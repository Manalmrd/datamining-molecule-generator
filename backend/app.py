from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import selfies as sf
import numpy as np
import json
import os
from datetime import datetime
import uuid
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
import base64
from io import BytesIO
import time
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# =====================================================
# CONFIGURATION
# =====================================================
MAX_LEN = 128
LATENT_DIM = 512
HIDDEN_DIM = 1024
EMBED_DIM = 256
NUM_PROPERTIES = 5

# Chemins vers les fichiers
VOCAB_PATH = "selfies_vocab.pt"
VAE_CHECKPOINT_PATH = "best_selfies_vae_epoch_4.pth"
PREDICTOR_PATH = "best_property_predictor.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Nom des propriétés
PROPERTY_NAMES = [
    'Molecular_Weight_norm',
    'XLogP3_norm',
    'Topological_Polar_Surface_Area_norm',
    'Rotatable_Bond_Count_norm',
    'Hydrogen_Bond_Acceptor_Count_norm'
]

PROPERTY_LABELS = {
    'Molecular_Weight_norm': 'Poids Moléculaire',
    'XLogP3_norm': 'LogP (Solubilité)',
    'Topological_Polar_Surface_Area_norm': 'Surface Polaire Topologique',
    'Rotatable_Bond_Count_norm': 'Liaisons Rotatives',
    'Hydrogen_Bond_Acceptor_Count_norm': 'Accepteurs de Liaison H'
}

# =====================================================
# 1. CHARGEMENT DU VOCABULAIRE
# =====================================================
print("\n1. Chargement du vocabulaire...")
vocab_data = torch.load(VOCAB_PATH, map_location=device)
symbol_to_idx = vocab_data['symbol_to_idx']
idx_to_symbol = vocab_data['idx_to_symbol']
VOCAB_SIZE = vocab_data['vocab_size']
print(f"Vocab size: {VOCAB_SIZE}")


# =====================================================
# 2. ARCHITECTURES DES MODÈLES
# =====================================================
class SelfiesVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_latent_to_h0 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        emb = self.embedding(x)
        _, h = self.encoder_rnn(emb)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dec_input = x[:, :-1]
        emb_dec = self.embedding(dec_input)
        h0 = self.fc_latent_to_h0(z).unsqueeze(0)
        out, _ = self.decoder_rnn(emb_dec, h0)
        return self.fc_output(out), mu, logvar

    def sample(self, z, max_len, symbol_to_idx, idx_to_symbol, device):
        batch_size = z.size(0)
        h = self.fc_latent_to_h0(z).unsqueeze(0)
        start_idx = symbol_to_idx['<start>']
        current_token = torch.tensor([start_idx] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        generated_indices = []
        end_idx = symbol_to_idx['<end>']

        for _ in range(max_len - 1):
            emb = self.embedding(current_token)
            out, h = self.decoder_rnn(emb, h)
            output = self.fc_output(out.squeeze(1))
            next_token = torch.argmax(output, dim=1).unsqueeze(1)
            generated_indices.append(next_token.squeeze(1).cpu().numpy())
            current_token = next_token
            if (next_token.squeeze(1) == end_idx).all():
                break

        if not generated_indices:
            return [""] * batch_size

        generated_indices = np.stack(generated_indices, axis=1)
        selfies_list = []

        for indices in generated_indices:
            selfie = []
            for idx in indices:
                if idx == end_idx:
                    break
                symbol = idx_to_symbol[idx]
                if symbol not in ['<start>', '[nop]']:
                    selfie.append(symbol)
            selfies_list.append("".join(selfie))

        return selfies_list


class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_properties):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_properties)

    def forward(self, z):
        h = torch.relu(self.bn1(self.fc1(z)))
        h = torch.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)


# =====================================================
# 3. CHARGEMENT DES MODÈLES
# =====================================================
print("\n2. Chargement du VAE...")
vae_model = SelfiesVAE(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
vae_checkpoint = torch.load(VAE_CHECKPOINT_PATH, map_location=device)
vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
vae_model.eval()
print(f"VAE loaded from: {VAE_CHECKPOINT_PATH}")

print("\n3. Chargement du prédicteur...")
predictor = PropertyPredictor(LATENT_DIM, HIDDEN_DIM, NUM_PROPERTIES).to(device)
predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=device))
predictor.eval()
print(f"Predictor loaded from: {PREDICTOR_PATH}")


# =====================================================
# 4. FONCTIONS UTILITAIRES AMÉLIORÉES
# =====================================================
def generate_molecule_api(target_properties, max_iterations=500):
    """Version améliorée de la fonction de génération"""
    print(f"\n[API] Génération avec propriétés: {target_properties}")

    # Validation des entrées
    if len(target_properties) != NUM_PROPERTIES:
        raise ValueError(f"Expected {NUM_PROPERTIES} properties, got {len(target_properties)}")

    for prop in target_properties:
        if prop < 0 or prop > 1:
            raise ValueError(f"Property value {prop} out of range [0, 1]")

    start_time = time.time()

    # Initialisation avec un meilleur learning rate
    latent_z = torch.randn(1, LATENT_DIM, device=device).requires_grad_(True)
    target_tensor = torch.tensor(target_properties, dtype=torch.float32, device=device).unsqueeze(0)
    optimizer = torch.optim.Adam([latent_z], lr=0.05)  # Réduit pour plus de stabilité
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_z = None
    convergence_iter = max_iterations

    # Optimisation avec patience
    patience = 50
    patience_counter = 0
    previous_loss = float('inf')

    for i in range(max_iterations):
        optimizer.zero_grad()
        predicted = predictor(latent_z)
        loss = loss_fn(predicted, target_tensor)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z = latent_z.detach().clone()
            convergence_iter = i

        # Early stopping avec patience
        if abs(previous_loss - loss.item()) < 1e-8:
            patience_counter += 1
        else:
            patience_counter = 0

        previous_loss = loss.item()

        if patience_counter >= patience:
            print(f"[API] Early stopping à l'itération {i}")
            break

        # Arrêt prématuré si convergence
        if loss.item() < 1e-5:
            print(f"[API] Convergence atteinte à l'itération {i}")
            break

    # Génération finale
    with torch.no_grad():
        final_selfies = vae_model.sample(best_z, MAX_LEN, symbol_to_idx, idx_to_symbol, device)
        final_pred = predictor(best_z).squeeze()

    # Conversion en SMILES
    smiles = None
    selfies_str = ""
    if final_selfies and final_selfies[0]:
        selfies_str = final_selfies[0]
        try:
            smiles = sf.decoder(selfies_str)
            print(f"[API] SMILES généré: {smiles}")
        except Exception as e:
            print(f"[API] Erreur de conversion SELFIES: {e}")
            smiles = None

    elapsed_time = time.time() - start_time
    print(f"[API] Génération terminée en {elapsed_time:.2f} secondes")

    return {
        'selfies': selfies_str,
        'smiles': smiles,
        'predicted_properties': final_pred.tolist(),
        'loss': best_loss,
        'convergence_iterations': convergence_iter,
        'generation_time': elapsed_time,
        'latent_vector': best_z.squeeze().tolist() if best_z is not None else None
    }


def analyze_molecule_with_fallback(smiles):
    """Analyse une molécule avec fallback pour garantir au moins l'image 2D"""
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"[API] Impossible de créer la molécule depuis SMILES: {smiles}")
            return None

        # Calcul des propriétés de base (toujours disponibles)
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'h_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'h_bond_donors': Descriptors.NumHDonors(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'num_rings': Chem.rdMolDescriptors.CalcNumRings(mol),
            'aromatic_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'is_valid': True
        }

        # Génération d'image 2D (toujours possible)
        try:
            img = Draw.MolToImage(mol, size=(400, 400))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            properties['image_2d'] = img_str
        except Exception as e:
            print(f"[API] Erreur génération image 2D: {e}")
            properties['image_2d'] = None

        # Génération des coordonnées 3D (optionnelle)
        atoms_3d = []
        bonds_3d = []
        num_atoms_3d = 0
        num_bonds_3d = 0

        try:
            # Créer une molécule avec hydrogènes pour la 3D
            mol_3d = Chem.AddHs(mol)

            # Générer la conformation 3D
            params = AllChem.ETKDGv3()
            params.randomSeed = 42

            # Les paramètres peuvent varier selon la version de RDKit
            # Essayons d'utiliser des paramètres génériques
            try:
                # Essayons d'abord sans maxAttempts
                params.useRandomCoords = True

                # Si votre RDKit supporte maxAttempts, utilisez-le
                if hasattr(params, 'maxAttempts'):
                    params.maxAttempts = 100

                # Vérifions d'autres paramètres disponibles
                if hasattr(params, 'maxIterations'):
                    params.maxIterations = 100

            except AttributeError as e:
                print(f"[API] Paramètre non supporté: {e}")
                # Continuer avec les paramètres par défaut

            success = AllChem.EmbedMolecule(mol_3d, params)

            if success == -1:
                print(f"[API] L'embedding 3D a échoué pour {smiles}, tentative alternative...")
                # Tentative alternative avec des paramètres différents
                params2 = AllChem.ETKDGv3()
                params2.randomSeed = 1234
                params2.useRandomCoords = True
                success = AllChem.EmbedMolecule(mol_3d, params2)

                if success == -1:
                    print(f"[API] Deuxième tentative d'embedding 3D échouée")
                    raise Exception("Embedding 3D failed after 2 attempts")

            # Optimiser la géométrie
            try:
                # Essayer plusieurs méthodes d'optimisation
                try:
                    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
                    except:
                        print(f"[API] Optimisation échouée, mais on garde la conformation brute")
                        # On peut essayer une optimisation minimale
                        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=50)
            except Exception as opt_error:
                print(f"[API] Erreur d'optimisation: {opt_error}")
                # Continuer même sans optimisation

            # Extraire les coordonnées 3D
            if mol_3d.GetNumConformers() > 0:
                conformer = mol_3d.GetConformer()
                for i in range(mol_3d.GetNumAtoms()):
                    atom = mol_3d.GetAtomWithIdx(i)
                    pos = conformer.GetAtomPosition(i)
                    atoms_3d.append({
                        'id': i,
                        'symbol': atom.GetSymbol(),
                        'x': float(pos.x),
                        'y': float(pos.y),
                        'z': float(pos.z),
                        'atomic_number': int(atom.GetAtomicNum()),
                        'is_aromatic': bool(atom.GetIsAromatic())
                    })

                # Extraire les liaisons
                for bond in mol_3d.GetBonds():
                    bonds_3d.append({
                        'from_atom': int(bond.GetBeginAtomIdx()),
                        'to_atom': int(bond.GetEndAtomIdx()),
                        'bond_type': str(bond.GetBondType()),
                        'is_aromatic': bool(bond.GetIsAromatic()),
                        'is_in_ring': bool(bond.IsInRing())
                    })

                num_atoms_3d = len(atoms_3d)
                num_bonds_3d = len(bonds_3d)
                print(f"[API] Structure 3D générée avec {num_atoms_3d} atomes et {num_bonds_3d} liaisons")

        except Exception as e:
            print(f"[API] Erreur lors de la génération 3D pour {smiles}: {e}")
            # On continue sans les données 3D, mais on garde tout le reste
            print(f"[API] Données 2D toujours disponibles pour {smiles}")

        return {
            'properties': properties,
            'atoms_3d': atoms_3d,
            'bonds_3d': bonds_3d,
            'num_atoms_3d': num_atoms_3d,
            'num_bonds_3d': num_bonds_3d,
            'has_3d': len(atoms_3d) > 0
        }

    except Exception as e:
        print(f"[API] Erreur d'analyse RDKit: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================================
# 5. ROUTES API AMÉLIORÉES
# =====================================================
@app.route('/')
def index():
    return jsonify({
        'message': 'API de Génération de Molécules avec Fallback',
        'version': '2.1.0',
        'endpoints': {
            '/generate': 'POST - Générer une molécule',
            '/health': 'GET - Vérifier l\'état de l\'API',
            '/config': 'GET - Obtenir la configuration'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'models_loaded': True,
        'rdkit_available': True
    })


@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({
        'num_properties': NUM_PROPERTIES,
        'property_names': PROPERTY_NAMES,
        'property_labels': PROPERTY_LABELS,
        'device': str(device),
        'max_iterations': 500,
        'features': {
            '2d_structure': True,
            '3d_structure': True,
            'property_prediction': True,
            'smiles_generation': True
        }
    })


@app.route('/generate', methods=['POST'])
def generate():
    """Génère une molécule avec les propriétés spécifiées - Version améliorée"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Aucune donnée reçue', 'success': False}), 400

        # Extraire les propriétés
        target_properties = []
        for prop_name in PROPERTY_NAMES:
            if prop_name in data:
                value = float(data[prop_name])
                value = max(0.0, min(1.0, value))
                target_properties.append(value)
            else:
                target_properties.append(0.5)

        print(f"[API] Génération demandée avec propriétés: {target_properties}")

        # Générer la molécule
        generation_result = generate_molecule_api(target_properties)

        # Créer un ID unique
        molecule_id = str(uuid.uuid4())[:8]

        # Initialiser la réponse
        response = {
            'success': True,
            'molecule_id': molecule_id,
            'name': f"Mol_{molecule_id}",
            'selfies': generation_result['selfies'],
            'smiles': generation_result['smiles'],
            'predicted_properties': {
                name: {
                    'target': target,
                    'predicted': pred,
                    'difference': abs(target - pred)
                }
                for name, target, pred in
                zip(PROPERTY_NAMES, target_properties, generation_result['predicted_properties'])
            },
            'predicted_properties_array': generation_result['predicted_properties'],
            'predicted_properties_formatted': [
                f"{PROPERTY_LABELS[name]}: {pred:.4f}"
                for name, pred in zip(PROPERTY_NAMES, generation_result['predicted_properties'])
            ],
            'generation_info': {
                'loss': float(generation_result['loss']),
                'convergence_iterations': generation_result['convergence_iterations'],
                'generation_time': generation_result['generation_time'],
                'timestamp': datetime.now().isoformat()
            }
        }

        # Analyser la molécule si SMILES valide
        if generation_result['smiles']:
            analysis = analyze_molecule_with_fallback(generation_result['smiles'])

            if analysis:
                # Générer un nom plus descriptif
                if analysis['properties']['formula']:
                    formula = analysis['properties']['formula']
                    if 'F' in formula:
                        response['name'] = f"Fluoro-compound-{molecule_id}"
                    elif 'Cl' in formula:
                        response['name'] = f"Chloro-compound-{molecule_id}"
                    elif 'Br' in formula:
                        response['name'] = f"Bromo-compound-{molecule_id}"
                    elif 'N' in formula and 'O' in formula:
                        response['name'] = f"Nitro-compound-{molecule_id}"
                    else:
                        response['name'] = f"Compound-{molecule_id}"

                # Générer la description
                props = analysis['properties']
                description_parts = []

                if props['heavy_atoms'] > 0:
                    description_parts.append(f"{props['heavy_atoms']} atomes lourds")
                if props['num_rings'] > 0:
                    description_parts.append(f"{props['num_rings']} cycle(s)")
                if props['rotatable_bonds'] > 0:
                    description_parts.append(f"{props['rotatable_bonds']} liaison(s) rotative(s)")

                if description_parts:
                    response['molecule_description'] = f"Molécule organique avec " + ", ".join(description_parts) + "."
                else:
                    response['molecule_description'] = "Molécule organique générée avec succès."

                # Ajouter l'analyse à la réponse
                response['analysis'] = {
                    'properties': analysis['properties'],
                    'atoms_3d': analysis['atoms_3d'],
                    'bonds_3d': analysis['bonds_3d'],
                    'num_atoms_3d': analysis['num_atoms_3d'],
                    'num_bonds_3d': analysis['num_bonds_3d'],
                    'has_3d': analysis['has_3d'],
                    'image_2d': analysis['properties'].get('image_2d')
                }

                # Ajouter l'image 2D directement dans la réponse principale
                if analysis['properties'].get('image_2d'):
                    response['image_2d'] = analysis['properties']['image_2d']

                # Ajouter un flag pour indiquer si la 3D est disponible
                response['has_3d_structure'] = analysis['has_3d']

                if not analysis['has_3d']:
                    response['3d_warning'] = "La structure 3D n'a pas pu être générée, mais la 2D est disponible."
            else:
                response['analysis'] = None
                response['molecule_description'] = "Molécule générée mais impossible d'analyser la structure."
                response['has_3d_structure'] = False
                response['warning'] = "Impossible d'analyser la molécule générée"
        else:
            response['analysis'] = None
            response['molecule_description'] = "Aucune molécule valide générée."
            response['has_3d_structure'] = False
            response['warning'] = "SMILES invalide généré"

        # Sauvegarder dans l'historique
        save_to_history(response)

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        print(f"[API] Erreur interne: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Erreur interne du serveur', 'success': False}), 500


@app.route('/convert-to-graph', methods=['POST'])
def convert_to_graph():
    """Convertit un SMILES en données de graphe - Version robuste"""
    try:
        data = request.get_json()

        if not data or 'smiles' not in data:
            return jsonify({'error': 'SMILES requis', 'success': False}), 400

        smiles = data['smiles'].strip()

        if not smiles:
            return jsonify({'error': 'SMILES vide', 'success': False}), 400

        print(f"[API] Conversion SMILES vers graphe: {smiles[:50]}...")

        # Analyser la molécule
        analysis = analyze_molecule_with_fallback(smiles)

        if not analysis:
            return jsonify({
                'success': False,
                'error': 'Impossible d\'analyser le SMILES',
                'smiles': smiles
            }), 400

        # Préparer la réponse
        response = {
            'success': True,
            'smiles': smiles,
            'analysis': {
                'properties': analysis['properties'],
                'has_3d': analysis['has_3d'],
                'image_2d': analysis['properties'].get('image_2d')
            }
        }

        # Ajouter les données 3D si disponibles
        if analysis['has_3d']:
            response['analysis']['atoms_3d'] = analysis['atoms_3d']
            response['analysis']['bonds_3d'] = analysis['bonds_3d']

        return jsonify(response)

    except Exception as e:
        print(f"[API] Erreur de conversion: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


def save_to_history(generation_data):
    """Sauvegarde une génération dans l'historique"""
    try:
        history = []
        history_file = 'generation_history.json'

        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

        # Simplifier les données pour l'historique
        simplified_data = {
            'id': generation_data.get('molecule_id', str(uuid.uuid4())[:8]),
            'name': generation_data.get('name', 'Unknown'),
            'smiles': generation_data.get('smiles'),
            'timestamp': datetime.now().isoformat(),
            'has_3d': generation_data.get('has_3d_structure', False),
            'properties': generation_data.get('predicted_properties_formatted', [])
        }

        history.append(simplified_data)

        # Garder seulement les 50 dernières entrées
        if len(history) > 50:
            history = history[-50:]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        print(f"[API] Erreur historique: {e}")


@app.route('/history', methods=['GET'])
def get_history():
    """Obtenir l'historique des générations"""
    try:
        if os.path.exists('generation_history.json'):
            with open('generation_history.json', 'r', encoding='utf-8') as f:
                history = json.load(f)
            return jsonify(history)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# 6. DÉMARRAGE DU SERVEUR
# =====================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" API de Génération de Molécules avec Fallback")
    print("=" * 60)
    print(f" Endpoint: http://localhost:5000")
    print(f" Génération: POST http://localhost:5000/generate")
    print(f" Conversion: POST http://localhost:5000/convert-to-graph")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )