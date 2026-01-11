import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import selfies as sf
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
from tqdm import tqdm
import os

# Supprimer l'avertissement de dtype de Pandas
warnings.filterwarnings("ignore", "Columns (2) have mixed types", category=pd.errors.DtypeWarning)

# =====================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES GLOBAUX
# =====================================================

# ⚠ À AJUSTER : Chemin d'accès à votre fichier CSV
DATA_PATH = r'C:\Users\ADMIN\Desktop\COMBINED_SELFIES_AVEC_PROPRIETES_normalized.csv'
COLUMN_NAME = 'SELFIES'

# --- PARAMÈTRES VAE ---
NUM_MOLECULES_TO_LOAD = 1_000_000
VAL_SIZE = 10_000
MAX_LEN = 128
BATCH_SIZE = 256
LATENT_DIM = 512
HIDDEN_DIM = 1024
EMBED_DIM = 256
NUM_VAE_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

# --- LOGIQUE DE REPRISE VAE MANUELLE (Pour commencer à l'époque 3) ---
# Mettre RESUME_EPOCH à 2 pour charger l'époque 2 et continuer à l'époque 3.
RESUME_EPOCH = 2
# Le nom du fichier de checkpoint pour la REPRISE:
LAST_SAVED_MODEL = f"best_selfies_vae_epoch_{RESUME_EPOCH}.pth" if RESUME_EPOCH > 0 else ""

# --- FICHIERS NÉCESSAIRES POUR L'INTERFACE ---
VOCAB_SAVE_PATH = "epoch/selfies_vocab.pt"
PREDICTOR_SAVE_PATH = "best_property_predictor.pth"

# --- CONFIGURATION DE LA GÉNÉRATION CONTRÔLÉE ---
PROPERTY_COLUMNS = [
    'Molecular_Weight_norm',
    'XLogP3_norm',
    'Topological_Polar_Surface_Area_norm',
    'Rotatable_Bond_Count_norm',
    'Hydrogen_Bond_Acceptor_Count_norm'
]
NUM_PROPERTIES = len(PROPERTY_COLUMNS)

# Les 5 valeurs cibles normalisées (entre 0 et 1)
TARGET_PROPERTIES_NORM = [0.4, 0.6, 0.3, 0.5, 0.4]


# =====================================================
# 1. PRÉPARATION ET CLASSES DE DONNÉES
# =====================================================

def save_vocab(symbol_to_idx, idx_to_symbol, path):
    """ Sauvegarde le vocabulaire pour une utilisation future et l'inférence. """
    torch.save({
        'symbol_to_idx': symbol_to_idx,
        'idx_to_symbol': idx_to_symbol,
        'vocab_size': len(symbol_to_idx)
    }, path)
    print(f"Vocabulaire sauvegardé dans : {path}")


def selfies_to_tensor(selfie, max_len, symbol_to_idx):
    """ Convertit une chaîne SELFIES en tenseur d'indices. """
    symbols = ['<start>'] + list(sf.split_selfies(selfie)) + ['<end>']
    if len(symbols) > max_len:
        symbols = symbols[:max_len - 1] + ['<end>']
    symbols += ['[nop]'] * (max_len - len(symbols))
    return torch.tensor([symbol_to_idx[s] for s in symbols], dtype=torch.long)


class SelfiesDataset(Dataset):
    def __init__(self, selfies_list, max_len, symbol_to_idx):
        self.selfies_list = selfies_list
        self.max_len = max_len
        self.symbol_to_idx = symbol_to_idx

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        return selfies_to_tensor(self.selfies_list[idx], self.max_len, self.symbol_to_idx)


# =====================================================
# 2. MODÈLE VAE (Pas de changement)
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
        """ Décode un vecteur latent z en une séquence SELFIES (méthode Greedy). """
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
            if (next_token.squeeze(1) == end_idx).all(): break

        if not generated_indices: return [""] * batch_size
        generated_indices = np.stack(generated_indices, axis=1)

        selfies_list = []
        for indices in generated_indices:
            selfie = []
            for idx in indices:
                if idx == end_idx: break
                symbol = idx_to_symbol[idx]
                if symbol not in ['<start>', '[nop]']: selfie.append(symbol)
            selfies_list.append("".join(selfie))

        return selfies_list


# =====================================================
# 3. MODÈLE DE PRÉDICTION DE PROPRIÉTÉ (MLP Régresseur)
# =====================================================

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_properties):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_properties)

    def forward(self, z):
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)


# =====================================================
# 4. FONCTIONS DE LOSS ET DE METRICS (Pas de changement)
# =====================================================

def vae_loss(recon_x, x, mu, logvar, kl_weight):
    target = x[:, 1:]
    CE = F.cross_entropy(recon_x.transpose(1, 2), target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return CE + kl_weight * KLD, CE, KLD


def token_metrics(recon_x, x, pad_idx):
    """ Calcule l'Accuracy au niveau des tokens non padding. """
    target = x[:, 1:]
    pred = recon_x.argmax(dim=-1)
    mask = target != pad_idx
    if not mask.any():
        return torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)
    pred = pred[mask]
    target = target[mask]
    correct = pred == target
    TP = correct.sum().float()
    FP = (~correct).sum().float()
    accuracy = TP / (TP + FP)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FP + 1e-8)
    return accuracy, precision, recall


# =====================================================
# 5. FONCTIONS D'ENTRAÎNEMENT VAE (Étape 1)
# =====================================================

def train_vae(model, train_loader, val_loader, optimizer, device, symbol_to_idx, VAE_CHECKPOINT, start_epoch):
    """ Entraîne le VAE et sauvegarde les checkpoints complets. """
    best_val_loss = float("inf")
    PAD_IDX = symbol_to_idx['[nop]']

    print(
        f"\n--- Démarrage de l'entraînement VAE pour {NUM_VAE_EPOCHS} époques, à partir de l'époque {start_epoch} ---")

    for epoch in range(start_epoch, NUM_VAE_EPOCHS + 1):
        # -------- TRAIN --------
        model.train()
        train_loss = 0
        kl_weight = min(1.0, epoch / 5.0)  # KL Annealing

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_VAE_EPOCHS} (TRAIN)", leave=False)

        for x in train_loop:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(Loss=loss.item() / x.size(0), KL_W=kl_weight)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # -------- VALIDATION --------
        model.eval()
        val_loss = acc_sum = prec_sum = rec_sum = 0
        num_val_samples = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, kl_weight)
                acc, prec, rec = token_metrics(recon, x, PAD_IDX)

                val_loss += loss.item()
                acc_sum += acc.item() * x.size(0)
                prec_sum += prec.item() * x.size(0)
                rec_sum += rec.item() * x.size(0)
                num_val_samples += x.size(0)

        avg_val_loss = val_loss / num_val_samples
        avg_acc = acc_sum / num_val_samples
        avg_prec = prec_sum / num_val_samples
        avg_rec = rec_sum / num_val_samples

        # Affichage des métriques complètes
        print(f"\n--- Époque {epoch:02d}/{NUM_VAE_EPOCHS} ---")
        print(
            f"TRAIN Loss : {avg_train_loss:.4f} | VAL Loss   : {avg_val_loss:.4f} | Accuracy : {avg_acc * 100:.2f}% | Precision : {avg_prec:.4f} | Recall : {avg_rec:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # NOM DU CHECKPOINT CORRIGÉ (sans le 5M)
            current_checkpoint_path = f"best_selfies_vae_epoch_{epoch}.pth"

            # Sauvegarde complète pour la reprise (inclus les métriques pour l'analyse)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'accuracy': avg_acc,
                'precision': avg_prec,
                'recall': avg_rec
            }, current_checkpoint_path)

            print(f">>> Nouveau meilleur modèle VAE sauvegardé: {current_checkpoint_path}")
            VAE_CHECKPOINT[0] = current_checkpoint_path

    print("\nEntraînement VAE terminé.")


# =====================================================
# 6. FONCTIONS D'ENTRAÎNEMENT PRÉDICTEUR (Étape 2)
# =====================================================

def train_predictor(vae_model, device, df_full, symbol_to_idx, max_len):
    """ Entraîne le Régresseur de Propriété. """
    PREDICTOR_LR = 1e-3
    PREDICTOR_EPOCHS = 50
    PREDICTOR_BATCH_SIZE = 1024

    latent_dim = LATENT_DIM
    hidden_dim = HIDDEN_DIM

    predictor = PropertyPredictor(latent_dim, hidden_dim, NUM_PROPERTIES).to(device)
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=PREDICTOR_LR)
    loss_fn = nn.MSELoss()

    print(f"\n--- Démarrage de l'entraînement du Régresseur de Propriété (MSE) avec {NUM_PROPERTIES} cibles ---")

    # --- EXTRACTION DES VECTEURS LATENTS (Z) ---
    print("Extraction des vecteurs latents (Z)...")
    vae_model.eval()

    full_selfies_list = df_full[COLUMN_NAME].tolist()

    full_dataset = SelfiesDataset(full_selfies_list, max_len, symbol_to_idx)
    full_loader = DataLoader(
        full_dataset,
        batch_size=PREDICTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    all_z = []
    with torch.no_grad():
        for x in tqdm(full_loader, desc="Encodage VAE"):
            x = x.to(device)
            mu, _ = vae_model.encode(x)
            all_z.append(mu.cpu())

    Z_data = torch.cat(all_z, dim=0).to(device)
    C_data = torch.tensor(df_full[PROPERTY_COLUMNS].values, dtype=torch.float32).to(device)

    predictor_dataset = torch.utils.data.TensorDataset(Z_data, C_data)
    predictor_loader = DataLoader(predictor_dataset, batch_size=PREDICTOR_BATCH_SIZE, shuffle=True)

    # --- BOUCLE D'ENTRAÎNEMENT DU RÉGRESSEUR ---
    best_predictor_loss = float('inf')

    for epoch in range(1, PREDICTOR_EPOCHS + 1):
        predictor.train()
        epoch_loss = 0
        predictor_loop = tqdm(predictor_loader, desc=f"Predictor Epoch {epoch:02d}/{PREDICTOR_EPOCHS}", leave=False)

        for z_batch, c_batch in predictor_loop:
            predictor_optimizer.zero_grad()
            c_pred = predictor(z_batch)
            loss = loss_fn(c_pred, c_batch)
            loss.backward()
            predictor_optimizer.step()
            epoch_loss += loss.item() * z_batch.size(0)
            predictor_loop.set_postfix(MSE=loss.item())

        avg_loss = epoch_loss / len(df_full)

        # Sauvegarde du Prédicteur
        if avg_loss < best_predictor_loss:
            best_predictor_loss = avg_loss
            # NOM DU FICHIER À SAUVEGARDER POUR L'INFÉRENCE
            torch.save(predictor.state_dict(), PREDICTOR_SAVE_PATH)

        if epoch % 10 == 0 or epoch == PREDICTOR_EPOCHS:
            print(f"Predictor Époque {epoch:02d}: MSE Loss = {avg_loss:.6f}")
            if avg_loss == best_predictor_loss: print(">>> Nouveau meilleur prédicteur sauvegardé")

    return predictor


# =====================================================
# 7. FONCTION DE GÉNÉRATION CONTRÔLÉE (Étape 3) (Pas de changement)
# =====================================================

def generate_controlled_molecule(vae_model, predictor, target_properties, symbol_to_idx, idx_to_symbol,
                                 max_iterations=2000):
    """ Optimisation dans l'Espace Latent. """
    Z_OPTIM_LR = 1e-1
    print(f"\n--- Démarrage de l'Optimisation pour les {NUM_PROPERTIES} cibles : {target_properties} ---")

    latent_z = torch.randn(1, LATENT_DIM, device=device).requires_grad_(True)
    target_c = torch.tensor(target_properties, dtype=torch.float32, device=device).unsqueeze(0)
    z_optimizer = optim.Adam([latent_z], lr=Z_OPTIM_LR)
    loss_fn = nn.MSELoss()

    predictor.eval()
    vae_model.eval()
    best_loss = float('inf')
    best_z = latent_z.detach().clone()

    for i in range(max_iterations):
        z_optimizer.zero_grad()
        predicted_c = predictor(latent_z)
        loss = loss_fn(predicted_c, target_c)
        loss.backward()
        z_optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z = latent_z.detach().clone()

        if i % 400 == 0:
            print(
                f"Iter {i}/{max_iterations}: Loss = {loss.item():.6f}, Prédit = {[f'{x:.4f}' for x in predicted_c.squeeze().tolist()]}")

    print(f"Optimisation terminée. Meilleure Loss finale : {best_loss:.6f}")

    # Décoder le meilleur Z trouvé
    with torch.no_grad():
        final_selfies = vae_model.sample(
            best_z, MAX_LEN, symbol_to_idx, idx_to_symbol, device
        )
        final_pred = predictor(best_z).squeeze()

    return final_selfies[0], best_z, final_pred


# =====================================================
# 8. EXÉCUTION PRINCIPALE (Orchestration des 3 Étapes)
# =====================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device.type.upper()}")

    # --- 8.1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
    print("\n--- 8.1. Chargement et Préparation des données ---")
    df = pd.read_csv(DATA_PATH, nrows=NUM_MOLECULES_TO_LOAD, low_memory=False)

    required_cols = [COLUMN_NAME] + PROPERTY_COLUMNS
    df_full = df[df[COLUMN_NAME].notna()]
    df_full = df_full[df_full[PROPERTY_COLUMNS].notna().all(axis=1)].reset_index(drop=True)

    if df_full.empty:
        print(
            "ERREUR: Le DataFrame est vide après le nettoyage. Veuillez vérifier le chemin DATA_PATH et les noms des colonnes.")
        exit()

    full_selfies_data = df_full[COLUMN_NAME].astype(str).tolist()

    alphabet = sf.get_alphabet_from_selfies(full_selfies_data)
    alphabet.update({'[nop]', '<start>', '<end>', '.'})
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    idx_to_symbol = {i: s for s, i in symbol_to_idx.items()}
    VOCAB_SIZE = len(alphabet)
    print(f"Taille du vocabulaire SELFIES: {VOCAB_SIZE}")

    # 📢 NOUVEAU : Sauvegarde du Vocabulaire (nécessaire pour l'interface)
    save_vocab(symbol_to_idx, idx_to_symbol, VOCAB_SAVE_PATH)

    train_selfies, val_selfies = train_test_split(
        full_selfies_data, test_size=VAL_SIZE, random_state=42
    )

    train_loader = DataLoader(SelfiesDataset(train_selfies, MAX_LEN, symbol_to_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(SelfiesDataset(val_selfies, MAX_LEN, symbol_to_idx), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    # --- 8.2. ENTRAÎNEMENT DU VAE (Étape 1) ---
    model = SelfiesVAE(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    VAE_CHECKPOINT = [""]
    START_EPOCH = 1

    # Logique de reprise AMÉLIORÉE (Chargement de l'époque 2 pour commencer l'époque 3)
    if RESUME_EPOCH > 0:
        if os.path.exists(LAST_SAVED_MODEL):
            try:
                # Tentative de chargement complet
                checkpoint = torch.load(LAST_SAVED_MODEL, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                START_EPOCH = checkpoint['epoch'] + 1
                VAE_CHECKPOINT[0] = LAST_SAVED_MODEL
                print(f" REPRISE RÉUSSIE. L'entraînement va CONTINUER à partir de l'époque {START_EPOCH}.")
            except Exception as e:
                print(f" Échec du chargement complet/partiel du VAE ({e}). Redémarrage à l'époque 1.")
                START_EPOCH = 1
        else:
            print(
                f"⚠ Avertissement : Fichier de checkpoint {LAST_SAVED_MODEL} non trouvé. Démarrage à l'époque 1 (RESUME_EPOCH ignoré).")
            START_EPOCH = 1

    train_vae(model, train_loader, val_loader, optimizer, device, symbol_to_idx, VAE_CHECKPOINT, START_EPOCH)

    # Recharger le MEILLEUR modèle VAE (probablement l'époque 3 si c'est la meilleure) pour l'étape 2
    if VAE_CHECKPOINT[0]:
        print(f"Chargement du meilleur VAE ({VAE_CHECKPOINT[0]}) pour l'étape 2.")
        loaded_checkpoint_data = torch.load(VAE_CHECKPOINT[0], map_location=device)
        model.load_state_dict(loaded_checkpoint_data['model_state_dict'])

    # --- 8.3. ENTRAÎNEMENT DU PRÉDICTEUR DE PROPRIÉTÉ (Étape 2) ---
    if NUM_PROPERTIES == 0:
        print("\nATTENTION: Aucune propriété cible définie. Saut de l'étape 2 et 3.")
    else:
        # ✅ L'étape train_predictor sauvegardera automatiquement 'best_property_predictor.pth'
        predictor_model = train_predictor(model, device, df_full, symbol_to_idx, MAX_LEN)

        # --- 8.4. GÉNÉRATION CONTRÔLÉE (Étape 3) ---
        generated_selfie, final_z, final_pred = generate_controlled_molecule(
            model,
            predictor_model,
            TARGET_PROPERTIES_NORM,
            symbol_to_idx,
            idx_to_symbol
        )

        print("\n=============================================")
        print("RÉSULTAT DE LA GÉNÉRATION CONTRÔLÉE (VÉRIFICATION FINALE)")
        print(f"Propriétés Cibles ({', '.join(PROPERTY_COLUMNS)}) : {TARGET_PROPERTIES_NORM}")
        print(f"Propriétés Prédites : {[f'{x:.4f}' for x in final_pred.tolist()]}")
        print(f"SELFIES Généré : {generated_selfie}")

        try:
            smiles = sf.decoder(generated_selfie)
            print(f"SMILES correspondant : {smiles} (Validité: OK)")
        except Exception:
            print("SMILES correspondant : (Invalide) - Erreur de décodage SELFIES")