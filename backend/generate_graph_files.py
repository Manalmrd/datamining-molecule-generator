import os
import sys
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import json
import uuid

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PT_2D_DIR = os.path.join(BASE_DIR, "generated_graphs_2d")
PT_3D_DIR = os.path.join(BASE_DIR, "generated_graphs_3d")

# Créer les dossiers s'ils n'existent pas
for dir_path in [PT_2D_DIR, PT_3D_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[Graph Generator] Dossier créé: {dir_path}")

print(f"[Graph Generator] Dossier 2D: {PT_2D_DIR}")
print(f"[Graph Generator] Dossier 3D: {PT_3D_DIR}")


# =====================================================
# FONCTIONS D'ENCODAGE
# =====================================================

def one_hot_encoding(x, allowable_set):
    """Encode une valeur en one-hot."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    """Extrait les caractéristiques d'un atome."""
    atom_type = one_hot_encoding(atom.GetSymbol(),
                                 ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Other'])
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    implicit_h = one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    in_ring = [atom.IsInRing()]
    hybridization = one_hot_encoding(str(atom.GetHybridization()),
                                     ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Other'])
    return np.array(atom_type + degree + implicit_h + in_ring + hybridization, dtype=np.float32)


def get_bond_features(bond):
    """Extrait les caractéristiques d'une liaison."""
    bond_type = one_hot_encoding(str(bond.GetBondType()),
                                 ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
    in_ring = [bond.IsInRing()]
    return np.array(bond_type + in_ring, dtype=np.float32)


def mol_to_pyg_data(mol, target_props, is_3d=False):
    """Convertit un objet RDKit Mol en PyG Data."""
    # 1. Caractéristiques des Nœuds (x)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_features), dtype=torch.float)

    # 2. Structure du Graphe (edge_index et edge_attr)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_features = get_bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bond_features)
        edge_attrs.append(bond_features)

    edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    # 3. Caractéristiques Cibles (y)
    y = torch.tensor([target_props], dtype=torch.float)

    pos = None
    if is_3d and mol.GetNumConformers() > 0:
        pos_coords = []
        conformer = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pt = conformer.GetAtomPosition(i)
            pos_coords.append([pt.x, pt.y, pt.z])
        pos = torch.tensor(pos_coords, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos,
                    smiles=Chem.MolToSmiles(mol), num_atoms=mol.GetNumAtoms())
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    smiles=Chem.MolToSmiles(mol), num_atoms=mol.GetNumAtoms())


# =====================================================
# FONCTION PRINCIPALE
# =====================================================

def generate_graph_files(smiles, properties, molecule_id=None):
    """
    Génère les fichiers .pt (2D et 3D) à partir d'un SMILES
    Retourne: (success_2d, success_3d, pt_files_info)
    """
    print(f"\n{'=' * 60}")
    print(f"GÉNÉRATION DE FICHIERS GRAPHIQUES")
    print(f"{'=' * 60}")
    print(f"SMILES: {smiles}")
    print(f"Propriétés: {properties}")

    if molecule_id is None:
        molecule_id = str(uuid.uuid4())[:8]

    success_2d = False
    success_3d = False
    pt_files_info = {}

    try:
        # 1. Convertir SMILES en molécule
        print("[Graph Generator] Conversion SMILES...")
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print("[Graph Generator] ERREUR: SMILES invalide")
            return False, False, {}

        print(f"[Graph Generator] Molécule créée: {mol.GetNumAtoms()} atomes")

        # 2. Générer fichier 2D
        try:
            print("[Graph Generator] Génération du graphe 2D...")
            data_2d = mol_to_pyg_data(mol, properties, is_3d=False)

            pt_2d_filename = f"mol_{molecule_id}_2d.pt"
            pt_2d_path = os.path.join(PT_2D_DIR, pt_2d_filename)

            torch.save(data_2d, pt_2d_path)
            success_2d = True

            # Préparer les données pour le frontend
            graph_data_2d = {
                'nodes': [],
                'edges': [],
                'smiles': smiles,
                'num_atoms': data_2d.num_atoms if hasattr(data_2d, 'num_atoms') else mol.GetNumAtoms()
            }

            # Extraire les atomes
            for i in range(data_2d.x.shape[0]):
                atom_features = data_2d.x[i].tolist()
                # Déterminer le type d'atome
                atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Other']
                atom_type_idx = atom_features[:11].index(1.0) if 1.0 in atom_features[:11] else 10

                graph_data_2d['nodes'].append({
                    'id': i,
                    'type': atom_types[atom_type_idx],
                    'features': atom_features
                })

            # Extraire les liaisons
            if hasattr(data_2d, 'edge_index'):
                edge_index = data_2d.edge_index.tolist()
                for i in range(edge_index[0].shape[0] if hasattr(edge_index[0], 'shape') else len(edge_index[0])):
                    graph_data_2d['edges'].append({
                        'source': int(edge_index[0][i]),
                        'target': int(edge_index[1][i])
                    })

            pt_files_info['2d'] = {
                'filename': pt_2d_filename,
                'path': pt_2d_path,
                'url': f"/get-graph/{molecule_id}/2d",
                'graph_data': graph_data_2d
            }

            print(f"[Graph Generator] Fichier 2D créé: {pt_2d_path}")

        except Exception as e:
            print(f"[Graph Generator] Erreur 2D: {e}")
            import traceback
            traceback.print_exc()

        # 3. Générer fichier 3D
        try:
            print("[Graph Generator] Génération du graphe 3D...")

            # Ajouter hydrogènes et générer coordonnées 3D
            mol_3d = Chem.AddHs(mol)

            # Essayer d'embarquer la molécule
            embed_result = AllChem.EmbedMolecule(mol_3d, randomSeed=42, maxAttempts=10)

            if embed_result == 0:
                # Optimiser la géométrie
                try:
                    AllChem.MMFFOptimizeMolecule(mol_3d)
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_3d)
                    except:
                        pass

                # Créer les données 3D
                data_3d = mol_to_pyg_data(mol_3d, properties, is_3d=True)

                pt_3d_filename = f"mol_{molecule_id}_3d.pt"
                pt_3d_path = os.path.join(PT_3D_DIR, pt_3d_filename)

                torch.save(data_3d, pt_3d_path)
                success_3d = True

                # Préparer les données pour le frontend
                graph_data_3d = {
                    'nodes': [],
                    'edges': [],
                    'positions': [],
                    'smiles': smiles,
                    'num_atoms': data_3d.num_atoms if hasattr(data_3d, 'num_atoms') else mol_3d.GetNumAtoms()
                }

                # Extraire les atomes
                for i in range(data_3d.x.shape[0]):
                    atom_features = data_3d.x[i].tolist()
                    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Other']
                    atom_type_idx = atom_features[:11].index(1.0) if 1.0 in atom_features[:11] else 10

                    graph_data_3d['nodes'].append({
                        'id': i,
                        'type': atom_types[atom_type_idx],
                        'features': atom_features
                    })

                # Extraire les liaisons
                if hasattr(data_3d, 'edge_index'):
                    edge_index = data_3d.edge_index.tolist()
                    for i in range(edge_index[0].shape[0] if hasattr(edge_index[0], 'shape') else len(edge_index[0])):
                        graph_data_3d['edges'].append({
                            'source': int(edge_index[0][i]),
                            'target': int(edge_index[1][i])
                        })

                # Extraire les positions 3D
                if hasattr(data_3d, 'pos'):
                    graph_data_3d['positions'] = data_3d.pos.tolist()

                pt_files_info['3d'] = {
                    'filename': pt_3d_filename,
                    'path': pt_3d_path,
                    'url': f"/get-graph/{molecule_id}/3d",
                    'graph_data': graph_data_3d
                }

                print(f"[Graph Generator] Fichier 3D créé: {pt_3d_path}")

            else:
                print("[Graph Generator] Impossible de générer les coordonnées 3D")
                # Créer quand même un fichier 3D sans positions
                data_3d_no_pos = mol_to_pyg_data(mol, properties, is_3d=False)
                pt_3d_filename = f"mol_{molecule_id}_3d.pt"
                pt_3d_path = os.path.join(PT_3D_DIR, pt_3d_filename)
                torch.save(data_3d_no_pos, pt_3d_path)
                success_3d = True

                # Copier les données 2D pour 3D
                pt_files_info['3d'] = pt_files_info.get('2d', {}).copy()
                pt_files_info['3d']['filename'] = pt_3d_filename
                pt_files_info['3d']['path'] = pt_3d_path
                pt_files_info['3d']['url'] = f"/get-graph/{molecule_id}/3d"

                print(f"[Graph Generator] Fichier 3D créé (sans positions 3D): {pt_3d_path}")

        except Exception as e:
            print(f"[Graph Generator] Erreur 3D: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n[Graph Generator] RÉSULTAT:")
        print(f"  2D: {'✓' if success_2d else '✗'}")
        print(f"  3D: {'✓' if success_3d else '✗'}")
        print(f"  ID: {molecule_id}")

        return success_2d, success_3d, pt_files_info

    except Exception as e:
        print(f"[Graph Generator] Erreur globale: {e}")
        import traceback
        traceback.print_exc()
        return False, False, {}


# =====================================================
# EXÉCUTION DIRECTE (pour test)
# =====================================================
if __name__ == "__main__":
    # Exemple de test
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    test_properties = [0.5, 0.5, 0.5, 0.5, 0.5]

    success_2d, success_3d, pt_info = generate_graph_files(test_smiles, test_properties)

    if success_2d or success_3d:
        print("\nFichiers créés avec succès!")
        print(f"Informations: {json.dumps(pt_info, indent=2, default=str)}")
    else:
        print("\nÉchec de la génération des fichiers.")