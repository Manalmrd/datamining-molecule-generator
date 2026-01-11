\# Générateur de Molécules IA - Projet Data Mining



Application web full-stack utilisant l'intelligence artificielle pour générer et analyser des molécules chimiques via un modèle VAE (Variational Autoencoder).



!\[Interface Principale](screenshots/interface-principale.png)

!\[Interface Principale](screenshots/interface-principale1.png)

!\[Propriétés Prédites](screenshots/proprietes-predites.png)





\## Fonctionnalités



\- Génération intelligente : Créez des molécules basées sur des propriétés cibles personnalisables

\- Prédiction de propriétés : Analyse automatique de 5+ caractéristiques moléculaires

\- Visualisation 3D interactive : Structure moléculaire rotative en temps réel avec Three.js

\- Interface intuitive : Contrôles par sliders pour ajuster les paramètres

\- Propriétés détaillées : Poids moléculaire, solubilité (LogP), surface polaire topologique, liaisons rotables, accepteurs H

\- Export PDF : Génération automatique de rapports détaillés

\- Export images : Téléchargement des structures 2D et 3D





\## Technologies utilisées



\### Frontend

\- React.js 18 - Framework JavaScript moderne

\- Three.js - Visualisation 3D interactive des molécules

\- CSS3 - Design responsive et animations

\- Axios - Communication avec l'API

\- jsPDF - Génération de rapports PDF

\- html2canvas - Capture d'écran pour PDF



\### Backend

\- Python 3.8+ - Langage principal

\- Flask 2.x - Framework API REST

\- PyTorch 2.x - Deep Learning et modèles VAE

\- RDKit - Chimie computationnelle et analyse moléculaire

\- SELFIES - Représentation robuste des molécules

\- Flask-CORS - Gestion des requêtes cross-origin



\### Machine Learning

\- VAE (Variational Autoencoder) - Génération de molécules dans l'espace latent

\- Property Predictor - Réseau neuronal pour prédire les propriétés chimiques

\- Espace latent 512D - Navigation optimisée dans l'espace des molécules



