# 🫁 Détection de Pneumonie par Deep Learning

##  Description
Ce projet utilise un réseau de neurones convolutionnels (CNN) pour détecter automatiquement la pneumonie sur des radiographies thoraciques.

##  Objectif
Classifier les radiographies en deux catégories :
- **NORMAL** : Poumons sains
- **PNEUMONIA** : Pneumonie détectée

##  Dataset
- **Source** : [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Taille** : 1.15 GB
- **Format** : Images JPEG en niveaux de gris

### Répartition des données
- **Train** : 1,341 NORMAL + 3,875 PNEUMONIA
- **Test** : 234 NORMAL + 390 PNEUMONIA
- **Validation** : 8 NORMAL + 8 PNEUMONIA

##  Architecture du modèle
- **Type** : CNN (Convolutional Neural Network)
- **Input** : Images 150x150 pixels (grayscale)
- **Couches** : 3 blocs convolutionnels + couche dense
- **Activation finale** : Sigmoid (classification binaire)

##  Technologies utilisées
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

##  Installation
```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle
python main.py

Auteur: BOUEKE Omer Bokassa 
HPC&IA Ingenieur
