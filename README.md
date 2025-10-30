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

# Conclusion

Dans le fichier texte Theorie.txt, je présente les approches et les modifications qui m’ont permis d’améliorer les performances de mon modèle.

Dans ce projet, j’ai entraîné un réseau de neurones convolutif (CNN) avec pour objectif d’obtenir un modèle moins gourmand en ressources — notamment en puissance de calcul et en mémoire (RAM).
Grâce à plusieurs optimisations détaillées dans ce document, j’ai réussi à améliorer la précision (accuracy) du modèle, passant de 78% à environ 89 %....91 %.

Les quatre éléments qui ont eu le plus d’impact sur cette amélioration, et qu’il serait pertinent de conserver pour les futurs projets, sont :

1. Le choix approprié du learning rate ;

2. L’ajustement de la taille du batch size ;

3. L’utilisation des class weights ;

4. Le nombre de neurones dans la dernière couche de classification (couche entièrement connectée).

Dans ce projet, certains participants ont utilisé des modèles contenant jusqu’à 14 millions de
paramètres pour atteindre une telle performance, d’autres environ 1,5 million, et moi 3,7 millions.
D’autres encore ont atteint une très haute précision avec 28 millions de paramètres.

Ce projet, hébergé sur Kaggle, a surtout été pour moi une occasion d’apprendre l’ingénierie de l’entraînement
d’un modèle : comprendre comment chaque choix influence les performances est la clé pour progresser dans ce domaine.

Auteur: BOUEKE Omer Bokassa 
HPC&IA Ingenieur
