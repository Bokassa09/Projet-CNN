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

## Résultats et Conclusion

Les résultats détaillés et les réflexions sur les différentes approches sont présentés dans le fichier Theorie.txt.

Dans ce projet, j’ai entraîné un réseau de neurones convolutionnel (CNN) avec pour objectif d’obtenir un modèle performant mais peu coûteux en ressources (temps de calcul et mémoire).
Grâce à plusieurs optimisations, la précision du modèle est passée d’environ 78 % à près de 90 %, avec une structure contenant environ 3,7 millions de paramètres.

Facteurs clés d’amélioration :

1. Le choix du learning rate approprié

2. L’ajustement du batch size

3. L’utilisation de pondérations de classes (class weights) pour corriger le déséquilibre du dataset

4. L’optimisation du nombre de neurones dans la couche de sortie

Certains modèles très performants du même dataset utilisent jusqu’à 14 à 28 millions de paramètres, mais ce projet montre qu’il est possible d’obtenir d’excellents résultats avec un modèle plus léger et bien calibré.

Ce travail a été pour moi une expérience formatrice dans l’ingénierie et l’optimisation des modèles de deep learning.
J’ai pu constater concrètement comment chaque paramètre — du taux d’apprentissage à la structure du réseau — influence la qualité du modèle.

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
