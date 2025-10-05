### Partie 0 : STATISTIQUES DU DATASET
import os # Permet de lire les dossiers et fichiers dans CNN
# Chemins vers les données
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
VAL_DIR = 'data/val'

def compter_images(dossier):
    """Compte combien d'images il y a dans un dossier"""
    if os.path.exists(dossier):
        return len(os.listdir(dossier))
    else:
        return 0

def afficher_stats():
    """Affiche le nombre d'images dans chaque catégorie"""
    print("=== STATISTIQUES DU DATASET ===\n")
    
    # Train
    train_normal = compter_images(TRAIN_DIR + '/NORMAL')
    train_pneumonia = compter_images(TRAIN_DIR + '/PNEUMONIA')
    print(f"TRAIN:")
    print(f"  - Normal: {train_normal}")
    print(f"  - Pneumonia: {train_pneumonia}")
    print(f"  - Total: {train_normal + train_pneumonia}\n")
    
    # Test
    test_normal = compter_images(TEST_DIR + '/NORMAL')
    test_pneumonia = compter_images(TEST_DIR + '/PNEUMONIA')
    print(f"TEST:")
    print(f"  - Normal: {test_normal}")
    print(f"  - Pneumonia: {test_pneumonia}")
    print(f"  - Total: {test_normal + test_pneumonia}\n")
    
    # Val
    val_normal = compter_images(VAL_DIR + '/NORMAL')
    val_pneumonia = compter_images(VAL_DIR + '/PNEUMONIA')
    print(f"VALIDATION:")
    print(f"  - Normal: {val_normal}")
    print(f"  - Pneumonia: {val_pneumonia}")
    print(f"  - Total: {val_normal + val_pneumonia}")

#afficher_stats()

### Partie 1 : Afficher quelques images
import matplotlib.pyplot as plt
from PIL import Image
def afficher_exemples():
    print("\n...Affichage d'exemple")
    # Créer une grille de 2 lignes x 3 colonnes
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    # Chemin vers les images NORMAL
    dossier_normal = TRAIN_DIR + '/NORMAL'
    dossier_pneumonia = TRAIN_DIR + '/PNEUMONIA'
    
    # Lister tous les fichiers dans ce dossier
    images_normal = os.listdir(dossier_normal)
    images_pneumonia = os.listdir(dossier_pneumonia)
    
    # Prendre seulement les 3 premières
    images_normal = images_normal[:3]
    images_pneumonia= images_pneumonia[:3]
    
    # Afficher les 3 images NORMAL dans la première ligne
    for i in range(3):
        # Construire le chemin complet de l'image
        chemin_image = dossier_normal + '/' + images_normal[i]
        chemin_image1 = dossier_pneumonia + '/' + images_pneumonia[i]
        
        # Ouvrir l'image
        img = Image.open(chemin_image)
        img1=Image.open(chemin_image1)
        # Afficher l'image dans la case [0, i] (ligne 0, colonne i)
        axes[0, i].imshow(img, cmap='gray')
        axes[1, i].imshow(img1, cmap='gray')
        
        # Mettre un titre
        axes[0, i].set_title('NORMAL', color='green', fontsize=14, fontweight='bold')
        axes[1, i].set_title('PNEUMONIA', color='red', fontsize=14, fontweight='bold')
        
        # Enlever les axes
        axes[0, i].axis('off')
    
    # Titre général
    fig.suptitle('Exemples de Radiographies', fontsize=16, fontweight='bold')
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    # Afficher !
    plt.show()

#afficher_exemples()

### Partie 2: 