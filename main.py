########################################
#                                      #
#  DÉTECTION AUTOMATIQUE DE            #
#  PNEUMONIE PAR DEEP LEARNING         #
#                                      #
########################################



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
    
    
    fig.suptitle('Exemples de Radiographies', fontsize=16, fontweight='bold')
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    
    plt.show()

#afficher_exemples()

### Partie 2: Prétraitement 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def pretraiter_image(chemin_image):
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% pour validation
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False # True utile pour des image symetrique par exemple visage d'un humai
)

    train_gen = train_datagen.flow_from_directory(
    chemin_image,
    target_size=(150, 150),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary",
    subset="training"
)

    val_gen = train_datagen.flow_from_directory(
    chemin_image,
    target_size=(150, 150),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary",
    subset="validation"
)
    
    return train_gen,val_gen

#train_gen, val_gen = pretraiter_image(TRAIN_DIR)

########################################################
# Quelques verifications sur le data set d'entrainement#
########################################################
# Récupérer un batch (x_batch, y_batch) genre un lot d'image
"""
print("\nQuelques verifications sur le data set d'entrainement")
x_batch, y_batch = next(train_gen)

print(x_batch.shape)
print(y_batch.shape)

# Exemple : accéder à la première image du batch
image_0 = x_batch[0]        # array 150x150x1
label_0 = y_batch[0]        # 0 ou 1

# Pour afficher
import matplotlib.pyplot as plt
plt.imshow(image_0.squeeze(), cmap='gray')  # squeeze pour enlever la dim 1
plt.title(f"Label: {label_0}")
plt.show()
"""


### Partie 2: Construire le modèle CNN


# Utilisation optimale du CPU
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)
# !!! En fait j'utilise du cpu ici car j'ai pas de graphique nvidia (cuda), je suis sur une vieille machine 😊


tf.get_logger().setLevel('ERROR')
print("TensorFlow configuré pour utiliser 6 threads en parallèle.")


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

##################################################
#    Input (150x150x1)
#        ↓
#    [Conv2D(32) + MaxPooling + Dropout + Padding]  ← Bloc 1
#        ↓
#    [Conv2D(64) + MaxPooling + Dropout]  ← Bloc 2
#        ↓
#    [Conv2D(128) + MaxPooling ] ← Bloc 3
#        ↓
#    [Flatten]
#        ↓
#    [Dense(50) + ReLU + Dropout]
#        ↓
#    [Dense(1) + Sigmoid] ← Sortie : 0 ou 1
##################################################

def construire_modele():
    """Construit l'architecture du CNN"""
    print("\nConstruction du modèle CNN...")
    
    # Créer un modèle séquentiel 
    model=Sequential()
    
    # Entrée
    model.add(layers.Input(shape=(150, 150, 1)))
     
    # Bloc 1
    model.add(layers.Conv2D(32,(3, 3),activation='relu',padding='same')) # Ajout de padding pour ne pas perdre trop d’informations spatiales
    model.add(layers.MaxPooling2D((2, 2)))  # Réduit la taille de moitié
    model.add(layers.Dropout(0.25))  # Dropout de 25%
    # Bloc 2
    model.add(layers.Conv2D(64,(3, 3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  
    model.add(layers.Dropout(0.25))  
    # Bloc 3
    model.add(layers.Conv2D(128,(3, 3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  
    model.add(layers.Dropout(0.20))  
   
    # FLATTEN : Transformer en vecteur 1D 
    model.add(layers.Flatten())
    
    # DENSE : Classification
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.5))

    # OUTPUT : 1 neurone avec Sigmoid
    model.add(layers.Dense(1, activation='sigmoid'))
    # Afficher l'architecture
    print("\nArchitecture du modèle :")
    model.summary()
    
    return model

# Construire le modèle

#model=construire_modele()

### Partie 3 : Entrainement du model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

def compiler_modele(model):
    print("\nCompilation du modele")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']

    )
    print("Fin du compilation")
    return model

#model=compiler_modele(model)

def entrainer_modele(model,train_gen,val_gen):
    print("\n....Debut de l'entrainement")

    class_weight={0:1.945,1:0.673}
    # On ajoute ici class_weight pour resoudre le probleme de la desequilibre du dataset trainning

    checkpoint = ModelCheckpoint(
    'meilleur_modele.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


    history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[checkpoint, early_stop],
    verbose=1,
    class_weight=class_weight
)
    
    print("\n....Fin de l'entrainement")

    return history


#history = entrainer_modele(model, train_gen, val_gen)

# Petit graphe pour voir l'evolution de 'accuracy'

import matplotlib.pyplot as plt

# Courbe de l'accuracy 
"""
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='red')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
plt.title("Évolution de l'Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("Courbe_accuracy.png")
plt.show()
"""
# Courbe de la loss
"""
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title("Évolution de la Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("Courbe_loss.png")
plt.show()
"""


### Partie 4: Test sur des data 

# je vais faire mon premier test sur des données jamais vue par le modele

# a) Petit stats sur les données de test 
# Chemins vers les données
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
VAL_DIR = 'data/val'

#afficher_stats()

# la sortie 
""" TEST:
  - Normal: 234
  - Pneumonia: 390
  - Total: 624 """


#b) Test 

from tensorflow.keras.models import load_model

print("\n...Debut des test")
model=load_model('meilleur_modele.keras')

test_datagen=ImageDataGenerator(rescale=1./255.0)

test_gen=test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150,150),   
    color_mode="grayscale",         
    batch_size=16,
    class_mode="binary",
    shuffle=False   
)

# Evaluer le modele

test_loss, test_accuracy=model.evaluate(test_gen)

print("\nResultat sur les premieres test du model")
print(f"\nLa perte (loss):{test_loss:.4f}")
print(f"\nLa precision (accuracy):{test_accuracy*100:.2f}")


# Faire des prediction

predictions=model.predict(test_gen)

y_pred=(predictions>0.5).astype(int).flatten()

y_tru=test_gen.classes

print("la taille de y_pred", len(y_pred))
print("la taille de y_tru", len(y_tru))


# Un peu stats

correct=0
mauvais=0
for i in range(0,len(y_pred)):
    if y_tru[i]==y_pred[i]:
        correct+=1

print("Les bonnes predictions sur le dataset test (Total: 624) est:", correct)


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_tru, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.savefig("Matrice_de_conf.png")
plt.show()

print("\nRapport de classification :")
print(classification_report(y_tru, y_pred, target_names=['Normal', 'Pneumonia']))