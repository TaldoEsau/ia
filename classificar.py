import os
import numpy as np
import shutil
from PIL import Image
import tensorflow as tf

modelo = tf.keras.models.load_model('modelo_placas.h5')

def classificar_imagem(imagem):
    imagem = Image.open(imagem).resize((224, 224))
    imagem = np.array(imagem) / 255.0
    imagem = np.expand_dims(imagem, axis=0)
    predicao = modelo.predict(imagem)
    return np.argmax(predicao)

def mover_imagem(imagem, classe):
    destino = f'dataset/{classe}'
    if not os.path.exists(destino):
        os.makedirs(destino)
    shutil.move(imagem, destino)

# Testar com uma nova imagem
nova_imagem = r'C:\Users\User\Downloads\img\aa.jpg'  # Substitua 'nome_da_imagem.jpg' pelo nome da sua imagem

resultado = classificar_imagem(nova_imagem)
classe = 'placas' if resultado == 0 else 'outras'
mover_imagem(nova_imagem, classe)
