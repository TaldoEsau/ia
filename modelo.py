import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Criar gerador de dados
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory('dataset', target_size=(224, 224), class_mode='categorical')

# Criar modelo
modelo = tf.keras.Sequential([
    tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(train_data, epochs=5)

# Salvar o modelo
modelo.save('modelo_placas.h5')
