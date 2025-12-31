#%%
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
import os
from keras import ops
#import cv2
#%%
def backbone_model():
    model = Sequential()

    # --- BLOC 1 ---
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # --- BLOC 2 ---
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    # --- CLASSIFICADOR ---
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(0.1))

    return model
def load_asycon_feature_extractor(weight_path):
    base_model = backbone_model()   
    base_model.load_weights(weight_path) 
    return base_model


feature_layer_model = load_asycon_feature_extractor('trained_models/asycon/asycon_backbone.h5')

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

n_samples = 1000
features = feature_layer_model.predict(x_test[:n_samples])
labels = y_test[:n_samples].flatten()


class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 
            'perro', 'rana', 'caballo', 'barco', 'camión']
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(features)

# Dibujar
plt.figure(figsize=(10, 8))
for i in range(10):
    indices = np.where(labels == i) 
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=class_names[i], alpha=0.7)
plt.legend()
plt.title("Espacio de Características (t-SNE) - CIFAR-10 Step 6")
plt.savefig('feature_space2d_asycon_backbone.png')
plt.close()

#%%
last_conv_layer_name = None
for layer in full_model.layers:
    if isinstance(layer, tf.keras.Model) or isinstance(layer, tf.keras.Sequential):
        last_conv_layer_name = [l.name for l in layer.layers if "conv2d" in l.name][-1]
        conv_layer_output = layer.get_layer(last_conv_layer_name).output
        break

grad_model = tf.keras.models.Model(
    inputs=full_model.input,
    outputs=[conv_layer_output, full_model.output[1]] 
)
#%%
folder = 'trained_models/ar1'
os.chdir(folder)

modelos = sorted(os.listdir())
for modelo in modelos:

    full_model = load_model(modelo, compile=False)

    input_layer = Input(shape=(32, 32, 3))

    
    x = input_layer
    for layer in full_model.layers[:-3]: 
        x = layer(x)

    feature_layer_model = Model(inputs=input_layer, outputs=x)
    
    print("Modelo de extracción de características listo.")


    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0

    n_samples = 1000
    features = feature_layer_model.predict(x_test[:n_samples])
    labels = y_test[:n_samples].flatten()
    
    
    class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 
                'perro', 'rana', 'caballo', 'barco', 'camión']
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i in range(10):
        indices = np.where(labels == i) 
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=class_names[i], alpha=0.7)
    plt.legend()
    plt.title("Espacio de Características (t-SNE) - CIFAR-10 Step 6")
    plt.savefig('feature_space2d_'+modelo+'.png')
    plt.close()
# %%
os.chdir('..')
#%%
full_model = load_model('trained_models/ctive_cifarnet_cifar10_step_9.h5', compile=False)
full_model.summary()
#%%
def get_gradcam_heatmap(img_array, grad_model, last_conv_layer_name):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()
# %%
for i in range(10):
    full_model = load_model('trained_models/AsyCLR_cifarnet2_cifar10_step_9.keras', compile=False)

    input_layer = Input(shape=(32, 32, 3))

    last_conv_layer_name = full_model.layers[-9].name
    x = input_layer
    for layer in full_model.layers: 
        x = layer(x)
        if layer.name == last_conv_layer_name:
            x_conv_output = x



    
    grad_model = tf.keras.models.Model(
        inputs=input_layer,
        outputs=[x_conv_output, x]
    )
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0

    idx = np.where(y_test == i)[0][0]
    img = x_test[idx] 
    img_array = np.expand_dims(img, axis=0)

    heatmap = get_gradcam_heatmap(img_array, grad_model, "conv2d_279")

    heatmap_3d = np.expand_dims(heatmap, axis=-1) 
    heatmap_resized = tf.image.resize(heatmap_3d, (32, 32)).numpy()
    heatmap_resized = np.squeeze(heatmap_resized) 

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Imagen Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5) 
    plt.title("¿Dónde mira el modelo? (Rasgos)")
    plt.axis('off')
    plt.savefig('grad_heatmap_'+str(i)+'asy.png')
    plt.close()
    

# %%
