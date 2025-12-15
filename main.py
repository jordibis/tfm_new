#IMPORTS NECESSARIS
from tensorflow.keras.datasets import cifar10,cifar100
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras import layers, models, Model, Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
def save_unique(filepath, **savefig_kwargs):
    """
    Guarda la figura dins de 'filepath'. Si ja existeix, afegeix un sufix numèric per evitar sobreescriure.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    # si existeix el fitxer, afegir un sufix numèric
    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1

    plt.savefig(new_filepath, **savefig_kwargs)
    print(f"Guardado como: {new_filepath}")
class IncrementalLearning:
    def __init__(self,metodo,dataset,NC_NIC,CNN,pretr_model=None,save_plots=False):
        self.metodo=metodo
        self.dataset=dataset
        self.tipo_CNN=CNN
        self.NC_NIC=NC_NIC
        self.pretrained=pretr_model
        self.save_plots=save_plots
       
        if self.dataset=='cifar10':
            (train_images, self.train_labels), (test_images, self.test_labels) = cifar10.load_data()
        if self.dataset=='cifar100':
            (train_images, self.train_labels), (test_images, self.test_labels) = cifar100.load_data()
        
        self.train_images = train_images.astype("float32") / 255.0
        self.test_images = test_images.astype("float32") / 255.0
 
    def loss_lwf(self, y_true, y_pred):
        """funció de pèrdua LwF amb lambda_lwf i y_lwf actualitzats"""
        if self.y_lwf is None:
            return categorical_crossentropy(y_true, y_pred)
        y_combined = tf.add(tf.cast(self.lambda_lwf, y_true.dtype) * tf.cast(self.y_lwf, y_true.dtype),
                    tf.cast((1 - self.lambda_lwf), y_true.dtype) * y_true)
        return categorical_crossentropy(y_combined, y_pred)
    
    def update_lambda_lwf(self, old_classes,num_classes):
        """Actualitza el valor de lambda_lwf segons la fórmula donada"""
        new_lambda = (old_classes/num_classes)**0.5
        return new_lambda
    
    def on_train_batch_begin_lwf(self,input_data,model):
        """Guarda les prediccions del model antic per a les dades d'entrada donades""" 
                
        self.y_lwf = model(input_data,training=False)
        

    ##FUNCIONS COMUNS PER A LA GESTIÓ DELS DADES
    def extract_from_class(self,tag,size,images,labels):
        """Extrau 'size' mostres de la classe 'tag' de les dades donades."""
        images_=[]
        labels_=[]
        for ind, value in enumerate(labels):
            if len(images_)>=size:
                break
            if value == tag:
                images_.append(images[ind])
                labels_.append(labels[ind])

        images_ = np.array(images_, dtype=images.dtype) 
        labels_ = np.array(labels_, dtype=labels.dtype)  
        return images_, labels_

    def data_batch(self,num_of_classes, train_images, train_labels,batch_size):
        """Crea un batch equilibrat amb mostres de cada classe."""
        size = batch_size // num_of_classes  

        train_images_batch = []
        train_labels_batch = []

        for num_class in range(num_of_classes):
            images, labels = self.extract_from_class(num_class, size, train_images, train_labels)
            train_images_batch.append(images)
            train_labels_batch.append(labels)

        train_images_batch = np.concatenate(train_images_batch, axis=0)
        train_labels_batch = np.concatenate(train_labels_batch, axis=0)

        return train_images_batch, train_labels_batch
    
    ##FUNCIONS COMUNS PARA LA CONSTRUCCIÓ DE LA CNN
    def build_model(self,num_classes):
        """Construeix el model segons el tipus de CNN especificat."""
        if self.tipo_CNN=='cifarnet_mod':
            model = Sequential()

            # --- BLOC 1 ---
            model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # --- BLOC 2 ---
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

           

            # --- CLASSIFICADOR ---
            model.add(Flatten())

            model.add(Dense(num_classes, activation='softmax'))

            return model
        if self.tipo_CNN=='cifarnet':
            model = Sequential()

            # --- BLOC 1 ---
            model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # --- BLOC 2 ---
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

           

            # --- CLASSIFICADOR ---
            model.add(Flatten())
            model.add(Dense(512))
            model.add(LeakyReLU(0.1))
            model.add(Dropout(0.5))

            model.add(Dense(num_classes, activation='softmax'))

            return model
        if self.tipo_CNN=='lenet':
            model = models.Sequential([
            tf.keras.Input(shape=(32, 32, 3)),

            layers.Conv2D(32, (5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(48, (5, 5), padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),

            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),

            layers.Dense(84),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.4),

            layers.Dense(num_classes, activation='softmax')])


            return model
        if self.tipo_CNN=='new':
            model = Sequential()
            model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(256))
            model.add(LeakyReLU(0.1))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))
    
            return model
        if self.tipo_CNN=='lenet_mod':
            model = models.Sequential([
            tf.keras.Input(shape=(32, 32, 3)),

            layers.Conv2D(32, (5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(48, (5, 5), padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),

            layers.Dense(num_classes, activation='softmax')
        ])

            return model

    def build_backbone(self):
        """Construeix el model backbone per a l'extracció de característiques."""
        inputs = Input(shape=(32, 32, 3), name="backbone_input")
        x = inputs

        # --- BLOC 1 ---
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # --- BLOC 2 ---
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # --- CLASSIFICADOR ---
        x = Flatten()(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.5)(x)

        outputs = x

        model = Model(inputs=inputs, outputs=outputs, name="backbone")
        return model
    
    def build_head(self, input_shape,name, init_value=0.01):
        """construeix el model head per a la classificació."""
        initializer = tf.keras.initializers.Constant(init_value)
    
        inp = Input(shape=input_shape, name="head_input")
        
        x = layers.Dense(
            128, activation='relu',
            kernel_initializer=initializer, bias_initializer=initializer,
            name="dense_128"
        )(inp)

        x = layers.Dense(
            64, activation='relu',
            kernel_initializer=initializer, bias_initializer=initializer,
            name="dense_64"
        )(x)

        out = layers.Dense(
                1, activation='sigmoid',
            kernel_initializer=initializer, bias_initializer=initializer,
            name="dense_out"
        )(x)

        return Model(inputs=inp, outputs=out, name=name)
    #FUNCIONS AUXILIARS PER ALGUNS MÈTODES
    def fisher_diagonal(self,X, y, model, loss_fn=tf.keras.losses.categorical_crossentropy, batch_size=10, max_samples=1000):
        """
        Calcula la diagonal de Fisher per al model donat utilitzant les dades (X, y).
        Utilitza mini-batches per a l'eficiència de memòria.
        Args:
            X: Dades d'entrada (numpy array).
            y: Etiquetes corresponents (numpy array).
            model: Model de Keras entrenat.
            loss_fn: Funció de pèrdua per calcular els gradients.
            batch_size: Mida del mini-batch per al càlcul dels gradients.
            max_samples: Nombre màxim de mostres a utilitzar per al càlcul.
        Returns:
            fisher_diag: Diagonal de Fisher com un vector numpy.
        """
        n = X.shape[0]
        num_samples = min(n, max_samples)

        # Mezclar índices de forma aleatoria
        indices = np.random.permutation(n)[:num_samples]
        X_sampled = X[indices]
        y_sampled = y[indices]

        fisher_diags = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_grads = []

            for i in range(start, end):
                flat_grad = self.step_f(X_sampled[i], y_sampled[i], model, loss_fn)  
                batch_grads.append(flat_grad)

            grads_tensor = tf.stack(batch_grads, axis=0)
            batch_var = tf.math.reduce_variance(grads_tensor, axis=0)
            fisher_diags.append(batch_var)

        fisher_diag = tf.reduce_mean(tf.stack(fisher_diags, axis=0), axis=0)

        return fisher_diag

    def sumar_vectores_desiguales(self,a, b):
        """
        Suma dos vectors de numpy de longitud diferent
        """
        len_a = len(a)
        len_b = len(b)
        min_len = min(len_a, len_b)

        suma_comun = a[:min_len] + b[:min_len]

        if len_a > len_b:
            suma_total = np.concatenate([suma_comun, a[min_len:]])
        else:
            suma_total = np.concatenate([suma_comun, b[min_len:]])

        return suma_total

    def sumar_vectores_desiguales_tf(self,a, b):

        a = tf.convert_to_tensor(a)
        b = tf.convert_to_tensor(b)
        len_a = tf.shape(a)[0]
        len_b = tf.shape(b)[0]
        min_len = tf.minimum(len_a, len_b)
        suma_comun = a[:min_len] + b[:min_len]

        def concat_a():
            return tf.concat([suma_comun, a[min_len:]], axis=0)
        def concat_b():
            return tf.concat([suma_comun, b[min_len:]], axis=0)

        return tf.cond(len_a > len_b, concat_a, concat_b)

    def actualizar_Fisher(self,X,y,model,actual_num_class,num_batch,actual_fisher,max_F,m=False,epsilon=1e-6):
        """
        Actualitza la diagonal de Fisher utilitzant les dades (X, y) i el model donat.
        Args:
            X: Dades d'entrada (numpy array).
            y: Etiquetes corresponents (numpy array).
            model: Model de Keras entrenat.
            actual_num_class: Nombre actual de classes.
            num_batch: Nombre de batches utilitzats per al càlcul.
            actual_fisher: Diagonal de Fisher actual (numpy array) o None si no existeix.
            max_F: Valor màxim per al clip del Fisher.
            m: Si és True, retorna només el Fisher calculat en aquesta crida.
            epsilon: Valor petit per evitar divisió per zero.
        Returns:
            fisher_hat: Diagonal de Fisher actualitzada com un vector numpy.
        """
        fisher_diag=self.fisher_diagonal(X,y,model)
        if m:
            fisher=fisher_diag

        else:
            fisher=self.sumar_vectores_desiguales(actual_fisher, fisher_diag)


        F_frac = fisher / num_batch

        fisher_hat = np.clip(F_frac, 0, max_F)


        return fisher_hat

    def step_f(self,X, y,model,loss_fn):
        """
        Calcula els gradients de la pèrdua respecte als pesos del model per a una sola mostra (X, y).
        """
        with tf.GradientTape() as tape:
            pred = model(tf.expand_dims(X, axis=0), training=False)
            loss = loss_fn(tf.expand_dims(y, axis=0), pred)

        grads = tape.gradient(loss, model.trainable_variables)

        # Aplanar cad gradient i concatenarlos en un sol vector
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return flat_grads

    def get_flat_weights(self,weights):
        """
        Retorna tots els pesos entrenables com un únic vector numpy (float32).
        """

        flat_weights = tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
        return flat_weights

    def get_flat_weights_tf(self,weights):

        if weights is None or len(weights) == 0:
            return None
        try:
            flat_weights = tf.concat(
                [tf.reshape(tf.cast(w, tf.float32), [-1]) for w in weights],
                axis=0
            )
            return flat_weights
        except Exception:
            return None
   
    def EWC_loss_penalty(self,lambda_, fisher_matrix, opt_weights, model):
        """
        Calcula el terme de regularització EWC.
        Si fisher_matrix u opt_weights no estan definits encara, retorna 0.0
        """
        current_weights = model.get_weights()
        current_weights = np.concatenate([w.flatten() for w in current_weights])
        opt_weights = [ w.numpy() if isinstance(w, tf.Tensor) else w for w in opt_weights ]
        opt_weights = np.concatenate([w.flatten() for w in opt_weights])


        min_len = min(len(current_weights), len(opt_weights),len(fisher_matrix))
        current_weights = current_weights[:min_len]
        opt_weights = opt_weights[:min_len]
        fisher_matrix = fisher_matrix[:min_len]

        weight_diff = current_weights - opt_weights



        # Terme EWC
        ewc_term = 0.5 * lambda_ * tf.reduce_sum(fisher_matrix * tf.square(weight_diff))

        return ewc_term
    
    def loss_ewc(self,y_true, y_pred):
        """
        Funció de pèrdua EWC amb fisher_matrix i opt_weights actualitzats
        """
        if self.step>1:
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            penalty = self.EWC_loss_penalty(self.lambda_real, self.fisher_matrix, self.opt_weights, self.model)
            return ce_loss + penalty
        else:
            return categorical_crossentropy(y_true, y_pred)

    def SI_fisher_diagonal(self,initial_weights, previous_weights, actual_weights, grads, delta_L, xi=1e-7):
        """
        Calcula la diagonal de Fisher segons el mètode SI (Synaptic Intelligence).
        Retorna fisher_diag (tf.Tensor) i delta_L (tf.Tensor).
        Args:
            initial_weights: Pesos inicials del model (l'últim abans de començar l'aprenentatge incremental).
            previous_weights: Pesos del model abans de l'última actualització.
            actual_weights: Pesos actuals del model.
            grads: Gradients actuals del model.
            delta_L: Delta L acumulat fins ara (tf.Tensor) o None si no existeix.
            xi: Valor petit per a l'estabilització numèrica.
        Returns:    
            fisher_diag: Diagonal de Fisher com un tf.Tensor.
            delta_L: Delta L actualitzat com un tf.Tensor.

        """
        # Convertir a tensores planos
        old_weights = self.to_flat_tf(previous_weights)
        new_weights = self.to_flat_tf(actual_weights)

        # Si alguno es None => devolvemos fisher_diag=0 y mantenemos delta_L
        if old_weights is None or new_weights is None:
            fisher_diag = tf.zeros(1, dtype=tf.float32)
            if delta_L is None:
                delta_L = tf.zeros(1, dtype=tf.float32)
            return fisher_diag, delta_L

        # Calcular delta_theta
        delta_theta = tf.abs(self.sumar_vectores_desiguales_tf(new_weights, -old_weights))

        # Aplanar gradientes
        flat_grads = self.to_flat_tf(grads)
        if flat_grads is None:
            flat_grads = tf.zeros_like(delta_theta)

        # Ajustar longitudes
        min_len = tf.minimum(tf.shape(delta_theta)[0], tf.shape(flat_grads)[0])
        delta_theta = delta_theta[:min_len]
        flat_grads = flat_grads[:min_len]

        # Delta L actual
        delta_L_now = tf.abs(delta_theta * flat_grads)
        if delta_L is None:
            delta_L = delta_L_now
        else:
            delta_L = tf.convert_to_tensor(delta_L, dtype=tf.float32)
            delta_L = self.sumar_vectores_desiguales_tf(delta_L, delta_L_now)

        # Diferencia total T_k
        initial_w_tf = self.to_flat_tf(initial_weights)
        if initial_w_tf is None:
            fisher_diag = tf.zeros_like(delta_theta)
        else:
            min_len2 = tf.reduce_min([tf.shape(new_weights)[0],
                          tf.shape(initial_w_tf)[0],
                          tf.shape(delta_L)[0]])
            T_k = new_weights[:min_len2] - initial_w_tf[:min_len2]
            fisher_diag = delta_L[:min_len2] / (tf.square(T_k) + xi)

        return fisher_diag, delta_L
    
    def actualizar_Fisher_SI(self,initial_weights,previous_weights,actual_weights,grads, delta_L ,w_i, actual_fisher, max_F, xi=1e-7):
        """
        Actualitza la diagonal de Fisher segons el mètode SI (Synaptic Intelligence).
        Args:
            initial_weights: Pesos inicials del model (l'últim abans de començar l'aprenentatge incremental).
            previous_weights: Pesos del model abans de l'última actualització.
            actual_weights: Pesos actuals del model.
            grads: Gradients actuals del model.
            delta_L: Delta L acumulat fins ara (tf.Tensor) o None si no existeix.
            w_i: Pes específic per a aquesta tasca.
            actual_fisher: Diagonal de Fisher actual (tf.Tensor) o None si no existeix.
            max_F: Valor màxim per al clip del Fisher.
            xi: Valor petit per a l'estabilització numèrica.
        Returns:
            fisher_hat: Diagonal de Fisher actualitzada com un tf.Tensor.
            delta_L: Delta L actualitzat com un tf.Tensor.
        """
        try:
            if actual_fisher is None:
                fisher, delta_L = self.SI_fisher_diagonal(initial_weights, previous_weights, actual_weights, grads, delta_L, xi=xi)
            else:
                fisher_diag, delta_L = self.SI_fisher_diagonal(initial_weights, previous_weights, actual_weights, grads, delta_L, xi=xi)
                actual_fisher_tf = tf.convert_to_tensor(actual_fisher, dtype=tf.float32)
                added = w_i * fisher_diag
                fisher = self.sumar_vectores_desiguales_tf(actual_fisher_tf, added)
        except Exception as e:
            fisher_diag, delta_L = self.SI_fisher_diagonal(initial_weights, previous_weights, actual_weights, grads, delta_L, xi=xi)
            if actual_fisher is None:
                fisher = fisher_diag
            else:
                actual_fisher_tf = tf.convert_to_tensor(actual_fisher, dtype=tf.float32)
                fisher = self.sumar_vectores_desiguales_tf(actual_fisher_tf, w_i * fisher_diag)

        fisher_hat = tf.clip_by_value(fisher, clip_value_min=0.0, clip_value_max=max_F)
        
        return fisher_hat, delta_L
    
    def SI_loss_penalty(self,lambda_, fisher_matrix, opt_weights, model):
        """
        Calcula el terme de regularització SI.
        """
        # Caso inicial: todavía no tenemos fisher ni pesos óptimos
        if fisher_matrix is None or opt_weights is None:
            return tf.constant(0.0, dtype=tf.float32)

        # Aplanar pesos actuales y óptimos como tensores
        current_weights = self.get_flat_weights_tf(model.trainable_variables)
        opt_weights_tf = self.get_flat_weights_tf(opt_weights)

        if current_weights is None or opt_weights_tf is None:
            return tf.constant(0.0, dtype=tf.float32)

        # normalizar longitudes y convertir fisher a tf.Tensor
        fisher_tf = tf.convert_to_tensor(fisher_matrix, dtype=tf.float32)

        min_len = tf.minimum(
            tf.shape(current_weights)[0],
            tf.minimum(tf.shape(opt_weights_tf)[0], tf.shape(fisher_tf)[0])
        )

        current_weights = current_weights[:min_len]
        opt_weights_tf = opt_weights_tf[:min_len]
        fisher_tf = fisher_tf[:min_len]

        # Diferencia de pesos
        weight_diff = current_weights - opt_weights_tf

        # Término SI
        si_term = 0.5 * lambda_ * tf.reduce_sum(fisher_tf * tf.square(weight_diff))
        return si_term
    
    def loss_si(self,y_true, y_pred):
        """
        Funció de pèrdua SI amb fisher_matrix i opt_weights actualitzats
        """
        if self.step>1:
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            penalty = self.SI_loss_penalty(self.lambda_real, self.fisher_matrix, self.opt_weights, self.model)
            return ce_loss + penalty
        else:
            return categorical_crossentropy(y_true, y_pred)
    
    def histograma_F(self,fisher):
        """
        Mostra un histograma dels valors de la diagonal de Fisher.
        """
        plt.hist(fisher, bins=30, color='skyblue', edgecolor='black')
        plt.title('Histograma de valores de F')
        plt.yscale('log')  
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.show()

    def to_flat_tf(self,weights):
        """
        Converteix els pesos donats a un únic vector tf.Tensor aplanat (float32).
        """
        if weights is None:
            return None
        flat_parts = []
        if isinstance(weights, np.ndarray):
            return tf.convert_to_tensor(weights.flatten().astype(np.float32))
        for w in weights:
            if isinstance(w, (np.ndarray,)):
                arr = w.astype(np.float32).reshape(-1)
                flat_parts.append(tf.convert_to_tensor(arr))
            elif isinstance(w, tf.Variable) or isinstance(w, tf.Tensor):
                flat_parts.append(tf.reshape(tf.cast(w, tf.float32), [-1]))
            else:
                try:
                    arr = np.array(w).astype(np.float32).reshape(-1)
                    flat_parts.append(tf.convert_to_tensor(arr))
                except:
                    raise TypeError("Tipo de peso no reconocido en to_flat_tf: " + str(type(w)))
        if len(flat_parts) == 0:
            return None
        return tf.concat(flat_parts, axis=0)
    #FUNCIONS PER MODIFICAR LA CAPA DE SORTIDA
    def modify_output_layer(self,model, old_num_classes, new_num_classes,custom_loss=categorical_crossentropy):
        """Modifica la capa de sortida del model per adaptar-la a un nou nombre de classes.
        Args:
            model: Model de Keras existent.
            old_num_classes: Nombre de classes actuals.
            new_num_classes: Nou nombre de classes.
            custom_loss: Funció de pèrdua personalitzada (opcional).
        Returns:
            nuevo_modelo: Model de Keras amb la capa de sortida modificada."""
        # Guardar pesos de las capas intermedias (todas menos la última)
        pesos_intermedios = [capa.get_weights() for capa in model.layers[:-1]]

        # Reconstruir el modelo con el nuevo número de clases
    
        #nuevo_modelo = self.build_model_lenet(num_classes=new_num_classes,custom_loss=custom_loss)
        nuevo_modelo = self.build_model(num_classes=new_num_classes)

        # Forzar la construcción del modelo pasando un dummy input
        dummy_input = np.random.rand(1, 32, 32, 3)
        nuevo_modelo.predict(dummy_input)

        # Asignar los pesos guardados a las capas compartidas
        for i in range(len(pesos_intermedios)):
            nuevo_modelo.layers[i].set_weights(pesos_intermedios[i])

        # Inicializar la nueva capa de salida:
        # Obtener los pesos antiguos de la capa de salida
        old_weights, old_bias = model.layers[-1].get_weights()
        # Crear nuevos pesos con la forma (número de neuronas de la penúltima capa, new_num_classes)
        new_weights = np.random.normal(size=(old_weights.shape[0], new_num_classes))
        new_bias = np.zeros((new_num_classes,))

        # Copiar los pesos y sesgos antiguos en las primeras posiciones
        if old_num_classes<new_num_classes:
            new_weights[:, :old_num_classes] = old_weights
            new_bias[:old_num_classes] = old_bias
            # Asignar los nuevos pesos a la capa de salida del nuevo modelo
            nuevo_modelo.layers[-1].set_weights([new_weights, new_bias])
        
        return nuevo_modelo
    
    def modify_output_layer_CWR(self,old_model, old_num_classes, new_num_classes, init_value=0.01):
        """
        Modifica la capa de sortida del model per adaptar-la a un nou nombre de classes segons el mètode CWR.
        """
        if self.step==1:
            pesos_intermedios = [capa.get_weights() for capa in old_model.layers[:-1]]
            nuevo_modelo = self.build_model(num_classes=new_num_classes)
            old_weights, old_bias = old_model.layers[-1].get_weights()  
            # Forzar la construcción del modelo pasando un dummy input
            dummy_input = np.random.rand(1, 32, 32, 3)
            nuevo_modelo.predict(dummy_input)

            # Asignar los pesos guardados a las capas compartidas
            for i in range(len(pesos_intermedios)):
                nuevo_modelo.layers[i].set_weights(pesos_intermedios[i])
            
            # Inicializar la nueva capa de salida:
            new_weights = np.full((old_weights.shape[0], new_num_classes), init_value, dtype=np.float32)
            new_bias = np.full((new_num_classes,), init_value, dtype=np.float32)

            nuevo_modelo.layers[-1].set_weights([new_weights, new_bias])
            return nuevo_modelo,None,None
            
        
        
        else:
            pesos_intermedios = [capa.get_weights() for capa in old_model.layers[:-1]]
            nuevo_modelo = self.build_model(num_classes=new_num_classes)
            old_weights, old_bias = old_model.layers[-1].get_weights()  
                      
            # Forzar la construcción del modelo pasando un dummy input
            dummy_input = np.random.rand(1, 32, 32, 3)
            nuevo_modelo.predict(dummy_input)

            # Asignar los pesos guardados a las capas compartidas
            for i in range(len(pesos_intermedios)):
                nuevo_modelo.layers[i].set_weights(pesos_intermedios[i])
            for layer in nuevo_modelo.layers[:-1]:
                layer.trainable = False
            # Inicializar la nueva capa de salida:
            new_weights = np.full((old_weights.shape[0], new_num_classes), init_value, dtype=np.float32)
            new_bias = np.full((new_num_classes,), init_value, dtype=np.float32)

            nuevo_modelo.layers[-1].set_weights([new_weights, new_bias])
            
            return nuevo_modelo,old_weights, old_bias
    
    def modify_MH_output_layer(self,model,old_num_classes, new_num_classes, input_shape=(32, 32, 3), is_first_epoch=False,freeze_backbone=False):
        """MODIFICA LA CAPA DE SORTIDA DEL MODEL SEGONS EL MÈTODE MULTI-HEAD."""
        total_backbone_layers=len(self.build_backbone().layers)
        
        # Paso 1: recuperar pesos si hay modelo anterior
        if not is_first_epoch:
            base_layers = model.layers[:total_backbone_layers]
            # Guardar pesos del backbone
            pesos_intermedios = [layer.get_weights() for layer in base_layers]
            
        else:
            base_layers = model.layers[:total_backbone_layers]
            pesos_intermedios = None

        # Paso 2: construir nuevo backbone
        backbone_model = self.build_backbone()
        # Restaurar pesos del backbone si existen
        if pesos_intermedios is not None:
            for i, layer in enumerate(backbone_model.layers):
                layer.set_weights(pesos_intermedios[i])
                print(f"Weights set for layer {layer.name}")
                        
        base_output = backbone_model.output
        base_output_shape = base_output.shape[1:]     
        if freeze_backbone:
            # Congelar backbone si no es la primera época
            if not is_first_epoch:
                for layer in backbone_model.layers:
                    layer.trainable = False
        
        # Paso 3: construir heads
        all_heads = []
        head_models = []
        for i in range(new_num_classes):
            head = self.build_head(base_output_shape,name=f"head_class_{i}")            
            head_output = head(base_output)
            all_heads.append(head_output)
            head_models.append(head)

        # Paso 4: construir modelo completo
        final_output = layers.Concatenate(name="multi_head_output")(all_heads)
        new_model = Model(inputs=backbone_model.input, outputs=final_output)

        new_model.summary()

        return new_model
    
    def save_mh_model(self,model_tw,model_cw,num_classes,old_num_classes):
        """Guarda els pesos del model multi-head """
        total_backbone_layers=len(self.build_backbone().layers)

        base_layers = model_tw.layers[:total_backbone_layers]
        base_output_tensor = base_layers[-1].output
        base_output_shape = base_output_tensor.shape[1:]

        # Guardar pesos del backbone
        pesos_intermedios = [layer.get_weights() for layer in base_layers]

        # Guardar pesos de heads anteriores
        pesos_heads = {}
        if old_num_classes>0:
           
            for i in range(old_num_classes):
                
                head_layers = model_cw.get_layer(f"head_class_{i}").layers
                pesos_head_i = [layer.get_weights() for layer in head_layers ]
                pesos_heads[f"head_class_{i}"]=pesos_head_i
                
    
        #normalitzar noves heads
        for i in range(old_num_classes,num_classes):
            head=model_tw.get_layer(f"head_class_{i}")
            pesos_head_i=[]
            for layer in head.layers:
                if len(layer.get_weights()) == 0:
                    pesos_head_i.append([])
                    continue
                print(f"Normalizando pesos de la head {i}, capa {layer.name}")
                tw_pesos, tw_biases = layer.get_weights()
                norm=np.mean(tw_pesos)
                #if norm>0:
                    #tw_pesos-=norm
                
                pesos_head_i.append([tw_pesos, tw_biases])
        #afegir noves heads
            pesos_heads[f"head_class_{i}"]=pesos_head_i

        print("Pesos guardados de backbone y heads:")
        print(f"Backbone layers: {len(pesos_intermedios)}")
        print(f"Heads guardadas:",pesos_heads.keys())
        backbone_model = self.build_backbone()      

        # Restaurar pesos del backbone si existen
        if pesos_intermedios is not None:
            for i, layer in enumerate(backbone_model.layers):
                if len(layer.get_weights()) != 0: 
                    layer.set_weights(pesos_intermedios[i])
                
        base_output = backbone_model.output
             
        # Paso 3: construir heads
        all_heads = []
        head_models = []
        for i in range(num_classes):
            head = self.build_head(base_output_shape,name=f"head_class_{i}")
            # Restaurar pesos 
           
            pesos_head_i = pesos_heads[f"head_class_{i}"]
            for j, layer in enumerate(head.layers):
                
                if len(pesos_head_i[j]) == len(layer.get_weights()):
                    print(f"Restaurando pesos en head {i}, capa {layer.name}") 
                    layer.set_weights(pesos_head_i[j])
        
        
            head_output = head(base_output)
            all_heads.append(head_output)
            head_models.append(head)

        # Paso 4: construir modelo completo
        final_output = layers.Concatenate(name="multi_head_output")(all_heads)
        new_model = Model(inputs=backbone_model.input, outputs=final_output)

        return new_model           
    
    def update_last_layer(self,model_tw, cw_pesos, cw_biases, num_classes,old_num_classes):
        """
        Actualitza la capa de sortida del model .
        """
        tw_pesos, tw_biases = model_tw.layers[-1].get_weights()     
        # Normaliza cada vector de pesos de la última capa
        for i in range(num_classes):
            norm = np.mean(tw_pesos[:, i])
            if norm > 0:
                tw_pesos[:,i] -= norm
     
        # Crear nuevos pesos combinados
        new_w = np.zeros((tw_pesos.shape[0], num_classes))
        new_b = np.zeros((num_classes,))
        
        # Copiar clases anteriores
        new_w[:, :old_num_classes] = cw_pesos[:, :old_num_classes]
        new_b[:old_num_classes] = cw_biases[:old_num_classes]

        # Copiar nuevas clases
        new_w[:, old_num_classes:num_classes] = tw_pesos[:, old_num_classes:num_classes]
        new_b[old_num_classes:num_classes] = tw_biases[old_num_classes:num_classes]

        # Asignar los nuevos pesos
        model_tw.layers[-1].set_weights([new_w, new_b])
        model_tw.layers[-1].trainable = True

        return model_tw
    
    #FUNCIÓ DE TEST
   
    def test(self,test_labels, test_images, model,y=True):


        test_labels_normal = test_labels
        test_predictions = model.predict(test_images)
        predicted_classes_test = np.argmax(test_predictions, axis=1)
        if self.dataset=='cifar10':
            class_names= ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

        if self.dataset=='cifar100':
            class_names = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
                'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
                'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
                'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
                'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
                'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
                'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
                'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
                'wolf', 'woman', 'worm'
            ]
        if y:
            cm_test = confusion_matrix(test_labels_normal, predicted_classes_test)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels= class_names)
            plt.xlabel('Predicción')
            plt.ylabel('Etiqueta Real')
            plt.title('Matriz de Confusión - Conjunto de Test')
            if self.save_plots==False:
                plt.show()
            else:
                filepath=str(self.metodo+'/'+'confusion_matrix.png')
                save_unique(filepath)
            
        accuracy = accuracy_score(test_labels_normal, predicted_classes_test)
        print(f'Accuracy del modelo: {accuracy:.4f}')
        return accuracy
    
    #FUNCIONS DE PREENTRENAMENT I ENTRENAMENT
    def pretrain(self):
        """Preentrena el model amb el conjunt de dades CIFAR-10 si no hi ha un model preentrenat especificat.
        Retorna el model preentrenat o carregat."""
        if self.pretrained==None:
            print("Preentrenant el model amb CIFAR-10")
            model=self.build_model(num_classes=10)
            train_images=self.train_images
            train_labels=self.train_labels
            train_images_batch, train_labels_batch = self.data_batch(10, train_images, train_labels, 10000)
            train_labels_batch = to_categorical(train_labels_batch, num_classes=10)

            indices = np.random.permutation(train_images_batch.shape[0])
            train_images_batch = train_images_batch[indices]
            train_labels_batch = train_labels_batch[indices]

            train_images_final = train_images_batch[:8000]
            train_labels_final = train_labels_batch[:8000]

            val_images = train_images_batch[8000:]
            val_labels = train_labels_batch[8000:]

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9),loss=categorical_crossentropy,metrics=['accuracy'])
            model.fit(train_images_final, train_labels_final, epochs=30, batch_size=64, validation_data=(val_images, val_labels),callbacks=[early_stop])
        
            name_model=str(self.tipo_CNN+'_'+self.dataset)
            model.save('pretrained_models/'+name_model+'.keras')
            return model
        else:
            print("Carregant el model preentrenat...")
            model=models.load_model(self.pretrained)
            return model
    
    
    
    def train(self,num_samples,landa_si=1e7,landa_ewc=0.1,pat=20,min_epoch=10,L_R=0.05,max_F=50,wi=0.005,mh_freeze=False,num_epocas=30, num_classes_ini=2, num_classes_fin=10, incr_class=1,early_stop_th=0.01):
        
        
        #Preparació de les dades d'entrenament
        train_images=self.train_images
        train_labels=self.train_labels


        #Inicialització del model i variables d'entrenament
        do_train=True
        self.model = self.pretrain()
        learning_rate=L_R
        num_steps = num_classes_fin
        num_classes = num_classes_ini
        early_stop_min_epoch=min_epoch

       
       
        #Inicialització de llistes per a l'entrenament
        train_losses = []
        val_losses = []
        accuracies_test=[]   
        accuracies_test_class=[]                
        learning_rate=L_R
        first=True
   
   
        #Bucle d'entrenament incremental
        for step in range(num_steps):
            self.step=step
            if step>1:
                old_classes = num_classes - incr_class
            else:
                old_classes=num_classes
            #Actualització de les dades d'entrenament segons el mode NC o NIC i el pas actual
            if self.NC_NIC=="NIC":
                train_images_batch, train_labels_batch = self.data_batch(num_classes, train_images, train_labels, num_samples) #extracció de dades segons el nombre de classes actual   
                train_labels_batch = to_categorical(train_labels_batch, num_classes=num_classes)

                len_train = train_images_batch.shape[0]
                val_len = int(len_train * 0.2)
                batch_size = int(len_train * 0.05)

                indices = np.random.permutation(train_images_batch.shape[0]) #barreja les dades
                train_images_batch = train_images_batch[indices]
                train_labels_batch = train_labels_batch[indices]

                train_images_final = train_images_batch[:len_train - val_len]
                train_labels_final = train_labels_batch[:len_train - val_len]
                val_images = train_images_batch[len_train - val_len:]
                val_labels = train_labels_batch[len_train - val_len:]


                test_images_batch,test_labels_batch=self.data_batch(num_classes,self.test_images,self.test_labels,num_samples) #extracció de dades de test segons el nombre de classes actual

            elif self.NC_NIC=="NC":
                if step>1:
                    train_images_batch, train_labels_batch = self.extract_from_class(num_classes-1,num_samples/num_classes_fin,train_images,train_labels) #extracció de dades per a les classes actuals
                    
                    train_labels_batch = to_categorical(train_labels_batch, num_classes=num_classes)

                    len_train = train_images_batch.shape[0]
                    val_len = int(len_train * 0.2)

                    indices = np.random.permutation(train_images_batch.shape[0])
                    train_images_batch = train_images_batch[indices]
                    train_labels_batch = train_labels_batch[indices]

                    train_images_final = train_images_batch[:len_train - val_len]
                    train_labels_final = train_labels_batch[:len_train - val_len]
                    val_images = train_images_batch[len_train - val_len:]
                    val_labels = train_labels_batch[len_train - val_len:]


                    test_images_batch,test_labels_batch=self.data_batch(num_classes,self.test_images,self.test_labels,num_samples)
                else:
                     
                    train_images_batch, train_labels_batch = self.data_batch(num_classes, train_images, train_labels, num_samples) #extracció de dades per a les classes actuals
                    train_labels_batch = to_categorical(train_labels_batch, num_classes=num_classes)

                    len_train = train_images_batch.shape[0]
                    val_len = int(len_train * 0.2)

                    indices = np.random.permutation(train_images_batch.shape[0])
                    train_images_batch = train_images_batch[indices]
                    train_labels_batch = train_labels_batch[indices]

                    train_images_final = train_images_batch[:len_train - val_len]
                    train_labels_final = train_labels_batch[:len_train - val_len]
                    val_images = train_images_batch[len_train - val_len:]
                    val_labels = train_labels_batch[len_train - val_len:]


                    test_images_batch,test_labels_batch=self.data_batch(num_classes,self.test_images,self.test_labels,num_samples)

                

            if step==0:
                print(f"Step {step + 1}/{num_steps} - Guardando valores iniciales")
                accuracies_test.append(1/num_classes_fin)
                accuracies_test_class.append(1)

            if step>0:
                print(f"Step {step + 1}/{num_steps} - Entrenando con {num_classes} clases")
                #definció de variables i datasets---------------------------------

               
                best_val_loss=10
                patience=pat
                batch_size=64  
                epochs = num_epocas
                
                
                
                train_dataset = tf.data.Dataset.from_tensor_slices((train_images_final, train_labels_final)).shuffle(1000).batch(batch_size)
                val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)
                #actualització del model segons cada mètode-------------------------------
                
                if self.metodo=='lwf':
                    if step==1:
                        self.model = self.modify_output_layer(self.model, 10, num_classes)

                    else:
                        self.model = self.modify_output_layer(self.model, old_classes, num_classes)

                    old_model = tf.keras.models.clone_model(self.model)
                    old_model.set_weights(self.model.get_weights())
                                        
                    self.lambda_lwf=self.update_lambda_lwf(num_classes-1,num_classes)
                    lwf_loss=self.loss_lwf
                    
                    loss_fn=lwf_loss

                if self.metodo=='ewc':
                    self.lambda_real = min(landa_ewc, 1 / (learning_rate * max_F))
                    if step==1:
                        self.model=self.modify_output_layer(self.model,10,num_classes)
                        self.fisher_matrix=0
                        epochs=10
                        lr=0.01
                    else:
                        lr=L_R

                        self.model=self.modify_output_layer(self.model,old_classes,num_classes)

                        self.fisher_matrix=self.actualizar_Fisher(train_images_final,train_labels_final,self.model,num_classes,step,self.fisher_matrix,max_F,m=first,epsilon=1e-6)


                    self.opt_weights=[tf.identity(var) for layer in self.model.layers for var in layer.trainable_variables]
                    loss_fn=self.loss_ewc



                if self.metodo=='si':
                    self.model=self.modify_output_layer(self.model,old_classes,num_classes)
                    initial_weights =[tf.identity(var) for layer in self.model.layers for var in layer.trainable_variables]


                    if step==1:
                        #w_i=0.00001
                        #self.lambda_real = min(1e-18, 1 / (learning_rate * max_F))
                        
                        epochs=10
                        lr=0.01
                        self.fisher_matrix=None
                        delta_L=None
                    else:
                        w_i=wi
                        self.lambda_real = min(landa_si, 1 / (learning_rate * max_F))/step
                        lr=L_R



                    self.opt_weights=self.model.get_weights()
                    loss_fn=self.loss_si

                if self.metodo=='cwr':
                    lr=L_R
                    do_train=False
                    loss_fn=categorical_crossentropy
                    if step==1:
                        self.model,cw_weights,cw_bias=self.modify_output_layer_CWR(self.model,10,num_classes)
                        epochs=10
                        lr=0.001
                    
                    else:
                        
                        self.model,cw_weights,cw_biases=self.modify_output_layer_CWR(self.model,old_classes,num_classes)
                        
                        if self.NC_NIC=="NIC":
                            ### CAMBIO: actualizar cw_counts SOLO con clases nuevas
                            class_counts_train = np.sum(train_labels_final, axis=0)
                            for classe,count in enumerate(class_counts_train):
                                if classe<num_classes-1:  
                                    cw_counts[classe]+=count
                    self.model.summary()
                
                if self.metodo=='ar1':
                    if step==1:
                        do_train=False
                        loss_fn=categorical_crossentropy
                        delta_L=None
                        self.fisher_matrix=None
                        lr=0.001
                        epochs=15
                        self.model,cw_weights,cw_biases=self.modify_output_layer_CWR(self.model,10,num_classes)
                    
                    else:
                        if step==8 or step==9:
                            epochs=20
                            lr=0.00001
                        self.model,cw_weights,cw_biases = self.modify_output_layer_CWR(self.model, old_classes, num_classes)
                        initial_weights =[tf.identity(var) for layer in self.model.layers[:-1] for var in layer.trainable_variables]
                        w_i=wi
                        #self.lambda_real = min(landa_si, 1 / (learning_rate * max_F))
                        self.lambda_real=landa_si
                        lr=L_R


                        loss_fn=self.loss_si

                    self.opt_weights=self.model.get_weights()
                
                if self.metodo=='armh':
                    initial_weights =[tf.identity(var) for layer in self.model.layers for var in layer.trainable_variables]
                    if step==1:
                        do_train=False
                        loss_fn=categorical_crossentropy
                        delta_L=None
                        self.fisher_matrix=None
                        lr=0.01
                        epochs=25
                        self.model=self.modify_MH_output_layer(self.model,0,num_classes_ini,is_first_epoch=True,freeze_backbone=mh_freeze)
                    else:
                        w_i=wi
                        self.lambda_real = min(landa_si, 1 / (learning_rate * max_F))
                        lr=L_R
                        self.opt_weights=self.model.get_weights()
                        self.model=self.modify_MH_output_layer(self.model,old_classes,num_classes,is_first_epoch=False,freeze_backbone=mh_freeze)
                        loss_fn=self.loss_si

                if self.metodo=='ctive':
                    lr=L_R
                    self.model=self.build_model(num_classes)
                    loss_fn=categorical_crossentropy

                if self.metodo=='naive':
                    lr_=L_R
                    if step==1:
                        self.model=self.modify_output_layer(self.model,10,num_classes)

                    else:
                        self.model=self.modify_output_layer(self.model,old_classes,num_classes)
                    loss_fn=categorical_crossentropy


            


                opt= tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
                #bucle entrenament------------------------------------
                for epoch in range(epochs):
                    print(f"Epoch {epoch + 1}/{epochs}")
                    epoch_loss = []
                    epoch_val_loss = []
                    ##entrenament
                    for X_batch, y_batch in train_dataset:

                        if self.metodo=='lwf':
                            if step==1:
                                self.y_lwf=None
                            else:
                                #print("vamos a cambiar el valor de y_lwf")
                                self.on_train_batch_begin_lwf(X_batch,old_model)
                            lwf_loss=self.loss_lwf
                            loss_fn=lwf_loss

                        if self.metodo=='si':
                            actual_weights = [tf.identity(w) for w in self.model.trainable_variables]
                            if step==1:
                                previous_weights=actual_weights
                            else:
                                self.fisher_matrix,delta_L=self.actualizar_Fisher_SI(initial_weights,previous_weights,actual_weights,grads, delta_L ,w_i, self.fisher_matrix, max_F, xi=1e-7)
                                previous_weights = [tf.identity(w) for w in self.model.trainable_variables]  #para SI
                            
                            
                        if self.metodo=='ar1':
                            actual_weights = [var for layer in self.model.layers[:-1] for var in layer.trainable_variables]
                            
                            if step>1:
                                self.fisher_matrix,delta_L=self.actualizar_Fisher_SI(initial_weights,previous_weights,actual_weights,grads, delta_L ,w_i, self.fisher_matrix, max_F, xi=1e-7)
                                
                            previous_weights = [tf.identity(var) for layer in self.model.layers[:-1] for var in layer.trainable_variables]
                                
                        if self.metodo=='armh':
                            actual_weights = [tf.identity(w) for w in self.model.trainable_variables]
                            if step==1:
                                previous_weights=actual_weights
                            else:
                                self.fisher_matrix,delta_L=self.actualizar_Fisher_SI(initial_weights,previous_weights,actual_weights,grads, delta_L ,w_i, self.fisher_matrix, max_F, xi=1e-7)
                                previous_weights = [tf.identity(w) for w in self.model.trainable_variables]  #para SI
                                #print("fisher matrix", self.fisher_matrix)
                        with tf.GradientTape() as tape:
                            pred = self.model(X_batch, training=do_train)
                            loss = tf.reduce_mean(loss_fn(y_batch,pred))
                            #print("loss batch:", loss.numpy())
                            
                            
                        


                        ############################   
                        
                        
                        
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        #print("grads:", grads)
                        opt.apply_gradients(zip(grads, self.model.trainable_variables))
                        epoch_loss.append(loss.numpy())
                        
                    if epoch==5 and step>1:
                        print("Fisher matrix distribution after 5 epochs:")
                        print("Fisher matrix", self.fisher_matrix)
                        self.histograma_F(self.fisher_matrix)  
                    
                    
                    ##validació
                    for X_val_batch, y_val_batch in val_dataset:
                        pred_val = self.model(X_val_batch, training=False)
                        if self.metodo=='lwf':
                            self.on_train_batch_begin_lwf(X_val_batch,old_model)
                            if step!=1:
                                val_loss = tf.reduce_mean(loss_fn(y_val_batch, pred_val))
                            else:
                                val_loss = tf.reduce_mean(categorical_crossentropy(y_val_batch, pred_val))
                        else:
                            if step!=1:
                                val_loss = tf.reduce_mean(loss_fn(y_val_batch, pred_val))
                            else:
                                val_loss = tf.reduce_mean(categorical_crossentropy(y_val_batch, pred_val))
                        #val_loss=tf.reduce_mean(categorical_crossentropy(y_val_batch, pred_val))
                        epoch_val_loss.append(val_loss.numpy())



                    ##guardar  pèrdues mitjanes
                    mean_train_loss = np.mean(epoch_loss)
                    mean_val_loss = np.mean(epoch_val_loss)

                    train_losses.append(mean_train_loss)
                    val_losses.append(mean_val_loss)

                    model_now = self.model

                    if mean_val_loss < best_val_loss and abs(mean_val_loss-best_val_loss)>early_stop_th:
                        best_val_loss = mean_val_loss
                        patience = pat
                        model_best = model_now
                    else:
                        if patience>0:
                            patience -= 1
                        if epoch>early_stop_min_epoch:
                            if patience == 0:
                                break


                    print(f"Loss: {mean_train_loss:.4f} - Val Loss: {mean_val_loss:.4f}")

                    ##guardar millor model

                self.model=model_best
                
                if step>1 and (self.metodo=='cwr' or self.metodo=='ar1'):                   
                    self.model=self.update_last_layer(self.model,cw_weights, cw_biases, num_classes, old_classes)
                if self.metodo=='armh':
                    if step==1:
                        model_cw=self.save_mh_model(self.model,None,num_classes,0)
                    else:
                        model_cw=self.save_mh_model(self.model,model_cw,num_classes,old_classes)
                    
                    
                    self.model = tf.keras.models.clone_model(model_cw)
                    self.model.set_weights(model_cw.get_weights())





            if step==1 and self.metodo in ['ewc','si','ar1','armh']:
                self.fisher_matrix=self.actualizar_Fisher(train_images_final,train_labels_final,self.model,num_classes,step,0,max_F,m=first,epsilon=1e-6)
            if step>0:
                accuracy_test=self.test(self.test_labels, self.test_images, self.model)
                accuracies_test.append(accuracy_test)
                accuracy_test_2=self.test(test_labels_batch,test_images_batch,self.model,y=False)
                accuracies_test_class.append(accuracy_test_2)

                num_classes += incr_class
          
                first=False

          # Graficar la pérdida
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        if self.save_plots==False:
                plt.show()
        else:
            filepath=str(self.metodo+'/'+'loss.png')
            save_unique(filepath)



        ###IMPORTANT: APART DEL GRÀFIC QUE TORNI TMB EL VECTOR D'ACCURACIIES

        print("Accuracies per a tot el dataset: ",accuracies_test)
        print("Accuracies per a les classes entrenades: ",accuracies_test_class)
        
        #GRAFICO ACCURACY
        inicio = 1
        fin = len(accuracies_test)
        epochs=range(1,11)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs,accuracies_test, marker='x', label='Test Accuracy for all dataset')
        plt.plot(epochs,accuracies_test_class, marker='o', label='Test Accuracy for trained classes')
        # Añadir líneas verticales y etiquetas "2 classes", "3 classes", etc.
        for epoch in range(inicio, fin):
            plt.axvline(x=epoch, color='gray', linestyle='--', linewidth=1)
            plt.text(epoch + 0.1, max(accuracies_test)*0.95, f"{epoch} classes",
                    rotation=90, verticalalignment='top', fontsize=9, color='gray')
        plt.annotate(f"{accuracies_test[-1]:.2f}",
                    xy=(epochs[-1], accuracies_test[-1]),
                    xytext=(epochs[-1] + 0.2, accuracies_test[-1]),
                    textcoords='data',
                    fontsize=9,
                    color='blue',
                    arrowprops=dict(arrowstyle="->", color='blue'))
        # Ajustes del gráfico
        plt.xticks(epochs)
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if self.save_plots==False:
                plt.show()
        else:
            filepath=str(self.metodo+'/'+'accuracy.png')
            save_unique(filepath)
        
                