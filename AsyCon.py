from tensorflow.keras.datasets import cifar10,cifar100   
import tensorflow as tf
#from keras import ops
from tensorflow.keras import layers, Model,optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
def alignment_loss_keras(p, z, y):
    y = tf.reshape(y, [-1, 1])
    
    label_mask = tf.cast(tf.equal(y, tf.transpose(y)), dtype=tf.float32)
    
    p_norm = tf.math.l2_normalize(p, axis=-1)
    z_norm = tf.math.l2_normalize(z, axis=-1)
    
    cosine_similarity = tf.matmul(p_norm, z_norm, transpose_b=True)
    

    cosine_pos_sum = tf.reduce_sum(cosine_similarity * label_mask, axis=1)
    pos_count = tf.reduce_sum(label_mask, axis=1) + 1e-6
    cosine_pos_mean = cosine_pos_sum / pos_count
    
    loss = -1.0 * tf.reduce_mean(cosine_pos_mean)
    return loss

def uniform_loss_keras(z0, z1, y, tmp=0.5):
    y = tf.reshape(y, [-1, 1])
    
    label_mask = tf.cast(tf.equal(y, tf.transpose(y)), dtype=tf.float32)
    neg_mask = 1.0 - label_mask
    
    neg_num = tf.cast(tf.shape(y)[0], dtype=tf.float32) - tf.reduce_sum(label_mask, axis=1)
    
    z0_norm = tf.math.l2_normalize(z0, axis=-1)
    z1_norm = tf.math.l2_normalize(z1, axis=-1)
    
    dot_product = tf.matmul(z0_norm, z1_norm, transpose_b=True)
    
    neg = dot_product * neg_mask
    
   
    exp_neg = tf.math.exp(neg / tmp) * neg_mask 
    sum_exp_neg = tf.reduce_sum(exp_neg, axis=1)
    
    loss = tf.math.log(sum_exp_neg / (neg_num + 1e-6) + 1e-6)
    return tf.reduce_mean(loss)

def get_projection_head(input_dim):
    return tf.keras.Sequential([
        layers.Dense(512, input_shape=(input_dim,), use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization()
    ], name="projection_head")

def get_prediction_head(input_dim=128):
    return tf.keras.Sequential([
        layers.Dense(64, input_shape=(input_dim,), use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(128)
    ], name="prediction_head")

class AsyCon(Model):
    
    def __init__(self, backbone_model, num_ftrs, **kwargs):
        super(AsyCon, self).__init__(**kwargs)
        self.backbone = backbone_model
        self.projection_head = get_projection_head(num_ftrs)
        self.prediction_head = get_prediction_head(128)

    def call(self, x, training=False):
        features = self.backbone(x, training=training)
        
        z = self.projection_head(features, training=training)
        
        p = self.prediction_head(z, training=training)

        if not training:
            return p
        
        return z, p
    


class AsyConTrainer(AsyCon): 
    def __init__(self, backbone, num_ftrs, lamb, tmp):
        super().__init__(backbone, num_ftrs)
        self.lamb = lamb
        self.tmp = tmp
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ssl_tracker = tf.keras.metrics.Mean(name="ssl_loss")
        self.neg_tracker = tf.keras.metrics.Mean(name="neg_loss")

    def train_step(self, data):
        (x0, x1), y = data

        with tf.GradientTape() as tape:
            z0, p0 = self(x0, training=True)
            z1, p1 = self(x1, training=True)

            loss_ssl = 0.5 * (
                alignment_loss_keras(p0, tf.stop_gradient(z1), y) + 
                alignment_loss_keras(p1, tf.stop_gradient(z0), y)
            )

            loss_neg = uniform_loss_keras(z0, z1, y, tmp=self.tmp) * self.lamb

            total_loss = loss_ssl + loss_neg

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(total_loss)
        self.ssl_tracker.update_state(loss_ssl)
        self.neg_tracker.update_state(loss_neg)
        
        return {
            "loss": self.loss_tracker.result(),
            "ssl": self.ssl_tracker.result(),
            "neg": self.neg_tracker.result()
        }

def main_keras(backbone, train_dataset, args):
    model = AsyConTrainer(backbone, num_ftrs=args.num_ftrs, lamb=args.lamb, tmp=args.tmp)


    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=args.lr,
    decay_steps=args.epochs * (len(X_train) // args.batch_size))
    optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    model.compile(optimizer=optimizer)

    model.fit(train_dataset, epochs=args.epochs)
    
    model.backbone.save_weights('asycon_backbone.h5')


class Args:
    num_ftrs = 640    
    lamb = 1.0       
    tmp = 0.5         
    lr = 0.1         
    epochs = 100
    batch_size = 1024

args = Args()


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

def prepare_dataset(images, labels, batch_size):
    images = images.astype("float32") / 255.0
    def augment(x, y):
        
        return (data_augmentation(x, training=True), 
                data_augmentation(x, training=True)), y

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(1000).batch(batch_size)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)
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
if __name__ == "__main__":
    backbone = backbone_model()
    args.num_ftrs = 512 
    (X_train, y_train), (test_images, test_labels) = cifar10.load_data()
    train_ds = prepare_dataset(X_train, y_train, args.batch_size)
    main_keras(backbone, train_ds, args)
