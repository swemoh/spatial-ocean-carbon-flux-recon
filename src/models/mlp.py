# models/simple_regressor.py
import tensorflow as tf

def build_mlp(base_layer_size=256, dropout_rate=0.1, l2_reg=1e-6, lr=5e-4, activation='relu'):
    """
    Builds and compiles a simple feedforward regression model.
    
    Parameters
    ----------
    base_layer_size : int
        Number of neurons in the first Dense layer.
    dropout_rate : float
        Dropout rate between 0 and 1.
    l2_reg : float
        L2 regularization factor.
    lr : float
        Learning rate for Adam optimizer.

    Returns
    -------
    model : tf.keras.Model
        A compiled Keras model ready for training.
    """
    if activation == 'LeakyReLU':
        activation = tf.keras.layers.LeakyReLU(alpha=0.1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(base_layer_size, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(base_layer_size * 2, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(base_layer_size, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(1, activation='linear')
    ])

 
    return model
