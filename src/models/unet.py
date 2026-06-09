import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
    Cropping2D, BatchNormalization, Activation, SpatialDropout2D
)
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size):
    """Convolutional block with two Conv2D + BatchNorm + ReLU layers."""
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_unet(input_shape, base_filters=64, kernel_size=(3,3), dropout_rate=0.1):
    """Builds a U-Net model with variable base filters, kernel size, and dropout."""
    inputs = Input(input_shape)
    
    # Padding to handle odd dimensions
    padded_inputs = tf.keras.layers.ZeroPadding2D(((0,1),(0,0)))(inputs)
    
    # Encoder
    c1 = conv_block(padded_inputs, base_filters, kernel_size)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = conv_block(p1, base_filters*2, kernel_size)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = conv_block(p2, base_filters*4, kernel_size)
    p3 = MaxPooling2D((2,2))(c3)
    
    # Bottleneck
    bnx = conv_block(p3, base_filters*8, kernel_size)
    bn = SpatialDropout2D(dropout_rate)(bnx)
    
    # Decoder
    u1 = Conv2DTranspose(base_filters*4, kernel_size, strides=(2,2), padding='same', activation='relu')(bn)
    m1 = concatenate([u1, c3])
    c4 = conv_block(m1, base_filters*4, kernel_size)
    
    u2 = Conv2DTranspose(base_filters*2, kernel_size, strides=(2,2), padding='same', activation='relu')(c4)
    m2 = concatenate([u2, c2])
    c5 = conv_block(m2, base_filters*2, kernel_size)
    
    u3 = Conv2DTranspose(base_filters, kernel_size, strides=(2,2), padding='same', activation='relu')(c5)
    m3 = concatenate([u3, c1])
    c6 = conv_block(m3, base_filters, kernel_size)
    
    # Output
    outputs = Conv2D(1, (1,1), activation='linear')(c6)
    outputs = Cropping2D(((0,1),(0,0)))(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
