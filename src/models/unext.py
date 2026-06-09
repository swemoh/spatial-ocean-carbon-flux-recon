import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.utils import register_keras_serializable

# Depthwise convolution used inside ShiftMLP
@register_keras_serializable(package="models.unext")
class DepthwiseConv(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=3, padding='same')

    def call(self, x, *, H, W):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, [B, C, H, W])
        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.dwconv(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [B, C, H * W])
        x = tf.transpose(x, [0, 2, 1])
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
        })
        return config

# MLP with shift operation
@register_keras_serializable(package="models.unext")
class ShiftMLP(tf.keras.layers.Layer):
    def __init__(self, dim, shift_size=4, dropout=0.0,**kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.fc1 = layers.Dense(dim)
        self.dwconv = DepthwiseConv(dim)
        self.act = layers.Activation('relu')
        self.fc2 = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def shift(self, x, *, H, W, axis):
        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, [-1, self.dim, H, W])
        x = tf.pad(x, [[0, 0], [0, 0], [self.pad, self.pad], [self.pad, self.pad]])
        x = tf.transpose(x, [0, 2, 3, 1])
        splits = tf.split(x, self.shift_size, axis=-1)
        shifts = [tf.roll(s, shift, axis=axis)
                  for s, shift in zip(splits, range(-self.pad, self.pad + 1))]
        x = tf.concat(shifts, axis=-1)
        x = x[:, self.pad:self.pad + H, self.pad:self.pad + W, :]
        x = tf.reshape(x, [-1, H * W, self.dim])
        return tf.transpose(x, [0, 2, 1])

    def call(self, x, *, H, W):
        x = self.shift(x, H=H, W=W, axis=1)
        x = tf.transpose(x, [0, 2, 1])
        x = self.fc1(x)
        x = self.dwconv(x, H=H, W=W)
        x = self.act(x)
        x = self.dropout(x)
        x = tf.transpose(x, [0, 2, 1])
        x = self.shift(x, H=H, W=W, axis=2)
        x = tf.transpose(x, [0, 2, 1])
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "shift_size": self.shift_size,
            "dropout": self.dropout,
        })
        return config

# One shifted block
@register_keras_serializable(package="models.unext")
class ShiftedBlock(tf.keras.layers.Layer):
    def __init__(self, dim, drop_path=0.0,**kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = ShiftMLP(dim)
        self.drop_path = layers.Dropout(drop_path)

    def call(self, x, *, H, W):
        res = x
        x = self.norm(x)
        x = self.mlp(x, H=H, W=W)
        return res + self.drop_path(x)

# Patch embedding layer
@register_keras_serializable(package="models.unext")
class OverlapPatchEmbed(tf.keras.layers.Layer):
    def __init__(self, patch_size=7, stride=4, embed_dim=768, **kwargs):
        super().__init__(**kwargs,)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim,
                                  kernel_size=patch_size,
                                  strides=stride,
                                  padding='same',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                  bias_initializer='zeros')
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H * W, C])
        x = self.norm(x)
        return x, H, W
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "stride": self.stride,
            "embed_dim": self.embed_dim,
        })
        return config
    

@register_keras_serializable(package="models.unext")
class CyclicHorizontalPadding2D(tf.keras.layers.Layer):
    def __init__(self, pad=(1,1), **kwargs):
        super().__init__(**kwargs)
        if isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            self.pad = pad

    def call(self, inputs):
        pad_l, pad_r = self.pad

        # wrap horizontal
        left = inputs[:, :, -pad_l:, :]
        right = inputs[:, :, :pad_r, :]
        x = tf.concat([left, inputs, right], axis=2)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"pad": self.pad})
        return config


def build_unext(input_shape=(192, 384, 12), filter_width = 32, dropout=0.0):
    inputs = layers.Input(shape=input_shape)
    inputs_padded = CyclicHorizontalPadding2D((12,12))(inputs)
    inputs_padded = ZeroPadding2D(((12, 13), (0, 0)))(inputs_padded)

    # Encoder Block 1
    x1 = layers.Conv2D(filter_width, 3, padding='same')(inputs_padded)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    p1 = layers.MaxPooling2D()(x1)

    # Encoder Block 2
    x2 = layers.Conv2D(filter_width*2, 3, padding='same')(p1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    p2 = layers.MaxPooling2D()(x2)

    # Encoder Block 3
    x3 = layers.Conv2D(filter_width*4, 3, padding='same')(p2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    p3 = layers.MaxPooling2D()(x3)

    # MLP Block 1
    patch_embed1 = OverlapPatchEmbed(patch_size=3, stride=2, embed_dim=filter_width*8)
    x4, _, _ = patch_embed1(p3)
    x4 = ShiftedBlock(filter_width*8, drop_path=dropout)(x4, H=12, W=24)
    x4 = layers.LayerNormalization()(x4)
    x4 = layers.Reshape((12, 24, filter_width*8))(x4)

    # MLP Block 2 (Bottleneck)
    patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, embed_dim=filter_width*16)
    x5, _, _ = patch_embed2(x4)
    x5 = ShiftedBlock(filter_width*16, drop_path=dropout)(x5, H=6, W=12)
    x5 = layers.LayerNormalization()(x5)
    x5 = layers.Reshape((6, 12, filter_width*16))(x5)

    # Decoder Block 1
    x = layers.Conv2D(filter_width*8, 3, padding='same')(x5)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, x4])
    x = layers.Reshape((-1, filter_width*8))(x)
    x = ShiftedBlock(filter_width*8, drop_path=dropout)(x, H=12, W=24)
    x = layers.LayerNormalization()(x)
    x = layers.Reshape((12, 24, filter_width*8))(x)

    # Decoder Block 2
    x = layers.Conv2D(filter_width*4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, p3])
    x = layers.Reshape((-1, filter_width*4))(x)
    x = ShiftedBlock(filter_width*4, drop_path=dropout)(x, H=24, W=48)
    x = layers.LayerNormalization()(x)
    x = layers.Reshape((24, 48, filter_width*4))(x)

    # Decoder Block 3
    x = layers.Conv2D(filter_width*2, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, p2])

    # Decoder Block 4
    x = layers.Conv2D(filter_width, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, p1])

    # Decoder Block 5
    x = layers.Conv2D(filter_width, 3, padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.ReLU()(x)

    # Output layer
    output = layers.Conv2D(1, 1, activation='linear')(x)
    output_cropped = layers.Cropping2D(((12, 13), (12, 12)))(output)

    return models.Model(inputs, output_cropped)
