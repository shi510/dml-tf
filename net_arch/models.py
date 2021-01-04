import net_arch.mobilenet_v2
import net_arch.mobilenet_v3
import net_arch.efficient_net

import tensorflow as tf


def conv_bn_relu(x, filter):
    y = x
    y = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    return y


def VGGVariant(shape):
    y = x = tf.keras.Input(shape)
    y = conv_bn_relu(y, 32)
    y = conv_bn_relu(y, 32)
    y = conv_bn_relu(y, 32)
    y = tf.keras.layers.MaxPooling2D((2, 2))(y)
    y = conv_bn_relu(y, 64)
    y = conv_bn_relu(y, 64)
    y = conv_bn_relu(y, 64)
    y = tf.keras.layers.MaxPooling2D((2, 2))(y)
    y = conv_bn_relu(y, 64)
    y = conv_bn_relu(y, 64)
    y = conv_bn_relu(y, 64)
    y = tf.keras.layers.MaxPooling2D((2, 2))(y)

    return tf.keras.Model(x, y)


def LENET(shape):
    y = x = tf.keras.Input(shape)
    y = conv_bn_relu(y, 20)
    y = tf.keras.layers.MaxPooling2D((2, 2))(y)
    y = conv_bn_relu(y, 40)
    y = tf.keras.layers.MaxPooling2D((2, 2))(y)

    return tf.keras.Model(x, y)


def ResNet50V2(shape):
    model = tf.keras.applications.ResNet50V2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights=None)
    return model


model_list = {
    "MobileNetV2": net_arch.mobilenet_v2.MobileNetV2,
    "MobileNetV3": net_arch.mobilenet_v3.MakeMobileNetV3,
    "EfficientNetB3": net_arch.efficient_net.EfficientNetB3,
    "ResNet50": ResNet50V2,
    "VGGVariant": VGGVariant,
    "LENET": LENET
}


def get_model(name, shape):
    return model_list[name](shape)
