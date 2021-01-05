import tensorflow as tf

def MobileNetV2(shape):
    model = tf.keras.applications.MobileNetV2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model


def MobileNetV3(shape):
    model = tf.keras.applications.MobileNetV3(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model


def ResNet50V2(shape):
    model = tf.keras.applications.ResNet50V2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model


def InceptionV3(shape):
    model = tf.keras.applications.InceptionV3(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model


model_list = {
    "MobileNetV2": MobileNetV2,
    "MobileNetV3": MobileNetV3,
    "ResNet50": ResNet50V2,
    "InceptionV3": InceptionV3
}


def get_model(name, shape):
    return model_list[name](shape)
