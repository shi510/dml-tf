config = {
    #
    # 1. mnist: shape=(28, 28, 1)
    # 2. cifar10: shape=(32, 32, 3)
    # 3. cifar100: shape=(32, 32, 3)
    # 4. cars196: shape=(160, 160, 3)
    # 5. cub: shape=(160, 160, 3)
    #
    'dataset': 'cifar100',
    'model_name': 'mnist_test',
    'batch_size' : 64,
    'shape' : [32, 32, 3],

    #
    # If 'saved_model' not exsits, then it will be built with this architecture.
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. MobileNetV3
    # 3. EfficientNetB3
    # 4. ResNet50
    # 5. LENET (only for mnist)
    # 6. VGGVariant (only for cifar10)
    #
    'model' : 'VGGVariant',
    'embedding_dim': 64,

    #
    # 1. ProxyNCA
    #
    'loss': 'ProxyNCA',

    #
    # There are two options.
    #  1. adam
    #  2. sgd with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'adam',
    'epoch' : 30,

    #
    # initial learning rate.
    #
    'lr' : 1e-4,

    #
    # lr * decay_rate ^ (steps / decay_steps)
    #
    'lr_decay': False,
    'lr_decay_steps' : 10000,
    'lr_decay_rate' : 0.9
}
