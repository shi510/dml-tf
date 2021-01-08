config = {
    #
    # 1. cars196: shape=(224, 224, 3)
    # 2. cub: shape=(224, 224, 3)
    # 3. sop: shape=(224, 224, 3)
    #
    'dataset': 'cars196',
    'model_name': 'mnist_test',
    'batch_size' : 64,
    'shape' : [224, 224, 3],

    #
    # Choose your architecture:
    # 1. InceptionV3
    # 2. ResNet50
    # 3. MobileNetV2
    # 4. MobileNetV3
    #
    'model' : 'InceptionV3',
    'embedding_dim': 64,

    #
    # 1. ProxyNCA
    #
    'loss': 'ProxyNCA',

    'loss_param':{
        'ProxyNCA':{
            'embedding_scale': 1,
            'proxy_scale': 8
        }
    },

    'eval':{
        'linear': False,
        'recall':[1, 2, 4, 8],
        # Calculating NMI is too slow.
        # (SOP dataset takes a long time)
        # If False, it is only executed at the end of the training.
        'NMI': False
    },

    #
    # There are two options.
    #  1. adam
    #  2. sgd with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'adam',
    'epoch' : 15,

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
