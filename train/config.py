config = {
    #
    # 1. cars196: shape=(224, 224, 3)
    # 2. cub: shape=(224, 224, 3)
    # 3. sop: shape=(224, 224, 3)
    # 4. inshop: shape=(224, 224, 3), need to download manually
    #
    'dataset': 'cub',
    'model_name': 'proxynca_cub',
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
    # 2. ProxyAnchor
    #
    'loss': 'ProxyAnchor',

    'loss_param':{
        'ProxyNCA':{
            'scale': 32,
            'proxy_lr': 1e-2,
        },
        'ProxyAnchor':{
            'scale': 32,
            'delta': 0.1,
            'proxy_lr': 1e-2,
        }
    },

    'eval':{
        'metric': 'cos',
        'recall':[1, 2, 4, 8],
        # Calculating NMI is too slow.
        # (SOP dataset takes a long time)
        # If False, it is only executed at the end of the training.
        'NMI': False
    },

    'epoch' : 50,

    #
    # There are two options.
    #  1. Adam
    #  2. AdamW
    #  3. RMSprop
    #  4. SGD with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'AdamW',

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
