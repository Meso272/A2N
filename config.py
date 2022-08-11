class DefaultConfig(object):
    train_root = '/home/jinyang.liu/lossycompression/cesm-multisnapshot-5fields/CLDHGH/CESM_CLDHGH.h5'
    validation_root = ''
    lr = 5e-4
    batch_size = 32
    num_workers = 8
    epoch = 200

    cuda = True

    opts1 = {
        'title': 'train_loss',
        'xlabel': 'epoch',
        'ylabel': 'loss',
        'width': 300,
        'height': 300,
    }

    opts2 = {
        'title': 'eval_psnr',
        'xlabel': 'epoch',
        'ylabel': 'psnr',
        'width': 300,
        'height': 300,
    }

opt = DefaultConfig()