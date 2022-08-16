class DefaultConfig(object):
    train_root = '/home/jinyang.liu/lossycompression/cesm-multisnapshot-5fields/CLDHGH/CESM_CLDHGH.h5'
    validation_root = ''
    lr = 5e-4
    patch_size = 64
    batch_size = 32
    num_workers = 32
    epoch = 200
    fix_length= 1600
    scale = 2 
    aug = 1
    loss_fn = "CL1"
    gamma = None

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