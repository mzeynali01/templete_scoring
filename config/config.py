class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = True
    finetune = False

    train_root = '/home/mm/projects/arcface-pytorch/data/Dataset/webface/CASIA-maxpy-clean-crop-144/'
    train_list = '/home/mm/projects/arcface-pytorch/data/Dataset/webface/train_data_13938.txt'
    val_list = '/home/mm/projects/arcface-pytorch/data/Dataset/webface/val_data_13938.txt'

    test_root = '/home/mm/projects/arcface-pytorch/data/Dataset/lfw'
    test_list = '/home/mm/projects/arcface-pytorch/data/Dataset/lfw_test_pair.txt'

    images_root = '/home/mm/projects/arcface-pytorch/data/Dataset/lfw'
    images_test_list = '/home/mm/projects/arcface-pytorch/data/Dataset/lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 2  # batch size
    test_batch_size = 2

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = False  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 2  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
