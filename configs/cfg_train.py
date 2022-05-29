from easydict import EasyDict

cfg_train = EasyDict()

cfg_train.device = 'cuda'

cfg_train.nrof_epochs = 100
cfg_train.lr = 1e-3
cfg_train.momentum = 0.45
cfg_train.weight_decay = 5e-5

cfg_train.weight_sampling = True

cfg_train.use_mixup_training = True

cfg_train.scheduler = True
cfg_train.ROP_factor = 0.5
cfg_train.ROP_patience = 5
cfg_train.watch_on = 'test'

cfg_train.optimizer = 'Adam'
cfg_train.loss = 'CrossEntropyLoss'
cfg_train.label_smoothing = 0.1



cfg_train.eval_on_train = False
cfg_train.eval_on_test = True


cfg_train.verbose = False
cfg_train.tensorboard = True
cfg_train.experiment_name = 'test_4'
cfg_train.log_path = f'../logs/{cfg_train.experiment_name}'

cfg_train.save_folder = f'../saves/{cfg_train.experiment_name}'
cfg_train.load_folder = cfg_train.save_folder

cfg_train.save_best_epoch = True
cfg_train.save_each_epoch = True

cfg_train.load = False
cfg_train.type_checkpoint = 'last_epoch' # 'best_epoch'
