from easydict import EasyDict
import os

cfg_dataloader = EasyDict()

cfg_dataloader.batch_size = 32
cfg_dataloader.shuffle = True
cfg_dataloader.drop_last = True

cfg_dataloader.num_workers = 12

