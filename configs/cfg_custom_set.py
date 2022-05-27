from easydict import EasyDict
import os

cfg_custom_set = EasyDict()
cfg_custom_set.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_custom_set.path ='/home/kirito/custom_set'# os.path.join(cfg_custom_set.root_dir, "data/custom_set")#'/home/kirito/custom_set'##