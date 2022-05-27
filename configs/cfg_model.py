from easydict import EasyDict
import os

cfg_model = EasyDict()
cfg_model.name_model = 'ResNet50'
cfg_model.use_stochastic_depth = False
cfg_model.prob_stochastic_depth = 0.0
cfg_model.zero_bn_init = False
cfg_model.nb_classes = 12


cfg_model.input_channels = 3

version_shortcut = 1
strides = [2, 1, 1]


cfg_model.version_first_conv_block = 1
# first version
cfg_model.first_conv_block_kernel_size = 7
cfg_model.first_conv_block_stride = 2
cfg_model.first_conv_block_padding = 3
# second version
cfg_model.in_channels = [32, 32]
cfg_model.out_channels = [32, 32, 64]
cfg_model.kernel_sizes = [3, 3, 3]
cfg_model.strides = [2, 1, 1]
cfg_model.paddings = [1, 1, 1]


cfg_model.first_max_pool_kernel = 3
cfg_model.first_max_pool_stride = 2
cfg_model.first_max_pool_padding = 1

cfg_model.nb_channels = [64, 128, 256, 512]
cfg_model.strides_conv_block = [1, 2, 2, 2]
cfg_model.resblock_kernel_sizes = [1, 3, 1]
cfg_model.resblock_paddings = [0, 1, 0]
cfg_model.nb_res_blocks = [3, 4, 6, 3]


cfg_model.orders = [
    {
        '1_Weight':dict(stride=strides[0]),
        '1_Norm':{},
        '1_Activation':{},

        '2_Weight':dict(stride=strides[1]),
        '2_Norm':{},
        '2_Activation':{},

        '3_Weight':dict(stride=strides[2]),
        '3_Norm':{},
        '1_Shortcut':{'version':version_shortcut},
        '3_Activation':{}
    },
    {
        '1_Weight':dict(stride=strides[0]),
        '1_Norm':{},
        '1_Activation':{},

        '2_Weight':dict(stride=strides[1]),
        '2_Norm':{},
        '2_Activation':{},

        '3_Weight':dict(stride=strides[2]),
        '1_Shortcut':{'version':version_shortcut},
        '3_Norm':{},
        '3_Activation':{}
    },
    {
        '1_Weight':dict(stride=strides[0]),
        '1_Norm':{},
        '1_Activation':{},

        '2_Weight':dict(stride=strides[1]),
        '2_Norm':{},
        '2_Activation':{},

        '3_Weight':dict(stride=strides[2]),
        '3_Norm':{},
        '3_Activation':{},
        '1_Shortcut':{'version':version_shortcut}
    },
    {
        '1_Activation':{},
        '1_Weight':dict(stride=strides[0]),
        '1_Norm':{},

        '2_Activation':{},
        '2_Weight':dict(stride=strides[1]),
        '2_Norm':{},

        '3_Activation':{},
        '3_Weight':dict(stride=strides[2]),
        '3_Norm':{},
        '1_Shortcut':{'version':version_shortcut}
    },
    {
        '1_Norm':{},
        '1_Activation':{},
        '1_Weight':dict(stride=strides[0]),
        '2_Norm': {},
        '2_Activation': {},
        '2_Weight': dict(stride=strides[1]),
        '3_Norm': {},
        '3_Activation': {},
        '3_Weight': dict(stride=strides[2]),
        '1_Shortcut':{'version':version_shortcut},
    }
]
cfg_model.order = cfg_model.orders[0]
