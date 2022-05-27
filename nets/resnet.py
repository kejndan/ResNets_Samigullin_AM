import torch
import torch.nn as nn
from nets.modules import Shortcut
from nets.nets_utils import DropPath

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, 'last_bn', None) is not None and m.last_bn:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
        else:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)




class ResBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, order, downsample, stride, config_model,idx_res_block=0):
        super().__init__()
        self.operations = []
        self.position_shortcut = None
        self.idx_res_block = idx_res_block
        last_out_chan = in_channels
        for i, (operation, args) in enumerate(order.items()):
            num_oper = int(operation[0])
            name_oper = operation[2:]
            if name_oper == 'Weight':
                cur_stride = args['stride'] if 'stride' in args and stride != 1 and downsample else 1
                if num_oper == 1:
                    in_chan = in_channels
                    out_chan = out_channels
                    k_size = config_model.resblock_kernel_sizes[0]
                    padding = config_model.resblock_paddings[0]
                elif num_oper == 2:
                    in_chan = out_channels
                    out_chan = out_channels
                    k_size = config_model.resblock_kernel_sizes[1]
                    padding = config_model.resblock_paddings[1]
                elif num_oper == 3:
                    in_chan = out_channels
                    out_chan = out_channels*self.expansion
                    k_size = config_model.resblock_kernel_sizes[2]
                    padding = config_model.resblock_paddings[2]
                last_out_chan = out_chan
                self.operations.append(nn.Conv2d(in_channels=in_chan,
                                                 out_channels=out_chan,
                                                 kernel_size=k_size,
                                                 stride=cur_stride,
                                                 padding=padding))
            elif name_oper == 'Norm':
                last_bn = True if num_oper == 3 and config_model.zero_bn_init else False

                bn = nn.BatchNorm2d(num_features=last_out_chan)
                bn.last_bn = last_bn
                self.operations.append(bn)

            elif name_oper == 'Activation':
                self.operations.append(nn.ReLU())

            elif name_oper == 'Shortcut':
                self.position_shortcut = i
                stride = args['stride'] if 'stride' in args else 2
                version = args['version'] if 'version' in args else 1

                prod_drop_path = self.idx_res_block/sum(config_model.nb_res_blocks)*(1-config_model.prob_stochastic_depth) + 1e-4\
                    if config_model.use_stochastic_depth else 0

                drop_path = DropPath(drop_prob=prod_drop_path)
                self.operations.append(Shortcut(downsample, in_channels=in_channels,
                                                out_channels=out_channels*self.expansion,
                                                stride=stride,
                                                version=version,
                                                drop_path=drop_path))
            else:
                raise ValueError('Incorrect operation name')

        self.operations = nn.Sequential(*self.operations)

    def forward(self, x):
        self.operations[self.position_shortcut].identity = x.clone()
        return self.operations(x)



class ResNet(nn.Module):
    def __init__(self, config_model, nb_classes):
        super().__init__()
        self.config_model = config_model
        list_nb_blocks = config_model.nb_res_blocks
        self.in_channels = config_model.nb_channels[0]
        self.order = config_model.order
        self.conv1 = self.__create_first_conv_block(config_model.input_channels, config_model.version_first_conv_block)
        self.pool1 = nn.MaxPool2d(kernel_size=config_model.first_max_pool_kernel,
                                  stride=config_model.first_max_pool_stride,
                                  padding=config_model.first_max_pool_padding)
        self.relu = nn.ReLU()
        self.count_res_blocks = 0

        self.conv2 = self.__create_conv_block(config_model.nb_channels[0], list_nb_blocks[0],
                                              stride=config_model.strides_conv_block[0])

        self.conv3 = self.__create_conv_block(config_model.nb_channels[1], list_nb_blocks[1],
                                              stride=config_model.strides_conv_block[1])

        self.conv4 = self.__create_conv_block(config_model.nb_channels[2], list_nb_blocks[2],
                                              stride=config_model.strides_conv_block[2])

        self.conv5 = self.__create_conv_block(config_model.nb_channels[3], list_nb_blocks[3],
                                              stride=config_model.strides_conv_block[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config_model.nb_channels[3]*ResBlock.expansion, nb_classes)

        self.apply(init_weights)

    def __create_first_conv_block(self, num_channels, version_first_conv_block=1):
        operations = []
        if version_first_conv_block == 1:
            operations.append(
                nn.Conv2d(in_channels=num_channels,
                          out_channels=self.config_model.nb_channels[0],
                          kernel_size=self.config_model.first_conv_block_kernel_size,
                          stride=self.config_model.first_conv_block_stride,
                          padding=self.config_model.first_conv_block_padding,
                          bias=False)
            )
            operations.append(nn.BatchNorm2d(self.config_model.nb_channels[0]))
        elif version_first_conv_block == 2:
            operations.append(
                nn.Conv2d(in_channels=num_channels,
                          out_channels=self.config_model.out_channels[0],
                          kernel_size=self.config_model.kernel_sizes[0],
                          stride=self.config_model.strides[0],
                          padding=self.config_model.paddings[0],
                          bias=False))
            operations.append(nn.BatchNorm2d(self.config_model.out_channels[0]))
            operations.append(
                nn.Conv2d(in_channels=self.config_model.in_channels[0],
                          out_channels=self.config_model.out_channels[1],
                          kernel_size=self.config_model.kernel_sizes[1],
                          stride=self.config_model.strides[1],
                          padding=self.config_model.paddings[1],
                          bias=False))
            operations.append(nn.BatchNorm2d(self.config_model.out_channels[1]))
            operations.append(
                nn.Conv2d(in_channels=self.config_model.in_channels[1],
                          out_channels=self.config_model.out_channels[2],
                          kernel_size=self.config_model.kernel_sizes[2],
                          stride=self.config_model.strides[2],
                          padding=self.config_model.paddings[2],
                          bias=False))
            operations.append(nn.BatchNorm2d(self.config_model.out_channels[2]))

        else:
            raise ValueError(f'Version {version_first_conv_block} first conv block  does not exist!')

        return nn.Sequential(*operations)

    def __create_conv_block(self, planes, nb_res_blocks, stride):
        res_blocks = []
        self.order['1_Shortcut']['stride'] = stride


        for i in range(nb_res_blocks):
            if i == 0:
                downsample = True
            else:
                downsample = False
            self.count_res_blocks += 1
            r_block = ResBlock(in_channels=self.in_channels,
                                       out_channels=planes,
                                       order=self.order,
                                       downsample=downsample,
                                       stride=stride,
                                       config_model=self.config_model,
                                        idx_res_block=self.count_res_blocks)

            res_blocks.append(r_block)

            if i == 0:
                self.in_channels = planes * ResBlock.expansion

        return nn.Sequential(*res_blocks)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)


        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    order = {'1_Weight':dict(stride=1),
             '1_Norm':{},
             '1_Activation':{},
             '2_Weight':dict(stride=2),
             '2_Norm':{},
             '2_Activation':{},
             '3_Weight':dict(stride=1),
             '3_Norm':{},
             '1_Shortcut':{'version':1},
             '3_Activation':{}
             }
    from configs.cfg_model import cfg_model
    x = torch.rand(1,3,224,224)
    rn = ResNet(cfg_model, 10)

    from torchsummary import summary
    rn(x)
    # summary(rn, x.size(),batch_size=128, device='cpu')







