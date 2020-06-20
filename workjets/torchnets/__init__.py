# -*- coding: utf-8 -*-
"""
    @ description：

    @ date:
    @ author: achange
"""
import torch
import torch.nn as nn

from enum import Enum, unique


@unique
class NetTypes(Enum):
    UNet3D = 1
    ResidualUNet3D = 2
    RAUnet3D = 3

    # add other loss networks here...

    def __str__(self):
        return self._name_




def get_networks(net_type, config):
    if net_type == NetTypes.UNet3D:
        from .Unet3D.unet3d import UNet3D
        model = UNet3D(
            in_channels=config.UNet3D.in_channels,
            out_channels=config.UNet3D.out_channels,
            f_maps=config.UNet3D.f_maps,
            layer_order=config.UNet3D.layer_order,
            num_groups=config.UNet3D.num_groups
        )

    elif net_type == NetTypes.ResidualUNet3D:
        from .Unet3D.unet3d import ResidualUNet3D
        model = ResidualUNet3D(
            in_channels=config.ResidualUNet3D.in_channels,
            out_channels=config.ResidualUNet3D.out_channels,
            f_maps=config.ResidualUNet3D.f_maps,
            conv_layer_order=config.ResidualUNet3D.layer_order,
            num_groups=config.ResidualUNet3D.num_groups
        )

    elif net_type == NetTypes.RAUnet3D:
        from .RAUnet3D.RAUnet3D import RAUnet3D
        model = RAUnet3D(
            in_channels=config.RAUnet3D.in_channels,
            out_channels=config.RAUnet3D.out_channels,
            feature_scale=config.RAUnet3D.feature_scale,
            norm_type=config.RAUnet3D.norm_type
        )

    # add other networks here...
    # elif net_type == xxx:
    #     from .RAUnet3D... import ...
    #     model = ...

    else:
        raise NotImplementedError('Model `{}` have not been implemented.'.format(str(net_type)))

    # todo: 将model 放在指定设备GPU/CPU上计算
    if config.device == 'cpu':
        print('[info] Computing on CPU.')
        device = torch.device('cpu')
        model.to(device)
    elif config.device == 'cuda':
        gpu_nums = torch.cuda.device_count()
        if torch.cuda.is_available():
            if gpu_nums > 1:
                # todo: DataParallel默认将模型输出放在cuda:0,使用DataParallel后模型类方法的访问使用`modle.module.METHOD_NAME`
                print('[info] Computing on Multi-GPUs.')
                device = torch.device('cuda')
                model = nn.DataParallel(model)
                model.to(device)
            else:
                print('[info] Computing on Single GPU.')
                device = torch.device('cuda:0')
                model.to(device)
        else:
            print('[info] No GPU device can be used, computing on CPU.')
            device = torch.device('cpu')
            model.to(device)

    return model


if __name__ == '__main__':
    pass

