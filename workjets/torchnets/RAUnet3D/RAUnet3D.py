# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
        implemention of Residual Block  残差机制
        https://arxiv.org/pdf/1603.05027.pdf
    """
    def __init__(self, in_channels=None, out_channels=None):
        super(ResidualBlock, self).__init__()
        self.Relu = nn.ReLU()

        mid_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

        self.triple_conv = nn.Sequential(
            # nn.BatchNorm3d(in_channels),
            # nn.ReLU(),
            nn.Conv3d(in_channels, mid_channels, kernel_size=1),
            # todo: 不是pooling+BN+Activation+Conv  而是pooling+Conv+BN+Activation
            #   参数的差别就在于先BN还是后BN
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),

            # nn.BatchNorm3d(mid_channels),
            # nn.ReLU(),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),

            # nn.BatchNorm3d(mid_channels),
            # nn.ReLU(),  # todo: 这里不用relu  最后该block输出的时候在activation一次！也可以都有吧
            nn.Conv3d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),  # todo: 这里不用relu  最后该block输出的时候在activation一次！也可以都有吧
        )

    def forward(self, x):
        # residual block
        out = self.triple_conv(x)

        # apply first convolution and save the output as a residual
        residual = self.conv1(x)

        out += residual
        # out = self.Relu(out)
        return out



class AttentionResidualBlock(nn.Module):
    """
        implemention of `SELF` attention mechanism  自注意力机制！！
        注：输入和输出大小相同！！
    """

    def __init__(self, n_channels, depth=1):
        super(AttentionResidualBlock, self).__init__()
        self.depth = depth

        # First Residual Block
        self.start = ResidualBlock(in_channels=n_channels, out_channels=n_channels)

        # Step1. Trunk Branch:注意力机制模块的主干
        self.trunk = nn.Sequential(
            ResidualBlock(in_channels=n_channels, out_channels=n_channels),
            ResidualBlock(in_channels=n_channels, out_channels=n_channels),
        )

        # Step2. Soft Mask Branch: 软掩码分支
        # 2.1 encoder
        self.encoder_downsampling = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=n_channels, out_channels=n_channels),
        )
        self.encoder_skip_connection = ResidualBlock(in_channels=n_channels, out_channels=n_channels)

        # 2.2 decoder
        self.decoder_skip_connection = 'Add'    # 使用张量相加即可
        self.decoder_upsampling = nn.Sequential(
            ResidualBlock(in_channels=n_channels, out_channels=n_channels),
            # nn.ConvTranspose3d(n_channels, n_channels, kernel_size=3)
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            # todo: 这里注意有多种上采样方式，对照参数看选用哪种 --> 使用nn.Upsample无参数
            #   本质的区别是：使用卷积 还是 使用差值 进行上采样  -->  使用差值
            #  注意 F.interpolate()是差值计算 需要放在forward中

            #  todo: 【知识点】pytorch 上采样 upsample 时align_corners 设为true 还是false:（直接百度align_corners）
            #   https://blog.csdn.net/wangweiwells/article/details/101820932
        )

        self.soft_mask_end = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.end = ResidualBlock(in_channels=n_channels, out_channels=n_channels)


    def forward(self, inputs):
        inputs = self.start(inputs)

        output_trunk = self.trunk(inputs)

        output_soft_mask = self.encoder_downsampling(inputs)
        skip_connections = []
        for i in range(self.depth-1):
            skip_connections.append(self.encoder_skip_connection(output_soft_mask))
            output_soft_mask = self.encoder_downsampling(output_soft_mask)
        # ----------  对称  -----------
        for i in range(self.depth - 1):
            output_soft_mask = self.decoder_upsampling(output_soft_mask)
            output_soft_mask += skip_connections[self.depth - 2 - i]    # todo:此处注意下标反过来 逆序遍历
        output_soft_mask = self.decoder_upsampling(output_soft_mask)

        output_soft_mask = self.soft_mask_end(output_soft_mask)
        output = output_soft_mask + 1
        output = output * output_trunk

        output = self.end(output)
        return output



class RAUnet3D(nn.Module):
    """
        implemention of `SELF` attention residual  Unet3D
    """

    def __init__(self, in_channels, out_channels, feature_scale=1, norm_type='BN'):
        super(RAUnet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_scale = feature_scale

        # 1.确定4次卷积、反卷积时卷积核数目
        filters = [32, 64, 128, 256, 512]     # 卷积核数目
        filters = [int(_ * self.feature_scale) for _ in filters]
        self.norm_type = norm_type  # in [BN='BatchNorm', GN='GroupNorm', IN='InstanceNorm']
        # # 使用BN GN IN? TODO: 目前就使用batch normalization，如果要微调再实现其他Norm方式
        # self.Norm3d = None
        # if norm_type == 'BN':
        #    self.Norm3d = nn.BatchNorm3d
        # elif norm_type == 'GN':
        #     self.Norm3d = nn.GroupNorm3d

        # todo: 下采样/上采样，使用maxpooling、upsampling即可
        self.downsampling = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.start_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=filters[0], kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU()
        )

        self.res_block1 = ResidualBlock(in_channels=filters[0], out_channels=filters[1])
        self.res_block2 = ResidualBlock(in_channels=filters[1], out_channels=filters[2])
        self.res_block3 = ResidualBlock(in_channels=filters[2], out_channels=filters[3])
        self.res_block4 = ResidualBlock(in_channels=filters[3], out_channels=filters[4])
        self.botoom_conv = nn.Sequential(
            ResidualBlock(in_channels=filters[4], out_channels=filters[4]),
            ResidualBlock(in_channels=filters[4], out_channels=filters[4])
        )


        self.att_block1 = AttentionResidualBlock(n_channels=filters[4])
        self.att_block2 = AttentionResidualBlock(n_channels=filters[3])
        self.att_block3 = AttentionResidualBlock(n_channels=filters[2])
        self.att_block4 = AttentionResidualBlock(n_channels=filters[1])

        self.conv_after_cat1 = ResidualBlock(in_channels=2*filters[4], out_channels=filters[4])
        self.conv_after_cat2 = ResidualBlock(in_channels=filters[4] + filters[3], out_channels=filters[3])
        self.conv_after_cat3 = ResidualBlock(in_channels=filters[3] + filters[2], out_channels=filters[2])
        self.conv_after_cat4 = ResidualBlock(in_channels=filters[2] + filters[1], out_channels=filters[1])

        self.end_conv = nn.Sequential(
            nn.Conv3d(in_channels=filters[1] + filters[0], out_channels=filters[0], kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),

            # todo: output conv 1x1x1 --> to out channels
            nn.Conv3d(in_channels=filters[0], out_channels=self.out_channels, kernel_size=1),
        )


    def forward(self, inputs):
        start_conv = self.start_conv(inputs)      # todo: start_conv可以使用 `res block` 吗
        pool = self.downsampling(start_conv)

        res1 = self.res_block1(pool)
        pool1 = self.downsampling(res1)

        res2 = self.res_block2(pool1)
        pool2 = self.downsampling(res2)

        res3 = self.res_block3(pool2)
        pool3 = self.downsampling(res3)

        res4 = self.res_block4(pool3)
        pool4 = self.downsampling(res4)

        res5 = self.botoom_conv(pool4)

        atb1 = self.att_block1(res4)                # torch.Size([1, 512, 4, 4, 4])
        up1 = self.upsampling(res5)                 # torch.Size([1, 512, 4, 4, 4])
        merged1 = torch.cat((atb1, up1), dim=1)     # torch.Size([1, 1024, 4, 4, 4])
        res5 = self.conv_after_cat1(merged1)        # torch.Size([1, 512, 4, 4, 4])

        atb2 = self.att_block2(res3)                # torch.Size([1, 256, 8, 8, 8])
        up2 = self.upsampling(res5)                 # torch.Size([1, 512, 8, 8, 8])
        merged2 = torch.cat((atb2, up2), dim=1)     # torch.Size([1, 768, 8, 8, 8])
        res6 = self.conv_after_cat2(merged2)        # torch.Size([1, 256, 8, 8, 8])


        atb3 = self.att_block3(res2)                # torch.Size([1, 128, 16, 16, 16])
        up3 = self.upsampling(res6)                 # torch.Size([1, 256, 16, 16, 16])
        merged3 = torch.cat((atb3, up3), dim=1)     # torch.Size([1, 384, 16, 16, 16])
        res7 = self.conv_after_cat3(merged3)        # torch.Size([1, 128, 16, 16, 16])


        atb4 = self.att_block4(res1)                # torch.Size([1, 64, 32, 32, 32])
        up4 = self.upsampling(res7)                 # torch.Size([1, 128, 32, 32, 32])
        merged4 = torch.cat((atb4, up4), dim=1)     # torch.Size([1, 192, 32, 32, 32])
        res8 = self.conv_after_cat4(merged4)        # torch.Size([1, 64, 32, 32, 32])


        # todo: No attention residual block
        up = self.upsampling(res8)                      # torch.Size([1, 64, 64, 64, 64])
        merged = torch.cat((up, start_conv), dim=1)     # torch.Size([1, 96, 64, 64, 64])
        logits = self.end_conv(merged)
        # todo: 最后一层根据final_sigmoid参数确定使用 nn.sigmoid或者nn.softmax，最终转换为分割结果视情况而定
        #   这样方便使用不同的loss functions  ----> 统一返回logits
        return logits



if __name__ == '__main__':

    """
    来自nn.Upsample的注释：
    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.
        from torch.nn import functional as F
        F.interpolate(x, size=output_size, mode='nearest')
        nn.functional.interpolate
    """

    from torchsummary import summary

    net = RAUnet3D(in_channels=4, out_channels=3)
    net.to(device=torch.device('cpu'))
    # from torch import optim
    # optimizer = optim.Adam(list(net.parameters()), 0.002, (0.8, 0.99))

    # data = torch.rand(1, 1, 160, 128, 128)  # (batchSize, channels, voxelBlockSize)
    # output = net(data)
    # print(output.size())
    # TODO: 目前内存不足，无法测试该size图像块的大小
    # 打印模型：
    summary(net, batch_size=1, input_size=(4, 64, 64, 64), device='cpu')

    # TODO: 使用一个

