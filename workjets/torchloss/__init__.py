# -*- coding: utf-8 -*-
"""
    @ description：
        枚举出全部损失函数类型，
    @ date:
    @ author: achange
"""

from enum import Enum, unique
from .loss import *


@unique
class LossTypes(Enum):  # 有时间再学习具体细节吧 包括怎么改变打印行为等
    BCELOSS = 1
    BalancedBCELoss = 2
    DiceLoss = 3
    BCEDiceLoss = 4
    GeneralizedDiceLoss = 5
    BinaryFocalLoss = 6
    TverskyLoss = 7

    # add other loss funciton here...

    def __str__(self):
        return self._name_


def get_loss_function(loss_type, config):
    if loss_type == LossTypes.BCELOSS:
        loss_fun = BCELoss()

    elif loss_type == LossTypes.BalancedBCELoss:
        loss_fun = BalancedBCELoss()

    elif loss_type == LossTypes.DiceLoss:
        loss_fun = DiceLoss()

    elif loss_type == LossTypes.BCEDiceLoss:
        loss_fun = BCEDiceLoss(
            alpha=config.BCEDiceLoss.alpha,
            beta=config.BCEDiceLoss.beta
        )

    elif loss_type == LossTypes.GeneralizedDiceLoss:
        loss_fun = GeneralizedDiceLoss()

    elif loss_type == LossTypes.BinaryFocalLoss:
        loss_fun = BinaryFocalLoss(
            alpha=config.BinaryFocalLoss.alpha,
            gamma=config.BinaryFocalLoss.gamma,
            ignore_index=config.BinaryFocalLoss.ignore_index,
            reduction=config.BinaryFocalLoss.reduction
        )

    elif loss_type == LossTypes.TverskyLoss:
        loss_fun = TverskyLoss(
            apply_nonlin=config.TverskyLoss.apply_nonlin,
            batch_dice=config.TverskyLoss.batch_dice,
            do_bg=config.TverskyLoss.do_bg,
            smooth=config.TverskyLoss.smooth,
            square=config.TverskyLoss.square,
            alpha=config.TverskyLoss.alpha,
            beta=config.TverskyLoss.beta
        )

    # register other loss functions here...
    # elif loss_type == LossTypes.TverskyLoss:
    #     loss_fun = TverskyLoss()

    else:
        raise NotImplementedError('Loss function `{}` have not been implemented.'.format(str(loss_type)))

    return loss_fun


if __name__ == '__main__':
    pass
