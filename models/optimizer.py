from utils.helpers import Nadam
from torch.optim import Adam, RMSprop, SGD


def get_optimizer(optim_params, optim_name, optim_lr, optim_momentum):

    optim_name = optim_name.lower()

    if optim_name == 'adam':
        return Adam(optim_params, lr=optim_lr)
    elif optim_name == 'sgd':
        return SGD(optim_params, lr=optim_lr, momentum=optim_momentum)
    elif optim_name == 'rmsprop':
        return RMSprop(optim_params, lr=optim_lr, momentum=optim_momentum)
    elif optim_name == 'nadam':
        return Nadam(optim_params, lr=optim_lr)
    else:
        raise ValueError(f"❌ Unknown optimizer: {optim_name}")