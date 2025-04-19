from utils.helpers import Nadam
from torch.optim import Adam, RMSprop, SGD, AdamW, lr_scheduler


def get_optimizer(optim_params, optim_name, optim_lr, optim_momentum, change_after, lr_step, weight_decay):
    optim_name = optim_name.lower()

    if optim_name == 'adam':
        optimizer = Adam(optim_params, lr=optim_lr, weight_decay=weight_decay)

    elif optim_name == 'sgd':
        optimizer = SGD(optim_params, lr=optim_lr, momentum=optim_momentum, weight_decay=weight_decay)

    elif optim_name == 'rmsprop':
        optimizer = RMSprop(optim_params, lr=optim_lr, momentum=optim_momentum, weight_decay=weight_decay)

    elif optim_name == 'nadam':
        optimizer = Nadam(optim_params, lr=optim_lr)

    elif optim_name == 'adamw':
        optimizer = AdamW(optim_params, lr=optim_lr)

    else:
        raise ValueError(f"❌ Unknown optimizer: {optim_name}")

    # Attach a scheduler to decay LR after `change_after` epochs by a factor of 1/lr_step
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=change_after,
        gamma=1 / float(lr_step)
    )

    print("🔧 Initializing Optimizer:")
    print(f"   • Type         : {optim_name.upper()}")
    print(f"   • Learning Rate: {optim_lr}")
    print(f"   • Momentum     : {optim_momentum if optim_name in ['sgd', 'rmsprop'] else 'N/A'}")
    print(f"   • LR Decay     : Step every {change_after} epochs by factor 1/{lr_step} ≈ {round(1/float(lr_step), 4)}")

    return optimizer, scheduler


"""
optim.SGD([
                {'params': model.base.named_parameters(), 'lr': 1e-2},
                {'params': model.classifier.named_parameters()}
            ], lr=1e-3, momentum=0.9)
"""