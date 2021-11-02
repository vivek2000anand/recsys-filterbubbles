import torch
from copy import deepcopy

def save_tracin_checkpoint(model, epoch, loss, optimizer, path):
    """Saves a checkpoint for tracin to a path

    Args:
        model ([type]): [description]
        epoch ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        path ([type]): [description]
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    return

def load_tracin_checkpoint(model, optimizer, path):
    """Loads a tracin checkpoint from a path

    Args:
        model ([type]): [description]
        optimizer ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def calculate_tracin_influence(model, source, source_label, target, target_label, optimizer, criterion, paths):
    """Calculates influence of source on target datapoint based on TracIn method from checkpoints

    Args:
        model ([type]): [description]
        source ([type]): [description]
        target ([type]): [description]
        optimizer ([type]): [description]
        criterion ([type]): [description]
        paths ([type]): [description]
    """
    num_checkpoints = len(paths)
    influence = 0
    for model_index in range(num_checkpoints):
        # TODO get the initialization ready
        # TODO add get gradients to the model
        # Load the models and get the informations
        curr_model = model()
        curr_model, model_optimizer, model_epoch, _ = load_tracin_checkpoint(curr_model, deepcopy(optimizer), paths[model_index])
        lr = get_lr(model_optimizer)
        # Get source gradients 
        model_optimizer.zero_grad()
        source_outputs = curr_model(source)
        source_loss = criterion(source_outputs, source_label)
        source_loss.backward()
        source_gradients = curr_model.get_gradients()
        # Get target gradients
        model_optimizer.zero_grad()
        target_outputs = curr_model(target)
        target_loss = criterion(target_outputs, target_label)
        target_loss.backward()
        target_gradients = curr_model.get_gradients()
        # Calculate influence for this epoch. Flatten weights and dot product.
        val = torch.dot(torch.flatten(source_gradients, target_gradients))
        influence += val
    return influence

def get_lr(optimizer):
    """Gets learning rate given an optimizer

    Args:
        optimizer ([type]): [description]
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    print("This doesn't do jack")