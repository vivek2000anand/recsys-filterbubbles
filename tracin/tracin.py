import torch
from torch.optim import SGD
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
    if optimizer != "SGD":
        raise Exception("Wrong optimizer, can only use SGD")
    num_checkpoints = len(paths)
    influence = 0
    print("Source ", source)
    for model_index in range(num_checkpoints):
        # TODO get the initialization ready
        # TODO add get gradients to the model
        # Load the models and get the informations
        curr_model = model(input_size=128, output_size=5673, hidden_dim=64, n_layers=1) 
        curr_model.LSTM.flatten_parameters()
        optimizer = SGD(curr_model.parameters(), lr=5e-2, momentum=0.9)
        curr_model, model_optimizer, _, _ = load_tracin_checkpoint(curr_model,optimizer, paths[model_index])
        lr = get_lr(model_optimizer)
        # print("LR is ", lr)
        # Get source gradients 
        model_optimizer.zero_grad()
        source_outputs = curr_model.forward(source)
        print("Source outputs are ", source_outputs, source_outputs[0].shape)
        print("Source label is ", source_label)
        source_loss = criterion(source_outputs, source_label)
        # print("source loss is ", source_loss)
        source_loss.backward()
        source_gradients = curr_model.get_gradients()
        # Get target gradients
        model_optimizer.zero_grad()
        target_outputs = curr_model.forward(target)
        print("target outputs are ", target_outputs)
        target_loss = criterion(target_outputs, target_label)
        # print("target loss is ", target_loss)
        target_loss.backward()
        target_gradients = curr_model.get_gradients()
        # Calculate influence for this epoch. Flatten weights and dot product.
        val = torch.dot(source_gradients, target_gradients)
        influence += val * lr
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