import torch
from torch._C import device
from torch.nn.modules.module import _forward_unimplemented
from torch.optim import SGD
from copy import deepcopy
from torch import nn
from torch.serialization import SourceChangeWarning
from tqdm import tqdm
import time

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


def get_lr(optimizer):
    """Gets learning rate given an optimizer
    Args:
        optimizer ([type]): [description]
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def approximate_tracin_batched(model, sources, source_labels, targets, target_labels, paths, device, num_items=5673, batch_size=2048, optimizer="SGD"):
    total_length = len(sources)
    num_checkpoints = len(paths)
    # Initialization
    sources = [torch.LongTensor(s)for s in sources]
    targets = [torch.LongTensor(s) for s in targets]
    source_labels = torch.LongTensor(source_labels).to(device)
    target_labels = torch.LongTensor(target_labels).to(device)
    # print(f"Source labels shape is {source_labels.shape}")
    # print(f"Target labels shape is {target_labels.shape}")
    start_time = time.time()
    influence = 0
    for model_index in range(num_checkpoints):
        checkpoint_start_time = time.time()
        print(f"In checkpoint number: {model_index}")
        # Initialize model
        curr_model = model(input_size=128, output_size=num_items, hidden_dim=64, n_layers=1, device=device)
        curr_model.LSTM.flatten_parameters()
        optimizer = SGD(curr_model.parameters(), lr=5e-2, momentum=0.9)
        curr_model, model_optimizer, _, _ = load_tracin_checkpoint(curr_model,optimizer, paths[model_index])
        lr = get_lr(model_optimizer)
        # Get Embeddings
        sources_emb = [curr_model.item_emb(torch.LongTensor(i)).to(device) for i in sources]
        targets_emb = [curr_model.item_emb(torch.LongTensor(i)).to(device) for i in targets]
        curr_model.to(device)
        criterion = nn.CrossEntropyLoss()
        # Batch the inputs
        for iteration in range(int(total_length/batch_size)+1):
            st_idx,ed_idx = iteration*batch_size, (iteration+1)*batch_size
            if ed_idx>total_length:
                ed_idx = total_length
            curr_length = ed_idx -st_idx
            # Sources 
            optimizer.zero_grad()
            output, hidden = curr_model.forward(torch.stack([sources_emb[i] for i in range(st_idx,ed_idx)],dim=0).detach())
            # print("output shape ", output.shape)
            # print(f"st_idx {st_idx} ed_idx {ed_idx}")
            # print(f"source labels {source_labels[st_idx:ed_idx].shape}")
            loss = criterion(output, source_labels[st_idx:ed_idx])
            loss.backward()
            source_gradients = curr_model.get_gradients(device)
            # Targets
            optimizer.zero_grad()
            output, hidden = curr_model.forward(torch.stack([targets_emb[i] for i in range(st_idx,ed_idx)],dim=0).detach())
            # print("output shape ", output.shape)
            # print(f"st_idx {st_idx} ed_idx {ed_idx}")
            # print(f"target labels {target_labels[st_idx:ed_idx].shape}")
            loss = criterion(output, target_labels[st_idx:ed_idx])
            loss.backward()
            target_gradients = curr_model.get_gradients(device)

            # Get total Influence
            val = torch.dot(source_gradients, target_gradients)
            influence += val * lr * (curr_length / total_length)
        checkpoint_end_time = time.time()
        print(f"Total time for checkpoint {model_index} : {checkpoint_end_time - checkpoint_start_time}")
    end_time = time.time()
    print(f"Total time taken is {end_time - start_time}")
    return influence