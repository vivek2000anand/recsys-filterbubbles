import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from tracin import save_tracin_checkpoint, load_tracin_checkpoint, calculate_tracin_influence

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_gradients(self):
        """Get gradients for tracin method. (Do not call individually)
        """
        list_params = list(self.parameters())
        gradients = torch.cat([torch.flatten(l.grad) for l in list_params])
        return gradients

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
train_subset = torch.utils.data.Subset(trainset, [0, 1, 5, 6, 999])
trainloader_subset = torch.utils.data.DataLoader(train_subset, batch_size=1, num_workers=0, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_subset = torch.utils.data.Subset(testset, [0, 1, 10, 6, 8])
testloader_subset = torch.utils.data.DataLoader(test_subset, batch_size=1, num_workers=0, shuffle=False)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

paths = []
# Train loop
for epoch in range(20):
    # Adding the train loop
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    # Saving the tracin checkpoint
    path = "conv_test_epoch" + str(epoch) + ".pt"
    paths.append(path)
    save_tracin_checkpoint(model, epoch, running_loss, optimizer, path)

trainloader_subset = iter(trainloader_subset)
testloader_subset = iter(testloader_subset)
# TracIn testing
source, source_label = next(trainloader_subset)
target, target_label = next(testloader_subset)
# print("source ", source)
influence = calculate_tracin_influence(Net, source, source_label, target, target_label, "SGD", nn.CrossEntropyLoss(), paths)
print("Influence for train 1 and test 1 is ", influence)
print("______________________________________________________________________________________________________________________")
source, source_label = next(trainloader_subset)
target, target_label = next(testloader_subset)
# print("source ", source)
influence = calculate_tracin_influence(Net, source, source_label, target, target_label, "SGD", nn.CrossEntropyLoss(), paths)
# print("Influence for train 2 and test 2 is ", influence)
print("______________________________________________________________________________________________________________________")
source, source_label = next(trainloader_subset)
target, target_label = next(testloader_subset)
# print("source ", source)
influence = calculate_tracin_influence(Net, source, source_label, target, target_label, "SGD", nn.CrossEntropyLoss(), paths)
print("Influence for train 5 and test 10 is ", influence)
print("______________________________________________________________________________________________________________________")
source, source_label = next(trainloader_subset)
# print("source ", source)
target, target_label = next(testloader_subset)
influence = calculate_tracin_influence(Net, source, source_label, target, target_label, "SGD", nn.CrossEntropyLoss(), paths)
print("Influence for train 6 and test 6 is ", influence)
print("______________________________________________________________________________________________________________________")
source, source_label = next(trainloader_subset)
target, target_label = next(testloader_subset)
# print("source ", source)
influence = calculate_tracin_influence(Net, source, source_label, target, target_label, "SGD", nn.CrossEntropyLoss(), paths)
print("Influence for train 999 and test 8 is ", influence)
print("______________________________________________________________________________________________________________________")