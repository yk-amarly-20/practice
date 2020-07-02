import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from Model import Net
import mlflow

def make_parse():
    """
    コマンド引数受け取り
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    return parser
    
args = make_parse().parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=args.batch_size,
                                            shuffle=False, 
                                            num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

# 学習
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[int(args.epochs*0.8), int(args.epochs*0.9)],
    gamma=0.1
)

epochs = args.epochs
result = {}
for epoch in range(epochs):
    running_loss = 0.0
    training_num = 0
    results = {}

    for i, (inputs, labels) in enumerate(trainloader, 0):
    
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
   
        optimizer.step()
        training_num += labels.size(0)

        running_loss += loss.item()
    running_loss /= training_num

    results['running_loss'] = running_loss
    print("epoch: {}, loss: {}".format(epoch + 1, running_loss))
    running_loss = 0.0

print("Finish Training")

# 精度検証

correct = 0
total = 0

"""
with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
"""
"""
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))
"""

with mlflow.start_run() as run:
    tracking_uri = './mlruns'
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "pra"
    mlflow.set_experiment(experiment_name)
    tracking = mlflow.tracking.MlflowClient()
    experiment = tracking.get_experiment_by_name('pra')
    print(experiment.experiment_id)
    mlflow.log_param('running_loss', results['running_loss'])

