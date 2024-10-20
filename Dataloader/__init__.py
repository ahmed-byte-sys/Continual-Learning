from loading import loadTaskDataset,loadTaskTestDataset
from torchvision import datasets, transforms


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
trainDataset=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

def createLoader(taskClasses):
    return loadTaskDataset(trainDataset,taskClasses)