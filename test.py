import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Dataloader import loadTaskTestDataset



def testModel(featureExtractor,wpModel,testDataset,classGroups):
    wpModel.eval()
    totalCorrect=0
    totalSamples=0
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for taskNum,taskClasses in enumerate(classGroups):
            print(f"\nTesting on task {taskNum+1} with classes: {taskClasses}")
            taskTestLoader=loadTaskTestDataset(testDataset,taskClasses)
            taskCorrect=0
            taskTotal=0
            taskLoss=0
            for images,labels in taskTestLoader:
                features=featureExtractor(images,taskNum)
                outputs=wpModel(features,taskNum)
                loss=criterion(outputs,labels)
                taskLoss+=loss.item()
                _,predicted=torch.max(outputs,1)
                taskTotal+=labels.size(0)
                taskCorrect+=(predicted==labels).sum().item()
            taskAccuracy=100*taskCorrect/taskTotal
            avgLoss=taskLoss/len(taskTestLoader)
            print(f"Task {taskNum+1} Accuracy: {taskAccuracy:.2f}%")
            print(f"Task {taskNum+1} Loss: {avgLoss:.4f}")
            totalCorrect+=taskCorrect
            totalSamples+=taskTotal
    overallAccuracy=100*totalCorrect/totalSamples
    print(f"\nOverall Accuracy on all tasks: {overallAccuracy:.2f}%")

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
testDataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
classGroups=[[0,1],[2,3],[4,5],[6,7],[8,9]]
featureExtractor = model.load_state_dict(torch.load("featureExtractor.pth"))
wpModel = model.load_state_dict(torch.load("wpModel"))
testModel(featureExtractor,wpModel,testDataset,classGroups)
