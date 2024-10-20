import random
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
from Dataloader import loadTaskDataset
from models import OODCNN,HATCNN






def traingOODheadWithFeatures(trainData,taskClasses,oodModel,featureExtractor,epochs=10):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(list(featureExtractor.parameters())+list(oodModel.parameters()),lr=0.001)
    for taskNum,taskClasses in enumerate(classGroups):
        print(f"\nTraining on task {taskNum+1} with classes: {taskClasses}")
        taskLoader=loadTaskDataset(trainDataset,taskClasses)
        XTaskList,yTaskList=[],[]
        for images,labels in taskLoader:
            XTaskList.append(images)
            yTaskList.append(labels)
        XTask=torch.cat(XTaskList)
        yTask=torch.cat(yTaskList)
        for epoch in range(epochs):
            featureExtractor.train()
            oodModel.train()
            optimizer.zero_grad()
            features=featureExtractor(XTask,taskNum)
            outputs=oodModel(features,taskNum)
            loss=criterion(outputs,yTask)
            loss.backward()
            optimizer.step()
            print(f"Task {taskNum+1}, Epoch {epoch+1}, Loss: {loss.item()}")

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
trainDataset=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
testDataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
classGroups=[[0,1],[2,3],[4,5],[6,7],[8,9]]
featureExtractor=HATCNN(numTasks=len(classGroups))
oodModel=OODCNN()
traingOODheadWithFeatures(trainDataset,classGroups,oodModel,featureExtractor,epochs=1)
ood_model_path='ood_model.pth'
feature_extractor_path='feature_extractor.pth'
torch.save(oodModel.state_dict(),ood_model_path)
torch.save(featureExtractor.state_dict(),feature_extractor_path)
print("Models saved successfully.")
