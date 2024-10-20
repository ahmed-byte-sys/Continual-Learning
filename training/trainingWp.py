import random
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
from Dataloader import loadTaskDataset
from models import MultiTaskCNN
from Buffer import ReplayBuffer



def evaluateTaskAccuracy(model,featureExtractor,taskLoader,taskNum):
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for images,labels in taskLoader:
            features=featureExtractor(images,taskNum)
            outputs=model(features,taskNum).argmax(dim=1)
            correct+=(outputs==labels).sum().item()
            total+=labels.size(0)
    return correct/total if total>0 else 0

def trainIncrementalClassTasks(featureExtractor,wpModel,trainDataset,classGroups,epochs=10):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(list(wpModel.parameters()),lr=0.001)
    buffer=ReplayBuffer(size=1000)
    task_accuracies,avg_forgetting_rate=[],[]
    for taskNum,taskClasses in enumerate(classGroups):
        print(f"\nTraining on task {taskNum+1} with classes: {taskClasses}")
        taskLoader=loadTaskDataset(trainDataset,taskClasses)
        XTaskList,yTaskList=[],[]
        for images,labels in taskLoader:
            XTaskList.append(images)
            yTaskList.append(labels)
        XTask=torch.cat(XTaskList)
        yTask=torch.cat(yTaskList)
        XBuffer,yBuffer=buffer.getSamples()
        if XBuffer is not None and len(XBuffer)>0:
            XReplay,yReplay=XBuffer,yBuffer
        else:
            XReplay,yReplay=None,None
        for epoch in range(epochs):
            featureExtractor.train()
            wpModel.train()
            optimizer.zero_grad()
            features=featureExtractor(XTask,taskNum)
            outputs=wpModel(features,taskNum)
            loss=criterion(outputs,yTask)
            if XReplay is not None:
                features=featureExtractor(XReplay,taskNum)
                replayOutputs=wpModel(features,taskNum)
                replayLoss=criterion(replayOutputs,yReplay)
                loss+=replayLoss
            loss.backward()
            optimizer.step()
            print(f"Task {taskNum+1}, Epoch {epoch+1}, Loss: {loss.item()}")
        task_accuracy=evaluateTaskAccuracy(wpModel,featureExtractor,taskLoader,taskNum)
        print(f"Accuracy on task {taskNum+1}: {task_accuracy*100:.2f}%")
        task_accuracies.append([task_accuracy])
        for i in range(taskNum):
            previous_task_loader=loadTaskDataset(trainDataset,classGroups[i])
            accuracy_after_task=evaluateTaskAccuracy(wpModel,featureExtractor,previous_task_loader,taskNum)
            task_accuracies[i].append(accuracy_after_task)
            print(f"Accuracy on task {i+1} after learning task {taskNum+1}: {accuracy_after_task*100:.2f}%")
        if taskNum>0:
            forgetting_rates=[]
            for i in range(taskNum):
                initial_accuracy=task_accuracies[i][0]
                current_accuracy=task_accuracies[i][-1]
                forgetting=initial_accuracy-current_accuracy
                forgetting_rates.append(forgetting)
            avg_forgetting_rate.append(sum(forgetting_rates)/len(forgetting_rates))
            print(f"Average Forgetting Rate after task {taskNum+1}: {avg_forgetting_rate[-1]:.4f}")
        buffer.addSamples(XTask,yTask)
    aca=sum([task_accuracies[i][-1] for i in range(len(classGroups))])/len(classGroups)
    print(f"\nAverage Classification Accuracy (ACA) after the last task: {aca:.4f}")
    return aca,avg_forgetting_rate

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
trainDataset=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
testDataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
classGroups=[[0,1],[2,3],[4,5],[6,7],[8,9]]
wpModel=MultiTaskCNN()
featureExtractor=model.load_state_dict(torch.load("feature_extractor.pth"))
trainIncrementalClassTasks(featureExtractor,wpModel,trainDataset,classGroups)
torch.save(wpModel.state_dict(),"wpModel.pth")

