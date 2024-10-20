
import torch
from torch.utils.data import DataLoader

def loadTaskDataset(trainData,taskClasses):
    taskIndices=[i for i,(_,label) in enumerate(trainData) if label in taskClasses]
    taskDataset=torch.utils.data.Subset(trainData,taskIndices)
    def remapLabels(data):
        images,labels=data
        labelMap={taskClasses[0]:0,taskClasses[1]:1}
        return images,labelMap[labels]
    remappedDataset=[(remapLabels(item)) for item in taskDataset]
    return DataLoader(remappedDataset,batch_size=64,shuffle=True)


def loadTaskTestDataset(testData,taskClasses):
    taskIndices=[i for i,(_,label) in enumerate(testData) if label in taskClasses]
    taskDataset=torch.utils.data.Subset(testData,taskIndices)
    def remapLabels(data):
        images,labels=data
        labelMap={taskClasses[0]:0,taskClasses[1]:1}
        return images,labelMap[labels]
    remappedDataset=[(remapLabels(item)) for item in taskDataset]
    return DataLoader(remappedDataset,batch_size=64,shuffle=False)