import torch
import torch.nn as nn


class HATCNN(nn.Module):
    def __init__(self,numTasks):
        super(HATCNN,self).__init__()
        self.num_tasks=numTasks
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*8*8,128)  
        self.fc2=nn.Linear(128,2)  
        self.masks = nn.Parameter(torch.ones(numTasks,128)) #Masking per Tasks
    def forward(self,x,task_id):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(x.size(0),-1)
        mask=self.masks[task_id]
        x=torch.relu(self.fc1(x)*mask)
        x=self.fc2(x)
        return x
    
    
class MultiTaskCNN(nn.Module):
        def __init__(self):
            super(MultiTaskCNN, self).__init__()
            self.task_heads = nn.ModuleList([
                nn.Linear(128,2),  #Task 0
                nn.Linear(128,2), #Task 1
                nn.Linear(128,2),  #Task 2
                nn.Linear(128,2),  #Task 3
                nn.Linear(128,2)])   #Task 4 
        def forward(self, x, task_id):
            print(x.shape)
            return self.task_heads[task_id](x) 
    
    
    
class OODCNN(nn.Module):
    def __init__(self):
        super(OODCNN, self).__init__()
        self.task_heads = nn.ModuleList([
            nn.Linear(128,2),  #Task 0
            nn.Linear(128,2), #Task 1
            nn.Linear(128,2),  #Task 2
            nn.Linear(128,2),  #Task 3
            nn.Linear(128,2)])   #Task 4 
    def forward(self, x, task_id):
        return self.task_heads[task_id](x) 