import torch

class ReplayBuffer:
    def __init__(self,size):
        self.size=size
        self.buffer=[]
    def addSamples(self,X,y):
        if len(self.buffer)>=self.size:
            self.buffer=self.buffer[len(X):] 
        self.buffer.extend(list(zip(X,y)))
    def getSamples(self):
        if len(self.buffer)==0:
            return None,None
        X,y=zip(*self.buffer)
        return torch.stack(X),torch.stack(y)