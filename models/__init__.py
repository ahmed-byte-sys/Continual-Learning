from .model import HATCNN,MultiTaskCNN,OODCNN


def createModel(HATCNN,MutliTaskCNN,OODCNN):
    featureExtractor = HATCNN()
    wpModel=MultiTaskCNN()
    oodModel=OODCNN()
    return featureExtractor,wpModel,oodModel
