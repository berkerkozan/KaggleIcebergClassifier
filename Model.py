"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import ModelTest


class FullyConnectedForPrediction(nn.Module):

    def __init__(self, hidden = 10, drop = 0):

        super(FullyConnectedForPrediction, self).__init__()
        self.batch = nn.BatchNorm2d(hidden)
        drop1 = torch.nn.Dropout(p=drop)
        
        self.fc1 = nn.Linear(7, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.fc3 = nn.Linear(hidden, 1)
        self.fc4 = nn.Linear(hidden, 1)
        
        self.relu = nn.ReLU(inplace=True)

        self.dense1 = nn.Sequential(
            self.fc1,self.batch, self.relu, drop1,self.fc2
        )
    def forward(self, x):
        return self.dense1(x)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


class CNN2(nn.Module):

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=3,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=2, dropout=0.5):

        super(CNN2, self).__init__()
        channels, height, width = input_dim

        self.convLayer = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        hidden = 20
        dropout = 0

        self.flat_fts = self.get_flat_fts([3, 75, 75], self.convLayer)
        self.dropout1 = nn.Dropout(dropout)
        self.batch1 = nn.BatchNorm1d(hidden)

        self.fc1 = nn.Linear(self.flat_fts, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.Relu = nn.ReLU(inplace=True)

        self.dense1 = nn.Sequential(
            self.fc1,
            self.batch1,
            self.Relu,
            self.dropout1
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        out = self.convLayer(x)
        out = self.convLayer(x)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        out = self.fc2(out)
        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU()
            ,nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU()
            ,nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU()
            ,nn.MaxPool2d(2)
        )

        self.convLayer = nn.Sequential(self.layer1,self.layer2,self.layer3)
        hidden = 50
        hidden2 = 50
        flat_fts = get_flat_fts([3, 75, 75], self.convLayer)

        self.fc1 = nn.Linear(flat_fts, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.fc3 = nn.Linear(hidden2, 1)

        self.batch1 = nn.BatchNorm1d(hidden)
        self.batch2 = nn.BatchNorm1d(hidden2)
        self.drop1 = torch.nn.Dropout(p=0.0)

    def forward(self, x):
        out = self.convLayer(x)
        out = out.view(out.size(0), -1)

        #out = self.drop1(F.relu(self.batch1(self.fc1(x))))
        out = self.fc1(out)
        out = F.relu(self.batch1(out))
        out = self.drop1(out)
        out = self.fc2(out)
        '''out = F.relu(self.batch2(out))
        out = self.drop1(out)
        out = self.fc3(out)'''
        return out
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

class CNN4(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU()
            ,nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU()
            ,nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU()
            ,nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(5 * 5 * 50, 50)
        self.fc2 = nn.Linear(50, 1)

        self.batch1 = nn.BatchNorm1d(50)
        #self.batch2 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)

        #out = self.drop1(F.relu(self.batch1(self.fc1(x))))
        out = self.fc1(out)
        out = F.relu(self.batch1(out))
        out = self.drop1(out)
        out = self.fc2(out)
        return out
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

class VGG16_1(nn.Module):

    def __init__(self, hidden = 1024, dropout = 0.):
        super(VGG16_1, self).__init__()
        model_conv = torchvision.models.vgg16_bn(pretrained=True)
        self.convLayers = model_conv.features

        '''for i,param in enumerate(self.convLayers.parameters()):
            #if i<=23:
                param.requires_grad = False
            #else:
            #    break'''
        self.flat_fts = self.get_flat_fts([3,75,75], self.convLayers)

        #len(self.convLayers.parameters)
        self.hiddenUnit = hidden
        self.dropout = dropout
        hidden1 = hidden
        hidden2 = hidden
        dropout = dropout

        fc1 = nn.Linear(self.flat_fts, hidden1)
        fc2 = nn.Linear(hidden1, hidden2)
        fc3 = nn.Linear(hidden2, 1)
        batch1 = nn.BatchNorm1d(hidden1)
        batch2 = nn.BatchNorm1d(hidden2)
        drop1 = torch.nn.Dropout(p=dropout)

        self.dense1 = nn.Sequential(
            drop1, fc1, batch1, nn.ReLU(inplace=True), drop1
        )
        self.dense2 = nn.Sequential(
            fc2, batch2, nn.ReLU(inplace=True), drop1, fc3
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        x = self.convLayers(x)
        x = x.view(x.size(0),-1)
        x = self.dense1(x)
        return self.dense2(x)

    @property
    def is_cuda(self):

        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

class VGG16_2(nn.Module):

    def __init__(self, hidden = 1024, dropout = 0.):
        super(VGG16_2, self).__init__()
        #model_conv = torchvision.models.vgg16_bn(pretrained=True)
        model_conv = ModelTest.vgg16_bn()
        self.convLayers = model_conv.features

        '''for i,param in enumerate(self.convLayers.parameters()):
            #if i<=23:
                param.requires_grad = False
            #else:
            #    break'''
        self.flat_fts = self.get_flat_fts([2,75,75], self.convLayers)

        #len(self.convLayers.parameters)
        self.hiddenUnit = hidden
        self.dropout = dropout
        hidden1 = hidden
        hidden2 = hidden
        dropout = dropout

        fc1 = nn.Linear(self.flat_fts, hidden1)
        fc2 = nn.Linear(hidden1, hidden2)
        fc3 = nn.Linear(hidden2, 1)
        batch1 = nn.BatchNorm1d(hidden1)
        batch2 = nn.BatchNorm1d(hidden2)
        drop1 = torch.nn.Dropout(p=dropout)

        self.dense1 = nn.Sequential(
            drop1, fc1, batch1, nn.ReLU(inplace=True), drop1
        )
        self.dense2 = nn.Sequential(
            fc2, batch2, nn.ReLU(inplace=True), drop1, fc3
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        x = self.convLayers(x)
        x = x.view(x.size(0),-1)
        x = self.dense1(x)
        return self.dense2(x)

    @property
    def is_cuda(self):

        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

class VGG16_Original_Classifier(nn.Module):

    def __init__(self, hidden = 1024, dropout = 0.5):
        super(VGG16_Original_Classifier, self).__init__()
        model_conv = torchvision.models.vgg16_bn(pretrained=True)
        self.convLayers = model_conv.features

        '''for i,param in enumerate(self.convLayers.parameters()):
            #if i<=23:
                param.requires_grad = False
            #else:
            #    break'''
        self.flat_fts = self.get_flat_fts([3,75,75], self.convLayers)

        #len(self.convLayers.parameters)

        drop1 = torch.nn.Dropout(p=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, hidden),
            nn.ReLU(True),
            drop1,
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            drop1,
            nn.Linear(hidden, 1),
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        x = self.convLayers(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

    @property
    def is_cuda(self):

        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

def get_flat_fts(in_size, fts):
    f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
    return int(np.prod(f.size()[1:]))

def get_flat_fts_and_all_shape(in_size, fts):
    f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
    return f.data.shape, int(np.prod(f.size()[1:]))

class Resnet152_1(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet152_1, self).__init__()
        model_ft = torchvision.models.resnet152(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([3, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([3, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet18_1(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet18_1, self).__init__()
        model_ft = torchvision.models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([3, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([3, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet18_2(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet18_2, self).__init__()
        model_ft = ModelTest.resnet18()

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([2, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([2, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet34_1(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet34_1, self).__init__()
        model_ft = torchvision.models.resnet34(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([3, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([3, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet50_1(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet50_1, self).__init__()
        model_ft = torchvision.models.resnet50(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([3, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([3, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet50_2(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet50_2, self).__init__()
        model_ft = ModelTest.resnet50()

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([2, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([2, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet101_1(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet101_1, self).__init__()
        model_ft = torchvision.models.resnet101(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([3, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([3, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Resnet101_2(nn.Module):
    def __init__(self,dropout = 0):
        super(Resnet101_2, self).__init__()
        model_ft = ModelTest.resnet101()

        self.features = nn.Sequential(*list(model_ft.children())[:-2])

        shape,_ = get_flat_fts_and_all_shape([2, 75, 75], self.features)
        self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        nnSize = get_flat_fts([2, 75, 75], self.features)



        #num_ftrs = model_ft.fc.in_features
        #self.fc = nn.Linear(nnSize, 1)
        #drop1 = torch.nn.Dropout(p=dropout)

        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))

        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class InceptionV3_1(nn.Module):
    def __init__(self,dropout = 0):
        super(InceptionV3_1, self).__init__()
        #model_ft = torchvision.models.inception_v3(pretrained=True,num_classes=1, aux_logits = False)
        model_ft = torchvision.models.inception_v3(pretrained=False, num_classes=1, aux_logits=False)

        self.features = nn.Sequential(*list(model_ft.children())[:-1])

        nnSize = get_flat_fts([3, 75, 75], self.features)

        #self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        #nnSize = get_flat_fts([3, 75, 75], self.features)

        #num_ftrs = model_ft.fc.in_features
        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))
        #drop1 = torch.nn.Dropout(p=dropout)
        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''


    def forward(self, x):

        x = self.features(x)
        #x = x.view(x.size(0),-1)
        #x = self.dense1(x)
        #return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Densenet121_1(nn.Module):
    def __init__(self, dropout=0):
        super(Densenet121_1, self).__init__()
        # model_ft = torchvision.models.inception_v3(pretrained=True,num_classes=1, aux_logits = False)
        model_ft = torchvision.models.densenet121(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-1])

        nnSize = get_flat_fts([3, 75, 75], self.features)

        # self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        # nnSize = get_flat_fts([3, 75, 75], self.features)

        # num_ftrs = model_ft.fc.in_features
        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))
        # drop1 = torch.nn.Dropout(p=dropout)
        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0),-1)
        # x = self.dense1(x)
        # return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Densenet161_1(nn.Module):
    def __init__(self, dropout=0):
        super(Densenet161_1, self).__init__()
        # model_ft = torchvision.models.inception_v3(pretrained=True,num_classes=1, aux_logits = False)
        model_ft = torchvision.models.densenet161(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-1])

        nnSize = get_flat_fts([3, 75, 75], self.features)

        # self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        # nnSize = get_flat_fts([3, 75, 75], self.features)

        # num_ftrs = model_ft.fc.in_features
        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))
        # drop1 = torch.nn.Dropout(p=dropout)
        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0),-1)
        # x = self.dense1(x)
        # return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Densenet201_1(nn.Module):
    def __init__(self, dropout=0):
        super(Densenet201_1, self).__init__()
        # model_ft = torchvision.models.inception_v3(pretrained=True,num_classes=1, aux_logits = False)
        model_ft = torchvision.models.densenet201(pretrained=True)

        self.features = nn.Sequential(*list(model_ft.children())[:-1])

        nnSize = get_flat_fts([3, 75, 75], self.features)

        # self.features = nn.Sequential(*list(model_ft.children())[:-2],nn.AvgPool2d(shape[3]))
        # nnSize = get_flat_fts([3, 75, 75], self.features)

        # num_ftrs = model_ft.fc.in_features
        self.fc = nn.Sequential(torch.nn.Dropout(p=dropout), nn.Linear(nnSize, 1))
        # drop1 = torch.nn.Dropout(p=dropout)
        # Everything except the last linear layer

        '''self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        self.modelName = 'resnet'''''

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0),-1)
        # x = self.dense1(x)
        # return self.dense2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class VGG11_1(nn.Module):

    def __init__(self, hidden = 1024, dropout = 0.):
        super(VGG11_1, self).__init__()
        model_conv = torchvision.models.vgg11_bn(pretrained=True)
        self.convLayers = model_conv.features

        '''for i,param in enumerate(self.convLayers.parameters()):
            #if i<=23:
                param.requires_grad = False
            #else:
            #    break'''
        self.flat_fts = self.get_flat_fts([3,75,75], self.convLayers)
        drop1 = torch.nn.Dropout(p=dropout)
        batch1 = nn.BatchNorm1d(hidden)

        self.classifier = nn.Sequential(
            drop1,  #Try deleting this..
            nn.Linear(self.flat_fts, hidden),
            batch1,
            nn.ReLU(inplace=True),
            drop1,
            nn.Linear(hidden, hidden),
            batch1,
            nn.ReLU(inplace=True),
            drop1,
            nn.Linear(hidden, 1)
        )



    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        x = self.convLayers(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

    @property
    def is_cuda(self):

        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

class VGG11_Original_Classifier(nn.Module):

    def __init__(self, hidden = 1024, dropout = 0.5):
        super(VGG11_Original_Classifier, self).__init__()
        model_conv = torchvision.models.vgg11_bn(pretrained=True)
        self.convLayers = model_conv.features

        '''for i,param in enumerate(self.convLayers.parameters()):
            #if i<=23:
                param.requires_grad = False
            #else:
            #    break'''
        self.flat_fts = self.get_flat_fts([3,75,75], self.convLayers)

        #len(self.convLayers.parameters)

        drop1 = torch.nn.Dropout(p=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, hidden),
            nn.ReLU(True),
            drop1,
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            drop1,
            nn.Linear(hidden, 1),
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        x = self.convLayers(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

    @property
    def is_cuda(self):

        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

class VGG19_1(nn.Module):

    def __init__(self, hidden = 1024, dropout = 0.):
        super(VGG19_1, self).__init__()
        model_conv = torchvision.models.vgg19_bn(pretrained=True)
        self.convLayers = model_conv.features

        '''for i,param in enumerate(self.convLayers.parameters()):
            #if i<=23:
                param.requires_grad = False
            #else:
            #    break'''
        self.flat_fts = self.get_flat_fts([3,75,75], self.convLayers)
        drop1 = torch.nn.Dropout(p=dropout)
        batch1 = nn.BatchNorm1d(hidden)

        self.classifier = nn.Sequential(
            drop1,  # Try deleting this..
            nn.Linear(self.flat_fts, hidden),
            batch1,
            nn.ReLU(inplace=True),
            drop1,
            nn.Linear(hidden, hidden),
            batch1,
            nn.ReLU(inplace=True),
            drop1,
            nn.Linear(hidden, 1)
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(torch.autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        x = self.convLayers(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

    @property
    def is_cuda(self):

        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)
