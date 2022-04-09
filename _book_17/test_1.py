



## SOURCE -- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
## SOURCE --  https://github.com/PacktPublishing/Python-Deep-Learning-Second-Edition/blob/master/Chapter05/chapter_05_001.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms

batch_size = 50

# training data
train_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])
## Normalization will prevent the --exploding gradient problem. 
## [0.485, 0.456, 0.406] == MEAN --- of the ENTIRE Dataset in IMAGE-NeT
## [0.229, 0.224, 0.225] == STD_DEVIATION --- of the ENTIRE Dataset in IMAGE-NeT
## transforms.Normalize(
##FOOBAR## https://discuss.pytorch.org/t/understanding-transform-normalize/21730/5

train_set = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=train_data_transform)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

# validation data
val_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_set = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=val_data_transform)

val_order = torch.utils.data.DataLoader(val_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    print("----in_here----train_model(----")
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        print("----train_model(--labels----",labels)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels) ## certain cases -- its TARGETS and not LABELS 

            # backward
            loss.backward()
            optimizer.step() 
            #update weights of network -- so as to make loss function return as small a loss as possible.
            # gradient descent is happening here - trying to converge on best SET of WEIGHTS - also not getting trapped into a local minima
            # each HOP along the descent is the Value of the LEARNING RATE - lr 

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train_Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval() #SO -- https://stackoverflow.com/a/60018731/4928635 
    # model.eval() have effect only on Layers, not on gradients, 
    #by default grad comp is switch on, but using context manager torch.no_grad() 
    #during evaluation allows you easily turn off and then autimatically turn on gradients comp

    current_loss = 0.0
    current_acc = 0  ## CURRENT ACCURACY 

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        print("----labels----",labels)
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False): # NOT TRAIN -- test_model-- thus FALSE 
            #This can be used to conditionally enable gradients.
            #https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html
            outputs = model(inputs)
            print("----type(outputs---",type(outputs))
            print("----outputs.shape---",outputs.shape)

            _, predictions = torch.max(outputs, 1) ##https://pytorch.org/docs/stable/generated/torch.max.html
            loss = loss_function(outputs, labels) # loss_function = nn.CrossEntropyLoss()
            # certain cases -- its TARGETS ( dict ) and not LABELS 

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
        print("--current_loss--\n",current_loss)
        print("--current_acc--\n",current_acc)
    print("-----len(data_loader.dataset)------",len(data_loader.dataset))

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test_Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def tl_feature_extractor(epochs=3):
    """
    ## Epochs == 3 -- Means we ITERATE over the TRAINING  data 3 Times 

    Locally disabling gradient computation -- For more fine-grained exclusion of subgraphs from gradient computation, 
    there is setting the --- requires_grad --- field of a tensor.
    https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc

    """
    # load the pre-trained model
    model = torchvision.models.resnet18(pretrained=True)

    # exclude existing parameters from backward pass for performance
    #print("--type(model.parameters()---\n",type(model.parameters())) #FOOBAR# <class 'generator'>
    for param in model.parameters():
        #print("----type(param---\n",type(param)) #FOOBAR##<class 'torch.nn.parameter.Parameter'>
        #print("----tl_feature_extractor--param---\n",param) #Look at LOG FILE -->> /_book_15_yolo/log_7_.log
        print("----tl_feature_extractor--type(param.data), param.size()---\n",type(param.data), param.size())
        param.requires_grad = False

    # newly constructed layers have requires_grad=True by default
    ## https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    num_features = model.fc.in_features 
    print("---model.fc.in_features--\n",num_features) ## 512
    #FOOBAR# https://discuss.pytorch.org/t/what-does-the-fc-in-feature-mean/4889
    ## Number of Input_Features --  ACTUAL COUNT of Input Features for Linear Layer -- in_feature is the number of inputs for your linear layer

    model.fc = nn.Linear(num_features, 10) #10 is count of OUTPUT Features --# 10 OutPuts from this layer -- mapping to each CIFAR-10 Class
    #FOOBAR -- nn.Linear-- Applies a linear transformation to the incoming data #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    print("----tl_feature_extractor-type(model.fc----",type(model.fc)) ##<class 'torch.nn.modules.linear.Linear'>


    ### #Look at LOG FILE -->> /_book_15_yolo/log_7_.log
    # from torchinfo import summary
    # test_model_batch_size = 10
    # print("---tl_feature_extractor---summary(model, input_size=(test_model_batch_size, 3, 28, 28)----",summary(model, input_size=(test_model_batch_size, 3, 28, 28)))
    # print("---tl_feature_extractor---summary(model)-----",summary(model))
    # #
    # print("---tl_feature_extractor---model----",model)
    model = model.to(device) # transfer to GPU (if available)

    ### FOOBAR --
    loss_function = nn.CrossEntropyLoss()

    
    ### FOOBAR -- as seen below -- only the  Newly created -  Fully Connected Layer is being passed in to the OPTIMIZER
    ## SO -- https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter
    # Newly created - for CIFAR 10 , Fully Connected Layer's Parameters -->> model.fc.parameters()
    #print("------model.fc.parameters()------\n",model.fc.parameters()) ##<generator object Module.parameters at 0x7f0846f07d60>
    # only parameters of the final layer are being optimized
    #optimizer = optim.Adam(model.fc.parameters()) ## Original book code 
    ## FOOBAR -- why dont we specify a LEARNING RATE here ?? 
    ## could be == optimizer = optim.Adam(model.fc.parameters(),lr=0.001)
    optimizer = optim.Adam(model.fc.parameters(),lr=0.001) # TEST -- ,lr=0.001

    # train
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, val_order)


def tl_fine_tuning(epochs=3):
    ## Epochs == 3 -- Means we ITERATE over the TRAINING  data 3 Times
    # load the pre-trained model
    model = models.resnet18(pretrained=True)

    # replace the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10) # 10 OutPuts from this layer -- mapping to each CIFAR-10 Class

    # transfer the model to the GPU
    model = model.to(device)

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # We'll optimize all parameters
    optimizer = optim.Adam(model.parameters())

    # train
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, val_order)


if __name__ == '__main__':
    tl_feature_extractor(epochs=1) # Original = 5 
    #tl_fine_tuning(epochs=5)



"""
### https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/9
#https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/12

import argparse
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms

dataset_names = ('cifar10','cifar100','mnist')

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')

args = parser.parse_args()

data_dir = os.path.join('.', args.dataset)

print(args.dataset)

if args.dataset == "cifar10":
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    #print(vars(train_set))
    print(train_set.train_data.shape)
    print(train_set.train_data.mean(axis=(0,1,2))/255)
    print(train_set.train_data.std(axis=(0,1,2))/255)

elif args.dataset == "cifar100":
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    #print(vars(train_set))
    print(train_set.train_data.shape)
    print(np.mean(train_set.train_data, axis=(0,1,2))/255)
    print(np.std(train_set.train_data, axis=(0,1,2))/255)

elif args.dataset == "mnist":
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    #print(vars(train_set))
    print(list(train_set.train_data.size()))
    print(train_set.train_data.float().mean()/255)
    print(train_set.train_data.float().std()/255)
"""    