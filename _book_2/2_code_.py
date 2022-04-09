import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

"""
Setting up DataLoaders
We'll use the built-in dataset of torchvision.datasets.
ImageFolder to quickly set up some dataloaders of downloaded cat and fish images.

check_image is a quick little function that is passed to the is_valid_file 
parameter in the ImageFolder and will do a sanity check to make sure PIL 
can actually open the file. We're going to use this in lieu of 
cleaning up the downloaded dataset.
"""

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

#
"""
Set up the transforms for every image:

Resize to 64x64
Convert to tensor
Normalize using ImageNet mean & std        
"""
#
img_transforms = transforms.Compose([
    transforms.Resize((64,64)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

#

"""
Our First Model, SimpleNet
SimpleNet is a very simple combination of three Linear layers and ReLu activations between them. 
Note that as we don't do a softmax() in our forward(), 
we will need to make sure we do it in our training function during the validation phase.
"""
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)
    
    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#


        
## Test -- Cat == 100
## Test - airplane == 
## Train - airplane == 
## Val - airplane == 

train_data_path = "./train" # TODO --glass/
#train_data_path = "./train/cat"
val_data_path = "./val" # TODO 
#val_data_path = "./val/cat"
#test_data_path = "./test/airplane" # TODO 
#
"""
# Optimizers --- Official >> https://pytorch.org/docs/stable/optim.html
# Smaller -- Lr == better convergence , lesser chance to be stuck in 
# Local Minima during Gradient Descent. 

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adam([var1, var2], lr=0.0001)
"""
# Define -- TRAIN 
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    print("--train--device----\n",device)
    for epoch in range(1, epochs+1):
        training_loss = 0.0 ## INIT Vals 
        valid_loss = 0.0
        model.train()
        #
        for batch in train_loader:
            optimizer.zero_grad() # Delete All Gradients from Last RUN | Last Batch
            inputs, targets = batch # Targets DICT 
            print("---targets----",targets)
            inputs = inputs.to(device) # Inputs to CUDA - GPU 
            targets = targets.to(device) # Targets DICT to CUDA - GPU 
            print("--train>>-model--summary----",model)
            output = model(inputs)
            loss = loss_fn(output, targets)
            print("--train>>-model-loss---",loss)

            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval() ## Evaluation Mode
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            print("--------targets.shape------",targets.shape)
            print("--------inputs.size(0)------",inputs.size(0))
            inputs = inputs.to(device) # Inputs to CUDA - GPU 
            output = model(inputs)
            print("--train>>-model--summary-2---",model)
            print("--------output.size(0)------",output.size(0))
            targets = targets.to(device) # Targets DICT to CUDA - GPU 
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0] # INCRement 
        valid_loss /= len(val_loader.dataset)
        print("-TRAIN----torch.cuda.memory_allocated---> {:.2f} GB".format(torch.cuda.memory_allocated()/1024**3))
        print('Epoch:{},TRAIN_Loss:{:.2f},VAL_Loss:{:.2f}, Accuracy = {:.2f}'.format(epoch,training_loss,valid_loss, num_correct / num_examples))

"""
Making predictions --- 
Labels are in alphanumeric order, so cat will be 0, fish will be 1. 
We'll need to transform the image and also make sure that the resulting tensor is 
copied to the appropriate device before applying our model to it.
"""
#

if __name__ == "__main__":
    
    train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)
    print("-------train_data----",type(train_data)) #-------train_data---- <class 'torchvision.datasets.folder.ImageFolder'>
    
    val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms, is_valid_file=check_image)
    print("-------val_data----",type(val_data)) #-------val_data---- <class 'torchvision.datasets.folder.ImageFolder'>
    
    #TODO -- test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image)
    #TODO -- print("-------test_data----",type(test_data)) #-------test_data---- <class 'torchvision.datasets.folder.ImageFolder'>

    batch_size = 64 #Book-->> 64 | MiniBatches -- official == 1

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4)
    print("-------train_data_loader----",type(train_data_loader))
    val_data_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=True, num_workers=4) 
    #TODO -- test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=4) 
    #
    simplenet = SimpleNet()
    print("-----type(simplenet-------\n",type(simplenet))
    #
    optimizer = optim.Adam(simplenet.parameters(), lr=0.001)
    print("-----type(optimizer-------\n",type(optimizer))
    #
    print("--1-torch.cuda.memory_allocated---> {:.2f} GB".format(torch.cuda.memory_allocated()/1024**3))
    # move model to GPU 
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    simplenet.to(device)
    #
    print("--2-torch.cuda.memory_allocated---> {:.2f} GB".format(torch.cuda.memory_allocated()/1024**3))
    #
    train(simplenet, optimizer,torch.nn.CrossEntropyLoss(), train_data_loader,val_data_loader, epochs=10, device=device)
    #
    #labels = ['airplane','not_plane']
    labels = ['glass','not_glass']

    # img1 = Image.open("./val/not_plane/image_0001.jpg")
    # img2 = Image.open("./val/not_plane/image_0003.jpg")
    img1 = Image.open("./val/glass/glass22.jpg")
    img2 = Image.open("./val/glass/glass21.jpg")
    img3 = Image.open("./val/glass/glass13.jpg")
    img4 = Image.open("./val/not_glass/image_0007.jpg")
    img5 = Image.open("./val/not_glass/image_0011.jpg")
    img6 = Image.open("./val/not_glass/image_0001.jpg")
    img7 = Image.open("./val/not_glass/image_0003.jpg")
    img8 = Image.open("./val/glass/glass4.jpg")
    img9 = Image.open("./val/glass/glass5.jpg")
    img10 = Image.open("./val/glass/glass6.jpg")

    ls_test_images = [img3,img4,img9,img8,img7,img6,img10,img1,img2,img3]
    #ls_test_images = [img1,img2,img3,img4,img5,img10,img9,img8,img7,img6,img10]


    for image_test in range(len(ls_test_images)):
        test_image = ls_test_images[image_test]
        test_image.show()


        img = img_transforms(ls_test_images[image_test]).to(device)
        img = torch.unsqueeze(img, 0)

        simplenet.eval()
        prediction = F.softmax(simplenet(img), dim=1)
        prediction = prediction.argmax()
        print("---labels[prediction]----\n",labels[prediction]) 











        # #TypeError: Expected Ptr<cv::UMat> for argument 'mat'
        # img = img.cpu()
        # image = np.array(image_test)
        # cv2.imshow('Output', image_test)
        # cv2.waitKey(0)

        # #TypeError: Expected Ptr<cv::UMat> for argument 'mat'
        # img = img.cpu()
        # image = np.array(img)
        # cv2.imshow('Output', image)
        # cv2.waitKey(0)