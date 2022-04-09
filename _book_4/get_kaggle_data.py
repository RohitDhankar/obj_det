## SOURCE -- #_book_4 ==  https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch

# TODO--07MARCH_Save_Model_STATE_DICT- ## https://pytorch.org/tutorials/beginner/saving_loading_models.html
## FOOBAR -- Save Model STATE_DICT -->>  Optimizer objects (torch.optim) also have a state_dict, which contains 
## information about the optimizer’s state, as well as the hyperparameters used.



#TODO - ownExpCode_MemmoryIssues

## Similar -- Official Pytorch Tutorial -->  https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
## Similar -- Official Pytorch Tutorial -->  _book_4/mask_rcnn_official_PytorchExample/torchvision_finetuning_instance_segmentation.py



# kaggle data sixhky/open-images-bus-trucks/
# 
#/kaggle_data/images_bus_trucks/images/images/

import os
import glob
from torch_snippets import *
from torch_snippets import Glob, stem, show, read
from PIL import Image
import pandas as pd
import torch
import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
#import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_ROOT = '../../kaggle_data/images_bus_trucks/images/images/'

DF_RAW = df_init_labels = pd.read_csv('../../kaggle_data/images_bus_trucks/df.csv')
#print("----",DF_RAW.head())

"""
Source Official Tute PyTorch --> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

One note on the labels. The model considers class 0 as background. 
If your dataset does not contain the background class, you should not have 0 in your labels. 
For example, assuming you have just two classes, cat and dog, you can define 1 (not 0) ## NOT ==0 , it will be BACKGROUND
to represent cats and 2 to represent dogs. So, for instance, if one of the images has both classes,
your labels tensor should look like [1,2].
"""
label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
print("-label2target---\n",label2target)
label2target['background'] = 0 ## add a key to DICT -- named background -- VALUE == 0 
print("-label2target---\n",label2target)
target2label = {t:l for l,t in label2target.items()}
print("-target2label---\n",target2label) ## Reverse DICT -- VALS are now KEYS --- and KEYS are VALS 
background_class = label2target['background']
num_classes = len(label2target)
#
def preprocess_image(img):
    """ display each PyTorch-Tensor [IMAGE] as an image """
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()
#
# 
class OpenDataset(torch.utils.data.Dataset):
    """
    Source Official Tute PyTorch --> _book_4/mask_rcnn_official_PytorchExample/torchvision_finetuning_instance_segmentation.py
    See the DocString for ==> # Defining the Dataset
    
    1/ The reference scripts for training object detection, 
    instance segmentation and 
    person keypoint detection -- allows for easily supporting adding new custom datasets. 
    
    2/ The dataset should inherit from the standard torch.utils.data.Dataset class, and implement 
    METHOD ==>> __len__ 
    and 
    METHOD ==>> __getitem__.
    
    3/ As has been done below in the book code for their Custom dataSet 

    4/ Additionally, if you want to use aspect ratio grouping during training 
    (so that each batch only contains images with similar aspect ratios), 
    then it is recommended to also implement a get_height_and_width method, 
    which returns the height and the width of the image. 
    If this method is not provided, we query all elements of the dataset via __getitem__ , 
    which loads the image in memory and is slower than if a custom method is provided
    """

    w, h = 224, 224
    def __init__(self, df_init_labels, image_dir=IMAGE_ROOT):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir+'/*')
        self.df = df_init_labels
        self.image_infos = df_init_labels.ImageID.unique()
    #
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix] ## FOOBAR -- ix is ID ?? 
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        
        #print("---__getitem__----df_init_labels.head()-----",df_init_labels.head())

        data = df_init_labels[df_init_labels['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist() # Convert Pandas Series Values to List 
        data = data[['XMin','YMin','XMax','YMax']].values

        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates
        #print("----foobar----len(boxes",len(boxes))

        # torch FRCNN expects ground truths as a dictionary of tensors
        ## FOOBAR -- See here for details of the STRUCTURE of the DICT ==> target 
        ## https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor([label2target[i] for i in labels]).long()
        #print("--------target--",target) #list(target.keys())

        img = preprocess_image(img)
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch)) 

    def __len__(self):
        return len(self.image_infos)


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

trn_ids, val_ids = train_test_split(df_init_labels.ImageID.unique(), test_size=0.1, random_state=99)
trn_df, val_df = df_init_labels[df_init_labels['ImageID'].isin(trn_ids)], df_init_labels[df_init_labels['ImageID'].isin(val_ids)] ## Slicing DF with Indexing 
# print("----len(trn_df)----",len(trn_df))
# print("----len(val_df)----",len(val_df))
#


train_ds = OpenDataset(trn_df)
#print("----type(train_ds)----",type(train_ds))
test_ds = OpenDataset(val_df)
#
low_batch_size=1 ## Original Book Code == 4

train_loader = DataLoader(train_ds, batch_size=low_batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
#print("----type(train_loader)----",type(train_loader)) #  <class 'torch.utils.data.dataloader.DataLoader'>
test_loader = DataLoader(test_ds, batch_size=low_batch_size, collate_fn=test_ds.collate_fn, drop_last=True)
#



import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model():
    """
    All code below is as-is from Official Tute -- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    """
    ## # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    ### get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    ### replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #print("----get_model---",model)
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_fasterrcnn_resnet50 = get_model().to(device)

"""
See own Log File for the SUMMARY of Model >> _book_4/logging_get_kaggle_data_1.log

### FOOBAR -- book comment --PAGE-363- We notice the following: GeneralizedRCNNTransform is a simple resize followed 
by a normalize transformation:
-- BackboneWithFPN is a neural network that transforms input into a feature map.
-- RegionProposalNetwork generates the anchor boxes for the preceding feature map 
and predicts individual feature maps for classification and regression tasks
"""

from torchsummary import summary
# model_summary = summary(model, input_size=(3, 800, 800))
# print("--model_summary---",model_summary)
#


# Defining training and validation functions for a single batch
def train_batch(inputs, model, optimizer):
    model.train() # model_fasterrcnn_resnet50
    input, targets = inputs
    
    #print("--train_batch--type(input----",type(input))

    input = list(image.to(device) for image in input)
    #print("--train_batch--len(input----",len(input))
    #
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #
    optimizer.zero_grad()
    #
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    #
    optimizer.step()
    return loss, losses

#torch.no_grad() -- Already done for validation 

@torch.no_grad() # decorator method - this will disable gradient computation in the function below
def validate_batch(inputs, model):
    model.train() # to obtain the losses, model needs to be in train mode only. 
    # Note that here we are not defining the model's forward method 
    #and hence need to work per the way the model class is defined
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses

#
from torch_snippets import Report
#model = get_model().to(device) ## Done above 

## TODO--07MARCH_Save_Model_STATE_DICT
# Own Code -- Not run in Jupyter
optimizer = torch.optim.SGD(model_fasterrcnn_resnet50.parameters(), lr=0.5)#,momentum=0.9, weight_decay=0.0005)

#Book Code --->> run in Jupyter
#optimizer = torch.optim.SGD(model_fasterrcnn_resnet50.parameters(), lr=0.005,momentum=0.9, weight_decay=0.0005)
#print("----type(optimizer--aaaa---",type(optimizer)) ## FOOBAR -- this prints only Once 
## code dies somewhere after 1 EPOCH in the For Loop below -- >> for epoch in range(n_epochs):



n_epochs = 1 # 5
log = Report(n_epochs)
#print("--log--",log) ##<torch_snippets.torch_loader.Report object at 0x7f33132f2eb0>

#
try:
    #
    for epoch in range(n_epochs):
        print("----epoch__now---",epoch)
        len_train_loader  = len(train_loader) ## Book code == _n
        for ix, inputs in enumerate(train_loader): ## train_loader-->>  <class 'torch.utils.data.dataloader.DataLoader'>
            # print("----train_loader--->>-- type(inputs",type(inputs)) # tuple
            # print("----train_loader--->>-- inputs...\n",inputs) ## See own Log File >> /_book_4/logging_get_kaggle_data_5.log
            # print("----train_loader--->>-- inputs[0]...\n",inputs[0]) ## See own Log File >> /_book_4/logging_get_kaggle_data_5.log
            # print("----train_loader--->>-- inputs[1]...\n",inputs[1]) ## See own Log File >> /_book_4/logging_get_kaggle_data_5.log
            
            #print("----train_loader--->>-- inputs[2]...\n",inputs[2]) ## See own Log File >> /_book_4/logging_get_kaggle_data_5.log
            ### IndexError: tuple index out of range

            #if type(optimizer) optimizer = optimizer.zero_grad() ##TODO - ownExpCode_MemmoryIssues

            #optimizer = optimizer.zero_grad() ##TODO - ownExpCode_MemmoryIssues
            
            #print("-----------type(optimizer----bbb----",type(optimizer))

            loss, losses = train_batch(inputs, model_fasterrcnn_resnet50, optimizer)

            #print("-----------type(optimizer----cccc----",type(optimizer))

            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
            pos = (epoch + (ix+1)/len_train_loader) ## Book code == _n
            log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss.item(), 
                    trn_regr_loss=regr_loss.item(), trn_objectness_loss=loss_objectness.item(),
                    trn_rpn_box_reg_loss=loss_rpn_box_reg.item(), end='\r')

        if (epoch+1)%(n_epochs//5)==0: log.report_avgs(epoch+1)
except Exception as err_1:
    print("--Exception--err_1-----\n",err_1)
    pass     



try:
    for epoch in range(n_epochs):
        print("----epoch__now---",epoch)
        len_test_loader = len(test_loader)
        for ix,inputs in enumerate(test_loader):
            loss, losses = validate_batch(inputs, model_fasterrcnn_resnet50)

            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
            pos = (epoch + (ix+1)/len_test_loader) ## Book code == _n

            log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss.item(), 
                    val_regr_loss=regr_loss.item(), val_objectness_loss=loss_objectness.item(),
                    val_rpn_box_reg_loss=loss_rpn_box_reg.item(), end='\r')
        if (epoch+1)%(n_epochs//5)==0: log.report_avgs(epoch+1)
except Exception as err_2:
    print("--Exception--err_2-----\n",err_2)
    pass    

#     
log.plot_epochs(['trn_loss','val_loss'])
## In Jupyter -->> AttributeError: 'Report' object has no attribute 'val_loss'
#


from torchvision.ops import nms
def decode_output(output):
    'convert tensors to numpy arrays'
    print("----INIT---output['boxes']---\n",output['boxes'])
    print("----INIT---output['scores']---\n",output['scores'])
    print("----INIT---output['labels']---\n",output['labels'])

    bbs_original = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ## bbs_original
    ixs = nms(torch.tensor(bbs_original.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs_original, confs, labels = [tensor[ixs] for tensor in [bbs_original, confs, labels]]


    print("--ORIGINAL-type(confs-aaa--",type(confs))
    print("--ORIGINAL-confs-aaa.shape--",confs.shape)
    print("--ORIGINAL-confs-aaa--",confs)
    #
    ls_temp = []
    ls_temp_bbs = []
    ls_temp_labels = []

    for iter_k in range(len(confs)):
        if len(confs) > 0:
            #
            if confs[iter_k] > 0.5:
                print("-----confs[iter_k] -->> 0.5----",confs[iter_k])
                ls_temp.append(confs[iter_k])
                ls_temp = np.array(ls_temp)

                
                #ls_temp.append(confs[iter_k])
                bbs_optimized = output['boxes'].cpu().detach().numpy().astype(np.uint16)
                print("-----bbs_optimized[iter_k] -->> 0.5----",bbs_optimized[iter_k])
                print("--bbs_optimized-type(--",type(bbs_optimized))
                print("--bbs_optimized.shape--",bbs_optimized.shape)
                print("--bbs_optimized-aaa--",bbs_optimized)
                ls_temp_bbs.append(bbs_optimized[iter_k])
                ls_temp_bbs = np.array(ls_temp_bbs)
                        
                #ixs = nms(torch.tensor(bbs_optimized.astype(np.float32)), torch.tensor(confs), 0.05)
                ixs = nms(torch.tensor(ls_temp_bbs.astype(np.float32)), torch.tensor(ls_temp), 0.05)

                print("--ixs-type(--",type(ixs))
                print("--ixs.shape--",ixs.shape)
                print("--ixs-aaa--",ixs)
                print("-----ixs[iter_k] -->> 0.5----",ixs[iter_k])

                ##RuntimeError: boxes and scores should have same number of elements in dimension 0, got 43 and 12
                bbs_optimized, confs, labels = [tensor[ixs] for tensor in [bbs_optimized, confs, labels]]
                
                
                ls_temp_labels.append(labels)
                #print("-------confs -->> 0.5--bbb---",confs)
        
            else:
                pass
        else:
            pass
        
    if len(ixs) == 1:
        bbs_original, confs, labels = [np.array([tensor]) for tensor in [bbs_original, confs, labels]]
    
    confs = ls_temp
    print("-----len(confs",len(confs))
    print("-----len(ls_temp_bbs",len(ls_temp_bbs))
    print("-----len(ls_temp_bbs",ls_temp_bbs)
    #
    print("-----len(bbs_original",len(bbs_original.tolist()))
    print("-----bbs_original",bbs_original.tolist())

    #return bbs_original.tolist(), confs, labels.tolist()
    return ls_temp_bbs, confs, ls_temp_labels# labels.tolist()
#
# 
### FOOBAR --- Do separate EVAL
#  
model_fasterrcnn_resnet50.eval()
print("------here----beyond ---eval() -----------")



# for ix, (images, targets) in enumerate(test_loader):

#     if ix==3: break
#     images = [im for im in images]
#     outputs = model_fasterrcnn_resnet50(images)
#     print("-----outputs----",outputs)
#     for ix, output in enumerate(outputs):
#         bbs, confs, labels = decode_output(output)
#         info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
#         print("-----info----",info)


#         #show(images[ix].cpu().permute(1,2,0), bbs=bbs, texts=labels, sz=5)

### Jupyter Code - OK 
for ix, (images, targets) in enumerate(test_loader):

    if ix==15: break ## get 15 Images 
    images = [im for im in images]
    outputs = model_fasterrcnn_resnet50(images)
    #print("-----outputs----",outputs)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_output(output)
        print("---confs[1]----",confs[1])
        info = [f'{l}@{c:.2f}' for l,c in zip(labels, confs)]
        print("-----info----",info)
        for iter_k in range(len(confs)):
            if confs[iter_k] > 0.5:
                print("-------confs[iter_k] > 0.5----",confs[iter_k])
                show(images[ix].cpu().permute(1,2,0), bbs=bbs, texts=labels, sz=10)
            else:
                print("---Low Confidence ---",confs[iter_k])





# ---foobar----len(boxes 1
# --------target-- {'boxes': tensor([[ 67., 105.,  91., 138.]]), 'labels': tensor([1])}
# --ORIGINAL-type(confs-aaa-- <class 'numpy.ndarray'>
# --ORIGINAL-confs-aaa.shape-- (2,)
# --ORIGINAL-confs-aaa-- [0.44923633 0.0841101 ]
# -----len(confs 0
# -----len(ls_temp_bbs 0
# -----len(ls_temp_bbs []
# -----len(bbs_original 2
# -----bbs_original [[36, 81, 195, 148], [171, 121, 218, 142]]
# -----info---- []
# ----foobar----len(boxes 2
# --------target-- {'boxes': tensor([[150., 114., 159., 124.],
#         [185., 177., 223., 204.]]), 'labels': tensor([2, 2])}
# --ORIGINAL-type(confs-aaa-- <class 'numpy.ndarray'>
# --ORIGINAL-confs-aaa.shape-- (3,)
# --ORIGINAL-confs-aaa-- [0.4863741  0.0649074  0.05783449]
# -----len(confs 0
# -----len(ls_temp_bbs 0
# -----len(ls_temp_bbs []
# -----len(bbs_original 3
# -----bbs_original [[18, 99, 205, 215], [70, 189, 92, 218], [179, 113, 222, 136]]
# -----info---- []
# ----foobar----len(boxes 1
# --------target-- {'boxes': tensor([[  0.,  78.,  69., 150.]]), 'labels': tensor([1])}
# --ORIGINAL-type(confs-aaa-- <class 'numpy.float32'>
# --ORIGINAL-confs-aaa.shape-- ()
# --ORIGINAL-confs-aaa-- 0.47315246

# ------------------------------------



# ----foobar----len(boxes 1
# --------target-- {'boxes': tensor([[ 67., 105.,  91., 138.]]), 'labels': tensor([1])}
# ----INIT---output['boxes'] tensor([[ 68.5039, 106.7684,  91.9937, 139.0323],
#         [ 37.5546,  74.2412, 138.4572, 137.6332],
#         [ 17.6182,  68.3099, 197.7660, 215.2834],
#         [  9.1009,  66.2336, 213.0801, 213.4284],
#         [  6.2085,   2.5430, 138.9150, 147.4917],
#         [ 67.5203, 103.4926,  92.4275, 139.8420],
#         [ 36.6589,  36.5740, 105.9258, 132.3919]], device='cuda:0',
#        grad_fn=<StackBackward>)
# ----INIT---output['scores'] tensor([0.5440, 0.4333, 0.3645, 0.3227, 0.1550, 0.1479, 0.0786],
#        device='cuda:0', grad_fn=<IndexBackward>)
# --ORIGINAL-type(confs-aaa-- <class 'numpy.ndarray'>
# --ORIGINAL-confs-aaa.shape-- (2,)
# --ORIGINAL-confs-aaa-- [0.5439906  0.36449778]
# -----confs[iter_k] -->> 0.5---- 0.5439906
# --bbs_optimized-type(-- <class 'numpy.ndarray'>
# --bbs_optimized.shape-- (7, 4)
# --bbs_optimized-aaa-- [[ 68 106  91 139]
#  [ 37  74 138 137]
#  [ 17  68 197 215]
#  [  9  66 213 213]
#  [  6   2 138 147]
#  [ 67 103  92 139]
#  [ 36  36 105 132]]
