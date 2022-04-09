import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##
##RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or 
# input should be a MKLDNN tensor and weight is a dense tensor
##

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ls_img_tensor = [] ## FOOBAR_ownCode

    #print("-------train_one_epoch-----INHERE-------aaa----")
  
    model.train()
    print("-------train_one_epoch---model.train()--INHERE-----bbbb------")

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        print("-------train_one_epoch-----INHERE-----------")
        # warmup_factor = 1. / 1000
        # warmup_iters = min(1000, len(data_loader) - 1)
        # lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        from PIL import Image
        import torchvision.transforms as transforms
        # image to a Torch tensor --- ## FOOBAR_ownCode
        #transform = transforms.Compose([transforms.PILToTensor()])
        transform = transforms.Compose([transforms.ToTensor()])
        for image in images: ## FOOBAR_ownCode
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            #image.to(device)
            img_tensor = transform(image) ## FOOBAR_ownCode
            print("--train_one_epoch--type(img_tensor----aaa---\n",type(img_tensor))
            cuda_tensor = img_tensor.to(device)
            print("--train_one_epoch--type(cuda_tensor----bbb---\n",type(cuda_tensor))
            print(img_tensor.dtype, type(img_tensor), img_tensor.type())
            print(cuda_tensor.dtype, type(cuda_tensor), cuda_tensor.type())

            ls_img_tensor.append(cuda_tensor) ## FOOBAR_ownCode


        images = ls_img_tensor
        #images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print("--train_one_epoch--type(targets-------\n",type(targets))
        #print("--train_one_epoch--targets-------\n",targets)

        loss_dict = model(images, targets)
        print("--train_one_epoch--type(loss_dict-------\n",type(loss_dict))

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        print("--train_one_epoch-----loss_value-------\n",loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
