#encoding=utf-8
from __future__ import division

from yolo_models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import argparse
import tqdm

import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms
from torch.autograd import Variable
# import torch.optim as optim

data_path_base = './data/'

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="final_models/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
# parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["valid"]
num_classes = int(data_config["classes"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval()

#返回指定下标的图像的人数
def returnNum(category,index):
    # Get dataloader
    test_path=data_path_base+category+'/'+str(index)+'.png'
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    #print("Compute mAP...")

    all_detections = []
    all_annotations = []

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        for output, annotations in zip(outputs, targets):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= opt.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    #average_precisions = {}
    #for label in range(num_classes):
    for label in range(1):
        # true_positives = []
        # scores = []
        # num_annotations = 0

        #for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
        for i in tqdm.tqdm(range(len(all_annotations)), desc="Computing AP for class '{}'".format(label)):
            detections = all_detections[i][label]
            #annotations = all_annotations[i][label]
            return detections.shape[0]
            #print(detections.shape[0])
