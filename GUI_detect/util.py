
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import os
import shutil
import time
import cv2
import sys
import itertools
import torch
from yolov5.utils.torch_utils import select_device
sys.path.insert(0, './yolov5')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, total_count,identities=None, offset=(0,0)):
    label_n = "total people:{}".format(total_count)
    cv2.putText(img, label_n, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2,[0,255,0], 2)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        x = (x1 + x2) // 2   # 圆心坐标x
        y = (y1 + y2) // 2   # 圆心坐标y
        pts = np.array([[x-10, y-10], [x-10, y+10],  [x, y+15], [x+10, y+10], [x+10, y-10], [x, y-15]],  np.int32) 
        pts = pts.reshape((-1, 1, 2)) 
        cv2.circle(img, (x, y), 5,(212, 255, 127), -1)
        cv2.polylines(img, [pts], True, (30,144, 255), 2)
        
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] - 10, y1 + t_size[1] - 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] - 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


def distancing(people_coords, img, dist_thres_lim=(200,400)):
    # Plot lines connecting people
    already_red = dict() # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers = []
    for i in people_coords:
        centers.append(((int(i[2])+int(i[0]))//2,(int(i[3])+int(i[1]))//2))
    for j in centers:
        already_red[j] = 0
    x_combs = list(itertools.combinations(people_coords,2))
    radius = 4
    thickness = 2
    low_risk_count = 0
    high_risk_count = 0
    for x in x_combs:
        xyxy1, xyxy2 = x[0],x[1]
        cntr1 = ((int(xyxy1[2])+int(xyxy1[0]))//2,(int(xyxy1[3])+int(xyxy1[1]))//2)
        cntr2 = ((int(xyxy2[2])+int(xyxy2[0]))//2,(int(xyxy2[3])+int(xyxy2[1]))//2)
        dist = ((cntr2[0]-cntr1[0])**2 + (cntr2[1]-cntr1[1])**2)**0.5

        if dist > dist_thres_lim[0] and dist < dist_thres_lim[1]:
            color = (255, 0, 0)
            label = "Low Risk "
            cv2.line(img, cntr1, cntr2, color, thickness)
            low_risk_count += 1
            if already_red[cntr1] == 0:
                cv2.circle(img, cntr1, radius, color, -1)
            if already_red[cntr2] == 0:
                cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                cntr = ((int(xy[2])+int(xy[0]))//2,(int(xy[3])+int(xy[1]))//2)
                if already_red[cntr] == 0:
                    c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
                    c2 = c1[0] + t_size[0] // 2, c1[1] - t_size[1] + 6
                    cv2.rectangle(img, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    

        elif dist < dist_thres_lim[0]:
            color = (0, 0, 255)
            label = "High Risk"
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            high_risk_count += 1
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
                c2 = c1[0] + t_size[0] // 2, c1[1] - t_size[1] + 6
                cv2.rectangle(img, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                
    return low_risk_count, high_risk_count


def draw_description(img, total_count, low_risk_count, high_risk_count):
    color = [0, 255, 0]
    label_n = "total people:{}".format(total_count)
    label_low = "low risk line count:{}".format(low_risk_count)
    label_high = "high risk line count:{}".format(high_risk_count)
    cv2.putText(img, label_n, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2,color, 2)
    cv2.putText(img, label_low, (20, 65), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(img, label_high, (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


def init_model(weights,half,img_size):
    device = select_device("0")
    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()
    if half:
       model.half()  # to FP16

    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    return model, device
