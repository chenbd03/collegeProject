# -*- coding: UTF-8 -*-
import time

from libs.models import *
from libs.utils.datasets import *
from libs.utils.utils import *
import os
import torch

def detect(
        images,  # input folder,
        img_name,
        cfg='./libs/cfg/yolov3.cfg',
        data_cfg='./libs/data/head.data',
        weights='./libs/weights/latest.pt',
        img_size=416,
        conf_thres=0.6,
        nms_thres=0.5,
):
    with torch.no_grad():
        t = time.time()
        device = torch_utils.select_device()
        torch.backends.cudnn.benchmark = False  # set False for reproducible results
        output = os.getcwd() + os.sep + "output" + os.sep
        if not os.path.exists(output):
            os.mkdir(output)

        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Set Dataloader
        dataloader = UnowLoadImages(images, img_size=img_size)

        # Get classes and colors
        classes = load_classes(os.path.join('./libs', parse_data_cfg(data_cfg)['names']))

        # Get detections
        img = torch.from_numpy(dataloader.img).unsqueeze(0).to(device)
        #print("img111:{}".format(img))
        pred, _ = model(img)
        #print("pred:{}".format(pred))
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        #print("det:{}".format(det))
        result = {'head_num': 0, 'face_list': [], 'cost_time': 0}
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], dataloader.img0.shape).round()

            # Print results to screen
            # print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                head_num = (det[:, -1] == c).sum()
            face_list = []
            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                plot_one_box(xyxy, dataloader.img0, color=(0, 0, 255),label="score:"+str('%.3f'%conf.item()))
                cv2.imwrite(output+img_name, dataloader.img0)
                face_info = {'head_score': float('%.3f'%conf.item()),
                            'location': {'left': xyxy[0].item(), 'top': xyxy[1].item(), 'width': xyxy[2].item() - xyxy[0].item(), 'height': xyxy[3].item() - xyxy[1].item()}}
                face_list.append(face_info)
            result = {'head_num': head_num.item(), 'face_list': face_list, 'cost_time': '%d' % round((time.time() - t)*1000)}
        else:
            result['cost_time'] = '%d' % round((time.time() - t)*1000)
        return result