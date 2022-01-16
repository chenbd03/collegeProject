from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
import itertools
import numpy as np
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


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
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


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    print("webcam:{}".format(webcam))
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        print("1")
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        print("2")
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    print("1111")
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    low_risk_count, high_risk_count = distancing(bbox_xyxy, im0)
                    draw_description(im0, n, low_risk_count, high_risk_count)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:  
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                # cv2.resizeWindow(p, 640, 480)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                # print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    # print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # parser.add_argument('--source', type=str, default='rtsp://admin:Abcd1234@192.168.3.22:554/cam/realmonitor?channel=1&subtype=0', help='source')
    parser.add_argument('--source', type=str, default='./inference/video/people.jpg', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
