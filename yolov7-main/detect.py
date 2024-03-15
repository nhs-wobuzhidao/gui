import argparse
import time
from pathlib import Path
#make a new text file and write the number of steps
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    framenum = 0
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    def overlappingArea(l1, r1, l2, r2):
            x = 0
            y = 1
 
            # Area of 1st Rectangle
            area1 = abs(l1[x] - r1[x]) * abs(l1[y] - r1[y])
            #print("This is area1", area1)
            # Area of 2nd Rectangle
            area2 = abs(l2[x] - r2[x]) * abs(l2[y] - r2[y])
            #print("This is area2", area2)
            ''' Length of intersecting part i.e  
            start from max(l1[x], l2[x]) of  
            x-coordinate and end at min(r1[x], 
            r2[x]) x-coordinate by subtracting  
            start from end we get required  
            lengths '''
            x_dist = (min(r1[x], r2[x]) -
              max(l1[x], l2[x]))
 
            y_dist = (min(r1[y], r2[y]) -
              max(l1[y], l2[y]))
            areaI = 0
            if x_dist > 0 and y_dist > 0:
                areaI = x_dist * y_dist
            return areaI

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    l_vid = []
    r_vid = []
    print("yoyoyngltr")
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                l = []
                r = []
                i = 0
                dict = {}
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        #define the center and length of bbox
                        x_center = xywh[0]
                        y_center = xywh[1]
                        x_length = xywh[2]
                        y_length = xywh[3]
                        #add to list of upper right coordinates of bbox
                        r.append([x_center+x_length/2, y_center+y_length/2])
                        #add to list of lower left coordinates of bbox
                        l.append([x_center-x_length/2, y_center-y_length/2])
                        #define label as class
                        label = f'{names[int(cls)]}'
                        #create a dictionary for each frame that organizes those coordinates by class
                        keys = list(dict.keys())
                        if (keys.count(label)==1):
                            temp = dict.get(label)
                            temp.append([[x_center+x_length/2, y_center+y_length/2],[x_center-x_length/2, y_center-y_length/2]])
                        else:
                            temp = [[[x_center+x_length/2, y_center+y_length/2],[x_center-x_length/2, y_center-y_length/2]]]
                        dict.update({label:temp})
                        #print(overlappingArea(l[x], r[x], l[x+1], r[x+1]))
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    if save_img or view_img:  # Add bbox to image
                        #if (keys.count(label)==1):
                        #   temp = dict.get(label)
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #print(xyxy[0])
                        #print(overlappingArea(l[x], r[x], l[x+1], r[x+1]))
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    i += 1
                l_vid.append(l)
                r_vid.append(r)
                print("dict")
                print(dict)
                which = 'person'
                if (which):
                    if (keys.count(which)==1):
                        f = open("output.txt", "a")
                        f.write("***")
                        f.close()
                        for i in range(len(dict.get(which))):
                            for j in range(i+1, len(dict.get(which))):
                                #print('pair')
                                p = dict.get(which)[i]
                                q = dict.get(which)[j]
                                #print(p,'\n')
                                #print(q, '\n')
                                f = open("output.txt", "a")
                                f.write(str(overlappingArea(p[1], p[0], q[1], q[0])) + "\n")
                                f.close()
                                print('Area ', overlappingArea(p[1], p[0], q[1], q[0]))
            # Print time (inference + NMS)
            #print (l[0], r[0])
            #print (l[1], r[1])
            #for x in range (len(l)-1):
            #    print(overlappingArea(l[x], r[x], l[x+1], r[x+1]))
            #    print('Area ', overlappingArea(l[1], r[1], l[1], r[1]))
            #    print('Percentage intersection ', (overlappingArea(l[x], r[x], l[x+1], r[x+1])/overlappingArea(l[x], r[x], l[x], r[x])))
            
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            print("wow")
            print(framenum)
            framenum = framenum + 1
            print(framenum)
            print("bow")
            g = open("framenum.txt", "w")
            g.write(str(framenum))
            g.close()
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        #print(l_vid)
        #print(r_vid)

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
