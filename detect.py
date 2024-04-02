# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


# @torch.no_grad()  # 该标注使得方法中所有计算得出的tensor的requires_grad都自动设置为False，也就是说不进行梯度的计算(当然也就没办法反向传播了)， 节约显存和算
@smart_inference_mode()  # 用于自动切换模型的推理模式，如果是FP16模型，则自动切换为FP16推理模式，否则切换为FP32推理模式，这样可以避免模型推理时出现类型不匹配的错误
def run(weights=ROOT / "yolov5s.pt",
        # model.pt path(s) 事先训练完成的权重文件，比如yolov5s.pt,默认 weights/，假如使用官方训练好的文件（比如yolov5s）,则会自动下载
        source=ROOT / "data/images",
        # file/dir/URL/glob, 0 for webcam 预测时的输入数据，可以是文件/路径/URL/glob, 输入是0的话调用摄像头作为输入，默认data/images/
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path, data文件路径，包括类别/图片/标签等信息
        imgsz=(640, 640),  # inference size (pixels)  预测时的放缩后图片大小(因为YOLO算法需要预先放缩图片), 两个值分别是height, width。默认640*640
        conf_thres=0.25,  # confidence threshold 置信度阈值, 高于此值的bounding_box才会被保留。默认0.25，用在nms中
        iou_thres=0.45,  # NMS IOU threshold IOU阈值,高于此值的bounding_box才会被保留。默认0.45，用在nms中
        max_det=1000,  # maximum detections per image 一张图片上检测的最大目标数量，用在nms中
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu 所使用的GPU编号，如果使用CPU就写cpu
        view_img=False,  # show results 是否展示预测之后的图片或视频，默认False
        save_txt=False,
        # save results to *.txt 是否将预测的框坐标以txt文件形式保存, 默认False, 使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels 是否将结果中的置信度保存在txt文件中，默认False
        save_crop=False,
        # save cropped prediction boxes 是否保存裁剪后的预测框，默认为False, 使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
        nosave=False,  # do not save images/videos 不保存图片、视频, 要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
        classes=None,  # filter by class: --class 0, or --class 0 2 3 过滤指定类的预测结果
        agnostic_nms=False,  # class-agnostic NMS 进行NMS去除不同类别之间的框, 默认False
        augment=False,  # augmented inference TTA测试时增强/多尺度预测，可以提分
        visualize=False,  # visualize features 是否可视化网络层输出特征
        update=False,  # update all models 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        project=ROOT / "runs/detect",  # save results to project/name 预测结果保存的路径
        name="exp",  # save results to project/name 结果保存文件夹的命名前缀
        exist_ok=False,  # existing project/name ok, do not increment True: 推理结果覆盖之前的结果 False: 推理结果新建文件夹保存,文件夹名递增
        line_thickness=3,  # bounding box thickness (pixels) 绘制Bounding_box的线宽度
        hide_labels=False,  # hide labels 若为True: 隐藏标签
        hide_conf=False,  # hide confidences  若为True: 隐藏置信度
        half=False,  # use FP16 half-precision inference 是否使用半精度推理（节约显存）
        dnn=False,  # use OpenCV DNN for ONNX inference 是否使用OpenCV DNN预测
        vid_stride=1,  # video frame-rate stride 视频帧采样间隔，默认为1，即每一帧都进行检测
        ):
    '''====================================================1.载入参数================================================================='''

    source = str(source)  # 将source转换为字符串，source为输入的图片、视频、摄像头等
    save_img = not nosave and not source.endswith(".txt")  # 判断是否保存图片，如果nosave为False，且source不是txt文件，则保存图片

    # 判断source是否是文件.Path(source)使用source创建一个Path对象，用于获取输入源信息，suffix获取文件扩展名：.jpg,.mp4等，suffix[1:]获取文件后缀，
    # 判断后缀是否在IMG_FORMATS和VID_FORMATS中，如果是，则is_file为True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    # source.isnumeric()判断source是否是数字，source.endswith('.streams')判断source是否以.streams结尾，(is_url and not is_file)判断source是否是url，
    # 且不是文件，上述三个条件有一个为True，则webcam为True
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)

    screenshot = source.lower().startswith("screen")  # 判断source是否是截图，如果是，则screenshot为True
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories, 创建保存结果的文件夹
    # increment run，增加文件或目录路径，即运行/exp——>运行/exp{sep}2，运行/exp{sep}3，…等。exist_ok为True时，如果文件夹已存在，则不会报错
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model，初始化模型
    device = select_device(device)
    # 加载模型，DetectMultiBackend()函数用于加载模型，weights为模型路径，device为设备，dnn为是否使用opencv dnn，data为数据集，fp16为是否使用fp16推理
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size,验证图像大小是每个维度的stride=32的倍数

    # Dataloader,初始化数据集
    bs = 1  # batch_size
    if webcam:  # 如果source是摄像头，则创建LoadStreams()对象
        view_img = check_imshow(warn=True)
        # 创建LoadStreams()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:  # 如果source是截图，则创建LoadScreenshots()对象
        # 创建LoadScreenshots()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 创建LoadImages()对象，直接加载图片，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化vid_path和vid_writer，vid_path为视频路径，vid_writer为视频写入对象

    '''====================================================2.开始推理================================================================='''
    # warmup，预热，用于提前加载模型，加快推理速度，imgsz为图像大小，如果pt为True或者model.triton为True，则bs=1，否则bs为数据集的长度。
    # 3为通道数，*imgsz为图像大小，即(1,3,640,640)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    # 初始化seen，windows，dt，seen为已检测的图片数量，windows为空列表，dt为时间统计对象
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 将图片转换为tensor，并放到模型的设备上，pytorch模型的输入必须是tensor
            im = im.half() if model.fp16 else im.float()  # 如果模型使用fp16推理，则将图片转换为fp16，否则转换为fp32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:  # 如果图片的维度为3，则添加batch维度
                # 在前面添加batch维度，即将图片的维度从3维转换为4维，即(3,640,640)转换为(1,3,640,640)，pytorch模型的输入必须是4维的
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            # 如果visualize为True，则创建visualize文件夹，否则为False
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                # 推理，model()函数用于推理，im为输入图片，augment为是否使用数据增强，
                # visualize为是否可视化,输出pred为一个列表，形状为（n,6）,n代表预测框的数量，6代表预测框的坐标和置信度，类别
                pred = model(im, augment=augment, visualize=visualize)

        # NMS,非极大值抑制，用于去除重复的预测框
        with dt[2]:
            # non_max_suppression()函数用于NMS，
            # pred为输入的预测框，conf_thres为置信度阈值，iou_thres为iou阈值，classes为类别，agnostic_nms为是否使用类别无关的NMS，max_det为最大检测框数量
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        '''====================================================3.处理预测结果================================================================='''
        # Process predictions
        for i, det in enumerate(pred):  # per image，历每张图片,enumerate()函数将pred转换为索引和值的形式，i为索引，det为对应的元素，即每个物体的预测框
            seen += 1
            if webcam:  # batch_size >= 1，如果是摄像头，则获取视频帧率
                # path[i]为路径列表，ims[i].copy()为将输入图像的副本存储在im0变量中，dataset.count为当前输入图像的帧数
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "  # 在打印输出中添加当前处理的图像索引号i，方便调试和查看结果。在此处，如果是摄像头模式，i表示当前批次中第i张图像；否则，i始终为0，因为处理的只有一张图像。
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)  # 如果不是摄像头，frame为0

            p = Path(p)  # 将路径转换为Path对象
            save_path = str(save_dir / p.name)  # im.jpg，保存图片的路径，save_dir为保存图片的文件夹，p.name为图片名称
            # im.txt，保存预测框的路径，save_dir为保存图片的文件夹，p.stem为图片名称，dataset.mode为数据集的模式，如果是image，则为图片，否则为视频
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop，如果save_crop为True，则将im0复制一份，否则为im0

            # 创建Annotator对象，用于在图片上绘制预测框和标签,im0为输入图片，line_width为线宽，example为标签
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将预测框的坐标从归一化坐标转换为原始坐标,im.shape[2:]为图片的宽和高，det[:, :4]为预测框的坐标，im0.shape为图片的宽和高
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():  # 遍历每个类别,unique()用于获取检测结果中不同类别是数量
                    n = (det[:, 5] == c).sum()  # n为每个类别的预测框的数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # s为每个类别的预测框的数量和类别

                # Write results
                # 遍历每个预测框,xyxy为预测框的坐标，conf为置信度，cls为类别,reversed()函数用于将列表反转，
                # *是一个扩展语法，*xyxy表示将xyxy中的元素分别赋值给x1,y1,x2,y2
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        # normalized xywh,将预测框的坐标从原始坐标转换为归一化坐标
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # label format,如果save_conf为True，则将置信度也写入txt文件中
                        with open(f"{txt_path}.txt", "a") as f:  # 打开txt文件,'a'表示追加
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image,如果save_img为True，则将预测框和标签绘制在图片上
                        c = int(cls)  # integer class
                        # 如果hide_labels为True，则不显示标签，否则显示标签，如果hide_conf为True，则不显示置信度，否则显示置信度
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))  # 绘制预测框和标签
                    if save_crop:  # 如果save_crop为True，则保存裁剪的图片
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results,在图片上绘制预测框和标签展示
            im0 = annotator.result()
            if view_img:  # 如果view_img为True，则展示图片
                if platform.system() == "Linux" and p not in windows:  # 如果系统为Linux，且p不在windows中
                    windows.append(p)  # 将p添加到windows中
                    # allow window resize (Linux),允许窗口调整大小,WINDOW_NORMAL表示用户可以调整窗口大小，WINDOW_KEEPRATIO表示窗口大小不变
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小，使其与图片大小一致
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:  # 如果save_img为True，则保存图片
                if dataset.mode == "image":  # 如果数据集模式为image
                    cv2.imwrite(save_path, im0)  # 保存图片
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果vid_writer[i]是cv2.VideoWriter类型
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:  # 如果save_txt为True，则打印保存的标签数量
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


'''=======================二、设置main函数==================================='''


def main(opt):
    # 检查环境/打印参数,主要是requrement.txt的包是否安装，用彩色显示设置的参数
    check_requirements(exclude=('tensorboard', 'thop'))
    # 执行run()函数
    run(**vars(opt))


# 命令使用
# python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source  data/images/fishman.jpg # webcam
if __name__ == "__main__":
    opt = parse_opt()  # 解析参数
    main(opt)  # 执行主函数
