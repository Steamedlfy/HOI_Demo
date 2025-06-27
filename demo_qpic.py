# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
import collections
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets_qpic.hico import make_hico_transforms
import util_qpic.misc as utils
from datasets_qpic import build_dataset, get_coco_api_from_dataset
from engine_qpic import evaluate, train_one_epoch, evaluate_hoi
from models_qpic import build_model
from PIL import Image
import datasets_qpic.transforms as T

# HICO 117类动词名称（完整列表）
HICO_VERBS = [
    "adjust", "assemble", "block", "blow", "board", "break", "brush_with", "buy", "carry", "catch",
    "chase", "check", "clean", "control", "cook", "cut", "cut_with", "direct", "drag", "dribble",
    "drink_with", "drive", "dry", "eat", "eat_at", "exit", "feed", "fill", "flip", "flush",
    "fly", "greet", "grind", "groom", "herd", "hit", "hold", "hop_on", "hose", "hug",
    "hunt", "inspect", "install", "jump", "kick", "kiss", "lasso", "launch", "lick", "lie_on",
    "lift", "light", "load", "lose", "make", "milk", "move", "open", "operate", "pack",
    "paint", "park", "pay", "peel", "pet", "pick", "pick_up", "point", "pour", "pull",
    "push", "race", "read", "release", "repair", "ride", "row", "run", "sail", "scratch",
    "serve", "set", "shear", "sign", "sip", "sit_at", "sit_on", "slide", "smell", "spin",
    "squeeze", "stab", "stand_on", "stand_under", "stick", "stir", "stop_at", "straddle", "swing", "tag",
    "talk_on", "teach", "text_on", "throw", "tie", "toast", "turn", "type_on", "walk", "wash",
    "watch", "wave", "wear", "wield", "zip"
]
OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def find_similar_pair_key(best_hoi, subj_box, obj_box, iou_thresh=0.7):
    for (s_box, o_box) in best_hoi:
        if compute_iou(s_box, subj_box) > iou_thresh and compute_iou(o_box, obj_box) > iou_thresh:
            return (s_box, o_box)  # 已存在近似交互对
    return None

def is_similar_box(box1, box2, iou_thresh=0.9):
    """判断两个框是否相似（IOU超过阈值）"""
    return compute_iou(box1, box2) > iou_thresh

def filter_and_visualize_hoi(frame_bgr, result, score_thresh=0.25, iou_thresh=0.9):
    labels = result['labels'].numpy()
    boxes = result['boxes'].numpy().astype(int)
    verb_scores = result['verb_scores'].numpy()  # [M, K]
    sub_ids = result['sub_ids'].numpy()  # [M]
    obj_ids = result['obj_ids'].numpy()  # [M]
    img = frame_bgr.copy()

    # 1. 收集所有HOI对
    hoi_list = []
    for i in range(len(verb_scores)):
        subj = sub_ids[i]
        obj = obj_ids[i]
        subj_box = tuple(boxes[subj])
        obj_box = tuple(boxes[obj])
        for verb_id, score in enumerate(verb_scores[i]):
            if score < score_thresh:
                continue
            verb = HICO_VERBS[verb_id] if verb_id < len(HICO_VERBS) else str(verb_id)
            subj_cls = OBJECTS[labels[subj]] if labels[subj] < len(OBJECTS) else str(labels[subj])
            obj_cls = OBJECTS[labels[obj]] if labels[obj] < len(OBJECTS) else str(labels[obj])
            hoi_list.append({
                'h_box': subj_box,
                'o_box': obj_box,
                'i_name': verb,
                'i_score': float(score),
                'h_name': subj_cls,
                'o_name': obj_cls,
            })

    # 2. 合并每个人/物体的动作
    human_actions = collections.defaultdict(list)
    for hoi in hoi_list:
        human_actions[hoi['h_box']].append(f"{hoi['i_name']} {hoi['o_name']}")


    # 3. 画人框和动作（使用模糊匹配避免重复绘制，但保留所有动作）
    drawn_human_boxes = {}  # 改为字典，键为相似人物框的代表框，值为所有动作

    for h_box, actions in human_actions.items():
        # 检查是否与已绘制的框相似
        similar_key = None
        for drawn_box in drawn_human_boxes.keys():
            if is_similar_box(h_box, drawn_box, iou_thresh):
                similar_key = drawn_box
                break
        
        if similar_key is not None:
            # 已有相似框，合并动作但不重复画框
            drawn_human_boxes[similar_key].extend(actions)
        else:
            # 新的人物框，记录并绘制
            drawn_human_boxes[h_box] = actions.copy()
            
            # 绘制人物框
            x1, y1, x2, y2 = h_box
            color_human = (255, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_human, 2)

# 统一绘制所有人物的动作（此时已收集所有相似人物的动作）
    for h_box, all_actions in drawn_human_boxes.items():
        x1, y1, x2, y2 = h_box
        
        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        line_height = 32  # 每行高度
        margin = 6  # 边距
        
        # 获取去重后的动作列表
        unique_actions = list(set(all_actions))
        
        # 垂直排列显示动作（限制显示数量避免超出框外）
        displayed_actions = []
        current_y = y1 + margin
        for action in unique_actions:
            (text_width, text_height), _ = cv2.getTextSize(action, font, font_scale, thickness)
            if current_y + text_height > y2 - margin:
                if not displayed_actions:
                    displayed_actions.append("...")
                break
            displayed_actions.append(action)
            current_y += text_height + line_height
        
        for i, action in enumerate(displayed_actions):
            text_x = x1 + margin
            text_y = y1 + margin + +16+i * (line_height)
            cv2.putText(img, action, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    # 4. 画物体框（使用模糊匹配避免重复绘制）
    drawn_object_boxes = []
    for hoi in hoi_list:
        o_box = hoi['o_box']
        is_duplicate = False
        for drawn_box in drawn_object_boxes:
            if is_similar_box(o_box, drawn_box, iou_thresh):
                is_duplicate = True
                break
        if is_duplicate:
            continue
        drawn_object_boxes.append(o_box)
        x1, y1, x2, y2 = o_box
        color_object = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_object, 2)

    # 5. 画人-物体连线（确保每对只画一次）
    drawn_pairs = set()
    for hoi in hoi_list:
        h_box = hoi['h_box']
        o_box = hoi['o_box']
        pair = (h_box, o_box)
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        hx1, hy1, hx2, hy2 = h_box
        ox1, oy1, ox2, oy2 = o_box
        cx_h = int((hx1 + hx2) / 2)
        cy_h = int((hy1 + hy2) / 2)
        cx_o = int((ox1 + ox2) / 2)
        cy_o = int((oy1 + oy2) / 2)
        cv2.line(img, (cx_h, cy_h), (cx_o, cy_o), (0, 0, 255), 2)
        cv2.circle(img, (cx_h, cy_h), 4, (0, 0, 255), -1)
        cv2.circle(img, (cx_o, cy_o), 4, (0, 0, 255), -1)

    return img

def load_model(path,args):
    # 初始化分布式训练模式（用于多卡训练或并行计算）
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))  # 输出当前git commit哈希
    normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = T.Compose([
    T.RandomResize([800], max_size=1333),
    normalize,
    ])
    # 如果开启冻结权重（用于分割），必须开启mask
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)  # 打印全部参数配置
    # 设置设备（GPU或CPU）
    device = torch.device(args.device)
    # 设置随机种子，保证实验可复现
    seed = args.seed + utils.get_rank()  # 每个进程不同种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 构建模型、损失函数和后处理函数
    model, criterion, postprocessors = build_model(args)
    model.to(device)  # 模型移至设备上
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model,postprocessors
def pretrain(frame):
    # OpenCV BGR 转 RGB，再转 PIL 处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # 定义 transforms
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 预处理
    img_tensor, _ = transforms(pil_image, None)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor
def main(args):
    # 初始化分布式训练模式（用于多卡训练或并行计算）
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))  # 输出当前git commit哈希
    normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = T.Compose([
    T.RandomResize([800], max_size=1333),
    normalize,
    ])
    # 如果开启冻结权重（用于分割），必须开启mask
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)  # 打印全部参数配置

    # 设置设备（GPU或CPU）
    device = torch.device(args.device)

    # 设置随机种子，保证实验可复现
    seed = args.seed + utils.get_rank()  # 每个进程不同种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型、损失函数和后处理函数
    model, criterion, postprocessors = build_model(args)
    model.to(device)  # 模型移至设备上

    model_without_ddp = model  # 单GPU/CPU情况直接使用
    if args.distributed:
        # 分布式训练封装
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module  # 取原始模型引用

    # 统计模型参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 如果指定了 frozen_weights，则只加载 DETR 权重（用于分割任务）
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 若指定了 resume，加载之前训练的 checkpoint 继续训练或验证
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 如果是继续训练，则恢复优化器和调度器状态
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # args.start_epoch = checkpoint['epoch'] + 1

    # 若指定了 pretrained，则加载预训练模型（用于评估或迁移训练）
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "无法打开摄像头"

    print("按 'q' 退出摄像头推理")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # OpenCV BGR 转 RGB，再转 PIL 处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
    
        # 预处理
        img_tensor, _ = transforms(pil_image, None)
        img_tensor = img_tensor.unsqueeze(0).to(device)
    
        # 模型推理
        with torch.no_grad():
            start_time = time.time()
            outputs = model(img_tensor)
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.2f} seconds")
            orig_size = torch.tensor([[frame.shape[0], frame.shape[1]]]).to(device)
            results = postprocessors['hoi'](outputs, orig_size)
            # print(results)  # 打印结果以调试
        # 可视化（OpenCV）
        vis_frame = filter_and_visualize_hoi(frame.copy(), results[0])  # 可添加 verb_names 映射
    
        # 展示结果
        cv2.imshow("HOI Detection", vis_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
def create_default_args():
    """创建默认参数"""
    args = argparse.Namespace()

    # 训练超参数
    args.lr = 1e-4
    args.lr_backbone = 1e-5
    args.batch_size = 1
    args.weight_decay = 1e-4
    args.epochs = 250
    args.lr_drop = 200
    args.clip_max_norm = 0.1

    # Transformer 结构
    args.position_embedding = 'sine'
    args.enc_layers = 6
    args.dec_layers = 6
    args.dim_feedforward = 2048
    args.hidden_dim = 256
    args.dropout = 0.1
    args.nheads = 8
    args.num_queries = 100
    args.pre_norm = False

    # 辅助损失
    args.aux_loss = True

    # 匹配损失系数
    args.set_cost_class = 1
    args.set_cost_bbox = 5
    args.set_cost_giou = 2
    args.set_cost_obj_class = 1
    args.set_cost_verb_class = 1

    # 损失权重
    args.dice_loss_coef = 1
    args.bbox_loss_coef = 5
    args.giou_loss_coef = 2
    args.mask_loss_coef = 1
    args.obj_loss_coef = 1
    args.verb_loss_coef = 1
    args.eos_coef = 0.02

    # 模型参数
    args.backbone = 'resnet50'
    args.dilation = False
    args.frozen_weights = None
    args.masks = False
    args.hoi = True
    args.num_obj_classes = 80
    args.num_verb_classes = 117
    args.pretrained = ''
    args.subject_category_id = 0
    args.verb_loss_type = 'focal'

    # 路径参数
    args.dataset_file = 'coco'
    args.coco_path = ''
    args.coco_panoptic_path = ''
    args.remove_difficult = False
    args.hoi_path = 'G:\模式识别课设数据集\HICO-DET\hico'
    args.output_dir = ''

    # 系统参数
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 42
    args.resume = ''
    args.start_epoch = 0
    args.eval = False
    args.num_workers = 0

    # 分布式
    args.world_size = 1
    args.dist_url = 'env://'

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
