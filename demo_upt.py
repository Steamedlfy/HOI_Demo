"""
Visualise detected human-object interactions in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import time
import os
import torch
import warnings
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import DataFactory
from upt import build_detector

warnings.filterwarnings("ignore")

def draw_box_pairs_cv(image_cv, boxes_1, boxes_2, width=2):
    """
    Draw bounding box pairs onto an OpenCV (BGR) image.
    - 人框(蓝色)
    - 物体框(绿色)
    - 框心连线(红色)
    
    Args:
        image_cv: OpenCV BGR 图像 (np.ndarray)
        boxes_1: (N, 4) tensor/list/array，格式 (x1, y1, x2, y2)，人框
        boxes_2: (N, 4) tensor/list/array，格式 (x1, y1, x2, y2)，物体框
        width: 框线宽度
    """
    # if isinstance(boxes_1, (torch.Tensor, list)):
    #     boxes_1 = np.asarray(boxes_1)
    # elif not isinstance(boxes_1, np.ndarray):
    #     raise TypeError("boxes_1 必须是 torch.Tensor, np.ndarray 或 list")
    # if isinstance(boxes_2, (torch.Tensor, list)):
    #     boxes_2 = np.asarray(boxes_2)
    # elif not isinstance(boxes_2, np.ndarray):
    #     raise TypeError("boxes_2 必须是 torch.Tensor, np.ndarray 或 list")
    
    boxes_1 = boxes_1.reshape(-1, 4)
    boxes_2 = boxes_2.reshape(-1, 4)
    
    assert len(boxes_1) == len(boxes_2), "两个分组的框数量不匹配"

    for b1, b2 in zip(boxes_1, boxes_2):
        x1, y1, x2, y2 = b1.int().tolist()
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), width)  # 蓝色框（人）
        
        x1, y1, x2, y2 = b1.int().tolist()
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), width)  # 绿色框（物体）

        # 中心点
        b1_cx, b1_cy = ((b1[:2] + b1[2:]) / 2).int().tolist()
        b2_cx, b2_cy = ((b2[:2] + b2[2:]) / 2).int().tolist()

        # 中心连线
        cv2.line(image_cv, (b1_cx, b1_cy), (b2_cx, b2_cy), (0, 0, 255), width)

        # 中心点小圆
        cv2.circle(image_cv, (b1_cx, b1_cy), width, (0, 0, 255), -1)
        cv2.circle(image_cv, (b2_cx, b2_cy), width, (0, 0, 255), -1)
    return image_cv

OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes_cv(image, boxes):
    """
    使用OpenCV在image上绘制boxes,返回绘制后的图像
    """
    image = np.array(image)[:, :, ::-1].copy()  # PIL -> BGR
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            image, str(i + 1), (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            image, str(i + 1), (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return image

def visualise_entire_image(image, output, actions, action=None, thresh=0.2, fps=None):
    """
    显示所有符合阈值的动作（不再只显示最高得分）
    - 人框：紫色(255, 0, 255)
    - 物体框：绿色(0, 255, 0)
    - 连线：红色(0, 0, 255)
    - 动作文本：红色，垂直排列在人框左上角
    """
    # 转换PIL图像为OpenCV格式(BGR)
    image_cv = np.array(image)[:, :, ::-1].copy()
    ow, oh = image.size
    h, w = output['size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale_fct = torch.as_tensor([ow / w, oh / h, ow / w, oh / h]).unsqueeze(0).to(device)
    boxes = output['boxes'] * scale_fct  # 还原到原始图像尺寸
    
    scores = output['scores']
    objects = output['objects']
    pred = output['labels']
    pairing = output['pairing']
    
    # 筛选符合阈值的交互对（不再筛选最高得分）
    if action is None:
        keep_all = torch.nonzero(scores >= thresh).squeeze(1)
    else:
        keep_all = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
    
    # 收集所有符合条件的交互对
    hoi_list = []
    for idx in keep_all:
        h_idx, o_idx = pairing[:, idx].tolist()
        h_box = boxes[h_idx].cpu().numpy()
        o_box = boxes[o_idx].cpu().numpy()
        verb_id = pred[idx].item()
        obj_id = objects[idx].item()
        score = scores[idx].item()
        
        verb = actions[verb_id] if verb_id < len(actions) else f"verb_{verb_id}"
        obj = OBJECTS[obj_id] if obj_id < len(OBJECTS) else f"obj_{obj_id}"
        
        hoi_list.append({
            'h_box': h_box,
            'o_box': o_box,
            'i_name': verb,
            'i_score': score,
            'o_name': obj
        })

    # 合并同一个人的所有动作
    human_actions = collections.defaultdict(list)
    for hoi in hoi_list:
        human_actions[tuple(hoi['h_box'])].append(f"{hoi['i_name']} {hoi['o_name']}")

    # 模糊匹配去重绘制人物框
    drawn_human_boxes = {}  # key: 人物框, value: 所有动作
    for h_box_tuple, actions_list in human_actions.items():
        h_box = np.array(h_box_tuple)
        similar_key = None
        for drawn_box in drawn_human_boxes.keys():
            if is_similar_box(h_box, drawn_box, iou_thresh=0.9):
                similar_key = drawn_box
                break
        if similar_key:
            drawn_human_boxes[similar_key].extend(actions_list)
        else:
            drawn_human_boxes[h_box_tuple] = actions_list
            x1, y1, x2, y2 = h_box.astype(int)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 255), 2)  # 紫色人框

    # 绘制人物动作
    for h_box_tuple, all_actions in drawn_human_boxes.items():
        h_box = np.array(h_box_tuple)
        x1, y1, x2, y2 = h_box.astype(int)
        
        # 垂直排列动作文本
        unique_actions = list(set(all_actions))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        line_height = 32  # 减小行高以显示更多动作
        margin = 4       # 减小边距
        current_y = y1 + margin+16
        
        for action_text in unique_actions:
            (text_w, text_h), _ = cv2.getTextSize(action_text, font, font_scale, thickness)
            if current_y + text_h > y2 - margin:
                if not unique_actions:
                    cv2.putText(image_cv, "...", (x1+margin, y1+margin), 
                               font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                break
            cv2.putText(image_cv, action_text, (x1+margin, current_y), 
                       font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            current_y += text_h + line_height

    # 模糊匹配去重绘制物体框
    drawn_object_boxes = set()
    for hoi in hoi_list:
        o_box = hoi['o_box']
        o_box_tuple = tuple(o_box.astype(int))
        is_duplicate = False
        for drawn_box in drawn_object_boxes:
            if is_similar_box(o_box, np.array(drawn_box), iou_thresh=0.9):
                is_duplicate = True
                break
        if not is_duplicate:
            drawn_object_boxes.add(o_box_tuple)
            x1, y1, x2, y2 = o_box_tuple
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色物体框

    # 绘制人-物连线（所有交互对）
    drawn_pairs = set()
    for hoi in hoi_list:
        h_box = hoi['h_box'].astype(int)
        o_box = hoi['o_box'].astype(int)
        pair = (tuple(h_box), tuple(o_box))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        
        h_cx = int((h_box[0] + h_box[2]) // 2)
        h_cy = int((h_box[1] + h_box[3]) // 2)
        o_cx = int((o_box[0] + o_box[2]) // 2)
        o_cy = int((o_box[1] + o_box[3]) // 2)
        
        cv2.line(image_cv, (h_cx, h_cy), (o_cx, o_cy), (0, 0, 255), 2)  # 红色连线
        cv2.circle(image_cv, (h_cx, h_cy), 3, (0, 0, 255), -1)  # 红色中心点
        cv2.circle(image_cv, (o_cx, o_cy), 3, (0, 0, 255), -1)

    # 绘制FPS
    if fps is not None:
        cv2.putText(image_cv, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image_cv

def is_similar_box(box1, box2, iou_thresh=0.9):
    """计算两框IOU判断是否相似"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union > iou_thresh if union > 0 else False

@torch.no_grad()
def main(args):
    # 初始化数据集
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' else 24
    actions = dataset.dataset.verbs if args.dataset == 'hicodet' else \
        dataset.dataset.actions
    
    # 构建模型
    upt = build_detector(args, conversion)
    upt.eval()

    # 加载预训练权重
    if os.path.exists(args.resume):
        print(f"=> 从检查点加载权重: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> 使用随机初始化的模型")

    
    # 摄像头模式
    if args.camera:
        print("=> 启动摄像头模式...")
        cap = cv2.VideoCapture(0)  # 0 代表默认摄像头
        
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS计算变量
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        print("按ESC键退出摄像头模式")
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break
                
            # 转换BGR为RGB PIL图像
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_frame)

            # 预处理图像
            image_tensor, _ = dataset.transforms(image_pil, None)
            image_tensor = image_tensor.unsqueeze(0)
            print(type(image_tensor), image_tensor.shape)
            # 推理
            inference_start = time.time()
            output = upt(image_tensor)
            inference_end = time.time()
            
            inference_time = inference_end - inference_start

            # 计算FPS
            frame_count += 1
            if frame_count >= 5:  # 每5帧计算一次平均FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # 可视化结果
            vis_frame = visualise_entire_image(image_pil, output[0], actions, 
                                             args.action, args.action_score_thresh, fps)
            
            # 显示处理时间
            cv2.putText(vis_frame, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示结果
            cv2.imshow('Human-Object Interaction Detection', vis_frame)
            
            # 按ESC键退出
            if cv2.waitKey(1) == 27:
                print("退出摄像头模式")
                break
                
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
    # 单张图像模式
    elif args.image_path is not None:
        print(f"=> 处理图像: {args.image_path}")
        image = dataset.dataset.load_image(args.image_path)
        image_tensor, _ = dataset.transforms(image, None)
        image_tensor = image_tensor.unsqueeze(0)
        
        # 推理
        start_time = time.time()
        output = upt(image_tensor)
        inference_time = time.time() - start_time
        print(f"推理时间: {inference_time:.4f}秒")
        
        # 可视化结果
        vis_frame = visualise_entire_image(image, output[0], actions, 
                                         args.action, args.action_score_thresh)
        
        # 显示结果
        cv2.imshow('Human-Object Interaction Detection', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # 测试集图像模式
    else:
        print(f"=> 处理测试集图像索引: {args.index}")
        image, _ = dataset[args.index]
        image_tensor = image.unsqueeze(0)
        print(type(image_tensor), image_tensor.shape)
        output = upt(image_tensor)
        
        # 加载原始图像
        orig_image = dataset.dataset.load_image(
            os.path.join(dataset.dataset._root,
                dataset.dataset.filename(args.index)
        ))
        
        # 可视化结果
        vis_frame = visualise_entire_image(orig_image, output[0], actions, 
                                         args.action, args.action_score_thresh)
        
        # 显示结果
        cv2.imshow('Human-Object Interaction Detection', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--data-root', default=r'D:\2025春PPT\模式识别课设\代码\transformer\transformer\hicodet')
    parser.add_argument('--human-idx', type=int, default=0)

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
        help="Path to an image file.")
    # 添加摄像头参数
    parser.add_argument('--camera', action='store_true',
        help="Use camera for real-time detection")
    
    args = parser.parse_args()
    print("配置参数:")
    print(f"- 设备: {args.device}")
    print(f"- 数据集: {args.dataset}")
    print(f"- 模型权重: {args.resume if args.resume else '随机初始化'}")
    print(f"- 摄像头模式: {'启用' if args.camera else '禁用'}")
    if args.image_path:
        print(f"- 图像路径: {args.image_path}")
    else:
        print(f"- 测试集索引: {args.index}")

    main(args)