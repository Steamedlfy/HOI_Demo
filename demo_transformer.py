# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------

import argparse
import random
import os
from tqdm import tqdm
from PIL import Image

import cv2
import numpy as np
import torch
import torchvision
from datasets_transformer.hico import hoi_interaction_names as hoi_interaction_names_hico
from datasets_transformer.hico import coco_instance_ID_to_name as coco_instance_ID_to_name_hico
from datasets_transformer.hoia import hoi_interaction_names as hoi_interaction_names_hoia
from datasets_transformer.hoia import coco_instance_ID_to_name as coco_instance_ID_to_name_hoia
from datasets_transformer.vcoco import hoi_interaction_names as hoi_interaction_names_vcoco
from datasets_transformer.vcoco import coco_instance_ID_to_name as coco_instance_ID_to_name_vcoco
from models_transformer import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Backbone.
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101'], required=True,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer.
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

    # Loss.
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher.
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss coefficients.
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # Dataset parameters.
    parser.add_argument('--dataset_file', choices=['hico', 'vcoco', 'hoia'], required=True)

    parser.add_argument('--model_path', required=True,
                        help='Path of the model to evaluate.')
    parser.add_argument('--log_dir', default='./',
                        help='path where to save temporary files in test')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # Distributed training parameters.
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Visualization.
    parser.add_argument('--img_sheet', help='File containing image paths.')
    parser.add_argument('--camera', action='store_true', help='Use camera for real-time detection')
    parser.add_argument('--camera_id', default=0, type=int, help='Camera device ID (default: 0)')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--video_output', default='output.avi', help='Output video filename')
    return parser


def random_color():
    rdn = random.randint(1, 1000)
    b = int(rdn * 997) % 255
    g = int(rdn * 4447) % 255
    r = int(rdn * 6563) % 255
    return b, g, r


def intersection(box_a, box_b):
    # box: x1, y1, x2, y2
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float((x2 - x1 + 1) * (y2 - y1 + 1))


def IoU(box_a, box_b):
    inter = intersection(box_a, box_b)
    box_a_area = (box_a[2]-box_a[0]+1) * (box_a[3]-box_a[1]+1)
    box_b_area = (box_b[2]-box_b[0]+1) * (box_b[3]-box_b[1]+1)
    union = box_a_area + box_b_area - inter
    return inter / float(max(union, 1))


def triplet_nms(hoi_list):
    hoi_list.sort(key=lambda x: x['h_cls'] * x['o_cls'] * x['i_cls'], reverse=True)
    mask = [True] * len(hoi_list)
    for idx_x in range(len(hoi_list)):
        if mask[idx_x] is False:
            continue
        for idx_y in range(idx_x+1, len(hoi_list)):
            x = hoi_list[idx_x]
            y = hoi_list[idx_y]
            iou_human = IoU(x['h_box'], y['h_box'])
            iou_object = IoU(x['o_box'], y['o_box'])
            if iou_human > 0.5 and iou_object > 0.5 and x['i_name'] == y['i_name'] and x['o_name'] == y['o_name']:
                mask[idx_y] = False
    new_hoi_list = []
    for idx in range(len(mask)):
        if mask[idx] is True:
            new_hoi_list.append(hoi_list[idx])
    return new_hoi_list


def load_model(model_path, args):
    checkpoint = torch.load(model_path, map_location='cpu')
    print('epoch:', checkpoint['epoch'])
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, device


def read_cv2_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_hh, img_ww = img.shape[0:2]
    return img, (img_hh, img_ww)


def resize_ensure_shortest_edge(img, size, max_size):
    def get_size_with_aspect_ratio(img_size, _size, _max_size=None):
        h, w = img_size
        if _max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * _size > _max_size:
                _size = int(round(_max_size * min_original_size / max_original_size))
        if (w <= h and w == _size) or (h <= w and h == _size):
            return h, w
        if w < h:
            ow = _size
            oh = int(_size * h / w)
        else:
            oh = _size
            ow = int(_size * w / h)
        return ow, oh

    rescale_size = get_size_with_aspect_ratio(img_size=img.shape[0:2], _size=size, _max_size=max_size)
    img_rescale = cv2.resize(img, rescale_size)
    return img_rescale


def prepare_cv2_image4nn(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
    img = torchvision.transforms.functional.to_tensor(img)
    img_tensor = torchvision.transforms.functional.normalize(img, mean=mean, std=std)
    return img_tensor


def parse_object_box(org_cid, org_box, org_cls, img_size, coco_instance_id_to_name):
    cid = org_cid
    cx, cy, w, h = org_box
    hh, ww = img_size
    cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
    n_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
    n_cls = org_cls
    n_name = coco_instance_id_to_name[int(cid)]
    return n_box, n_cls, n_name


def predict_on_one_image(args, model, device, img_tensor, img_size, hoi_th, human_th, object_th, top_k=100):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file == 'hico':
        num_classes = 91
        num_actions = 118
        hoi_interaction_names = hoi_interaction_names_hico
        coco_instance_id_to_name = coco_instance_ID_to_name_hico
    elif args.dataset_file == 'vcoco':
        num_classes = 91
        num_actions = 30
        hoi_interaction_names = hoi_interaction_names_vcoco
        coco_instance_id_to_name = coco_instance_ID_to_name_vcoco
    else:
        num_classes = 12
        num_actions = 11
        hoi_interaction_names = hoi_interaction_names_hoia
        coco_instance_id_to_name = coco_instance_ID_to_name_hoia

    samples = torch.unsqueeze(img_tensor, dim=0)
    samples = samples.to(device)
    outputs = model(samples)
    action_pred_logits = outputs['action_pred_logits'][0]
    object_pred_logits = outputs['object_pred_logits'][0]
    object_pred_boxes = outputs['object_pred_boxes'][0]
    human_pred_logits = outputs['human_pred_logits'][0]
    human_pred_boxes = outputs['human_pred_boxes'][0]

    act_cls = torch.nn.Softmax(dim=1)(action_pred_logits).detach().cpu().numpy()[:, :-1]
    human_cls = torch.nn.Softmax(dim=1)(human_pred_logits).detach().cpu().numpy()[:, :-1]
    human_box = human_pred_boxes.detach().cpu().numpy()
    object_cls = torch.nn.Softmax(dim=1)(object_pred_logits).detach().cpu().numpy()[:, :-1]
    object_box = object_pred_boxes.detach().cpu().numpy()

    keep = (act_cls.argmax(axis=1) != num_actions)
    keep = keep * (human_cls.argmax(axis=1) != 2)
    keep = keep * (object_cls.argmax(axis=1) != num_classes)

    human_idx_max_list = human_cls[keep].argmax(axis=1)
    human_val_max_list = human_cls[keep].max(axis=1)
    human_box_max_list = human_box[keep]
    object_idx_max_list = object_cls[keep].argmax(axis=1)
    object_val_max_list = object_cls[keep].max(axis=1)
    object_box_max_list = object_box[keep]
    keep_act_scores = act_cls[keep]

    keep_act_scores_1d = keep_act_scores.reshape(-1)
    top_k_idx_1d = np.argsort(-keep_act_scores_1d)[:top_k]
    box_action_pairs = [(idx_1d // num_actions, idx_1d % num_actions) for idx_1d in top_k_idx_1d]

    hoi_list = []
    for idx_box, idx_action in box_action_pairs:
        # action
        i_box = (0, 0, 0, 0)
        i_cls = keep_act_scores[idx_box, idx_action]
        i_name = hoi_interaction_names[int(idx_action)]
        if i_name in ['__background__', 'walk', 'smile', 'run', 'stand']:
            continue
        # human
        h_box, h_cls, h_name = parse_object_box(
            org_cid=human_idx_max_list[idx_box], org_box=human_box_max_list[idx_box],
            org_cls=human_val_max_list[idx_box], img_size=img_size, coco_instance_id_to_name=coco_instance_id_to_name,
        )
        # object
        o_box, o_cls, o_name = parse_object_box(
            org_cid=object_idx_max_list[idx_box], org_box=object_box_max_list[idx_box],
            org_cls=object_val_max_list[idx_box], img_size=img_size, coco_instance_id_to_name=coco_instance_id_to_name,
        )
        if i_cls < hoi_th or h_cls < human_th or o_cls < object_th:
            continue
        pp = dict(
            h_cls=float(h_cls), o_cls=float(o_cls), i_cls=float(i_cls),
            h_box=h_box, o_box=o_box, i_box=i_box, h_name=h_name, o_name=o_name, i_name=i_name,
        )
        hoi_list.append(pp)

    hoi_list = triplet_nms(hoi_list)
    return hoi_list


def viz_hoi_result(img, hoi_list):
    print('hoi_list:', hoi_list)
    img_result = img.copy()
    for idx_box, hoi in enumerate(hoi_list):
        color = (0, 0, 255)
        # action
        i_cls, i_name = hoi['i_cls'], hoi['i_name']
        cv2.putText(img_result, '%s:%.2f' % (i_name, i_cls),
                    (10, 50 * idx_box + 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        # human
        color = (255, 0, 255)
        x1, y1, x2, y2 = hoi['h_box']
        h_cls, h_name = hoi['h_cls'], hoi['h_name']
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_result, '%s:%.2f' % (h_name, h_cls), (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        # object
        color = (0, 255, 0)
        x1o, y1o, x2o, y2o = hoi['o_box']
        o_cls, o_name = hoi['o_cls'], hoi['o_name']
        cv2.rectangle(img_result, (x1o, y1o), (x2o, y2o), color, 2)
        cv2.putText(img_result, '%s:%.2f' % (o_name, o_cls), (x1o, y2o), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        # 连线：连接人框中心和物体框中心（红色线）
        x1h, y1h, x2h, y2h = hoi['h_box']
        cx_h = int((x1h + x2h) / 2)
        cy_h = int((y1h + y2h) / 2)
        cx_o = int((x1o + x2o) / 2)
        cy_o = int((y1o + y2o) / 2)
        cv2.line(img_result, (cx_h, cy_h), (cx_o, cy_o), (0, 0, 255), 3)  # 红色连线
    if img_result.shape[0] > 480:
        ratio = img_result.shape[0] / 480
        img_result = cv2.resize(img_result, (int(img_result.shape[1] / ratio), int(img_result.shape[0] / ratio)))
    return img_result


def run_on_images(args, img_path_list):
    model, device = load_model(model_path=args.model_path, args=args)
    log_dir = os.path.join(args.log_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    for idx_img, img_path in enumerate(tqdm(img_path_list)):
        # read image data
        img, img_size = read_cv2_image(img_path=img_path)

        # inference on one image
        img_rescale = resize_ensure_shortest_edge(img=img, size=672, max_size=1333)
        img_tensor = prepare_cv2_image4nn(img=img_rescale)
        hoi_list = predict_on_one_image(
            args, model, device, img_tensor, img_size, hoi_th=0.6, human_th=0.6, object_th=0.6, top_k=25,
        )
        img_name = 'img_%s_%06d.jpg' % (os.path.basename(img_path), idx_img)
        img_result = viz_hoi_result(img=img, hoi_list=hoi_list)
        cv2.imwrite(os.path.join(log_dir, img_name), img_result)

def run_on_camera(args):
    model, device = load_model(model_path=args.model_path, args=args)
    
    # 初始化摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera_id}")
        return
    
    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 视频保存设置
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.video_output, fourcc, fps, (width, height))
    
    print("Press 'q' to quit, 's' to save current frame")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera")
                break
            
            img_size = (frame.shape[0], frame.shape[1])
            
            # 图像预处理
            img_rescale = resize_ensure_shortest_edge(img=frame, size=672, max_size=1333)
            img_tensor = prepare_cv2_image4nn(img=img_rescale)
            
            # HOI检测
            hoi_list = predict_on_one_image(
                args, model, device, img_tensor, img_size, 
                hoi_th=0.7, human_th=0.7, object_th=0.7, top_k=3,
            )
            
            # 可视化结果
            img_result = viz_hoi_result(img=frame, hoi_list=hoi_list)
            
            # 添加FPS信息
            info_text = f"Frame: {frame_count}, Detections: {len(hoi_list)}"
            text_y = img_result.shape[0] - 20  # 距离底部20像素
            cv2.putText(img_result, info_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('HOI Detection', img_result)
            
            # 保存视频帧（如果启用）
            if args.save_video:
                out.write(img_result)
            
            frame_count += 1
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                save_path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, img_result)
                print(f"Saved frame to {save_path}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # 清理资源
        cap.release()
        if args.save_video:
            out.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")
    

def create_default_args():
    """创建默认参数"""
    args = argparse.Namespace()
    args.lr = 1e-4
    args.lr_backbone = 1e-5
    args.batch_size = 1
    args.weight_decay = 1e-4
    args.epochs = 250
    args.lr_drop = 200
    args.clip_max_norm = 0.1
    args.position_embedding = 'sine'
    args.enc_layers = 6
    args.dec_layers = 6
    args.dim_feedforward = 2048
    args.hidden_dim = 256
    args.dropout = 0.1
    args.nheads = 8
    args.num_queries = 100
    args.pre_norm = False
    args.aux_loss = True
    args.set_cost_class = 1
    args.set_cost_bbox = 5
    args.set_cost_giou = 2
    args.dice_loss_coef = 1
    args.bbox_loss_coef = 5
    args.giou_loss_coef = 2
    args.eos_coef = 0.02
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 42
    args.resume = ''
    args.start_epoch = 0
    args.num_workers = 0
    args.world_size = 1
    args.dist_url = 'env://'
    return args



def main():
    """
    # 图像模式
    python3 test_on_images.py --dataset_file=hico --backbone=resnet50 \
        --batch_size=1 --log_dir=./ --model_path=your_model_path --img_sheet=your_image_sheet_file
    
    # 摄像头模式
    python3 test_on_images.py --dataset_file=hico --backbone=resnet50 \
        --batch_size=1 --model_path=your_model_path --camera
    
    # 摄像头模式并保存视频
    python3 test_on_images.py --dataset_file=hico --backbone=resnet50 \
        --batch_size=1 --model_path=your_model_path --camera --save_video --video_output=hoi_output.avi
    """
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)

    if args.camera:
        # 摄像头实时检测模式
        run_on_camera(args=args)
    else:
        # 原有的图像批处理模式
        if args.img_sheet is None:
            img_path_list = [
                './data/hico/images/test2015/HICO_test2015_00000001.jpg',
                './data/hoia/images/test/test_000000.png',
            ]
        else:
            img_path_list = [l.strip() for l in open(args.img_sheet, 'r').readlines()]
        
        run_on_images(args=args, img_path_list=img_path_list)
    
    print('done')



if __name__ == '__main__':
    main()
