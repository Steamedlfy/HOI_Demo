#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOI Detection UI Application
基于tkinter的人-物交互检测图形界面程序
"""
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import queue
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import demo_qpic
import demo_upt
from utils import DataFactory
from upt import build_detector
# 导入原有的demo模块
from demo import (
    load_model, create_default_args, read_cv2_image, resize_ensure_shortest_edge,
    prepare_cv2_image4nn, predict_on_one_image, viz_hoi_result, triplet_nms
)


class HOIDetectionApp:
    def __init__(self, root):
        self.config = None
        self.root = root
        self.root.title("HOI Detection Demo - 人物交互检测系统")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # 模型相关变量
        self.model = None
        self.device = None
        self.args = None
        self.is_model_loaded = False
        self.UPT_dataset = None
        self.UPT_model = None
        self.UPT_device = None
        
        # 摄像头相关变量
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        self.canvas_image_id = None
        # 视频相关变量
        self.video_cap = None
        self.is_video_running = False
        self.video_thread = None
        self.video_path = None
        
        # 图像显示相关
        self.display_label = None
        self.current_image = None
        
        self.setup_ui()
        self.setup_styles()
        self.postprocesser = None
    def setup_styles(self):
        """设置UI样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
    
    def setup_ui(self):
        """设置UI界面"""
        # 创建主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 右侧显示面板
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 标题
        title_label = ttk.Label(parent, text="HOI Detection System", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # 模型配置区域
        self.setup_model_config(parent)
        
        # 功能选择区域
        self.setup_function_selection(parent)
        
        # 参数设置区域
        self.setup_parameters(parent)
        
        # 控制按钮区域
        self.setup_control_buttons(parent)
    
    def setup_model_config(self, parent):
        """设置模型配置区域"""
        model_frame = ttk.LabelFrame(parent, text="模型配置", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 数据集选择
        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.dataset_var = tk.StringVar(value="HoiTransformer")
        dataset_combo = ttk.Combobox(model_frame, textvariable=self.dataset_var, 
                                   values=["HoiTransformer", "QPIC",  "UPT"], state="readonly", width=15)
        dataset_combo.grid(row=0, column=1, pady=2, sticky=tk.W)
        
        # 骨干网络选择
        ttk.Label(model_frame, text="骨干网络:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.backbone_var = tk.StringVar(value="resnet50")
        backbone_combo = ttk.Combobox(model_frame, textvariable=self.backbone_var,
                                    values=["resnet50", "resnet101"], state="readonly", width=15)
        backbone_combo.grid(row=1, column=1, pady=2, sticky=tk.W)
        
        # 模型路径
        ttk.Label(model_frame, text="模型路径:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.model_path_var = tk.StringVar()
        model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=25)
        model_path_entry.grid(row=2, column=1, pady=2, sticky=tk.W)
        
        browse_btn = ttk.Button(model_frame, text="浏览", command=self.browse_model, width=8)
        browse_btn.grid(row=3, column=0, pady=5, sticky=tk.W)
        
        load_btn = ttk.Button(model_frame, text="加载模型", command=self.load_model, width=8)
        load_btn.grid(row=3, column=1, pady=5, sticky=tk.W)
        
        # 模型状态
        self.model_status_label = ttk.Label(model_frame, text="模型状态: 未加载", 
                                           foreground="red", style='Status.TLabel')
        self.model_status_label.grid(row=4, column=0, columnspan=2, pady=5)
    
    def setup_function_selection(self, parent):
        """设置功能选择区域"""
        function_frame = ttk.LabelFrame(parent, text="检测功能", padding="10")
        function_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.function_var = tk.StringVar(value="image")
        
        # 图像检测
        image_radio = ttk.Radiobutton(function_frame, text="图像检测", 
                                    variable=self.function_var, value="image")
        image_radio.pack(anchor=tk.W, pady=2)
        
        # 视频检测
        video_radio = ttk.Radiobutton(function_frame, text="视频检测", 
                                    variable=self.function_var, value="video")
        video_radio.pack(anchor=tk.W, pady=2)
        
        # 摄像头检测
        camera_radio = ttk.Radiobutton(function_frame, text="摄像头检测", 
                                     variable=self.function_var, value="camera")
        camera_radio.pack(anchor=tk.W, pady=2)
        
        # 摄像头ID设置
        camera_frame = ttk.Frame(function_frame)
        camera_frame.pack(fill=tk.X, pady=2)
        ttk.Label(camera_frame, text="摄像头ID:").pack(side=tk.LEFT)
        self.camera_id_var = tk.StringVar(value="0")
        camera_id_entry = ttk.Entry(camera_frame, textvariable=self.camera_id_var, width=5)
        camera_id_entry.pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_parameters(self, parent):
        """设置参数区域"""
        param_frame = ttk.LabelFrame(parent, text="检测参数", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # HOI阈值
        ttk.Label(param_frame, text="HOI阈值:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.hoi_th_var = tk.DoubleVar(value=0.6)
        hoi_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.hoi_th_var, 
                            orient=tk.HORIZONTAL, length=150)
        hoi_scale.grid(row=0, column=1, pady=2, sticky=tk.W)
        
        # 人体阈值
        ttk.Label(param_frame, text="人体阈值:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.human_th_var = tk.DoubleVar(value=0.6)
        human_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.human_th_var,
                              orient=tk.HORIZONTAL, length=150)
        human_scale.grid(row=1, column=1, pady=2, sticky=tk.W)
        
        # 物体阈值
        ttk.Label(param_frame, text="物体阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.object_th_var = tk.DoubleVar(value=0.6)
        object_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.object_th_var,
                               orient=tk.HORIZONTAL, length=150)
        object_scale.grid(row=2, column=1, pady=2, sticky=tk.W)
        
        # Top K
        ttk.Label(param_frame, text="Top K:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.top_k_var = tk.IntVar(value=25)
        top_k_spin = ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.top_k_var, width=10)
        top_k_spin.grid(row=3, column=1, pady=2, sticky=tk.W)
    
    def setup_control_buttons(self, parent):
        """设置控制按钮区域"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="开始检测", command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(control_frame, text="停止检测", command=self.stop_detection, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.save_btn = ttk.Button(control_frame, text="保存结果", command=self.save_result)
        self.save_btn.pack(side=tk.LEFT)
    
    def setup_right_panel(self, parent):
        """设置右侧显示面板"""
        # 图像显示区域
        display_frame = ttk.LabelFrame(parent, text="检测结果显示", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建Canvas用于显示图像
        self.canvas = tk.Canvas(display_frame, bg="white", width=500, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(parent, text="运行日志", padding="10")
        log_frame.pack(fill=tk.X)
        
        # 创建文本框和滚动条
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_text_frame, height=8, font=('Consolas', 9))
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def browse_model(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth *.pt"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """加载模型"""
        if not self.model_path_var.get():
            messagebox.showerror("错误", "请先选择模型文件！")
            return
        
        if not os.path.exists(self.model_path_var.get()):
            messagebox.showerror("错误", "模型文件不存在！")
            return
        
        try:
            self.log_message("正在加载模型...")
            if self.dataset_var.get() == 'HoiTransformer':
                # 创建参数对象
                self.args = create_default_args()
                self.args.dataset_file = 'hico'
                self.args.backbone = self.backbone_var.get()
                self.args.model_path = self.model_path_var.get()
                # 加载模型
                self.model, self.device = load_model(self.model_path_var.get(), self.args)
                self.is_model_loaded = True

            elif self.dataset_var.get() == "QPIC":
                # 使用Q-PIC模型加载
                self.args = demo_qpic.create_default_args()
                self.args.dataset_file = self.dataset_var.get()
                self.args.backbone = self.backbone_var.get()
                self.args.model_path = self.model_path_var.get()                
                # 加载Q-PIC模型
                self.model, self.postprocesser = demo_qpic.load_model(self.model_path_var.get(), self.args)
                self.is_model_loaded = True

            elif self.dataset_var.get() == "UPT":
                # 构造args
                self.args = demo_upt.argparse.Namespace()
                self.args.backbone = self.backbone_var.get()
                self.args.dilation = False
                self.args.position_embedding = 'sine'
                self.args.repr_dim = 512
                self.args.hidden_dim = 256
                self.args.enc_layers = 6
                self.args.dec_layers = 6
                self.args.dim_feedforward = 2048
                self.args.dropout = 0.1
                self.args.nheads = 8
                self.args.num_queries = 100
                self.args.pre_norm = False
                self.args.aux_loss = True
                self.args.set_cost_class = 1
                self.args.set_cost_bbox = 5
                self.args.set_cost_giou = 2
                self.args.bbox_loss_coef = 5
                self.args.giou_loss_coef = 2
                self.args.eos_coef = 0.1
                self.args.alpha = 0.5
                self.args.gamma = 0.2
                self.args.dataset = 'hicodet'
                self.args.partition = 'test2015'
                self.args.data_root = r'D:\2025春PPT\模式识别课设\代码\transformer\transformer\hicodet'  # 改为你的数据集路径
                self.args.human_idx = 0
                self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.args.pretrained = ''
                self.args.box_score_thresh = 0.2
                self.args.fg_iou_thresh = 0.5
                self.args.min_instances = 3
                self.args.max_instances = 15
                self.args.resume = self.model_path_var.get()
                self.args.index = 0
                self.args.action = None
                self.args.action_score_thresh = 0.2
                self.args.image_path = None
                self.args.camera = False

                # 加载模型
                self.UPT_dataset = DataFactory(name=self.args.dataset, partition=self.args.partition, data_root=self.args.data_root)
                conversion = self.UPT_dataset.dataset.object_to_verb if self.args.dataset == 'hicodet' else list(self.UPT_dataset.dataset.object_to_action.values())
                self.args.num_classes = 117 if self.args.dataset == 'hicodet' else 24
                self.model = build_detector(self.args, conversion)
                checkpoint = torch.load(self.model_path_var.get(), map_location=self.args.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.device = torch.device(self.args.device)
                self.model.to(self.device)
                self.is_model_loaded = True      
            # 更新状态
            self.model_status_label.config(text="模型状态: 已加载", foreground="green")
            self.log_message(f"模型加载成功，运行设备: {self.device}")
            
        except Exception as e:
            self.log_message(f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
    
    def start_detection(self):
        """开始检测"""
        if not self.is_model_loaded:
            messagebox.showerror("错误", "请先加载模型！")
            return
        
        function_type = self.function_var.get()
        
        if function_type == "image":
            self.start_image_detection()
        elif function_type == "video":
            self.start_video_detection()
        elif function_type == "camera":
            self.start_camera_detection()
    
    def start_image_detection(self):
        """开始图像检测"""
        filenames = filedialog.askopenfilenames(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        
        if not filenames:
            return
        
        self.log_message(f"选择了 {len(filenames)} 张图像进行检测")
        
        # 在新线程中处理图像检测
        threading.Thread(target=self.process_images, args=(filenames,), daemon=True).start()
    
    def process_images(self, filenames):
        """处理图像检测"""
        try:
            for i, img_path in enumerate(filenames):
                self.log_message(f"正在处理第 {i+1}/{len(filenames)} 张图像: {os.path.basename(img_path)}")
                
                # 读取图像
                img, img_size = read_cv2_image(img_path)
                print(f"self.dataset_var = {self.dataset_var.get()}")
                if self.dataset_var.get() =='HoiTransformer':
                    # 预处理
                    img_rescale = resize_ensure_shortest_edge(img=img, size=672, max_size=1333)
                    img_tensor = prepare_cv2_image4nn(img=img_rescale)
                    
                    # 检测
                    hoi_list,merged_humans, merged_objects = predict_on_one_image(
                        self.args, self.model, self.device, img_tensor, img_size,
                        hoi_th=self.hoi_th_var.get(),
                        human_th=self.human_th_var.get(),
                        object_th=self.object_th_var.get(),
                        top_k=5
                    )                   
                    # 可视化结果
                    img_result = viz_hoi_result(img=img, hoi_list=hoi_list,merged_humans=merged_humans, merged_objects=merged_objects)
                    self.current_image = img_result

                elif self.dataset_var.get() == 'QPIC':
                    img_tensor = demo_qpic.pretrain(img)
                    outputs = self.model(img_tensor)
                    orig_size = torch.tensor([[img.shape[0], img.shape[1]]]).to(self.device)
                    results = self.postprocesser['hoi'](outputs, orig_size)
                    img_result = demo_qpic.filter_and_visualize_hoi(img,results[0])
                    self.current_image = img_result  
                # 显示结果
                elif self.dataset_var.get() == 'UPT':
                    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    image_tensor, _ = self.UPT_dataset.transforms(image_pil, None)
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.model(image_tensor)
                    actions = self.UPT_dataset.dataset.verbs if hasattr(self.UPT_dataset.dataset, 'verbs') else self.UPT_dataset.dataset.actions
                    img_result = demo_upt.visualise_entire_image(image_pil, output[0], actions, thresh=self.hoi_th_var.get())
                    self.current_image = img_result
                self.display_image(img_result)
                
        except Exception as e:
            self.log_message(f"图像检测出错: {str(e)}")
            messagebox.showerror("错误", f"图像检测出错: {str(e)}")

    
    def start_video_detection(self):
        """开始视频检测"""
        filename = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )

        if not filename:
            return

        self.video_path = filename
        self.log_message(f"选择视频文件: {os.path.basename(filename)}")

        if not self.is_video_running:
            self.is_video_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # 初始化队列和缓存
            self.frame_queue = queue.Queue(maxsize=30)
            self.result_queue = queue.Queue()
            self.current_processed_result = None
            self.last_processed_frame = -5

            # 只启动一个线程，执行process_video，内部再启动两个线程
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()

    def process_video(self):
        """处理视频检测（多线程跳帧检测）"""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                self.log_message("无法打开视频文件")
                return

            # 获取视频信息
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            preprocess_frames = int(fps)  # 预处理1秒的帧数
            
            result_queue = queue.Queue(maxsize=100)  # 进一步增大队列
            detection_ready = threading.Event()  # 检测预处理完成信号
            detection_complete = threading.Event()  # 检测完全完成信号
            last_detection_result = None  # 新增：保存最后一次检测结果

            def detection_worker():
                """检测线程工作函数 - 独立读取视频流"""
                detection_cap = None
                try:
                    detection_cap = cv2.VideoCapture(self.video_path)
                    if not detection_cap.isOpened():
                        self.log_message("检测线程无法打开视频文件")
                        return

                    step = 2  # 每5帧处理一次
                    frame_idx = 0
                    processed_count = 0
                    
                    self.log_message(f"检测线程开始预处理，目标帧数: {preprocess_frames}, 每{step}帧处理一次")
                    
                    while self.is_video_running and frame_idx < total_frames:
                        # 精确跳到目标帧位置
                        detection_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = detection_cap.read()
                        
                        if not ret:
                            print(f"[Detection] Failed to read frame {frame_idx}")
                            break
                        
                        print(f"[Detection] Processing frame {frame_idx}")
                        
                        # 执行检测
                        img_result = None
                        try:
                            if self.dataset_var.get() == 'HoiTransformer':
                                img_rescale = resize_ensure_shortest_edge(img=frame, size=672, max_size=1333)
                                img_tensor = prepare_cv2_image4nn(img=img_rescale)
                                hoi_list, merged_humans, merged_objects = predict_on_one_image(
                                    self.args, self.model, self.device, img_tensor, (frame.shape[0], frame.shape[1]),
                                    hoi_th=self.hoi_th_var.get(),
                                    human_th=self.human_th_var.get(),
                                    object_th=self.object_th_var.get(),
                                    top_k=5
                                )
                                img_result = viz_hoi_result(img=frame, hoi_list=hoi_list, merged_humans=merged_humans, merged_objects=merged_objects)
                                
                        except Exception as e:
                            self.log_message(f"帧 {frame_idx} 检测出错: {str(e)}")
                        
                        # 将检测结果放入队列
                        if img_result is not None:
                            try:
                                result_queue.put((frame_idx, img_result), timeout=5)
                                processed_count += 1
                                print(f"[Detection] Put result for frame {frame_idx} (processed: {processed_count})")
                                
                                # 检查是否完成预处理
                                if not detection_ready.is_set() and frame_idx >= preprocess_frames:
                                    detection_ready.set()
                                    self.log_message(f"检测预处理完成，已处理到第 {frame_idx} 帧")
                                    
                            except queue.Full:
                                print(f"[Detection] Result queue full, skipping frame {frame_idx}")
                        
                        frame_idx += step
                        
                        # 检查是否需要停止
                        if not self.is_video_running:
                            break
                    
                    detection_complete.set()
                    self.log_message(f"检测线程完成，共处理 {processed_count} 帧")
                    
                except Exception as e:
                    self.log_message(f"检测线程出错: {str(e)}")
                finally:
                    if detection_cap:
                        detection_cap.release()
                    detection_ready.set()  # 确保即使出错也能设置信号
                    detection_complete.set()

            # 启动检测线程
            detect_thread = threading.Thread(target=detection_worker, daemon=True)
            detect_thread.start()

            # 等待预处理完成
            self.log_message("等待检测预处理完成...")
            if not detection_ready.wait(timeout=30):  # 最多等待30秒
                self.log_message("检测预处理超时，开始播放")
            else:
                self.log_message("检测预处理完成，开始播放")

            # 收集所有可用的检测结果
            detection_results = {}
            while not result_queue.empty():
                try:
                    result_frame_idx, img_result = result_queue.get_nowait()
                    detection_results[result_frame_idx] = img_result
                    result_queue.task_done()
                except queue.Empty:
                    break

            self.log_message(f"预处理收集到 {len(detection_results)} 个检测结果")
            
            # 开始主循环播放
            frame_count = 0
            self.log_message(f"开始播放视频，总帧数: {total_frames}")

            while self.is_video_running:
                ret, frame = self.video_cap.read()
                if not ret:
                    self.log_message("主线程视频读取完毕")
                    break
                
                frame_count += 1
                
                # 持续收集新的检测结果（非阻塞）
                new_results_count = 0
                while new_results_count < 10:  # 限制每次处理的数量
                    try:
                        result_frame_idx, img_result = result_queue.get_nowait()
                        detection_results[result_frame_idx] = img_result
                        print(f"[Main] Received new detection result for frame {result_frame_idx}")
                        result_queue.task_done()
                        new_results_count += 1
                    except queue.Empty:
                        break

                # 修改显示逻辑：延续上一次检测结果
                if frame_count in detection_results:
                    display_frame = detection_results[frame_count]
                    last_detection_result = display_frame  # 保存最新检测结果
                    print(f"[Main] Displaying detection result for frame {frame_count}")
                elif last_detection_result is not None:
                    display_frame = last_detection_result  # 使用上一次检测结果
                    print(f"[Main] Displaying last detection result for frame {frame_count}")
                else:
                    display_frame = frame  # 没有任何检测结果时显示原始帧
                    print(f"[Main] Displaying original frame {frame_count}")

                # 显示当前帧
                self.current_image = display_frame
                self.display_image(display_frame)
                
                # 清理过期的检测结果（保留当前帧前后50帧的结果）
                frames_to_remove = []
                for detection_frame_idx in detection_results.keys():
                    if abs(detection_frame_idx - frame_count) > 50:
                        frames_to_remove.append(detection_frame_idx)
                
                for f in frames_to_remove:
                    del detection_results[f]

                # 控制播放速度
                time.sleep(0.03)  # 约33fps
                
                # 每200帧输出一次状态
                if frame_count % 200 == 0:
                    available_results = len([f for f in detection_results.keys() if f >= frame_count])
                    self.log_message(f"播放进度: {frame_count}/{total_frames}, 可用检测结果: {available_results}, 队列大小: {result_queue.qsize()}")

            # 等待检测线程完成
            self.log_message("等待检测线程完成...")
            detection_complete.wait(timeout=10)
            
            # 处理剩余的检测结果
            remaining_results = 0
            while not result_queue.empty():
                try:
                    result_frame_idx, img_result = result_queue.get_nowait()
                    remaining_results += 1
                    result_queue.task_done()
                except queue.Empty:
                    break
            
            if remaining_results > 0:
                self.log_message(f"处理了 {remaining_results} 个剩余检测结果")
            
            self.video_cap.release()
            self.log_message("视频处理完成")

        except Exception as e:
            self.log_message(f"视频检测出错: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_video_running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")        

    def start_camera_detection(self):
        """开始摄像头检测"""
        try:
            pass
        except ValueError:
            messagebox.showerror("错误", "摄像头ID必须是数字！")
            return
        
        if not self.is_camera_running:
            self.is_camera_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # 在新线程中处理摄像头
            self.camera_thread = threading.Thread(target=self.process_camera, args=(0,), daemon=True)
            self.camera_thread.start()
    
    def process_camera(self, camera_id):
        """处理摄像头检测"""
        try:
            # self.pipeline.start(self.config)
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                self.log_message(f"无法打开摄像头 {camera_id}")
                return
            # 
            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.log_message(f"摄像头启动成功")
            frame_count = 0
            
            while self.is_camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                img_size = (frame.shape[0], frame.shape[1])
                
                # 每隔几帧处理一次（提高实时性能）
                if frame_count % 3 == 0:  # 每3帧处理一次
                    # 预处理
                    if self.dataset_var.get() == 'HoiTransformer':
                        img_rescale = resize_ensure_shortest_edge(img=frame, size=672, max_size=1333)
                        img_tensor = prepare_cv2_image4nn(img=img_rescale)
                        
                        # 检测
                        hoi_list,merged_humans, merged_objects = predict_on_one_image(
                            self.args, self.model, self.device, img_tensor, img_size,
                            hoi_th=self.hoi_th_var.get(),
                            human_th=self.human_th_var.get(),
                            object_th=self.object_th_var.get(),
                            top_k=5
                        )
                        
                        # 可视化结果
                        img_result = viz_hoi_result(img=frame, hoi_list=hoi_list,merged_humans=merged_humans, merged_objects=merged_objects)
                    elif self.dataset_var.get() == 'QPIC':
                        img_tensor = demo_qpic.pretrain(frame)
                        outputs = self.model(img_tensor)
                        orig_size = torch.tensor([[frame.shape[0], frame.shape[1]]]).to(self.device)
                        results = self.postprocesser['hoi'](outputs, orig_size)
                        img_result = demo_qpic.filter_and_visualize_hoi(frame, results[0])
                        self.current_image = img_result  
                    
                    elif self.dataset_var.get() == 'UPT':
                        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image_tensor, _ = self.UPT_dataset.transforms(image_pil, None)
                        image_tensor = image_tensor.unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            output = self.model(image_tensor)
                        actions = self.UPT_dataset.dataset.verbs if hasattr(self.UPT_dataset.dataset, 'verbs') else self.UPT_dataset.dataset.actions
                        img_result = demo_upt.visualise_entire_image(image_pil, output[0], actions, thresh=self.hoi_th_var.get())
                        self.current_image = img_result
                    self.current_image = img_result                 
                    # 显示结果
                    self.display_image(img_result)                
                # 控制帧率
                time.sleep(0.01)
            
            self.pipeline.stop()
            self.log_message("摄像头检测停止")
            
        except Exception as e:
            self.log_message(f"摄像头检测出错: {str(e)}")
        finally:
            self.is_camera_running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
    
    def stop_detection(self):
        """停止检测"""
        if self.is_video_running:
            self.is_video_running = False
            self.log_message("正在停止视频检测...")
        
        if self.is_camera_running:
            self.is_camera_running = False
            self.log_message("正在停止摄像头检测...")
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    def display_image(self, img):
        """在Canvas上显示图像"""
        try:
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:
                    img_width, img_height = img_pil.size
                    scale_w = canvas_width / img_width
                    scale_h = canvas_height / img_height
                    scale = min(scale_w, scale_h, 1.0)

                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    img_tk = ImageTk.PhotoImage(img_resized)

                    # 更新画布图像而不是每次重新创建
                    if self.canvas_image_id is None:
                        x = (canvas_width - new_width) // 2
                        y = (canvas_height - new_height) // 2
                        self.canvas_image_id = self.canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
                    else:
                        self.canvas.itemconfig(self.canvas_image_id, image=img_tk)

                    # 防止图像被回收
                    self.canvas.image = img_tk

        except Exception as e:
            self.log_message(f"显示图像出错: {str(e)}")

    def save_result(self):
        """保存检测结果"""
        if self.current_image is None:
            messagebox.showwarning("警告", "没有可保存的检测结果！")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存检测结果",
            defaultextension=".jpg",
             filetypes=[("JPEG图像", "*.jpg"), ("PNG图像", "*.png"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.current_image)
                self.log_message(f"检测结果已保存到: {filename}")
                messagebox.showinfo("成功", "检测结果保存成功！")
            except Exception as e:
                self.log_message(f"保存结果出错: {str(e)}")
                messagebox.showerror("错误", f"保存结果出错: {str(e)}")
    
    def on_closing(self):
        """程序关闭时的清理工作"""
        # 停止所有检测
        self.stop_detection()
        
        # 等待线程结束
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
        
        # 释放资源
        # if self.cap:
            # self.cap.release()
        if self.video_cap:
            self.video_cap.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    """主函数"""
    # 检查CUDA可用性
    device_info = "CUDA可用" if torch.cuda.is_available() else "仅CPU可用"
    print(f"PyTorch版本: {torch.__version__}")
    print(f"计算设备: {device_info}")
    
    # 创建主窗口
    root = tk.Tk()
    app = HOIDetectionApp(root)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动应用
    print("启动HOI检测UI应用...")
    app.log_message("HOI检测系统启动成功")
    app.log_message(f"计算设备: {device_info}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
