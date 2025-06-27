#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOI Detection UI Application
基于tkinter的人-物交互检测图形界面程序
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch

# 导入原有的demo模块
from demo import (
    load_model, create_default_args, read_cv2_image, resize_ensure_shortest_edge,
    prepare_cv2_image4nn, predict_on_one_image, viz_hoi_result, triplet_nms
)


class HOIDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HOI Detection Demo - 人物交互检测系统")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # 模型相关变量
        self.model = None
        self.device = None
        self.args = None
        self.is_model_loaded = False
        
        # 摄像头相关变量
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        
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
        ttk.Label(model_frame, text="数据集:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.dataset_var = tk.StringVar(value="hico")
        dataset_combo = ttk.Combobox(model_frame, textvariable=self.dataset_var, 
                                   values=["hico", "vcoco", "hoia"], state="readonly", width=15)
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
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
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
            
            # 创建参数对象
            self.args = create_default_args()
            self.args.dataset_file = self.dataset_var.get()
            self.args.backbone = self.backbone_var.get()
            self.args.model_path = self.model_path_var.get()
            
            # 加载模型
            self.model, self.device = load_model(self.model_path_var.get(), self.args)
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
                
                # 预处理
                img_rescale = resize_ensure_shortest_edge(img=img, size=672, max_size=1333)
                img_tensor = prepare_cv2_image4nn(img=img_rescale)
                
                # 检测
                hoi_list = predict_on_one_image(
                    self.args, self.model, self.device, img_tensor, img_size,
                    hoi_th=self.hoi_th_var.get(),
                    human_th=self.human_th_var.get(),
                    object_th=self.object_th_var.get(),
                    top_k=self.top_k_var.get()
                )
                
                # 可视化结果
                img_result = viz_hoi_result(img=img, hoi_list=hoi_list)
                self.current_image = img_result
                
                # 显示结果
                self.display_image(img_result)
                self.log_message(f"检测到 {len(hoi_list)} 个HOI交互")
                
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
            
            # 在新线程中处理视频
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
    
    def process_video(self):
        """处理视频检测"""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            
            if not self.video_cap.isOpened():
                self.log_message("无法打开视频文件")
                return
            
            frame_count = 0
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while self.is_video_running:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                frame_count += 1
                img_size = (frame.shape[0], frame.shape[1])
                
                # 每隔几帧处理一次（提高性能）
                if frame_count % 10 == 0:  # 每5帧处理一次
                    # 预处理
                    img_rescale = resize_ensure_shortest_edge(img=frame, size=672, max_size=1333)
                    img_tensor = prepare_cv2_image4nn(img=img_rescale)
                    
                    # 检测
                    hoi_list = predict_on_one_image(
                        self.args, self.model, self.device, img_tensor, img_size,
                        hoi_th=self.hoi_th_var.get(),
                        human_th=self.human_th_var.get(),
                        object_th=self.object_th_var.get(),
                        top_k=self.top_k_var.get()
                    )
                    
                    # 可视化结果
                    img_result = viz_hoi_result(img=frame, hoi_list=hoi_list)
                    self.current_image = img_result
                    
                    # 显示结果
                    self.display_image(img_result)
                    
                    if frame_count % 25 == 0:  # 每25帧打印一次日志
                        self.log_message(f"处理进度: {frame_count}/{total_frames}, 检测到 {len(hoi_list)} 个HOI交互")
                
                # 控制播放速度
                time.sleep(0.03)
            
            self.video_cap.release()
            self.log_message("视频处理完成")
            
        except Exception as e:
            self.log_message(f"视频检测出错: {str(e)}")
        finally:
            self.is_video_running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
    
    def start_camera_detection(self):
        """开始摄像头检测"""
        try:
            camera_id = int(self.camera_id_var.get())
        except ValueError:
            messagebox.showerror("错误", "摄像头ID必须是数字！")
            return
        
        if not self.is_camera_running:
            self.is_camera_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # 在新线程中处理摄像头
            self.camera_thread = threading.Thread(target=self.process_camera, args=(camera_id,), daemon=True)
            self.camera_thread.start()
    
    def process_camera(self, camera_id):
        """处理摄像头检测"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                self.log_message(f"无法打开摄像头 {camera_id}")
                return
            
            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.log_message(f"摄像头 {camera_id} 启动成功")
            frame_count = 0
            
            while self.is_camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("无法从摄像头读取帧")
                    break
                
                frame_count += 1
                img_size = (frame.shape[0], frame.shape[1])
                
                # 每隔几帧处理一次（提高实时性能）
                if frame_count % 3 == 0:  # 每3帧处理一次
                    # 预处理
                    img_rescale = resize_ensure_shortest_edge(img=frame, size=672, max_size=1333)
                    img_tensor = prepare_cv2_image4nn(img=img_rescale)
                    
                    # 检测
                    hoi_list = predict_on_one_image(
                        self.args, self.model, self.device, img_tensor, img_size,
                        hoi_th=self.hoi_th_var.get(),
                        human_th=self.human_th_var.get(),
                        object_th=self.object_th_var.get(),
                        top_k=5  # 摄像头模式使用较小的top_k提高速度
                    )
                    
                    # 可视化结果
                    img_result = viz_hoi_result(img=frame, hoi_list=hoi_list)
                    self.current_image = img_result
                    
                    # 显示结果
                    self.display_image(img_result)
                    
                    if frame_count % 30 == 0:  # 每30帧打印一次日志
                        self.log_message(f"实时检测中，当前检测到 {len(hoi_list)} 个HOI交互")
                
                # 控制帧率
                time.sleep(0.01)
            
            self.cap.release()
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
            # 将OpenCV图像转换为PIL图像
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # 获取Canvas尺寸
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # 确保Canvas已经显示
                    # 计算缩放比例
                    img_width, img_height = img_pil.size
                    scale_w = canvas_width / img_width
                    scale_h = canvas_height / img_height
                    scale = min(scale_w, scale_h, 1.0)  # 不放大图像
                    
                    # 调整图像尺寸
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 转换为tkinter可用的格式
                    img_tk = ImageTk.PhotoImage(img_resized)
                    
                    # 清除之前的图像并显示新图像
                    self.canvas.delete("all")
                    x = (canvas_width - new_width) // 2
                    y = (canvas_height - new_height) // 2
                    self.canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
                    
                    # 保持图像引用（防止被垃圾回收）
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
        if self.cap:
            self.cap.release()
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
