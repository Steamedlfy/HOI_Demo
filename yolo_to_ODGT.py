import os
import json
from PIL import Image

def convert_yolo_to_odgt(yolo_dir, image_dir, output_file, class_names=None):
    """
    将YOLO格式标注转换为ODGT格式，HOI部分留空
    """
    # 读取类别名称
    if class_names is None:
        classes_path = os.path.join(yolo_dir, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = []
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历YOLO标注文件
        for filename in os.listdir(yolo_dir):
            if filename.endswith('.txt') and filename != 'classes.txt':
                yolo_path = os.path.join(yolo_dir, filename)
                image_name = os.path.splitext(filename)[0] + '.jpg'
                image_path = os.path.join(image_dir, image_name)
                
                # 检查图像存在性
                if not os.path.exists(image_path):
                    print(f"警告: 缺少图像 {image_name}，跳过")
                    continue
                
                try:
                    # 获取图像尺寸
                    with Image.open(image_path) as img:
                        width, height = img.size
                    
                    # 解析标注
                    annotations = []
                    with open(yolo_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            class_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            
                            # 转换为像素坐标并取整
                            x1 = int((xc - w/2) * width)
                            y1 = int((yc - h/2) * height)
                            x2 = int((xc + w/2) * width)
                            y2 = int((yc + h/2) * height)
                            
                            # 获取类别名称
                            category = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
                            annotations.append({"tag": category, "box": [x1, y1, x2, y2]})
                    
                    # 生成ODGT条目（HOI留空）
                    odgt_entry = {
                        "file_name": image_name,
                        "gtboxes": annotations,
                        "width": width,
                        "height": height,
                        "hoi": []
                    }
                    outfile.write(json.dumps(odgt_entry, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")

def main():
    # ================== 请修改以下路径 ==================
    yolo_dir = r"D:\2025春PPT\模式识别课设\代码\transformer\transformer\Additional_dataset\labels"      # YOLO标注目录（.txt文件）
    image_dir = r"D:\2025春PPT\模式识别课设\代码\transformer\transformer\Additional_dataset\images"          # 图像目录（.jpg文件）
    output_file = r"D:\2025春PPT\模式识别课设\代码\transformer\transformer\data\additional_trainval.odgt"   # 输出ODGT文件路径
    
    # 类别名称（按class_id顺序排列）
    class_names = ["person", "cell phone"]  # 请根据实际类别修改
    
    # 执行转换
    convert_yolo_to_odgt(yolo_dir, image_dir, output_file, class_names)
    print(f"转换完成，结果保存至 {output_file}")

if __name__ == "__main__":
    main()