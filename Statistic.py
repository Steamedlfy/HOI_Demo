import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontProperties

odgt_file = r'D:\2025春PPT\模式识别课设\代码\transformer\transformer\data\hico_trainval_remake.odgt'

interaction_counter = Counter()

with open(odgt_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        gtboxes = data['gtboxes']
        hoi_list = data['hoi']
        for hoi in hoi_list:
            interaction = hoi['interaction']
            if interaction == "no_interaction":
                continue  # 跳过无交互
            object_id = hoi['object_id']
            object_tag = gtboxes[object_id]['tag']
            key = f"{interaction}+{object_tag}"
            interaction_counter[key] += 1

# 绘制分布图
plt.figure(figsize=(12, 6))
sorted_items = sorted(interaction_counter.items(), key=lambda x: x[1])
labels, values = zip(*sorted_items)
# 生成颜色列表，默认蓝色，hold+apple和wash+apple为红色
bar_colors = []
for label in labels:
    if label in ['hold+apple', 'wash+apple']:
        bar_colors.append('red')
    else:
        bar_colors.append('C0')  # 默认蓝色
plt.bar(range(len(labels)), values)
plt.xticks([], [])  # 不显示横坐标的类别名称
#plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel('Interaction+Object')
plt.ylabel('Count')
plt.title('Distribution of Interaction+Object (no "no_interaction")')
plt.tight_layout()

# 标注 hold+apple 和 wash+apple
for action in ['hold+apple', 'wash+apple']:
    if action in labels:
        idx = labels.index(action)
        val = values[idx]
                # 在柱子上方显示数值
        plt.text(idx, val + max(values)*0.01, str(val),
                 ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
        # 在x轴下方显示动作名称
        plt.text(idx, -max(values)*0.03, action,
                 rotation=0, ha='center', va='top', fontsize=10, color='red', fontweight='bold')
    else:
        print(f"{action} 不在统计结果中")
        
plt.show()

# 列举出现次数小于等于3的类别，并以列表格式输出
rare_interactions = [(k, v) for k, v in interaction_counter.items() if v <= 3]
print("出现次数小于等于5的交互类别及其次数：")

print(rare_interactions)
# 画出稀有类别的分布图
if rare_interactions:
    rare_labels, rare_values = zip(*rare_interactions)
    plt.figure(figsize=(10, 4))
    # 生成渐变色（以橙色为基础，从浅到深）
    cmap = plt.get_cmap('Purples')
    colors = [cmap(i / (len(rare_labels)-1 if len(rare_labels)>1 else 1)) for i in range(len(rare_labels))]
    bars = plt.bar(range(len(rare_labels)), rare_values, color=colors)
    font_prop = FontProperties(style='italic')
    plt.xticks(range(len(rare_labels)), rare_labels, rotation=88, fontproperties=font_prop)
    plt.xlabel('Interaction(in dataset order)')
    plt.yticks(range(0, max(rare_values)+1, 1))
    plt.ylabel('Count')
    plt.title('Rare Interaction (Count <= 3)')
    plt.tight_layout()
    plt.xlim(-0.5, len(rare_labels)-0.5)
    plt.show()
else:
    print("没有出现次数小于等于3的交互类别。")