import os
from collections import Counter

label_dir = 'data/labels/train'  # 修改为你的标签路径

if not os.path.exists(label_dir):
    print(f"标签文件夹不存在: {label_dir}")
    exit(1)

files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
print(f"发现 {len(files)} 个标签文件。")

class_counter = Counter()
empty_files = 0

for file in files:
    file_path = os.path.join(label_dir, file)
    with open(file_path, 'r') as f:
        found_label = False
        for line in f:
            if line.strip():  # 非空行
                class_id = int(line.strip().split()[0])
                class_counter[class_id] += 1
                found_label = True
        if not found_label:
            empty_files += 1

print(f"空标签文件数量: {empty_files}")
if not class_counter:
    print("没有发现任何目标标签，请检查标签内容是否正确！")
else:
    names = ['表面划伤', '表面磕伤', '表面污渍', '表面压痕']
    for class_id, name in enumerate(names):
        print(f"{name}（类别{class_id}）: {class_counter[class_id]}")
