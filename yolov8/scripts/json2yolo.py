# scripts/json2yolo.py
import os
import json

"""
将 json_clean 下清理后的 JSON 标注批量转换为 YOLO 格式的 txt 标签文件。
- 从 data/labels/json_clean/{train,val} 读取 JSON
- 标签类别：表面划伤、表面磕伤、表面污渍、表面压痕
- pts 可为 dict(xmin,ymin,xmax,ymax) 或 list[[x,y],...]
- 输出到 data/labels/{train,val} 同名 .txt
"""

# 定义类别列表，已去除 "表面墨点"
CLASSES = ['表面划伤', '表面磕伤', '表面污渍', '表面压痕']

# 加载 JSON 文件，优先尝试 UTF-8，再 GBK、Latin-1
def load_json(path):
    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, errors='replace') as f:
                return json.load(f)
        except Exception:
            continue
    raise ValueError(f"无法用 utf-8/gbk/latin-1 打开 {path}")

# 单文件转换
def convert_json(json_path, txt_path):
    data = load_json(json_path)
    w = data.get('imageWidth') or data.get('imgWidth')
    h = data.get('imageHeight') or data.get('imgHeight')
    if not w or not h:
        print(f"跳过 {os.path.basename(json_path)}：无尺寸信息")
        return

    lines = []
    for shape in data.get('shapes', []):
        raw = shape.get('label', '')
        matched = next((cls for cls in CLASSES if raw.endswith(cls)), None)
        if not matched:
            continue
        cls_id = CLASSES.index(matched)

        pts = shape.get('points')
        # 支持 dict 或 list 两种格式
        if isinstance(pts, dict):
            xs = [pts.get('xmin',0), pts.get('xmax',0)]
            ys = [pts.get('ymin',0), pts.get('ymax',0)]
            x1, x2 = sorted(xs)
            y1, y2 = sorted(ys)

        elif isinstance(pts, list):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            continue

        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # 写入 txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# 主流程
if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for split in ['train', 'val']:
        json_dir = os.path.join(base_dir, f'data/labels/json_clean/{split}')
        txt_dir  = os.path.join(base_dir, f'data/labels/{split}')
        os.makedirs(txt_dir, exist_ok=True)

        for fname in os.listdir(json_dir):
            if not fname.lower().endswith('.json'):
                continue
            name      = os.path.splitext(fname)[0]
            json_path = os.path.join(json_dir, fname)
            txt_path  = os.path.join(txt_dir, name + '.txt')
            convert_json(json_path, txt_path)

    print('JSON → YOLO txt 转换完成！')
