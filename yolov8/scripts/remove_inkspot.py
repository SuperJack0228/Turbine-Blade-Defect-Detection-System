# scripts/remove_inkspot.py
import os, json, shutil

# 要去掉的完整标签字符串
REMOVE_LABEL = "墨点"

# 原始 JSON 目录（如果之前已经集中到 data/labels/json_raw）
SRC_DIRS = {
    "train": "data/labels/json_raw/train",
    "val":   "data/labels/json_raw/val"
}
# 备份并写入新 JSON 到 data/labels/json_clean
DST_DIRS = {
    "train": "data/labels/json_clean/train",
    "val":   "data/labels/json_clean/val"
}

def clean_json(src, dst):
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if not fn.endswith(".json"):
            continue
        data = None
        # 兼容多种编码
        for enc in ('gbk','utf-8','latin-1'):
            try:
                data = json.load(open(os.path.join(src,fn), encoding=enc, errors='replace'))
                break
            except:
                continue
        if data is None:
            print(f"无法读取 {fn}")
            continue
        # 过滤 shapes
        shapes = data.get("shapes", [])
        new_shapes = [s for s in shapes if s.get("label","").split("_")[-1] != REMOVE_LABEL]
        if len(new_shapes) != len(shapes):
            print(f"{fn}: 删除 {len(shapes)-len(new_shapes)} 条 ‘{REMOVE_LABEL}’ 标注")
        data["shapes"] = new_shapes
        # 写回到新目录
        with open(os.path.join(dst,fn), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    for split in ("train","val"):
        clean_json(SRC_DIRS[split], DST_DIRS[split])
    print("JSON 清理完成，已输出到 data/labels/json_clean/")
