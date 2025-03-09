import json
import numpy as np
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 设置后端为无图形界面的 'Agg'

# ========== 一些函数 ==========

def box_cxcywh_to_xyxy(boxes):
    """将 [cx, cy, w, h] 转为 [xmin, ymin, xmax, ymax]."""
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def rescale_bboxes(out_bbox, size):
    """
    将 DETR 输出的相对坐标 [0,1] 转回原图坐标 [xmin, ymin, xmax, ymax].
    out_bbox: shape (N,4) (cx, cy, w, h)
    size: (width, height)
    """
    img_w, img_h = size
    bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return bbox_xyxy * scale

def coco_box_to_xyxy(box):
    """
    COCO ann['bbox'] = [x, y, w, h] 转为 [x_min, y_min, x_max, y_max].
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]

def box_iou_xyxy(box1, box2):
    """
    计算 box1 与 box2 的 IoU. 均为 xyxy: [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

# ========== 1. 数据和模型 ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.1 加载 COCO
coco_annotation_file = r"F:\IRPMODEL\detr-main-test\data\coco\annotations\instances_val2017.json"
coco = COCO(coco_annotation_file)

# 1.2 定义类别
# 包括 0~3 这四个真正类别 + 第 4 类 "UNMATCHED" 用来记录漏检/误检
CLASSES = ["N/A", "panel", "body", "line", "UNMATCHED"]
num_cls = len(CLASSES)  # = 5

# 1.3 图像预处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# 1.4 构建模型 + 加载权重
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
model.class_embed = nn.Linear(model.transformer.d_model, num_cls)  # 这里 num_cls=5
ckpt_path = "data/output/checkpoint.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# ========== 2. 混淆矩阵初始化 ==========

cmatrix = np.zeros((num_cls, num_cls), dtype=np.int32)

# ========== 3. 遍历图片并做匹配 ==========

image_ids = coco.getImgIds()

conf_thresh = 0.7  # 预测框置信度阈值
iou_thresh = 0.5   # IoU 匹配阈值

for img_id in image_ids:
    # --- 3.1 读取图像
    img_info = coco.loadImgs(img_id)[0]
    width, height = img_info["width"], img_info["height"]
    filename = img_info["file_name"]
    img_path = f"F:\\IRPMODEL\\detr-main-test\\data\\coco\\val2017\\{filename}"
    img_pil = Image.open(img_path).convert("RGB")

    # --- 3.2 前向推理
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)

    # --- 3.3 获取预测框 + 类别 (过滤置信度)
    pred_logits = outputs["pred_logits"][0]  # [100, num_cls]
    pred_boxes  = outputs["pred_boxes"][0]   # [100, 4], cxcywh

    # softmax 去掉最后一类(通常是 no_object)也可保留看你训练怎么设置
    # 这里假设你把最后一类当 no_object, 仅取 [:, :-1]
    # 注意这跟你实际数据定义相关, 需自行核对
    probs = F.softmax(pred_logits, dim=-1)[:, :-1]
    max_probs, max_labels = probs.max(dim=-1)  # 每个 query 最可能的类别(不含 no_object)
    keep = (max_probs > conf_thresh)
    max_probs = max_probs[keep]
    max_labels = max_labels[keep].cpu().numpy()

    # 缩放到原图
    pred_boxes = pred_boxes[keep].cpu()
    pred_boxes_xyxy = rescale_bboxes(pred_boxes, (width, height)).numpy()  # [K,4]

    # --- 3.4 读取真实框
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    gt_boxes_xyxy = []
    gt_classes = []
    for ann in anns:
        # COCO: [x, y, w, h]
        x, y, w, h = ann['bbox']
        x2 = x + w
        y2 = y + h
        gt_boxes_xyxy.append([x, y, x2, y2])

        # category_id 可能不是 [0,3], 需映射
        c_id = ann["category_id"]
        # 如果你的数据标注是 1->panel, 2->body, 3->line, etc.,
        # 需要减1 或别的映射. 此处仅做示例:
        if c_id < 0 or c_id > 3:
            c_id = 0  # 把不识别的类当成 "N/A"
        gt_classes.append(c_id)

    gt_boxes_xyxy = np.array(gt_boxes_xyxy)
    gt_classes = np.array(gt_classes)

    # --- 3.5 贪心 IoU 匹配
    used_pred = set()
    for i, gbox in enumerate(gt_boxes_xyxy):
        gcls = gt_classes[i]
        best_iou = 0.0
        best_j = -1
        for j, pbox in enumerate(pred_boxes_xyxy):
            if j in used_pred:
                continue
            iou_val = box_iou_xyxy(gbox, pbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        # 判断是否匹配成功
        if best_j >= 0 and best_iou >= iou_thresh:
            used_pred.add(best_j)
            pcls = max_labels[best_j]
            cmatrix[gcls, pcls] += 1
        else:
            # 漏检 -> (gcls, "UNMATCHED")
            cmatrix[gcls, 4] += 1

    # --- 3.6 剩余未匹配的预测框是误检 -> ("UNMATCHED", pcls)
    for j in range(len(pred_boxes_xyxy)):
        if j not in used_pred:
            pcls = max_labels[j]
            cmatrix[4, pcls] += 1

# ========== 4. 可视化混淆矩阵 ==========

# plt.figure(figsize=(8,6))
# sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=CLASSES, yticklabels=CLASSES)
# ======= 原来生成的 cmatrix 是 num_cls=5 大小 (含 N/A) =======
# cmatrix = np.zeros((num_cls, num_cls), dtype=np.int32)
# ... 省略中间匹配过程 ...
# 此时 cmatrix 形状是 (5, 5)，行列顺序都对应 ["N/A", "panel", "body", "line", "UNMATCHED"]

# ======= 截掉 N/A 对应的行、列 =======
# 这里假设第一项 "N/A" 在索引 0 处
filtered_cmatrix = cmatrix[1:, 1:]  # 去掉第 0 行和第 0 列
filtered_cmatrix = filtered_cmatrix[:-1, :-1]

# 更新类名，去掉 "N/A"
filtered_classes = CLASSES[1:]  # ["panel", "body", "line", "UNMATCHED"]
filtered_classes = filtered_classes[:-1]
plt.rcParams['font.sans-serif'] = ['SimHei']
# 替换类别标签为中文
filtered_classes = ["太阳能板", "主体", "天线"]  # 根据你的实际类别名称修改

# ======= 绘制图像 =======
plt.figure(figsize=(8,6))
ax=sns.heatmap(filtered_cmatrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=filtered_classes, yticklabels=filtered_classes,annot_kws={"size": 22},  # 调整矩阵中数字的字体大小
            cbar=True)
# 调整 colorbar 刻度字体大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('预测',fontsize=20)
plt.ylabel('实际',fontsize=20)
plt.title('基于 IoU 匹配的混淆矩阵',fontsize=22)

save_path = r"F:\IRPMODEL\output_charts\confusion_matrix.png"
plt.savefig(save_path)
plt.show()

print("Done. Confusion matrix saved to:", save_path)
