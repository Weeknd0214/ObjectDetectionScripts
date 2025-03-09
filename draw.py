
import math
import numpy as np

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from torch.nn.functional import dropout,linear,softmax
torch.set_grad_enabled(False)
import matplotlib
matplotlib.use('Agg')  # 设置后端为无图形界面的 'Agg'

 
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
 
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
 
# COCO classes
CLASSES = [
     'N/A','panel', 'body', 'line'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
 
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


###########################load model



# 加载线上的模型
# model = torch.load('output/checkpoint.pth', 'detr_resnet50', pretrained=True)
# model.eval()
# 首先，定义模型架构
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
num_classes = 7 # 你的类别数，修改为3
model.class_embed = nn.Linear(model.transformer.decoder.layers[0].self_attn.embed_dim, num_classes)
# 加载你自己训练的权重
checkpoint = torch.load('output/checkpoint.pth')
model.load_state_dict(checkpoint['model'])  # 加载权重

model.eval()
for name, parameters in model.named_parameters():
    # 获取训练好的object queries，即pq:[100,256]
    if name == 'query_embed.weight':
        pq = parameters
    # 获取解码器的最后一层的交叉注意力模块中q和k的线性权重和偏置:[256*3,256]，[768]
    if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_weight':
        in_proj_weight = parameters
    if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_bias':
        in_proj_bias = parameters
###

print(name)

# --------------------------------------------2.下载图像并进行预处理和前馈过程--------------------------------------------------
# 线上下载图像
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# im = Image.open(requests.get(url, stream=True).raw)
mg_path = '/home/xiaolongbing/detr/detr-main-test/detr-main-test/data/test/image36.png'
im = Image.open(mg_path)
 
# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)
 
# propagate through the model
outputs = model(img)
 
# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.5
 
# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# ------------------------------------------------3. 准备存储前馈该图片时的值---------------------------------------------------
# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []
cq = []     # 存储detr中的 cq
pk =  []    # 存储detr中的 encoder pos
memory = [] # 编码器最后一层的输入/解码器的输入特征
 
# 注册hook
hooks = [
    # 获取resnet最后一层特征图
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    # 获取encoder的图像特征图memory
    model.transformer.encoder.register_forward_hook(
        lambda self, input, output: memory.append(output)
    ),
    # 获取encoder的最后一层layer的self-attn weights
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    # 获取decoder的最后一层layer中交叉注意力的 weights
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
    # 获取decoder最后一层self-attn的输出cq
    model.transformer.decoder.layers[-1].norm1.register_forward_hook(
        lambda self, input, output: cq.append(output)
    ),
    # 获取图像特征图的位置编码pk
    model.backbone[-1].register_forward_hook(
        lambda self, input, output: pk.append(output)
    ),
]
 
# propagate through the model
outputs = model(img)
 
# 用完的hook后删除
for hook in hooks:
    hook.remove()
 
# don't need the list anymore
conv_features = conv_features[0]       # [1,2048,25,34]
enc_attn_weights = enc_attn_weights[0] # [1,850,850]   : [N,L,S]
dec_attn_weights = dec_attn_weights[0] # [1,100,850]   : [N,L,S] --> [batch, tgt_len, src_len]
memory = memory[0] # [850,1,256] # 编码器最后一层的输入/解码器的输入特征
 
cq = cq[0]    # decoder的self_attn:最后一层输出[100,1,256]
pk = pk[0]    # [1,256,25,34]

# ----------------------------------------4， 求attn_output_weights以绘制各个head的注意力权重------------------------------------
pk = pk.flatten(-2).permute(2,0,1)           # [1,256,850] --> [850,1,256]
pq = pq.unsqueeze(1).repeat(1,1,1)           # [100,1,256]
q = pq + cq
 
k = pk
 
# 将q和k完成线性层的映射，代码参考自nn.MultiHeadAttn()
_b = in_proj_bias
_start = 0
_end = 256
_w = in_proj_weight[_start:_end, :]
if _b is not None:
    _b = _b[_start:_end]
q = linear(q, _w, _b)
 
_b = in_proj_bias
_start = 256
_end = 256 * 2
_w = in_proj_weight[_start:_end, :]
if _b is not None:
    _b = _b[_start:_end]
k = linear(k, _w, _b)
 
scaling = float(256) ** -0.5
q = q * scaling
q = q.contiguous().view(100, 8, 32).transpose(0, 1)
k = k.contiguous().view(-1, 8, 32).transpose(0, 1)
attn_output_weights = torch.bmm(q, k.transpose(1, 2))
 
attn_output_weights = attn_output_weights.view(1, 8, 100, 1125)
attn_output_weights = attn_output_weights.view(1 * 8, 100, 1125)
attn_output_weights = softmax(attn_output_weights, dim=-1)
attn_output_weights = attn_output_weights.view(1, 8, 100, 1125)
 
# 后续可视化各个头
attn_every_heads = attn_output_weights # [1,8,100,850]
attn_output_weights = attn_output_weights.sum(dim=1) / 8 # [1,100,850]

# ----------------------------------------------------------5. 画图---------------------------------------------------------
h, w = conv_features['0'].tensors.shape[-2:]
 
fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=10, figsize=(22, 28))  # [11,2]
colors = COLORS * 100
if len(bboxes_scaled) == 0:
    print("none")
    exit()
# 可视化
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    if len(bboxes_scaled) == 1:
        ax_i = axs
    else:
        ax_i = axs[:,idx]
     
    # 可视化decoder的注意力权重
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}',fontsize = 30)
 
 
    # 可视化框和类别
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[probas[idx].argmax()],fontsize = 30)
 
 
    # 分别可视化8个头部的位置特征图
    for head in range(2, 2 + 8):
        ax = ax_i[head]
        ax.imshow(attn_every_heads[0, head-2, idx].view(h,w))
        ax.axis('off')
        ax.set_title(f'head:{head-2}',fontsize = 30)
 
plot_path = "output_charts/zhuyili1.png"
plt.savefig(plot_path)
