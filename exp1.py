# 调用包从图像识别包中调用，图像标签,工具包
from image_sdk.utils import encode_to_base64
from image_sdk.image_tagging import image_tagging_aksk
from image_sdk.utils import init_global_env
# 调用json解析传回的结果
import json
# 操作系统文件/文件夹的包
import os
import shutil
# 图像处理展示相关的包

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

init_global_env('cn-north-4')
# 准备ak,sk
app_key = "OWZJHZ6MGYURXYVKQOLC"
app_secret = "m7DqODSvMLGnjIPKqLgb8ugkVBf08hO1SrQps1j4"

# # 使用网络图像测试
# demo_data_url = 'https://sdk-obs-source-save.obs.cn-north-4.myhuaweicloud.com/tagging-normal.jpg'  # call interface use the url
# result = image_tagging_aksk(app_key, app_secret, '', demo_data_url, 'zh', 5, 30)  # check "
# # 转化成Python字典形式
# tags = json.loads(result)
# print(tags)

# 确定电子相册位置
file_path = 'data/'
# 保存图片标签的字典
labels = {}

items = os.listdir(file_path)
for i in items:
    # 判断是否为文件，而不是文件夹
    if os.path.isfile:
        # 华为云El目前支持JPG/PNG/BMP格式的图片
        if i.endswith('jpg') or i.endswith('jpeg') or i.endswith('bmp') or i.endswith('png'):
            # 为图片打上标签
            result = image_tagging_aksk(app_key, app_secret, encode_to_base64(file_path + i), '', 'zh', 5, 60)
            # 解析返回的结果
            result_dic = json.loads(result)
            # 将文件名与图片对齐
            labels[i] = result_dic['result']['tags']
# 显示结果
print(labels)
# 将标签字典保存到文件
save_path = './label'
# 如果文件夹不存在则创建文
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 创建文件,执行写入操作，并关闭
with open(save_path + '/labels.json', 'w+') as f:
    f.write(json.dumps(labels))

# 打开刚刚保存的文件
label_path = 'label/labels.json'
with open(label_path, 'r') as f:
    labels = json.load(f)

# 搜索关键词
key_word = input('请输入搜索词')
# 设置可信百分比
threshold = 60
# 设置一个集合(集合内只存在唯一的元素)
valid_list = set()
# 遍历labels 中的字典获取所有包含关键字的图片名字
for k, v in labels.items():
    for item in v:
        if key_word in item['tag'] and float(item['confidence']) >= threshold:
            valid_list.add(k)
# 展示结果
valid_list = list(valid_list)
print(valid_list)

# 生成一个临时文件夹if not os.path.exists('tmp'):
os.mkdir('tmp')
# 将所有搜索到的图像转化为gif格式，并存储在临时文件夹中
gif_list = []
for k, pic in enumerate(valid_list):
    pic_path = 'data/' + pic
    img = Image.open(pic_path)
    img = img.resize((640, 380))
    save_name = 'tmp/' + str(k) + '.gif'
    img.save(save_name)
    gif_list.append(save_name)
# 打开已经所有静止的gif图片
images = []
for i in gif_list:
    pic_path = i
    images.append(Image.open(pic_path))
# 存储成动图gif
images[0].save('相册动图.gif',
               save_all=True,
               append_images=images[1:],
               duration=1000,
               loop=0)
# 释放内存
del images
# 删除临时文件夹
shutil.rmtree('tmp')
print('gif 相册制作完成')

# 打开保存的labels文件
label_path = 'label/labels.json'
with open(label_path, 'r') as f:
    labels = json.load(f)
print(labels)
# 获取置信度最高的文件分类
classes = [[v[0]['tag'], k] for k, v in labels.items()]
classes

for cls in classes:
    if not os.path.exists('data/' + cls[0]):
        os.mkdir('data/' + cls[0])
    # 复制被对应的图片
    shutil.copy('data/' + cls[1], 'data/' + cls[0] + '/' + cls[1])
print('已完成移复制!')
