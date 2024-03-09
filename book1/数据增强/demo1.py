import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
#取图形,使用PIL格式
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
#中心裁剪
def center_crop(image):     #crop:裁剪
    #在中心裁剪出一个(300, 300)的矩形区域
    CenterCrop = transforms.CenterCrop(size = (300, 300))
    cropped_image = CenterCrop(image)
    return cropped_image

#r是一个特殊的前缀，表示后面的字符串是原始字符串
#确保字符串中的 \ 被视为字面意义上的反斜杠，而不是转义字符
img = read_PIL(r'./images/5.jpg')
print(img.size)
#os.makedirs:创建文件夹(exist_ok保证如果已存在文件夹则继续执行)
outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

center_cropped_image = center_crop(img)
center_cropped_image.save(os.path.join(outDir, 'center_cropped_image.jpg'))



