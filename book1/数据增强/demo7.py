import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
def BCSH_transform(image):
    img = transforms.ColorJitter(hue=0.4)(image)        # 色度
    img = transforms.ColorJitter(brightness=1)(img)     # 亮度
    img = transforms.ColorJitter(saturation=0.6)(img)   # 饱和度
    img = transforms.ColorJitter(contrast=1)(img)       # 对比度
    return img

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

bcsh_image = BCSH_transform(img)
bcsh_image.save(os.path.join(outDir, 'bcsh_image.jpg'))

