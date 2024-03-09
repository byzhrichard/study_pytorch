import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
def random_rotation(image):
    #顺or逆 旋转10-80度
    RR = transforms.RandomRotation(degrees=(10,80))
    rr_image = RR(image)
    return rr_image

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

rr_image = random_rotation(img)
rr_image.save(os.path.join(outDir, 'rr_image.jpg'))



