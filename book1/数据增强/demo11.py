import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image

def gamma_transform(image, gamma_value):
    gamma_value = TF.adjust_gamma(img=image,gamma=gamma_value)
    return gamma_value

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

gamma_image = gamma_transform(img,0.1)
gamma_image.save(os.path.join(outDir, 'gamma_image.jpg'))

