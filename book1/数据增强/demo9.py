import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
def pad(image):
    pad_image = transforms.Pad( (0, (image.size[0]-image.size[1])//2) )(image)
    return pad_image

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

pad_image = pad(img)
pad_image.save(os.path.join(outDir, 'pad_image.jpg'))

