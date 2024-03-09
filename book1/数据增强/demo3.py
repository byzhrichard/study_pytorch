import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
def resize(image):
    Resize = transforms.Resize(size=(100,150))
    resized_image = Resize(image)
    return resized_image

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

resized_image = resize(img)
resized_image.save(os.path.join(outDir, 'resized_image.jpg'))

