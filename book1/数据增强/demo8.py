import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
def random_gray(image):
    gray_image = transforms.RandomGrayscale(p=0.5)(image)
    return gray_image

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

random_gray_image = random_gray(img)
random_gray_image.save(os.path.join(outDir, 'random_gray_image.jpg'))

