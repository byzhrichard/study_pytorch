import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image

def random_crop(image):
    RandomCrop = transforms.RandomCrop(size = (200, 200))
    random_image = RandomCrop(image)
    return random_image

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

randown_cropped_image = random_crop(img)
randown_cropped_image.save(os.path.join(outDir, 'random_cropped_image.jpg'))




