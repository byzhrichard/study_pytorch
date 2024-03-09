import PIL.Image as Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
def erase_image(image, position, size):
    img = TF.to_tensor(image)
    erase_image = TF.to_pil_image(TF.erase(img=img,
                                           i=position[0],
                                           j=position[1],
                                           h=size[0],
                                           w=size[1],
                                           v=1))
    return erase_image

img = read_PIL(r'./images/5.jpg')
print(img.size)

outDir = r'./images/result'
os.makedirs(outDir, exist_ok=True)

erase_image = erase_image(img, (100, 100), (50, 200))
erase_image.save(os.path.join(outDir, 'erased_image.jpg'))

