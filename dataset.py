import os, config, numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as t

def add_white_background(image):
    """
    Converts an image with an alpha channel to RGB with a white background.
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and 'transparency' in image.info):
        # Create a white background image
        background = Image.new("RGB", image.size, (255, 255, 255))
        # Paste the image on the background. 
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background
    else:
        return image


class Class1Class2Dataset(Dataset):
    def __init__(self, root_c1, root_c2, transform=None):
        self.root_c1 = root_c1
        self.root_c2 = root_c2
        self.transform = transform

        self.class1_images = os.listdir(root_c1)
        self.class2_images = os.listdir(root_c2)
        
        self.class1_len = len(self.class1_images)
        self.class2_len = len(self.class2_images)
        
        self.length_dataset = max(self.class1_len, self.class2_len)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        class1_img = self.class1_images[index % self.class1_len]
        class1_path = os.path.join(self.root_c1, class1_img)
        
        
        class2_img = self.class2_images[index % self.class2_len]
        class2_path = os.path.join(self.root_c2, class2_img)

        if config.ADD_WHITE_BACKGROUND:
            class1_img = np.array(Image.open(class1_path)).convert('RGB')
            class2_img = np.array(Image.open(class2_path)).convert('RGB')
        else:
            class1_img = np.array(Image.open(class1_path)).convert('RGB')
            class2_img = np.array(Image.open(class2_path)).convert('RGB')

        if self.transform:
            augmentations = self.transform(image=class1_img, image0=class2_img)
            class1_img = augmentations['image']
            class2_img = augmentations['image0']

        return class1_img, class2_img

# save 5 augmented
def test():
    ccD = Class1Class2Dataset(config.C1_TRAIN_DIR, config.C2_TRAIN_DIR, config.transforms)
    print(len(ccD))
    print(ccD[0])
    transform = t.ToPILImage()
    for i in range(1, 6):
        (transform(ccD[i][0] * 0.5 + 0.5)).save(f'{i}_c1.png')
        transform(ccD[i][1] * 0.5 + 0.5).save(f'{i}_c2.png')


if __name__ == '__main__':
    test()
