import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchmetrics.image.inception import InceptionScore

# class to load images from a directory
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# path to directory with images
generated_images_dir = "generated_images"
image_dataset = ImageDataset(generated_images_dir)

# process data using batches
batch_size = 32
data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

# compute the inception score
inception_score = InceptionScore(normalize=False)
for batch in data_loader:
    inception_score.update(batch)

score = inception_score.compute()
print("Inception Score:", score[0])
