from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class ImageDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform

        for root, _, files in os.walk(self.main_dir):
            for f in files:
                fullpath = os.path.join(root, f)
                try:
                    if os.path.getsize(fullpath) < 100:
                        os.remove(fullpath)
                except WindowsError:
                    print("Error" + fullpath)


        self.total_imgs = os.listdir(main_dir)


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image_id = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, image_id

