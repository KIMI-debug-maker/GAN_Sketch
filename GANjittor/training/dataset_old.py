import random
import pathlib
from PIL import Image
from jittor import transform
from jittor import dataset

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class ImagePathDataset(dataset.Dataset):
    def __init__(self, path, image_mode='L', tran=None, max_images=None):
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        if max_images is None:
            self.files = files
        elif max_images < len(files):
            self.files = random.sample(files, max_images)
        else:
            print(f"max_images larger or equal to total number of files, use {len(files)} images instead.")
            self.files = files
        self.transformer = tran
        self.image_mode = image_mode

    def __getitem__(self, index):
        image_path = self.files[index]
        image = Image.open(image_path).convert(self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
            image=transform.to_tensor(image),
        return image

    def __len__(self):
        return len(self.files)

def create_dataloader(data_dir, size, batch, img_channel=3):
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]

    transformer = transform.Compose([
            transform.Resize(size),
            transform.ImageNormalize(mean=[0.5], std=[0.5]),
        ])

    if img_channel == 1:
        image_mode = 'L'
    elif img_channel == 3:
        image_mode = 'RGB'
    else:
        raise ValueError("image channel should be 1 or 3, but got ", img_channel)
    
    loader = dataset.ImageFolder(data_dir,transform=transformer).set_attrs(batch_size=batch, shuffle=True, drop_last=True)
    return loader

def yield_data(loader):
    epoch = 0
    while True:
        for batch in loader:
            yield batch
        epoch += 1

fuck=create_dataloader('../data/image', 256, 16, img_channel=3)
yield_data(fuck)