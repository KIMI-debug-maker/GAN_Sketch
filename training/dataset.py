from jittor import transform
from jittor import dataset

def create_dataloader(data_dir, size, batch):

    transformer = transform.Compose([
        transform.Resize(size),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])

    loader = dataset.ImageFolder(data_dir,transform=transformer).set_attrs(batch_size=batch, shuffle=True, drop_last=False)
    
    show=False
    if show:
        for a,b in loader:
            print(a)
            break
    #a->pic,channel_first,b->label   

    return loader