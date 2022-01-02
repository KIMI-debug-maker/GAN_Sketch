import numpy as np
from . import pix2pix
from .misc import set_requires_grad
import jittor as jt
from jittor import nn
#from .diffaug import DiffAugment
class OutputTransform(nn.Module):
    def __init__(self, opt, process='', diffaug_policy=''):
        super(OutputTransform, self).__init__()
        self.opt = opt
        self.process=process
        if diffaug_policy == '':
            self.augment = None
        else:
            self.augment = DiffAugment(policy=diffaug_policy)

        transforms = []
        process = process.split(',')
        for p in process:
            if p == 'to3ch':
                pass
            elif p == 'toSketch':
                sketch = self.setup_sketch(opt)
                transforms.append(sketch)
            else:
                ValueError("Transforms contains unrecognizable key: %s" % p)
        self.transforms = nn.Sequential(*transforms)

    def setup_sketch(self, opt):
        ###all is torch method in this func!
        sketch = pix2pix.ResnetGenerator(3, 1, n_blocks=9, use_dropout=False)

        state_dict = jt.load(opt.photosketch_path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        sketch.load_state_dict(state_dict)

        sketch.train()
        set_requires_grad(sketch.parameters(), False)
        return sketch

    def execute(self, img, apply_aug=True):
        img = self.transforms(img)
        if apply_aug and self.augment is not None:
            img = self.augment(img)
        return img
