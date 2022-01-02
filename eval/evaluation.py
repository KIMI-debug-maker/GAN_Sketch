import os
import numpy as np
import jittor as jt
from cleanfid import fid
from run_metrics import make_eval_images

class Evaluator():
    def __init__(self, opt, gan_model):
        self.opt = opt
        self.gan_model = gan_model
        self.fid_stat = get_fid_stats(opt.eval_dir, opt.eval_batch)
        # record the best fid so far
        self.best_fid = float('inf')
        if opt.resume_iter is not None:
            load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "best_FID.npy")
            self.best_fid = float(np.load(load_path))

    def run_metrics(self, iters):
        print("Running metrics and gathering images...")
        cache_folder = f'cache_files/{self.opt.name}'

        print("Gathering images...")
        with jt.no_grad():
            make_eval_images(self.gan_model.netG,
                             cache_folder,
                             2500,
                             self.opt.eval_batch)
            print('images prepared')
            metrics = metrics_process(cache_folder, self.fid_stat)
        best_so_far = False
        if metrics['fid'] < self.best_fid:
            self.best_fid = metrics['fid']
            save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            np.save(save_path + "/best_FID.npy", self.best_fid)
            with open(save_path + "/best_iter.txt", 'w') as f:
                f.write('%d\n' % iters)
            best_so_far = True

        print("Ran metrics successfully...")
        return metrics, best_so_far

# return metrics
def metrics_process(cache_folder, fid_stats):
    metrics = {}
    print("Evaluating FID...")
    metrics['fid'] = fid.compute_fid(cache_folder+'/image/', num_workers=0, dataset_name=fid_stats, dataset_split="custom")
    print('fid test:'+str(fid))
    return metrics

# get fid data
def get_fid_stats(eval_dir, eval_batch):
    fid_stat = os.path.basename(eval_dir.rstrip('/')) + '_image'
    if not fid.test_stats_exists(fid_stat, 'clean'):
        fid.make_custom_stats(fid_stat, eval_dir + '/image/', batch_size=eval_batch)
    return fid_stat