import argparse
import numpy as np
import os
from skimage import measure
from multiprocessing.dummy import Pool
from functools import partial
from tqdm.contrib.concurrent import process_map
import SimpleITK as sitk

EPS = 1e-10

CLASS_NAMES = ['TC', 'WT', 'ET']

def safe_divide(a, b):
    return (a + EPS) / (b + EPS)

# Dice系数
def cal_dsc(pre, gt):        
    intersection = np.count_nonzero(pre * gt)
    dsc = safe_divide(2 * intersection, np.count_nonzero(pre) + np.count_nonzero(gt))
    return dsc

def load(path):
    nii = sitk.ReadImage(path)
    label = sitk.GetArrayFromImage(nii)
    return label

def get_class(label, class_id):
    if class_id == 0:
        return (label == 1) | (label == 4)
    elif class_id == 1:
        return (label == 1) | (label == 4) | (label == 2)
    elif class_id == 2:
        return (label == 4)

class MetricValues:

    def __init__(self):
        self.dsc = np.zeros((len(CLASS_NAMES)))
        self.case_num = 0

    def update(self, obj):
        for k, v in self.__dict__.items():
            sum = v + getattr(obj, k)
            self.__setattr__(k, sum)

    def __repr__(self) -> str:
        str = 'MetricValues:{'
        for k, v in self.__dict__.items():
            str += f'{k}:{v}, '
        str += '}'
        return str
    
    def print_result(self):
        dsc_msg = ""
        for i in range(len(CLASS_NAMES)):
            dsc_i = safe_divide(self.dsc[i], self.case_num)
            dsc_msg += f'dsc({CLASS_NAMES[i]}):{dsc_i:.4f} '
        dsc_msg += f'dsc(mean): {np.mean(self.dsc)/self.case_num:.4f}'
        print(dsc_msg)


def run(args, filename):
    metric = MetricValues()
    metric.case_num = 1
    gt_path = os.path.join(args.gt_dir, filename)
    pre_path = os.path.join(args.pre_dir, filename)
    if not os.path.exists(pre_path):
        print(f'{filename} not exist in predictions')
        return metric
    gt = load(gt_path)
    pre = load(pre_path)
    for i in range(len(CLASS_NAMES)):
        gt_i = get_class(gt, i)
        pre_i = get_class(pre, i)
        metric.dsc[i] = cal_dsc(pre_i, gt_i)
    return metric
    
def main(args):
    metric = MetricValues()
    
    pool = Pool(args.thread_num)
    filenames = os.listdir(args.pre_dir)
    filenames = [name for name in filenames if name.endswith('npz') or name.endswith('.nii.gz')]
    results = process_map(partial(run, args), filenames, max_workers=args.thread_num)
    pool.close()
    pool.join()
    
    for result in results:
        metric.update(result)
    print(metric)
    metric.print_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hi_measure")
    parser.add_argument('--pre_dir', type=str, default=r'/data4/sanleizheng_output/ceshi_guangxi')
    parser.add_argument('--gt_dir', type=str, default=r"/data4/sanleizheng/ceshi_guangxi")
    parser.add_argument('--series_num', type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument('--thread_num', type=int, default=20)
    args = parser.parse_args()
    
    main(args)
