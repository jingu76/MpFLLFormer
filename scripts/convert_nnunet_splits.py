from dataset.brats_dataset import datafold_read
import argparse
import os
import json


def main(args):
    with open(args.src_path) as f:
        data = json.load(f)
    result = []
    for i in range(5):
        temp = {"train":[], "val":[]}
        for x in data['training']:
            name = x['image'].split('volume-')[-1][:-4]
            fold_idx = x['fold']
            if fold_idx == i:
                temp['val'].append(name)
            else:
                temp['train'].append(name)
        result.append(temp)
    with open(args.target_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hi_measure")
    parser.add_argument('--src_path', type=str, default="/data0/yangcunyuan/datasets/lits_preprocess.json")
    parser.add_argument('--target_path', type=str, default="/data0/yangcunyuan/datasets/nnUNet/preprocessed/Dataset003_LiTS/splits_final.json")
    args = parser.parse_args()
    main(args)