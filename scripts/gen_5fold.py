"""
    对已经转换完成的数据集，生成decathlon challenge格式的json描述文件
"""
import argparse
import os
import json
import random

basic_info = {
    "training": [], # need
}

def get_list(root_dir, data_dir):
    results = []
    
    data_dir_abs = os.path.join(root_dir, data_dir)
    images_dir_abs = os.path.join(data_dir_abs, "scan")
    labels_dir_abs = os.path.join(data_dir_abs, "label")
    
    image_names = os.listdir(images_dir_abs)
    random.shuffle(image_names)
    
    for idx, image_name in enumerate(image_names):
        label_name = image_name.replace('CT', 'SEG').replace('volume', 'segmentation')
        label_path_abs = os.path.join(labels_dir_abs, label_name)
        if not os.path.exists(label_path_abs):
            raise Exception(f"{label_path_abs} not exist")
        
        image_path = os.path.join(data_dir, "scan", image_name)
        label_path = os.path.join(data_dir, "label", label_name)
        fold = int((idx)*5/len(image_names))
        results.append({"image":image_path, "label":label_path, "fold":fold})
    
    return results

def main(args):
    data = basic_info.copy()
    data["training"] = get_list(args.root_dir, args.train_dir)
    
    with open(args.output_path, "w") as f:
        json.dump(data, f)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--root_dir", default="/home/yangcunyuan/datasets", type=str, help="root directory")
    parser.add_argument("--train_dir", default="lits_train", type=str, help="input dataset directory")
    parser.add_argument("--output_path", default="/home/yangcunyuan/datasets/lits.json")
    args = parser.parse_args()
    main(args)