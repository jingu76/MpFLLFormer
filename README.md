# Background

The project is the code for paper:  
    Region-based Fusion Transformer for Focal Liver Lesion Segmentation

Multi-phase computed tomography images provide crucial complementary information 
for accurate liver tumor segmentation. State-of-the-art multi-phase LiTS methods 
typically fuse cross-phase features using phase-weighted attention based on 
concatenation or summation. However, these methods neglect the spatial regional 
relationships between different phases, leading to insufficient feature integration.
To address this issue, we propose the region-based aggregation fusion block to fuse 
feature maps between the prediction phase and auxiliary phases at the regional level. 
In addition, the performance of existing methods are still limited by insufficient 
feature extraction, we propose residual shift windows transformer within sliding 
widow to exact feature.
In this work, we introduce a novel method, MpFLLFormer, for focal liver lesion 
segmentation. We have conducted extensive testing on both brain and liver datasets.
Our approach demonstrates an improvement in Dice similarity coefficient segmentation 
accuracy of 2.53\% on our in-house MPFLLsDS dataset and 0.15\% on the BraTS public
dataset when compared to the most recent state-of-the-art methods, respectively.

# Structure

- algorithms: 
    - method-name
        - network
        - train
        - test
- dataset:
- loss: loss function
- optimizer:
- scripts: 
- utils: Store some general utility class codes, such as index calculation, post-processing, etc.
- assets: 

# Installation

 install torch 1.13.1, refer to torch official site

```
pip install -r requirements.txt
pip install 'monai[all]'
pip install monai==1.2.0
```


# How to train

## single phase train
```
  python ./MpFLLFormer/train/main_single.py
```

## four phase train
```
  python ./MpFLLFormer/train/main_four.py
```

# How to test

```
  python ./script/hi_measure.py
```

