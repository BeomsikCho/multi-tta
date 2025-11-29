# Test Time Adaptation with Multi Types of Domain Shift

This repostiory is designed to implement unified framework addressing multi types of domain shift problem.

## Baseline

### Model
The baseline of our methods is as follows:
* microsoft/resnet-50 [[paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)] 
* ViT [[paper](https://arxiv.org/abs/2010.11929)]


### Dataset
We train/validate/test with the below datasets
* ImageNet-C [[paper](https://openreview.net/forum?id=HJz6tiCqYm&hl=es)] [[download](https://zenodo.org/records/2235448)]
* ImageNet-LT [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.html)] [[github](https://github.com/zhmiao/OpenLongTailRecognition-OLTR?tab=readme-ov-file)] [[download](https://drive.google.com/uc?export=download&id=0B7fNdx_jAqhtckNGQ2FLd25fa3c)]
* VisDA-C [[paper](https://arxiv.org/abs/1710.06924)] [[github](https://github.com/VisionLearningGroup/taskcv-2017-public)] [[download](https://ai.bu.edu/visda-2017/)]
* OfficeHome [[paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.html)] [[github](https://github.com/hemanthdv/da-hash)] [[download](https://www.hemanthdv.org/officeHomeDataset.html)]
* DomainNet [[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html)] [[github](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit)] [[download](https://ai.bu.edu/M3SDA/)]


## Setup
1. Download the above [datasets](#dataset) and place on same folder named __data__ as follows:
```zsh
data/
├── ImageNet-C/ 
    ├── images/ 
    └── labels.csv
├── ImageNet-LT/
    ├── images/
    └── labels.csv
├── VisDA-C/
`   ├── train/
    ├── validation/
    └── test/
├── OfficeHome/
    ├── Art/
    ├── Clipart/
    ├── Product/
    └── Real_World/
└── DomainNet/
    ├── images/
    └── labels.csv
```

2. Install the Requirement Libraries
    
    First, Install the python >= 3.12
    ```bash
    conda create --name multiTTA python=3.12
    ```
    
    Second, Install Pytorch >= 2.4
    ```bash
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    ```

    Third, Install the requirements
    ```bash
    pip install -U -r requirements.txt 
    ```

## Train
To train your model you should instruct the below code to run main.py  

```bash
python3 main.py --config [path/to/your/config_file] --mode train --device cuda:[gpu_number]
```



## Test