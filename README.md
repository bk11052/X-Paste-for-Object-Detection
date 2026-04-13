# X-Paste, ICML 2023

The repo is the official implementation of ["X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion"](https://arxiv.org/abs/2212.03863).

## Introduction

![X-Paste Pipeline](pipleline_figure.png)
  X-Paste is built upon Copy-Paste to train the instance segmentation model but aims to make Copy-Paste more scalable, i.e., obtain large-scale object instances with high-quality masks for unlimited categories in an efficient and automatic way. 

## Requirements

```
pip install -r requirements.txt
```

Download [COCO](https://cocodataset.org/#download) and  [LVIS](https://www.lvisdataset.org/dataset) dataset, place them under $DETECTRON2_DATASETS following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/datasets)


Download pretrained backbone 
```
mkdir models
cd models
wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth
python tools/convert-thirdparty-pretrained-model-to-d2.py --path resnet50_miil_21k.pth

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
python tools/convert-thirdparty-pretrained-model-to-d2.py --path swin_base_patch4_window7_224_22k.pth
```

## Getting Started 
1. generate images with stablediffusion: generation/text2im.py

```
cd generation
pip install -U diffusers transformers xformers
python text2im.py --model diffusers --samples 100 --category_file /mnt/data/LVIS/lvis_v1_train.json --output_dir /mnt/data/LVIS_gen_FG
```

2. Segment foreground object segment_methods/reseg.py 

```
cd segment_methods

## for each segment method, you should manually download their models and edit the model path in export.py 

python reseg.py --input_dir /mnt/data/LVIS_gen_FG --output_dir /mnt/data/LVIS_gen_FG_segs/ --seg_method clipseg
python reseg.py --input_dir /mnt/data/LVIS_gen_FG --output_dir /mnt/data/LVIS_gen_FG_segs/ --seg_method UFO
python reseg.py --input_dir /mnt/data/LVIS_gen_FG --output_dir /mnt/data/LVIS_gen_FG_segs/ --seg_method U2Net
python reseg.py --input_dir /mnt/data/LVIS_gen_FG --output_dir /mnt/data/LVIS_gen_FG_segs/ --seg_method selfreformer
```

3. Filtering object and create object pool 

```
cd segment_methods

python clean_pool.py --input_dir /mnt/data/LVIS_gen_FG_segs/ --image_dir /mnt/data/LVIS_gen_FG --output_file /mnt/data/LVIS_instance_pools.json --min_clip 21 --min_area 0.05 --max_area 0.95 --tolerance 1

```

4. train network 

```
## edit INST_POOL_PATH in config file as your instance pool json
bash launch.sh --config-file configs/Xpaste_swinL.yaml

```

5. demo
```
python demo.py --config-file configs/Xpaste_swinL.yaml --input example.jpg --output annotated.jpg --opts MODEL.WEIGHTS Xpaste_swinL_final.pth
```
![](visualize.png)
Qualitative results of X-Paste and baseline on LVIS test set. Left: X-Paste,
Right: baseline (Swin-L)

## Models (LVIS dataset)
| Backbone  | method | $AP^{box}$ | $AP^{mask}$ | $AP_r^{box}$ | $AP_r^{mask}$ | checkpoint |
|:--------:|:----------:|:----------:|:-----------:|:------------:|:-------------:|:-------------:|
| ResNet50  | baseline |    34.5    |     30.8    |     24.0     |      21.6    | [model](https://drive.google.com/drive/folders/1vVwrZ4ad0xiWVO-JLaxRdLMDq4vdRZwT?usp=sharing) |
| ResNet50  | X-Paste  |    37.4    |     33.2    |     33.9     |      29.7     | [model](https://drive.google.com/drive/folders/1vVwrZ4ad0xiWVO-JLaxRdLMDq4vdRZwT?usp=sharing) |
|  Swin-L  | baseline |    47.5    |     42.3    |     41.4    |      36.8     | [model](https://drive.google.com/drive/folders/1vVwrZ4ad0xiWVO-JLaxRdLMDq4vdRZwT?usp=sharing) |
|  Swin-L  | X-Paste  |    50.9    |     45.4    |     48.7     |      43.8    | [model](https://drive.google.com/drive/folders/1vVwrZ4ad0xiWVO-JLaxRdLMDq4vdRZwT?usp=sharing) |

## Small Object Generation (Extended)

기존 X-Paste 파이프라인을 확장하여 small object (COCO 기준 area < 32² = 1024px) 생성을 지원합니다.

### 추가된 옵션 (`generation/text2im.py`)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--prompt_template` | `"a photo of a single {}"` | 프롬프트 템플릿. `{}`에 카테고리명이 들어감 |
| `--image_size` | `512` | 생성 이미지 해상도 (8의 배수). 작을수록 작은 객체 생성에 유리 |

### Small Object 생성 예시

```bash
# 기존 방식 (변경 없음)
python generation/text2im.py --model diffusers --samples 100 \
  --category_file lvis_v1_train.json --output_dir LVIS_gen_FG

# Small object 전용 생성
python generation/text2im.py --model diffusers --samples 100 \
  --prompt_template "a photo of a small {}" \
  --image_size 256 \
  --category_file lvis_v1_train.json \
  --output_dir LVIS_gen_FG_small
```

### Docker로 실행하기

#### 1. Docker 이미지 빌드

```bash
# XPaste/ 디렉토리에서 실행 (Dockerfile이 여기 있음)
cd XPaste
docker build -t xpaste .
```

#### 2. 컨테이너 실행

```bash
# GPU 사용, XPaste 디렉토리 마운트
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/XPaste \
  xpaste bash
```

#### 3. 컨테이너 안에서 실행

```bash
# Small object 생성
python generation/text2im.py --model diffusers --samples 100 \
  --prompt_template "a photo of a small {}" \
  --image_size 256 \
  --category_file lvis_v1_train.json \
  --output_dir LVIS_gen_FG_small
```

#### 추천 프롬프트 템플릿

| 템플릿 | 용도 |
|--------|------|
| `"a photo of a single {}"` | 기본 (원본 X-Paste) |
| `"a photo of a small {}"` | small object |
| `"a photo of a tiny {}"` | 더 작은 object |
| `"a close-up photo of a tiny {}"` | 클로즈업 + tiny |

### SD 1.5 해상도 제한

SD 1.5는 512x512로 학습되었으므로 1920x1080 등 고해상도 직접 생성 시 반복 패턴, 구조 붕괴, VRAM 부족 문제가 발생한다. 장면 내 작은 객체를 SD에 직접 요청해도 객체가 크게 생성되므로, **생성(256~512px) → 축소 paste(20x20)** 2단계 방식을 사용한다.

### Bbox Label 생성 (Object Detection용)

생성된 전경 이미지에서 배경을 제거하고 bbox를 COCO JSON으로 추출한다.

```bash
# Bbox label 추출 (threshold 방식 - 빠름)
python segment_methods/gen_bbox_labels.py \
  --input_dir output/LVIS_gen_FG \
  --output_file output/bbox_labels.json \
  --method threshold

# Bbox label 추출 (grabcut 방식 - 정확)
python segment_methods/gen_bbox_labels.py \
  --input_dir output/LVIS_gen_FG \
  --output_file output/bbox_labels.json \
  --method grabcut

# 결과 확인
python -c "import json; d=json.load(open('output/bbox_labels.json')); \
  print(f'images: {len(d[\"images\"])}, annotations: {len(d[\"annotations\"])}, categories: {len(d[\"categories\"])}')"

# Bbox 시각화
python segment_methods/visualize_bbox.py \
  --input_dir output/LVIS_gen_FG \
  --label_file output/bbox_labels.json \
  --output_dir output/bbox_vis \
  --max_images 5
```

### 커스텀 카테고리 (군사용 등)

LVIS에 없는 카테고리도 커스텀 JSON으로 생성 가능:

```bash
# generation/military_categories.json 사용
python generation/text2im.py --model diffusers --samples 5 --batchsize 5 \
  --prompt_template "a photo of a small {}" \
  --category_file generation/military_categories.json \
  --output_dir output/LVIS_gen_FG_military
```

LVIS 내 군사 관련 카테고리: `army_tank`(id=1058), `fighter_jet`(id=436), `gun`(id=523), `helicopter`(id=555), `rifle`(id=884)

## Acknowledgements

We use code from [Detic](https://github.com/facebookresearch/Detic), [CenterNet2](https://github.com/xingyizhou/CenterNet2) and [Detectron2](https://github.com/facebookresearch/detectron2)

## Lisence 

The majority of our X-Paste is licensed under the Apache 2.0 license, portions of the project are available under separate license terms: SWIN-Transformer, CLIP, CLIPSeg, UFO and TensorFlow Object Detection API are licensed under the MIT license; UniDet, U2Net and Detic are licensed under the Apache 2.0 license; Selfreformer is lisenced under BSD 3-Clause License; Stable Diffusion is lisenced under CreativeML Open RAIL M License and the LVIS API is licensed under a custom license. If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than CC-BY-NC, MIT, or CC0

## Citation

X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion

```
@inproceedings{Zhao2022XPasteRC,
  title={X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion},
  author={Hanqing Zhao and Dianmo Sheng and Jianmin Bao and Dongdong Chen and Dong Chen and Fang Wen and Lu Yuan and Ce Liu and Wenbo Zhou and Qi Chu and Weiming Zhang and Nenghai Yu},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
