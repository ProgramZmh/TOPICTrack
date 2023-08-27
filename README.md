# TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes

> [**TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes**](https://arxiv.org/pdf/2308.11157v1.pdf)
> 
> Xiaoyan Cao, Yiyao Zheng, Yao Yao, Huapeng Qin, Xiaoyu Cao and Shihui Guo
> 
> *[arXiv 2110.06864](https://arxiv.org/pdf/2308.11157v1.pdf)*


## News
* (2023.08) We provide a complete reproduction tutorial and release the proposed [BEE23](https://drive.google.com/file/d/1kcq3wV-sjr8H_HGNoefaGr_nx7OlVfPo/view) dataset as a new benchmark.

## Demo

<p align="center"><img src="figs/demo.gif" width="800"/></p>

```
python3 demo.py --exp_name mot17_test --dataset mot17 --test_dataset
python3 demo.py --exp_name mot20_test --dataset mot20 --test_dataset
python3 demo.py --exp_name dance_test --dataset dance --test_dataset
python3 demo.py --exp_name gmot_test --dataset gmot --test_dataset
python3 demo.py --exp_name bee_test --dataset BEE23 --test_dataset
```


## Abstract
Video data and algorithms have driven advances in multi-object tracking (MOT). While existing MOT datasets focus on occlusion and appearance similarity, complex motion patterns are widespread yet often overlooked. To address this issue, we introduce a novel dataset named BEE23 to highlight complex motions. Identity association algorithms have long been the focus of MOT research. Existing trackers can be categorized into two association paradigms: the single-feature paradigm (based on either motion or appearance features) and the serial paradigm (where one feature serves as secondary while the other is primary). However, these paradigms fall short of fully utilizing different features. In this paper, we propose a parallel paradigm and present the Two-round Parallel Matching Mechanism (TOPIC) to implement it. TOPIC leverages both motion and appearance features, adaptively selecting the preferable one as the assignment metric based on motion level. Furthermore, we provide an Attention-based Appearance Reconstruct Module (AARM) to enhance the representation of appearance feature embeddings. Comprehensive experiments demonstrate that our approach achieves state-of-the-art performance on four public datasets and BEE23. Importantly, our proposed parallel paradigm outperforms existing association paradigms significantly, e.g., reducing false negatives by 12% to 51% compared to the single-feature association paradigm. The dataset and association paradigm introduced in this work provide a fresh perspective for advancing the MOT field.

<p align="center"><img src="figs/data_mot.png" width="800"/></p>

> Comparison of different datasets' properties. In addition to occlusion and highly similar appearance, BEE23 stands out for its remarkable property of complex motion patterns. This is evident in the diversity of motion patterns between objects and the variability of motion patterns within a single object. In the legend, “Complex” and “Simple” denote objects with the most complex and simplest motion patterns in the scene, respectively.

<p align="center"><img src="figs/pipeline.png" width="800"/></p>

> Comparison of existing association paradigms with our proposed parallel paradigm. (a) The single-feature association paradigm uses either motion or appearance features as the assignment metric. (b) The serial association paradigm manually specifies a feature to filter association candidates, followed by another feature as the primary assignment metric, akin to taking the “intersection” of motion and appearance matches. (c) Our proposed parallel association paradigm utilizes motion and appearance features as assignment metrics in parallel, similar to taking the union set, and can effectively resolve conflicts.


## Installation
### Installing on the host machine

- Install Python dependencies. We utilize Python 3.8 and PyTorch 1.8.1.

  ```
  git clone https://github.com/holmescao/TOPICTrack
  cd TOPICTrack
  conda create -n TOPICTrack python=3.8
  conda activate TOPICTrack
  pip install --upgrade pip
  pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
  cd external/YOLOX/
  pip install -r requirements.txt && python setup.py develop
  cd ../deep-person-reid/
  pip install -r requirements.txt && python setup.py develop
  cd ../fast_reid/
  pip install -r docs/requirements.txt
  ```

## Data preparation

Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [DanceTrack](https://drive.google.com/drive/folders/1ASZCFpPEfSOJRktR8qQ_ZoT9nZR0hOea), [GMOT-40](https://spritea.github.io/GMOT40/download.html), [BEE23](https://drive.google.com/file/d/1kcq3wV-sjr8H_HGNoefaGr_nx7OlVfPo/view) and put them under <TOPICTrack_HOME>/data in the following structure:
```
data
|-- mot
|   |-- test
|   `-- train
|-- MOT20
|   |-- test
|   `-- train
|-- dancetrack
|   |-- test
|   |-- train
|   `-- val
|-- gmot
|   |-- test
|   `-- train
`-- BEE23
    |-- test
    `-- train
```

Note that the GMOT-40 contains 4 categories with 10 sequences each. Since the official description does not split the training and test sets for these sequences, this work treats 3 of them (sequentially numbered 0,1,3) for each category as the training set and the remaining 1 (sequentially numbered 2) as the test set.



Then, you need to turn the datasets to COCO format:

```shell
cd <TOPICTrack_HOME>
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_dance_to_coco.py
python3 tools/convert_gmot_to_coco.py
python3 tools/convert_bee_to_coco.py
```

## Model zoo

We provide some pretrained YOLO-X weights and FastReID weights for TOPICTrack.

Please download the required pre-trained weights by yourself and put them into `external/weights`.

| Dataset         | HOTA | MOTA | IDF1 | FP | FN | Model (Detection) | Model (Re-ID) |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| MOT17-half-val | 69.9 | 79.8 | 81.6 | 3,065 | 7,568 | topictrack_ablation.pth.tar [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | mot17_sbs_S50.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| MOT17-test | 63.9 | 78.8 | 78.7 | 17,000 | 101,100 | topictrack_mot17.pth.tar [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | mot17_sbs_S50.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| MOT20-half-val | 57.6 | 73.0 | 72.3 | 28,702 | 135,881 | topictrack_mot17.pth.tar [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | mot20_sbs_S50.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| MOT20-test | 62.6 | 72.4 | 77.6 | 11,000 | 131,100 | topictrack_mot20.pth.tar [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | mot20_sbs_S50.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| DanceTrack-val | 55.9 | 89.3 | 54.5 | 12,816 | 10,622 | topictrack_dance.pth.tar  [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | dance_sbs_S50.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| DanceTrack-test | 58.3 | 90.9 | 56.6 | 5,555 | 19,246 | topictrack_dance.pth.tar  [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | dance_sbs_S50.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| GMOT40-test | 84.7 | 96.6 | 92.5 | 205 | 327 | topictrack_gmot.pth.tar  [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | gmot_AGW.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |
| BEE23-test | 71.9 | 86.7 | 86.3 | 644 | 634 | topictrack_bee.pth.tar  [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] | bee_AGW.pth [[google](https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN)] |

* For more YOLO-X weights, please refer to the model zoo of [ByteTrack](https://github.com/ifzhang/ByteTrack).

## Training detector
You can use TOPICTrack without training by adopting existing detectors. But we borrow the training guidelines from ByteTrack in case you want work on your own detector. 

Download the COCO-pretrained YOLOX weight [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) and put it under *\<TOPICTrack_HOME\>/pretrained*.

#### Train ablation model MOT17 half train

You can run the follow command:

```shell
sh run/mot17_half_train.sh
```

Or

```shell
python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 1 -b 4 --fp16 -o -c external/weights/yolox_x.pth
```

#### Train MOT17 test model 

You can run the follow command:

```shell
sh run/mot17_train.sh
```

Or

```shell
python3 tools/train.py -f exps/example/mot/yolox_x_mot17_train.py -d 1 -b 4 --fp16 -o -c external/weights/yolox_x.pth
```

#### Train MOT20 test model 

You can run the follow command:

```shell
sh run/mot20_train.sh
```

Or

```shell
python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 1 -b 4 --fp16 -o -c external/weights/yolox_x.pth
```

#### Train DanceTrack test model 

You can run the follow command:

```shell
sh run/dancetrack_train.sh
```

Or

```shell
python3 tools/train.py -f exps/example/mot/yolox_x_dance_train.py -d 1 -b 4 --fp16 -o -c external/weights/yolox_x.pth
```

#### Train GMOT-40 test model 

You can run the follow command:

```shell
sh run/gmot_train.sh
```

Or

```shell
python3 tools/train.py -f exps/example/mot/yolox_x_gmot_train.py -d 1 -b 4 --fp16 -o -c external/weights/yolox_x.pth
```

#### Train BEE23 test model 

You can run the follow command:

```shell
sh run/bee_train.sh
```

Or

```shell
python3 tools/train.py -f exps/example/mot/yolox_x_bee23_train.py -d 1 -b 4 --fp16 -o -c external/weights/yolox_x.pth
```

#### Train custom dataset

First, you need to prepare your dataset in COCO format. You can refer to [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) or [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py). Then, you need to create a Exp file for your dataset. You can refer to the [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) training Exp file. Don't forget to modify get_data_loader() and get_eval_loader in your Exp file. Finally, you can train bytetrack on your dataset by running:

```shell
python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

## Training Re-ID model
It is coming...

## Evaluation

Please download the pre-trained model mentioned in the Model zoo chapter and put it in `external/weights` before proceeding to the following steps.

### MOT17

#### Evaluation on MOT17 half val

You can run the follow command:

```shell
sh run/mot17_val.sh
```

The results are saved in: `results/trackers/MOT17-val/mot17_val_post` path.

Evaluate tracking results:

```shell
exp=mot17_val dataset=MOT17 sh eval_metrics.sh
```

#### Test on MOT17

You can run the follow command:

```shell
sh run/mot17_test.sh
```

The results are saved in: `results/trackers/MOT17-val/mot17_test_post` path.

### MOT20

#### Evaluation on MOT20 half val

You can run the follow command:

```shell
sh run/mot20_val.sh
```

The results are saved in: `results/trackers/MOT20-val/mot20_val_post` path.

Evaluate tracking results:

```shell
exp=mot20_val dataset=MOT20 sh eval_metrics.sh
```

#### Test on MOT20

You can run the follow command:

```shell
sh run/mot20_test.sh
```

The results are saved in: `results/trackers/MOT20-val/mot20_test_post` path.

### DanceTrack

#### Evaluation on DanceTrack half val

You can run the follow command:

```shell
sh run/dancetrack_val.sh
```

The results are saved in: `results/trackers/DANCE-val/dance_val_post` path.

Evaluate tracking results:

```shell
exp=dance_val dataset=DANCE sh eval_metrics.sh
```

#### Test on DanceTrack

You can run the follow command:

```shell
sh run/dancetrack_test.sh
```

The results are saved in: `results/trackers/DANCE-val/dance_test_post` path.

### GMOT-40

#### Evaluation/Test on GMOT-40 half val

GMOT-40 has only training set and test set without dividing the validation set. Therefore the validation set and test set have the same evaluation script. You can run the follow command:

```shell
sh run/gmot_test.sh
```

The results are saved in: `results/trackers/GMOT-val/gmot_test_post` path.

Evaluate tracking results:

```shell
exp=gmot_test dataset=GMOT sh eval_metrics.sh
```

### BEE23

#### Evaluation/Test on BEE23 half val

BEE23 has only training set and test set without dividing the validation set. Therefore the validation set and test set have the same evaluation script. You can run the follow command:

```shell
sh run/bee_test.sh
```

The results are saved in: `results/trackers/BEE23-val/bee_test_post` path.

Evaluate tracking results:

```shell
exp=bee_test dataset=BEE23 sh eval_metrics.sh
```

## Citation
If you find this project useful, please consider to cite our paper. Thank you!

```
@misc{cao2023topic,
      title={TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes}, 
      author={Xiaoyan Cao and Yiyao Zheng and Yao Yao and Huapeng Qin and Xiaoyu Cao and Shihui Guo},
      year={2023},
      eprint={2308.11157},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [OC-SORT](https://github.com/noahcao/OC_SORT), [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/) and [FastReID](https://github.com/JDAI-CV/fast-reid). Many thanks for their wonderful works.