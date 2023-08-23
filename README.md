# TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes

> [**TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes**](https://arxiv.org/pdf/2308.11157v1.pdf)
> 
> Xiaoyan Cao, Yiyao Zheng, Yao Yao, Huapeng Qin, Xiaoyu Cao and Shihui Guo
> 
> *[arXiv 2110.06864](https://arxiv.org/pdf/2308.11157v1.pdf)*

## Abstract
Video data and algorithms have driven advances in multi-object tracking (MOT). While existing MOT datasets focus on occlusion and appearance similarity, complex motion patterns are widespread yet often overlooked. To address this issue, we introduce a novel dataset named BEE23 to highlight complex motions. Identity association algorithms have long been the focus of MOT research. Existing trackers can be categorized into two association paradigms: the single-feature paradigm (based on either motion or appearance features) and the serial paradigm (where one feature serves as secondary while the other is primary). However, these paradigms fall short of fully utilizing different features. In this paper, we propose a parallel paradigm and present the Two-round Parallel Matching Mechanism (TOPIC) to implement it. TOPIC leverages both motion and appearance features, adaptively selecting the preferable one as the assignment metric based on motion level. Furthermore, we provide an Attention-based Appearance Reconstruct Module (AARM) to enhance the representation of appearance feature embeddings. Comprehensive experiments demonstrate that our approach achieves state-of-the-art performance on four public datasets and BEE23. Importantly, our proposed parallel paradigm outperforms existing association paradigms significantly, e.g., reducing false negatives by 12% to 51% compared to the single-feature association paradigm. The dataset and association paradigm introduced in this work provide a fresh perspective for advancing the MOT field.

<p align="center"><img src="figs/data_mot.png" width="800"/></p>

> Comparison of different datasets' properties. In addition to occlusion and highly similar appearance, BEE23 stands out for its remarkable property of complex motion patterns. This is evident in the diversity of motion patterns between objects and the variability of motion patterns within a single object. In the legend, “Complex” and “Simple” denote objects with the most complex and simplest motion patterns in the scene, respectively.

<p align="center"><img src="figs/pipeline.png" width="800"/></p>

> Comparison of existing association paradigms with our proposed parallel paradigm. (a) The single-feature association paradigm uses either motion or appearance features as the assignment metric. (b) The serial association paradigm manually specifies a feature to filter association candidates, followed by another feature as the primary assignment metric, akin to taking the “intersection” of motion and appearance matches. (c) Our proposed parallel association paradigm utilizes motion and appearance features as assignment metrics in parallel, similar to taking the union set, and can effectively resolve conflicts.

## Reproduction
### 1 Installation

- Install Python dependencies. We utilize Python 3.6 and PyTorch 1.10.1.

  ```
  conda create -n BeeTrack python=3.6
  conda activate BeeTrack
  pip install --upgrade pip
  cd ${BEETRACK_ROOT}
  pip install -r requirements.txt
  ```

### Other Reproduction Steps for the Algorithm
✨ The code mentioned in the paper has been updated.

⏳ Detailed documentation for the reproduction process is expected to be completed by August 27, 2023.

## Dataset Preparation
⏳ The BEE23 dataset is currently being prepared and will be made available by August 27, 2023.

## Citation
If you find this project useful, please cite it. Thank you!

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