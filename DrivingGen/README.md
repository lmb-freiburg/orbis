---
license: apache-2.0
task_categories:
- image-to-video
- image-text-to-video
language:
- en
tags:
- Autonomous
size_categories:
- n<1K
pretty_name: DrivingGen
dataset_info:
- config_name: open_domain
  features:
  - name: imgs
    dtype: image
  - name: caption
    dtype: string
  splits:
  - name: eval
    num_examples: 222
- config_name: ego_condition
  features:
  - name: imgs
    dtype: image
  - name: caption
    dtype: string
  - name: ego_motion
    dtype: image
  splits:
  - name: eval
    num_examples: 222
---

# DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving

<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
    <a href='https://arxiv.org/abs/2601.01528'><img src='https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red'></a>
    <a href='https://drivinggen-bench.github.io/'><img src='https://img.shields.io/badge/DrivingGen-Website-green?logo=googlechrome&logoColor=green'></a>
    <a href='https://huggingface.co/datasets/yangzhou99/DrivingGen'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a>
    <!-- <a href='https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard'><img src='https://img.shields.io/badge/Leaderboard-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a> -->
</div>

> #### [DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving]()
>
> ##### [Yang Zhou*](https://yang-zhou-me.github.io/), [Hao Shao*](https://hao-shao.com/), [Letian Wang*](https://letian-wang.github.io/), [Zhuofan Zong](https://zongzhuofan.github.io/), [Hongsheng Li](http://www.ee.cuhk.edu.hk/~hsli/), [Steven Waslander](https://www.trailab.utias.utoronto.ca/) ("*" denotes equal contribution)

<!-- ## Citation -->

```bibtex
@misc{zhou2026drivinggencomprehensivebenchmarkgenerative,
      title={DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving}, 
      author={Yang Zhou and Hao Shao and Letian Wang and Zhuofan Zong and Hongsheng Li and Steven L. Waslander},
      year={2026},
      eprint={2601.01528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.01528}, 
}
```
<br>