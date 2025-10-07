# Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models
**Official Implementation**
## [Paper](https://arxiv.org/abs/2507.13162) | [Project Page](https://lmb-freiburg.github.io/orbis.github.io/) 

> [Arian Mousakhan](https://lmb.informatik.uni-freiburg.de/people/mousakha/), [Sudhanshu Mittal](https://lmb.informatik.uni-freiburg.de/people/mittal/) , [Silvio Galesso](https://lmb.informatik.uni-freiburg.de/people/galessos/), [Karim Farid](https://lmb.informatik.uni-freiburg.de/people/faridk/), [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox/index.html)
> <br>University of Freiburg<br>


![Teaser](imgs/Rollout.png)



## Installation
```bash
conda create -n orbis python=3.10
conda activate orbis
git clone https://github.com/lmb-freiburg/orbis.git
cd orbis
pip install -r requirements.txt
```

## Checkpoints 
Link to [Checkpoints](https://drive.google.com/file/d/18qGkgOS83TvSSVkTINgckA7DO9VVoz4r/view?usp=sharing)


## Autoregressive Video Generation (Roll-out)
To predict next frames from a set of input images, use:
```bash
python evaluate/rollout.py --exp_dir STAGE2_EXPERIMENT_DIR --num_gen_frames 120 --num_steps 30
```

Alternatively, you can specify a configuration file for the inference data:
```bash
python evaluate/rollout.py --exp_dir STAGE2_EXPERIMENT_DIR --val_config val_config.yaml --num_gen_frames 120 --num_steps 30
```

The generated latents can be saved to disk using the `--save_latents` flag. This can be useful for Monte-Carlo sampling and uncertainty evaluation using multiple random seeds.


##  Training

### Tokenizer
You can train a first stage model (tokenizer) with the following command:
```bash
python main.py  --base configs/stage1.yaml --name stage1 -t --n_gpus 1 --n_nodes 1
```
You can train the first stage in Image Folder style using the following command:
```bash
python main.py  --base configs/stage1_if.yaml --name stage1 -t --n_gpus 1 --n_nodes 1
```
You may need to adapt a few paths in the config file to your own system, or you can override them from CLI.

#### Features
##### In the config file:

- `cont_ratio_training`: Sets the ratio at which vector quantization is bypassed, allowing the decoder to receive continuous (non-discrete) tokens.
- `only_decoder`: If set to true, only the decoder is trained while other components remain frozen.
- `scale_equivariance`: Enables Scale Equivariance training. `se_weight` is defiend as SE loss temperature. 
- You can use `callbacks/logging.py` to calculate enc_scale in tokenizer config.


### 🔮 Video Generation Model
You can train a second stage model with the following command:
```bash
python main.py  --base configs/stage2.yaml --name stage2 -t --n_gpus 1 --n_nodes 1
```
If you trained Image Folder tokenizer you can use the following command line:
```bash
python main.py  --base configs/stage2_if.yaml --name stage2 -t --n_gpus 1 --n_nodes 1
```
You may need to adapt a few paths in the config file to your own system, or you can override them from CLI.

#### Additional points to consider
- Two variables $TK_WORK_DIR and $WM_WORK_DIR are defined that refer to tokenizer and World Model directory. By setting them, experiment outputs will be automatically saved in the specified directory.

## Acknowledgement
This codebase builds upon several excellent open-source projects, including:
- [DiT](https://github.com/facebookresearch/DiT)
- [Taming-Transformers](https://github.com/CompVis/taming-transformers)

We sincerely thank the authors for making their work publicly available.

## BibTeX
```bibtex
@article{mousakhan2025orbis,
  title={Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models},
  author={Mousakhan, Arian and Mittal, Sudhanshu and Galesso, Silvio and Farid, Karim and Brox, Thomas},
  journal={arXiv preprint arXiv:2507.13162},
  year={2025}
}
```
