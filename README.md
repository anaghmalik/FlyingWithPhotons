# Flying with Photons: Rendering Novel Views of Propagating Light 
### [Project Page](http://www.anaghmalik.com/FlyingWithPhotons/) | [Video](https://www.youtube.com/watch?v=dSFR8rs11vI) | [Paper](https://arxiv.org/abs/2404.06493)

## Installation

1. To create a virtual environment use these commands
```
python3 -m virtualenv venv
. venv/bin/activate
```
2. Additionally please install PyTorch, we tested on `torch1.12.1+cu116`

```
# torch 1.12.1+cu116
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. Install the requirements file with 

```
pip install -r requirements.txt
```

## Dataset 

Datasets can be downloaded using the `download_datasets.py` script. 
With flags `--scenes o1 o2 o3`, replacing `o1`, `o2` and `o3` with scenes you want to download. You can use shorthand `all`, `captured` or `simulated` or otherwise specify scenes by their names. 

For more info check `loaders/README.md`, you can also find the dataset on [Dropbox](https://www.dropbox.com/scl/fo/x15w32otenucx94dakqri/h?rlkey=3k5567qzhki4j8dxzc6qol0c5&dl=0).

## Training

You can train the transient field by specifying a config of the scene you want to train on, for coke bottle you would use

```
python train.py -c="./configs/train/captured/coke.ini"
```

You can then evaluate the same scene (to get quantitative and image results) with
```
python eval.py -c="./configs/train/captured/coke.ini" -tc="./configs/test/captured/coke.ini" -checkpoint_dir=[trained model directory root]
```

To see the summary during training, run the following
```
tensorboard --logdir=./results/ --port=6006
```

## Figures 

The script to make the peaktime visualizations can be found in `misc/figures.py`.


## Multiview videos 

In `configs/other` you can find configs to create multiview videos using the transforms files `transforms_bezier.json` or `transforms_spiral.json` and 

```
python eval.py -c=[trainign config path] -tc=[video config path] -checkpoint_dir=[trained model directory root]
```

You can also use the `misc/trajectory_parametrization.py` script to define anchor camera points and trajectories between them. Example trajectories are given in the file too. 

## Relativistic rendering

For relativistic renderings again you can find configs in `configs/other` and run

```
python misc/relativistic_rendering.py -c=[training config path] -tc=[video config path] -checkpoint_dir=[trained model directory root]
```

## Direct/Global separation 

The script in `misc/direct_separation.py` provides code to create a separate dataset for the direct and global componets of light. To use the script please first download the `pulse.npy` file from the [Dropbox directory](https://www.dropbox.com/scl/fo/x15w32otenucx94dakqri/h?rlkey=3k5567qzhki4j8dxzc6qol0c5&dl=0). 


## Citation

```
@article{malik2024flying,
  author = {Malik, Anagh and Juravsky, Noah and Po, Ryan and Wetzstein, Gordon and Kutulakos, Kiriakos N. and Lindell, David B.},
  title = {Flying with Photons: Rendering Novel Views of Propagating Light},
  journal = {arXiv},
  year = {2024}
}
```
## Acknowledgments

We thank [NerfAcc](https://www.nerfacc.com/) for their implementation of Instant-NGP. 
