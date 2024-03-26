This repo contains the code for the implementation of Align MacridVAE. If you want to learn more about the paper you can check the [article](https://link.springer.com/chapter/10.1007/978-3-031-56027-9_5) presented at [ECIR 2024](https://www.ecir2024.org/). This implements a multimodal recommender that can suggest items to users based on their preferences

The project is implemented in Torch and implements a shallow Variational Autoencoder with a pre-training step to Align image and textual representation. 

## Requirements

About software, you will need the following tools:
- [Python](https://www.python.org/) 3.10 or later
- Recommended for training with a GPU
	- [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) or later, if you plan to train this on a GPU
	- [CUDNN](https://developer.nvidia.com/rdp/cudnn-archive) (for example, v.8.9.7 for CUDA 11.x)

Regarding harwate, this project is meant to run in NVIDIA GPUs, like the ones personal laptops, or in datacenters. It can also run on the CPU but it will be much slower. We tested it in [V100](), [RTX 20 series](https://www.nvidia.com/en-gb/geforce/20-series/) . The model is relatively simple and small and we don't load larger models like CLIP, BERT or ViT during training or inference. Items are preprocessed before running through the model to simplify training.

## Installation

First, install the `requirements.txt` file which specifies the dependencies
```shell
pip install -r requirements.txt
```
Next fetch the datasets. The datasets are hosted in Kaggle [here](https://www.kaggle.com/datasets/ignacioavas/alignmacrid-vae) and it is available to download  through the web UI or using the [command line tools](https://github.com/Kaggle/kaggle-api). For example, if you already have set up your Kaggle credentials.

```bash
# Optional, you can download the dataset through the website
kaggle datasets download ignacioavas/alignmacrid-vae

unzip alignmacrid-vae.zip -d RecomData/
rm alignmacrid-vae.zip
```
The dataset contains data from subcategories [Amazon Dataset](https://nijianmo.github.io/amazon/index.html), [Movielens 25M](https://grouplens.org/datasets/movielens/25m/), [Bookcrossing](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset).. Those datasets were prepared by adding images and filtering missing items, and then passing textual and visual representation through and encoders like BERT, CLIP or ViT. You can learn more by reading the README.md in the dataset root directory. The preprocessing code for building the datasets is available at [Align-MacridVAE-data](https://github.com/igui/Align-MacridVAE-data/t),

## Running

Once you have the datasets downloaded, you can train a model by running the `main.py` script with the `train` argument. For example, to train the Amazon Musical Instruments dataset encoded with CLIP for visual and textual modality run the following command:

```bash
python main.py train --data Musical_Instruments-clip_clip
```

The training code will generate a file in  the `run/` directory with a name depending on the dataset and the model parameters, for example: `Musical_Instruments-clip_clip-AlignMacridVAE-50E-100B-0.001L-0.0001W-0.5D-0.2b-7k-200d-0.1t-98765s`. The `model.pkl` file contains the trained model.

Run `python main.py --help` to see all available parameters.

## Evaluating 

To evaluate a given model, we can pass the `test` mode. It will try to load a model from the `run` directory provided it was already trained. For example, to evaluate the same model as above run the following command:

```python
python main.py test --data Musical_Instruments-clip_clip
```
