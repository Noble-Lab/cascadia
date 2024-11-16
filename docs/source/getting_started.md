# Getting Started

### Installation 

We recommend using [conda](https://docs.conda.io/en/latest/) to manage dependencies for Cascadia. Create a new conda enviornment with:

```sh
conda create --name cascadia_env python=3.10
```

This will create an environment called `cascadia_env` with Python 3.10 installed. Activate it by running:

```sh
conda activate cascadia
```

Finally, you can install Cascadia and all of its dependencies with:

```sh
pip install cascadia
```

### Run de novo sequencing on new data with a trained model 

```{note}
We recommend using linux and a dedicated GPU to achieve optimal runtime performance.
```

Most users will want to use a pretrained Cascadia model to perform de novo sequencing on a new dataset. Cascadia takes input MS data in the [mzML](file_formats.md) format. A small demo dataset, along with the pretrained model checkpoints from the paper, are available [here](https://drive.google.com/drive/folders/1UTrZIrCdUqYqscbqga_KdX8kc8ZjMMfr?usp=sharing). The following example on the provided demo dataset should take approximately 5 minutes to run on a GPU:

```sh
    cascadia sequence \
      demo.mzML  \
      --checkpoint cascadia.ckpt \
      --out demo_results
```

<!-- For larger inference jobs, in order to reduce runtime we recommend using a GPU and setting the batch size to the largest value that still fits on GPU memory.  -->

Cascadia will produce an output file, [`demo_results.ssl`](file_formats.md), containing the de novo sequencing results. This file contains one row for each prediction, and can be loaded int [skyline](https://skyline.ms/wiki/home/software/BiblioSpec/page.view?name=default) as a spectral library to visualize the results. 

A full description of additional optional paramaters to Cascadia sequencing is available [here](usage.md). 

### Train a new model from scratch

To train Cascadia on new data, you need a labeled training and validation set in [.asf format](file_formats.md) as positional arguments:
```sh
    cascadia train training_data.asf validation_data.asf
```
A full list of additional optional arguments for training are descried [here](usage.md). 

### Fine tune a model on new data

To fine tune a pre-trained model checkpoint on new data, you can simply pass it as an additional keyword argument to `train`:  

```sh
    cascadia train training_data.asf validation_data.asf \
        --model pretrained_checkpoint.ckpt
```

<!-- FIXME describe adding a new PTM -->
