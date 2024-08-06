# Cascadia

A transformer deep learning model for de novo sequencing of data-independent acquisition mass spectrometry data. 

## General information

Full documentation and further functionality are still a work in progress. A short demo for running our trained version of Cascadia on your data is available below. Please check back soon for an updated tool! 

In the meantime, you can read our preprint here: 
https://www.biorxiv.org/content/10.1101/2024.06.03.597251v1 

Thanks for you patience! 

## Demo 

### Dependencies 

Cascadia requires the following packages - we recommend using a package manager such as conda to manage your python enviornment:
- lightning >= 2.0
- numpy < 2.0
- torch >= 2.1
- pyteomics
- tqdm 
- pandas
- tensorboard

Currently, you need to pull the Cascadia github repo and install the above dependencies yourself. Cascadia will be added to pip and conda soon, which will manage the installation of all dependencies automatically in a couple of minutes. 

### Run de novo sequencing on new data with a trained model 

To run a pretrained Cascadia model on a new dataset, you just need to provide an mzML file. The following example on a small demo dataset should take approximately 5 minutes to run:

    python3 cascadia.py \
      --mode sequence \
      --t demo.mzML  \
      --checkpoint cascadia.ckpt \
      --out demo_results

The demo dataset and model checkpoint is available [here](https://drive.google.com/drive/folders/1UTrZIrCdUqYqscbqga_KdX8kc8ZjMMfr?usp=sharing). For larger inference jobs, in order to reduce runtime we recommend using a GPU and setting the batch size to the largest value that still fits on GPU memory. 

The output is a mztab text file containing one row for each Cascadia prediction. The relevent columns are: 
-	predicted_sequence
- predicted_score
- precursor_mz
- precursor_charge
- precursor_rt

<!---
### Fine tune a model on new data

    FIXME 

### Train a new model from scratch

    FIXME 
--->
