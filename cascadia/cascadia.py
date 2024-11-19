from .depthcharge.data.spectrum_datasets import AnnotatedSpectrumDataset
from .depthcharge.data.preprocessing import scale_to_unit_norm, scale_intensity
from .depthcharge.tokenizers import PeptideTokenizer
import torch
import numpy as np
import pytorch_lightning as pl
import os
import sys
import argparse
from lightning.pytorch import loggers as pl_loggers
from .utils import *
from .model import AugmentedSpec2Pep
from .augment import *
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

def sequence():
  parser=argparse.ArgumentParser()
  parser.add_argument("spectrum_file")
  parser.add_argument("model", type= str, help="A path to a trained Cascadia model checkpoint.")
  parser.add_argument("-o", "--outfile", type= str, default= 'cascadia_results', help="Output file for inference.")
  parser.add_argument("-t", "--score_threshold", type= float, default= 0.8, help="Score threshold for Cascadia predictions.")
  parser.add_argument("-b", "--batch_size", type= int, default= 32, help="Number of spectra to include in a batch.")
  parser.add_argument("-w", "--width", type= int, default= 2, help="Number of adjacent scans to use when constructing each augmented spectrum.")
  parser.add_argument("-c", "--max_charge", type= int, default= 4, help="Maximum precursor charge state to consider")
  parser.add_argument("-p", "--modifications", type= str, default= 'mskb', help="A path to a json file containing a list of the PTMs to consider.")
  
  args = parser.parse_args(args=sys.argv[2:])

  spectrum_file = args.spectrum_file
  model_ckpt_path = args.model
  results_file = args.outfile
  batch_size = args.batch_size
  score_threshold = args.score_threshold
  augmentation_width = args.width
  max_charge = args.max_charge
  mods = args.modifications
  
  temp_path = os.getcwd() + '/cascadia_' +  datetime.now().strftime("%m-%d-%H:%M:%S")
  os.mkdir(temp_path)
  train_index_filename = temp_path + "/index.hdf5"

  print("Augmenting spectra from:", spectrum_file)
  asf_file, isolation_window_size, cycle_time = augment_spectra(spectrum_file, temp_path, max_charge=max_charge)
  
  if mods == 'mskb':
    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
  else:
    with open('ptms.json', 'r') as f:
      proforma = json.load(f)
    tokenizer = PeptideTokenizer.from_proforma(proforma, reverse=False, replace_isoleucine_with_leucine=True)
  
  train_dataset = AnnotatedSpectrumDataset(tokenizer, asf_file, index_path=train_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
  train_loader = train_dataset.loader(batch_size=batch_size, num_workers=4, pin_memory=True)

  if os.path.exists(asf_file):
      os.remove(asf_file)

  model = AugmentedSpec2Pep.load_from_checkpoint(
      model_ckpt_path,
      d_model = 512,
      n_layers = 9,
      n_head = 8,
      dim_feedforward = 1024,
      dropout = 0,
      rt_width = 2,
      tokenizer=tokenizer,
      max_charge=10,
  )

  print("Running inference on augmented spectra from:", spectrum_file)
  if torch.cuda.is_available():
    device = 'gpu'
    print('GPU found')
  else:
    device = 'cpu'
    print(f'No GPU found - running inference on cpu')

  trainer = pl.Trainer(max_epochs=50, log_every_n_steps=1, accelerator=device)
  preds = trainer.predict(model, dataloaders=train_loader)

  print("Writing results to:", results_file + '.ssl')
  write_results(preds, results_file, spectrum_file, isolation_window_size, score_threshold, augmentation_width*cycle_time)
  
  os.remove(train_index_filename)
  os.rmdir(temp_path)
  
  return parser

def train():
  print("train", sys.argv)
  parser=argparse.ArgumentParser()
  parser.add_argument("train_spectrum_file")
  parser.add_argument("val_spectrum_file")
  parser.add_argument("-m", "--model", type= str, required=False, help="A path to a Cascadia model checkpoint to fine tune.")
  parser.add_argument("-b", "--batch_size", type= int, default= 32, help="Number of spectra to include in a batch.")
  parser.add_argument("-w", "--width", type= int, default= 2, help="Number of adjacent scans to use when constructing each augmented spectrum.")
  parser.add_argument("-c", "--max_charge", type= int, default= 4, help="Maximum precursor charge state to consider.")
  parser.add_argument("-e", "--max_epochs", type= int, default= 10, help="Maximum number of training epochs.")
  parser.add_argument("-lr", "--learning_rate", type= float, default= 1e-5, help="Learning rate for model training.")
  parser.add_argument("-p", "--modifications", type= str, default= 'mskb', help="A path to a json file containing a list of the PTMs to consider.")
  
  args = parser.parse_args(args=sys.argv[2:])

  train_spectrum_file = args.train_spectrum_file
  val_spectrum_file = args.val_spectrum_file
  model_ckpt_path = args.model
  batch_size = args.batch_size
  augmentation_width = args.width
  max_charge = args.max_charge
  max_epochs = args.max_epochs
  lr = args.learning_rate
  mods = args.modifications
  
  if not torch.cuda.is_available():
    print("No GPU Available - training on CPU will be extremely slow!")

  print("Training on spectra from:", train_spectrum_file)
  print("Validating on spectra from:", val_spectrum_file)
  
  ckpt_path = os.getcwd() + '/checkpoint_' +  datetime.now().strftime("%m-%d-%H:%M:%S")
  os.mkdir(ckpt_path)
  train_index_filename = ckpt_path + "/train_gpu.hdf5"
  val_index_filename = ckpt_path + "/val_gpu.hdf5"

  if os.path.exists(train_index_filename):
    os.remove(train_index_filename)
  if os.path.exists(val_index_filename):
    os.remove(val_index_filename)

  if mods == 'mskb':
    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
  else:
    with open(mods, 'r') as f:
      proforma = json.load(f)
    tokenizer = PeptideTokenizer.from_proforma(proforma, reverse=False, replace_isoleucine_with_leucine=True)
  
  if '.hdf5' in train_spectrum_file:
    train_dataset = AnnotatedSpectrumDataset(tokenizer, index_path=train_spectrum_file, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
    val_dataset = AnnotatedSpectrumDataset(tokenizer, index_path=val_spectrum_file, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
  else:
    train_dataset = AnnotatedSpectrumDataset(tokenizer, train_spectrum_file, index_path=train_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm]) 
    val_dataset = AnnotatedSpectrumDataset(tokenizer, val_spectrum_file, index_path=val_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
  
  train_loader = train_dataset.loader(batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
  val_loader = val_dataset.loader(batch_size=batch_size, num_workers=10, pin_memory=True)

  if model_ckpt_path is None:
    print("Training model from scratch") 
    model = AugmentedSpec2Pep(
        d_model = 512,
        n_layers = 9,
        n_head = 8,
        dim_feedforward = 1024,
        dropout = 0,
        rt_width = augmentation_width,
        tokenizer=tokenizer,
        max_charge=max_charge,
        lr=lr
    )
    
  else:
    print("Loading model from checkpoint:", model_ckpt_path) 
    model = AugmentedSpec2Pep.load_from_checkpoint(
      model_ckpt_path, 
      d_model = 512,
      n_layers = 9,
      n_head = 8,
      dim_feedforward = 1024,
      dropout = 0,
      rt_width = 2,
      tokenizer=tokenizer,
      max_charge=max_charge,
      lr=lr
    )

  tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
  ckpt_callback = pl.callbacks.ModelCheckpoint(
      dirpath=ckpt_path,
      filename="Cascadia-{epoch}-{step}",
      monitor="Val Pep. Acc.",
      mode='max',
      save_top_k=2
  )
  trainer = pl.Trainer(max_epochs=max_epochs, logger=tb_logger, log_every_n_steps=10000, val_check_interval = 10000, check_val_every_n_epoch=None, callbacks=[ckpt_callback], accelerator='gpu')
  trainer.fit(model, train_loader, val_loader)
  
def main():
  parser=argparse.ArgumentParser()
  parser.add_argument("mode", choices=['sequence', 'train'], help='Which command to call')
  args = parser.parse_args(args=sys.argv[1:2])
  mode = args.mode
  if mode == 'sequence':
    sequence()
  elif mode == 'train':
    train()

if __name__ == '__main__':
    main()