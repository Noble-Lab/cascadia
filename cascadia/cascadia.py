from .depthcharge.data.spectrum_datasets import AnnotatedSpectrumDataset
from .depthcharge.data.preprocessing import scale_to_unit_norm, scale_intensity
from .depthcharge.tokenizers import PeptideTokenizer
import torch
import numpy as np
import pytorch_lightning as pl
import os
import argparse
from lightning.pytorch import loggers as pl_loggers
from .utils import *
from .model import AugmentedSpec2Pep
from .augment import *
from datetime import datetime

def sequence():
  parser=argparse.ArgumentParser()
  parser.add_argument("spectrum_file")
  parser.add_argument("--model", type= str, default= '', help="A model checkpoint for fine tuning")
  parser.add_argument("-o", "--outfile", type= str, default= 'cascadia_results', help="Output file for inference")
  parser.add_argument("--batch_size", type= int, default= 32, help="Number of spectra to include in a batch.")
  args=parser.parse_args()

  spectrum_file = args.spectrum_file
  model_ckpt_path = args.model
  lr = args.lr
  results_file = args.outfile
  batch_size = args.batch_size


  score_threshold = 0.8
  augmentation_width = 2
  
  temp_path = os.getcwd() + '/cascadia_' +  datetime.now().strftime("%m-%d-%H:%M:%S")
  os.mkdir(temp_path)
  train_index_filename = temp_path + "/train_gpu.hdf5"

  print("Augmenting spectra from:", spectrum_file)
  asf_file, isolation_window_size, cycle_time = augment_spectra(spectrum_file, temp_path)
  
  tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
  train_dataset = AnnotatedSpectrumDataset(tokenizer, asf_file, index_path=train_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
  train_loader = train_dataset.loader(batch_size=batch_size, num_workers=4, pin_memory=True)

  # if os.path.exists(asf_file):
  #     os.remove(asf_file)

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
      lr=lr
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
  return parser

  
def main():
  parser=argparse.ArgumentParser()
  parser.add_argument("--mode", type= str)
  parser.add_argument("--t", type= str, default= '')
  parser.add_argument("--v", type= str, default= '')
  parser.add_argument("--checkpoint", type= str, default= '', help="A model checkpoint for fine tuning")
  parser.add_argument("--out", type= str, default= '', help="Output file for inference")
  parser.add_argument("--lr", type= float, help="Model learning rate")
  parser.add_argument("--batch_size", type= int, default= 8, help="Number of spectra to include in a batch.")
  args=parser.parse_args()

  mode = args.mode
  train_file = args.t
  val_file = args.v
  model_ckpt_path = args.checkpoint
  lr = args.lr
  results_file = args.out
  batch_size = args.batch_size

  if mode == 'train': 

    if not torch.cuda.is_available():
      print("No GPU Available!")
      #exit()

    print("Training on spectra from:", train_file)
    print("Validating on spectra from:", val_file)
    
    ckpt_path = os.getcwd() + '/checkpoint_' +  datetime.now().strftime("%m-%d-%H:%M:%S")
    os.mkdir(ckpt_path)
    train_index_filename = ckpt_path + "/train_gpu.hdf5"
    val_index_filename = ckpt_path + "/val_gpu.hdf5"

    if os.path.exists(train_index_filename):
      os.remove(train_index_filename)
    if os.path.exists(val_index_filename):
      os.remove(val_index_filename)

    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
    
    if '.hdf5' in train_file:
      train_dataset = AnnotatedSpectrumDataset(tokenizer, index_path=train_file, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
      val_dataset = AnnotatedSpectrumDataset(tokenizer, index_path=val_file, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
    else:
      train_dataset = AnnotatedSpectrumDataset(tokenizer, train_file, index_path=train_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm]) 
      val_dataset = AnnotatedSpectrumDataset(tokenizer, val_file, index_path=val_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
    
    train_loader = train_dataset.loader(batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = val_dataset.loader(batch_size=batch_size, num_workers=10, pin_memory=True)

    if model_ckpt_path == "":
      print("Training model from scratch") 
      model = AugmentedSpec2Pep(
          d_model = 512,
          n_layers = 9,
          n_head = 8,
          dim_feedforward = 1024,
          dropout = 0,
          rt_width = 2,
          tokenizer=tokenizer,
          max_charge=10,
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
        lr=lr
      )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_path,
        filename="Cascadia-{epoch}-{step}",
        monitor="Val MSKB Pep. Acc./dataloader_idx_0",
        mode='max',
        save_top_k=5
    )
    trainer = pl.Trainer(max_epochs=200, logger=tb_logger, log_every_n_steps=2000, val_check_interval = 1000, callbacks=[ckpt_callback], accelerator='gpu')
    trainer.fit(model, train_loader, val_loader)
  
  elif mode == 'sequence':

    score_threshold = 0.8
    augmentation_width = 2
    
    temp_path = os.getcwd() + '/cascadia_' +  datetime.now().strftime("%m-%d-%H:%M:%S")
    os.mkdir(temp_path)
    train_index_filename = temp_path + "/train_gpu.hdf5"

    print("Augmenting spectra from:", train_file)
    asf_file, isolation_window_size, cycle_time = augment_spectra(train_file, temp_path)
    
    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
    train_dataset = AnnotatedSpectrumDataset(tokenizer, asf_file, index_path=train_index_filename, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
    train_loader = train_dataset.loader(batch_size=32, num_workers=4, pin_memory=True)

    # if os.path.exists(asf_file):
    #     os.remove(asf_file)

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
        lr=lr
    )

    print("Running inference on augmented spectra from:", train_file)
    if torch.cuda.is_available():
      device = 'gpu'
      print('GPU found')
    else:
      device = 'cpu'
      print(f'No GPU found - running inference on cpu')

    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=1, accelerator=device)
    preds = trainer.predict(model, dataloaders=train_loader)

    print("Writing results to:", results_file + '.ssl')
    write_results(preds, results_file, train_file, isolation_window_size, score_threshold, augmentation_width*cycle_time)

if __name__ == '__main__':
    main()