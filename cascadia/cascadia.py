from depthcharge.depthcharge.data.spectrum_datasets import AnnotatedSpectrumDataset
from depthcharge.depthcharge.data.preprocessing import scale_to_unit_norm, scale_intensity
from depthcharge.depthcharge.tokenizers import PeptideTokenizer
from depthcharge.depthcharge.primitives import MassSpectrum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
import numpy as np
import pytorch_lightning as pl
import os
import sys, argparse
from lightning.pytorch import loggers as pl_loggers
from model import AugmentedSpec2Pep
from datetime import datetime

def filter_intensity(max_num_peaks: int=150):
    def filter(spectrum: MassSpectrum) -> MassSpectrum:
        sorted_intensity_idxs = np.argsort(spectrum._inner._intensity)[-max_num_peaks:]
        spectrum._inner._intensity = spectrum._inner._intensity[sorted_intensity_idxs]
        spectrum._inner._mz = spectrum._inner._mz[sorted_intensity_idxs]
        spectrum.rt = spectrum.rt[sorted_intensity_idxs]
        spectrum.level = spectrum.level[sorted_intensity_idxs]
        return spectrum
    return filter

def main():
  parser=argparse.ArgumentParser()
  parser.add_argument("--mode", type= str)
  parser.add_argument("--t", type= str, default= '')
  parser.add_argument("--v", type= str, default= '')
  parser.add_argument("--checkpoint", type= str, default= '', help="A model checkpoint for fine tuning")
  parser.add_argument("--lr", type= float, help="Model learning rate")
  args=parser.parse_args()

  mode = args.mode
  train_file = args.t
  val_file = args.v
  model_ckpt_path = args.checkpoint
  lr = args.lr

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
      train_dataset = AnnotatedSpectrumDataset(tokenizer, index_path=train_file, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm]) #scale_intensity(scaling="root"), scale_to_unit_norm
      val_dataset = AnnotatedSpectrumDataset(tokenizer, index_path=val_file, preprocessing_fn=[scale_intensity(scaling="root"), scale_to_unit_norm])
    else:
      train_dataset = AnnotatedSpectrumDataset(tokenizer, train_file, index_path=train_index_filename, preprocessing_fn=[scale_to_unit_norm]) #filter_intensity(max_num_peaks=500),
      val_dataset = AnnotatedSpectrumDataset(tokenizer, val_file, index_path=val_index_filename, preprocessing_fn=[scale_to_unit_norm])

    train_loader = train_dataset.loader(batch_size=32, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = val_dataset.loader(batch_size=32, num_workers=10, pin_memory=True)

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
      interval = 50000

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
        max_charge=5,
        lr=lr
      )
      interval = 5000

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_path,
        filename="Cascadia-{epoch}-{step}",
        monitor="Val MSKB Pep. Acc./dataloader_idx_0",
        mode='max',
        save_top_k=5
    )
    trainer = pl.Trainer(max_epochs=200, logger=tb_logger, log_every_n_steps=2000, val_check_interval = interval, callbacks=[ckpt_callback], accelerator='gpu')

    #Train the model
    trainer.fit(model,
                train_loader,
                val_loader)

  elif sys.argv[1] == 'sequence':

    train_file = sys.argv[2]
    chk_path = sys.argv[3]
    results_file = sys.argv[4]

    print("Predicting spectra from:", train_file)

    ckpt_path = os.getcwd() + '/checkpoint_' +  datetime.now().strftime("%m-%d-%H:%M:%S")
    os.mkdir(ckpt_path)
    train_index_filename = ckpt_path + "/train_gpu.hdf5"

    if os.path.exists(train_index_filename):
      os.remove(train_index_filename)

    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)

    if os.path.exists(train_index_filename):
      os.remove(train_index_filename)

    train_dataset = AnnotatedSpectrumDataset(tokenizer, train_file, index_path=train_index_filename, preprocessing_fn=[])
    train_loader = train_dataset.loader(batch_size=32, num_workers=4, pin_memory=True)

    model = AugmentedSpec2Pep.load_from_checkpoint(
        chk_path,
        d_model = 768,
        n_layers = 10,
        n_head = 12,
        dim_feedforward = 1024,
        dropout = 0,
        rt_width = 2,
        tokenizer=tokenizer,
        max_charge=10,
        lr = lr
    )

    if torch.cuda.is_available():
      device = 'gpu'
    else:
      device = 'cpu'

    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=1, accelerator=device)
    preds = trainer.predict(model, dataloaders=train_loader)

    pred_seqs, aa_conf, pep_conf, true_seq = [], [], [], []

    for pred_seqs_c, true_seq_c, pep_conf_c, aa_conf_c in preds:
        pred_seqs.append(pred_seqs_c)
        aa_conf.append(aa_conf_c)
        pep_conf.append(pep_conf_c)
        true_seq.append(true_seq_c)

    pred_seqs = np.concatenate(pred_seqs)
    pep_conf = np.concatenate(pep_conf)
    true_seq = np.concatenate(true_seq)

    print("Writing results to:", results_file + '.tsv')
    with open(results_file + '.tsv', 'w') as out:
        for pred, true, conf in zip(pred_seqs, true_seq, pep_conf):
            out.write("\t".join([pred.split('$')[0], true, str(conf)]) + '\n')

if __name__ == '__main__':
    main()
