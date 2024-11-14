# Usage

## Sequence
```
cascadia sequence SPECTRUM_FILE -m MODEL [-o OUTFILE] [-t SCORE_THRESHOLD] [-b BATCH_SIZE] [-w WIDTH] [-c MAX_CHARGE]
```

Argument | Description
---|---
|spectrum_file |  __(required)__ The mzML file to perform de novo sequencing on.|
|-o, --outfile | The output file to save de novo sequencing results to. (default: cascadia_results.ssl)|
|-t, --score_threshold] | The score threshold applied to predictions. (default: 0.8)|
|-b, --batch_size | The batch size for inference. For the fastest inference the largest batch size that fits in GPU memory is recommended.  (default: 32)|
|-w, --width | The number of adjacent scans to use when construcing augmented spectra. (default: 2)|
|-c, --max_charge | The maximum precursor charge to consider when making predictions. (default 4) |


## Train
```
cascadia train TRAIN_SPECTRUM_FILE VAL_SPECTRUM_FILE [-m MODEL] [-b BATCH_SIZE] [-w WIDTH] [-c MAX_CHARGE] [-e MAX_EPOCHS] [-lr LEARNING_RATE]
```

Argument | Description
---|---
|spectrum_file |  __(required)__ A labeled .asf file to use for model training. |
|spectrum_file |  __(required)__ A labeled .asf file used for validation during training. |
|-m, --model | A pre-trained model checkpoint to use for fine-tuning. If none is provided, the model is trained from scratch. (default: None)|
|-b, --batch_size | The batch size to use for training. (default: 32)|
|-w, --width | The number of adjacent scans used when constructing augmented spectra in the training data. (default: 2)|
|-c, --max_charge | The maximum precursor charge to be considered by the model. (default 4) |
|-e, --max_epochs | The maximum number of epochs to train the model for. The model checkpoint with the lowest validation loss after max_epochs will be saved. (default 10) |
|-lr, --learning_rate | The learning rate to use for model training. (default 1e-5) |