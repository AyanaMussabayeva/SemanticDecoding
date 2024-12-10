# Semantic Decoding

This repository contains code used in the paper "Semantic reconstruction of continuous language from non-invasive brain recordings" by Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth. 



## Original Ridge Encoder running instructions 
### ALL of the original instructions and code base can be accessed at https://github.com/HuthLab/semantic-decoding

- Download [language model data](https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip) and extract contents into new `data_lm/` directory. 

- Download [training data](https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip) and extract contents into new `data_train/` directory. Stimulus data for `train_stimulus/` and response data for `train_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds003020/).

- Download [test data](https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip) and extract contents into new `data_test/` directory. Stimulus data for `test_stimulus/[EXPERIMENT]` and response data for `test_response/[SUBJECT_ID]` can be downloaded from [OpenNeuro](https://openneuro.org/datasets/ds004510/).

- Estimate the encoding model. The encoding model predicts brain responses from contextual features of the stimulus extracted using GPT. The `--gpt` parameter determines the GPT checkpoint used. Use `--gpt imagined` when estimating models for imagined speech data, as this will extract features using a GPT checkpoint that was not trained on the imagined speech stories. Use `--gpt perceived` when estimating models for other data. The encoding model will be saved in `MODEL_DIR/[SUBJECT_ID]`. Alternatively, download [pre-fit encoding models](https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz).


```bash
python3 decoding/train_EM.py --subject [SUBJECT_ID] --gpt perceived
```


## Proposed Neural Network approach running instructions 

```bash
python3 decoding/train_Encoder.py --pickle_path semantic_data.pkl --nn EncoderNN --opt SGD
```

You can choose between EncoderNN (simple non-linear model) or RidgeRegressionNN (same original ridge approach but in neural network form). 
The optimizers can be chosen from: Adam, RMSProp and SGD 

