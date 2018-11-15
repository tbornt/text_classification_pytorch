# Text classification models implemented in Pytorch

## Description

This repository contains the implmentation of various text classification models. The design of neural models in this repository is fully configurable through a configuration file, which does not require any code work.

Current implemented model:

* basic LSTM
* basci GRU
* LSTM + attention
* GRU + attention
* TextCNN model [Convolutional Neural Networks for Sentence Classification
](https://arxiv.org/abs/1408.5882)
* DPCNN model [Deep Pyramid Convolutional Neural Networks for Text Categorization
](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)

Current supported loss:

* cross-entropy loss
* focal loss [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## Requirements

* python3
* pytorch 4.0
* requirements.txt

First install python3 and pytorch4.0. Then run `pip3 install -r requirements.txt`.

## Experiments

### Toxic Comment Classification Challenge

First We apply the text classification models to [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). There are six fields to be predicted in this challenge. So we create six individual models to predict each field.
The main purpose of this experiment is to test different models, so feature engineering is not included.

#### steps

* create a `data` folder on the same level with main.py
* download data from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and put all files to `data` folder.
* run `bash egs/kaggle_toxic/run.sh model_type`. `model_type` can be one of `cnn`, `rnn`, `dpcnn`.

#### some results

| 模型       |   word embedding   | 参数    |  kaggle score  |
| :--------: |:----:| :----:   | :----: |
| Basic Bi-GRU        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0      |   0.9718    |
| Basic Bi-GRU        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3      |   0.9771    |
| Basic Bi-GRU+Attention       |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3     |   0.9756    |
| Basic Bi-GRU+Attention+FocalLoss        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3 attention FocalLoss     |   0.9763    |
| Basic Bi-LSTM       |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0      |   0.9710    |
| TextCNN        |glove.6b.100d |   same as described in paper    |   0.9525    |
| DPCNN        |glove.6b.100d |   same as described in paper    |   0.9773    |

