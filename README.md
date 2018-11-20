# Text classification models implemented in Pytorch

## Description

This repository contains the implmentation of various text classification models. The design of neural models in this repository is fully configurable through a configuration file, which does not require any code work.

Current implemented model:

* basic LSTM
* basci GRU
* LSTM + attention(dot, general) [Effective Approaches to Attention-based Neural Machine Translation](http://aclweb.org/anthology/D15-1166)
* GRU + attention(dot, general) [Effective Approaches to Attention-based Neural Machine Translation](http://aclweb.org/anthology/D15-1166)
* TextCNN [Convolutional Neural Networks for Sentence Classification
](https://arxiv.org/abs/1408.5882)
* DPCNN [Deep Pyramid Convolutional Neural Networks for Text Categorization
](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)
* RCNN [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
* Transformer Encoder [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

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

#### Steps

* create a `data` folder on the same level with main.py
* download data from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and put all files to `data` folder.
* run `bash egs/kaggle_toxic/run.sh model_type`. `model_type` can be one of `cnn`, `rnn`, `dpcnn`.

#### Some Results

| 模型       |   word embedding   | 参数    |  kaggle score  |
| :--------: |:----:| :----:   | :----: |
| Basic Bi-GRU        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0      |   0.9718    |
| Basic Bi-GRU        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3      |   0.9771    |
| Basic Bi-GRU        |glove.6b.50d |hidden_size:200 n_layer:1 input_dropout:0.3      |   0.9745    |
| Basic Bi-GRU        |glove.6b.300d |hidden_size:200 n_layer:1 input_dropout:0.3      |   0.9766    |
| Basic Bi-GRU+2 hidden layer       |glove.6b.100d |hidden_size:200 n_layer:2 input_dropout:0.3 dropout:0.5      |   0.9793    |
| Basic Bi-GRU+FocalLoss        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3 FocalLoss     |   0.9755    |
| Basic Bi-GRU+Attention(general)       |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3     |   0.9773    |
| Basic Bi-GRU+Attention(dot)       |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3     |   0.9756    |
| Basic Bi-GRU+Attention(dot)+FocalLoss        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0.3 attention FocalLoss     |   0.9763    |
| Basic Bi-LSTM       |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0      |   0.9710    |
| TextCNN        |glove.6b.100d |   same as described in paper    |   0.9525    |
| DPCNN        |glove.6b.100d |   same as described in paper    |   0.9773    |
| RCNN        |glove.6b.100d |   bi-gru + max_pooling    |   0.9797    |
| RCNN        |glove.6b.100d |   bi-gru + 2 layer + max_pooling    |   0.9789    |

### Stanford Sentiment Treebank

#### Description

The dataset contains movie reviews parsed and labeled by Socher et al. (2013). The labels are Very Negative, Negative, Neutral, Positive, and Very Positive.

| Train | Dev | Test | labels |
| :---: | :---: | :---: | :---:|
| 8544  | 1101  | 2210  | 5    |

#### Result

| 模型       |   word embedding   | 参数    |  Accuracy  |
| :--------: |:----:| :----:   | :----: |
| Basic Bi-GRU        |glove.6b.100d |hidden_size:200 n_layer:1 input_dropout:0      |   0.5067873303167421    |

ConfusionMatrix

|真实\预测| very negative | negative | neutral | positive | very positive |
| :---: | :---: | :---: | :---: | :---: | :---: |
|very negative|0|1|107|0|0|
|negative|0|12|387|5|2|
|neutral|0|38|1103|14|2|
|positive|1|8|412|5|0|
|very positive|0|6|106|1|0|
