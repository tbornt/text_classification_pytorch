# Text classification models implemented in Pytorch

## Description

This repository contains the implmentation of various text classification models. The design of neural models in this repository is fully configurable through a configuration file, which does not require any code work.

Current implemented model:

* RNN based model. (LSTM,GRU,num layers,hidden size are all configurable)
* TextCNN model. [Convolutional Neural Networks for Sentence Classification
](https://arxiv.org/abs/1408.5882)
* DPCNN model.[Deep Pyramid Convolutional Neural Networks for Text Categorization
](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)

## Requirements

* python3
* pytorch4.0
* requirements.txt

First install python3 and pytorch4.0. Then run `pip3 install -r requirements.txt`

## Experiments

We apply the text classification models to [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
