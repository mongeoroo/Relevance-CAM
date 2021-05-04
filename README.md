# PyTorch Implementation of [Relevance-CAM: Your Model Already Knows Where to Look] [CVPR 2021]

This is implementation of **[Relevance-CAM: Your Model Already Knows Where to Look]** which is accepted by CVPR 2021

<img src="https://github.com/mongeoroo/Relevance-CAM/blob/main/images/Comparison.jpg?raw=true" width="400px" height="350px" title="px(픽셀) 크기 설정" alt="Comparison"></img><br/>



## Introduction
Official implementation of [Relevance-CAM: Your Model Already Knows Where to Look].

We introduce a novel method which allows to analyze the intermediate layers of a model as well as the last convolutional layer.

Method consists of 3 phases:

1. Propagation to get the activation map of a target layer of a model.

2. Backpropagation of to the target layer with Layer-wise Relevance Propagation Rule to calculate the weighting component

3. Weighted summation with activation map and weighting component.

<img src="https://github.com/mongeoroo/Relevance-CAM/blob/main/images/R_CAM_pipeline.jpg?raw=true" width="400px" height="350px" title="px(픽셀) 크기 설정" alt="Relevance-CAM pipeline"></img><br/>


### Code explanation




