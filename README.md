## Problem Description
Real vs. Fake image classification is a critical task in digital content authentication and security systems. The objective of this project is to develop a Convolutional Neural Network (CNN) model that can classify images as either real or fake (synthetically generated). Each image belongs to one of two categories: real (genuine natural images) or fake (AI-generated or manipulated images). This project utilizes CNN architecture to classify images with a reasonable degree of accuracy.

## Input-Output
1. Input: Images in various formats (.jpg, .png), where each image corresponds to either the real or fake class.

2. Output: A classification label representing the type of image:

0 → Real Image

1 → Fake Image

## Data Source
The dataset used in this project consists of real and fake images collected from a variety of sources, including publicly available datasets and synthetically generated content. The dataset is structured into two classes, with each class containing an approximately balanced number of examples. Additionally, the dataset includes separate test images for evaluating the model’s performance. A structured folder format or a labels.csv file accompanies the dataset for mapping images to their respective classes during model evaluation.
## Model Architecture
For this classification task, I will use a basic Convolutional Neural Network (CNN) model due to its effectiveness in image classification tasks. The CNN model consists of several convolutional layers, followed by pooling layers to reduce dimensionality, and fully connected layers to make final predictions. A sigmoid activation function is used in the output layer to classify images into one of the two classes.

## Downloading dataset
The dataset was downloaded from kaggle. [Download page ](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification)

## Using the predict function.
If input is a list of images, we can use the predict function as it is.
If input is similar to a data folder, with subfolders containing examples of each class, then please use the validate_data_folder function with the data folder path as input.

## Evaluation Metric
The performance of the model is evaluated using accuracy, which measures the proportion of correctly classified images out of the total number of images. Accuracy is a straightforward and widely used metric for classification tasks, providing a clear indication of the model's overall performance in identifying the correct class for each image