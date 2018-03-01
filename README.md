# ID2223-project
Final project in the the course ID2223 Scalable Machine Learning and Deep Learning.

This repository contains code for a convolutional neural network that is able to colorize images. 

Main files are located under *hops-notebooks/* and contain jupyter notebooks that run on [Hops](https://hops.site), a platform that allows users to develop and deploy stream processing projects (Flink/Spark), machine learning projects (tensorflow) and more. 

## Setup
Create container:
`docker run -p 8888:8888 --name tensorflow-id2223 -v /path/to/ID2223-project:/notebooks -it gcr.io/tensorflow/tensorflow`

## Run
Run container:
`docker start --interactive tensorflow-id2223`

## Data
Extract the data.zip file in the root folder

## Version Control
Remember to clear all output in the notebooks before commiting changes.
__Cell > All Output > Clear__
