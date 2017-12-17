# ID2223-project

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
