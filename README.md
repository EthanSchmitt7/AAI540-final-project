# AAI540-final-project

## Build Instructions
- Clone the repo
    `git clone <ssh/html link>`
- Install required libraries
    `pip install -r requirements.txt`
- Run the `data_generation.ipynb` script
- Run the `model_pipeline.ipynb` script
- Run the `endpoint.ipynb` script

## Script Descriptions
### test_model.ipynb
This file is not required to run this project. This file was created for initial exploration of the dataset and possible avenues for our LSTM model.

### data_generation.ipynb
This file will generate the required files for the model pipeline to be built properly and for the endpoint to simulate a production environment. It will produce two CSV files in a folder named `data`. The file named `sensor_data.csv` contains the training/evaluation/testing data for our model and will be used in the pipeline creation. The file named `production_data.csv` contains the data reserved for production testing of our model and will be used in our endpoint script to simulate the inflow of data in real time.

## model_pipeline.ipynb
This file contains the code to configure, run, and execute the pipeline for our ML model. This includes preprocessing, feature stores, training, and inference.

## endpoint.ipynb
This file simulates our model in a production environment. It deploys our model to a live endpoint and then sends queries periodically to that model so its performance can be logged and viewed in Cloudwatch on our dashboard.