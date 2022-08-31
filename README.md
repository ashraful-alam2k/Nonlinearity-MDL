# Multi-domain Learning for Neural Network-Based Equalizers in a Coherent Optical Transmission System: Solving the Flexibility Problem

This repository contains source codes and saved models for training and testing Deep Neural Network (DNN) model. 
This models were focused on mitigating nonlinearity in optical signal propagation.
We used Multi-domain learning approach (MDL) where multiple domains are different span/power/symbol rate in the transmission.

N.B. channel_model refers to module used for signal propagation with Manakov equation in 1 km step.

The trained models and source codes can be found in the scenario specific folders.

Please follow the below mentioned steps to install the required modules and packages beforehand. This will set up your environment and run the codes without hassle.

### Install 

    #python3.9
    pip install --upgrade pip
    pip install -r requirements.txt

## Results for different symbol rates (SR) scenarios.
![Different SR scenarios for 6 dBm](https://user-images.githubusercontent.com/96380861/187230486-96767673-b760-4272-ba53-c004f7687960.png)
![Different SR scenarios for 8 dBm](https://user-images.githubusercontent.com/96380861/187230847-cc7dbddc-dd5a-4e26-8ac5-6879a40397dd.png)

## Results for different span numbers scenarios.
![Different span numbers scenarios for 6 dBm](https://user-images.githubusercontent.com/96380861/187390130-1489d0e3-ba8d-47dd-a8e7-130488943d7c.png)
![Different span numbers scenarios for 8 dBm](https://user-images.githubusercontent.com/96380861/187390138-05cee1ed-75d1-426d-8fa6-38c2718e4f08.png)


## Results for different launch power scenarios.
![MicrosoftTeams-image (9)](https://user-images.githubusercontent.com/96380861/187741228-99cb1877-e737-4b25-92b5-f88d5ec92688.png)


## Results for different span and power scenarios.
![MicrosoftTeams-image (8)](https://user-images.githubusercontent.com/96380861/187741176-5ffaa023-8aad-47a8-aefc-c902b19ea2d1.png)
