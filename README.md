# Purpose

The purpose of this repository is to share the code and models developed in during the Predictive Analysis internship project done by Rishit Arora, Avani Yerva and Max Rontal

# Installation

- Clone this repository and install the necessary Python packages
- Make sure to have the data_preprocess.py file in the same directory as the other .ipynb files as they are dependent on the python file

# Usage

Many functions and lines of code have comments that explain what they do
- basic_graphs.ipynb is used to visualize basic FFT graphs
- BEA_with_STFFT_and_SK.ipynb is used to perform bearing envelope analysis with information gathered through spectral kurtosis of Short Time Fast Fourier Transform on the microphone data
- data_preprocess.py is used to preprocess the microphone data such as performing FFT and cutting off certain frequencies
- ml_analysis_cnn_inside_oven.ipynb is used to create models
- model_testing.ipynb is used to test models that are created
- convert_model.ipynb is used to convert the model to ONNX format for the Microsoft.ML.OnnxRuntime library to use in .NET Desktop Applications (Deployment)