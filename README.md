# Power_Spectrum_Generation_And_Analysis_Using_ML
Generating Power Spectrum And Predicting Astrophysical Parameters from Power Spectrum Data using Machine Learning

This research project focuses on predicting astrophysical parameters, namely 'HII_EFF_FACTOR', 'ION_Tvir_MIN', and 'R_BUBBLE_MAX', based on power spectrum data in astrophysics. The power spectrum data captures the statistical properties of the early universe, and by understanding the relationship between the power spectrum and astrophysical parameters, we gain insights into the underlying physical processes.

The research workflow involves three key steps:

1. Data Generation and Cleansing:
   - Generate a large dataset of power spectra by varying the astrophysical parameters within specified ranges.
   - Cleanse the data by removing irrelevant information and handling any inconsistencies or missing values.

2. Machine Learning Model Development:
   - Implement machine learning models to predict the astrophysical parameters based on the power spectrum data.
   - Two approaches are explored:
 	- Approach 1: Regression Method:
   	- Utilize dense and convolutional neural networks (CNN) to establish a relationship between power spectra and parameters.
   	- Train the models using the generated dataset and evaluate their performance.

 	- Approach 2: Classification Method:
   	- Transform the parameter prediction problem into a multi-class classification task.
   	- Assign classes to the parameter values based on their ranges.
   	- Train dense and CNN models to classify the parameters using power spectrum data.

3. Model Evaluation and Analysis:
   - Assess the performance of the developed models using appropriate evaluation metrics such as mean absolute error, R-squared score, accuracy, precision, recall, and F1-score.
   - Compare the results of the regression and classification approaches to determine the most effective method for predicting astrophysical parameters.

The project includes the necessary code, data, and documentation to reproduce and understand the research. The code is organized into distinct sections, including data generation, data cleansing, model development, training, and evaluation. The accompanying documentation provides detailed explanations of the methods employed and the results obtained.

The research findings contribute to the field of astrophysics by advancing our understanding of the relationship between power spectra and astrophysical parameters. The predictive models developed through this research can be utilized to estimate these parameters for new power spectra, enabling further exploration and analysis of the early universe.
