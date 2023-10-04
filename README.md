# ResDeepSurv
Survival analysis, as a widely used method for analyzing and predicting the timing of event occurrence, plays a crucial role in the field of medicine. Medical profes- sionals utilize survival models to gain insights into the re- lationship between patient covariates and the effectiveness of various treatment strategies. This knowledge is essential for the development of treatment plans and the enhance- ment of treatment approaches. Conventional survival mod- els, such as the Cox proportional hazards model, require extensive feature engineering or prior knowledge to facili- tate personalized modeling. To overcome these limitations, we propose a novel residual-based self-attention deep neu- ral network for survival modeling, called ResDeepSurv, which combines the advantages of neural networks and the Cox proportional hazards regression model. The proposed model in our study explicitly simulates the distribution of survival time and the relationship between covariates and outcomes without making any underlying assumptions. This approach effectively accounts for both linear and non- linear risk functions in the analysis of survival data. The performance of our model in analyzing survival data with various risk functions is on par with or even superior to that of other existing survival analysis methods. Further- more, we validate the superior performance of our model in comparison to currently existing methods by conducting evaluation on multiple publicly available clinical datasets. Through this study, we demonstrate the effectiveness of our proposed model in the field of survival analysis, pro- viding a promising alternative to traditional approaches. The application of deep learning techniques and the ability to capture complex relationships between covariates and survival outcomes without relying on extensive feature en- gineering make our model a valuable tool for personalized medicine and decision-making in clinical practice.
# Overview
![pic11](https://github.com/HaoWuLab-Bioinformatics/ResDeepSurv/assets/55370215/bdcf544e-9c3b-4dfe-810e-b30d828973a6)
# Dependency
config	0.5.1	
h5py	2.9.0	
hdf5	1.10.4	
numpy	1.17.3	
pandas	0.25.3	
pytorch	1.11.0	
scikit-learn	0.23.2	
# Introduction 
The 'model' file is the code used to build the model
’Datasets' is the code that processes data
'ibs' is the code used to calculate IBS scores
'95ci' is the code for calculating confidence intervals
'recommended 'is the code related to recommendation algorithms
# Usage
main.py
