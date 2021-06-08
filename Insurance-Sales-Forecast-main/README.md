# Insurance Sales Forecast

# Overview
* This project is a simulation of a bussines problem usually experienced by insurance companies.The objective is condute an analyses trought a dataset of vehicle insurance users who also adhere to combined health insurance.For this purpose I create a machine learning solution that will help an insurance company to detect who of your users will adhere to a combined insurance plan. I perfomed my analyses on approximately 381,000 user data and use LightGBM, KNN and Random Forest algorithms to construct my models.

# Code and Resources Used:
**Python Version:** 3.7

**Libraries:** pandas , numpy , matplotlib , plotly , folium , sklearn , yellowbrick 

**Original Dataset:** https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

# Data Cleaning
 * The dataset had no irregularities to be corrected, that is, there were no missing values and the data entries already had adequate classes.

# Exploratory Data Analyses
* Initially, in order to have a better understanding of the characteristics of the customers that made up the data set, the
following analyses :  

* Distribution by gender of customers, where we can see that there is a balance between the two sexs: 
![Gender](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gr%C3%A1ficos/zGender.PNG)

* Distribution by age of customers, which shows a main concentration between 21 to 25 years and a secondary one between 40 - 50 years :
![Age](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/zCAge.PNG)

* Furthermore, crossing the two previous analyses, we can see that for women the concentration of clients between 21-25 years old is significantly higher than for men in the same age group : 
!['Gender/Age](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/zGenderAge.PNG)

* I found that virtually all customers have a driver's license, which certainly makes it possible to sell vehicle insurance.
![license](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/zLicence.PNG)

* Surveying the vehicle data I concluded that most customers have cars with a maximum of 1 year of use:
![VAge](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/zVAge.PNG)

* Also, I found that half of the customers had already damaged their vehicles, which can increase the chances of selling the insurance.:
![VDamage](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/zVD.PNG)

* Finally, when performing the last analysis of this step, I found that only 12.3% of customers had signed up for vehicle insurance. At this point, I have already paid attention to a possible bias in the machine learning models to be generated.
![Adhesion](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/zVeicAdesion.PNG)

# Model Building
* At first, I used a variety of algorithms in order to find difficulties and facilities in learning each of the models.

* The algorithms used were: LightGBM, RandomForest and KNearest Neighbors

* First of all, I trained the models with data containing a certain degree of blurring between POSITIVE and NEGATIVE responses (Only 12% of cases were POSITIVE), and obtained the following metrics for each model:
![Metrics](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/Metrics.PNG) 

* It is important to emphasize that I used the *AUC and Accuracy* metrics as the basis for comparing the performance between the models, evaluating through the other metrics a possible bias or overfitting of the models.

* When creating a new dataset with equal proportions between POSITIVE and NEGATIVE responses, I retrained each of the previous models with this new data. However, the only model to show significant variation was that of KNearest Neighbors. Soon after, I performed an optimization of its K parameter to obtain the highest values for the analyzed metrics.

![KnnOPT](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/KNN_Graph.PNG) 
!![KnnOPT2](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/KNN%20Opt.PNG)

* Furthermore, as the model using LighGBM proved to be one of the best performing ones, I applied a *Bayesian Optimization* to it so that I could find the parameters that resulted in the highest possible value for *Precision* in relation to the true predictions.
![OPTLGBM](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Insurance-Sales-Forecast-main/gráficos/LGBM%20OPT.PNG)

# Conclusion
* As seen, after individual optimizations and data balancing, the three models performed with similar results 
* However, due to its shorter processing time to generate the results, the LightGBM model is the most recommended to perform the predictions of this dataset.
