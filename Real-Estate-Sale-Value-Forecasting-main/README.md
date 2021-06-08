# Real Estate Sale Value Forecasting
# Overview
* Conduct an analysis of real estate sales data in King County-Washington,D.C. 
* Analyzes performed on approximately 22 thousand transactions
* Construction of a Linear Regression model to forecast the sale value of properties

# Code and Resources Used
**Python Version:** 3.7

**Libraries:** pandas , numpy , matplotlib , plotly , sklearn , scipy , seaborn , 

**Original Dataset:** https://www.kaggle.com/shivachandel/kc-house-data

# Data Cleaning
* The dataset did not have any irregularities to be corrected, that is, there were no null values and the data entries already had adequate classes.
* At this stage, the only correction was to remove the 'id' and 'date' columns that would not be useful in building the model
``` python 
# Importing the libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly 
from plotly.offline   import iplot,init_notebook_mode
import plotly.graph_objs as go
from cufflinks.offline import go_offline

init_notebook_mode(connected = True)
go_offline(connected=True)

# Importing the dataset and droping the columns
df = pd.read_csv('kc_house_data.csv')
df = df.drop(['date','id'],axis=1)

# Searching for null values
fig = plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(),cmap ='viridis');
fig.savefig('teste.png')
``` 
![image](https://user-images.githubusercontent.com/82520450/121061652-8084a600-c79a-11eb-8a6e-ec091db911de.png)


![image](https://user-images.githubusercontent.com/82520450/121061720-97c39380-c79a-11eb-9e66-380f1ba063de.png)

# Exploratory Data Analyses 
* Initially I tried to filter the data in order to select the regions that simultaneously had
a large amount of sales and also a high sales value per square meter. Defining then,
some zipcodes as being the most favorable for investment.

![bar_plots](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/bar_plot.PNG)

![ZipCodes](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/a-top_zipcodes.PNG)
   

* Within these 'most favorable regions' the data are explored in order to better understand
the type of real estate marketed in them.

![Quartos](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/Graphs.PNG)

* In addition, looking at the chart below, you can get an idea of the values of properties traded in this region.
```python
# Generating the average graph 
df_mean = pd.DataFrame(range(553),columns=['clas'])
df_mean['Values'] = range(553)

for i in range(553):
    df_mean['Values'].loc[i] = valores.mean()


# Ploting the graph
plt.figure(figsize=(15,8))
plt.title('House Sales Prices in the Golden Region')
plt.ylabel('Million Dollars')
plt.xlabel('Index')
plt.bar(nclas,valores,color = 'Red')
plt.plot(range(553),df_mean['Values'],color='Black',label='Valor Médio = 0.576')
plt.legend()
plt.show()
```
![Histograma](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/pre%C3%A7o%20de%20venda.PNG)

# Feature Engineering
* It is important to point out that the worked features will all belong to properties traded in the previously "most favorable region", and consequently  model will generate sales value forecasts only for these areas. 

* Initially, I analyzed the correlations between the features and the target values ('prices') obtaining the following result:
```python
df = df.drop(['id,date],axis=1)
corr = df.corr()
corr.iplot(kind='heatmap',colorscale='Blues',hoverinfo='all',
           layout = go.Layout(title='Correlation Heatmap',titlefont=dict(size=20),autosize = False ,width = 700,height=500))
```
![image](https://user-images.githubusercontent.com/82520450/121062473-7e6f1700-c79b-11eb-92af-d0fd92171b89.png)


* Assuming as strong correlations the ones with |z| > 0.15 , the features wished reduced to the followings:
```python
bad_ft =['condition','yr_built','yr_renovated','zipcode','long','lat','waterfront']
ndf = df.drop(bad_ft,axis=1)
``` 
![DF](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/a-%20df1.PNG)

* However, analyzing once again the matrix of correlations between the features, I noticed the presence of strong dependencies between the features themselves. Thus, to avoid multicollinearity problems that would compromise the stability and quality of the final results, the features chosen to describe the targets were as follows:
```python
n2df = ndf.drop(['bedrooms','bathrooms','sqft_living15','sqft_above','sqft_lot15','view','floors'],axis=1)
n2df.head(3)
```
![df_2](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/a%20-%20df2.PNG)


* Next, I analyzed the presence of outliers in the data taken to avoid possible learning problems in the model to be generated.It is important to know that outliers do not always represent incorrect data, it is possible that they also represent totally possible variations of the analyzed variable. Therefore, it is necessary to create a treshhold 
a little higher, that is, to allow some extreme values to be present in the data set so that the model can learn with a greater variety of situations.
```python 
# Analyzing Outliers
plt.figure(figsize=(20,12))
plt.subplot(2,2,1)
sns.boxplot(y='sqft_living',data=n2df)
plt.title('Sqft_Living Box Plot')

plt.subplot(2,2,2)
sns.boxplot(y='sqft_basement',data=n2df)
plt.title('Sqft_Basement Box Plot ')

plt.subplot(2,2,3)
sns.boxplot(y='sqft_lot',data=n2df)
plt.title('Sqft_Lot Box Plot ')

plt.subplot(2,2,4)
sns.boxplot(y='price',data=n2df)
plt.title('Price Box Plot ')
plt.show()
```
![box_plot](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/a-box_plot1.PNG)


* So, after creating a treshhold for each analyzed variable, I filtered the  outliers. 
```python
# Example of construction of trashholds
# Finding OutLiers in Sqft Living, and Building a ThresHold
sqft_liv= n2df['sqft_living']

# Defining the quartiles 
q25,q75 = np.percentile(sqft_liv,25),np.percentile(sqft_liv,75)
print(f'q25 = {q25}, q75 = {q75}')

liv_iqr = q75-q25
print(f'iqr = {liv_iqr}')

liv_cutoff = liv_iqr*1.95
liv_upper = (q75+liv_cutoff)
print(f'Cut Off = {liv_cutoff}')
print(f'Sqft_Living Upper = {liv_upper}')

# Finding the outliers 
outliers = [x for x in sqft_liv if x >liv_upper]
print(f'Feature Sqft_Living Outliers {len(outliers)}')  

q25 = 1220.0, q75 = 2080.0
iqr = 860.0
Cut Off = 1677.0
Sqft_Living Upper = 3757.0
Feature Sqft_Living Outliers 6

# Removing the outliers founded from the main dataset
cut_out_prices =  n2df.drop(n2df[(n2df['price'] > price_upper) | (n2df['price'] < price_lower) ].index)
cut_outl_living = cut_out_prices.drop(cut_out_prices[(cut_out_prices['sqft_living'] > liv_upper)].index)
cut_out_basement = cut_outl_living.drop(cut_outl_living[(cut_outl_living['sqft_basement'] > base_upper)].index)
cut_out_lot = cut_out_basement.drop(cut_out_basement[(cut_out_basement['sqft_lot'] > lot_upper) | (cut_out_basement['sqft_lot'] < lot_lower) ].index)

new_df = cut_out_lot
```
* In order to have a better understanding of the data, I performed an analysis on the distribution of the features in question
``` python
# Analyzing the distribution of selected features and price values 
plt.figure(figsize=(18,11))
plt.subplot(2,2,1)
sns.distplot(new_df['sqft_living'],color ='Red')
plt.title('Sqft_Living Distribution')

plt.subplot(2,2,2)
sns.distplot(new_df['sqft_basement'],color='black')
plt.title('Sqft_Basement Distribution')

plt.subplot(2,2,3)
sns.distplot(new_df['sqft_lot'],color='orange')
plt.title('Sqft_Lot Distribution')

plt.subplot(2,2,4)
sns.distplot(new_df['price'],color = 'blue')
plt.title('Price Distribution')
plt.show()
```
![dist_plots](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/1-dist.PNG)


* Based on this analysis, I chose to normalize the data and transform their distributions as close to Gaussian as possible.This process is important because normalized data facilitates the learning of the model and at the same time avoids a bias in the predictions due to the difference between the orders of magnitude of the training values
```python
# Example of the feature normalization process
from scipy import stats

# Normalizing the sqft_living features
living = new_df['sqft_living'].values
n_living = (stats.rankdata(living)/(len(living)+1)-0.5)*2
n_living = (np.arctanh(n_living) + np.max(np.arctanh(n_living)) + 0.01)
new_df['sqft_living'] = n_living
```
![new_plot](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/a-n_dist.PNG)

* NOTE: The 'price' data used as target values for the training of the model did not undergo the normalization process, so the new distribution of these values shown above was only for the purpose of visualizing the method. 

# Model Building
* As I said before, the model was built from a Linear Regression algorithm, and after training the model with the data already processed in the previous steps, the resulting metrics were:

``` python
# Importing data already prepared for modeling
df = pd.read_csv('processed_data.csv')

# Separating data between training and testing
x_data = df.drop('price',axis=1)
y = df['price'].values

x_train,x_test,y_train,y_test = train_test_split(x_data,
                                                y,
                                                test_size = 0.3,
                                                random_state=0)
# Building and Training the Model
rl = LinearRegression()
rl.fit(x_train,y_train)
predicts = rl.predict(x_test)

# Evaluating the metrics
mea =  round((mean_absolute_error(y_test,predicts)/100000),5)
r2 =    round((r2_score(y_test,predicts)),5)
me_log = round((mean_squared_log_error(y_test,predicts)),5)
``` 
![métrics](https://github.com/gui-miranda/Data-Science-Projects/blob/main/Real-Estate-Sale-Value-Forecasting-main/Graphs/a-metrics.PNG)

* Analyzing the Mean Abosolute Error metric we can see that its value is in the order of 10^4 dollars, and as the predicted values are in the order of 10^6 dollars, we can interpret it as a strong indicator of low forecast errors.

* It is also possible to analyze the Mean Squared Logaritmh Error, which, due to the fact that the interpretation of the order of magnitude is exempt, indicates a low variation between forecast and real value.

* Finally, the r^2 Score metric tells us how well the model is describing the expected values, and how its value is approximately 65% also points to a satisfactory model response.

# Conclusion
* It was found that, even though the original dataset had several features with strong correlations with the target values, the effects of multicollinearity between these features ended up being a depreciation factor for the performance of the generated models, thus being necessary to remove them from the training dataset.

* And in addition, after removing outliers and normalizing the distribution of data, it was possible to generate a forecast model for the sale value of properties that presented a low error value in relation to the amount analyzed. 




