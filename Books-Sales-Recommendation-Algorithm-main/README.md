# Books Promotions Recommendation Algorithm

# Obective 
* Create a machine learning solution that will help users or small companies to filter the best books promotions according to theirs personal interests on 
the Brazilian website "Mercado Livre". 

# Overview
* Construction of  a full machine learning solution following this steps:
 
  * Make the Web Scrapping
  
  * Process the collected Data
  
  * Build,Optimize and Validate some diferents ML algorithms like : Random Forest,Light GBM and Logistical Regression
  
  * Merge the best models in a single ensanble with better performance
  
  * Make the deploy of the final ensamble  

# Code and Resources : 
**Python Version:** 3.7

**Libraries:** pandas , numpy , requests , sklearn , yellowbrick 

**Website:** https://www.mercadolivre.com.br/

# Web Scrapping 
* The data was collected based on the folowwing search queries : 'financas",'ficcao-cientifica','exatas'. Initially, I scpraped the 20 first pages for each query and extract all the advertaisments links from them.  

``` python
# Colleting the links of each adverstaisment from the main webpage 
# Selected queries
queries = ["financas",'ficcao-cientifica','exatas']

# # Collecting the data from the main search page for each query  
for query in queries:
    for i in range(1,21):
        if i == 1:
            url =(f'https://livros.mercadolivre.com.br/livros/{query}')  
        else:
            page = (48*i - 47)
            url =(f'https://livros.mercadolivre.com.br/livros/{query}/_Desde_{page}')

        print(url)
        response = rq.get(url)
        with open(f'{query}_pg.{i}.html', 'w+',encoding='utf-8') as output:
            output.write(response.text)
            time.sleep(2) 
            
# Scraping the individual data From each advertsiment 
for query in queries:
    for i in range(1,21):
            
        with open(f'{query}_pg.{i}.html','r+',encoding='utf-8') as inp:
            page_html = inp.read()
            parsed = bs4.BeautifulSoup(page_html)
            tags = parsed.findAll('a')
            
            for e in tags:
                if e.has_attr('class') and e.has_attr('title'):
                    if e['class'] == ['ui-search-link']:
                    link = e['href']
                    title = e['title']
                    with open("parsed_videos.json", 'a+') as output:
                          data = {"link": link, "title": title,"query" : query}
                          output.write("{}\n".format(json.dumps(data)))

``` 

* Lately, I accessed the advertisements webpage and extracted the wished data by searching then for their HTML tags. The extracted features were : Price , Author , Title , Total Sale , Publisher ,Language and Format 


``` python
for link in lista_links:
    try:
        with open(f'{link[36:49]}.html','r+',encoding='utf-8') as inp:
                    page_html = inp.read()
            
                    parsed = bs4.BeautifulSoup(page_html)
            
                    # Collecting the data of the Prices 
                    class_price = parsed.find_all(attrs={"itemprop":re.compile(r"price")})[0]
                    price.append(float(class_price['content']))
            
                    # Collecting the data of the Total Sales
                    try:
                        class_sales = parsed.find_all(attrs={"class":re.compile(r"ui-pdp-subtitle")})
                        sales.append(class_sales[0].text.strip())
                    except:
                        continue
```



* After finished the scraping, I saved all the data in a CSV file to add the properly classification of the ads.  (Non-Interasting : 0 or Interasting : 1)  


# Data Cleaning
* It was necessary to convert the 'Total Sales' feature from a text to a integer number. And for this, I use the .replace() function to remove the strings that accompanied the numbers. 


``` python
strings = df['sales']
newsales = []
for e in strings:
    try:    
        try:
            k = e.replace('Novo  |  ','')
            try:
                newsales.append(float(k.replace('vendidos','')))
            except:
                newsales.append(float(k.replace('vendido','')))
        except:
                k = e.replace('Usado  |  ','')
                try:
                    newsales.append(float(k.replace('vendidos','')))
                except:
                    newsales.append(float(k.replace('vendido','')))       
    except:
        newsales.append('null')
        continue
newsales = pd.DataFrame(newsales)
``` 


* Futhermore, some advertisements were poor edited and all of their values were missing, so it was necessary drop them from the data set.


```python
# Importing the raw data directly from a feather file previously generated 
df = pd.read_feather('raw_data_unclean.feather').drop(['formato','language','editora'],axis=1)

df.isnull().sum()

titles    316
price       0
sales       6
author    341
dtype: int64

df = df.dropna()

df.isnull().sum()
titles    0
price     0
sales     0
author    0
dtype: int64
```

# Setting the Labels
* First of all, its very important note that the followings criterias were used to determine if a book advertisement is interesting:
  * Advertisents of single books are more interesting than box of books
  * 'Finanças' and 'Exatas' books are more interesting than 'ficção cientifica' books
  * 'Pai Rico e Pai Pobre',Stephen Hawking books and Stock Exchange Themed Books are the preferences of the users.

# Feature Engineering 
* First of all I splited the dataset in Training and Validation 


``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Impporting the dataset with the labels already defined:
df = pd.read_csv('raw_data_wlabels.csv').drop('Unnamed: 0',axis=1).dropna()

# Spliting the dataset in training and validation
y = df['y']
xdata = df.drop('y',axis=1)

data_train,data_val,ytrain,yval = train_test_split(xdata,
                                                    y,
                                                    test_size = 0.5,
                                                    random_state=0)
``` 


* So that the models can use the title and the authors of the advertisements as features, I converted the text entries in columns 'Titles' and 'Authors' to  sparse matrixs using the TfidfVectorizer method. 


* The TfidfVectorizer will generate a matrix that relate each word that appears in the strings with the occurency of them in the features. Besides that, the convertion of a matrix to a sparse matrix is necessary to supress the null elements, so the stored information is much smaller than in an ordinary matrix structure.


``` python
# Vetorazing the titles strings
title_train = data_train['titles']
title_val = data_val['titles']

title_vec = TfidfVectorizer(min_df = 2,ngram_range=(1,1))

title_bow_train = title_vec.fit_transform(title_train)
title_bow_val = title_vec.transform(title_val)

# Vetorazing the Author strings
author_train = data_train['author']
author_val = data_val['author']

autor_vec = TfidfVectorizer(min_df = 2,ngram_range=(1,2))

autor_bow_train = autor_vec.fit_transform(autor_train)
autor_bow_val = autor_vec.transform(autor_val)

```
* After that, to merge all the dataset again, I used the .hstack() function from Scipy librarie. This function allow me to concatanet horizontally the sparse matrixs and the commom features.

```python
from scipy.sparse import hstack
# Sorting out the "simple features"
mask_train = data_train.drop(['titles','author'],axis=1)
mask_val = data_val.drop(['titles','author'],axis=1)

# Merging all the data again
xtrain_wvec = hstack([title_bow_train,autor_bow_train,mask_train])
xval_wvec = hstack([title_bow_val,autor_bow_val,mask_val])
```

# Model Building
* The final dataset had no more than 1200 advertisements, and just 340 were classified as Interesting (y = 1 ) 

* So, to avoid the overfiting problem an analisys using the confusion matrix were used. Also that,the followings metrics were used to validate the models:Mean Average Score, Roc Auc Score.Because the need to ranking the best advertisement I chose to test the followings algorithms: Random Forest,Logistical Regression,LightGBM.

 * Random Forest : 
 ```python
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import average_precision_score , roc_auc_score

rdf = ConfusionMatrix(RandomForestClassifier(n_estimators = 1000,random_state=0,min_samples_leaf=3,class_weight = 'balanced', n_jobs=6))
rdf = rdf.fit(xtrain_wvec,ytrain)
rdf.score(xval_wvec, yval)
rdf.poof()
``` 
![RDFMetrics](https://github.com/gui-miranda/Books-Sales-Recommendation-Algorithm/blob/main/Images/rdf.matrix.PNG)

```python
# Forecasting the probabilities
p = rdf.predict_proba(xval_wvec)[:,1]
print(f'Random Forest Metrics \nAVG : {average_precision_score(yval,p)} \nROC : {roc_auc_score(yval,p)}')

Random Forest Metrics 
AVG : 0.48340197848954136 
ROC : 0.6663858403912681
```
 

* Logistic Regression : 
  * Because the nature of the Logistic Regression and the presence of the sigmoid function in its construction, before build this model I first normalized the data entries
```python
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import average_precision_score , roc_auc_score

# Normalizing the data
scaler = MaxAbsScaler()

xtrain_wvec2 = scaler.fit_transform(xtrain_wvec2)
xval_wvec2 = scaler.transform(xval_wvec2)

lr = ConfusionMatrix( LogisticRegression(C=30,n_jobs=6,random_state=0))
lr.fit(xtrain_wvec2,ytrain)
lr.score(xval_wvec2,yval)
lr.poof()
```

![LRMetrics](https://github.com/gui-miranda/Books-Sales-Recommendation-Algorithm/blob/main/Images/lr.matrix.PNG)

```python
# Forecasting the probabilities
p = lr.predict_proba(xval_wvec2)[:,1]
print(f'Logistical Regression Metrics \nAVG : {average_precision_score(yval,p)} \nROC : {roc_auc_score(yval,p)}')

Logistical Regression Metrics 
AVG : 0.5169900249550464 
ROC : 0.7117976857926757

```
 * LightGBM
 
 ```python
 from lightgbm import LGBMClassifier
 from yellowbrick.classifier import ConfusionMatrix
 from sklearn.metrics import average_precision_score , roc_auc_score
 
 lgbm = ConfusionMatrix(LGBMClassifier(random_state = 0,class_weight ='balanced',n_jobs=6))
 lgbm = lgbm.fit(xtrain_wvec,ytrain)
 lgbm.score(xval_wvec, yval)
 lgbm.poof()
 ```
 ![LGBMMetrics](https://github.com/gui-miranda/Books-Sales-Recommendation-Algorithm/blob/main/Images/lgb.matrix.PNG)
 
 ```python
 p = lgbm.predict_proba(xval_wvec)[:,1]
 print(f'AVG : {average_precision_score(yval,p)} \nROC : {roc_auc_score(yval,p)}')
 AVG : 0.3434728669544407 
 ROC : 0.5557303471310986
``` 

* Performing a Baysean Optmization

   * As the features were already properly prepared, the most viable way to improve the performance of this model would be to tune its hyperparameters. However, as it is a more complex model than the previous ones, changing each parameter manually would be very difficult. So I decided to use a Baysean Optimization to determine which parameters would return the highest value for the average_precision_score metric.
   * In a Baysean optimization a model is trained inside a function generated by the user. This function uses the input values ​​as the hyperparameters of the model, in this case the model is based on LightGBM algorithm. After being trained, the function returns one of the model's performance metrics as a value.
After the construction of the function, a variation space for the input variables is defined. Thus, each variable can take any value within these created spaces. Then, this variable space created is used as the inputs of the function that will train the model, and through the skopt library we can use the forest_minimize() function to vary the inputs within the defined spaces and thus find different values ​​for the desired metric . The greater the number of tests allowed, the more likely it is to find better values for the metric.
   
``` python 
from skopt import forest_minimize

# Creating the function that will train the model
# The whole feature engineering will be done again 

def tune_lgbm(params):
    print(params)
    
    # Defining the hyperparameters as inputs
    lr = params[0]
    max_depth = params[1]
    min_child_samples = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    n_estimators = params[5]
    min_df = params[6]
    ngram_range = (1,params[7])
    
    # Vetorazing the titles and authors strings
    title_vec = TfidfVectorizer(min_df = min_df,ngram_range=ngram_range)
    title_bow_train = title_vec.fit_transform(title_train)
    title_bow_val = title_vec.transform(title_val)
    autor_vec = TfidfVectorizer(min_df = min_df,ngram_range=ngram_range)
    autor_bow_train = title_vec.fit_transform(autor_train)
    autor_bow_val = title_vec.transform(autor_val)
    
    # Merging the dataset once more
    xtrain_wvec = hstack([title_bow_train,autor_bow_train,mask_train])
    xval_wvec = hstack([title_bow_val,autor_bow_val,mask_val])
    
    # Building and Training the model
    mdl = LGBMClassifier(learning_rate=lr, num_leaves= 2**max_depth, max_depth=max_depth,
                         min_child_samples=min_child_samples, subsample=subsample,
                         colsample_bytree=colsample_bytree, bagging_freq = 1, n_estimators=n_estimators, 
                         random_state = 0, class_weight ='balanced',n_jobs=6)
    mdl.fit(xtrain_wvec,ytrain)
    
    # Forecasting the probabilities
    p = mdl.predict_proba(xval_wvec)[:,1]
    return -average_precision_score(yval,p)
    
    # Defining the variation space of parameters
    space = [(1e-3,1e-1,'log-uniform'), #lr
         (1,10),    # max_depth
         (1,10),    #min_child_samples
         (0.05,1.), #subsamples
         (0.05,1.), #colsample_bytree
         (100,1000),#n_estimators
         (1,5),     #min_df
         (1,5)      #ngram_range 
        ] 
# Starting the tunning function
res = forest_minimize(tune_lgbm,space,random_state=160745,n_random_starts=20,n_calls=50,verbose=1)
``` 

* So, after retraining the model with the best hyperparametrs found, this were the results for the LightGBM metrics:

```python
LGBM Metrics 
AVG : 0.5296939830564515 
ROC : 0.6951121316950972
``` 



* And then, to create a solution capable of handling a wider variety of situations I built an ensamble that combines the three models probabilities acording to te following realation:
  
  ```python
  p = (0.25*p_rd + 0.25*p_lr + 0.5*p_lgbm)
  
  Average_Precison_Score : 0.5512737816445318 
  ROC_auc_score          : 0.7172849815101994
  ```
  
 * This is a important step in the solution because , as its possible to see in the confusion matrixs, some models are better  predicting NEGATIVES (Logistic Regression and Random Forest) responses while other are better predcting POSITIVES (LightGBM) responses.
 * I chose this proportions, because its significantly increased the metrics and put the major part in predict probabilitie in the hands of the model with best metrics  (LightGBM)
 
 # Deploy
 * To deploy the resultant ML model and then use it to make real recomendations, I adapted the whole project (From Web Scraping to Ensamble Model Predictions) for a Spyder-script. I chose do the deploy this way because is relative simple to run a python file using Spyder, and Its a good replacement for building a more complex API that I don't have domain yet
 
 ![Capture](https://github.com/gui-miranda/Books-promotions-recommendation-algorithm/blob/main/Images/Capture.PNG)
 * The result are a csv file that returns the 15 more interasting advertisements, like this one:
 
 ![Results]( https://github.com/gui-miranda/Books-promotions-recommendation-algorithm/blob/main/Images/Results.PNG)

   


