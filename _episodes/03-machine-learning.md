---
title: Machine Learning for Physical Sciences
teaching: 1
exercises: 0
questions:
- "What are the fundamental concepts in ML I can use in sklearn framewrok ?"
- "How do I write documentation for my ML code?"
- "How do I train and test ML models for Physical Sciences Problems?"
objectives:
- "Gain an understanding of fundamental machine learning concepts relevant to physical sciences."
- "Develop proficiency in optimizing data preprocessing techniques for machine learning tasks in Python."
- "Learn and apply best practices for training, evaluating, and interpreting machine learning models in the domain of physical sciences."
keypoints:
- "Data representations are crucial for ML in science, including spatial data (vector, raster), point clouds, time series, graphs, and more"
- "ML algorithms like linear regression, k-nearest neighbors,support vector Machine, xgboost and random forests are vital algorithms"
- "Supervised learning is a popular ML approach, with decision trees, random forests, and neural networks being widely used"
- "Fundamentals of data engineering are crucial for building robust ML pipelines, including data storage, processing, and serving"
---



# Machine Learning Concepts

Machine learning is a field of study that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves


## Types of ML Algorithms

Types of Machine Learning
There are three main categories of machine learning algorithms:

- Supervised Learning: Algorithms learn from labeled training data to predict outcomes. It includes classification (predicting categories) and regression (predicting numerical values).
- Unsupervised Learning: Algorithms find hidden patterns in unlabeled data. A common task is clustering, which groups similar examples together.
- Reinforcement Learning: Algorithms learn by interacting with an environment and receiving rewards or penalties for actions to maximize performance

## Preprocessing

Data preprocessing describes the steps needed to encode data with the purpose of transforming it into a numerical state that machines can read. Data preprocessing techniques are part of data mining, which creates end products out of raw data which is standardized/normalized, contains no null values, and more.

### Replacing Null Values 

Replacing null values is usually the most common of data preprocessing techniques because it allows us to have a full dataset of values to work with. To execute replacing null values as part of data preprocessing, I suggest using Google Colab or opening a Jupyter notebook. For simplicity’s sake, I will be using Google Colab. Your first step will be to import `SimpleImputer` which is part of the `sklearn` library. The SimpleImputer class provides basic strategies for imputing, or representing missing values.

~~~
from sklearn.impute import SimpleImputer
~~~
{: python}

Next, you’re going to want to specify which missing values to replace. We will be replacing those missing values with the mean of that row of the dataset, which we can do by setting the strategy variable equal to the mean.

~~~
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
~~~
{: .python}

The imputer fills missing values with some statistics (e.g., mean, median) of the data. To avoid data leakage during cross-validation, it computes the statistic on the train data during the fit and then stores it. It then uses that data on the test portion, done during the transform. 

~~~
imputer.fit(X[:, 1:3]) #looks @ rows and columns X[:, 1:3] = imputer.transform(X[:, 1:3])
~~~
{: .python}

### Feature Scaling

Feature scaling is a data preprocessing technique used to normalize our set of data values. The reason we use feature scaling is that some sets of data might be overtaken by others in such a way that the machine learning model disregards the overtaken data. The sets of data, in this case, represent separate features.

Normalization puts all values between 0 and 1. However, normalization is a recommended data preprocessing technique when most of your features exhibit a normal distribution – which may not always be the case.


## Evaluation Metrics and Model Selection

## Supervised Learning



### Regression

### Classification

* Classification is a **supervised learning** task where the objective is to predict **categorical labels** for new instances based on past observations. It is widely used in various applications, such as email filtering, medical diagnosis, and image recognition. 
* In the context of physical sciences, particularly climate science, classification can be used to categorize weather patterns, predict climate events, and classify types of vegetation based on satellite imagery.

* **Types of Classification**

**Binary Classification**:  involves distinguishing between two classes.

_**Examples**_:
- Spam vs. Non-Spam: Identifying whether an email is spam or not.
- Disease vs. No Disease: Diagnosing whether a patient has a disease or not.
* Climate Example: Predicting whether a day will be rainy or not based on weather data.

**Multiclass Classification**: involves distinguishing among more than two classes.

_**Examples**_:
- Handwritten Digit Recognition: Classifying digits from 0 to 9.
- Animal Species Classification: Identifying different species of animals.
- Climate Example: Classifying weather events into categories such as sunny, cloudy, rainy, and snowy.

**Multilabel Classification**: involves assigning multiple labels to each instance.

_**Examples**_:
- Image Tagging: Identifying multiple objects in an image.
- News Categorization: Categorizing news articles into multiple topics.
- Climate Example: Classifying a satellite image of a region by both vegetation type and the presence of various climate phenomena (e.g., drought, flooding).

**Example: Climate Classification**
To illustrate classification in climate science, consider the task of classifying weather patterns. This can help meteorologists predict and understand climate phenomena, leading to better preparation and response strategies.

**Problem Statement**: Predict the type of weather (e.g., sunny, cloudy, rainy, snowy) for a given day based on historical weather data.
**Dataset** : 
    * Features: Temperature, humidity, wind speed, atmospheric pressure, previous day's weather.
    * Labels: Weather type (sunny, cloudy, rainy, snowy).


* To thoroughly explore classification algorithms, we will examine the following examples and predict whether it will rain tomorrow using various classification techniques.


###### Importing libraries
~~~
import numpy as np
import pandas as pd
~~~
{: .python}

##### Loading Dataset
~~~
data = pd.read_csv('weatherAUS.csv')
data.head()
~~~
{: .python}
~~~
Date	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	...	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RainTomorrow
0	01-12-2008	Albury	13.4	22.9	0.6	NaN	NaN	W	44.0	W	...	71.0	22.0	1007.7	1007.1	8.0	NaN	16.9	21.8	No	No
1	02-12-2008	Albury	7.4	25.1	0.0	NaN	NaN	WNW	44.0	NNW	...	44.0	25.0	1010.6	1007.8	NaN	NaN	17.2	24.3	No	No
2	03-12-2008	Albury	12.9	25.7	0.0	NaN	NaN	WSW	46.0	W	...	38.0	30.0	1007.6	1008.7	NaN	2.0	21.0	23.2	No	No
3	04-12-2008	Albury	9.2	28.0	0.0	NaN	NaN	NE	24.0	SE	...	45.0	16.0	1017.6	1012.8	NaN	NaN	18.1	26.5	No	No
4	05-12-2008	Albury	17.5	32.3	1.0	NaN	NaN	W	41.0	ENE	...	82.0	33.0	1010.8	1006.0	7.0	8.0	17.8	29.7	No	No
5 rows × 23 columns
~~~
{: .output}


## Ensemble Learning Techniques

## Unsupervised Learning

## Hyperparameter Tuning and Optimization

## Machine learning modelling for spatial data
### sklearn-xarray
### Pyspatialml





