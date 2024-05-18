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

**Dataset**  
    * Features: Temperature, humidity, wind speed, atmospheric pressure, previous day's weather.
    * Labels: Weather type (sunny, cloudy, rainy, snowy).


* To thoroughly explore classification algorithms, we will examine the following examples and predict whether it will rain tomorrow using various classification techniques.


###### 1. Importing libraries
~~~
import numpy as np
import pandas as pd
~~~
{: .python}

##### 2. Loading Dataset
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

###### 3. Understanding the Data Structure
* Check the shape, data types, and basic statistics of the dataset.
~~~
# Shape of the dataset
print("Shape of the dataset:", data.shape)
~~~
{: .python}

~~~
Shape of the dataset: (145460, 23)
~~~
{: .output}

~~~
# Data types and non-null counts
data.info()
~~~
{: .python}

~~~
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 145460 entries, 0 to 145459
Data columns (total 23 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   Date           145460 non-null  object 
 1   Location       145460 non-null  object 
 2   MinTemp        143975 non-null  float64
 3   MaxTemp        144199 non-null  float64
 4   Rainfall       142199 non-null  float64
 5   Evaporation    82670 non-null   float64
 6   Sunshine       75625 non-null   float64
 7   WindGustDir    135134 non-null  object 
 8   WindGustSpeed  135197 non-null  float64
 9   WindDir9am     134894 non-null  object 
 10  WindDir3pm     141232 non-null  object 
 11  WindSpeed9am   143693 non-null  float64
 12  WindSpeed3pm   142398 non-null  float64
 13  Humidity9am    142806 non-null  float64
 14  Humidity3pm    140953 non-null  float64
 15  Pressure9am    130395 non-null  float64
 16  Pressure3pm    130432 non-null  float64
 17  Cloud9am       89572 non-null   float64
 18  Cloud3pm       86102 non-null   float64
 19  Temp9am        143693 non-null  float64
 20  Temp3pm        141851 non-null  float64
 21  RainToday      142199 non-null  object 
 22  RainTomorrow   142193 non-null  object 
dtypes: float64(16), object(7)
memory usage: 25.5+ MB
~~~
{: .output}

~~~
data.describe(include='all')
~~~
{: .python}

~~~

Date	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	...	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RainTomorrow
count	145460	145460	143975.000000	144199.000000	142199.000000	82670.000000	75625.000000	135134	135197.000000	134894	...	142806.000000	140953.000000	130395.00000	130432.000000	89572.000000	86102.000000	143693.000000	141851.00000	142199	142193
unique	3436	49	NaN	NaN	NaN	NaN	NaN	16	NaN	16	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	2	2
top	12-11-2013	Canberra	NaN	NaN	NaN	NaN	NaN	W	NaN	N	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	No	No
freq	49	3436	NaN	NaN	NaN	NaN	NaN	9915	NaN	11758	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	110319	110316
mean	NaN	NaN	12.194034	23.221348	2.360918	5.468232	7.611178	NaN	40.035230	NaN	...	68.880831	51.539116	1017.64994	1015.255889	4.447461	4.509930	16.990631	21.68339	NaN	NaN
std	NaN	NaN	6.398495	7.119049	8.478060	4.193704	3.785483	NaN	13.607062	NaN	...	19.029164	20.795902	7.10653	7.037414	2.887159	2.720357	6.488753	6.93665	NaN	NaN
min	NaN	NaN	-8.500000	-4.800000	0.000000	0.000000	0.000000	NaN	6.000000	NaN	...	0.000000	0.000000	980.50000	977.100000	0.000000	0.000000	-7.200000	-5.40000	NaN	NaN
25%	NaN	NaN	7.600000	17.900000	0.000000	2.600000	4.800000	NaN	31.000000	NaN	...	57.000000	37.000000	1012.90000	1010.400000	1.000000	2.000000	12.300000	16.60000	NaN	NaN
50%	NaN	NaN	12.000000	22.600000	0.000000	4.800000	8.400000	NaN	39.000000	NaN	...	70.000000	52.000000	1017.60000	1015.200000	5.000000	5.000000	16.700000	21.10000	NaN	NaN
75%	NaN	NaN	16.900000	28.200000	0.800000	7.400000	10.600000	NaN	48.000000	NaN	...	83.000000	66.000000	1022.40000	1020.000000	7.000000	7.000000	21.600000	26.40000	NaN	NaN
max	NaN	NaN	33.900000	48.100000	371.000000	145.000000	14.500000	NaN	135.000000	NaN	...	100.000000	100.000000	1041.00000	1039.600000	9.000000	9.000000	40.200000	46.70000	NaN	NaN
11 rows × 23 columns
~~~
{: .output}

### 4. Pre-Processing
When applying any predictive algorithm, we can never use it immediately without having done any pre-processing of the data. This step is extremely important, and can never be overlooked. For this data set, we perform the following pre-processing steps:

* Drop features that do not seem to add any value to our model
To determine which columns are less important for predicting the target variable RainTomorrow, we need to consider several factors:

a. **Missing Values**: Columns with a high proportion of missing values might be less useful unless they carry significant predictive power that justifies the effort to handle these missing values.

b. **Correlation with Target**: Columns with little or no correlation to the target variable might be less important.

c. **Domain Knowledge**: Certain features might be more relevant based on domain knowledge about weather prediction.

**Calculate missing value percentages for all columns**

~~~
data.isnull().sum()
~~~
{: .python}

~~~
Date                 0
Location             0
MinTemp           1485
MaxTemp           1261
Rainfall          3261
Evaporation      62790
Sunshine         69835
WindGustDir      10326
WindGustSpeed    10263
WindDir9am       10566
WindDir3pm        4228
WindSpeed9am      1767
WindSpeed3pm      3062
Humidity9am       2654
Humidity3pm       4507
Pressure9am      15065
Pressure3pm      15028
Cloud9am         55888
Cloud3pm         59358
Temp9am           1767
Temp3pm           3609
RainToday         3261
RainTomorrow      3267
dtype: int64
~~~
{: .output}
~~~
# Calculate missing value percentages for all columns
missing_percentages = (data.isnull().sum() / len(data)) * 100

# Display missing value percentages for all columns
print("Missing value percentages for all columns:")
print(missing_percentages)
~~~
{: .python}

~~~
Missing value percentages for all columns:
Date              0.000000
Location          0.000000
MinTemp           1.020899
MaxTemp           0.866905
Rainfall          2.241853
Evaporation      43.166506
Sunshine         48.009762
WindGustDir       7.098859
WindGustSpeed     7.055548
WindDir9am        7.263853
WindDir3pm        2.906641
WindSpeed9am      1.214767
WindSpeed3pm      2.105046
Humidity9am       1.824557
Humidity3pm       3.098446
Pressure9am      10.356799
Pressure3pm      10.331363
Cloud9am         38.421559
Cloud3pm         40.807095
Temp9am           1.214767
Temp3pm           2.481094
RainToday         2.241853
RainTomorrow      2.245978
dtype: float64
~~~
{: .output}

Columns with a high percentage of missing values are likely to be less important unless they have a strong correlation with the target.

- Evaporation (missing 42.82%): Likely less important.
- Sunshine (missing 48.02%): Likely less important.
- Cloud9am (missing 38.42%): Possibly less important.
- Cloud3pm (missing 40.79%): Possibly less important.
~~~
#Visualize missing values
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
~~~
{: .python}
~~~
![](../fig/missing.png)
~~~
{: .output}
**Get the data types of all columns**
~~~
# Get the data types of all columns
column_data_types = data.dtypes

# Separate columns into numerical and categorical
numerical_columns = column_data_types[column_data_types != 'object'].index.tolist()
categorical_columns = column_data_types[column_data_types == 'object'].index.tolist()

# Print the lists of numerical and categorical columns
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)
~~~
{: .python}
~~~
Numerical columns: ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
Categorical columns: ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
~~~
{: .output}
#### 5. Replace missing values
~~~
from sklearn.impute import SimpleImputer
# Assuming df is your DataFrame

# Numerical columns
numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

# Categorical columns
categorical_cols = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

# Impute numerical columns with mean
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Impute categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

print(data.head())
~~~
{: .python}

~~~
         Date Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
0  01-12-2008   Albury     13.4     22.9       0.6     5.468232  7.611178   
1  02-12-2008   Albury      7.4     25.1       0.0     5.468232  7.611178   
2  03-12-2008   Albury     12.9     25.7       0.0     5.468232  7.611178   
3  04-12-2008   Albury      9.2     28.0       0.0     5.468232  7.611178   
4  05-12-2008   Albury     17.5     32.3       1.0     5.468232  7.611178   

  WindGustDir  WindGustSpeed WindDir9am  ... Humidity9am  Humidity3pm  \
0           W           44.0          W  ...        71.0         22.0   
1         WNW           44.0        NNW  ...        44.0         25.0   
2         WSW           46.0          W  ...        38.0         30.0   
3          NE           24.0         SE  ...        45.0         16.0   
4           W           41.0        ENE  ...        82.0         33.0   

   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \
0       1007.7       1007.1  8.000000   4.50993     16.9     21.8         No   
1       1010.6       1007.8  4.447461   4.50993     17.2     24.3         No   
2       1007.6       1008.7  4.447461   2.00000     21.0     23.2         No   
3       1017.6       1012.8  4.447461   4.50993     18.1     26.5         No   
4       1010.8       1006.0  7.000000   8.00000     17.8     29.7         No   

   RainTomorrow  
0            No  
1            No  
2            No  
3            No  
4            No  

[5 rows x 23 columns]
~~~
{: .output}
~~~
data.isnull().sum()
~~~
{: .python}
~~~
Date             0
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
RainTomorrow     0
dtype: int64
~~~
{: .output}
###### Data Encoding
Data encoding is a process of transforming categorical data into a numerical format suitable for analysis by machine learning algorithms. Categorical data consists of discrete labels, such as colors, types, or categories, which are not inherently numerical. Two common encoding techniques are Label Encoding and One-Hot Encoding.

**Label Encoding**:

* Method: In label encoding, each unique category is assigned a unique integer label.
* Example: If we have categories like 'Red,' 'Green,' and 'Blue,' label encoding might assign them labels 0, 1, and 2, respectively.
* Use Case: Label encoding is often used when there is an ordinal relationship between categories, meaning there is a meaningful order or ranking among them.

**One-Hot Encoding**:

* Method: One-hot encoding creates binary columns for each category and indicates the presence or absence of the category with a 1 or 0, respectively.
* Example: For the categories 'Red,' 'Green,' and 'Blue,' one-hot encoding would create three binary columns, each representing one color, with values like [1, 0, 0] for 'Red,' [0, 1, 0] for 'Green,' and [0, 0, 1] for 'Blue.'
* Use Case: One-hot encoding is commonly used when there is no inherent order among categories, and each category is considered equally distinct.

**Considerations**:

* Label encoding might introduce unintended ordinal relationships in the data, which can be problematic for some algorithms.
* One-hot encoding avoids this issue by representing categories independently, but it can lead to a large number of features, especially when dealing with a high number of categories.
* The choice between label encoding and one-hot encoding depends on the nature of the data and the requirements of the machine learning algorithm being used.

**Application**:

* Data encoding is crucial when working with machine learning models that require numerical input, such as linear regression, support vector machines, or neural networks.
* Many machine learning libraries and frameworks provide convenient functions for implementing these encoding techniques.

~~~
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode categoricalencode categorical data into numerical data  columns
for column in categorical_cols:
    data[column] = label_encoder.fit_transform(data[column])

# Print the first few rows to verify encoding
print(data.head())
~~~
{: .python}
~~~
 Date  Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
0   105         2     13.4     22.9       0.6     5.468232  7.611178   
1   218         2      7.4     25.1       0.0     5.468232  7.611178   
2   331         2     12.9     25.7       0.0     5.468232  7.611178   
3   444         2      9.2     28.0       0.0     5.468232  7.611178   
4   557         2     17.5     32.3       1.0     5.468232  7.611178   

   WindGustDir  WindGustSpeed  WindDir9am  ...  Humidity9am  Humidity3pm  \
0           13           44.0          13  ...         71.0         22.0   
1           14           44.0           6  ...         44.0         25.0   
2           15           46.0          13  ...         38.0         30.0   
3            4           24.0           9  ...         45.0         16.0   
4           13           41.0           1  ...         82.0         33.0   

   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \
0       1007.7       1007.1  8.000000   4.50993     16.9     21.8          0   
1       1010.6       1007.8  4.447461   4.50993     17.2     24.3          0   
2       1007.6       1008.7  4.447461   2.00000     21.0     23.2          0   
3       1017.6       1012.8  4.447461   4.50993     18.1     26.5          0   
4       1010.8       1006.0  7.000000   8.00000     17.8     29.7          0   

   RainTomorrow  
0             0  
1             0  
2             0  
3             0  
4             0  

[5 rows x 23 columns]
~~~
{: .output}

**Correlation with Target**
* Calculate the correlation of numerical features with RainTomorrow. For categorical features, we can use other methods like Chi-square test for independence or converting them to numerical and checking correlation.
~~~
# Calculate correlation of each numerical feature with the target variable
correlation_with_target = data.corr()['RainTomorrow'].sort_values(ascending=False)

# Print correlation values
print(correlation_with_target)
~~~
{: .python}
~~~
RainTomorrow     1.000000
Humidity3pm      0.433179
RainToday        0.305744
Cloud3pm         0.298050
Humidity9am      0.251470
Cloud9am         0.249978
Rainfall         0.233900
WindGustSpeed    0.220442
WindSpeed9am     0.086661
WindSpeed3pm     0.084207
MinTemp          0.082173
WindGustDir      0.048774
WindDir9am       0.035341
WindDir3pm       0.028890
Date             0.005732
Location        -0.005498
Temp9am         -0.025555
Evaporation     -0.088288
MaxTemp         -0.156851
Temp3pm         -0.187806
Pressure3pm     -0.211977
Pressure9am     -0.230975
Sunshine        -0.321533
Name: RainTomorrow, dtype: float64
~~~
{: .output}
~~~
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Target Variable (RainTomorrow)')
plt.show()
~~~
{: .python}
~~~
![](../fig/heatmap_tomo.png)
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}
~~~
~~~
{: .python}

~~~
~~~
{: .output}

## Ensemble Learning Techniques

## Unsupervised Learning

## Hyperparameter Tuning and Optimization

## Machine learning modelling for spatial data
### sklearn-xarray
### Pyspatialml





