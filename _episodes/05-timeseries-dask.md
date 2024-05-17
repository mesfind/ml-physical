---
title: Time Series Analysis with Python
teaching: 1
exercises: 0
questions:
- "What are the basic timeseries I can use in pandas ?"
- "How do I write documentation for my Python code?"
- "How do I install and manage packages?"
objectives:
- "Brief overview of basic datatypes like lists, tuples, & dictionaries."
- "Recommendations for proper code documentation."
- "Installing, updating, and importing packages."
- "Verify that everyone's Python environment is ready."
keypoints:
- "Are you flying yet?"
---



### Python's Basic in Time Series Data

### Some Useful Pandas Tools

#### Changing Index to Datetime

To convert the index of a pandas DataFrame to datetime format, you can use the `to_datetime` function. Here's how you can do it:

```python
df.index = df.to_datetime(df.index)
```

#### Plotting the Time Series Data

Once the index is in datetime format, you can plot the time series data using the `plot` function:

~~~
df.plot()
~~~
{: .python}

#### Slicing the Data

After converting the index to datetime format, you can slice the data based on specific dates. For example, to get the data for the year 2012:

~~~
df['2012']
~~~
{: .python}

#### Join two dataframes

~~~
df1.join(df2)
~~~
{: .python}


- computing the percentage change and differenes in time series

~~~
df['col'].pct_changes()
df['col'].diff()
~~~
{: .python} 

- pandas correlation method of Series

~~~
df['ABC'].corr(df['XYZ'])
~~~

- Pandas autocorrelation

~~~
df['ABC'].autocorr()
~~~


These tools are essential for working with time series data in Python using the pandas library.

## Correlation of Two Time Series

To understand the correlation between two time series variables, especially in the context of time-dependent data, several key points need to be considered based on the provided sources:

1. **Pearson Correlation and Time Series**:
   - Pearson correlation is commonly used to measure the linear relationship between two variables. However, when dealing with time series data, the assumption of independence between data points may not hold true due to the temporal nature of the data[2].
   - Time series data often exhibits within-series dependence, where observations are correlated over time. This can lead to misleading results when using Pearson correlation, as it assumes independence between data points[2].

2. **Cross-Correlation Function**:
   - The cross-correlation function (CCF) is used to explore how one time series may predict or explain another. It helps in understanding the relationship between two time series variables and can reveal patterns like leading or lagging effects between them[4].

3. **Computing Correlations in Time Series Data**:
   - When analyzing multiple variables in a time series dataset, computing correlations between variables can provide valuable insights. Visualizing correlations as a heat map can help identify patterns, such as positive or negative correlations between variables over time[5].

4. **Practical Application**:

   - In practice, it is essential to consider the temporal aspect of data when calculating correlations between time series variables. Understanding the nuances of time-dependent relationships and the impact of within-series dependence is crucial for accurate analysis and interpretation.

When assessing the correlation between two time series variables, it is important to account for the temporal dependencies inherent in time series data. Utilizing appropriate methods like cross-correlation functions and considering within-series dependence can lead to more meaningful insights into the relationship between the variables over time.
