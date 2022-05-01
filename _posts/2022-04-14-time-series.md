---
title: 'How to Analyze and Forecast Time Series Data'
date: 2022-04-14
permalink: /posts/2022/04/time-series/
tags:
  - tutorial
---

With this tutorial, you will learn how to analyze and forecast time series data using various statistical and machine learning models.

## What is a Time Series?

A **time series** is a sequence of data observed over time. For example, at specific points in time, we can observe features such as the unemployment rate of a country, sales success of a product, or demand of a certain quantity. The key is that each of these variables _varies with respect to time_. 

Analysis of temporal data allows us to obtain useful insights on how a variable changes over time (as in a **univariate time series**) or how it depends on the change in the values of other variables (as in a **multivariate time series**). This relationship can then be further analyzed for time series **forecasting** (extrapolation into the future), which has numerous applications for artificial intelligence and machine learning.

## Python Libraries Used for Time Series Analysis

There are several open-source Python libraries we will use for time series analysis, some of which you may have seen before. These include:
* `numpy`: provides fast and basic mathematical functionality on array objects
* `pandas`: provides highly efficient data structures such as Series and DataFrame objects
* `scipy`: provides functionalities for optimization, signal and image processing, integration, interpolation and linear algebra
* `scikit-learn`: a scipy toolkit widely used for statistical modeling, machine learning, and deep learning
* `statsmodels`: provides tools for statistical data exploration and modeling
* `matplotlib`: provides tools for data visualization in various formats (scatterplots, histograms, etc.)
* `datetime`: provides all the necessary functionality for reading, formatting, and manipulating time

Here's a flowchart of some of these packages for reference:
![Flowchart of python packages for time series analysis](/education-blogs/images/2022-04-14-time-series/libflow.png)
