---
title: 'Choosing a Good Dataset'
excerpt: This tutorial will contain content relating to creating your own data science project! Often, topics pertaining to beginning projects, such as formulating a question and finding the right data sets can be difficult. We hope to provide insight to get you started.
date: 2022-04-14
permalink: /posts/2022/04/choosing-dataset/
tags:
  - tutorial
---

## How to Get Started on Data Science Projects
This tutorial will contain content relating to creating  your own data science project! Often, topics pertaining to beginning projects, such as formulating a question and finding the right data sets can be difficult. We hope to provide insight to get you started. 

## Contents: 

1. How to develop a question

2. Where can you find datasets

4. How to pick a good dataset and create questions from it

5. I have a topic in mind.. How do I formulate a question?? & I have a question in mind… how do I find the data??

## 1. How to develop a question: 

Good Examples of Questions: 
  - What work and lifestyle conditions greatly impact mental health, and in what way?
  - Based on this data, what factors can be used to predict a candidate’s success within a Canadian election?
  - What features best predict the amount of solar radiation the Earth gets based on data collected by NASA?

Why these are good examples: 
  - In Data Science problems, you want features and a target variable 
  - Features are the inputs to your model. For example, in the mental health question, the features would be different lifestyle conditions  such as fitness, work hours, and healthy eating. 
  - A target variable is the variable you would like to predict. Using the same example, the target variable would be mental health outcomes. 
  - All these questions contain objective features and target variables. 

Bad Examples of Questions: 
  - What can the data tell me about mental health?
  - Is there a relationship between the data and a candidate’s success in a Canadian election?
  - Can we predict the amount of solar radiation the earth gets?

Why these are bad examples:
- These examples are broad and appear broad and hard to decipher features and target variables. 
- For the mental health example, what data are you looking at? For the radiation example, what are you basing this prediction on?

## 2. Where to find Datasets
Resources include: Kaggle, Government datasets, and web scraping

Kaggle:  <https://www.kaggle.com/datasets>
  - Search for all different types of datasets that can be downloaded to your computer
  - Topics include country population datasets, national GDP datasets, car prices, pistachio images, and much more!

Government data sets: 
  - Use search engines to find government datasets related to topics such as health, sustainability, and economics. 
  - Useful site: <https://data.gov>
  
Web Scraping: 
  - Extract data from websites
  - Exports data into a more user-friendly arrangement 
  - We will not go into detail and how to webscrape, but here are some resources:
    <https://realpython.com/beautiful-soup-web-scraper-python/>
    <https://oxylabs.io/blog/python-web-scraping>
  
## 3. How to Pick a Good Dataset and Derive Questions from it
Once you’ve found an interesting topic, here are some things to consider about good datasets:
  - Is the data formatted well? 
  - Are there many missing values? 
  - Will a lot of data manipulation/imputation be necessary?
  - Is the dataset large enough to make valid conclusions?

Here is an example of a poor dataset to use in a project, from Kaggle: 

![Good Dataset](/education-blogs/images/2022-04-14-choosing-dataset/MessyDataset.png)
[Source](https://medium.com/well-red/cleaning-a-messy-dataset-using-python-7d7ab0bf199b)


This dataset is missing some values, leading to a need for data imputation, and does not contain many entries. 


How to derive a good question from a dataset:
  - What is a variable that you would like to predict within the dataset?
  - What other features in the dataset may relate to this target?
  - Be sure to explicitly define these to allow for an easier project

## 4. I have a topic in mind.. How do I formulate a question?? & I have a question in mind… how do I find the data??
Example: I want to determine different factors that may cause obesity. 

  - First, discover different factors that may cause obesity. Using some common knowledge and discovery, one could determine that things like genetics, physical activity, diet, and others may have an effect on obesity. 
  - After finding a dataset that includes some of these factors and obesity status, one can formulate a good question. 
  - One good question could be, what impact do physical activity, a balanced diet, and one’s environment have on obesity risk? 
  - This is a good question because it’s subjective and clearly lists features and the target variable 

How would I find the data for this?
  - Since this is a global medical issue, I could likely use a government database, like <https://data.gov/>, and search “obesity database” in the search bar!
	
  - If I can’t find a dataset I like, I could also try this search on Kaggle!












