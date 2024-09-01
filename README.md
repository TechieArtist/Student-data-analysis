# Student Data Analysis

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TechieArtist/Student-data-analysis/blob/main/LICENSE)

## About  <a name="about"></a>
This project is a comprehensive analysis of student performance data. The primary objective is to explore the relationships between various factors such as study habits, extracurricular activities, and academic performance. Additionally, several machine learning models were trained and evaluated to predict student success based on these factors.

### Libraries Overview <a name="lib_overview"></a>

All the custom libraries and scripts used in this project are located under [\<project root\>/source](https://github.com/TechieArtist/Student-data-analysis/tree/main/source)
- [\<project root\>/source/data_preprocessing.py](https://github.com/TechieArtist/Student-data-analysis/blob/main/source/data_preprocessing.py): Contains all preprocessing functions, including feature scaling, encoding, and handling missing values.
- [\<project root\>/source/model_training.py](https://github.com/TechieArtist/Student-data-analysis/blob/main/source/model_training.py): Main script used to train and evaluate machine learning models.
- [\<project root\>/source/predict_nn.py](https://github.com/TechieArtist/Student-data-analysis/blob/main/source/predict_nn.py): Script for making predictions using the trained neural network model.

### Where to Put the Code  <a name="#putcode"></a>
- Place any additional preprocessing functions/classes in [source/data_preprocessing.py](https://github.com/TechieArtist/Student-data-analysis/blob/main/source/data_preprocessing.py)
- Place new model definitions or enhancements in [source/model_definitions](https://github.com/TechieArtist/Student-data-analysis/tree/main/source/model_definitions)
- Add any new analysis or visualization scripts to [source/data_analysis.py](https://github.com/TechieArtist/Student-data-analysis/blob/main/source/data_analysis.py) or [source/data_visualization.py](https://github.com/TechieArtist/Student-data-analysis/blob/main/source/data_visualization.py)

## Table of Contents

+ [About](#about)
  + [Libraries Overview](#lib_overview)
  + [Where to Put the Code](#putcode)
+ [Prerequisites](#prerequisites)
+ [Bootstrap Project](#bootstrap)
+ [Running the code using Jupyter](#jupyter)
  + [Configuration](#configuration)
  + [Local Jupyter](#local_jupyter)
  + [Google Collab](#google_collab)
+ [Adding New Libraries](#adding_libs) 
+ [TODO](#todo)
+ [License](#license)

## Prerequisites <a name="prerequisites"></a>

You need to have Python >= 3.9 installed on your machine. It's recommended to use a virtual environment for managing dependencies.

```Shell
$ python3.9 -V
Python 3.9.7
