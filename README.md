# Ecommerce_Recommendation_System

## Introduction
Welcome to the Spotifly Recommendation System project! This project aims to build a recommendation system for the Spotifly app using user playback data and social network connections among users. The system predicts the number of times a particular user will listen to a specific artist.

## Data
### Data Files
`user_artist.csv`: This file contains playback data, including user IDs, artist IDs, and the number of times a user listened to a specific artist.

`friends_user.csv`: Describes the social network connections between users. Each row represents a pair of neighboring users.

`test.csv`: Contains pairs of users and artists for which predictions need to be made.
### Data Transformation
To prepare the data for modeling, a logarithmic transformation was applied to the playback counts in the user_artist.csv file.

## Code
The project code is written in Python and utilizes several libraries, including:
`pandas` for data manipulation.
`numpy` for numerical operations.
`surprise` (Simple Python Recommendation System Engine) for building recommendation models.
`GridSearchCV` for hyperparameter tuning.
### Code Organization
The project code is organized into different sections, including data transformation, model selection, and prediction.
Detailed comments within the code provide explanations for each step.
### Model Selection
The project employs various recommendation models, including K-nearest neighbors (KNN) and matrix factorization-based models like SVD and SVDpp.
Hyperparameters for each model were selected using cross-validation (GridSearchCV) to minimize the root mean squared error (RMSE) on the training data.
The best-performing model is selected based on the lowest RMSE value.

## Results and Evaluation
The results of the model selection process indicate that the SVDpp model achieved the lowest RMSE, making it the chosen recommendation model for this project.
The project minimizes the risk of overfitting through cross-validation, which trains the model with different parameter combinations on different subsets of the training data.

## Instructions for Running the Code
Ensure you have the required dependencies installed, including pandas, numpy, and surprise.
Place the data files (user_artist.csv and test.csv) in the appropriate directory.
Run the Python code provided (task1.py) to perform data transformation, model selection, and prediction.
The final predictions will be saved in the task1.csv file in the working directory.

_Note: The model selection process can be time-consuming (approximately two hours). You may choose to comment out this part of the code if not necessary._

## Conclusion
The Spotifly Recommendation System project demonstrates the process of building a recommendation system using various models and data transformation techniques. The selected SVDpp model offers promising results for predicting user-artist interactions. Future enhancements and applications of this system can explore additional features and optimizations.
