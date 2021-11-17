


# Using NLP Techniques to Predict Song Skips on Spotify based on Acoustic Data




Puspha Raj Pandeya<br>
puspharajpandey@gmail.com<br>
Date: 2021-11-15<br>


# Abstract
Recommender systems play an essential role in music streaming services, prominently in the form of personalized playlists. Exploring the user interactions within these listening sessions can be beneficial to understanding the user preferences in the context of a single session. Spotify constantly seeks to improve its user experience. Using Machine Learning we can build a model that could predict if the user will skip the song or not. By predicting whether users will skip certain songs, Spotify can better understand its user’s needs and based on these insights improve its music recommendation system. The models used for this project are logistic regression and gradient boosted trees. The results were obtained by training these models on a large dataset that consists of about 10000 listening sessions. The accuracy of our models was around 87%. Our code is released at https://github.com/Puspha22/spotify-skip-prediction.

<br>Keywords: Machine Learning, Logistic Regression, Gradient Boosted Trees, Skip Prediction



# Introduction
In many of the music streaming services such as Spotify, personalized music recommendation systems play a prominent role. These recommendation systems allow the user to listen to suggested music based on a particular song, the user’s mood, time or location. The specific goal of the project was to predict whether a listener will skip a certain song. The vast amount of available music and diverse interests exhibited by various users, as well as by the same user during different situations pose considerable challenges to such systems.

Due to the large size of the dataset and lack of good hardware we choose to use a sample dataset containing about 10000 sessions only. The length of each session varies from 10 to 20 tracks. This means the model has to predict skipping behaviour for five tracks for the shortest sessions, and ten tracks for the longest. Metadata such as duration, release year, and US popularity estimate is provided for every track. Also, audio features such as acousticness, tempo, loudness are provided. For each track that the user was presented within the session, interactions such as seek forward/backwards, short/long pause before play are available. Finally, session information such as the time of the day, date, the premium user or not, context type of playlist is present. In the dataset, skipping behaviour is classified into four types: 
(1) skip_1: Boolean indicating if the track was only played very briefly 
(2) skip_2: Boolean indicating if the track was only played briefly 
(3) skip_3: Boolean indicating if most of the track was played 
(4) not_skipped: Boolean indicating that the track was played in its entirety 

The objective of the challenge is limited to predicting just the skip_2 behaviour.

# Methodology
For this project, we followed a simple methodology used in most of the machine learning projects. All the steps that we followed are discussed below:

## Data Gathering
The dataset provided by Spotify consists of two parts, session logs and tracks. The dataset provided by Spotify has around 130 million listening sessions. Due to the lack of good hardware, we choose to use a sample of the dataset. The session logs contain 10000 listening sessions. Each of these listening sessions is described by 21 features, such as:<br>
unique session ID<br>
the sequence of tracks played during the session<br>
the date and hour the session was played, etc.<br>
<br>The tracks dataset contains 50704 unique tracks users listened to during their listening sessions. Each of these tracks is described by 29 features, such as:<br><br>
unique track ID<br>
track duration<br>
track popularity rating<br>
track beat strength, etc.<br>


## Data Preprocessing
Data preprocessing is a very crucial step in any machine learning project. Data preprocessing generally refers to the process of handling missing values, Parsing Dates, Handling inconsistent data and so on. In our case, there were no missing values so handling missing values were not needed. The dataset was given in two separate files. So we first merged the datasets based on track_id. Since all the dates were of the same year and months were different, I converted the year feature into days. 

## Data Visualization
Data visualization is generally done to get insight into the data. We plotted histograms and bar plots of the features to see how they depend on the output feature. We also used an inbuilt python library called sweetviz that does all the visualization automatically in a beautiful format.


## Exploratory Data Analysis (EDA)
EDA is unavoidable and one of the major steps to fine-tune the given data sets in a different form of analysis to understand the insights of the key characteristics of various entities of the data set like columns, rows by applying Pandas, NumPy, Statistical Methods, and Data visualization packages. In EDA we checked for correlation and dropped highly positively and highly negatively correlated features. Then we encoded the categorical features using one-hot encoding. The reason we need encoding is that a machine learning model can only work with numeric data. After that, we checked for outliers in the dataset and removed them. After removing the outliers, we scaled the dataset using a min-max scaler, which scales the dataset into the range 0 to 1. 

## Train-Test split
Train-Test split is done to split the dataset into training and testing so that after training the model we can evaluate the accuracy on the new values that the model has not seen before. We used 80% of the data for training and the remaining 20% for testing. 

## Model Selection
As per the project guidelines, the task was to create models using Logistic Regression and Light Gradient Boosted Trees. I started with Logistic Regression as it was simple to make and works great for classification problems. The accuracy I got with Logistic Regression was about 87% which was pretty good as we were only working on a subset of the actual dataset. After that, I moved to Light Gradient Boosted Trees which is a tree-based model that uses boosting Technique. After tuning the hyperparameters for the model I was able to get an accuracy of 87.65%. The accuracy I got from both the models were pretty much the same (LBGT winning with a small margin). But with LGBT there is always room for improvement as it is a very robust algorithm with lots of parameters that can be tuned.

I also used the confusion matrix to see how many features were misclassified. And finally saved the models on a file using the pickle library.

## Model Deployment
For deployment, I used Flask which is a python framework for web development. I used HTML and CSS to create a simple frontend for my web app. I made a class that would extract the track_ID and track features from the Spotify API by providing the track name, and pass it to the model. I used Heroku which is a cloud platform to host my web app. 


# Conclusion and Future Work
Making a sequential skip prediction model was a challenging task. At the same time, these kinds of projects are very beneficial for the companies that provide music services to make recommendation systems. The accuracy I got is approx. 87%, which I think is pretty good considering the time constraint and hardware constraint. Also if we could use the whole dataset provided by Spotify instead of taking a sample of it, I am very sure the accuracy would have been a lot better.

In future, we can use models that use Neural Networks for this task. We can use LSTM, Bi-LSTM models as they are very good with long term memory. Also, we can use Encoder-Decoder architecture to preserve the sequence of the data. Also, transformers could be used which is a very complicated model but is very good for these kinds of sequential predictions.
