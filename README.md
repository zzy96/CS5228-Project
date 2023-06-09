# CS5228-Project

The goal of this project is to give you experience in a real-life regression task, which is one of the most practical and commonly encountered type of data-related task. It will give you experience applying approaches such as exploratory data analysis, visualization, preprocessing, and how to select and evaluate various data mining approaches.

The task is to predict the resale price of HDB flats. A full description of the dataset and evaluation metrics can be found at the Kaggle InClass competition page. The dataset comes from data.gov and 99.co, with around 20 attributes (such as asking price, property type, address, etc.)

In this project, we look into the Singapore housing market. Home ownership in Singapore is rather expensive, making buying a home one of the most significant financial decision for most people in their lives. Buyers therefore want to know what they can get for their money, where they can best save money, and simply spot bargains rip-offs. On the other hand, sellers and real estate agents aiming to maximize prices want to know how to best present and advertise their properties. Also, with the limited amount of available land, affordable housing is a major issue of the Singaporean government, which may deploy ”cooling measures” to influence the housing market. In short, there are many stakeholders that rely on and benefit from a deeper understanding of the Singaporean housing market.

## Submitting your Predictions on Kaggle

The Kaggle page (https://www.kaggle.com/t/f88bbbd685514d4aa0a2cf161a699e6e) contains training data train.csv, with the associated labels in the form of the “resale_price" variable. The test data test.csv contains the test set which your model should predict. The predictions you submit should be a csv file: for reference, see the sample predictions file sample_output.csv, which has been included in the dataset package, and shows what your submission should look like. Additionally, considering the importance of location and nearby amenities, an auxiliary data file auxiliary_data.zip has also been provided, which contains some csv files with additional data like the locations of MRT stations, shopping malls, schools, etc. It is up to you if and how you want to consider integrating this data for training your model.

To prevent overfitting to the leaderboard, you are allowed to make at most 5 submissions to the leaderboard per day. Note that you are expected to write your own procedures for evaluating how well your method is performing using the given datasets: for example, this can be done by cross-validation, splitting the data into a training and a validation set, or similar approaches. Do not mainly rely on the public leaderboard to evaluate your model: this could potentially overfit to the public leaderboard, resulting in poor performance on the private leaderboard, and the submission limit would hinder your model development process; instead, you should utilize your own evaluation procedures such as cross-validation (see the “Evaluation using Cross-Validation” link in the Helpful Resources section below for more details). We will place greater consideration on the private rather than the public leaderboard scores.

## Run Code

There are three steps for running the code and generate prediction:
1. Run extract_location_features.py to generate additional columns based on auxiliary data and output train_with_location.csv and test_with_location.csv
2. Run preprocessing.py to generate the preprocessed train_preprocessed.csv and test_preprocessed.csv
3. Run project.ipynb to generate prediction.csv
