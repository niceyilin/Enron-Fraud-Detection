# Enron-Fraud-Detection
Machine Learning project, fraud detection using Enron financial and email dataset, with Python sklearn package

## 0. Background
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.
This project attempts to predict the likelihood of someone being a suspect of Enron fraud conspiracy by looking at given dataset. We call the suspects Person of Interest (POI). The dataset contains insider pays to all Enron executives as well as emails sent through their company accounts, and their POI status.
We use machine learning to learn insider pays and emailing habits of POIs and non-POIs and see if we can find a pattern there, then use the model created to predict the likeliness of someone with a particular pattern of being a POI or not.

## 1. Data processing flow
![data flow](https://drive.google.com/drive/folders/0B8RA3Cz4irGiWnFwMFFWWkFON2s)

## 2. Dataset description
The financial dataset contains a total of 146 data points with 21 features. 18 are labeled as persons of interest.
 * Available features are as below, I identified three main type of features in the dataset:
 1.  Insider pays:
  - bonus
  - salary
  - exercised_stock_options
  - deferral_payments
  - deferred_income
  - total_payments
  - total_stock_value
 2.  Emailing habits:
  - email_addr
  - Number of emails sent: from_messages
  - Number of emails received: to_messages
  - Number of emails sent to POIs: from_this_person_to_poi
  - Number of emails received from POIs: from_poi_to_this_person
  - Receipts shared with poi: shared_receipt_with_poi
 3. target:
  - POI status as already known: poi

After visualizing the dataset as a scatter plot, I identified an outlier named TOTAL. This is a spreadsheet artifact and it was thus removed.

Emails sent and received by Enron's employees are also available, each person has his folder which contain the path to the email body that belong to him. There's a another list which contains POI's email address. 

## 2. Pre-processing
* sklearn.preprocessing.MinMaxScaler
* sklearn.decomposition.PCA : random solver, tuned by parameter n_pca_components
* sklearn.feature_selection.SelectPercentile : for email feature selection, tuned by parameter email_feature_percentage
* nltk.stem.snowball : email raw text processing
* sklearn.feature_extraction.text.TfidfVectorizer: tokenizer + Tfidf transform, also remove stop words
* filter out percentage of email content by random sampling, to fit data in memory, and speed up processing
 - tuned by parameter person_filter_percentage, nonpoi_filter_percentage, poi_filter_percentage

## 3. Algorithm selection and Tunning
* sklearn.naive_bayes.GaussianNB
* sklearn.svm.SVC
* sklearn.tree.DecisionTreeClassifier
* Tuning by sklearn.model_selection.GridSearchCV
 - kernel, C, gamma, min_samples_split, etc

## 4. Cross Validation and Evaluation 
- sklearn.model_selection.train_test_split : simple training and test data split, 70% training and 30% testing + random shuffling
- sklearn.model_selection.KFold : better for this dataset because of smaller data sample number, KF gives more stable/averaged/reasonable score

## 5. Result








