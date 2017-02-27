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
 2.  Emailing info:
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

* Email data
- Parse raw Email content, remove email header, extract email body
- nltk.stem.snowball : email raw text processing
- sklearn.feature_extraction.text.TfidfVectorizer: tokenizer + Tfidf transform, also remove stop words
- sklearn.feature_selection.SelectPercentile : for email feature selection, tuned by parameter email_feature_percentage
- filter out percentage of email content by random sampling, to fit data in memory, and speed up processing
 - tuned by parameter person_filter_percentage, nonpoi_filter_percentage, poi_filter_percentage

* Finicial data
- sklearn.preprocessing.MinMaxScaler
- Outlier removal by visual check (ploting)
- sklearn.decomposition.PCA : random solver, tuned by parameter n_pca_components
```
Extracting the top 40 principal component from 18208 features
1st PC explained  0.334949310051  of variance of data
2nd PC explained  0.146807349309  of variance of data
Projecting the input data on the principal component basis
X_train shape after PCA :  130 ,  40
```
- Assign processed email text to the persion in finacial dictionary whose email matches with the email header

## 3. Algorithm selection and Tunning
* sklearn.naive_bayes.GaussianNB
* sklearn.svm.SVC, with best tunned parameter set
```
Start training SVC...
SVC completes in  0.268102884293 seconds
SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=1e-07, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```
* sklearn.tree.DecisionTreeClassifier, with best tunned parameter set
```
Start training Decision Tree...
Decision Tree completes in  0.0639951229095 seconds
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=120, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
```
* Tuned by sklearn.model_selection.GridSearchCV
 - kernel, C, gamma, min_samples_split, etc

## 4. Cross Validation and Evaluation 
- sklearn.model_selection.train_test_split : simple training and test data split, 70% training and 30% testing + random shuffling
- sklearn.model_selection.KFold : better for this dataset because of smaller data sample number, KF gives more stable/averaged/reasonable score
- Evaluated by Accuracy, Recall, Precision, Confusion matrix 

## 5. Result
* below set of parameter gives best classification result
 - person_filter_percentage = 0
 - nonpoi_filter_percentage = 70
 - poi_filter_percentage = 40
 - email_feature_percentage = 10
 - n_pca_components = 40

* Among them, GaussianNB classifier gives the best POI Recall and Percision

  ```
GaussianNB accuracy:  0.948275862069
             precision    recall  f1-score   support

    non-POI       0.94      1.00      0.97        51
        POI       1.00      0.57      0.73         7

avg / total       0.95      0.95      0.94        58

[[51  0]
 [ 3  4]]
 ```

* fine tuned SVC gives relatively bad result
```
SVC accuracy:  0.879310344828
             precision    recall  f1-score   support

    non-POI       0.89      0.98      0.93        51
        POI       0.50      0.14      0.22         7

avg / total       0.85      0.88      0.85        58

[[50  1]
 [ 6  1]]
 ```

* Decision Tree classifier with min_samples_split = 80 also gives good result
```
Decision Tree accuracy:  0.844827586207
             precision    recall  f1-score   support

    non-POI       0.92      0.90      0.91        51
        POI       0.38      0.43      0.40         7

avg / total       0.85      0.84      0.85        58

[[46  5]
 [ 4  3]]
```
There's so many emails text involved, processing them is slow with limited CPU and memory.
Applying the trick to randomly filter out some percent of emails speedup the processing a lot, but still slow.
Also noted, 86 people in dataset has been matched with their email content, but other 60 people's email content are missing
```
Total count of samples:  145

Start to extract email features ...

Email_list extraction is done in  905.339167833  seconds
Count of email_list =  86
Total count of samples:  86
Selected  10 % text features out of total  181881
86  people have been added with email content data !
```

## Limitation

The number of samples in this dataset is very limited, 146 people in total.
Especially number of POI target is too limited, only 18 people. 

So even though the score measured above is promissing, it's not very convincing, because lack of enough data.
The training and testing data split is 70% vs 30% (randomly), different random split makes resulting score showing big variation.

Here in this project, I'm not able to collect more POI data points, but at least K fold CV can be used to make the relative training/testing seems a little more, and make resulting score more convincing.

* Below result is taken from K fold CV , where k = 10, and the scores are averaged by k
- All classifiers' performace drops, because some training/testing split happen to make classification very bad
- But GaussianNB Classifier is still the best among 3.
- GaussianNB Classifier is still next, SVC is still worst

```
****************** K Fold Validation Results ***************************

GaussianNB Classifier average Recall :  0.55  Precision :  0.345
SVM Classifier average Recall :  0.1  Precision :  0.05
Decision Tree Classifier average Recall :  0.266666666667  Precision :  0.233333333333
```






