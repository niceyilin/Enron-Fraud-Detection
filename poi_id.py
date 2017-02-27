#!/usr/bin/python

import os
import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 0. Data read in
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

del data_dict["TOTAL"] # remove outliers (invalid samples)
print "Total count of samples: ", len(data_dict)

valid_email_set = set()
for name, features in data_dict.items():
	valid_email_set.add(features["email_address"])
#print "Names in dataset : \n", sorted(valid_email_set)

#############################################
######## 0. Extract Email Feature ###########
#############################################
import random
from poi_email_addresses import poiEmails
from parse_out_email_text import parseOutText
print "\n\nStart to extract email features ...\n"

#### parameters to be tuned #########
debug_mode = True
debug_counter = 200
person_filter_percentage = 0
nonpoi_filter_percentage = 60
poi_filter_percentage = 40
email_feature_percentage = 10
n_pca_components = 40
n_kfolds = 10
######################################

addr_list = []
label_list = []
email_list = []
poiEmailList = poiEmails()
path = 'emails_by_address'
t0 = time()

# generate label_list and email_list for training
for file_name in os.listdir(path):
	# for debug run only
	if debug_mode :
		print "debug_counter = ", debug_counter
		debug_counter -= 1
		if(debug_counter < 0):
			break

	# go to each person's folder, get his label, and the path to his emails
	try:
		ToOrFrom, email_addr = file_name.split('_', 1)
	except ValueError:
		print "Error : ", file_name, " can't been split !"
				
	email_addr = email_addr.replace(".txt", "")	
	if(email_addr not in valid_email_set)	:
		continue
		
	poiLabel = 1 if email_addr in poiEmailList else 0
	
	randomRoll = random.random() * 100 # 0~100	
	if poiLabel == 0 and randomRoll < person_filter_percentage : # skip 90% of non-poi person for speedup
		continue
	
	if debug_mode :
		print "\t", email_addr, "*************** is POI ?", poiLabel
	
	file_name = os.path.join(path, file_name)
	
	try:	
		fp = open(file_name, 'r')
	except IOError:
    		print("Error: can't open file !")

	# go through path-list, get path to each email, parse email body and store them
	email_string = ""
	email_set = set()
	for raw_email_path in fp:
		if poiLabel == 0 and random.random() * 100 < nonpoi_filter_percentage : # skip 90% of non-poi person's email for speedup
			continue
		if poiLabel == 1 and random.random() * 100 < poi_filter_percentage : # skip 90% of non-poi person's email for speedup
			continue
		
		date, email_path = raw_email_path.split('/', 1)
		email_path = os.path.join(".", email_path[:-1])
		emailfp = open(email_path, 'r')
		email_body = parseOutText(emailfp) # remove header + stemmer
		
		if email_body not in email_set:
			email_string += email_body
			email_set.add(email_body)
		
		emailfp.close()
		
	if email_addr in addr_list:
		index = addr_list.index(email_addr)
		email_list[index] += email_string
	else:
		email_list.append(email_string)
		label_list.append(poiLabel)
		addr_list.append(email_addr)		
	
	fp.close()

print "Email_list extraction is done in ", time() - t0, " seconds"
print "Count of email_list = ", len(email_list)

# tf idf processing 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")
email_list = tfidf.fit_transform(email_list).toarray()

# Feature Selection
from sklearn.feature_selection import SelectPercentile
totalfeatures = len(email_list[0])
selector = SelectPercentile(percentile=email_feature_percentage)
email_list = selector.fit_transform(email_list, label_list)
print "Total count of samples: ", len(email_list)
print "Selected ", email_feature_percentage, "% text features out of total ",  totalfeatures

# append each person's email context to data_dict
email_content_counter = 0
for person, features in data_dict.items():
	email_addr = features['email_address']
	found = False
	for i in range(len(addr_list)):
		if email_addr == addr_list[i]:
			data_dict[person]['email_content'] = email_list[i]
			email_content_counter += 1
			found = True
			break
	if found == False:
		data_dict[person]['email_content'] = [0.0] * len(email_list[0])
			
			
print email_content_counter, " people have been added with email content data !"


#############################################
######## 1. Extract Final Feature ###########
#############################################
target = "poi"
f = {}
f[0] = "to_messages"
f[1] = "salary"
f[2] = "total_stock_value"
f[3] = "deferral_payments"
f[4] = "total_payments"
f[5] = 'exercised_stock_options'
f[6] = 'bonus'
f[7] = 'restricted_stock'
f[8] = 'shared_receipt_with_poi'
f[9] = 'restricted_stock_deferred'
f[10] = 'total_stock_value'
f[11] = 'expenses'
f[12] = 'loan_advances'
f[13] = 'from_messages'
f[14] = 'other'
f[15] = 'from_this_person_to_poi'
f[16] = 'director_fees'
f[17] = 'deferred_income'
f[18] = 'long_term_incentive'
f[19] = 'from_poi_to_this_person'
features_list = [target, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[18], f[19]]


data = featureFormat(data_dict, features_list, sort_keys = True, include_name = True)
names, labels, features = targetFeatureSplit(data, include_name = True)

# Feature Selection
#from sklearn.feature_selection import SelectKBest
#selector = SelectKBest(k=5)
#features = selector.fit_transform(features, labels)
#print "Total count of samples: ", len(data_dict)
#print "Total available features: ", data_dict["METTS MARK"].keys()
#print "Feature Selector scores : ", selector.scores_


# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Simper training/testing data split
features_with_emails = []
for i in range(len(names)):
	features_with_emails.append(np.concatenate((features[i], data_dict[names[i]]['email_content'])))

#X_train, X_test, y_train, y_test = train_test_split(features_with_emails, labels, test_size=0.3, random_state=42)

#K-Fold training/testing data split
fold_counter = 1
SV_recall = SV_precision = DT_recall = DT_precision = NB_recall = NB_precision = 0.0
	
kf = KFold(n_splits=n_kfolds, shuffle=True, random_state=42)
for train_index, test_index in kf.split(features_with_emails):
	print "\nCurrent running dataset fold ", fold_counter, "..."
	fold_counter += 1
	X_train = [features_with_emails[i] for i in train_index] 
	y_train = [labels[i] for i in train_index] 
	X_test  = [features_with_emails[i] for i in test_index] 
	y_test  = [labels[i] for i in test_index]

	# PCA
	from sklearn.decomposition import PCA
	print "Extracting the top %d principal component from %d features" % (n_pca_components, len(X_train[0]))
	pca = PCA(n_components=n_pca_components, svd_solver='auto', whiten=True).fit(X_train)
	print "1st PC explained ", pca.explained_variance_ratio_[0], " of variance of data"
	print "2nd PC explained ", pca.explained_variance_ratio_[1], " of variance of data"

	print "Projecting the input data on the principal component basis"
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)
	print "X_train shape after PCA : ", len(X_train), ", ", len(X_train[0])


	"""
	#############################################
	######## 2.1 Visualization ##################
	#############################################
	import matplotlib.pyplot as plt
	if debug_mode:
		x_axis = 1
		y_axis = 3
		print_label = True

		for feature, label in zip(features, labels):
			col = 'g' if label == 0 else 'r'
			lbl = 'non-POI' if label == 0 else "POI"
			if print_label:
				print_label = False
				plt.scatter(feature[x_axis], feature[y_axis], color=col, label=lbl)
			plt.scatter(feature[x_axis], feature[y_axis], color=col)

		plt.xlabel(f[x_axis])
		plt.ylabel(f[y_axis])
		plt.legend()
		plt.show()
	"""
	#############################################
	######## 3. Fit and Tune Classifier #########
	#############################################
	from sklearn.model_selection import GridSearchCV
	from sklearn.naive_bayes import GaussianNB
	from sklearn.svm import SVC
	from sklearn.tree import DecisionTreeClassifier

	print "\nStart training Naive Bayes..."
	NBclf = GaussianNB()
	NBclf.fit(X_train, y_train)
	print NBclf.get_params()

	print "\nStart training SVC..."
	#SVMparam = {'kernel':['linear', 'rbf', 'poly'], 'C':[1e2, 1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
	SVMparam = {'kernel':['linear', 'rbf'], 'C':[1e7, 1e6, 1e5, 1e4], 'gamma': [0.00001, 0.000001, 0.0000001]}
	#SVclf = SVC(kernel='rbf', C=1e7, gamma=0.00001)
	SVclf = GridSearchCV(SVC(), SVMparam)
	t0 = time()
	SVclf.fit(X_train, y_train)
	print "SVC completes in ", time() - t0, "seconds"
	print SVclf.best_estimator_

	print "\nStart training Decision Tree..."
	DTparam = {'min_samples_split' : [40, 60, 80, 120, 150]}
	#DTclf = DecisionTreeClassifier()
	DTclf = GridSearchCV(DecisionTreeClassifier(), DTparam)
	t0 = time()
	DTclf.fit(X_train, y_train)
	print "Decision Tree completes in ", time() - t0, "seconds"
	print DTclf.best_estimator_

	#############################################
	######## 4. Validation ######################
	#############################################
	from sklearn.metrics import recall_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix

	NB_pred = NBclf.predict(X_test)
	print "\nGaussianNB accuracy: ", NBclf.score(X_test, y_test)
	NB_recall += recall_score(y_test, NB_pred, pos_label='1.0')
	NB_precision += precision_score(y_test, NB_pred, pos_label='1.0')
	print classification_report(y_test, NB_pred, target_names=["non-POI", "POI"])
	print confusion_matrix(y_test, NB_pred)

	SV_pred = SVclf.predict(X_test)
	print "\nSVC accuracy: ", SVclf.score(X_test, y_test)
	SV_recall += recall_score(y_test, SV_pred, pos_label='1.0')
	SV_precision += precision_score(y_test, SV_pred, pos_label='1.0')
	print classification_report(y_test, SV_pred, target_names=["non-POI", "POI"])
	print confusion_matrix(y_test, SV_pred)

	DT_pred = DTclf.predict(X_test)
	print "\nDecision Tree accuracy: ", DTclf.score(X_test, y_test)
	DT_recall += recall_score(y_test, DT_pred, pos_label='1.0')
	DT_precision += precision_score(y_test, DT_pred, pos_label='1.0')
	print classification_report(y_test, DT_pred, target_names=["non-POI", "POI"])
	print confusion_matrix(y_test, DT_pred)

	
	
print "\n****************** K Fold Validation Results ***************************\n"	
print "GaussianNB Classifier average Recall : ", NB_recall / (n_kfolds * 1.0), " Precision : ", NB_precision / (n_kfolds * 1.0)
print "SVM Classifier average Recall : ", SV_recall / (n_kfolds * 1.0), " Precision : ", SV_precision / (n_kfolds * 1.0)
print "Decision Tree Classifier average Recall : ", DT_recall / (n_kfolds * 1.0), " Precision : ", DT_precision / (n_kfolds * 1.0)
			

