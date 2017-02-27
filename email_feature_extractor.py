#!/usr/bin/python

import os
import sys
import pickle
import random
sys.path.append("../tools/")
from time import time
from poi_email_addresses import poiEmails
from parse_out_email_text import parseOutText

#### parameters to be tuned #########
debug_mode = True
debug_counter = 500
person_filter_percentage = 0
nonpoi_filter_percentage = 50
poi_filter_percentage = 30
######################################

addr_list = []
label_list = []
email_list = []
poiEmailList = poiEmails()
path = 'emails_by_address'
poi_count = 0
nonpoi_count = 0
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
		email_path = os.path.join("..", email_path[:-1])
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
print "POI emails count : ", poi_count, ", non-POI emails count : ", nonpoi_count

# tf idf processing 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")
email_list = tfidf.fit_transform(email_list).toarray()

"""
# PCA
from sklearn.decomposition import RandomizedPCA
n_pca_components = 10
print "Extracting the top %d principal component from %d features" % (n_pca_components, email_list_tfidf.shape[1])
pca = RandomizedPCA(n_components=n_pca_components, whiten=True).fit(email_list_tfidf)
print "1st PC explained ", pca.explained_variance_ratio_[0], " of variance of data"
print "2nd PC explained ", pca.explained_variance_ratio_[1], " of variance of data"

print "Projecting the input data on the principal component basis"
email_list_tfidf_pca = pca.transform(email_list_tfidf)
print "email_list_tfidf shape after PCA : ", len(email_list_tfidf), ", ", len(email_list_tfidf[0])
"""

# Feature Selection
from sklearn.feature_selection import SelectPercentile
totalfeatures = len(email_list[0])
selector = SelectPercentile(percentile=10)
email_list = selector.fit_transform(email_list, label_list)
print "Total count of samples: ", len(email_list)
#print "Selected ", len(email_list[0]), " text features out of total ",  totalfeatures
#print "Feature: ", email_list[0][:30]
#print "Feature Selector scores : ", selector.scores_[:30]	

# dump label_list and email_list
#pickle.dump(email_list, open("final_email_list.pkl", "w"))
#pickle.dump(label_list, open("final_label_list.pkl", "w"))


# NB classifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(email_list, label_list, test_size=0.3, random_state=42)

print "Start training Naive Bayes..."
NBclf = GaussianNB()
NBclf.fit(X_train, y_train)

NB_pred = NBclf.predict(X_test)
print "GaussianNB accuracy: ", NBclf.score(X_test, y_test)
print classification_report(y_test, NB_pred, target_names=["non-POI", "POI"])
print confusion_matrix(y_test, NB_pred, labels=range(2))


# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

print "Start training Decision Tree..."
DTparam = {'criterion' : ('gini', 'entropy'), 'min_samples_split' : [50, 70, 150, 300, 500]}
DTclf = GridSearchCV(DecisionTreeClassifier(), DTparam)
#DTclf = DecisionTreeClassifier(min_samples_split=20)
t0 = time()
DTclf.fit(X_train, y_train)
print "\tDecision Tree completes in ", time() - t0, "seconds"
print DTclf.best_estimator_

DT_pred = DTclf.predict(X_test)
print "Decision Tree accuracy: ", DTclf.score(X_test, y_test)
print classification_report(y_test, DT_pred, target_names=["non-POI", "POI"])
print confusion_matrix(y_test, DT_pred, labels=range(2))





