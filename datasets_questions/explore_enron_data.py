#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)
print enron_data["PRENTICE JAMES"]["total_stock_value"]
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print np.sum([1 for name in enron_data if enron_data[name]["poi"]])

print "--------------"
print enron_data["LAY KENNETH L"]
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]
print "--------------"

print np.sum([1 for name in enron_data if enron_data[name]["salary"] != "NaN"])
print np.sum([1 for name in enron_data if enron_data[name]["email_address"] != "NaN"])

# data_list = featureFormat(enron_data, ["salary"])
# print targetFeatureSplit(data_list)

print np.sum([1 for name in enron_data if enron_data[name]["total_payments"] == "NaN"])
print np.sum([1 for name in enron_data if enron_data[name]["poi"] and enron_data[name]["total_payments"] == "NaN"])
print np.sum([1 for name in enron_data if enron_data[name]["poi"]])