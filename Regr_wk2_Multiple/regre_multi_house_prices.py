'''
@author: Gaio
@summary: Multiple Regression - Predicting House Prices
'''

#Start Graphlab create
import graphlab
import numpy as np

#import data
sales = graphlab.SFrame('kc_house_data.gl/')

#redirect output
graphlab.canvas.set_target('ipynb')

#split train & test sets
train_data,test_data = sales.random_split(.8,seed=0)

#add transformation of existing vars to generate new features
train_data['bedrooms_squared'] = train_data['bedrooms']*train_data['bedrooms']
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
train_data['log_sqft_living'] = np.log(train_data['sqft_living'])
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']

test_data['bedrooms_squared'] = test_data['bedrooms']*test_data['bedrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
test_data['log_sqft_living'] = np.log(test_data['sqft_living'])
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

#calculate mean
test_data['bedrooms_squared'].mean()#12.44667770158429
test_data['bed_bath_rooms'].mean()#7.503901631591395
test_data['log_sqft_living'].mean()#7.55027467964594
test_data['lat_plus_long'].mean()#-74.65333497217306

#create regression model based on new features
features1=['sqft_living','bedrooms','bathrooms','lat','long']
features2=['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']
features3=['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms','bedrooms_squared','log_sqft_living','lat_plus_long']

features_model1 = graphlab.linear_regression.create(train_data,target='price',features=features1)
features_model2 = graphlab.linear_regression.create(train_data,target='price',features=features2)
features_model3 = graphlab.linear_regression.create(train_data,target='price',features=features3)

#Quiz Question: What is the sign (positive or negative) for the coefficient/weight for ‘bathrooms’ in Model 1? +
features_model1['coefficients']
#Quiz Question: What is the sign (positive or negative) for the coefficient/weight for ‘bathrooms’ in Model 2? -
features_model2['coefficients']

#Which model (1, 2 or 3) has lowest RSS on TRAINING Data?
print features_model1.evaluate(train_data)
print features_model2.evaluate(train_data)
print features_model3.evaluate(train_data)#!yo!
#Which model (1, 2 or 3) has lowest RSS on TESTING Data?
print features_model1.evaluate(test_data)
print features_model2.evaluate(test_data)#!yo!
print features_model3.evaluate(test_data)