'''
@author: Gaio
@summary: predict house prices with regression models
'''

import matplotlib.pyplot as plt 
#%matplotlib inline

#Start Graphlab create
import graphlab
#only for cloud notebook
#graphlab.product_key.set_product_key('your licence key here')

#Load house sales data
sales = graphlab.SFrame('home_data.gl/')

#exploring the dataset
#redirects output to the python notebook - applicable just if you run on virtual nb
#graphlab.canvas.set_target('ipynb')

sales.show(view="Scatter Plot",x="sqft_living",y="price")

#split training & test data sets
train_data, test_data = sales.random_split(.8,seed=0)

#create regression model of sqft_living to price
sqft_model=graphlab.linear_regression.create(train_data,target='price',features=['sqft_living'])

#evaluate the model
print test_data['price'].mean()
print sqft_model.evaluate(test_data)

#visualize the predictions
plt.plot(test_data['sqft_living'],test_data['price'],'.',test_data['sqft_living'],sqft_model.predict(test_data),'-')

#check paramenters learned. Note the angle D(price)/D(sqft) ~ avg price/sqrft
sqft_model.get('coefficients')

#add additional features
add_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']

#visualize data ZIPCODE for assignment = 98039
sales[add_features].show()
sales.show(view='BoxWhisker Plot', x='zipcode',y='price')

#create regression model based on add_features
add_features_model = graphlab.linear_regression.create(train_data,target='price',features=add_features)

#compare the performance of the 2 models
print sqft_model.evaluate(test_data)
print add_features_model.evaluate(test_data)

#apply the learned model to predict house prices
house1=sales[sales['id']== '5309101200']
print house1['price']
print sqft_model.predict(house1)
print add_features_model.predict(house1)

house2=sales[sales['id']== '1925069082']
print house2['price']
print sqft_model.predict(house2)
print add_features_model.predict(house2)

#assignment

#part 1
zip_98039 =sales[sales['zipcode']== '98039']
avgPrice = zip_98039['price'].mean()
print avgPrice #2160606.6

#part 2
#logic filters: https://turi.com/learn/userguide/sframe/data-manipulation.html
range_houses = sales[(sales['sqft_living']>2000) & (sales['sqft_living'] <=4000)]
print range_houses.num_rows() #9118
print sales.num_rows() #21613
percRangeHouses = float(range_houses.num_rows())/float(sales.num_rows())
print percRangeHouses #0.421875722945

#part 3
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors
]

#add additional features
add_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']

#split training & test data sets
train_data, test_data = sales.random_split(.8,seed=0)

#create regression models 
add_features_model = graphlab.linear_regression.create(train_data,target='price',features=add_features)
advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features)

#compare the performance of the 2 models
print add_features_model.evaluate(test_data) #{'max_error': 3473149.1740256962, 'rmse': 179656.886148893} 
print advanced_features_model.evaluate(test_data) #{'max_error': 3615590.823853611, 'rmse': 165303.45649426847}