
# coding: utf-8

# In[2]:

'''
@author: Gaio
@summary: Multiple Regression - Predicting House Prices with graphlab & gradient descent
'''


# In[3]:

#imports
import graphlab
import numpy as np
import math


# In[4]:

#import data
sales = graphlab.SFrame('kc_house_data.gl/')


# In[5]:

#split data into train & test sets
train_data,test_data = sales.random_split(.8,seed=0)


# In[6]:

#apply multiple regression on feature set
example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data, target = 'price', features = example_features, 
                                                  validation_set = None)


# In[7]:

#print weigths
example_weight_summary = example_model.get("coefficients")
print example_weight_summary


# In[8]:

#apply model to train data
example_predictions = example_model.predict(train_data)
print example_predictions[0] #271789.505878


# In[9]:

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’
    features_sframe=data_sframe[features]
    print features_sframe
    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’
    output_sarray= data_sframe[output]
    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy() # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)


# In[10]:

#test function
output_array = get_numpy_data(sales,example_features, 'price')


# In[ ]:




# In[ ]:




# In[11]:

#predicted output = matrix mult of predicted weights (feature_matrix) & predicted output (output_array)
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


# In[12]:

#derivative of regression cost function = 2 dot product of feature & error predictions
def feature_derivative(errors, feature):
    derivative = 2 * np.dot(errors,feature)
    return(derivative)


# In[13]:

#gradient descent function
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors,feature_matrix[:, i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative**2
            # update the weight based on step size and derivative:
            #each feature weight by subtracting the step size times the derivative for that feature given the current weights
            weights[i] -= step_size * derivative 
            
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


# In[14]:

#test function input
simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7


# In[15]:

#run test
simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)


# In[16]:

print simple_weights


# In[17]:

print simple_feature_matrix


# In[18]:

#use the weights on the test data
simple_features = ['sqft_living']
my_output= 'price'
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)


# In[19]:

np.ma.round(a=281.91211912, decimals=1)


# In[20]:

predictions = predict_outcome(test_simple_feature_matrix,test_output)


# In[21]:

predictions = predict_outcome(test_simple_feature_matrix,simple_weights)


# In[22]:

#predicted price for the 1st house
print predictions[0]


# In[23]:

def compute_RSS(predictions, output):
     
    #residual
    residual = output - predictions

    # square up
    residual_squared = residual **2
    
    #sum of squared residuals
    RSS = residual_squared.sum()

    return(RSS)


# In[24]:

RSS = compute_RSS(predictions,test_output)


# In[25]:

print RSS


# In[26]:

#2nd model
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9


# In[28]:

#run test
model_2_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size,tolerance)


# In[29]:

print model_2_weights


# In[30]:

(test_model_feature_matrix, test__model_output) = get_numpy_data(test_data, model_features, my_output)


# In[31]:

predictions = predict_outcome(test_model_feature_matrix,model_2_weights)


# In[32]:

print prediction[0]


# In[33]:

print predictions[0]


# In[34]:

RSS = compute_RSS(predictions,test__model_output)


# In[35]:

print RSS


# In[36]:

print test_data[0]['price']

