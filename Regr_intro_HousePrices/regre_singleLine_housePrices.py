'''
@author: Gaio
@summary: Linear Regression - Predicting House Prices
'''

#Start Graphlab create
import graphlab

#import data
sales = graphlab.SFrame('kc_house_data.gl/')

#redirect output
graphlab.canvas.set_target('ipynb')

#split train & test sets
train_data,test_data = sales.random_split(.8,seed=0)


def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    num_inputs = input_feature.size()
    sum_input_feature = input_feature.sum()    
    sum_output = output.sum()    
    # compute the product of the output and the input_feature and its sum
    prod_I_O = input_feature * output
    sum_prodIO = prod_I_O.sum()        
    # compute the squared value of the input_feature and its sum
    sqrd_I = input_feature*input_feature
    sum_sqrd_I = sqrd_I.sum()   
    
    #numerator = (sum of X*Y) - (1/N)*((sum of X) * (sum of Y))
    numerator = sum_prodIO - (1/num_inputs) * (sum_input_feature *sum_output)
    print "numerator:", numerator 
    #denominator = (sum of X^2) - (1/N)*((sum of X) * (sum of X)) 
    denominator = sum_sqrd_I - (1/num_inputs) * (sum_input_feature * sum_input_feature)
    print "denominator: ", denominator
    # slope
    slope = numerator/denominator 
    #intercept = (mean of Y) - slope * (mean of X)
    intercept = output.mean() - slope * input_feature.mean()
    
    return(intercept, slope)

def simple_linear_regression_x(input_feature, output):
    Xi = input_feature
    Yi = output
    N = len(Xi)
    # compute the mean of  input_feature and output
    Ymean = Yi.mean()
    Xmean = Xi.mean()
    
    # compute the product of the output and the input_feature and its mean
    SumYiXi = (Yi * Xi).sum()
    YiXiByN = (Yi.sum() * Xi.sum()) / N
    
    # compute the squared value of the input_feature and its mean
    XiSq = (Xi * Xi).sum()
    XiXiByN = (Xi.sum() * Xi.sum()) / N
    
    # use the formula for the slope
    slope = (SumYiXi - YiXiByN) / (XiSq - XiXiByN)
    
    # use the formula for the intercept
    intercept = Ymean - (slope * Xmean)
    return (intercept, slope)

#test function
test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)

#use function to calculate the estimated slope and intercept on the training data to predict ‘price’ given ‘sqft_living’
input_feature = train_data['sqft_living']
output = train_data['price']
(squarfeet_intercept, squarfeet_slope) =  simple_linear_regression_x(input_feature, output)
print "Intercept: " + str(squarfeet_intercept)
print "Slope: " + str(squarfeet_slope)

def get_regression_predictions(input_feature, intercept, slope):
    # y = mx + q
    predicted_values = input_feature * slope + intercept
    
    return predicted_values

#use the function to predict house prices
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, squarfeet_intercept, squarfeet_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)
#The estimated price for a house with 2650 squarefeet is $689282.72

#compute RSS
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # predictions
    fitted_line = input_feature * slope + intercept
    # then compute the residuals , square them and add them up
    residuals = fitted_line - output
    sqrd_residuals = residuals * residuals
    RSS = sqrd_residuals.sum()
    return(RSS)

#apply to my data
print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) # 0.0
rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], squarfeet_intercept, squarfeet_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)
#The RSS of predicting Prices based on Square Feet is : 1.2072119188e+15

#predit sqrft from prices
def inverse_regression_predictions(output, intercept, slope):
    estimated_feature= (output - intercept)/slope
    return estimated_feature

my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, squarfeet_intercept, squarfeet_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)
#The estimated squarefeet for a house worth $800000.00 is 3070

#use bedrooms
input_feature_b = train_data['bedrooms']
output = train_data['price']
(squarfeet_intercept_b, squarfeet_slope_b) =  simple_linear_regression(input_feature_b, output)
print "Intercept: " + str(squarfeet_intercept_b)
print "Slope: " + str(squarfeet_slope_b)
#Intercept: 7388.31464205
#Slope: 157886.92748
rss_prices_on_sqft = get_residual_sum_of_squares(train_data['bedrooms'], train_data['price'], squarfeet_intercept_b, squarfeet_slope_b)
print 'The RSS of predicting Prices based on Bedrooms is : ' + str(rss_prices_on_sqft)
#The RSS of predicting Prices based on Bedrooms is : 2.15635611736e+15