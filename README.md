
# Regression: Simple Linear Regression

In this notebook, I will use data on house sales in King County to predict house prices using simple (one input) linear regression. This is the part of University of Washington Machine learning specialization. I will perform below things:

* Use Python 3
* Use Numpy, Pandas, Scikit-learn to compute important summary statistics
* Implement function to compute the Simple Linear Regression weights using the closed form solution
* Implement function to make predictions of the output given the input feature
* Turn the regression around to predict the input given the output
* Compare two different models for predicting house prices

# Load necessary libraries


```python
import numpy as np
import pandas as pd
from sklearn import cross_validation
```

# Load house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
sales = pd.read_csv('kc_house_data.csv',dtype = {'bathrooms':float, 'waterfront':int, 
                                                 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
                                                 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 
                                                 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 
                                                 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
```

# Split data into training and testing


```python
train_data,test_data =cross_validation.train_test_split(sales,test_size=0.2, random_state=0)
```

# Build a generic simple linear regression function 


```python
def simple_linear_regression(input_feature, output):
    
    df = pd.DataFrame({'X':input_feature.as_matrix(),'Y':output.as_matrix()})
    correlation = df.corr(method='pearson').iloc[1,0]
     
    # use the formula for the slope
    slope = correlation * np.std(output) / np.std(input_feature)
    
    # use the formula for the intercept
    intercept = np.mean(output) - slope * np.mean(input_feature)
    
    return (intercept, slope)
```


```python
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])
```

# Predicting Values

Now that we have the model parameters: intercept & slope we can make predictions.


```python
def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + slope * input_feature
    
    return predicted_values
```

Now that we can calculate a prediction given the slope and intercept let's make a prediction. Use (or alter) the following to find out the estimated price for a house with 2650 squarefeet according to the squarefeet model we estiamted above.


```python
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print ("The estimated price for a house with {} squarefeet is {}" .format(my_house_sqft, estimated_price))
```

    The estimated price for a house with 2650 squarefeet is 704259.6128700661
    

# Residual Sum of Squares

Now that we have a model and can make predictions let's evaluate our model using Residual Sum of Squares (RSS).


```python
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_value = intercept + slope * input_feature
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    RSS = (predicted_value - output) ** 2
    RSS = RSS.sum()
    return(RSS)
```


```python
rss_prices_on_sqft = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : {}'.format(str(rss_prices_on_sqft)))
```

    The RSS of predicting Prices based on Square Feet is : 267770022739753.47
    

# Predict the squarefeet given price

We will predict the squarefoot given the price. Since we have an equation y = a + b\*x we can solve the function for x. 


```python
def inverse_regression_predictions(output, intercept, slope):
    #Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept)/slope
    return estimated_feature
```

Now that we have a function to compute the squarefeet given the price from our simple regression model let's see how big we might expect a house that costs $800,000 to be.


```python
my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print("The estimated squarefeet for a house worth {} is {}".format(my_house_price, estimated_squarefeet))
```

    The estimated squarefeet for a house worth 800000 is 2987.151366648074
    

# New Model: estimate prices from bedrooms

We have made one model for predicting house prices using squarefeet, but there are many other features. 
Using simple linear regression function to estimate the regression parameters from predicting Prices based on number of bedrooms.


```python
# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
bedrooms_intercept, bedrooms_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])
```

# Test Linear Regression Algorithm

Now we have two models for predicting the price of a house. Calculate the RSS on the TEST data. Compute the RSS from predicting prices using bedrooms and from predicting prices using squarefeet.


```python
# Compute RSS when using bedrooms on TEST data:
rss_prices_on_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bedrooms_intercept, bedrooms_slope)
print('The RSS of predicting Prices based on Bedrooms Feet is : {}'.format(str(rss_prices_on_bedrooms)))
```

    The RSS of predicting Prices based on Bedrooms Feet is : 472745600358876.1
    


```python
# Compute RSS when using squarefeet on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : {}'.format(str(rss_prices_on_sqft)))
```

    The RSS of predicting Prices based on Square Feet is : 267770022739753.47
    
