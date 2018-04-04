# ml-assignments
Repository for work in the CU Machine Learning course

## A1 - Ridge Regression and Active Learning

### A1.1 - Ridge Regression

The formula for finding the weights $w_{RR}$ is:   

$$w_{RR} = arg\,min_w~||y - Xw||^2 + \lambda || w ||^2$$

To calculate the wights $w_{RR}$ we use this matrix math:   

$$w_{RR} = (X^TX \cdot \lambda I)^{-1} \cdot X^Ty$$


### A1.2 - Active Learning

