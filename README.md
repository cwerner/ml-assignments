# ml-assignments
Repository for work in the CU Machine Learning course

## A1 - Ridge Regression and Active Learning

### A1.1 - Ridge Regression

The formula for finding the weights $w_{RR}$ is:   

$$ w_{RR} = arg min_{w} || y - Xw ||^2 + \Lambda || w ||^2 $$

To calculate $w_{RR}$ we use this matrix math:   

$$ w_${RR} = (X^TX \dot \Lambda I)${-1} \dot X^Ty $$

