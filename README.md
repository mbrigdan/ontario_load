## Summary

Applying probabilistic programming methods (pyro-ppl and pymc) to Ontario zonal demand data

## Data

Data retrieved from the IESO website: http://www.ieso.ca/en/Power-Data/Data-Directory

### Notes

- Data from 2020 is used to fit a sklearn StandardScaler object, which is used on the 2021 data, which is in turn broken into training and test datasets.

gp_learning.py: Gaussian Process regression using pymc
    
- fitting/sampling/predicting process very slow
- attempts at using HSGP instead lead to out of memory errors
  - worth investigating, because HSGP is supposed to be an appoximation that is faster/less memory?

neural_pyro.py: Bayesian Neural network using pyro-ppl
- Training using Stochastic Variational Inference (SVI)
- Sampling using MCMC+NUTS
  - very slow - too large to fit onto GPU

### Results:
Samples of predictions showing BNN mean output as well as +/- 2 standard deviations interval (green shaded area): 
![Resulting plot of BNN trained using SVI #1](plots/BNN%20test%201.png)
![Resulting plot of BNN trained using SVI #2](plots/BNN%20test%202.png)

Samples of predictions showing output from MCMC sampling trained model (only 100 samples):
![Resulting plot of BNN trained using MCMC #1](plots/BNN%20test%20MCMC%201.png)
![Resulting plot of BNN trained using MCMC #1](plots/BNN%20test%20MCMC%202.png) 

Clearly strange results - but expected given the extremely limited sampling that was used due to the time sampling took.
Should only be viewed as a proof of concept.