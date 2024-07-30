# Maximum Likelihood Estimation of a normal random variable
1st class project  in intro ML and data science course.

Maximum Likelihood Estimation (MLE) is a method used in statistics to estimate the parameters of a probability distribution by maximizing a likelihood function. The likelihood function measures how likely it is to observe the given sample data for different parameter values. MLE selects the parameter values that maximize this likelihood, making the observed data most probable.

In this exercise we were given a small dataset and were asked to calculate the parameters of the model (𝜇, 𝜎^2), compare them with the real values of the parameters and generate new datasets and see how it incluences the accuracy of the model.

## The exercise
The following data is given:
data =[1.42, 5.01, 2.45, 1.92, 1.41, 4.83, 1.81] 
It is known to be sampled identicaly and without statistical dependency from a probability source upholding:
$$𝑃(𝑥|𝜇, 𝜎^2) \sim 𝒩(𝜇, 𝜎^2)$$

### Assignments:

    1. a. Calculate the model parameters according to the data.
       b. Plot a graph showing the distribution of 𝒩 on x axis.
    2. The true values of 𝜇 and 𝜎 are given. 𝜇 = 2, 𝜎 = 1.5, Plot it and compare to the distribution you calculated
    3. Generate a new sample series with length N=30 from a normal distribution with the true parameters 
       and repeate tasks 1, 2.
    4. Repeat task 3 10 times and create a single graph with the original distribuition and all the 10 distributions 
       with the estimated parameters.
    5. Repeate task 4 with N = 3000.
