#!/usr/bin/env python
# coding: utf-8

# ## Maximum likelihood estimation of a normal random variable

# The following data is given:
# data =[1.42, 5.01, 2.45, 1.92, 1.41, 4.83, 1.81] 
# It is known to be sampled identicaly and without statistical dependency from a probability source upholding:
# $$ 𝑃(𝑥|𝜇, 𝜎^2) \sim 𝒩(𝜇, 𝜎^2) $$
# 
# ### Assignments:
# 
#     1. a. Calculate the model parameters according to the data.
#        b. Plot a graph showing the distribution of 𝒩 on x axis.
#     2. The true values of 𝜇 and 𝜎 are given. 𝜇 = 2, 𝜎 = 1.5. Add to the real distribution to the plot.
#     3. Generate a new sample series with length N=30 from a normal distribution with the true parameters 
#        and repeate tasks 1, 2.
#     4. Repeat task 3 10 times and create a single graph with the original distribuition and all the 10 distributions 
#        with the estimated parameters.
#     5. Repeate task 4 with N = 3000.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


# ### Calculate paramteres:


def calc_parameters(data):
    mu = np.mean(data)     # data average
    data_sum = 0
    for x in data:
        data_sum += (x - mu)**2 # sum of each square of (data point-average) -> sigma square
    sigma_sqr = data_sum/len(data) # dividing the square of sigma by the length of data
    return mu, sigma_sqr


# ### Calculate the probability according to the formula:
# $$ p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot e^{-\frac{1}{2 \pi \sigma^2} (x - \mu)^2} $$



def calc_p(x, mu, sigma_sqr): # 
    exp_part = np.exp(-(1/(2*sigma_sqr))*(x - mu)**2) # the power of e
    fraction_part = 1/np.sqrt(2*np.pi*sigma_sqr) # the multiplier the exponent
    return exp_part * fraction_part


# Reminder: <br>
# 𝜇 is the average of the data. <br><br>
# $𝜎^2$ is $ \frac{\sum (d - \mu)^2}{N}$
# 

# ### Task 1.a
# ###### Calculating the model paramters


# Calculating model parameters

# given data
data=[1.42, 5.01, 2.45, 1.92, 1.41, 4.83, 1.81] 

mu, sigma_sqr = calc_parameters(data)
print ("𝜇 =", mu)
print("𝜎^2 =", sigma_sqr)
print ("𝜎 =",np.sqrt(sigma_sqr))


# Calculating the probability density function for the given data
p = calc_p(data, mu, sigma_sqr)
for i in range (len(data)):
    print("data point:", data[i], "--- probability density:", p[i] )
    
# We can see that the further a nu,ber is from 𝜇 (the average), its probabilty density is getting smaller.
# This means it is less likely "to draw" the number.


x_axs = np.arange(-6, 9, 0.2)
MLE = calc_p(x_axs, mu, sigma_sqr) 
# Calculating maximum likelihood estimation for the all the numbers between -6 to 9 with 0.2 gaps


# ### Task 1.b
# ###### Plotting the maximum likelihood estimation


plt.plot(x_axs,MLE, label=f"MLE; mu = {mu:.1f}, sigma^2 = {sigma_sqr:.1f}")
plt.xlabel("X")
plt.ylabel("P(X)")
plt.title(r"$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\left( \frac{1}{2\pi\sigma^2} \right) (x - \mu)^2}$")
plt.legend(loc = (0.65,0.8))


# ### Task 2
# ###### Real parameter were given

real_mu = 2
real_sigma = 1.5
real_sigma_sqr = 1.5**2
real_MLE = calc_p(x_axs, real_mu, real_sigma_sqr)


plt.plot(x_axs, MLE, label="MLE; mu = 2.7, sigma^2 = 2.1")
plt.plot(x_axs, real_MLE, label="REAL; mu = 2, sigma^2 = 1.5")
plt.xlabel("X")
plt.ylabel("P(X)")
plt.title(r"$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\left( \frac{1}{2\pi\sigma^2} \right) (x - \mu)^2}$")
plt.legend(loc = (0.65,0.8))


# ### Task 3
# ###### Generate 30 new samples around the real values of 𝜇 and $𝜎^2$

def gen_data(n, mu, sigma_sqr):
    samples = np.random.normal(mu, np.sqrt(real_sigma_sqr), n)
    #print (samples)
    return samples


samples_30 = gen_data(30, real_mu, real_sigma)
print(samples_30)

mu_30, sigma_sqr_30 = calc_parameters(samples_30)
MLE_30 = calc_p(x_axs, mu_30, sigma_sqr_30)


# #### plotting singular graph

plt.plot(x_axs,MLE_30, label=f"MLE; mu = {mu_30:.1f}, sigma^2 = {sigma_sqr_30:.1f}")
plt.xlabel("X")
plt.ylabel("P(X)")
plt.title(r"$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\left( \frac{1}{2\pi\sigma^2} \right) (x - \mu)^2}$")
plt.legend(loc = (0.65,0.8))


# #### plotting comparative graph

plt.plot(x_axs,MLE_30, label=f"MLE; mu = {mu_30:.1f}, sigma^2 = {sigma_sqr_30:.1f}")
plt.plot(x_axs, real_MLE, label="REAL; mu = 2, sigma^2 = 2.25")

plt.xlabel("X")
plt.ylabel("P(X)")
plt.title(r"$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\left( \frac{1}{2\pi\sigma^2} \right) (x - \mu)^2}$")
plt.legend(loc = (0.65,0.8))


# ### Task 4
# ###### Repeat task 3 10 times

for i in range (0,10):
    samples = gen_data(30, real_mu, real_sigma)
    mu, sigma_sqr = calc_parameters(samples)
    MLE = calc_p(x_axs, mu, sigma_sqr)
    plt.plot(x_axs,MLE, label=f"MLE; mu = {mu:.1f}, sigma^2 = {sigma_sqr:.1f}")
plt.xlabel("X")
plt.ylabel("P(X)")
plt.title(r"$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\left( \frac{1}{2\pi\sigma^2} \right) (x - \mu)^2}$")
plt.plot(x_axs, real_MLE, label="REAL; mu = 2, sigma^2 = 2.25",color ='b')
plt.legend(loc = (1,0.37))
plt.show()


# ### Task 5
# ###### Redo task 4 but with N=3000

for i in range (0,10):
    samples = gen_data(3000, real_mu, real_sigma)
    mu, sigma_sqr = calc_parameters(samples)
    MLE = calc_p(x_axs, mu, sigma_sqr)
    plt.plot(x_axs,MLE, label=f"MLE; mu = {mu:.1f}, sigma^2 = {sigma_sqr:.1f}")
plt.xlabel("X")
plt.ylabel("P(X)")
plt.title(r"$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\left( \frac{1}{2\pi\sigma^2} \right) (x - \mu)^2}$")
plt.plot(x_axs, real_MLE, label="REAL; mu = 2, sigma^2 = 2.25",color ='b')
plt.legend(loc = (1,0.37))
plt.show()



