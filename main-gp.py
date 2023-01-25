# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:18:12 2020

@author: ghaderi1
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import pandas as pd
import random 
from sklearn.model_selection import train_test_split

from GP import GP

np.random.seed(1234)

if __name__ == "__main__":    
    
    
    D = 1
    noise = 0.05
    
    
    dataset = pd.read_excel("SBR.xlsx", escapechar="\\")
    X00 = dataset.iloc[:,0:1].values
    Y00 = dataset.iloc[:,1:2].values
    
    X, x_test, y, y_test = train_test_split(X00, Y00, test_size=0.1)
    N = np.size(X) 

   
    # Training data    
    y = y 

    # Test data
    nn = 100
    X_star = np.linspace(1, 3.9, nn)[:,None]

    
    # Define model
    model = GP(X, y)
        
    # Train 
    model.train()
    
    # Predict
    y_pred, y_var = model.predict(X_star)
    y_var = np.abs(np.diag(y_var))
           
    
    # Draw samples from the prior and posterior
    Y0 = model.draw_prior_samples(X_star, 50)
    YP = model.draw_posterior_samples(X_star, 50)
    
    # Plot predictions
    plt.figure(1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    #plt.plot(X_star, y_star, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star, y_pred, 'r--', label = "Prediction", linewidth=2)
    lower = y_pred - 6.0*np.sqrt(y_var[:,None])
    upper = y_pred + 6.0*np.sqrt(y_var[:,None])
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X,y,'bo', markersize = 12, alpha = 0.5, label = "Data")
    plt.legend(frameon=False,loc='upper left')
    ax = plt.gca()
    plt.xlabel('$Stretch$')
    plt.ylabel('$Stress$')
    plt.axis([1,4,0,6])
    
    # Plot samples
    plt.figure(2, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(X_star,Y0,'g',linewidth=.2)
    ax = plt.gca()
    #ax.set_xlim([lb[0], ub[0]])
    plt.ylabel('$Stress$')
    plt.title("Prior samples")
    
    plt.figure(3, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(X_star,YP,'g',linewidth=.2)
    plt.plot(X,y,'bo', markersize = 12, alpha = 0.5, label = "Data")
    ax = plt.gca()
    plt.xlabel('$Stretch$')
    plt.ylabel('$Stress$')
    plt.title("Posterior samples")
    plt.axis([1,4,0,6])