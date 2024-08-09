#!/usr/bin/env python3
#
# Linear Regression for a dataset read from a csv file. We applied closed form
# equations to determine theta1 and theta0 (James et al., 2021). We have two
# datasets: CASF-2016 Ki (de Azevedo et al., 2024) and Amsterdam Apartments
# dataset (Wolf, 2022).
# This code is for linear models as expressed by the following equation:
# y = theta0 + theta1*X, where theta0 is the fit intercept and theta1 the
# coeficient, theta0 and theta1 are the model’s parameters (Géron, 2023).
#
# References
# de Azevedo WF Jr, Quiroga R, Villarreal MA, da Silveira NJF,
# Bitencourt-Ferreira G, da Silva AD, Veit-Acosta M, Oliveira PR, Tutone M,
# Biziukova N, Poroikov V, Tarasova O, Baud S. SAnDReS 2.0: Development of
# machine-learning models to explore the scoring function space. J Comput Chem.
# 2024 Jun 20. doi: 10.1002/jcc.27449. Epub ahead of print. PMID: 38900052.
#
# Géron, Aurélien. 2023. Hands-on Machine Learning with Scikit-Learn, Keras,
# and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems.
# 3rd ed. CA 95472: O’Reilly.
#
# James, G., Witten, D., Hastie, T., & Tibshirani, R. 2021. An introduction to
# statistical learning: With applications in R (2nd ed.). Springer.
#
# Wolf, Andrew. The Machine Learning Simplified: A Gentle Introduction to
# Supervised Learning. Andrew Wolf. Kindle Edition.
#
################################################################################
# Dr. Walter F. de Azevedo, Jr.                                                #
# https://github.com/azevedolab                                                #
# July 20, 2024                                                                #
################################################################################
#
# Import section
import numpy as np
import reg2D_model
import csv
from scipy.stats import pearsonr, spearmanr

# Read csv file
# CASF-2016 dataset
# de Azevedo WF Jr, Quiroga R, Villarreal MA, da Silveira NJF,
# Bitencourt-Ferreira G, da Silva AD, Veit-Acosta M, Oliveira PR, Tutone M,
# Biziukova N, Poroikov V, Tarasova O, Baud S. SAnDReS 2.0: Development of
# machine-learning models to explore the scoring function space. J Comput Chem.
# 2024 Jun 20. doi: 10.1002/jcc.27449. Epub ahead of print. PMID: 38900052.
file_in = "CASF-2016_Ki_training.csv"

# Amsterdam Apartments dataset (Amsterdam)
# Wolf, Andrew. The Machine Learning Simplified: A Gentle Introduction to
# Supervised Learning. Andrew Wolf. Kindle Edition.
#file_in = "Amsterdam_apartments_training_set.csv"

# Get header from a csv file
fo = open(file_in,"r")
csv_in = csv.reader(fo)
for line in csv_in:
    i_y = line.index("pKi")
    i_X = line.index("Gauss 2")
    #i_y = line.index("Price")
    #i_X = line.index("Area")
    break
fo.close()

# Get numerical data
data = np.genfromtxt(file_in, delimiter=',', skip_header = 1)
X = data[:,i_X]
y = data[:,i_y]

# Calculate parameters (theta0,theta1) as defined elsewhere
# James, G., Witten, D., Hastie, T., & Tibshirani, R. 2021. An introduction to
# statistical learning: With applications in R (2nd ed.). Springer.
X_bar = X.mean()
y_bar = y.mean()
theta1 = np.sum( (X - X_bar)*(y - y_bar) )/np.sum( (X - X_bar)**2 )
theta0 = y_bar - theta1*X_bar

# Make predictions
X_new = np.array([[X.min()],[X.max()]])
y_predict = theta0 + theta1*X_new

# Plotting with reg2D_model() function
theta0_str = "{:.8f}".format(theta0)
theta1_str = "{:.8f}".format(theta1)
#title_str = "CASF-2016 K$_i$ (Training Set)"
title_str = "Amsterdan Apartments (Training Set)"
label_str = "Predictions ( y = "+theta0_str+" + "+theta1_str+" x )"
reg2D_model.plotting(title_str,
                #X,y,"b.",[0,120,0,120],
                #X_new,y_predict,"r-",label_str,"Area(m$^2$)","Price(10K*Euros)",
                    X,y,"b.",[-200,3200,0,15],
                    X_new,y_predict,"r-",label_str,"Gauss 2","pK$_i$",
                    "upper left",True,file_in.replace(".csv",".png"),600)

# Show predictive performance
msg_out = "\nPredictive performance"
print(msg_out)
y_full_predictions = theta0 + theta1*X
rss = np.sum( np.square(y_full_predictions - y)   )
p_corr = pearsonr(y,y_full_predictions)
s_corr = spearmanr(y,y_full_predictions)
print("RSS = {:.5f}".format(rss))
print("r = {:.5f}".format(p_corr[0]),"\tp-value = {:.5e}".format(p_corr[1]))
print("rho = {:.5f}".format(s_corr[0]),"\tp-value = {:.5e}".format(s_corr[1]))