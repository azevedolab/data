#!/usr/bin/env python3
#
# Regression models for data from CASF-2016 Ki dataset
# (de Azevedo et al., 2024).
#
# References:
# de Azevedo WF Jr, Quiroga R, Villarreal MA, da Silveira NJF,
# Bitencourt-Ferreira G, da Silva AD, Veit-Acosta M, Oliveira PR, Tutone M,
# Biziukova N, Poroikov V, Tarasova O, Baud S. SAnDReS 2.0: Development of
# machine-learning models to explore the scoring function space. J Comput Chem.
# 2024 Jun 20. doi: 10.1002/jcc.27449. Epub ahead of print. PMID: 38900052.
#
# Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and
# TensorFlow. O'Reilly Media. Kindle Edition.
#
################################################################################
# Dr. Walter F. de Azevedo, Jr.                                                #
# https://github.com/azevedolab                                                #
# July 20, 2024                                                                #
################################################################################
#
# Define cross_validation() function
# Function to set up k-fold class. Kfold class to build a n fold
# cross-validation loop and test the generalization ability of regression. With
# cross-validation, we generally obtain a more conservative estimate(that is,
# the error is larger). The cross-validation estimate is a better estimate of
# how well we could generalize to predict on unseen data.
#
# Reference:
# Coelho LP, Richert W. (2015) Building Machine Learning Systems with
# Python. 2nd ed. Packt Publishing Ltd. Birmingham UK. 301 pp. See page 162
# (Cross-validation for regression)
def cross_validation(model,X,y,n_splits,random_state,verbose):

    # Import section
    from sklearn.model_selection import KFold, cross_val_score
    from warnings import simplefilter
    import numpy as np

    # Set up k-fold class
    kf = KFold(n_splits=n_splits,shuffle=True, random_state=random_state)

    # Ignore all future warnings
    simplefilter(action='ignore', category=DeprecationWarning)

    # Looping through kf.split()
    for train,test in kf.split(X):

        # Generate regression model
        model.fit(X[train],y[train])

    # Show Walsh metrics if requestes
    if verbose:
        # Show average coefficient of determination using n-fold crossvalidation
        scores = cross_val_score(model,X,y,cv=kf)
        msg_out = "Average coefficient of determination using n-fold "
        msg_out += "cross-validation"
        print("\n"+msg_out+":",np.mean(scores))

    # Return model
    return model

# Define regression_metrics() function
def regression_metrics(y,y_pred):
    """Function to calculate regression metrics as recommended by Walsh et 2021
    and de Azevedo Jr et al., 2024"""

    # Import section for this function
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import r2_score
    from scipy import stats
    import numpy as np

    # Determine metrics
    rmse = root_mean_squared_error(y,y_pred)
    mae = median_absolute_error(y,y_pred)
    R2 = r2_score(y,y_pred)
    r_pearson,p_pearson = stats.pearsonr(y,y_pred)
    rho,p_spearman = stats.spearmanr(y,y_pred)
    dome = np.sqrt(rmse**2 + mae**2 + (R2 -1)**2 )
    edomer2 = np.sqrt(rmse**2 + mae**2 + (R2 -1)**2 + (r_pearson**2 - 1)**2)
    edomerho = np.sqrt(rmse**2 + mae**2 + (R2 -1)**2 + (rho**2 - 1)**2)
    edome = np.sqrt(rmse**2 + mae**2+(R2-1)**2 +(r_pearson**2-1)**2 +(rho-1)**2)

    # Return metrics
    return r_pearson,p_pearson,rho,p_spearman,rmse,mae,R2,dome,edomer2,\
    edomerho,edome

# Define scatter_plot() function
def scatter_plot(y,y_pred,data_point,color_data,color_regression,plot_out,dpi):
    """Function to generate a basic scatter plot"""

    # Import section
    import numpy as np
    import matplotlib.pyplot as plt

    # Create basic scatterplot
    plt.plot(y, y_pred,data_point,color=color_data)

    # Obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(y,y_pred, 1)

    # Use color_regression as color for regression line
    plt.plot(y, m*y+b, color=color_regression)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid()
    plt.savefig(plot_out,dpi=dpi)
    plt.close()

# Define write_metric() function
def write_metrics(X_train,y_train,X_test,y_test,model_in,csv_out):

    # Predict
    y_train_pred = model_in.predict(X_train)
    y_test_pred = model_in.predict(X_test)

    # Set up lists
    y_list = [y_train,y_test]
    y_pred_list = [y_train_pred,y_test_pred]
    data_list = ["Training Set","Test Set"]

    # Set up empty string
    data_out = "Data,n,r,p-value(r),r2,rho,p-value(rho),RMSE,MAE,R2,DOME,"
    data_out += "EDOMEr2,EDOMErho,EDOME\n"

    # Looping through data
    for count,y in enumerate(y_list):

        # Call regression_metrics() function
        r_pearson,p_pearson,rho,p_spearman,rmse,mae,R2,dome,edomer2,edomerho,\
                    edome=regression_metrics(y_list[count],y_pred_list[count])

        # Set up output line
        line_out = data_list[count]+","+str(len(y))+","
        line_out+= "{:.4f}".format(r_pearson)+","+"{:.4e}".format(p_pearson)+","
        line_out += "{:.4f}".format(r_pearson**2)+","+"{:.4f}".format(rho)+","
        line_out += "{:.4e}".format(p_spearman)+","+"{:.4f}".format(rmse)+","
        line_out += "{:.4f}".format(mae)+","+"{:.4f}".format(R2)+","
        line_out += "{:.4f}".format(dome)+","+"{:.4f}".format(edomer2)+","
        line_out += "{:.4f}".format(edomerho)+","+"{:.4f}".format(edome)

        # Update data_out
        data_out += line_out+"\n"

    # Open a new file and write data_out
    fo_out = open(csv_out,"w")
    fo_out.write(data_out)
    fo_out.close()

    # Return prediction
    return y_test_pred

# Get data from csv files
from sklearn import preprocessing
import pandas as pd

msg_out = "\nReading and preprocessing CASF-2016 Ki dataset"
print(msg_out,end="...")

# Set up feature_list
feature_list = ["Gauss 2","C","Gauss 1","Hydrophobic","N","Torsional",
"B-factor ratio (Ligand/Receptor)","Torsions","S","Receptor B-factor(A2)","Q",
"Average Q","Hydrogen","O"]

# Get training set
data = pd.read_csv("datasets/CASF-2016_Ki_training.csv")
X_train = data[feature_list]
y_train = data.iloc[:,8]
scaler_train = preprocessing.StandardScaler().fit(X_train)
X_train = scaler_train.transform(X_train)

# Get test set
data = pd.read_csv("datasets/CASF-2016_Ki_test.csv")
X_test = data[feature_list]
y_test = data.iloc[:,8]
scaler_test = preprocessing.StandardScaler().fit(X_test)
X_test = scaler_test.transform(X_test)

print("done!")
################################################################################
# Build a regression model with Multilayer Perceptron
msg_out = "\nRegression with Multilayer Perceptron"
print(msg_out,end="...")
from sklearn.neural_network import MLPRegressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(300,300,300,),activation='relu',
solver='adam', alpha=0.0001, learning_rate='constant',
learning_rate_init=0.001, validation_fraction=0.2, max_iter=700, shuffle=True,
random_state=46).fit(X_train, y_train)

# Call cross_validation() function
mlp_reg = cross_validation(mlp_reg,X_train,y_train,5,46,False)

# Call write_metrics() function
csv_out = "metrics_CASF-2016_Ki_MLPRegressor.csv"
y_test_pred_mlp = write_metrics(X_train,y_train,X_test,y_test,mlp_reg,
                                csv_out)

# Call scatter_plot() function
plot_out = "scatter_plot_CASF-2016_Ki_MLPRegressor.png"
dpi = 600
scatter_plot(y_test,y_test_pred_mlp,".","blue","black",plot_out,dpi)
print("done!")
################################################################################
# Build a regression model with Random Forest
msg_out = "\nRegression with Random Forest"
print(msg_out,end="...")
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=1000, max_depth=100,
warm_start = False, random_state=46, oob_score=True,
criterion="friedman_mse", n_jobs = -1).fit(X_train, y_train)

# Call cross_validation() function
rf_reg = cross_validation(rf_reg,X_train,y_train,5,46,False)

# Call write_metrics() function
csv_out = "metrics_CASF-2016_Ki_RandomForestRegressor.csv"
y_test_pred_rf = write_metrics(X_train,y_train,X_test,y_test,rf_reg,
                                csv_out)

# Call scatter_plot() function
plot_out = "scatter_plot_CASF-2016_Ki_RandomForestRegressor.png"
dpi = 600
scatter_plot(y_test,y_test_pred_rf,".","blue","black",plot_out,dpi)
print("done!")
################################################################################
msg_out = "\nRegression with Stochastic Gradient Descent"
print(msg_out,end="...")
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(loss='squared_error',penalty="l2",alpha=0.0001,
fit_intercept=False, max_iter=100000,tol=1e-5, shuffle=True,verbose=0,
validation_fraction=0.2,random_state=46).fit(X_train, y_train)

# Call cross_validation() function
sgd_reg = cross_validation(sgd_reg,X_train,y_train,5,46,False)

# Call write_metrics() function
csv_out = "metrics_CASF-2016_Ki_SGDRegressor.csv"
y_test_pred_sgd = write_metrics(X_train,y_train,X_test,y_test,sgd_reg,
                                csv_out)

# Call scatter_plot() function
plot_out = "scatter_plot_CASF-2016_Ki_SGDRegressor.png"
dpi = 600
scatter_plot(y_test,y_test_pred_sgd,".","blue","black",plot_out,dpi)
print("done!")
################################################################################
# Open a new file with predicted values
fo_new =  open("predictions_CASF-2016_Ki_test_set.csv","w")
data_out = "y,MLPRegressor,RandomForestRegressor,SGDRegressor\n"
for count in range(len(y_test)):
    data_out += str(y_test[count])+","+str(y_test_pred_mlp[count])+","
    data_out += str(y_test_pred_rf[count])+","+str(y_test_pred_sgd[count])+"\n"
fo_new.write(data_out)
fo_new.close()