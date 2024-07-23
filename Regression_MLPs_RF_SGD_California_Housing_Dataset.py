# Regression models for dataset California Housing Prices. Models built using
# the following regression methods: MLPRegressor, RandomForestRegressor, and
# SGDRegressor available from Scikit-Learn (Pedregosa et al., 2021). We employ
# the metrics recommended by
# [Walsh et al., 2021](https://doi.org/10.1038/s41592-021-01205-4) to analyze
# the predictive performance of the regression methods
# (e.g., RMSE, MAE, and R2).
#
# Optimized using GridSearch()
# https://colab.research.google.com/github/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb#scrollTo=tbxpzEsyXHb5
# Source: https://colab.research.google.com/github/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb#scrollTo=tbxpzEsyXHb5
#
# References:
# Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and
# TensorFlow. O'Reilly Media. Kindle Edition.
#
# Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel
# M, Prettenhofer P, Weiss R, Dubourg V, Verplas J, Passos A, Cournapeau D,
# Brucher M, Perrot M, Duchesnay E. Scikitlearn: Machine Learning in Python.
# J Mach Learn Res., 2011, 12, 2825-2830.
# [PDF](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
#
# Walsh I, Fishman D, Garcia-Gasulla D, Titma T, Pollastri G; ELIXIR Machine
# Learning Focus Group, Harrow J, Psomopoulos FE, # Tosatto SCE. DOME:
# recommendations for supervised machine learning validation in biology.
# Nat Methods. 2021, 18(10), 1122-1127.
# [DOI](https://doi.org/10.1038/s41592-021-01205-4)
#
# Import section
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    plt.savefig(plot_out,dpi=dpi)
    plt.close()

# Define write_metric() function
def write_metrics(X_train_full,y_train_full,X_train,y_train,X_valid,y_valid,
                    X_test,y_test,pipeline_in,csv_out):

    # Predict
    y_train_full_pred = pipeline_in.predict(X_train_full)
    y_train_pred = pipeline_in.predict(X_train)
    y_valid_pred = pipeline_in.predict(X_valid)
    y_test_pred = pipeline_in.predict(X_test)

    # Set up lists
    y_list = [y_train_full,y_train,y_valid,y_test]
    y_pred_list = [y_train_full_pred,y_train_pred,y_valid_pred,y_test_pred]
    data_list = ["Training Set (Full)","Training Set","Validation Set",
                                                                    "Test Set"]

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

# Get data
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,
                                        housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                        y_train_full, random_state=42)

# Build a regression model with Multilayer Perceptron
from sklearn.neural_network import MLPRegressor
mlp_reg =  MLPRegressor(activation = "logistic",
            alpha = 5e-05,
            batch_size = "auto",
            beta_1 = 0.9,
            beta_2 = 0.999,
            early_stopping = False,
            epsilon = 1e-08,
            hidden_layer_sizes = (20,20,20,20,20,),
            learning_rate = "constant",
            learning_rate_init = 0.001,
            max_fun = 15000,
            max_iter = 100000,
            momentum = 0.9,
            n_iter_no_change = 30,
            nesterovs_momentum = True,
            power_t = 0.5,
            random_state = None,
            shuffle = True,
            solver = "adam",
            tol = 0.0001,
            validation_fraction = 0.1,
            verbose = False,
            warm_start = False)

# Build a regression model with Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(max_depth=8, random_state=0)

# Build a regression model with Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=100000, tol=1e-5)

################################################################################
msg_out = "\nRegression with Multilayer Perceptron"
print(msg_out,end="...")

# Pipeline Multilayer Perceptron Regression
pipeline_mlp = make_pipeline(StandardScaler(), mlp_reg)
pipeline_mlp.fit(X_train, y_train)

# Call write_metrics() function
csv_out = "metrics_MLPRegressor.csv"
y_test_pred_mlr = write_metrics(X_train_full,y_train_full,X_train,y_train,
                X_valid,y_valid,X_test,y_test,pipeline_mlp,csv_out)

# Call scatter_plot() function
plot_out = "scatter_plot_MLPRegressor.png"
dpi = 600
scatter_plot(y_test,y_test_pred_mlr,".","blue","black",plot_out,dpi)

print("done!")
################################################################################
msg_out = "\nRegression with Random Forest"
print(msg_out,end="...")

# Pipeline Random Forest Regression
pipeline_rf = make_pipeline(StandardScaler(), rf_reg)
pipeline_rf.fit(X_train, y_train)

# Call write_metrics() function
csv_out = "metrics_RandomForestRegressor.csv"
y_test_pred_rf = write_metrics(X_train_full,y_train_full,X_train,y_train,
                X_valid,y_valid,X_test,y_test,pipeline_rf,csv_out)

# Call scatter_plot() function
plot_out = "scatter_plot_RandomForestRegressor.png"
dpi = 600
scatter_plot(y_test,y_test_pred_rf,".","blue","black",plot_out,dpi)

print("done!")
################################################################################
msg_out = "\nRegression with Stochastic Gradient Descent"
print(msg_out,end="...")

# Pipeline for Stochastic Gradient Descent Regression
pipeline_sgd = make_pipeline(StandardScaler(), sgd_reg)
pipeline_sgd.fit(X_train, y_train)

# Call write_metrics() function
csv_out = "metrics_SGDRegressor.csv"
y_test_pred_sgd = write_metrics(X_train_full,y_train_full,X_train,y_train,
                X_valid,y_valid,X_test,y_test,pipeline_sgd,csv_out)

# Call scatter_plot() function
plot_out = "scatter_plot_SGDRegressor.png"
dpi = 600
scatter_plot(y_test,y_test_pred_sgd,".","blue","black",plot_out,dpi)

print("done!")
################################################################################
# Open a new file with predicted values
fo_new =  open("predictions_test_set.csv","w")
data_out = "y,MLPRegressor,RandomForestRegressor,SGDRegressor\n"
for count in range(len(y_test)):
    data_out += str(y_test[count])+","+str(y_test_pred_mlr[count])+","
    data_out += str(y_test_pred_rf[count])+","+str(y_test_pred_sgd[count])+"\n"
fo_new.write(data_out)
fo_new.close()