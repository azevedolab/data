#!/usr/bin/env python3
#
# Classification Methods Applied to MNIST-784 Dataset
# This Python code creates statistical learning models using the following
# classifiers: multi-layer Perceptron ([MLPClassifier]
# (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)),
# random forest
# ([RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)),
# and stochastic gradient descent ([SGDClassifier]
# (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html))
# from [Scikit-Learn](https://scikit-learn.org/stable/index.html)
# [(Pedregosa et al., 2011)]
# (https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf).
# It builds models to identify number 5 on the [MNIST-784]
# (https://openml.org/search?type=data&status=active&id=554)
# (Modified National Institute of Standards and Technology) dataset.
# You will find part of this code discussed in the book Géron 2023. This code
# generates receiver operating characteristic (ROC) curves and calculates the
# area under the curve (AUC)
# ([Fawcett, 2006](https://doi.org/10.1016/j.patrec.2005.10.010)) for models
# built using the previously highlighted classifiers. We employ the metrics
# recommended by [Walsh et al., 2021]
# (https://doi.org/10.1038/s41592-021-01205-4)
# to analyze the predictive performance of the classifiers (e.g., precision,
# recall, and F1 score). This code relies on the following libraries:
# [Scikit-Learn](https://scikit-learn.org/stable/index.html)
# [(Pedregosa et al., 2011)]
# (https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf),
# [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and
# [Matplotlib](https://matplotlib.org/).
#
# References
#
# Fawcett, T. An introduction to ROC analysis. Pattern Recognit. Lett., 2006,
# 27, 861–874.   [DOI](https://doi.org/10.1016/j.patrec.2005.10.010)
#
# Géron, Aurélien. 2023. Hands-on Machine Learning with Scikit-Learn, Keras, and
# TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. 3rd
# ed. CA 95472: O’Reilly.
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

# Define write_metrics() function
def write_metrics(csv_out,method_list,y_in,y_pred_list,verbose):

    # Open a new csv file for training set
    fo_new = open(csv_out,"w")
    data_o = "Method,TN,FP,FN,TP,Accuracy,F1,Precision,Recall,ROC_AUC,"
    data_o += "Specificity"

    # Looping through method_list
    for i,method_in in enumerate(method_list):

        # Call Walsh_classification_metrics() function
        tn,fp,fn,tp,accuracy,f1,precision,recall,roc_auc,specificity = \
        Walsh_classification_metrics(method_in,y_in,y_pred_list[i],verbose)
        data_o += "\n"+method_in+","+str(tn)+","+str(fp)+","+str(fn)+","
        data_o += str(tp)+","+str(accuracy)+","+str(f1)+","+str(precision)+","
        data_o += str(recall)+","+str(roc_auc)+","+str(specificity)

    fo_new.write(data_o)
    fo_new.close()

# Define Walsh_classification_metrics() function
# Function to Calculate Metrics (metrics)
# This function focuses on evaluating the predictive performance of classifiers
# using metrics recommended by  [Walsh et al., 2021]
# (https://doi.org/10.1038/s41592-021-01205-4).
def Walsh_classification_metrics(method_in,y,y_pred,verbose):

    # Import section
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score,recall_score,f1_score,\
                                accuracy_score,roc_auc_score

    c_mat = confusion_matrix(y,y_pred)
    tn = c_mat[0][0]
    fp = c_mat[0][1]
    fn = c_mat[1][0]
    tp = c_mat[1][1]
    accuracy = round(accuracy_score(y,y_pred,normalize = True),6)
    f1 = round(f1_score(y,y_pred),6)
    precision = round(precision_score(y,y_pred),6)
    recall = round(recall_score(y,y_pred),6)
    roc_auc = round(roc_auc_score(y,y_pred),6)
    specificity = round(tn/(tn+fp),6)

    # Show Walsh metrics if requestes
    if verbose:
        print("\nMetrics for a model built using: ",method_in)
        print("TN: ",tn)
        print("FP: ",fp)
        print("FN: ",fn)
        print("TP: ",tp)
        print("Accuracy: {:.6}".format(accuracy))
        print("F1 Score: {:.6}".format(f1))
        print("Precision: {:.6}".format(precision))
        print("Recall (Sensitivity) : {:.6}".format(recall))
        print("ROC AUC: {:.6}".format(roc_auc))
        print("Specificity: {:.6}".format(specificity))

    # Return Walsh metrics
    return tn,fp,fn,tp,accuracy,f1,precision,recall,roc_auc,specificity

# Define ROC() class
# This class generates ROC curves
# ([Fawcett, 2006](https://doi.org/10.1016/j.patrec.2005.10.010)) and determines
# the metrics ([Walsh et al., 2021](https://doi.org/10.1038/s41592-021-01205-4))
# to evaluate the predictive performance of the classification models.
class ROC(object):

    # Define constructor method
    def __init__(self,roc_out,dpi,method_list,X_in,y_in,y_score_list):

        # Set up attributes
        self.roc_out = roc_out
        self.dpi = dpi
        self.method_list = method_list
        self.X_in = X_in
        self.y_in = y_in
        self.y_score_list = y_score_list

    # Define curves() method
    def curves(self):

        # Import section
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve,roc_curve

        # Plotting
        plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")

        # Looping through models to plot ROC and determine metrics
        for count, method_in in enumerate(self.method_list):

            # Get precisions,recalls,thresholds
            precisions,recalls,thresholds=precision_recall_curve(self.y_in,
                                                    self.y_score_list[count])

            # Determine fpr, tpr, and thresholds
            fpr,tpr,thresholds = roc_curve(self.y_in, self.y_score_list[count])

            # Plot curve a model
            plt.plot(fpr, tpr, linewidth=2, label=self.method_list[count])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        plt.savefig(self.roc_out,dpi=self.dpi)
        plt.close()

# MNIST Dataset
# Here, we download the
# [MNIST-784]
# (https://openml.org/search?type=data&status=active&id=554&sort=runs)
# dataset. Then, we create the target vectors for this classification task.
# We use True for all 5s and False for all other digits.
msg_out = "\nDownloading MNIST-784 dataset from openml.org"
print(msg_out,end="...")
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
print("done!")

# Preprocessing MNIST-784 data
msg_out = "\nPreprocessing MNIST-784 data"
print(msg_out,end="...")
from sklearn import preprocessing
X, y = mnist.data, mnist.target
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = (y_train == '5')
y_test = (y_test == '5')
print("done!")

# MLPClassifier
# This part of the code builds a classification model employing a multi-layer
# Perceptron classifier
# ([MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html))
# from [Scikit-Learn](https://scikit-learn.org/stable/index.html)
# [(Pedregosa et al., 2011)]
# (https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
msg_out = "\nClassification with Multilayer Perceptron"
print(msg_out,end="...")
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(20,20,20),activation="relu",
solver="adam",alpha=0.0001,learning_rate="constant",learning_rate_init=0.001,
validation_fraction=0.2,max_iter=100000,shuffle=True,random_state=46)

# Call cross_validation() function
mlp_clf_cv = cross_validation(mlp_clf,X_train,y_train,5,46,False)
y_probas_train_mlp = mlp_clf_cv.predict_proba(X_train)
y_scores_train_mlp = y_probas_train_mlp[:, 1]
y_probas_test_mlp = mlp_clf_cv.predict_proba(X_test)
y_scores_test_mlp = y_probas_test_mlp[:, 1]
y_train_pred_mlp = mlp_clf_cv.predict(X_train)
y_test_pred_mlp = mlp_clf_cv.predict(X_test)
print("done!")

# RandomForestClassifier
# Now, we use a random forest classifier
# ([RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))
# from [Scikit-Learn](https://scikit-learn.org/stable/index.html)
# [(Pedregosa et al., 2011)]
# (https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
# to build a classification model.
msg_out = "\nClassification with Random Forest"
print(msg_out,end="...")
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None,
warm_start = False, random_state=46, oob_score=False,criterion="gini",n_jobs=-1)

# Call cross_validation() function
rf_clf_cv = cross_validation(rf_clf,X_train,y_train,5,46,False)
y_probas_train_rf = rf_clf_cv.predict_proba(X_train)
y_scores_train_rf = y_probas_train_rf[:, 1]
y_probas_test_rf = rf_clf_cv.predict_proba(X_test)
y_scores_test_rf = y_probas_test_rf[:, 1]
y_train_pred_rf = rf_clf_cv.predict(X_train)
y_test_pred_rf = rf_clf_cv.predict(X_test)
print("done!")

# SGDClassifier
# In this part, we employ the stochastic gradient descent
# ([SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html))
# from [Scikit-Learn](https://scikit-learn.org/stable/index.html)
# [(Pedregosa et al., 2011)]
# (https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
# to make a classifier model.
msg_out = "\nClassification with Stochastic Gradient Descent"
print(msg_out,end="...")
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss='log_loss',penalty="l2",alpha=0.0001,
fit_intercept=True, max_iter=1000,tol=0.001, shuffle=True,verbose=0,
validation_fraction=0.2,random_state=46).fit(X_train, y_train)

# Call cross_validation() function
sgd_clf_cv = cross_validation(sgd_clf,X_train,y_train,5,46,False)
y_probas_train_sgd = sgd_clf_cv.predict_proba(X_train)
y_scores_train_sgd = y_probas_train_sgd[:, 1]
y_probas_test_sgd = sgd_clf_cv.predict_proba(X_test)
y_scores_test_sgd = y_probas_test_sgd[:, 1]
y_train_pred_sgd = sgd_clf_cv.predict(X_train)
y_test_pred_sgd = sgd_clf_cv.predict(X_test)
print("done!")

# ROC Curves
# In the final part of the code, we generate ROC curves
# ([Fawcett, 2006](https://doi.org/10.1016/j.patrec.2005.10.010))
# and evaluate the predictive performance
# ([Walsh et al., 2021](https://doi.org/10.1038/s41592-021-01205-4))
# of all classifiers.
msg_out = "\nGenerating ROC Curves"
print(msg_out,end="...")
method_list = ["MLPClassifier","RandomForestClassifier","SGDClassifier"]
y_score_train_list = [y_scores_train_mlp,y_scores_train_rf,y_scores_train_sgd]
clf_list = [mlp_clf_cv,rf_clf_cv,sgd_clf]
r_clf_train = ROC("ROC_training_set_mnist_784.png",600,method_list,X_train,
                    y_train,y_score_train_list)
r_clf_train.curves()
print("done!")

# Walsh Metrics for Classfication
msg_out = "\nCalculating Walsh classfication metrics"
print(msg_out,end="...")
y_train_pred_list = [y_train_pred_mlp,y_train_pred_rf,y_train_pred_sgd]
y_test_pred_list = [y_test_pred_mlp,y_test_pred_rf,y_test_pred_sgd]

# Call write_metrics() function for training set
write_metrics("Walsh_classification_metrics_training_set_mnist_784.csv",
                                method_list,y_train,y_train_pred_list,False)

# Call write_metrics() function for test set
write_metrics("Walsh_classification_metrics_test_set_mnist_784.csv",method_list,
                                    y_test,y_test_pred_list,False)

print("done!")