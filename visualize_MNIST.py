# Visualization of data from the MNIST Dataset.
#
# https://atmamani.github.io/projects/ml/mnist-digits-classification-using-logistic-regression-scikit-learn/
#
# Reference:
# Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and
# TensorFlow. O'Reilly Media. Kindle Edition.
#
################################################################################
# Dr. Walter F. de Azevedo, Jr.                                                #
# https://github.com/azevedolab                                                #
# July 20, 2024                                                                #
################################################################################
#
# Import section
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Get data
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
X_train_in,y_train_in = X_train,y_train_5
X_test_in,y_test_in = X_test,y_test_5

# Show some information about this dataset
print("\nTargets: ",mnist.target)
print("Categories: ",mnist.categories)
print("Feature names: ",mnist.feature_names)
print("Details: ",mnist.details)
print("Description: ",mnist.DESCR)

# Preview digits
# For first half of the digits
print("\nGenerating plots to preview digits",end="...")
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(mnist.data[0:5],
                                           mnist.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.binary)
    plt.title('Digit: ' + label, fontsize = 20)

plt.savefig("mnist_preview_digits_1.png",dpi=1200)
plt.close()

# For the second half of the digits
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(mnist.data[5:10],
                                           mnist.target[5:10])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.binary)
    plt.title('Digit: ' + label, fontsize = 20)

plt.savefig("mnist_preview_digits_2.png",dpi=1200)
plt.close()

print("done!")
msg_out = "To preview digits open the following files: "
msg_out += "mnist_preview_digits_1.png and "
msg_out += "mnist_preview_digits_2.png"
print(msg_out)

# Generate bar graph for training set
print("\nGenerating bar graph (bar_graph_MNIST_training.png)",end="...")
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.xticks(unique)
plt.xlabel("Label")
plt.ylabel("Quantity")
plt.title("Labels in MNIST 784 dataset (training set)")
plt.savefig("bar_graph_MNIST_training.png",dpi=1200)
plt.close()
print("done!")

# Generate bar graph for test set
print("\nGenerating bar graph (bar_graph_MNIST_test.png)",end="...")
unique, counts = np.unique(y_test, return_counts=True)
plt.bar(unique, counts)
plt.xticks(unique)
plt.xlabel("Label")
plt.ylabel("Quantity")
plt.title("Labels in MNIST 784 dataset (test set)")
plt.savefig("bar_graph_MNIST_test.png",dpi=1200)
plt.close()
print("done!")

# Generate histogram for training set
print("\nGenerating histograms (histograms_MNIST_frequency.png)",end="...")
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(y_train);
plt.title('Frequency of different classes - Training data');

# Generate histogram for test set
plt.subplot(1,2,2)
plt.hist(y_test);
plt.title('Frequency of different classes - Test data');
plt.savefig("histograms_MNIST_frequency.png",dpi=1200)
plt.close()
print("done!")