# Visualization of data from California Housing Dataset
#
# Reference:
# Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and
# TensorFlow. O'Reilly Media. Kindle Edition.
#
# Import section
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

# Get data for California Housing Prices
feature_target_list =["longitude","latitude","housing_median_age","total_rooms",
"total_bedrooms","population","households","median_income","median_house_value"]
file_in = "housing.csv"
df = pd.read_csv(file_in)
df_data = df[feature_target_list]
df_target = df["median_house_value"]

# Show some information regading the data
data_info = df.info()

# Generate histogram
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
df.hist(bins=50, figsize=(12, 8))
plt.savefig("histogram.png",dpi=600)
print("\nHistogram saved in the file: histogram.png")
# Show map
df.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.ylabel("Latitude (°)")
plt.xlabel("Longitude (°)")
plt.savefig("map_plot.png",dpi=600)
print("\nMap saved in the file: map_plot.png")
# Show color map
df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=df["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.ylabel("Latitude (°)")
plt.xlabel("Longitude (°)")
plt.savefig("color_map_plot.png",dpi=600)
print("\nColor map saved in the file: color_map_plot.png")
# Looking for Correlations
corr_matrix = df.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
corr_matrix.to_csv("scatter_matrix.csv", sep=',', index=False, encoding='utf-8')
print("\nCorrelation matrix data saved in the file: scatter_matrix.csv")
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(df[attributes], figsize=(12, 8))
plt.savefig("scatter_matrix_plot.png",dpi=600)
print("\nCorrelation matrix plot saved in the file: scatter_matrix_plot.png")
df.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.savefig("income_vs_house_value_scatterplot.png",dpi=600)
msg_out = "\nScatter plot for income vs house_value saved in the file:"
msg_out += "income_vs_house_value_scatterplot.png"
print(msg_out)