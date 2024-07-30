#!/usr/bin/env python3
#
# Visualization of data from CASF-2016 Ki dataset (de Azevedo et al., 2024).
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
# Import section
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

# Get data
feature_target_list = ["Gauss 1","Gauss 2","Repulsion","Hydrophobic","Hydrogen",
"Torsional","Torsions","C","N","O","S","P","Receptor B-factor(A2)","Average Q",
"Q","B-factor ratio (Ligand/Receptor)","pKi"]

extended_feature_target_list = ["Resolution(A)","Gauss 1","Gauss 2","Repulsion",
"Hydrophobic","Hydrogen","Torsional","Torsions","C","N","O","S","P","F","Cl",
"Ligand B-factor(A2)","Receptor B-factor(A2)","Average Q","Q",
"B-factor ratio (Ligand/Receptor)","pKi"]

file_in = "datasets/CASF-2016_Ki.csv"
df = pd.read_csv(file_in)
df_extended =  df[extended_feature_target_list]
df_feature_target =  df[feature_target_list]

# Show some information regading the data
data_info = df.info()

# Generate histogram
plt.rc('font', size=9)
plt.rc('axes', labelsize=7, titlesize=9)
plt.rc('legend', fontsize=9)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
df_extended.hist(bins=15, figsize=(12, 10))
plt.savefig("histogram_CASF-2016_Ki.png",dpi=600)
print("\nHistogram saved in the file: histogram_CASF-2016_Ki.png")

# Looking for Correlations
corr_matrix = df_feature_target.corr(numeric_only=True)
corr_matrix["pKi"].sort_values(ascending=False)
corr_matrix.to_csv("scatter_matrix.csv", sep=',', index=False, encoding='utf-8')
print("\nCorrelation matrix data saved in the file: scatter_matrix.csv")
attributes = ["pKi", "C", "Gauss 1","Gauss 2"]
scatter_matrix(df_feature_target[attributes], figsize=(12, 8))
plt.savefig("scatter_matrix_plot.png",dpi=600)
print("\nCorrelation matrix plot saved in the file: scatter_matrix_plot.png")
df.plot(kind="scatter", x="Gauss 2", y="pKi",
             alpha=0.1, grid=True)
plt.savefig("Gauss 2_vs_pKi_scatterplot.png",dpi=600)
msg_out = "\nScatter plot for Gauss 2 vs pKi saved in the file:"
msg_out += "Gauss 2_vs_pKi_scatterplot.png"
print(msg_out)