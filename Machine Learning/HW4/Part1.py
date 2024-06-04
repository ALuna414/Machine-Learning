#************************************************************************************
# Aaron Luna
# ML – HW#4
# Filename: Part1.py
# Due: Oct. 18, 2023
#
# Objective:
# Use k-means++ to observe clusters in the data using the LEAP cluster and determine 
# the number of centroids by using the Elbow Method (provide the plot) for the
# 2011 dataset first. Then apply this to the rest of the years in dataset.
# Use the correct number of centroids and plot the clusters with its centers and 
# silhouettes for each individual year. 
# Determine the distortion score and save it to a text file for each individual year
#************************************************************************************
#Importing all required libraries
print('\nImporting Packages........')
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator
print('\t\t\t\t........DONE!')

####################################################
#
# Data Preprocessing
#
####################################################
print('\nData Preprocessing........')
# Path set for each of the folders we want to extract from
GT_2011_file_path =  r"Datasets/gt_2011.csv"
GT_2012_file_path =  r"Datasets/gt_2012.csv"
GT_2013_file_path =  r"Datasets/gt_2013.csv"
GT_2014_file_path =  r"Datasets/gt_2014.csv"
GT_2015_file_path =  r"Datasets/gt_2015.csv"

# Read the data and combine into a new data dataframe
df_gt_2011 = pd.read_csv(GT_2011_file_path, header=None)
df_gt_2012 = pd.read_csv(GT_2012_file_path, header=None)
df_gt_2013 = pd.read_csv(GT_2013_file_path, header=None)
df_gt_2014 = pd.read_csv(GT_2014_file_path, header=None)
df_gt_2015 = pd.read_csv(GT_2015_file_path, header=None)

# Convert the specified rows to float
df_gt_2011[1:] = df_gt_2011[1:].astype(float)
df_gt_2012[1:] = df_gt_2012[1:].astype(float)
df_gt_2013[1:] = df_gt_2013[1:].astype(float)
df_gt_2014[1:] = df_gt_2014[1:].astype(float)
df_gt_2015[1:] = df_gt_2015[1:].astype(float)

#  Create a NumPy array
np_gt_2011 = np.array(df_gt_2011[1:])
np_gt_2012 = np.array(df_gt_2012[1:])
np_gt_2013 = np.array(df_gt_2013[1:])
np_gt_2014 = np.array(df_gt_2014[1:])
np_gt_2015 = np.array(df_gt_2015[1:]) 

# standardize the data
std = StandardScaler()
gt_2011_std = std.fit_transform(np_gt_2011)
gt_2012_std = std.fit_transform(np_gt_2012)
gt_2013_std = std.fit_transform(np_gt_2013)
gt_2014_std = std.fit_transform(np_gt_2014)
gt_2015_std = std.fit_transform(np_gt_2015)

# PCA
pca = PCA(n_components=2)
gt_2011_pca = pca.fit_transform(gt_2011_std)
gt_2012_pca = pca.fit_transform(gt_2012_std)
gt_2013_pca = pca.fit_transform(gt_2013_std)
gt_2014_pca = pca.fit_transform(gt_2014_std)
gt_2015_pca = pca.fit_transform(gt_2015_std)

print('\t\t\t\t........DONE!')
####################################################
# End of Data Preprocessing CODE
####################################################


####################################################
#
# Determining Centroids by Elbow Method
#
####################################################
print('\nElbow Method Processing........')

# 2011 data
distortions = []

# Loop from 1 to 10 clusters to determine the optimal number of clusters
for i in range(1, 11):
    km = KMeans(n_clusters=i,   # Number of clusters to try
    init='k-means++',           # Method to initialize cluster centers
    n_init=10,                  # Number of times KMeans will be run with different initializations
    max_iter=300,
    random_state=0)
    
    # Fit the KMeans model to the PCA-transformed data
    km.fit(gt_2011_pca)
    distortions.append(km.inertia_)

# Get the inertia score for the last fitted KMeans model
score = km.inertia_

# Use the KneeLocator to find the "elbow point" which indicates the optimal number of clusters
k_elbow = KneeLocator(x=range(1, 11), y=distortions, curve='convex', direction='decreasing')
print("The best number of clusters is", k_elbow.elbow)

# Store the optimal number of clusters for the 2011 data 
n_clusters_2011 = k_elbow.elbow

# 2012 data
distortions2 = []

# Loop from 1 to 10 clusters to determine the optimal number of clusters
for i in range(1, 11):
    km2 = KMeans(n_clusters=i,  # Number of clusters to try
    init='k-means++',           # Method to initialize cluster centers
    n_init=10,                  # Number of times KMeans will be run with different initializations
    max_iter=300,               # Maximum number of iterations for each single run
    random_state=0)

    # Fit the KMeans model to the PCA-transformed data
    km2.fit(gt_2012_pca)
    distortions2.append(km2.inertia_)

# Get the inertia score for the last fitted KMeans model
score2 = km2.inertia_

# Use the KneeLocator to find the "elbow point" which indicates the optimal number of clusters
k_elbow2 = KneeLocator(x=range(1, 11), y=distortions2, curve='convex', direction='decreasing')
print("The best number of clusters is", k_elbow2.elbow) 

# Store the optimal number of clusters for the 2012 data
n_clusters_2012 = k_elbow2.elbow

# 2013 data
distortions3 = []

# Loop from 1 to 10 clusters to determine the optimal number of clusters
for i in range(1, 11):
    km3 = KMeans(n_clusters=i,  # Number of clusters to try
    init='k-means++',           # Method to initialize cluster centers
    n_init=10,                  # Number of times KMeans will be run with different initializations
    max_iter=300,               # Maximum number of iterations for each single run
    random_state=0)

    # Fit the KMeans model to the PCA-transformed data
    km3.fit(gt_2013_pca)
    distortions3.append(km3.inertia_)

score3 = km3.inertia_

# Use the KneeLocator to find the "elbow point" which indicates the optimal number of clusters
k_elbow3 = KneeLocator(x=range(1, 11), y=distortions3, curve='convex', direction='decreasing')
print("The best number of clusters is", k_elbow3.elbow) 
n_clusters_2013 = k_elbow3.elbow

# 2014 data
distortions4 = []

# Loop from 1 to 10 clusters to determine the optimal number of clusters
for i in range(1, 11):
    km4 = KMeans(n_clusters=i,  # Number of clusters to try
    init='k-means++',           # Method to initialize cluster centers
    n_init=10,                  # Number of times KMeans will be run with different initializations
    max_iter=300,               # Maximum number of iterations for each single run
    random_state=0)

    km4.fit(gt_2014_pca)
    distortions4.append(km4.inertia_)

# Get the inertia score for the last fitted KMeans model
score4 = km4.inertia_

# Use the KneeLocator to find the "elbow point" which indicates the optimal number of clusters
k_elbow4 = KneeLocator(x=range(1, 11), y=distortions4, curve='convex', direction='decreasing')
print("The best number of clusters is", k_elbow4.elbow) 
n_clusters_2014 = k_elbow4.elbow

# 2015 data
distortions5 = []
for i in range(1, 11):
    km5 = KMeans(n_clusters=i,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=0)

    # Fit the KMeans model to the PCA-transformed data
    km5.fit(gt_2015_pca)
    distortions5.append(km5.inertia_)

# Get the inertia score for the last fitted KMeans model
score5 = km5.inertia_

k_elbow5 = KneeLocator(x=range(1, 11), y=distortions5, curve='convex', direction='decreasing')
print("The best number of clusters is", k_elbow5.elbow) 
n_clusters_2015 = k_elbow5.elbow
    
print('\t\t\t\t........DONE!')
####################################################
# End of Elbow Method Processing CODE
####################################################


####################################################
#
# Elbow Method Distortion Plotting
#
####################################################
print('\nDistortion Plotting........')
# 2011 data plot
plt.plot(range(1,len(distortions) + 1), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig("Part1_2011_Elbow_Method.png")
plt.close()

# 2012 data plot
plt.plot(range(1,len(distortions2) + 1), distortions2, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig("Part1_2012_Elbow_Method.png")
plt.close()

# 2013 data plot
plt.plot(range(1,len(distortions3) + 1), distortions3, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig("Part1_2013_Elbow_Method.png")
plt.close()

# 2014 data plot
plt.plot(range(1,len(distortions4) + 1), distortions4, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig("Part1_2014_Elbow_Method.png")
plt.close()

# 2015 data plot
plt.plot(range(1,len(distortions5) + 1), distortions5, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig("Part1_2015_Elbow_Method.png")
plt.close()

print('\t\t\t\t........DONE!')
####################################################
# End of Elbow Method Plotting CODE
####################################################


####################################################
#
# Cluster Plotting
#
####################################################
print('\nCluster Plotting........')
colors = ['green', 'blue', 'yellow', 'orange', 'black', 'purple', 'pink', 'brown', 'gray', 'cyan']
title = "Silhouette"

# 2011 data cluster
n_clusters = k_elbow.elbow  # Use the previously determined best number of clusters

km = KMeans(n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,            # Tolerance to declare convergence
            random_state=0)

# Fit the KMeans model to the PCA-transformed data and get cluster labels
y_km = km.fit_predict(gt_2011_pca)

for i in range(n_clusters):
    # Plot data points belonging to the current cluster
    plt.scatter(gt_2011_pca[y_km == i, 0],
                gt_2011_pca[y_km == i, 1],
                s=5,              # Size of the markers
                c=colors[i],
                marker='s',       # Shape of the markers (square)
                edgecolor='black',
                label='cluster ' + str(i + 1))

# Plot the cluster centroids
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=50, marker='*',
            c='red', edgecolor='black',
            label='centroids')

plt.legend(scatterpoints=1)
plt.title(title)
plt.grid()
plt.savefig("Part1_2011_Clusters.png")
plt.close()


# 2012 data cluster
n_clusters2 = k_elbow2.elbow    # Use the previously determined best number of clusters

km2 = KMeans(n_clusters=n_clusters2,
            init='k-means++',   # Method to initialize cluster centers
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

# Fit the KMeans model to the PCA-transformed data and get cluster labels
y_km2 = km2.fit_predict(gt_2012_pca)

for i in range(n_clusters2):
    # Plot data points belonging to the current cluster
    plt.scatter(gt_2012_pca[y_km2 == i, 0],
                gt_2012_pca[y_km2 == i, 1],
                s=5,
                c=colors[i],
                marker='s',     # Shape of the markers (square)
                edgecolor='black',
                label='cluster ' + str(i + 1))

# Plot the cluster centroids
plt.scatter(km2.cluster_centers_[:, 0],
            km2.cluster_centers_[:, 1],
            s=50, marker='*',
            c='red', edgecolor='black',
            label='centroids')

plt.legend(scatterpoints=1)
plt.title(title)
plt.grid()
plt.savefig("Part1_2012_Clusters.png")
plt.close()

# 2013 data cluster
n_clusters3 = k_elbow3.elbow     # Use the previously determined best number of clusters

km3 = KMeans(n_clusters=n_clusters3,
            init='k-means++',    # Method to initialize cluster centers
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km3 = km3.fit_predict(gt_2013_pca)

for i in range(n_clusters3):
    # Plot data points belonging to the current cluster
    plt.scatter(gt_2013_pca[y_km3 == i, 0],
                gt_2013_pca[y_km3 == i, 1],
                s=5,
                c=colors[i],
                marker='s',
                edgecolor='black', # Edge color of the markers
                label='cluster ' + str(i + 1))

# Plot the cluster centroids
plt.scatter(km3.cluster_centers_[:, 0],
            km3.cluster_centers_[:, 1],
            s=50, marker='*',
            c='red', edgecolor='black',
            label='centroids')

plt.legend(scatterpoints=1)
plt.title(title)
plt.grid()
plt.savefig("Part1_2013_Clusters.png")
plt.close()

# 2013 data cluster
n_clusters4 = k_elbow4.elbow       # Use the previously determined best number of clusters

km4 = KMeans(n_clusters=n_clusters4,
            init='k-means++',
            n_init=10,             # Number of times KMeans will be run with different initializations
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km4 = km4.fit_predict(gt_2014_pca)

for i in range(n_clusters4):
    # Plot data points belonging to the current cluster
    plt.scatter(gt_2014_pca[y_km4 == i, 0],
                gt_2014_pca[y_km4 == i, 1],
                s=5,
                c=colors[i],      # Color for the markers
                marker='s',
                edgecolor='black',
                label='cluster ' + str(i + 1))

# Plot the cluster centroids
plt.scatter(km4.cluster_centers_[:, 0],
            km4.cluster_centers_[:, 1],
            s=50, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.title(title)
plt.grid()
plt.savefig("Part1_2014_Clusters.png")
plt.close()

# 2015 data cluster
n_clusters5 = k_elbow5.elbow  # Use the previously determined optimal number of clusters

km5 = KMeans(n_clusters=n_clusters5,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km5 = km5.fit_predict(gt_2015_pca)

for i in range(n_clusters5):
    # Plot data points belonging to the current cluster
    plt.scatter(gt_2015_pca[y_km5 == i, 0],
                gt_2015_pca[y_km5 == i, 1],
                s=5,
                c=colors[i],
                marker='s',
                edgecolor='black',
                label='cluster ' + str(i + 1))

# Plot the cluster centroids
plt.scatter(km5.cluster_centers_[:, 0],
            km5.cluster_centers_[:, 1],
            s=50, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.title(title)
plt.grid()
plt.savefig("Part1_2015_Clusters.png")
plt.close()

print('\t\t\t\t........DONE!')
####################################################
# End of Clustering Plotting CODE
#################################################### 


####################################################
#
# Silhouette Plotting
#
####################################################
print('\nSilhouette Plotting........')
# 2011 data silhouette
# Initialize variables for silhouette plot
silhouette_avg = silhouette_score(gt_2011_pca, y_km)
sample_silhouette_values = silhouette_samples(gt_2011_pca, y_km)

y_lower = 10
fig, ax1 = plt.subplots()
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(gt_2011_pca) + (n_clusters_2011 + 1) * 10])

# Compute and plot the silhouette scores
for i in range(n_clusters_2011):
    ith_cluster_silhouette_values = sample_silhouette_values[y_km == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters_2011)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax1.set_title("Silhouette plot for 2011 data")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

# Save the silhouette plot
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.savefig("Part1_2011_Silhouette.png")
plt.close()

# 2012 data silhouette
# Initialize variables for silhouette plot
silhouette_avg2 = silhouette_score(gt_2012_pca, y_km2)
sample_silhouette_values2 = silhouette_samples(gt_2012_pca, y_km2)

y_lower2 = 10
fig2, ax2 = plt.subplots()
ax2.set_xlim([-0.1, 1])
ax2.set_ylim([0, len(gt_2012_pca) + (n_clusters_2012 + 1) * 10])

# Compute and plot the silhouette scores
for i in range(n_clusters_2012):
    ith_cluster_silhouette_values2 = sample_silhouette_values2[y_km2 == i]
    ith_cluster_silhouette_values2.sort()
    size_cluster_i2 = ith_cluster_silhouette_values2.shape[0]
    y_upper2 = y_lower2 + size_cluster_i2

    color2 = cm.nipy_spectral(float(i) / n_clusters_2012)
    ax2.fill_betweenx(np.arange(y_lower2, y_upper2), 0, ith_cluster_silhouette_values2, facecolor=color2, edgecolor=color2, alpha=0.7)

    ax2.text(-0.05, y_lower2 + 0.5 * size_cluster_i2, str(i))
    y_lower2 = y_upper2 + 10

ax2.set_title("Silhouette plot for 2012 data")
ax2.set_xlabel("Silhouette coefficient values")
ax2.set_ylabel("Cluster label")
ax2.axvline(x=silhouette_avg2, color="red", linestyle="--")

# Save the silhouette plot
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.savefig("Part1_2012_Silhouette.png")
plt.close()

# 2013 data silhouette
# Initialize variables for silhouette plot
silhouette_avg3 = silhouette_score(gt_2013_pca, y_km3)
sample_silhouette_values3 = silhouette_samples(gt_2013_pca, y_km3)

y_lower3 = 10
fig3, ax3 = plt.subplots()
ax3.set_xlim([-0.1, 1])
ax3.set_ylim([0, len(gt_2013_pca) + (n_clusters_2013 + 1) * 10])

# Compute and plot the silhouette scores
for i in range(n_clusters_2013):
    ith_cluster_silhouette_values3 = sample_silhouette_values3[y_km3 == i]
    ith_cluster_silhouette_values3.sort()
    size_cluster_i3 = ith_cluster_silhouette_values3.shape[0]
    y_upper3 = y_lower3 + size_cluster_i3

    color3 = cm.nipy_spectral(float(i) / n_clusters_2013)
    ax3.fill_betweenx(np.arange(y_lower3, y_upper3), 0, ith_cluster_silhouette_values3, facecolor=color3, edgecolor=color3, alpha=0.7)

    ax3.text(-0.05, y_lower3 + 0.5 * size_cluster_i3, str(i))
    y_lower3 = y_upper3 + 10

ax3.set_title("Silhouette plot for 2013 data")
ax3.set_xlabel("Silhouette coefficient values")
ax3.set_ylabel("Cluster label")
ax3.axvline(x=silhouette_avg3, color="red", linestyle="--")

# Save the silhouette plot
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.savefig("Part1_2013_Silhouette.png")
plt.close()

# 2014 data silhouette
# Initialize variables for silhouette plot
silhouette_avg4 = silhouette_score(gt_2014_pca, y_km4)
sample_silhouette_values4 = silhouette_samples(gt_2014_pca, y_km4)

y_lower4 = 10
fig4, ax4 = plt.subplots()
ax4.set_xlim([-0.1, 1])
ax4.set_ylim([0, len(gt_2014_pca) + (n_clusters_2014 + 1) * 10])

# Compute and plot the silhouette scores
for i in range(n_clusters_2014):
    ith_cluster_silhouette_values4 = sample_silhouette_values4[y_km4 == i]
    ith_cluster_silhouette_values4.sort()
    size_cluster_i4 = ith_cluster_silhouette_values4.shape[0]
    y_upper4 = y_lower4 + size_cluster_i4

    color4 = cm.nipy_spectral(float(i) / n_clusters_2014)
    ax4.fill_betweenx(np.arange(y_lower4, y_upper4), 0, ith_cluster_silhouette_values4, facecolor=color4, edgecolor=color4, alpha=0.7)

    ax4.text(-0.05, y_lower4 + 0.5 * size_cluster_i4, str(i))
    y_lower4 = y_upper4 + 10

ax4.set_title("Silhouette plot for 2014 data")
ax4.set_xlabel("Silhouette coefficient values")
ax4.set_ylabel("Cluster label")
ax4.axvline(x=silhouette_avg4, color="red", linestyle="--")

# Save the silhouette plot
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.savefig("Part1_2014_Silhouette.png")
plt.close()

# 2015 data silhouette
# Initialize variables for silhouette plot
silhouette_avg5 = silhouette_score(gt_2015_pca, y_km5)
sample_silhouette_values5 = silhouette_samples(gt_2015_pca, y_km5)

y_lower5 = 10
fig5, ax5 = plt.subplots()
ax5.set_xlim([-0.1, 1])
ax5.set_ylim([0, len(gt_2015_pca) + (n_clusters_2015 + 1) * 10])

# Compute and plot the silhouette scores
for i in range(n_clusters_2015):
    ith_cluster_silhouette_values5 = sample_silhouette_values5[y_km5 == i]
    ith_cluster_silhouette_values5.sort()
    size_cluster_i5 = ith_cluster_silhouette_values5.shape[0]
    y_upper5 = y_lower5 + size_cluster_i5

    color5 = cm.nipy_spectral(float(i) / n_clusters_2015)
    ax5.fill_betweenx(np.arange(y_lower5, y_upper5), 0, ith_cluster_silhouette_values5, facecolor=color5, edgecolor=color5, alpha=0.7)

    ax5.text(-0.05, y_lower5 + 0.5 * size_cluster_i5, str(i))
    y_lower5 = y_upper5 + 10

ax4.set_title("Silhouette plot for 2015 data")
ax4.set_xlabel("Silhouette coefficient values")
ax4.set_ylabel("Cluster label")
ax4.axvline(x=silhouette_avg5, color="red", linestyle="--")

# Save the silhouette plot
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.savefig("Part1_2015_Silhouette.png")
plt.close()

print('\t\t\t\t........DONE!')
####################################################
# End of Silhouette Plotting CODE
#################################################### 


####################################################
#
# Distortion Score
#
####################################################
print('\nDistortion Scores........')
file_path = 'Part1_Distortion_Scores.txt'

with open(file_path, 'w') as file:
    file.write('2011 Distortion Score: ' + str(score) + '\n')
    file.write('2012 Distortion Score: ' + str(score2) + '\n')
    file.write('2013 Distortion Score: ' + str(score3) + '\n')
    file.write('2014 Distortion Score: ' + str(score4) + '\n')
    file.write('2015 Distortion Score: ' + str(score5) + '\n')

print('\t\t\t\t........DONE!')
####################################################
# End of Distortion Score CODE
#################################################### 