#************************************************************************************
# Aaron Luna
# 4331 HW2
#
# Objective: To show I can adjust and analyze data by using PCA/LDA, along with model
# types. Also by plotting those models and reporting results. 
#************************************************************************************
import pandas as pd
import os
import csv
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
####################################################
#
# Data Preprocessing
#
####################################################
# create empty list to store DataFrames and merge files together
combined_lab_list = []                               
directory = '/gpfs/home/a_l523/HW_2/Loc_0011_Lab'  # directory assignment 
for file_name in os.listdir(directory):                   
    f = os.path.join(directory, file_name)                 
if f.endswith('.csv'):                               
    with open (f,'r') as csv_reader:
        csv_reader = csv.reader(csv_reader, delimiter = ';')    # delimiter splits data 

        for index, line in enumerate(csv_reader):               # for loop to read files
            if index in [0, 1]:
                pass                                            
            elif index ==2:                                     
                lab = pd.DataFrame(columns = ['Freq','Real_S11','Imag_S11','Real_S21','Imag_S21','Room'])
            else:
                lab.loc[len(lab.index)] = line   
    combined_lab_list.append(lab)                 # append DataFrame

# DataFrames lists into single DataFrame
lab = pd.concat(combined_lab_list, ignore_index=True)

lab['Room'] = 1

missing_data = lab.isnull().sum()                        

# create empty list to store DataFrames and merge files together round 2
combined_sport_list = []
directory = '/gpfs/home/a_l523/HW_2/Loc_0011_Spo'     # directory assignment
for file_name in os.listdir(directory):                    
    f = os.path.join(directory, file_name)                 
if f.endswith('.csv'):                                
    with open (f,'r') as csv_reader:
        csv_reader = csv.reader(csv_reader, delimiter = ';')    # delimiter splits data 

        for index, line in enumerate(csv_reader):               # for loop to read
            if index in [0, 1]:
                pass                                            # skips two rows
            elif index ==2:                                     
                sport = pd.DataFrame(columns = ['Freq','Real_S11','Imag_S11','Real_S21','Imag_S21','Room'])
            else:
                sport.loc[len(sport.index)] = line    
    combined_sport_list.append(sport)                 # append DataFrame 

# DataFrames lists into single DataFrame
sport = pd.concat(combined_sport_list, ignore_index=True)

sport['Room'] = 2                                    # room number for sport room

missing_data = sport.isnull().sum()                        

# create empty list to store DataFrames and merge files together round 3
combined_main_list = []
directory = '/gpfs/home/a_l523/HW_2/Loc_0011_Main'      # directory assignment
for file_name in os.listdir(directory):                    # loop to read directory
    f = os.path.join(directory, file_name)                 
if f.endswith('.csv'):                                
    with open (f,'r') as csv_reader:
        csv_reader = csv.reader(csv_reader, delimiter = ';')    # delimiter splits data d

        for index, line in enumerate(csv_reader):               
            if index in [0, 1]:
                pass                                            # skips first two rows
            elif index ==2:                                     
                main = pd.DataFrame(columns = ['Freq','Real_S11','Imag_S11','Real_S21','Imag_S21','Room']) # column names given
            else:
                main.loc[len(main.index)] = line    
    combined_main_list.append(main)                 # appends DataFrame 

# DataFrames lists into single DataFrame
main = pd.concat(combined_main_list, ignore_index=True)

main['Room'] = 3                                # room number for main room

missing_data = main.isnull().sum()                  # check for missing data

# create empty list to store DataFrames and merge files together round 4
combined_corridor_list = []
directory = '/gpfs/home/a_l523/HW_2/Loc_0011_Cor'  
for file_name in os.listdir(directory):                    
    f = os.path.join(directory, file_name)                
if f.endswith('.csv'):                                # reads csv files
    with open (f,'r') as csv_reader:
        csv_reader = csv.reader(csv_reader, delimiter = ';')    # delimiter splits data 

        for index, line in enumerate(csv_reader):               
            if index in [0, 1]:
                pass                                            
            elif index ==2:                                     
                corridor = pd.DataFrame(columns = ['Freq','Real_S11','Imag_S11','Real_S21','Imag_S21','Room'])
            else:
                corridor.loc[len(corridor.index)] = line    
    combined_corridor_list.append(corridor)                 # append DataFrame 

# DataFrames lists into single DataFrame
corridor = pd.concat(combined_corridor_list, ignore_index=True)

corridor['Room'] = 4                     # room number for corridor room

missing_data = corridor.isnull().sum()                          

# combine for one master csv file
combined_df = pd.concat([lab, sport, main, corridor], ignore_index=True)
missing_combined_data = combined_df.isnull().sum()
print(missing_combined_data)   #check for missing data 

combined_df = combined_df.astype(float)    # float data type conversion

combined_df.to_csv('Master.csv', index = False)

####################################################
#
# Data Preprocessing
#
####################################################
#preprocess data
X = combined_df.drop('Room', axis = 1)  
y = combined_df['Room']                       

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# standardize data for svm model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)           # standardize training features
X_test_scaled = scaler.transform(X_test)                 # standardize testing features

# LDA 
lda = LDA(n_components=2)  # number of components
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

lda_coefficients = lda.coef_

max_weight_feature_index_ld1 = abs(lda_coefficients[0]).argmax()
max_weight_feature_name_ld1 = X.columns[max_weight_feature_index_ld1]

max_weight_feature_index_ld2 = abs(lda_coefficients[1]).argmax()
max_weight_feature_name_ld2 = X.columns[max_weight_feature_index_ld2]

####################################################
#
# Metric Printouts and Text File writes
#
###################################################
# text file output
output_text = f"The highest weight feature in LD1 is: {max_weight_feature_name_ld1}\n"
output_text += f"The highest weight feature in LD2 is: {max_weight_feature_name_ld2}\n"

# file path for the text file
file_path = 'model_results_txt'

# write text to the file
with open(file_path, 'w') as f:
    f.write(output_text)

# The kNN Model
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

#Testing the model data
y_pred = knn.predict(X_test)

# calculate accuracy                   
accuracy = accuracy_score(y_test, y_pred)

Name of the text file
with open("kNN_accuracies.txt", "w") as f:
f.write('Misclassified samples: %d' % (y_test != y_pred).sum())
f.write('Training Accuracy: %.2f' % knn.score(X_train, y_train))
f.write('Test Accuracy: %.2f' % knn.score(X_test, y_test))

print (combined_df)

# Combine all training and test data to single object variables
X_combined=np.vstack((X_train,X_test))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined, y=y_combined, classifier=knn,
test_idx=range(y_train.size, y_train.size + y_test.size))

#plot with labels and legend
plt.xlabel('Feature 2 [Standardized]')
plt.ylabel('Feature 1 [standardized]')
plt.legend(loc='upper left')
#Save plot ---> LEAP requires this part
plt.savefig("kNN_plot.png")


####################################################
#
# Plot setup and save image
#
####################################################
# color key
colorbar = plt.colorbar()
colorbar.set_label('Room')

plt.title('SVM Decision Boundary')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.savefig('svm_plot.png')

# perform t-SNE
tsne = TSNE(n_components=2, random_state=2)
X_train_tsne = tsne.fit_transform(X_train_lda)
X_test_tsne = tsne.fit_transform(X_test_lda)

# Plot t-SNE results
plt.figure(figsize=(12, 6))
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', s=80, edgecolors='k')
colorbar = plt.colorbar()
colorbar.set_label('Room')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('tsne_plot.png')
plt.close()

# UMAP
umap_model = umap.UMAP(n_components=2,)
X_train_umap = umap_model.fit_transform(X_train_lda)
X_test_umap = umap_model.transform(X_test_lda)

# Plot UMAP results
plt.figure(figsize=(12, 6))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', s=80, edgecolors='k')
colorbar = plt.colorbar()
colorbar.set_label('Room')
plt.title('UMAP Visualization')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.savefig('umap_plot.png')
plt.close()

