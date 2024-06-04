#************************************************************************************
# Aaron Luna
# ML – HW#3
# Filename: part1.py
# Due: Oct 8, 2023
#
# Objective:
# Classify targets in data, validate it, and plot neccessary graphs
#*************************************************************************************
print('\nImporting Packages........')
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
print('\t\t\t\t........DONE!')

####################################################
#
# Data Preprocessing
#
####################################################

# Path set for each of the folders we want to extract from
Corridor_rm155_71_path =         r"Datasets/Measurements_Upload/Corridor_rm155_7.1"
Lab139_71_loc0011_path =         r"Datasets/Measurements_Upload/Lab139_7.1"
Main_Lobby71_loc0011_path =      r"Datasets/Measurements_Upload/Main_Lobby_7.1"
Sport_Hall_71_loc0011_path =     r"Datasets/Measurements_Upload/Sport_Hall_7.1"

# Create DataFrame
combined_data = pd.DataFrame()

# Loop through files in the corridor folder, read the data, and combine into a new data dataframe
for filename in os.listdir(Corridor_rm155_71_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Corridor_rm155_71_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_corridor= pd.concat([combined_data, data], ignore_index=True)

# New comnimed DataFrame given file name
combined_data_corridor.to_csv("Corridor.csv", index=False)

# Loop through files in the lab folder, read the data, and combine into a new data dataframe
for filename in os.listdir(Lab139_71_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_lab = pd.concat([combined_data, data], ignore_index=True)

# New comnimed DataFrame given file name
combined_data_lab.to_csv("Lab.csv", index=False)

# Loop through files in main lobby folder, read the data, and combine into a new data dataframe
for filename in os.listdir(Main_Lobby71_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_main = pd.concat([combined_data, data], ignore_index=True)

# New comnimed DataFrame given file name
combined_data_main.to_csv("Main.csv", index=False)

# Loop through files in the sports folder, read the data, and combine into a new data dataframe
for filename in os.listdir(Sport_Hall_71_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_sport = pd.concat([combined_data, data], ignore_index=True)

# New comnimed DataFrame given file name
combined_data_sport.to_csv("SportHall.csv", index=False)

# add the labels to the newly combined dataframes
combined_data_corridor['label'] = "Corridor"
combined_data_lab['label'] = "Lab"
combined_data_main['label'] = "Main"
combined_data_sport['label'] = "Sport"

# combine all data frames into one 
combined_data_frames = [combined_data_corridor, combined_data_lab, combined_data_main, combined_data_sport]
combined_data_all = pd.concat(combined_data_frames, ignore_index=True)
combined_data_all.to_csv("Combined_data_all.csv", index=False)

# drop the 5th collumn
combined_data_all = combined_data_all.drop(columns=[5])

# Handling missing values by dropping rows with missing values
combined_data_all.dropna(inplace=True)

# Split the data into features (X) and labels (y)
X = combined_data_all.drop(columns=['label'])
y = combined_data_all['label']

# Convert empty strings to NaN
X[X == ''] = np.nan

# Convert the entire array to float
X = X.astype(float)

#X = X.iloc[:, :-1]  # Remove the last column

le = LabelEncoder()
y_encoded = le.fit_transform(y) #From lecture notes 10

# standardize the training and test inputs
print('\nStandardizing the Data.......')
sc = StandardScaler()
X_std = sc.fit_transform(X)

norm = MinMaxScaler()
X_t_std = norm.fit_transform(X)
print('\t\t\t\t........DONE!')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=0)
X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=1)
####################################################
# End of Data Preprocessing CODE
####################################################


####################################################
#
# Pipeline code
#
####################################################
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))
pipe_lr.fit(X_train, y_train)
y_pred1 = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
pipe_lr = Pipeline([('standardscaler', StandardScaler()),('pca', PCA()),('logisticregression', LogisticRegression())])
param_grid = {
    'pca__n_components': [1, 2, 3],  # Number of components for PCA
    'logisticregression__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
}

pipe_lr2 = make_pipeline(StandardScaler(), LDA(n_components=3), LogisticRegression(random_state=1))
pipe_lr2.fit(X_train, y_train)
y_pred2 = pipe_lr2.predict(X_test)
print('Test Accuracy2: %.3f' % pipe_lr2.score(X_test, y_test))
pipe_lr2 = Pipeline([('standardscaler', StandardScaler()),('LDA', LDA()),('logisticregression', LogisticRegression())])
param_grid2 = {
    'LDA__n_components': [1, 2, 3],  # Number of components for PCA
    'logisticregression__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
}

pipe_lr3 = make_pipeline(StandardScaler(), KernelPCA(n_components=2, kernel='rbf', gamma=15), LogisticRegression(random_state=1))
pipe_lr3.fit(X_train, y_train)
y_pred3 = pipe_lr3.predict(X_test)
print('Test Accuracy3: %.3f' % pipe_lr3.score(X_test, y_test))
pipe_lr3 = Pipeline([('standardscaler', StandardScaler()),('KPCA', KernelPCA(kernel='rbf')),('logisticregression', LogisticRegression())])
param_grid3 = {
    'KPCA__n_components': [1, 2, 3],  # Number of components for PCA
    'logisticregression__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
}

pipe_lr4 = make_pipeline(MinMaxScaler(), PCA(n_components=2), LogisticRegression(random_state=1))
pipe_lr4.fit(X_train, y_train)
y_pred4 = pipe_lr4.predict(X_test)
print('Test Accuracy4: %.3f' % pipe_lr4.score(X_test, y_test))
pipe_lr4 = Pipeline([('standardscaler', StandardScaler()),('pca', PCA()),('logisticregression', LogisticRegression())])
param_grid4 = {
    'pca__n_components': [1, 2, 3],  # Number of components for PCA
    'logisticregression__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
}

pipe_lr5 = make_pipeline(MinMaxScaler(), LDA(n_components=1), LogisticRegression(random_state=1))
pipe_lr5.fit(X_train, y_train)
y_pred5 = pipe_lr5.predict(X_test)
print('Test Accuracy5: %.3f' % pipe_lr5.score(X_test, y_test))
pipe_lr5 = Pipeline([('standardscaler', StandardScaler()),('LDA', LDA()),('logisticregression', LogisticRegression())])
param_grid5 = {
    'LDA__n_components': [1, 2, 3],  # Number of components for PCA
    'logisticregression__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
}

pipe_lr6 = make_pipeline(MinMaxScaler(), KernelPCA(n_components=3, kernel='rbf', gamma=15), LogisticRegression(random_state=1))
pipe_lr6.fit(X_train, y_train)
y_pred6 = pipe_lr6.predict(X_test)
print('Test Accuracy6: %.3f' % pipe_lr6.score(X_test, y_test))
pipe_lr6 = Pipeline([('standardscaler', StandardScaler()),('KPCA', KernelPCA(kernel='rbf')),('logisticregression', LogisticRegression())])
param_grid6 = {
    'KPCA__n_components': [1, 2, 3],  # Number of components for PCA
    'logisticregression__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'logisticregression__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
}
####################################################
# End of pipeline CODE
####################################################


####################################################
#
# Gridsearch code
#
####################################################
gs1 = GridSearchCV(estimator=pipe_lr,param_grid=param_grid,scoring='accuracy',cv=10, n_jobs=-1)
gs1 = gs1.fit (X_train, y_train)

gs2 = GridSearchCV(estimator=pipe_lr2,param_grid=param_grid2,scoring='accuracy',cv=10, n_jobs=-1)
gs2 = gs2.fit (X_train, y_train)

gs3 = GridSearchCV(estimator=pipe_lr3,param_grid=param_grid3,scoring='accuracy',cv=10, n_jobs=-1)
gs3 = gs3.fit (X_train, y_train)

gs4 = GridSearchCV(estimator=pipe_lr4,param_grid=param_grid4,scoring='accuracy',cv=10, n_jobs=-1)
gs4 = gs4.fit (X_train, y_train)

gs5 = GridSearchCV(estimator=pipe_lr5,param_grid=param_grid5,scoring='accuracy',cv=10, n_jobs=-1)
gs5 = gs5.fit (X_train, y_train)

gs6 = GridSearchCV(estimator=pipe_lr6,param_grid=param_grid6,scoring='accuracy',cv=10, n_jobs=-1)
gs6 = gs6.fit (X_train, y_train)

# Define a list of grid search objects
grid_searches = [gs1, gs2, gs3, gs4, gs5, gs6]

# Initialize variables to store the best score and best index
best_score = -1
best_index = -1

# Iterate through the grid searches to find the best score and index
for index, gs in enumerate(grid_searches):
    if gs.best_score_ > best_score:
        best_score = gs.best_score_
        best_index = index

# Use the best index to set best_model and other related variables
best_gs = grid_searches[best_index]
best_model = best_gs.best_estimator_
best_params = best_gs.best_params_

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred5)
print(f"Accuracy: {accuracy:.2f}")

print("\Best Score: ")
print (best_gs.best_score_)
print("\nBest Params: ")
print (best_params.best_params_)
print("\nBest Estimator: ")
print(best_model.best_estimator_)
####################################################
# End of gridsearch CODE
####################################################


####################################################
#
# Best Results code
#
####################################################
# Predictions for best model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n\n\nBest Model Train Accuracy: {train_accuracy:.2f}")
print(f"Best Model Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}\n")
print(f"Recall: {recall:.2f}\n")
print(f"F1: {f1:.2f}\n")
print("Best Model:")
print(best_model)
print("Best Params:")
print(best_params)
print("\n\n\n")

# print to a txt file
with open('Part1_LR_results.txt', 'w') as file:
    file.write(f"Best Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision Score: {precision:.2f}")
    file.write(f"\nRecall Score: {recall:.2f}")
    file.write(f"\nF1 Score: {f1:.2f}\n")
    file.write("\nBest Model:\n")
    file.write(str(best_model))
    file.write("\nBest Params:\n")
    file.write(str(best_params))
####################################################
# End of best results CODE
####################################################


####################################################
#
# Plot Decision code
#
####################################################
lr = LogisticRegression(C=100.0, random_state=1)
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    print('\nCreating the Plot Decision figure.......')
    markers = ('s','x','o','D')
    colors = ('red', 'blue', 'lightgreen','orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #Plot all the samples
    X_test,y_test=X[test_idx,:],y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl, edgecolor='black'))

    #Highlight test samples
    if test_idx:
        X_test,y_test =X[test_idx,:],y[test_idx]
    
    plt.scatter(X_test[:,0],X_test[:,1],facecolors='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

    print('\t\t\t\t........DONE!')
    
    plot_decision_regions(X, y, classifier=lr)
    plt.title("{} Decision Regions".format(name))
    plt.legend()
    plt.savefig("Part1_DecisionRegion_results.png".format(name))
    plt.close()
####################################################
# End of PLOT DECISION CODE
####################################################


####################################################
#
# Learning Curve Code
#
####################################################
print('\nCreating the Learning Curve figure.......')
train_sizes, train_scores, test_scores = learning_curve(estimator=best_model,X=X_train,y=y_train,train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.grid()
plt.fill_between(train_sizes, train_mean + train_std, alpha=0.15, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, alpha=0.15, color='green')
plt.plot (train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.legend(loc='lower right')
plt.ylim([0.1, 1.0])
plt.savefig('Part1_LearningCurve_results.png')
plt.show
print('\t\t\t\t........DONE!')
####################################################
# End of pipeline CODE
####################################################


####################################################
#
# Matrix Train and Test outcome code
#
####################################################
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
#Matplotlib’s matshow
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
   for j in range(confmat.shape[1]):
      ax.text(x=j, y=i,
      s=confmat[i, j],
      va='center', ha='center')
plt.xlabel('predicted  test label')
plt.ylabel('true test label')
plt.savefig('Part1_CM_train_results.png')
plt.show() 
plt.close()

confmat2 = confusion_matrix(y_true=y_train, y_pred=y_pred_train)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
#Matplotlib’s matshow
ax.matshow(confmat2, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat2.shape[0]):
   for j in range(confmat2.shape[1]):
      ax.text(x=j, y=i,
      s=confmat2[i, j],
      va='center', ha='center')
plt.xlabel('predicted train label')
plt.ylabel('true train label')
plt.savefig('Part1_CM_test_results.png')
plt.show() 
plt.close()
####################################################
# End of matrix train CODE
####################################################


####################################################
#
# ROC outcome code
#
####################################################
fpr = dict()
tpr = dict()
roc_auc = dict()
y_pred_prob = best_model.predict_proba(X_test)
n_classes = len(np.unique(y_test))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(7, 5))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))
plt.plot([0, 1],[0, 1],linestyle='--',color=(0.6, 0.6, 0.6),label='random guessing')
plt.plot([0, 0, 1], [0, 1, 1],linestyle=':',color='black',label='perfect performance')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC')
plt.legend(loc="lower right")
plt.savefig('Part1_roc_auc_results.png')
plt.show()
plt.close()
####################################################
# End of ROC CODE
####################################################