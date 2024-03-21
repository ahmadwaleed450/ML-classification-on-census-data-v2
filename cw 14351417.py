#!/usr/bin/env python
# coding: utf-8

# ## Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

# In[2]:


# Install package containing dataset -
# URL for dataset https://archive.ics.uci.edu/dataset/2/adult
# To obtain dataset uncomment and run the line below
# pip install ucimlrep

# Newer version of imbalanced-learn not compatible with python3.7
# pip install imbalanced-learn==0.7.0  
# pip install category_encodersimport pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import category_encoders as ce 
import seaborn as sns

# Setting seed to last 3 digits of my student ID as random seed for reproducibility
seed_value = 417


# In[5]:


# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
#print(adult.metadata) 
  
# variable information 
# print(adult.variables) 
raw_df = X
raw_df['income'] = y['income']


# ## Pre processing

# In[6]:


# Data exploration
df = raw_df.copy()
for col in raw_df.columns:
    unique_values = raw_df[col].unique()
    print(f"Unique values in {col} \n: {unique_values}, \n")


# ### Replace ? with NAN in few columns and remove "." from target variable causing fake more classes

# In[7]:


# Target variable 'income' has 4 unique elements (<=50K  <=50K. >50K >50K.) when actually its just 2
df['income'] = df['income'].str.replace('.', '', regex=False)

# Replace ? with None from the few columns it exists in.
df['workclass'] = df['workclass'].replace('?', None, regex=False)
df['native-country'] = df['native-country'].replace('?', None, regex=False)
df['occupation'] = df['occupation'].replace('?', None, regex=False)


# ## Handling missing values

# ### Listwise deletion (row-deletion) to handle missing values

# In[8]:


# Listwise deletion
pre_size = len(df)
df.dropna(inplace=True)
print (pre_size - len(df), 'rows deleted because of missing values') 


# In[9]:


# Mapping the columns with binary values to 0 and 1 
sex_mapping = {'Male': 0, 'Female': 1}
income_mapping = {'<=50K': 0, '>50K': 1}
df['sex'] = df['sex'].map(sex_mapping)
df['income'] = df['income'].map(income_mapping)


# In[ ]:


Column	Original Values	Mapped Values
sex	'Male', 'Female'	0 (Male), 1 (Female)
income	'<=50K', '>50K'	0 (<=50K), 1 (>50K)

Mapping Binary Columns:
Column	Original Values	Mapped Values
sex	'Male', 'Female'	0 (Male), 1 (Female)
income	'<=50K', '>50K'	0 (<=50K), 1 (>50K)
The binary columns 'sex' and 'income' were mapped to 0 and 1, respectively, for consistency and ease of analysis.


# 
# ## encode the categorical columns with label encoding, one hot, binary encoding

# ## in paper talk about multi collinearity and why dropping first column in one hot coding is necessarry

# In[10]:


categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

# One-Hot Encoding
df_one_hot_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Binary Encoding
binary_encoder = ce.BinaryEncoder(cols=categorical_cols)
df_binary_encoded = binary_encoder.fit_transform(df)

label_encoder = LabelEncoder()
df_label_encoded = df.copy()
for col in categorical_cols:
    df_label_encoded[col] = label_encoder.fit_transform(df[col])


encoded_dfs = {'binary': df_binary_encoded,'one-hot': df_one_hot_encoded, 'label': df_label_encoded}


# # Sample balancing
# ## Under Sampling, Over sampling, SMOTE 

# In[45]:


# function for resampling
def resample_data(df, sampler):
    X, y = sampler.fit_resample(df.drop('income', axis=1), df['income'])
    unshuffled = pd.concat([pd.DataFrame(X, columns=df.drop('income', axis=1).columns), 
                      pd.DataFrame(y, columns=['income'])], axis=1)
    return unshuffled.sample(frac=1, random_state=seed_value).reset_index(drop=True)

# Dictionary to store all data frames for each encoding
results_dfs = {}

for name, df in encoded_dfs.items():
    print(f"Unbalanced ({name}):", df['income'].value_counts(), sep='\n')
    
    # Dictionary to store data frames for each resampling method
    resampled_results = {}
    
    # Undersampling
    df_undersampled = resample_data(df, RandomUnderSampler(sampling_strategy='majority', random_state=seed_value))
    print(f"Undersampled ({name}):", df_undersampled['income'].value_counts(), df_undersampled.shape, sep='\n')
    resampled_results['under_sampled'] = df_undersampled
    
    
    # Oversampling
    df_oversampled = resample_data(df, RandomOverSampler(sampling_strategy='minority', random_state=seed_value))
    print(f"Oversampled ({name}):", df_oversampled['income'].value_counts(),df_oversampled.shape, sep='\n' )
    
    resampled_results['over_sampled'] = df_oversampled
    
    # SMOTE
    smote = SMOTE(random_state=seed_value)
    X_resampled, y_resampled = smote.fit_resample(df.drop('income', axis=1), df['income'])
    df_smote = pd.concat([pd.DataFrame(X_resampled, columns=df.drop('income', axis=1).columns), 
                          pd.DataFrame(y_resampled, columns=['income'])], axis=1).sample(frac=1, random_state=seed_value).reset_index(drop=True)
    resampled_results['smote'] = df_smote
    print(f"Smote ({name}):", df_smote['income'].value_counts(),df_smote.shape, sep='\n')
    
    
    # Save results for this encoding
    results_dfs[name] = resampled_results


# ### split dataframes in training and test for all datasets

# In[12]:


# Dictionary to store train and test sets for each encoding
dfs = {}

# Loop through each encoding
for encoding, resampled_results in results_dfs.items():
    train_test_results = {}
    
    # Loop through each resampled result
    for resampling_method, df_resampled in resampled_results.items():
        
        # Splitting data into train and test sets
        X = df_resampled.drop('income', axis=1)
        y = df_resampled['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)
        
        # Scaling the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        train_test_results[resampling_method] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
    
    # Storing train and test sets for this encoding
    dfs[encoding] = train_test_results

# Accessing the train and test sets for binary encoding:
# dfs['binary']['under_sampled']['X_train'], dfs['binary']['under_sampled']['y_train']


# In[48]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Dictionary to store PCA-transformed train and test data for each encoding
dfs_pca = {}

# Loop through each encoding
for encoding, train_test_results in dfs.items():
    pca_results = {}
    print(f'Encoding method : {encoding}')
     
    # Loop through each resampled result
    for resampling_method, data in train_test_results.items():
        print(f'Resampling method : {resampling_method}')
        
        X_train = data['X_train']
        X_test = data['X_test']
        
        # Initialize and fit PCA
        # one hot = 80 selected components, binary 30 selected components in labelling just 13 
        n_components = {'one-hot': 80, 'binary': 30, 'label': 13}[encoding]

        # Initialize and fit PCA for training data
        print('PCA started')
        # pca_train = PCA()
        pca_train = PCA(n_components=n_components)
        X_train_pca = pca_train.fit_transform(X_train)
        
        # Transform test data using PCA fitted on training data
        X_test_pca = pca_train.transform(X_test)
        
        # Store PCA results
        pca_results[resampling_method] = {
            'X_train_pca': X_train_pca,
            'X_test_pca': X_test_pca,
            'explained_variance_ratio_pca': pca_train.explained_variance_ratio_,
            'cumulative_explained_variance_pca': np.cumsum(pca_train.explained_variance_ratio_),
        }
       
        print(" Cummulative sum of ", n_components ,": variables", sum(pca_train.explained_variance_ratio_))
        

    # Store PCA-transformed train and test data for this encoding
    dfs_pca[encoding] = pca_results

# Accessing PCA-transformed train and test data for binary encoding, for example:
# dfs_pca['binary']['under_sampled']['X_train_pca'], dfs_pca['binary']['under_sampled']['X_test_pca']


# In[51]:


# Dictionary to store cumulative explained variance for each resampling method
cumulative_explained_variance = {}

# Loop through each encoding
for encoding, pca_results in dfs_pca.items():
    cumulative_explained_variance[encoding] = {}
    
    # Initialize plots for cumulative explained variance
    plt.figure(figsize=(10, 5))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title(f'Cumulative Variance Explained by PCA Components - {encoding}')
    plt.grid(True)
    
    # Loop through each resampling method
    for resampling_method, data in pca_results.items():
        cumulative_var = data['cumulative_explained_variance_pca']
        plt.plot(np.arange(1, len(cumulative_var) + 1), cumulative_var, label=resampling_method)
    
    # Add legend and show plot for each encoding
    plt.legend(title='Resampling Method')
    plt.tight_layout()
    plt.show()


# ## Classification methods

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
import os
import pickle



def get_classifier(classifier_name):
  if classifier_name == 'SVM':
    return SVC(kernel='linear', random_state=seed_value, probability=True)
  elif classifier_name =='KNN':
    return KNeighborsClassifier()
  elif classifier_name == 'RandomForest': 
    return RandomForestClassifier(random_state=seed_value)
  elif classifier_name == 'AdaBoost':
    return AdaBoostClassifier(random_state=seed_value)
  elif classifier_name == 'ExtraTrees':
    return ExtraTreesClassifier(random_state=seed_value)
  else:
    raise ValueError(str(classifier_name) + 'not recognised')


# Function to save data to pickle file
def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Loop through each classifier
for classifier_name in ['SVM', 'KNN', 'RandomForest', 'AdaBoost', 'ExtraTrees' ]:

    print('Classifier: ', classifier_name)
    
    if not os.path.exists(classifier_name):  # Check if directory exists for classifier
        os.makedirs(classifier_name)
    
    # Loop through each encoding
    for encoding, pca_results in dfs_pca.items():
        print('Encoding', encoding)
        classifier_model = get_classifier(classifier_name)
        
        if not os.path.exists(os.path.join(classifier_name, encoding)):  # Check if directory exists for encoding
            os.makedirs(os.path.join(classifier_name, encoding))
        
        # Loop through each resampled result
        for resampling_method, data in pca_results.items():
            filename = os.path.join(classifier_name, encoding, f'{resampling_method}.pkl')
            
            if os.path.exists(filename):  # Skip if file already exists
                print(f'Skipping {filename} as it already exists.')
                continue
            else:
              print('Sampling method', resampling_method)
            X_train_pca = data['X_train_pca']
            X_test_pca = data['X_test_pca']
            y_train = train_test_results[resampling_method]['y_train']
            y_test = train_test_results[resampling_method]['y_test']
            
            # Initialize and fit classifier model
            classifier_model.fit(X_train_pca, y_train)
            
            # Predict on test data
            y_pred = classifier_model.predict(X_test_pca)
            
            # Calculate accuracy measures
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store performance measures
            performance_measures = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
                      
            # Save all data to pickle file
            data_to_save = {
                'title': ', '.join([classifier_name, encoding, resampling_method]),
                'model': classifier_model,
                'performance_measures': performance_measures,
                'confusion_matrices': cm,
#                'roc_curves': roc_curves
            }
            save_data_to_pickle(data_to_save, filename)


# In[15]:


import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from pickle file
def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, title):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

# Directory containing pickle files
directory = './'

# Loop through each file in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.pkl'):  # Check if file is a pickle file
            filename = os.path.join(root, file)
            print(f'Reading file: {filename}')
            
            # Load data from pickle file
            data = load_data_from_pickle(filename)
            print (data)
            
            # Extract classifier, encoding, and performance measures
            title = data['title']
            classifier_name, encoding, resampling_method = title.split(', ')
            performance_measures = data['performance_measures']
            cm = data['confusion_matrices']
            model = data['model']
            # fpr, tpr, roc_auc = data['roc_curves']
            
            # Print performance measures
            print(f'Classifier: {classifier_name}, Encoding: {encoding}, Resampling Method: {resampling_method}')
            for metric, value in performance_measures.items():
                print(f'{metric}: {value}')
            
            # Plot confusion matrix
            # plot_confusion_matrix(cm, f'Confusion Matrix - {classifier_name} ({encoding} - {resampling_method})')
            


# In[ ]:


# List to store performance measures
performance_list = []

# Loop through each file in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.pkl'):  # Check if file is a pickle file
            filename = os.path.join(root, file)
            
            # Load data from pickle file
            data = load_data_from_pickle(filename)
            
            # Extract classifier, encoding, and performance measures
            title = data['title']
            classifier_name, encoding, resampling_method = title.split(', ')
            performance_measures = data['performance_measures']
            confusion_matrix = data['confusion_matrices']
            accuracy = performance_measures['accuracy']
            precision = performance_measures['precision']
            recall = performance_measures['recall']
            f1_score = performance_measures['f1_score']
            
            # Append performance measures to list
            performance_list.append({
                'Classifier': classifier_name,
                'Encoding': encoding,
                'Resampling Method': resampling_method,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'Confusion Matrix': confusion_matrix
            })

# Sort performance list based on best accuracy and then on precision
performance_list_sorted = sorted(performance_list, key=lambda x: (x['Accuracy'], x['F1 Score']), reverse=True)
performance_df = pd.DataFrame(performance_list_sorted)


# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Plot comparison of accuracy and F1 score for different encoding methods
plt.figure(figsize=(12, 6))

# Group by encoding, sampling method, and classifier
grouped_df = performance_df.groupby(['Encoding', 'Resampling Method', 'Classifier'])

# Calculate mean performance measures for each group
mean_performance = grouped_df.mean().reset_index()

# Mean accuracy by encoding method
plt.subplot(1, 2, 1)
sns.barplot(data=mean_performance, x='Encoding', y='Accuracy', hue='Resampling Method', palette='Set1')
plt.title('Mean Accuracy by Encoding Method')
plt.xlabel('Encoding Method')
plt.ylabel('Mean Accuracy')
plt.grid(axis='y')
plt.yticks(np.arange(0, 1.1, 0.1))
sns.despine()

# Mean F1 score by encoding method
plt.subplot(1, 2, 2)
sns.barplot(data=mean_performance, x='Encoding', y='F1 Score', hue='Resampling Method', palette='Set2')
plt.title('Mean F1 Score by Encoding Method')
plt.xlabel('Encoding Method')
plt.ylabel('Mean F1 Score')
plt.grid(axis='y')
plt.yticks(np.arange(0, 1.1, 0.1))
sns.despine()

plt.tight_layout()
plt.show()
performance_df.sort_values(by=['Accuracy', 'F1 Score'], ascending=[False, False]).reset_index().drop('index', axis=1)


# In[21]:


df_binary_smote = dfs_pca['binary']['smote']
df_binary_smote['y_train'], df_binary_smote['y_test'] = dfs['binary']['smote']['y_train'], dfs['binary']['smote']['y_test'] 
df_label_oversampled = dfs_pca['label']['over_sampled']
df_label_oversampled['y_train'], df_label_oversampled['y_test'] = dfs['label']['over_sampled']['y_train'], dfs['label']['over_sampled']['y_test'] 


# # Extra Trees

# In[33]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the Extra Trees classifier
et_classifier = ExtraTreesClassifier(random_state=seed_value)

# Define the hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(estimator=et_classifier, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(df_label_oversampled['X_train_pca'], df_label_oversampled['y_train'])

# Get the best parameters and best accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Extract the results of grid search
results = pd.DataFrame(grid_search.cv_results_)


# In[ ]:


# Convert cv_results_ to DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Select columns of interest
columns_of_interest = ['rank_test_score', 'param_n_estimators', 'param_max_depth', 'param_min_samples_split',
                       'param_min_samples_leaf', 'param_max_features', 'mean_test_score', ]
results_df = results_df[columns_of_interest]


# # Extra Trees

# In[34]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

# Best parameters obtained from grid search
best_params = {
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'auto'
}


# Initialize and train the Extra Trees Classifier with the best parameters
model = ExtraTreesClassifier(**best_params)
model.fit(df_label_oversampled['X_train_pca'], df_label_oversampled['y_train'])

# Make predictions on the test data
y_pred = model.predict(df_label_oversampled['X_test_pca'])

# Calculate probabilities for ROC curve
y_proba = model.predict_proba(df_label_oversampled['X_test_pca'])[:, 1]

# Evaluate the model's performance
conf_matrix = confusion_matrix(df_label_oversampled['y_test'], y_pred)
roc_auc = roc_auc_score(df_label_oversampled['y_test'], y_proba)
accuracy = accuracy_score(df_label_oversampled['y_test'], y_pred)
precision = precision_score(df_label_oversampled['y_test'], y_pred)
recall = recall_score(df_label_oversampled['y_test'], y_pred)
f1 = f1_score(df_label_oversampled['y_test'], y_pred)

# Print performance measures
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(df_label_oversampled['y_test'], y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# # SVM

# In[35]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the SVM classifier
svm_classifier = SVC(random_state=seed_value)

# Define the hyperparameters to search
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Perform grid search with 10-fold cross-validation
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(df_binary_smote['X_train_pca'], df_binary_smote['y_train'])

# Get the best parameters and best accuracy
best_params_svm = grid_search_svm.best_params_
best_accuracy_svm = grid_search_svm.best_score_

print("Best Parameters for SVM Classifier:", best_params_svm)
print("Best Accuracy for SVM Classifier:", best_accuracy_svm)


# ### Best Parameters for SVM Classifier: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}. Best Accuracy for SVM Classifier: 0.8862769390549351

# In[37]:


# Took backup as it is costly operation: save_data_to_pickle(grid_search_svm, 'grid_search_svm.pkl')


# ### Training model on best parameters,  and testing, and creating graphs

# In[38]:


from sklearn.svm import SVC

# Best parameters obtained from grid search
best_params_svm = {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}

# Initialize and train the SVM model with the best parameters
svm_model = SVC(**best_params_svm, probability=True)
svm_model.fit(df_binary_smote['X_train_pca'], df_binary_smote['y_train'])

# Make predictions on the test data
y_pred_svm = svm_model.predict(df_binary_smote['X_test_pca'])

# Calculate probabilities for ROC curve
y_proba_svm = svm_model.predict_proba(df_binary_smote['X_test_pca'])[:, 1]

# Evaluate the model's performance
conf_matrix_svm = confusion_matrix(df_binary_smote['y_test'], y_pred_svm)
roc_auc_svm = roc_auc_score(df_binary_smote['y_test'], y_proba_svm)
accuracy_svm = accuracy_score(df_binary_smote['y_test'], y_pred_svm)
precision_svm = precision_score(df_binary_smote['y_test'], y_pred_svm)
recall_svm = recall_score(df_binary_smote['y_test'], y_pred_svm)
f1_svm = f1_score(df_binary_smote['y_test'], y_pred_svm)

# Print performance measures
print(f'Accuracy: {accuracy_svm:.4f}')
print(f'Precision: {precision_svm:.4f}')
print(f'Recall: {recall_svm:.4f}')
print(f'F1 Score: {f1_svm:.4f}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (SVM)')
plt.show()

# Plot ROC curve
fpr_svm, tpr_svm, _ = roc_curve(df_binary_smote['y_test'], y_proba_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (SVM)')
plt.legend(loc='lower right')
plt.show()


# # Random Forest 

# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=seed_value)

# Define the hyperparameters to search
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform grid search with 10-fold cross-validation
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=10, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(df_label_oversampled['X_train_pca'], df_label_oversampled['y_train'])

# Get the best parameters and best accuracy
best_params_rf = grid_search_rf.best_params_
best_accuracy_rf = grid_search_rf.best_score_

print("Best Parameters for Random Forest Classifier:", best_params_rf)
print("Best Accuracy for Random Forest Classifier:", best_accuracy_rf)


# In[65]:


from sklearn.ensemble import RandomForestClassifier

# Best parameters obtained from grid search
best_params_rf = {
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'auto'
}

# Initialize and train the Random Forest model with the best parameters
rf_model = RandomForestClassifier(**best_params_rf)
rf_model.fit(df_label_oversampled['X_train_pca'], df_label_oversampled['y_train'])

# Make predictions on the test data
y_pred_rf = rf_model.predict(df_label_oversampled['X_test_pca'])

# Calculate probabilities for ROC curve
y_proba_rf = rf_model.predict_proba(df_label_oversampled['X_test_pca'])[:, 1]

# Evaluate the model's performance
conf_matrix_rf = confusion_matrix(df_label_oversampled['y_test'], y_pred_rf)
roc_auc_rf = roc_auc_score(df_label_oversampled['y_test'], y_proba_rf)
accuracy_rf = accuracy_score(df_label_oversampled['y_test'], y_pred_rf)
precision_rf = precision_score(df_label_oversampled['y_test'], y_pred_rf)
recall_rf = recall_score(df_label_oversampled['y_test'], y_pred_rf)
f1_rf = f1_score(df_label_oversampled['y_test'], y_pred_rf)

# Print performance measures
print(f'Accuracy: {accuracy_rf:.4f}')
print(f'Precision: {precision_rf:.4f}')
print(f'Recall: {recall_rf:.4f}')
print(f'F1 Score: {f1_rf:.4f}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# Plot ROC curve
fpr_rf, tpr_rf, _ = roc_curve(df_label_oversampled['y_test'], y_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Random Forest)')
plt.legend(loc='lower right')
plt.show()


# ### Creating model, predicting, calculating performance measures and drawing graphs

# In[ ]:


from sklearn.svm import SVC

# Best parameters obtained from grid search
best_params_svm = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# Initialize and train the SVM model with the best parameters
svm_model = SVC(**best_params_svm, probability=True)
svm_model.fit(df_binary_smote['X_train_pca'], df_binary_smote['y_train'])

# Make predictions on the test data
y_pred_svm = svm_model.predict(df_binary_smote['X_test_pca'])

# Calculate probabilities for ROC curve
y_proba_svm = svm_model.predict_proba(df_binary_smote['X_test_pca'])[:, 1]

# Evaluate the model's performance
conf_matrix_svm = confusion_matrix(df_binary_smote['y_test'], y_pred_svm)
roc_auc_svm = roc_auc_score(df_binary_smote['y_test'], y_proba_svm)
accuracy_svm = accuracy_score(df_binary_smote['y_test'], y_pred_svm)
precision_svm = precision_score(df_binary_smote['y_test'], y_pred_svm)
recall_svm = recall_score(df_binary_smote['y_test'], y_pred_svm)
f1_svm = f1_score(df_binary_smote['y_test'], y_pred_svm)

# Print performance measures
print(f'Accuracy: {accuracy_svm:.4f}')
print(f'Precision: {precision_svm:.4f}')
print(f'Recall: {recall_svm:.4f}')
print(f'F1 Score: {f1_svm:.4f}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (SVM)')
plt.show()

# Plot ROC curve
fpr_svm, tpr_svm, _ = roc_curve(df_binary_smote['y_test'], y_proba_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (SVM)')
plt.legend(loc='lower right')
plt.show()


# In[52]:


import matplotlib.pyplot as plt

# Class labels and counts
class_labels = ['<=5000', '>50000']
class_counts = [36080, 11541]

# Plotting the class imbalance
plt.figure(figsize=(8, 6))
plt.bar(class_labels, class_counts, color=['blue', 'orange'])
plt.xlabel('Income Class')
plt.ylabel('Number of Instances')
plt.title('Class Imbalance')
plt.show()


# In[80]:


import numpy as np
import matplotlib.pyplot as plt

# Baseline performance measures from previous research (blank for Extra Trees)
baseline_accuracy = [0, 84.6, 79.2]  # Accuracy for Extra Trees, Random Forest, and SVM
achieved_accuracy = [93.85, 92.64, 89.01]  # Accuracy for Extra Trees, Random Forest, and SVM
algorithms = ['Extra Trees', 'Random Forest', 'SVM']

bar_width = 0.35

index = np.arange(len(algorithms))

# Plot the bars for accuracy
plt.figure(figsize=(10, 6))
plt.bar(index, baseline_accuracy, bar_width, color='skyblue', label='Baseline')
plt.bar(index + bar_width, achieved_accuracy, bar_width, color='orange', label='Achieved')

plt.xlabel('Classification Method')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.xticks(index + bar_width / 2, algorithms)
plt.legend()

# Add grid lines every 5 points on the y-axis
plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.5)
plt.yticks(np.arange(0, 101, 5))

# Show plot
plt.tight_layout()
plt.show()


# In[82]:


# Baseline performance measures from previous research (blank for Extra Trees)
baseline_precision = [0, 81.0, 88.7]  # Precision for Extra Trees, Random Forest, and SVM
achieved_precision = [91.84, 89.09, 90.73]  # Precision for Extra Trees, Random Forest, and SVM

plt.figure(figsize=(10, 6))
plt.bar(index, baseline_precision, bar_width, color='lightgreen', label='Baseline')
plt.bar(index + bar_width, achieved_precision, bar_width, color='lightcoral', label='Achieved')

plt.xlabel('Classification Method')
plt.ylabel('Precision (%)')
plt.title('Precision Comparison')
plt.xticks(index + bar_width / 2, algorithms)
plt.legend()

plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.5)
plt.yticks(np.arange(0, 101, 5))

# Show plot
plt.tight_layout()
plt.show()


# In[88]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

roc_auc_rf = auc(fpr_rf, tpr_rf)
fpr_et, tpr_et = fpr, tpr
roc_auc_et = auc(fpr_et, tpr_et)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_et, tpr_et, color='green', lw=2, label='Extra Trees (AUC = %0.2f)' % roc_auc_et)
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM (AUC = %0.2f)' % roc_auc_svm)

# Plot random guessing line
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()




