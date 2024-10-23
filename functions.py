import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import warnings

def y_label_bar_graph(df):
    df.plot(kind='bar')
    plt.title('Balanced Distribution of Smoking (0 = Non-Smoker, 1 = Smoker)')
    plt.xlabel('Smoking')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Non-Smoker (0)', 'Smoker (1)'])
    plt.show()



def sns_plot(numeric_list, data, type):
    features = [col for col in numeric_list if col != 'smoking']

    n_features = len(features)
    n_cols = 4  # Number of columns in the grid
    n_rows = (n_features + n_cols - 1) // n_cols  # Number of rows required

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 5 * n_rows))
    axes = axes.flatten()

    # Loop over the features and create plots
    if type=="box":
        for i, feature in enumerate(features):
            sns.boxplot(ax=axes[i], x="smoking", y=feature, data=data, palette="Set2")
            axes[i].set_title(f"{feature} vs Smoking")
    elif type=="violin":
        for i, feature in enumerate(features):
            sns.violinplot(ax=axes[i], x="smoking", y=feature, data=data, palette="Set2")
            axes[i].set_title(f"{feature} vs Smoking")
    elif type=="histogram":
        for i, feature in enumerate(features):
            sns.histplot(ax=axes[i], x=feature, data=data, hue="smoking", palette="Set2", kde=True)
            axes[i].set_title(f"{feature} vs Smoking")
        

    # Delete any remaining unused axes if number of plots is less than grid size
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


    
def remove_outliers_zscore(df, column, threshold=3):
    mean = np.mean(df[column])
    std = np.std(df[column])
    z_scores = (df[column] - mean) / std
    return df[np.abs(z_scores) < threshold]



def model_classifier(X_train,X_test,y_train,y_test,model):
    mod = model
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)


    
def model_classifier_kfold(X, y, model, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=kf)
    
    # Calculate accuracy and metrics
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    
    # Print results
    print("Accuracy (K-Fold):", accuracy)
    print("Confusion Matrix (K-Fold):\n", conf_matrix)
    print("Classification Report (K-Fold):\n", report)
    
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation accuracy:", cv_scores.mean())
    

def plot_explained_variance(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA()
    pca.fit(X_train_scaled)
    
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.show()
    
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def model_classifier_pca(X_train, X_test, y_train, y_test, model, n_components=17):    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    mod = model
    mod.fit(X_train_pca, y_train)
    y_pred = mod.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)
