import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
        

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


    
def remove_outliers_zscore(df, column, threshold=3):
    mean = np.mean(df[column])
    std = np.std(df[column])
    z_scores = (df[column] - mean) / std
    return df[np.abs(z_scores) < threshold]



def model_classifier(X_train, X_test, y_train, y_test, model):
    # Split validation from the test set
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Compute log loss
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    y_test_final_proba = model.predict_proba(X_test_final)

    train_loss = log_loss(y_train, y_train_proba)
    val_loss = log_loss(y_val, y_val_proba)
    test_loss = log_loss(y_test_final, y_test_final_proba)

    # Report losses
    print("Train Loss:", train_loss)
    print("Validation Loss:", val_loss)
    print("Test Loss:", test_loss)

    # Classification reports
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test_final)

    print("\n=== Train Set ===")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))

    print("\n=== Validation Set ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Classification Report:\n", classification_report(y_val, y_val_pred))

    print("\n=== Test Set ===")
    print("Accuracy:", accuracy_score(y_test_final, y_test_pred))
    print("Classification Report:\n", classification_report(y_test_final, y_test_pred))


def model_classifier_kfold(X, y, model, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_losses = []
    val_losses = []
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split data for the fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train the model
        model.fit(X_train, y_train)

        # Compute log loss
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)

        train_loss = log_loss(y_train, y_train_proba)
        val_loss = log_loss(y_val, y_val_proba)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Validation accuracy
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        fold_accuracies.append(val_accuracy)

        print(f"Fold {fold}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.4f}")

    # Report mean metrics
    print("\n=== Summary ===")
    print(f"Mean Train Loss: {sum(train_losses) / n_splits:.4f}")
    print(f"Mean Validation Loss: {sum(val_losses) / n_splits:.4f}")
    print(f"Mean Validation Accuracy: {sum(fold_accuracies) / n_splits:.4f}")


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


def model_classifier_pca(X_train, X_test, y_train, y_test, model, n_components=17):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Split validation from the test set
    X_val_pca, X_test_final_pca, y_val, y_test_final = train_test_split(
        X_test_pca, y_test, test_size=0.5, random_state=42
    )

    # Train the model
    model.fit(X_train_pca, y_train)

    # Compute log loss
    y_train_proba = model.predict_proba(X_train_pca)
    y_val_proba = model.predict_proba(X_val_pca)
    y_test_final_proba = model.predict_proba(X_test_final_pca)

    train_loss = log_loss(y_train, y_train_proba)
    val_loss = log_loss(y_val, y_val_proba)
    test_loss = log_loss(y_test_final, y_test_final_proba)

    # Report losses
    print("Train Loss:", train_loss)
    print("Validation Loss:", val_loss)
    print("Test Loss:", test_loss)

    # Classification reports
    y_train_pred = model.predict(X_train_pca)
    y_val_pred = model.predict(X_val_pca)
    y_test_pred = model.predict(X_test_final_pca)

    print("\n=== Train Set ===")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))

    print("\n=== Validation Set ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Classification Report:\n", classification_report(y_val, y_val_pred))

    print("\n=== Test Set ===")
    print("Accuracy:", accuracy_score(y_test_final, y_test_pred))
    print("Classification Report:\n", classification_report(y_test_final, y_test_pred))
