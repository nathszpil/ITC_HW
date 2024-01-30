import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle


def main():
    df = pd.read_csv('cellular_churn_greece.csv')
    X = df.drop(columns=['churned'])
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    X_test.to_csv('X_test.csv', index=False)

    np.savetxt('preds.csv', y_pred, delimiter=',', fmt='%d')


if __name__ == "__main__":
    main()
