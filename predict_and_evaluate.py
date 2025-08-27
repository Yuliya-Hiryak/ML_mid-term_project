import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def predict_and_evaluate(model, X, y, name = ''):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    print(f"F1 Score {name}:", f1)
    print(f"AUC Score {name}:", auc)
    print(f"Classification Report {name}:\n", classification_report(y, y_pred))
    print(f"Confusion Matrix {name}:\n", confusion_matrix(y, y_pred))

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Signed', 'Signed'], yticklabels=['Not Signed', 'Signed'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix {name}')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC {name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.show()