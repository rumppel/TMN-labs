import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# Завантаження даних
url = "bioresponse.csv"
data = pd.read_csv(url)

# Розділення даних на тренувальний та тестовий набори
X = data.drop('Activity', axis=1)
y = data['Activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання класифікаторів
# Дрібне дерево рішень
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)

# Глибоке дерево рішень
deep_dt_model = DecisionTreeClassifier(max_depth=10)  # Вказати глибину дерева
deep_dt_model.fit(X_train, y_train)

# Випадковий ліс на дрібних деревах
rf_small_model = RandomForestClassifier(n_estimators=10, max_depth=5)  # Вказати кількість дерев та їх глибину
rf_small_model.fit(X_train, y_train)

# Випадковий ліс на глибоких деревах
rf_deep_model = RandomForestClassifier(n_estimators=10, max_depth=20)  # Вказати кількість дерев та їх глибину
rf_deep_model.fit(X_train, y_train)

shared_plot = True

# Оцінка якості моделей
if shared_plot:
    plt.figure(figsize=(10, 4))

models = [dt_model, deep_dt_model, rf_small_model, rf_deep_model]
names = ['дрібне дерево рішень', 'глибоке дерево рішень', 'випадковий ліс на дрібних деревах', 'випадковий ліс на глибоких деревах']
for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, Log-loss: {logloss}")

    # Precision-Recall і ROC-криві
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    if not shared_plot:
        plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=names[i])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=names[i])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')

    if not shared_plot:
        plt.show()

# Використаємо збалансованість класів для RandomForestClassifier
balanced_rf_model = RandomForestClassifier(max_depth=20, class_weight='balanced')
balanced_rf_model.fit(X_train, y_train)

# Оцінка якості моделі, яка уникає помилок II роду
y_pred_proba_balanced = balanced_rf_model.predict_proba(X_test)[:, 1]
threshold = 0.1
y_pred_balanced = (y_pred_proba_balanced > threshold).astype(int)

acc_balanced = accuracy_score(y_test, y_pred_balanced)
precision_balanced = precision_score(y_test, y_pred_balanced)
recall_balanced = recall_score(y_test, y_pred_balanced)
f1_balanced = f1_score(y_test, y_pred_balanced)
logloss_balanced = log_loss(y_test, y_pred_proba_balanced)

print(f"\nMetrics for Balanced Model:")
print(f"Accuracy: {acc_balanced}, Precision: {precision_balanced}, Recall: {recall_balanced}, F1-score: {f1_balanced}, Log-loss: {logloss_balanced}")

# Precision-Recall і ROC-криві
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

if not shared_plot:
    plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(recall, precision, label='Уникання помилок II роду')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label='Уникання помилок II роду')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')

plt.show()
