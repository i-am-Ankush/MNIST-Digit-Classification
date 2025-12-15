# MNIST Digit Classification


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns



print("Downloading MNIST... please wait")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print("Download finished")

print("X shape:", X.shape)
print("y shape:", y.shape)


X = np.array(X)
y = np.array(y)


plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y[0]}")
plt.axis('off')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)
print("Pixel range:", X_train.min(), "to", X_train.max())

model = LogisticRegression(
    max_iter=1000,
    solver="saga",
    n_jobs=-1
)





model.fit(X_train, y_train)

y_pred = model.predict(X_test
                       )
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)



'''  

# ==============================
# Optional: Random Forest Model
# ==============================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\nTraining Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Test Accuracy:", rf_accuracy)

'''









cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MNIST")
plt.show()


import random

wrong_idx = np.where(y_test != y_pred)[0]
idx = random.choice(wrong_idx)

plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
plt.title(f"Actual: {y_test[idx]} | Predicted: {y_pred[idx]}")
plt.axis("off")
plt.show()







