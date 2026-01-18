
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import BinaryAccuracy, AUC


# ------------------------------------------
# LOAD DATA
# ------------------------------------------
#df = pd.read_csv("Disaster.csv")
df = pd.read_csv("Disaster_5000.csv")

X = df[
    ["temp_min_c","temp_max_c","temp_avg_c",
     "rainfall_mm","humidity_percent","soil_moisture"]
].values

y = df[
    ["flood","cyclone_storm","heatwave",
     "landslide","wildfire","volcanic_eruption"]
].values


# ------------------------------------------
# NORMALIZATION
# ------------------------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = X.reshape(X.shape[0], X.shape[1], 1)


# ------------------------------------------
# SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------
# MODEL
# ------------------------------------------
model = Sequential([
    Conv1D(64, 3, padding="same", activation="relu",
           input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(),

##    Conv1D(128, 3, padding="same", activation="relu"),
##    BatchNormalization(),
##    MaxPooling1D(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.7),

    Dense(6, activation="sigmoid")
])


# ------------------------------------------
# COMPILE (IMPORTANT CHANGE)
# ------------------------------------------
model.compile(
    #optimizer=Adam(0.0005),
    optimizer=Adam(0.01,),
    loss="binary_crossentropy",
    metrics=[
        BinaryAccuracy(name="binary_accuracy"),
        AUC(name="auc")
    ]
)

model.summary()


# ------------------------------------------
# TRAIN
# ------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)


# ------------------------------------------
# EVALUATION
# ------------------------------------------
loss, binary_acc, auc = model.evaluate(X_test, y_test)

print(f"\nBinary Accuracy : {binary_acc*100:.2f}%")
print(f"AUC Score       : {auc*100:.2f}%")


# ------------------------------------------
# THRESHOLD PREDICTIONS
# ------------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))



model.save("dcnn_disaster_model_final.h5")
print("✅ Model Saved")


joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler Saved")




# ==========================================
# 13. ACCURACY & LOSS GRAPHS
# ==========================================

plt.figure(figsize=(12, 5))

# ---- Binary Accuracy ----
plt.subplot(1, 2, 1)
plt.plot(history.history["binary_accuracy"], label="Train Binary Accuracy")
plt.plot(history.history["val_binary_accuracy"], label="Validation Binary Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Binary Accuracy")
plt.title("DCNN Binary Accuracy")
plt.legend()
plt.grid(True)

# ---- Loss ----
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("DCNN Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
