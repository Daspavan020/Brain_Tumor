🟢 Final Tip

# 411 - 669 Unet Code


<!-- Every time you open PowerShell, you must activate venv: -->

------> D:\FinalProject\tf_gpu\Scripts\Activate.ps1

<!-- Run this Code When ever the block is shared to another system -->

----------> python -m venv tf_gpu
            ./tf_gpu/Scripts/Activate.ps1
            pip install -r requirements.txt


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
.\tf_gpu\Scripts\activate

-----------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Cox Partial Likelihood Loss (DeepSurv)
# ------------------------------------------------
class CoxLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        times = y_true[:, 0]
        events = y_true[:, 1]

        order = tf.argsort(times, direction='DESCENDING')
        times = tf.gather(times, order)
        events = tf.gather(events, order)
        scores = tf.gather(y_pred[:, 0], order)

        exp_scores = tf.exp(scores)
        risk_set = tf.cumsum(exp_scores)

        log_likelihood = scores - tf.math.log(risk_set + 1e-8)
        neg_log_likelihood = -tf.reduce_sum(log_likelihood * events) / (tf.reduce_sum(events) + 1e-8)

        return neg_log_likelihood


# ------------------------------------------------
# 2. Load Data
# ------------------------------------------------
df = pd.read_csv(r"D:\FinalProject\survival_data.csv")

y_time = df["time"].values
y_event = df["event"].values
X = df.drop(columns=["time", "event"])
X = pd.get_dummies(X)

feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

y_train = np.vstack([t_train, e_train]).T
y_test = np.vstack([t_test, e_test]).T


# ------------------------------------------------
# 3. Build Model
# ------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=CoxLoss())

model.summary()

# ------------------------------------------------
# 4. Train Model
# ------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)


# ------------------------------------------------
# 5. C-index Calculation & Display Table
# ------------------------------------------------
def calculate_c_index(model, X, time, event):
    risk = model.predict(X).flatten()
    return concordance_index(time, -risk, event)

train_c_index = calculate_c_index(model, X_train, t_train, e_train)
test_c_index = calculate_c_index(model, X_test, t_test, e_test)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "C-Index": [train_c_index, test_c_index]
})

print("\n===== MODEL PERFORMANCE (TABLE) =====\n")
print(results_df, "\n")


# ------------------------------------------------
# 6. Predict Risks for All Test Patients and Show Table
# ------------------------------------------------
test_pred = model.predict(X_test).flatten()
pred_df = pd.DataFrame({
    "Time": t_test,
    "Event": e_test,
    "Predicted Risk": test_pred
})

print("\n===== SAMPLE PREDICTIONS TABLE =====\n")
print(pred_df.head(), "\n")


# ------------------------------------------------
# 7. Plot Loss Curve
# ------------------------------------------------
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()


# ------------------------------------------------
# 8. Plot Predicted Risk Distribution
# ------------------------------------------------
plt.figure()
plt.hist(test_pred, bins=20)
plt.xlabel("Predicted Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution on Test Set")
plt.show()


# ------------------------------------------------
# 9. Predict for ONE Patient
# ------------------------------------------------
def predict_patient(patient_dict):
    df_input = pd.DataFrame([patient_dict])
    df_input = pd.get_dummies(df_input)

    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_names]
    x_scaled = scaler.transform(df_input)

    risk_score = model.predict(x_scaled)[0][0]
    return risk_score


example_patient = {
    "Tumor Type": "Glioma",
    "Location": "Frontal Lobe",
    "Size (cm)": 5.4,
    "Grade": "III",
    "Patient Age": 45,
    "Gender": "Male"
}

print("\n===== ONE PATIENT PREDICTION =====")
print("Predicted Survival Risk =", predict_patient(example_patient))



-----------------------------------------------------------------------------------------------------------------------------------------------------------------






import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Cox Partial Likelihood Loss (DeepSurv)
# ------------------------------------------------
class CoxLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        times = y_true[:, 0]
        events = y_true[:, 1]

        # Sort by descending time
        order = tf.argsort(times, direction='DESCENDING')
        times = tf.gather(times, order)
        events = tf.gather(events, order)
        scores = tf.gather(y_pred[:, 0], order)

        exp_scores = tf.exp(scores)
        risk_set = tf.cumsum(exp_scores)

        log_likelihood = scores - tf.math.log(risk_set + 1e-8)
        neg_log_likelihood = -tf.reduce_sum(log_likelihood * events) / (tf.reduce_sum(events) + 1e-8)

        return neg_log_likelihood


# ------------------------------------------------
# 2. Load Data
# ------------------------------------------------
df = pd.read_csv(r"D:\FinalProject\survival_data.csv")

y_time = df["time"].values
y_event = df["event"].values
X = df.drop(columns=["time", "event"])
X = pd.get_dummies(X)

feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

y_train = np.vstack([t_train, e_train]).T
y_test = np.vstack([t_test, e_test]).T


# ------------------------------------------------
# 3. Build Model
# ------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=CoxLoss())

model.summary()


# ------------------------------------------------
# 4. Train Model
# ------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)


# ------------------------------------------------
# 5. C-index Calculation & Display Table
# ------------------------------------------------
def calculate_c_index(model, X, time, event):
    risk = model.predict(X).flatten()
    return concordance_index(time, -risk, event)   # negative risk: higher risk = shorter survival

train_c_index = calculate_c_index(model, X_train, t_train, e_train)
test_c_index = calculate_c_index(model, X_test, t_test, e_test)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "C-Index": [train_c_index, test_c_index]
})

print("\n===== MODEL PERFORMANCE (TABLE) =====\n")
print(results_df, "\n")


# ------------------------------------------------
# 6. Predict risk for ALL patients in dataset
# ------------------------------------------------
all_risk_scores = model.predict(X).flatten()

# Start from original dataframe so all columns are included
pred_df = df.copy()

# Add patient ID (optional but useful)
pred_df.insert(0, "Patient_ID", range(1, len(df) + 1))

# Add predicted risk column
pred_df["Predicted_Risk"] = all_risk_scores

print("\n===== SAMPLE OF PATIENT RISK TABLE (FULL DATA) =====")
print(pred_df.head())



# ------------------------------------------------
# 7. Plot Loss Curve
# ------------------------------------------------
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()


# ------------------------------------------------
# 8. Plot Predicted Risk Distribution (on TEST SET)
# ------------------------------------------------
# compute test risk scores
test_pred = model.predict(X_test).flatten()

plt.figure()
plt.hist(test_pred, bins=20)
plt.xlabel("Predicted Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution on Test Set")
plt.show()


# ------------------------------------------------
# 9. Predict for ONE Patient
# ------------------------------------------------
def predict_patient(patient_dict):
    df_input = pd.DataFrame([patient_dict])
    df_input = pd.get_dummies(df_input)

    # align columns with training features
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_names]
    x_scaled = scaler.transform(df_input)

    risk_score = model.predict(x_scaled)[0][0]
    return risk_score


example_patient = {
    "Tumor Type": "Glioma",
    "Location": "Frontal Lobe",
    "Size (cm)": 5.4,
    "Grade": "III",
    "Patient Age": 45,
    "Gender": "Male"
}

print("\n===== ONE PATIENT PREDICTION =====")
print("Predicted Survival Risk =", predict_patient(example_patient))


# ------------------------------------------------
# 10. SAVE RESULTS TO ONE EXCEL FILE
# ------------------------------------------------
output_path = r"D:\FinalProject\survival_results.xlsx"  # change path if needed

# Make training history table
hist_df = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss']) + 1),
    "Train Loss": history.history['loss'],
    "Validation Loss": history.history['val_loss']
})

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Save accuracy table
    results_df.to_excel(writer, sheet_name="Model Accuracy", index=False)

    # Save predictions table (ALL patients)
    pred_df.to_excel(writer, sheet_name="Predictions", index=False)

    # Save training & validation loss over epochs
    hist_df.to_excel(writer, sheet_name="Training History", index=False)

print(f"\n===== FILE SAVED =====\n{output_path}\n")



----------------------------------------------------------------------------------------------------------------------------------------------------------
# The Previous Unet Code 411 - 669
"""
Optimized U-Net pipeline for DirectML GPU acceleration (6GB GPU version).
scm-history-item:d%3A%5CFinalProject?%7B%22repositoryId%22%3A%22scm0%22%2C%22historyItemId%22%3A%225da4f4ab8039482e6556af93fa31546490ec18e7%22%2C%22historyItemParentId%22%3A%223b30f0fd91740402b22531c00199f00477dade2f%22%2C%22historyItemDisplayId%22%3A%225da4f4a%22%7D
Changes applied:
- Random sampling for dataset truncation
- Reduced epochs for faster training (30–50 mins)
- Safe resource logging (DirectML compatible)
- Correct U-Net skip connections
- Optional mixed precision
"""

import os
os.environ["TF_USE_DIRECTML"] = "1"  # Required for AMD/NVIDIA mixed systems

# ---------------------- IMPORTS ----------------------
import glob, random, psutil
import numpy as np, h5py, cv2
from functools import partial
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

# Prevent TF from reserving 100% VRAM
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Mixed Precision (improves speed on DirectML)
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print("⚙️ Mixed Precision Enabled (FP16)\n")
except:
    print("⚠️ Mixed Precision Not Supported on this system.\n")

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback

# ---------------------- CONFIG ----------------------
DATA_DIR = "data"
OUT_DIR = "files_h5"
RESULTS_DIR = "results_h5"

IMG_H, IMG_W = 240, 240
INPUT_CHANNELS = 4
NUM_CLASSES = 3
LR = 1e-4

# Target training time: ~30–50 minutes
EPOCHS = 25  # Reduced from 150

USE_FILE_LIMIT = True
FILE_LIMIT = 6000  # Reduced dataset for speed

# ---------------------- BATCH SIZE ----------------------
def autotune_batch():
    # 6GB GPU → best safe range: 8–24
    return 16

BATCH_SIZE = autotune_batch()
print(f"\n🔧 Auto Batch Size = {BATCH_SIZE}\n")

# Create directories
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------- DATA PIPELINE ----------------------
def read_h5_file(path):
    with h5py.File(path, 'r') as f:
        img, msk = f['image'][:], f['mask'][:]

    img = img.astype(np.float32)
    if img.max() > 1.0: img /= img.max()

    msk = msk.astype(np.float32)
    if msk.ndim == 2:
        msk = tf.one_hot(msk.astype(np.int32), NUM_CLASSES).numpy().astype(np.float32)

    return img, msk

def _py_read_h5(path_bytes):
    path = path_bytes.decode()
    img, msk = read_h5_file(path)

    img = cv2.resize(img, (IMG_W, IMG_H))
    msk = cv2.resize(msk, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

    return img.astype(np.float32), msk.astype(np.float32)

def tf_load_h5(path):
    img, msk = tf.numpy_function(_py_read_h5, [path], [tf.float32, tf.float32])
    img.set_shape([IMG_H, IMG_W, INPUT_CHANNELS])
    msk.set_shape([IMG_H, IMG_W, NUM_CLASSES])
    return img, msk

def tf_augment(img, msk):
    if tf.random.uniform([]) < 0.5:
        img = tf.image.flip_left_right(img)
        msk = tf.image.flip_left_right(msk)
    if tf.random.uniform([]) < 0.5:
        img = tf.image.flip_up_down(img)
        msk = tf.image.flip_up_down(msk)
    return img, msk

def build_datasets():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    
    if USE_FILE_LIMIT:
        files = random.sample(files, min(len(files), FILE_LIMIT))

    random.shuffle(files)
    n = len(files)

    test_n, val_n = int(n*0.15), int(n*0.15)
    test_files = files[:test_n]
    val_files = files[test_n:test_n+val_n]
    train_files = files[test_n+val_n:]

    print(f"📁 Loaded {n} files → Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    def make_ds(file_list, aug=False):
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.map(tf_load_h5, num_parallel_calls=tf.data.AUTOTUNE)
        if aug: ds = ds.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return make_ds(train_files, True), make_ds(val_files), make_ds(test_files)

# ---------------------- MODEL ----------------------
def conv_block(x, filters):
    for _ in range(2):
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def build_unet():
    inputs = Input((IMG_H, IMG_W, INPUT_CHANNELS))

    s1 = conv_block(inputs, 64);  p1 = MaxPool2D()(s1)
    s2 = conv_block(p1, 128);     p2 = MaxPool2D()(s2)
    s3 = conv_block(p2, 256);     p3 = MaxPool2D()(s3)
    s4 = conv_block(p3, 512);     p4 = MaxPool2D()(s4)

    b = conv_block(p4, 1024)

    def up(x, skip, filters):
        x = Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = Concatenate()([x, skip])
        return conv_block(x, filters)

    d1 = up(b, s4, 512)
    d2 = up(d1, s3, 256)
    d3 = up(d2, s2, 128)
    d4 = up(d3, s1, 64)

    outputs = Conv2D(NUM_CLASSES, 1, activation="softmax", dtype="float32")(d4)
    return Model(inputs, outputs)

# ---------------------- LOSS ----------------------
SMOOTH = 1e-7

def dice_coef(y_true, y_pred):
    num = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
    den = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    return tf.reduce_mean((num+SMOOTH)/(den+SMOOTH))

def combined_loss(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))


# ---------------------- PREDICTION UTILITIES ----------------------

def create_masked_image(mri, mask, tumor_class=1):
    """
    mri  : (240, 240, 4)
    mask : (240, 240)
    output: (224, 224, 3) uint8 image for EfficientNet
    """
    binary_mask = (mask == tumor_class).astype(np.float32)

    # Use first MRI channel (e.g., FLAIR)
    masked = mri[:, :, 0] * binary_mask

    masked = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX)
    masked = cv2.resize(masked, (224, 224))

    masked_rgb = np.stack([masked] * 3, axis=-1)
    return masked_rgb.astype(np.uint8)


def save_predictions_for_efficientnet(model, dataset,
                                      out_dir="results_h5/efficientnet_inputs",
                                      mask_dir="results_h5/raw_masks"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    idx = 0
    for imgs, _ in dataset:
        preds = model.predict(imgs)
        masks = np.argmax(preds, axis=-1)

        imgs = imgs.numpy()

        for i in range(len(imgs)):
            eff_img = create_masked_image(imgs[i], masks[i])

            cv2.imwrite(f"{out_dir}/img_{idx}.png", eff_img)
            np.save(f"{mask_dir}/mask_{idx}.npy", masks[i])

            idx += 1

    print(f"\n✅ Saved {idx} EfficientNet-ready images.")

# ---------------------- CALLBACKS ----------------------
class ResourceLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        cpu = psutil.cpu_percent()
        try:
            import GPUtil
            g = GPUtil.getGPUs()[0]
            print(f"\n🧪 Epoch {epoch+1}: CPU {cpu}% | GPU {g.load*100:.1f}% | VRAM {g.memoryUsed}/{g.memoryTotal} MB")
        except:
            print(f"\n🧪 Epoch {epoch+1}: CPU {cpu}% | GPU: DirectML (usage not measurable)")

# ---------------------- TRAIN ----------------------
def train():
    train_ds, val_ds, test_ds = build_datasets()
    
    model = build_unet()
    model.compile(optimizer=Adam(LR), loss=combined_loss, metrics=[dice_coef])
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join(OUT_DIR, "best_model.h5"), save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger(os.path.join(OUT_DIR, "log.csv")),
        ResourceLogger()
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print("\n🎉 Training completed.")
    print("📤 Generating EfficientNet inputs from TEST set...")

    save_predictions_for_efficientnet(model, test_ds)

    return model


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    train()


---------------------------------------------------------------------------------------------------------------------------------------------------------------------