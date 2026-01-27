import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\FinalProject\BrainMRI"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
VAL_DIR   = os.path.join(BASE_DIR, "Validation")
TEST_DIR  = os.path.join(BASE_DIR, "Testing")

OUT_CSV = r"D:\FinalProject\efficientnet_features.csv"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# ---------------- LOAD DATA ----------------
print("📂 Loading folder-based MRI dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("🧠 Classes:", class_names)

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds   = val_ds.map(lambda x, y: (preprocess_input(x), y))

# ---------------- MODEL ----------------
print("🧠 Building EfficientNetB0...")

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs
)

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu", name="feature_layer")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
print("🚀 Training EfficientNet...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2
)

# ---------------- FEATURE EXTRACTION (FIXED) ----------------
print("📤 Extracting deep features (memory-safe)...")

feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer("feature_layer").output
)

# CSV setup
feature_cols = [f"f{i}" for i in range(1, 129)]
df_out = pd.DataFrame(columns=["Image_Name"] + feature_cols)

def extract_from_folder(folder):
    rows = []
    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue

        images = sorted([
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".png"))
        ])

        for i in range(0, len(images), BATCH_SIZE):
            batch_files = images[i:i+BATCH_SIZE]
            batch_imgs = []

            for img_path in batch_files:
                img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img = tf.keras.utils.img_to_array(img)
                img = preprocess_input(img)
                batch_imgs.append(img)

            batch_imgs = np.array(batch_imgs)
            feats = feature_model.predict(batch_imgs, verbose=0)

            for fname, feat in zip(batch_files, feats):
                rows.append([os.path.basename(fname)] + feat.tolist())

    return rows

# Process folders one by one
print("🔹 TRAIN")
df_out = pd.concat([df_out, pd.DataFrame(extract_from_folder(TRAIN_DIR),
                columns=df_out.columns)])

print("🔹 VALIDATION")
df_out = pd.concat([df_out, pd.DataFrame(extract_from_folder(VAL_DIR),
                columns=df_out.columns)])

print("🔹 TEST")
df_out = pd.concat([df_out, pd.DataFrame(extract_from_folder(TEST_DIR),
                columns=df_out.columns)])

# ---------------- SAVE CSV ----------------
df_out.to_csv(OUT_CSV, index=False)

print(f"\n✅ EfficientNet FEATURES saved safely to:\n{OUT_CSV}")
print("\n===== SAMPLE FEATURE ROW =====")
print(df_out.head())
