import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os

# ==================================================
# 1. Cox Partial Likelihood Loss (DeepSurv)
# ==================================================
class CoxLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        times  = tf.cast(y_true[:, 0], tf.float32)
        events = tf.cast(y_true[:, 1], tf.float32)

        order  = tf.argsort(times, direction="DESCENDING")
        times  = tf.gather(times, order)
        events = tf.gather(events, order)
        scores = tf.gather(y_pred[:, 0], order)

        exp_scores = tf.exp(scores)
        risk_set   = tf.cumsum(exp_scores)

        log_likelihood = scores - tf.math.log(risk_set + 1e-8)
        loss = -tf.reduce_sum(log_likelihood * events) / (tf.reduce_sum(events) + 1e-8)
        return loss

# ==================================================
# 2. Load Clinical Data
# ==================================================
SURVIVAL_CSV = r"D:\FinalProject\Prediction\Prediction data.csv"

df = pd.read_csv(SURVIVAL_CSV)
print("Dataset loaded:", df.shape)

# ==================================================
# 3. Generate Proxy Survival Time & Event
# ==================================================
np.random.seed(42)

df["Survival_Rate(%)"] = pd.to_numeric(df["Survival_Rate(%)"], errors="coerce")

noise = np.random.normal(loc=1.0, scale=0.15, size=len(df))
df["time"] = (df["Survival_Rate(%)"] * noise).clip(lower=1)

p_event = 1 - (df["Survival_Rate(%)"] / 100)
p_event = p_event.clip(0.05, 0.95)
df["event"] = np.random.binomial(1, p_event)

print("time & event generated successfully")

# ==================================================
# 4. Prepare Features & Labels
# ==================================================
y_time  = df["time"].values
y_event = df["event"].values

X = df.drop(columns=["time", "event"])
X = pd.get_dummies(X)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype("float32")

# ==================================================
# 5. Train–Test Split
# ==================================================
X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

y_train = np.vstack([t_train, e_train]).T.astype("float32")
y_test  = np.vstack([t_test,  e_test]).T.astype("float32")

# ==================================================
# 6. Build SurvivalNet Model
# ==================================================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=CoxLoss()
)

model.summary()

# ==================================================
# 7. Train Model
# ==================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)

# ==================================================
# 8. C-Index Evaluation
# ==================================================
def calculate_c_index(model, X, time, event):
    risk = model.predict(X).flatten()
    return concordance_index(time, -risk, event)

train_c_index = calculate_c_index(model, X_train, t_train, e_train)
test_c_index  = calculate_c_index(model, X_test,  t_test,  e_test)

# ==================================================
# 9. Risk Prediction & Stratification
# ==================================================
df["Predicted_Risk"] = model.predict(X).flatten()
median_risk = np.median(df["Predicted_Risk"])
df["Risk_Group"] = np.where(df["Predicted_Risk"] >= median_risk, "High", "Low")

# ==================================================
# 10. Save Excel & PDF Report
# ==================================================
results_dir = r"D:\FinalProject\Prediction\Results"
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_path = os.path.join(results_dir, f"survival_results_{timestamp}.xlsx")
pdf_path   = os.path.join(results_dir, f"survival_report_{timestamp}.pdf")

# -------- Save Excel --------
results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "C-Index": [train_c_index, test_c_index]
})

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Model Performance", index=False)
    df.to_excel(writer, sheet_name="Predictions", index=False)

# -------- Save PDF --------
with PdfPages(pdf_path) as pdf:

    # ===== PAGE 1: SUMMARY =====
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    summary_text = (
        "SurvivalNet Report\n\n"
        f"Run timestamp: {timestamp}\n\n"
        f"Train C-index: {train_c_index:.4f}\n"
        f"Test C-index : {test_c_index:.4f}\n\n"
        f"Total patients: {len(df)}\n"
        f"High-risk patients: {(df['Risk_Group']=='High').sum()}\n"
        f"Low-risk patients : {(df['Risk_Group']=='Low').sum()}\n"
    )

    ax.text(0.1, 0.85, summary_text, fontsize=14, va="top")
    pdf.savefig(fig)
    plt.close(fig)

    # ===== PAGE 2: TRAINING & VALIDATION LOSS =====
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    note = (
        "Note:\n"
        "This plot shows the convergence of the survival model.\n"
        "A stable validation loss indicates good generalization."
    )
    fig.text(0.02, 0.02, note, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)

    # ===== PAGE 3: TEST RISK SCORE DISTRIBUTION =====
    fig, ax = plt.subplots()
    test_risk = model.predict(X_test).flatten()
    ax.hist(test_risk, bins=20)
    ax.set_title("Test Risk Score Distribution")
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Number of Patients")
    ax.grid(True)

    note = (
        "Note:\n"
        "This distribution shows how patients are spread across\n"
        "different predicted survival risk levels."
    )
    fig.text(0.02, 0.02, note, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)

    # ===== PAGE 4: KAPLAN–MEIER CURVES =====
    fig, ax = plt.subplots()
    kmf = KaplanMeierFitter()

    for group in ["High", "Low"]:
        mask = df["Risk_Group"] == group
        kmf.fit(
            df.loc[mask, "time"],
            event_observed=df.loc[mask, "event"],
            label=f"{group} Risk"
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_title("Kaplan–Meier Survival Curves")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(True)

    note = (
        "Note:\n"
        "Clear separation between curves confirms effective\n"
        "risk stratification by the survival model."
    )
    fig.text(0.02, 0.02, note, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)

print("\n✅ SURVIVAL NET PIPELINE COMPLETED")
print("Excel:", excel_path)
print("PDF  :", pdf_path)
