import numpy as np
import pandas as pd

# =====================================================
# 1. Reproducibility
# =====================================================
np.random.seed(42)

N_SAMPLES = 10_000

# =====================================================
# 2. Feature domains
# =====================================================
transaction_types = ["P2P", "Merchant", "BillPay"]
merchant_categories = ["Groceries", "Fuel", "Electronics", "Jewelry", "Travel"]
device_types = ["Android", "iOS", "Unknown"]

# =====================================================
# 3. Generate features
# =====================================================
df = pd.DataFrame({
    "transaction_amount": np.round(
        np.random.lognormal(mean=7.4, sigma=1.0, size=N_SAMPLES), 2
    ),
    "hour_of_day": np.random.randint(0, 24, N_SAMPLES),
    "account_age_days": np.random.randint(1, 5000, N_SAMPLES),
    "num_txn_last_24h": np.random.poisson(3, N_SAMPLES),
    "num_failed_txn_last_24h": np.random.poisson(0.4, N_SAMPLES),
    "distance_from_usual_km": np.round(
        np.random.exponential(4.5, N_SAMPLES), 2
    ),

    "transaction_type": np.random.choice(
        transaction_types, N_SAMPLES, p=[0.6, 0.3, 0.1]
    ),
    "merchant_category": np.random.choice(
        merchant_categories, N_SAMPLES
    ),
    "device_type": np.random.choice(
        device_types, N_SAMPLES, p=[0.7, 0.25, 0.05]
    ),
    "is_new_payee": np.random.choice(
        ["Yes", "No"], N_SAMPLES, p=[0.3, 0.7]
    )
})

# =====================================================
# 4. Risk score engineering
# =====================================================
risk_score = np.zeros(N_SAMPLES)

# Amount-based risk
risk_score += np.log1p(df["transaction_amount"]) * 0.7

# Frequency & failure anomalies
risk_score += df["num_txn_last_24h"] * 0.5
risk_score += df["num_failed_txn_last_24h"] * 3.0

# Distance anomaly
risk_score += (df["distance_from_usual_km"] > 40).astype(int) * 3.0

# Night-time behavior
risk_score += df["hour_of_day"].isin([0, 1, 2, 3, 4]).astype(int) * 1.5

# New / immature account
risk_score += (df["account_age_days"] < 90).astype(int) * 2.0

# Merchant risk
risk_score += df["merchant_category"].isin(
    ["Jewelry", "Electronics"]
).astype(int) * 1.8

# Device risk
risk_score += (df["device_type"] == "Unknown").astype(int) * 2.5

# New payee risk
risk_score += (df["is_new_payee"] == "Yes").astype(int) * 1.6

# =====================================================
# 5. Convert risk score → labels (FIXED)
# =====================================================
df["risk_label"] = np.select(
    [
        risk_score < 6.0,
        (risk_score >= 6.0) & (risk_score < 10.0),
        risk_score >= 10.0
    ],
    ["safe", "suspicious", "fraud"],
    default="safe"   # IMPORTANT: avoid dtype error
)

# =====================================================
# 6. Shuffle dataset
# =====================================================
df = df.sample(frac=1.0).reset_index(drop=True)

# =====================================================
# 7. Save dataset
# =====================================================
OUTPUT_PATH = "upi_risk.csv"
df.to_csv(OUTPUT_PATH, index=False)

# =====================================================
# 8. Sanity checks
# =====================================================
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["risk_label"].value_counts(normalize=True))
