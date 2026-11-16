"""
House Price Prediction - Data Science Project
---------------------------------------------
Steps: load data -> EDA -> preprocess -> train models -> evaluate -> predict

Use your own dataset: put a CSV (e.g. Kaggle "House Prices" / Bengaluru house data)
in the same folder and set CSV_PATH and TARGET below.
If no CSV is found, a synthetic dataset is generated so the project runs end to end.

Install:  pip install pandas numpy scikit-learn matplotlib seaborn joblib
Run:      python house_price_prediction.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files, no Tk window (fixes IDLE error)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH = "house_data.csv"   # change to your dataset
TARGET = "price"              # change to your target column
RANDOM_STATE = 42


# 1. LOAD DATA -------------------------------------------------------------
def make_synthetic_data(n=2000):
    rng = np.random.default_rng(RANDOM_STATE)
    area = rng.normal(1500, 500, n).clip(400, 5000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5)
    age = rng.integers(0, 40, n)
    location = rng.choice(["Downtown", "Suburb", "Rural", "Lakeside"], n,
                          p=[0.25, 0.4, 0.2, 0.15])
    loc_factor = pd.Series(location).map(
        {"Downtown": 1.4, "Suburb": 1.0, "Rural": 0.7, "Lakeside": 1.6}).values
    price = (area * 3000 + bedrooms * 150000 + bathrooms * 100000
             - age * 8000) * loc_factor + rng.normal(0, 150000, n)
    return pd.DataFrame({"area_sqft": area.round(), "bedrooms": bedrooms,
                         "bathrooms": bathrooms, "age_years": age,
                         "location": location, "price": price.round()})


def load_data():
    if os.path.exists(CSV_PATH):
        print(f"Loading {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    print("CSV not found -> using synthetic data")
    return make_synthetic_data()


df = load_data()
print("\nShape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())

# 2. EDA -------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], kde=True)
plt.title("Price distribution")
plt.tight_layout()
plt.savefig("plots/price_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("plots/correlation.png")
plt.close()

# 3. PREPROCESSING ---------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

# 4. TRAIN & COMPARE MODELS ------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "CV R2": round(cv_r2, 4),
        "Test R2": round(r2_score(y_test, pred), 4),
        "MAE": round(mean_absolute_error(y_test, pred)),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, pred))),
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("Test R2", ascending=False)
print("\nModel comparison:\n", results_df.to_string(index=False))

# 5. BEST MODEL: PLOTS + SAVE ----------------------------------------------
best_name = results_df.iloc[0]["Model"]
best = fitted[best_name]
print(f"\nBest model: {best_name}")

pred = best.predict(X_test)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, alpha=0.4)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

model_step = best.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    names = best.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model_step.feature_importances_, index=names).sort_values().tail(10)
    imp.plot(kind="barh", figsize=(7, 4), title="Top feature importances")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()

joblib.dump(best, "house_price_model.pkl")
print("Saved model -> house_price_model.pkl, plots -> plots/")

# 6. PREDICT NEW HOUSE -----------------------------------------------------
if os.path.exists(CSV_PATH):
    sample = X_test.iloc[[0]]
else:
    sample = pd.DataFrame([{"area_sqft": 1800, "bedrooms": 3, "bathrooms": 2,
                            "age_years": 5, "location": "Suburb"}])
loaded = joblib.load("house_price_model.pkl")
print("\nSample input:\n", sample.to_string(index=False))
print(f"Predicted price: {loaded.predict(sample)[0]:,.0f}")
