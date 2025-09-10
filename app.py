# ===============================
# 1. Import libraries
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ===============================
# 2. Load dataset
# ===============================
file_path = "medicine_with_expiry.csv"
df = pd.read_csv(file_path)

# ===============================
# 3. Clean and standardize column names
# ===============================
df.columns = (
    df.columns
    .str.strip()
    .str.replace("Â", "", regex=False)
    .str.replace("°", "C", regex=False)
)

# Fix specific column that became "CC"
df = df.rename(columns={"Storage Temperature (CC)": "Storage Temperature (C)"})

print("✅ Final Cleaned Columns:\n", df.columns.tolist())

# ===============================
# 4. Handle missing values
# ===============================
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

# ===============================
# 5. Define features & target
# ===============================
y = df["Safe/Not Safe"]
le = LabelEncoder()
y = le.fit_transform(y)

numeric_cols = [
    "Days Until Expiry",
    "Storage Temperature (C)",
    "Dissolution Rate (%)",
    "Disintegration Time (minutes)",
    "Impurity Level (%)",
    "Assay Purity (%)",
    "Warning Labels Present"
]

# If Warning Labels is text (Yes/No), convert to numeric
if df["Warning Labels Present"].dtype == "object":
    df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0})

# Features
X = df[["Active Ingredient", "Disease/Use Case"] + numeric_cols]

# ===============================
# 6. Preprocessor (Text + Numeric)
# ===============================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("text_ing", TfidfVectorizer(), "Active Ingredient"),
        ("text_dis", TfidfVectorizer(), "Disease/Use Case"),
        ("num", numeric_transformer, numeric_cols)
    ]
)

# ===============================
# 7. Build model pipeline
# ===============================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ===============================
# 8. Train-test split & Train model
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# ===============================
# 9. Evaluate model
# ===============================
y_pred = model.predict(X_test)
print("\n📊 Model Performance Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ===============================
# 10. Safety Rule Thresholds
# ===============================
SAFETY_RULES = {
    "Days Until Expiry": {"min": 30, "message": "expiry is too close. Suggest medicines with ≥ 30 days shelf life."},
    "Storage Temperature (C)": {"range": (15, 30), "message": "temperature out of safe range (15–30°C)."},
    "Dissolution Rate (%)": {"min": 80, "message": "dissolution too low. Improve to ≥ 80%."},
    "Disintegration Time (minutes)": {"max": 30, "message": "disintegration too slow. Should be < 30 minutes."},
    "Impurity Level (%)": {"max": 2, "message": "impurity too high. Reduce to ≤ 2%."},
    "Assay Purity (%)": {"min": 90, "message": "purity too low. Increase to ≥ 90%."},
    "Warning Labels Present": {"min": 1, "message": "warning labels missing. Add required safety labels."}
}

# ===============================
# 11. User Input Prediction
# ===============================
print("\n💊 Medicine Safety Prediction System\n")

user_disease = input("Enter Disease/Use Case: ")
user_tablet = input("Enter Tablet (Active Ingredient): ")

# Take numeric feature inputs
user_features = {}
for col in numeric_cols:
    try:
        val = float(input(f"Enter value for '{col}': "))
    except:
        val = 0.0
    user_features[col] = val

# Build input row
input_data = {
    "Active Ingredient": user_tablet,
    "Disease/Use Case": user_disease
}
for col in numeric_cols:
    input_data[col] = user_features[col]

input_df = pd.DataFrame([input_data])

# ===============================
# 12. Prediction + Detailed Suggestions
# ===============================
pred = model.predict(input_df)[0]
result = le.inverse_transform([pred])[0]

print("\n✅ Prediction Result:", result)

if result == "Not Safe":
    print("\n⚠️ Suggested Improvements:")
    for feature, rule in SAFETY_RULES.items():
        value = input_data[feature]

        if "min" in rule and value < rule["min"]:
            print(f"- {feature} is {value} ({rule['message']})")

        if "max" in rule and value > rule["max"]:
            print(f"- {feature} is {value} ({rule['message']})")

        if "range" in rule:
            low, high = rule["range"]
            if not (low <= value <= high):
                print(f"- {feature} is {value} ({rule['message']})")
