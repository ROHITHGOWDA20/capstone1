import streamlit as st
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
# 1. Load dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("medicine_with_expiry.csv")

    # Clean columns
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("Â", "", regex=False)
        .str.replace("°", "C", regex=False)
    )
    df = df.rename(columns={"Storage Temperature (CC)": "Storage Temperature (C)"})

    # Handle missing
    df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
    df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["Safe/Not Safe"])

    # Numeric features
    numeric_cols = [
        "Days Until Expiry",
        "Storage Temperature (C)",
        "Dissolution Rate (%)",
        "Disintegration Time (minutes)",
        "Impurity Level (%)",
        "Assay Purity (%)",
        "Warning Labels Present"
    ]

    if df["Warning Labels Present"].dtype == "object":
        df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0})

    X = df[["Active Ingredient", "Disease/Use Case"] + numeric_cols]

    return df, X, y, le, numeric_cols

df, X, y, le, numeric_cols = load_data()

# ===============================
# 2. Preprocessor + Model
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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# ===============================
# 3. Streamlit UI
# ===============================
st.title("💊 Medicine Safety Prediction System")

# User inputs
user_disease = st.text_input("Enter Disease/Use Case")
user_tablet = st.text_input("Enter Tablet (Active Ingredient)")

user_features = {}
for col in numeric_cols:
    val = st.number_input(f"Enter value for {col}", value=0.0)
    user_features[col] = val

if st.button("Predict Safety"):
    # Build input row
    input_data = {
        "Active Ingredient": user_tablet,
        "Disease/Use Case": user_disease
    }
    for col in numeric_cols:
        input_data[col] = user_features[col]

    input_df = pd.DataFrame([input_data])

    pred = model.predict(input_df)[0]
    result = le.inverse_transform([pred])[0]

    st.subheader("✅ Prediction Result")
    st.write(result)

    # ===============================
    # 4. Safety Rules
    # ===============================
    SAFETY_RULES = {
        "Days Until Expiry": {"min": 30, "message": "expiry is too close. Suggest ≥ 30 days shelf life."},
        "Storage Temperature (C)": {"range": (15, 30), "message": "temperature out of safe range (15–30°C)."},
        "Dissolution Rate (%)": {"min": 80, "message": "dissolution too low. Improve to ≥ 80%."},
        "Disintegration Time (minutes)": {"max": 30, "message": "disintegration too slow. Should be < 30 minutes."},
        "Impurity Level (%)": {"max": 2, "message": "impurity too high. Reduce to ≤ 2%."},
        "Assay Purity (%)": {"min": 90, "message": "purity too low. Increase to ≥ 90%."},
        "Warning Labels Present": {"min": 1, "message": "warning labels missing. Add required safety labels."}
    }

    if result == "Not Safe":
        st.subheader("⚠️ Suggested Improvements")
        for feature, rule in SAFETY_RULES.items():
            value = input_data[feature]
            if "min" in rule and value < rule["min"]:
                st.write(f"- {feature} is {value} ({rule['message']})")
            if "max" in rule and value > rule["max"]:
                st.write(f"- {feature} is {value} ({rule['message']})")
            if "range" in rule:
                low, high = rule["range"]
                if not (low <= value <= high):
                    st.write(f"- {feature} is {value} ({rule['message']})")
