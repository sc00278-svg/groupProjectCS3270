import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

#Get rid of weird scientific notation stuff
pd.set_option('display.float_format', '{:.2f}'.format)


# Load Data
df = pd.read_csv("steam_games_data_adjusted.csv")

# Clean
df["Primary_Genre"] = df["Primary_Genre"].fillna("Unknown")
df["All_Tags"] = df["All_Tags"].fillna("")
df["Price_USD"] = df["Price_USD"].fillna(0)



# Features and targets
X = df[["Primary_Genre", "All_Tags", "Price_USD"]]

y_owners_raw = df["Estimated_Owners"]
y_score = df["Review_Score_Pct"]

# log transform owners (trying to mitigate skew)
y_owners = np.log1p(y_owners_raw)



# Vectorizing tags (for speed I guess)
tag_vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split(";"),
    token_pattern=None,
    min_df=3,
    max_df=0.9
)



# Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("genre", OneHotEncoder(handle_unknown="ignore"), ["Primary_Genre"]),
        ("tags", tag_vectorizer, "All_Tags"),
        ("price", FunctionTransformer(np.log1p), ["Price_USD"])
    ]
)



# One for score and one for owners
owners_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=5.0))
])

score_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=5.0))
])



# Train/Test split 80/20
X_train, X_test, y_owners_train, y_owners_test = train_test_split(
    X, y_owners, test_size=0.2, random_state=42
)

_, _, y_score_train, y_score_test = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)



# Train
owners_model.fit(X_train, y_owners_train)
score_model.fit(X_train, y_score_train)



# Test
owners_pred_log = owners_model.predict(X_test)
score_pred = score_model.predict(X_test)

owners_pred = np.expm1(owners_pred_log)
y_owners_true = np.expm1(y_owners_test)


# R2 and Mean Abs. Error
print("\nEstimated Owners")
print("MAE:", mean_absolute_error(y_owners_true, owners_pred))
print("R2:", r2_score(y_owners_true, owners_pred))

print("\nReview Score")
print("MAE:", mean_absolute_error(y_score_test, score_pred))
print("R2:", r2_score(y_score_test, score_pred))



# Showing the actual predictions
results = X_test.copy()

results["Actual_Owners"] = y_owners_true.values
results["Predicted_Owners"] = owners_pred

results["Actual_Score"] = y_score_test.values
results["Predicted_Score"] = score_pred


# Some more samples
results["Owner_Error"] = abs(results["Actual_Owners"] - results["Predicted_Owners"])
results = results.sort_values("Owner_Error", ascending=False)


print("\n\nSamples\n")

print(
    results[[
        "Primary_Genre",
        "Price_USD",
        "Actual_Owners",
        "Predicted_Owners",
        "Actual_Score",
        "Predicted_Score"
    ]].head(15).to_string(index=False)
)

