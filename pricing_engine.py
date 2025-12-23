import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
df = pd.read_csv("listings.csv")

# --------------------------------------------------
# 2. BASIC CLEANING
# --------------------------------------------------
# Remove rows with missing or zero price
df = df[df['price'] > 0]

# Fill missing values
df['number_of_reviews'] = df['number_of_reviews'].fillna(0)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['availability_365'] = df['availability_365'].fillna(0)

# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

# Convert last_review to datetime
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['month'] = df['last_review'].dt.month

# Season mapping
def get_season(month):
    if month in [6, 7, 8, 12]:
        return 'Peak'
    elif month in [1, 2, 3]:
        return 'OffPeak'
    else:
        return 'Shoulder'

df['season'] = df['month'].apply(lambda x: get_season(x) if pd.notnull(x) else 'Unknown')

# Listing quality score
df['quality_score'] = (
    df['number_of_reviews'] * 0.6 +
    df['reviews_per_month'] * 0.4
)

# --------------------------------------------------
# 4. SELECT MODEL FEATURES
# --------------------------------------------------
features = [
    'latitude',
    'longitude',
    'availability_365',
    'quality_score',
    'room_type',
    'neighbourhood',
    'season'
]

df_model = df[features + ['price']]

# --------------------------------------------------
# 5. ENCODE CATEGORICAL VARIABLES
# --------------------------------------------------
df_model = pd.get_dummies(
    df_model,
    columns=['room_type', 'neighbourhood', 'season'],
    drop_first=True
)

# --------------------------------------------------
# 6. TRAIN REGRESSION MODEL
# --------------------------------------------------
X = df_model.drop('price', axis=1)
y = df_model['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------
predictions = model.predict(X_test)

print("R2 Score:", round(r2_score(y_test, predictions), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))

# --------------------------------------------------
# 8. PRICE RECOMMENDATION FUNCTION
# --------------------------------------------------
def suggest_price(input_data):
    """
    input_data: pandas DataFrame with same columns as X
    """
    return model.predict(input_data)

# --------------------------------------------------
# 9. EXPORT CLEAN DATA FOR POWER BI
# --------------------------------------------------
df.to_csv("airbnb_powerbi.csv", index=False)

