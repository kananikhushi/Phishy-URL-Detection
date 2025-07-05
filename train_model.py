import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========== Load Dataset ==========
df = pd.read_csv("malicious_phish.csv")
df = df.dropna(subset=['url'])
df['url'] = df['url'].astype(str)
df['label'] = df['type'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)

# Sample smaller but representative data
df = df.sample(n=3000, random_state=42)

# ========== Feature Extraction ==========
def extract_features(url):
    return {
        'url_length': len(url),
        'has_https': int('https' in url.lower()),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_digits': sum(c.isdigit() for c in url),
        'has_suspicious_words': int(any(word in url.lower() for word in ['login', 'verify', 'update', 'secure', 'account', 'free'])),
        'count_special_chars': sum(c in url for c in ['@', '=', '?', '&']),
    }

X = pd.DataFrame(df['url'].apply(extract_features).tolist())
y = df['label']

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== Model ==========
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# ========== Evaluation ==========
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ========== Save ==========
joblib.dump(model, 'rf_model.joblib')
joblib.dump(X.columns.tolist(), 'rf_features.joblib')
print("\n✅ Random Forest model saved.")
