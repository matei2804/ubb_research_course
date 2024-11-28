import numpy as np
import pandas as pd
import re
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
from tqdm import tqdm
import os
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

columns = [
    'label',
    'method',
    'user_agent',
    'cache_control',
    'pragma',
    'accept',
    'accept_encoding',
    'accept_charset',
    'accept_language',
    'host',
    'cookie',
    'content_type',
    'connection',
    'content_length',
    'payload',
    'other',
    'url'
]

data = pd.read_csv(
    'csic_database.csv',
    header=None,
    names=columns,
    encoding='latin1',
    low_memory=False
)

data = data.dropna(subset=['label'])

label_mapping = {'Normal': 0, 'Anomalous': 1}
data['label'] = data['label'].map(label_mapping)

data = data.dropna(subset=['label'])

data['label'] = data['label'].astype(int)

benign_samples = data[data['label'] == 0]
malicious_samples = data[data['label'] == 1].sample(n=1000, random_state=42)
data = pd.concat([benign_samples, malicious_samples])

X = data['payload'].fillna('').astype(str)
y = data['label']

def preprocess_payload(payload):
    decoded = urllib.parse.unquote(payload)
    cleaned = decoded.strip()
    cleaned = cleaned.lower()
    return cleaned

X = X.apply(preprocess_payload)

X_train_payloads, X_test_payloads, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), analyzer='char_wb')
X_train_tfidf = vectorizer.fit_transform(X_train_payloads)
X_test_tfidf = vectorizer.transform(X_test_payloads)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(zip(np.unique(y_train), class_weights))

input_dim = X_train_tfidf.shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))

encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

benign_indices = y_train == 0
X_train_benign = X_train_tfidf[benign_indices]

X_train_benign_dense = X_train_benign.toarray()
X_test_tfidf_dense = X_test_tfidf.toarray()

autoencoder.fit(
    X_train_benign_dense,
    X_train_benign_dense,
    epochs=15,
    batch_size=64,
    shuffle=True,
    validation_data=(X_test_tfidf_dense, X_test_tfidf_dense),
    callbacks=[early_stopping]
)

reconstructions = autoencoder.predict(X_train_benign_dense)
mse = np.mean(np.power(X_train_benign_dense - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)
print(f"Threshold for anomaly detection set at: {threshold}")

reconstructions_test = autoencoder.predict(X_test_tfidf_dense)
mse_test = np.mean(np.power(X_test_tfidf_dense - reconstructions_test, 2), axis=1)

def anomaly_based_detection_batch(mse_values, threshold):
    return [1 if mse > threshold else 0 for mse in mse_values]

signatures = [
    r"union.*select", r"or\s+1=1", r"drop\s+table", r"insert\s+into", r"select\s+\*", r"--", r"';",
    r"<script>", r"onerror", r"onload", r"alert\(", r"document\.cookie",
    r";\s*/bin", r"`", r"\|\|", r"&&",
    r"(\.\./|\.\.\\)+", r"/etc/passwd", r"c:\\windows",
    r"eval\(", r"base64_decode", r"\/\*.*\*\/", r"benchmark\(", r"sleep\(",
]

def signature_based_detection_batch(payloads):
    results = []
    pattern = re.compile('|'.join(signatures), re.IGNORECASE)
    for payload in tqdm(payloads, desc="Signature-Based Detection"):
        if pattern.search(payload):
            results.append(1)
        else:
            results.append(0)
    return results

def parse_payload(payload):
    params = {}
    pairs = payload.split('&')
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            params[key.strip()] = value.strip()
    return params

def specification_based_detection_batch(payloads):
    results = []
    for payload in tqdm(payloads, desc="Specification-Based Detection"):
        params = parse_payload(payload)
        violation = False
        expected_params = {
            'id': r'^\d+$',
            'price': r'^\d+(\.\d{1,2})?$',
            'email': r'^\S+@\S+\.\S+$',
            'quantity': r'^\d+$',
            'username': r'^[a-zA-Z0-9_]{3,30}$',
            'password': r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$',
        }
        for param, pattern in expected_params.items():
            if param in params:
                if not re.match(pattern, params[param]):
                    violation = True
                    break
        results.append(1 if violation else 0)
    return results

rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

joblib.dump(rf_classifier, 'rf_classifier.pkl')
autoencoder.save('autoencoder.h5')

joblib.dump(vectorizer, 'vectorizer.pkl')

np.save('anomaly_threshold.npy', threshold)

def hybrid_detection_batch(X_test_payloads, X_test_tfidf, mse_test, threshold):
    sig_results = signature_based_detection_batch(X_test_payloads)
    ano_results = anomaly_based_detection_batch(mse_test, threshold)
    spec_results = specification_based_detection_batch(X_test_payloads)
    rf_results = rf_classifier.predict(X_test_tfidf).tolist()

    hybrid_results = []
    for sig, ano, spec, rf in zip(sig_results, ano_results, spec_results, rf_results):
        votes = [sig, ano, spec, rf]
        if sum(votes) >= 2:
            hybrid_results.append(1)
        else:
            hybrid_results.append(0)
    return hybrid_results

sig_predictions = signature_based_detection_batch(X_test_payloads)
ano_predictions = anomaly_based_detection_batch(mse_test, threshold)
spec_predictions = specification_based_detection_batch(X_test_payloads)
rf_predictions = rf_classifier.predict(X_test_tfidf)
hybrid_predictions = hybrid_detection_batch(X_test_payloads, X_test_tfidf, mse_test, threshold)

print("Signature-Based Predictions:", np.bincount(sig_predictions))
print("Anomaly-Based Predictions:", np.bincount(ano_predictions))
print("Specification-Based Predictions:", np.bincount(spec_predictions))
print("Random Forest Predictions:", np.bincount(rf_predictions))
print("Hybrid Model Predictions:", np.bincount(hybrid_predictions))

def evaluate_model(y_true, y_pred, model_name):
    print(f'\n{model_name} Classification Report:')
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious'], zero_division=0))

evaluate_model(y_test, sig_predictions, 'Signature-Based Detection')
evaluate_model(y_test, ano_predictions, 'Anomaly-Based Detection')
evaluate_model(y_test, spec_predictions, 'Specification-Based Detection')
evaluate_model(y_test, rf_predictions, 'Random Forest Classifier')
evaluate_model(y_test, hybrid_predictions, 'Hybrid Model Detection')

app = Flask(__name__)

rf_classifier = joblib.load('rf_classifier.pkl')
autoencoder = load_model('autoencoder.h5')
vectorizer = joblib.load('vectorizer.pkl')
threshold = np.load('anomaly_threshold.npy')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    payload_str = data.get('payload', '')
    preprocessed_payload = preprocess_payload(payload_str)
    vectorized_payload = vectorizer.transform([preprocessed_payload])
    vectorized_payload_dense = vectorized_payload.toarray()
    recon = autoencoder.predict(vectorized_payload_dense)
    mse = np.mean(np.power(vectorized_payload_dense - recon, 2), axis=1)
    ano_result = anomaly_based_detection_batch(mse, threshold)[0]
    sig_result = signature_based_detection_batch([preprocessed_payload])[0]
    spec_result = specification_based_detection_batch([preprocessed_payload])[0]
    rf_result = rf_classifier.predict(vectorized_payload)[0]

    votes = [sig_result, ano_result, spec_result, rf_result]
    detection_result = 1 if sum(votes) >= 2 else 0

    return jsonify({'detection_result': int(detection_result)})

if __name__ == '__main__':
    app.run(debug=False)
