from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re

MAX_LEN = 5000

app = Flask(__name__)

model = joblib.load("model/model.pkl")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_LEN]

def prepare_row(tx: dict) -> dict:
    items = tx.get("items") or []

    names = []
    prices = []

    for it in items:
        n = it.get("name")
        if n:
            names.append(str(n))

        p = it.get("price")
        if isinstance(p, (int, float)):
            prices.append(float(p))

    receipt_text = normalize_text(" ".join(names))

    terminal_desc = tx.get("terminal_description", "")
    city = tx.get("city", "")
    context_text = normalize_text(f"{terminal_desc} {city}")

    try:
        amount = float(tx.get("amount", 0))
    except Exception:
        raise ValueError("amount is not numeric")

    if amount < 0:
        raise ValueError("negative amount")

    if prices:
        mn = min(prices)
        sum_ratio = sum(prices) / (amount + 1e-5)
    else:
        mn = 0.0
        sum_ratio = 0.0

    return {
        "amount": amount,
        "item_count": len(items),
        "receipt_text": receipt_text,
        "context_text": context_text,
        "min": mn,
        "sum_ratio": sum_ratio,
        "l2_rec": len(receipt_text)
    }

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "bad json"}), 400

    try:
        row = prepare_row(data)
        X = pd.DataFrame([row])
        probas = model.predict_proba(X)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "inference failed"}), 500

    p = probas[0]
    idx = int(np.argmax(p))

    return jsonify({
        "transaction_id": data.get("transaction_id"),
        "predicted_mcc": int(model.classes_[idx]),
        "confidence": round(float(p[idx]), 3)
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    payload = request.get_json(silent=True)
    if not payload or "transactions" not in payload:
        return jsonify({"error": "bad json"}), 400

    rows = []
    txs = payload["transactions"]

    try:
        for tx in txs:
            rows.append(prepare_row(tx))
        X = pd.DataFrame(rows)
        probas = model.predict_proba(X)
    except Exception:
        return jsonify({"error": "batch inference failed"}), 500

    res = []
    for i, tx in enumerate(txs):
        p = probas[i]
        idx = int(np.argmax(p))
        res.append({
            "transaction_id": tx.get("transaction_id"),
            "predicted_mcc": int(model.classes_[idx]),
            "confidence": round(float(p[idx]), 3)
        })

    return jsonify({"predictions": res})


@app.route("/model/info")
def model_info():
    return jsonify({
        "name": "mcc-classifier",
        "version": "2.2"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
