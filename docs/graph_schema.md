Graph Schema - Fintech merchant Risk GNN

This document defines the node types, edge types, features and learning objectives for the merchant risk graph used in the project

## 1. Node Types

### 1.1 User

- **id**: `user_id` (string, e.g. `"user_42"`)
- **features**:
  - `age_bucket` (categorical: `"18-24"`, `"25-34"`, `"35-44"`, `"45-60"`)
  - `state` (categorical: `"KA"`, `"MH"`, `"DL"`, etc.)
  - `user_risk_score` (float in [0, 1], initially a heuristic or 0.0)

### 1.2 Merchant

- **id**: `merchant_id` (string, e.g. `"m_10"`)
- **features**:
  - `category` (categorical: `"Food"`, `"Electronics"`, `"Groceries"`, etc.)
  - `city` (categorical: `"Bangalore"`, `"Mumbai"`, etc.)
  - `avg_txn_amount` (float, average transaction amount for this merchant)
  - `is_fraud_merchant` (label: 0/1, used for training / evaluation)

### 1.3 Device

- **id**: `device_id` (string, e.g. `"device_7"`)
- **features**:
  - `os` (categorical: `"Android"`, `"iOS"`)
  - `device_risk_score` (float in [0, 1], initially 0.0 placeholder)



## 2. Edge Types

### 2.1 User → Merchant (Transaction Edge)

- **from**: `user_id`
- **to**: `merchant_id`
- **features**:
  - `amount` (float)
  - `timestamp` (ISO string)
  - `is_fraud` (0/1, fraud label at transaction level)

### 2.2 User → Device (Login Edge)

- **from**: `user_id`
- **to**: `device_id`
- **features**:
  - `login_freq` (float, to be derived later)
  - `last_seen` (timestamp)

### 2.3 Merchant → Merchant (Shared Attribute Edge)

- **from**: `merchant_id`
- **to**: `merchant_id`
- **features**:
  - `shared_city` (0/1)
  - `shared_category` (0/1)

These edges help us detect clusters of merchants that share many properties and users.


## 3. Learning Objectives

### 3.1 Node Classification — Merchant Risk

- **Task**: Predict `is_fraud_merchant` for merchant nodes.
- **Type**: Binary classification.
- **Input**:
  - Merchant’s own features.
  - Aggregated signals from connected users, devices, and neighboring merchants via GNN.
- **Output**:
  - Fraud probability `P(fraud | merchant)` in [0, 1].
  - Or a hard label `{0, 1}` after thresholding.

### 3.2Link Prediction — Laundering Rings

- **Task**: Predict if an edge between two merchants is suspicious (potential laundering / synthetic network).
- **Type**: Link prediction / anomaly detection.
- **Usage**:
  - Detect emerging fraud rings even before explicit labels exist.


## 4. Justification

- Users, merchants, and devices naturally form a heterogeneous graph in payment systems.
- Fraud patterns often involve:
  - The same users transacting with multiple risky merchants.
  - Shared devices / IPs between seemingly unrelated accounts.
  - Night-time, low-amount, high-frequency transaction patterns.
- Tabular models on individual transactions miss these relational patterns.
- By using a GNN:
  - Each merchant node can aggregate information from its neighborhood (users, devices, other merchants).
  - We can assign a more accurate merchant risk score that reflects both its own behavior and the behavior of its neighbors.
