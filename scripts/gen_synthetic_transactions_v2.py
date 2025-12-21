import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

n_users = 4000
n_merchants = 2500
n_transactions = 22000
# n_fraud_merchants = 5

states = ['KA', 'UP', 'MH', 'DL', 'TN', 'RJ', 'GJ', 'BH', 'AP']
city = ['Bangalore', 'Noida', 'Mumbai', 'Delhi', 'Chennai', 'Jaipur', 'Surat', 'Patna', 'Vizag']
categories = ["Food", "Electronics", "Fashion", "Groceries", "Travel", "Gaming", "Services", "Sports", "Entertainment"]
age_buckets = ["18-24", "25-34", "35-44", "45-60"]
device_os = ["Android", "IOS"]

def make_dirs():
    os.makedirs(os.path.join("data//raw"), exist_ok=True)


def generate_users():
    user_ids = [f"user_{i}" for i in range(n_users)]
    age_bucket = np.random.choice(age_buckets, size=n_users)
    age_list = [np.random.randint(int(age_range[:2]),int(age_range[3:])) for age_range in age_bucket]

    users = pd.DataFrame({
        "user_id": user_ids,
        "age_bucket": age_bucket,
        "age" : age_list,
        "state": np.random.choice(states,size=n_users),
        "user_risk_score": np.random.uniform(0.0,0.3,size=n_users)

        })
    return users

def generate_merchants():
    merchants = pd.DataFrame({
        "merchant_id":[f"merch_{i}" for i in range(n_merchants)],
        "category": np.random.choice(categories,size=n_merchants),
        "city" : np.random.choice(city,size=n_merchants)       
    })

    base_amounts = {
        "Food":500,
        "Electronics":2200,
        "Fashion":800,
        "Groceries":400,
        "Travel":5000,
        "Gaming":1500,
        "Services":3500,
        "Sports":1000,
        "Entertainment":1200
    }

    merchants['avg_trans_amount'] = merchants['category'].map(base_amounts)


    return merchants

# def pick_fraud_merchants(merchants: pd.DataFrame):
'''
    Removing this function because this function was marking merchants as fraud and creating transaction according to it.
    But in reality this never exists, because no merchants are born fraud, their behavior decides whether they are fraud or not.
    So, now creating new generator which checks merchant's behavior and then label them as fraud
'''
#     fraud_merchants = merchants.sample(n_fraud_merchants, random_state=random_seed)['merchant_id'].tolist()
#     return fraud_merchants


def generate_mule_users(users:pd.DataFrame):
    mule_user_size = (np.random.randint(5,10)*n_users)//100
    mule_users = users.sample(mule_user_size, random_state=random_seed)['user_id'].to_list()

    return mule_users




def timestamp(n_days:int = 30, night_bias: bool = False):
    '''
    Generate random timestamps in last n_days.
    # If night_bias=True -> more likely to be at night (e.g. 11pm-4am).
    
    '''

    now = datetime.now()
    days_back = np.random.randint(0,n_days)
    base_date = now - timedelta(days=(days_back))

    if night_bias:
        hour = np.random.choice([22,23,0,1,2,3,4])
    else:
        hour = np.random.randint(9,22)

    minute = np.random.randint(0,60)
    second = np.random.randint(0,60)

    return base_date.replace(hour=hour, minute=minute, second=second)


# def generate_transactions(users: pd.DataFrame, merchants:pd.DataFrame,fraud_merchants: list):
def generate_transactions(users: pd.DataFrame, merchants:pd.DataFrame,mule_users: list):

    """
    Generate normal + fraudulent transactions.

    Fraud pattern:
      - subset of users (mules) frequently hit limited fraud_merchants
      - smaller, more frequent amounts
      - often at night hours
    """

    user_ids = users['user_id'].tolist()
    merchant_ids = merchants['merchant_id'].tolist()

    transactions = []

    # mule_users = random.sample(user_ids,k=40)

    normal_count = int(n_transactions*0.8)

    for i in range(normal_count):
        user = random.choice(user_ids)
        merchant = random.choice(merchant_ids)

        merch_row = merchants.loc[merchants["merchant_id"] == merchant].iloc[0]
        base_mu = np.log(merch_row["avg_trans_amount"] + 1)

        amount = np.random.lognormal(mean=base_mu, sigma=0.4)
        amount = max(10.0, min(amount, 20000.0))  # clamp

        ts = timestamp(n_days=30, night_bias=False)
        device = f"device_{np.random.randint(0, n_users // 2)}"
        # is_fraud = 0

        transactions.append({
            "trans_id": f"t_{i}",
            "user_id": user,
            "merchant_id": merchant,
            "device_id": device,
            "amount": round(float(amount), 2),
            "timestamp": ts.isoformat()
            # "is_fraud": is_fraud,
        })

    # ---- Fraud ring transactions ----
    # fraud_count = n_transactions - normal_count

    # for j in range(fraud_count):
    #     idx = normal_count + j

    #     user = random.choice(mule_users)
    #     merchant = random.choice(fraud_merchants)

    #     amount = np.random.lognormal(mean=np.log(300), sigma=0.2)
    #     amount = max(50.0, min(amount, 1500.0))

    #     ts = timestamp(n_days=30, night_bias=True)
    #     device = f"device_fraud_{np.random.randint(0, 20)}"
    #     is_fraud = 1

    #     transactions.append({
    #         "trans_id": f"t_{idx}",
    #         "user_id": user,
    #         "merchant_id": merchant,
    #         "device_id": device,
    #         "amount": round(float(amount), 2),
    #         "timestamp": ts.isoformat(),
    #         "is_fraud": is_fraud,
    #     })

    txns_df = pd.DataFrame(transactions)
    return txns_df

def main():
    make_dirs()

    # Generating Users
    users = generate_users()

    # Generating Merchats
    merchants = generate_merchants()

    #picking fraud merchants
    # fraud_merchants = pick_fraud_merchants(merchants)
    merchants["is_fraud_merchant"] = merchants["merchant_id"].isin([]).astype("int32")

    # Generating transactions
    trans = generate_transactions(users=users,merchants=merchants,fraud_merchants=[])

    # Creating CSVs
    users.to_csv('data/raw/users.csv', index = False)
    merchants.to_csv('data/raw/merchants.csv', index=False)
    trans.to_csv('data/raw/transactions.csv',index=False)


if __name__ == "__main__":
    main()

