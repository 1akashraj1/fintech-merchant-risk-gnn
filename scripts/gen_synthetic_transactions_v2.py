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




# def timestamp(n_days:int = 30, night_bias: bool = False):
def timestamp(n_days:int = 90):
    '''
    Generate random timestamps in last n_days.
    # If night_bias=True -> more likely to be at night (e.g. 11pm-4am).
    
    '''

    now = datetime.now()
    days_back = np.random.randint(0,n_days)
    base_date = now - timedelta(days=(days_back))
    hour = np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    # if hour in [22,23,0,1,2,3,4]:
    #     night_bias = True
    # else:
    #     night_bias = False

    minute = np.random.randint(0,60)
    second = np.random.randint(0,60)

    return base_date.replace(hour=hour, minute=minute, second=second)



# def generate_ring(merchants,mule_users:list):
#     ring = {}
#     pref_merch_size = (np.random.randint(5,10)*n_merchants)//100
#     fix_preferred_merchants = merchants.sample(pref_merch_size,random_state=random_seed)['merchant_id'].tolist()
#     preferred_merchants = fix_preferred_merchants.copy()
#     sample_mule_users = np.random.choice(mule_users, size=5)
#     rest_mule_users = list(set(mule_users) - set(sample_mule_users))
#     while len(rest_mule_users) > 0:
#         sample_merchants = np.random.choice(preferred_merchants, size = np.random.randint(4,6))
#         preferred_merchants = list(set(preferred_merchants) - set(sample_merchants))

#         if len(preferred_merchants) > 0:
#             for i in range(len(sample_mule_users)):
#                 ring[sample_mule_users[i]] = sample_merchants
#         else:
#             sample_merchants = np.random.choice(fix_preferred_merchants, size = np.random.randint(4,6))
#             preferred_merchants = list(set(fix_preferred_merchants) - set(sample_merchants))
#             for i in range(len(sample_mule_users)):
#                 ring[sample_mule_users[i]] = sample_merchants
            
#         sample_mule_users = np.random.choice(rest_mule_users, size=np.random.randint(5,7))
#         rest_mule_users = list(set(rest_mule_users) - set(sample_mule_users))


#     for i in range(len(sample_mule_users)):
#             ring[sample_mule_users[i]] = sample_merchants


#     print(ring)
#     return ring



def generate_fraud_rings(users, merchants, num_rings=30):
    """
    Creates isolated fraud rings.
    Each ring has:
      - 6 merchants
      - 5 mule users
    """
    all_users = users["user_id"].tolist()
    all_merchants = merchants["merchant_id"].tolist()

    random.shuffle(all_users)
    random.shuffle(all_merchants)

    rings = []
    user_to_ring = {}

    user_idx = 0
    merch_idx = 0

    for ring_id in range(num_rings):
        ring_users = set(all_users[user_idx:user_idx + 5])
        ring_merchants = set(all_merchants[merch_idx:merch_idx + 6])

        user_idx += 5
        merch_idx += 6

        ring = {
            "ring_id": ring_id,
            "users": ring_users,
            "merchants": ring_merchants
        }

        rings.append(ring)

        for u in ring_users:
            user_to_ring[u] = ring

    return rings, user_to_ring



# def generate_transactions(users: pd.DataFrame, merchants:pd.DataFrame,fraud_merchants: list):
def generate_transactions(users, merchants, user_to_ring, fraud_merchants, normal_merchants):


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

    # normal_count = int(n_transactions*0.8)

    for i in range(n_transactions):
        user = random.choice(user_ids)

        if user in user_to_ring:
    # Mule user → ONLY ring merchants
            ring = user_to_ring[user]
            merchant = random.choice(list(ring["merchants"]))
        else:
            # Normal user → mostly normal merchants
            if random.random() < 0.97:   # 97% of the time
                merchant = random.choice(normal_merchants)
            else:                        # 3% rare exposure
                merchant = random.choice(fraud_merchants)

        

        merch_row = merchants.loc[merchants["merchant_id"] == merchant].iloc[0]
        base_mu = np.log(merch_row["avg_trans_amount"] + 1)

        amount = np.random.lognormal(mean=base_mu, sigma=0.4)
        amount = max(10.0, min(amount, 7000.0))  # clamp

        ts = timestamp()
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


def generate_fraud_behavior(x,y, trans_df:pd.DataFrame):
    # this function will decide if the merchant is fraud or not,
    # if the merchant shares >= x users with >= y merchants then the merchant is fraud
    
    is_fraud = {}
    connection_count = {}
    merch_user_connection = {}
    for i,row in trans_df.iterrows():
        if row['merchant_id'] in merch_user_connection:
            merch_user_connection[row['merchant_id']].add(row['user_id'])
        else:
            merch_user_connection[row['merchant_id']] = {row['user_id']}
            connection_count[row['merchant_id']] = 0
    
    merch_list = list(merch_user_connection.keys())
    merch_count = len(merch_list)
    for i in range(merch_count):
        count = 0
        for j in range(i+1, merch_count):
            share_user = len(merch_user_connection[merch_list[i]] & merch_user_connection[merch_list[j]])

            if share_user >= x:
                connection_count[merch_list[i]] += 1
                connection_count[merch_list[j]] += 1
                count += 1
            if count >= y:
                is_fraud[merch_list[i]] = 1
                break
        
    for merch in connection_count.keys():
        if connection_count[merch] >= y:
            is_fraud[merch] = 1
        else:
            is_fraud[merch] = 0


    fraud_merch = pd.DataFrame(list(is_fraud.items()),columns=['merchant_id','is_fraud']).sort_values('merchant_id')  

    return fraud_merch
    


def main():
    make_dirs()

    # Generating Users
    users = generate_users()

    # Generating Merchats
    merchants = generate_merchants()

    # Generating mule users
    rings, user_to_ring = generate_fraud_rings(
    users=users,
    merchants=merchants,
    num_rings=30
    )

    # After creating rings
    fraud_merchants = set()
    for ring in rings:
        fraud_merchants.update(ring["merchants"])

    all_merchants = set(merchants["merchant_id"].tolist())
    normal_merchants = list(all_merchants - fraud_merchants)
    fraud_merchants = list(fraud_merchants)


    # Generating transactions

    trans = generate_transactions(
    users=users,
    merchants=merchants,
    user_to_ring=user_to_ring,
    fraud_merchants=fraud_merchants,
    normal_merchants=normal_merchants
    )

    #picking fraud merchants
    # fraud_merchants = pick_fraud_merchants(merchants)
    # merchants["is_fraud_merchant"] = merchants["merchant_id"].isin([]).astype("int32")



    fraud_merch = generate_fraud_behavior(2,3,trans)


    merchants = merchants.merge(fraud_merch,how='left',on='merchant_id')
    merchants['is_fraud'] = merchants['is_fraud'].astype('Int64')

    # Creating CSVs
    users.to_csv('data/raw/users.csv', index = False)
    merchants.to_csv('data/raw/merchants.csv', index=False)
    trans.to_csv('data/raw/transactions.csv',index=False)


if __name__ == "__main__":
    main()

