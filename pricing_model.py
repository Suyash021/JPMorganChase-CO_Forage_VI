import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import logging

# Disable Prophet's logging to keep the output clean
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Load data from Task 1
def load_and_train_model():
    df = pd.read_csv('task_1/Nat_Gas.csv')
    df['Dates'] = pd.to_datetime(df['Dates'], format='mixed')
    df = df.rename(columns={'Dates': 'ds', 'Prices': 'y'})

    model = Prophet()
    model.fit(df)
    return model, df

# Initialize model and data
model, df_historical = load_and_train_model()

def get_forecasted_price(date_str):
    """
    Predicts the gas price for a given date string using the Prophet model.
    """
    input_date = pd.to_datetime(date_str)

    # Check if the date is within historical range or needs forecasting
    # We create a dataframe for prediction for the specific date
    future = pd.DataFrame({'ds': [input_date]})
    forecast = model.predict(future)

    return forecast.iloc[0]['yhat']

def price_contract(injection_dates, injection_volumes, injection_prices,
                   withdrawal_dates, withdrawal_volumes, withdrawal_prices,
                   rate, max_volume, storage_cost):
    """
    Prices a commodity storage contract.

    Args:
        injection_dates: List of strings or datetime objects for injection.
        injection_volumes: List of floats for volumes injected.
        injection_prices: List of floats for purchase prices on injection dates.
        withdrawal_dates: List of strings or datetime objects for withdrawal.
        withdrawal_volumes: List of floats for volumes withdrawn.
        withdrawal_prices: List of floats for sale prices on withdrawal dates.
        rate: Maximum volume per injection/withdrawal transaction.
        max_volume: Maximum storage capacity.
        storage_cost: Monthly cost per unit held in storage.

    Returns:
        The calculated value of the contract.
    """
    # 1. Consolidate and validate transactions chronologically
    injections = []
    for d, v, p in zip(injection_dates, injection_volumes, injection_prices):
        if v > rate:
            raise ValueError(f"Injection volume {v} exceeds max rate {rate} on {d}")
        injections.append({'date': pd.to_datetime(d), 'volume': v, 'price': p, 'type': 'injection'})

    withdrawals = []
    for d, v, p in zip(withdrawal_dates, withdrawal_volumes, withdrawal_prices):
        if v > rate:
            raise ValueError(f"Withdrawal volume {v} exceeds max rate {rate} on {d}")
        withdrawals.append({'date': pd.to_datetime(d), 'volume': -v, 'price': p, 'type': 'withdrawal'})

    all_transactions = sorted(injections + withdrawals, key=lambda x: x['date'])

    # 2. Implement inventory tracking and constraint validation
    inventory = 0.0
    total_revenue = 0.0
    total_injection_cost = 0.0
    total_storage_cost = 0.0

    last_date = None

    for tx in all_transactions:
        # Calculate storage cost for interval before this transaction
        if last_date is not None:
            days = (tx['date'] - last_date).days
            # Monthly rate, assuming 30 days in a month for calculation
            total_storage_cost += inventory * storage_cost * (days / 30.0)

        # Perform transaction
        inventory += tx['volume']

        # Check constraints
        if inventory > max_volume:
            raise ValueError(f"Storage capacity exceeded: {inventory} > {max_volume} on {tx['date']}")
        if inventory < 0:
            raise ValueError(f"Inventory cannot be negative: {inventory} on {tx['date']}")

        # Track cash flows
        if tx['type'] == 'injection':
            total_injection_cost += abs(tx['volume']) * tx['price']
        else:
            total_revenue += abs(tx['volume']) * tx['price']

        last_date = tx['date']

    return total_revenue - total_injection_cost - total_storage_cost

if __name__ == "__main__":
    # Test Case 1: Standard profitable scenario
    # Inject in Summer (low price), Withdraw in Winter (high price)
    inj_dates = ['2024-06-01', '2024-07-01']
    inj_vols = [1000, 1000]
    inj_prices = [get_forecasted_price(d) for d in inj_dates]

    with_dates = ['2024-12-01', '2025-01-01']
    with_vols = [1000, 1000]
    with_prices = [get_forecasted_price(d) for d in with_dates]

    rate = 1500
    max_vol = 5000
    storage_cost = 0.1 # $0.1 per unit per month

    print(f"Test Case 1: Standard Scenario")
    print(f"Injection Prices: {inj_prices}")
    print(f"Withdrawal Prices: {with_prices}")
    try:
        value = price_contract(inj_dates, inj_vols, inj_prices,
                               with_dates, with_vols, with_prices,
                               rate, max_vol, storage_cost)
        print(f"Contract Value: ${value:.2f}")
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 30)

    # Test Case 2: Max volume constraint violation
    print(f"Test Case 2: Storage Capacity Violation")
    try:
        price_contract(inj_dates, [3000, 3000], inj_prices,
                       with_dates, with_vols, with_prices,
                       4000, 5000, storage_cost)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("-" * 30)

    # Test Case 3: Rate constraint violation
    print(f"Test Case 3: Rate Violation")
    try:
        price_contract(['2024-06-01'], [2000], [10.0],
                       ['2024-12-01'], [2000], [15.0],
                       1500, 5000, storage_cost)
    except ValueError as e:
        print(f"Caught expected error: {e}")
