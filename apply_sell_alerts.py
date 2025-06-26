#!/usr/bin/env python3
"""
Script to apply manual sell alerts to smart_analytics.json and trigger cleanup.
"""

import sys
import os
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading import (
    SMART_ANALYTICS_FILE,
    TRADING_HISTORY_FILE,
    update_stock_sell_status,
    cleanup_sold_stocks_from_analytics,
    initialize_smart_analytics, # Ensure file exists for setup
    initialize_trading_history, # Ensure file exists for setup
    save_recommendations_to_smart_analytics # To initially add the stocks
)

def setup_stocks_as_bought_for_test():
    """Ensures the stocks from the sell alerts are marked as 'bought' in smart_analytics.json."""
    print("🧪 Setting up stocks (IDFCFIRSTB, NHPC, PNB) as 'bought' in smart_analytics.json for testing...")
    initialize_smart_analytics()
    initialize_trading_history()

    # Stocks to add/update with their buy details from Slack alerts
    stocks_to_buy = [
        {
            "symbol": "IDFCFIRSTB",
            "recommended_price": 70.35, # Using buy price as recommended to simulate
            "quantity": 1,
            "ai_reason": "Simulated prior purchase for testing",
            "technical_score": 70, # Dummy score
            "stop_loss_price": 68.0, # Dummy SL
            "stop_loss_pct": 3.0, # Dummy SL %
            "signals": ["Simulated buy"],
        },
        {
            "symbol": "NHPC",
            "recommended_price": 81.95,
            "quantity": 1,
            "ai_reason": "Simulated prior purchase for testing",
            "technical_score": 75,
            "stop_loss_price": 79.0,
            "stop_loss_pct": 3.0,
            "signals": ["Simulated buy"],
        },
        {
            "symbol": "PNB",
            "recommended_price": 103.62,
            "quantity": 1,
            "ai_reason": "Simulated prior purchase for testing",
            "technical_score": 80,
            "stop_loss_price": 100.0, # Dummy SL
            "stop_loss_pct": 3.0, # Dummy SL %
            "signals": ["Simulated buy"],
        }
    ]

    # Read existing smart_analytics data
    smart_data = {}
    if os.path.exists(SMART_ANALYTICS_FILE):
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            smart_data = json.load(f)
    existing_recs = smart_data.get("current_recommendations", [])

    # Filter out existing entries for these symbols if they are 'not_bought'
    # This prevents duplicates if the script is run multiple times
    cleaned_existing_recs = [rec for rec in existing_recs if rec["symbol"] not in [s["symbol"] for s in stocks_to_buy] or rec.get("status") == "bought"]

    # Append new entries for the test stocks
    for i, stock_data in enumerate(stocks_to_buy):
        # Create a unique ID for the test scenario
        unique_id = f"test_bought_{stock_data['symbol']}_{datetime.now().strftime("%H%M%S%f")}"
        cleaned_existing_recs.append({
            "id": unique_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": stock_data["symbol"],
            "recommended_price": stock_data["recommended_price"],
            "quantity": stock_data["quantity"],
            "ai_reason": stock_data["ai_reason"],
            "technical_score": stock_data["technical_score"],
            "stop_loss_price": stock_data["stop_loss_price"],
            "stop_loss_pct": stock_data["stop_loss_pct"],
            "signals": stock_data["signals"],
            "recommendation_order": len(cleaned_existing_recs) + i + 1,
            "status": "bought", # Explicitly set to bought for this test setup
            "actual_buy_price": stock_data["recommended_price"], # Use recommended as actual buy for test
            "actual_quantity": stock_data["quantity"],
            "buy_timestamp": datetime.now().isoformat(),
            "sell_price": None,
            "sell_timestamp": None,
            "actual_pl": None,
            "notes": "Setup for sell test"
        })
    
    smart_data["current_recommendations"] = cleaned_existing_recs
    smart_data["last_updated"] = datetime.now().isoformat()
    
    with open(SMART_ANALYTICS_FILE, 'w') as f:
        json.dump(smart_data, f, indent=4)
    print("✅ Stocks (IDFCFIRSTB, NHPC, PNB) successfully set as 'bought' for testing.")

def apply_sell_updates(sell_events):
    """Applies a list of sell events to the smart_analytics.json file.
    Each event should be a dict with 'symbol', 'sell_price', 'sell_reason'.
    """
    print("🚀 Applying sell updates from provided alerts...")
    
    # Ensure files are initialized if they don't exist
    initialize_smart_analytics()
    initialize_trading_history()

    # Before applying, let's read the current smart_analytics to confirm their presence
    try:
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            current_smart_data = json.load(f)
        current_symbols = {rec["symbol"] for rec in current_smart_data.get("current_recommendations", []) if rec.get("status") == "bought"}
        print(f"Initial 'bought' stocks in {SMART_ANALYTICS_FILE}: {current_symbols}")
    except Exception as e:
        print(f"Error reading {SMART_ANALYTICS_FILE} before updates: {e}")
        current_symbols = set()

    for event in sell_events:
        symbol = event['symbol']
        sell_price = event['sell_price']
        sell_reason = event['sell_reason']
        
        # Check if the stock is currently marked as 'bought' before trying to sell
        if symbol in current_symbols:
            print(f"Updating {symbol} to sold at ₹{sell_price:.2f} due to: {sell_reason}")
            update_stock_sell_status(symbol, sell_price, sell_reason)
        else:
            print(f"Skipping {symbol}: Not found or not marked as 'bought' in {SMART_ANALYTICS_FILE}.")
            
    print("\n--- Running cleanup to move sold stocks to trading_history.json ---")
    cleanup_sold_stocks_from_analytics()
    print("✅ Sell updates applied and cleanup performed!")

    # Verify final states
    print("\n--- Final smart_analytics.json content ---")
    try:
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            final_smart_content = json.load(f)
        print(json.dumps(final_smart_content, indent=4))
    except Exception as e:
        print(f"Error reading final {SMART_ANALYTICS_FILE}: {e}")

    print("\n--- Final trading_history.json content ---")
    try:
        with open(TRADING_HISTORY_FILE, 'r') as f:
            final_history_content = json.load(f)
        print(json.dumps(final_history_content, indent=4))
    except Exception as e:
        print(f"Error reading final {TRADING_HISTORY_FILE}: {e}")


if __name__ == "__main__":
    # 0. Setup: Ensure these stocks are in 'bought' status first
    setup_stocks_as_bought_for_test()

    # 1. Define the sell events from your Slack alerts
    sell_events_to_apply = [
        {
            "symbol": "IDFCFIRSTB",
            "sell_price": 69.73,
            "sell_reason": "Consistent downward trend, technical breakdown. Late in trading day, 2% profit target unlikely."
        },
        {
            "symbol": "NHPC",
            "sell_price": 81.01,
            "sell_reason": "Clear technical breakdown with sustained downward momentum, hitting new lows. Limited time remaining."
        },
        {
            "symbol": "PNB",
            "sell_price": 103.11,
            "sell_reason": "Late in trading day, downward trend hitting recent lows. Unlikely to reach 2% profit target."
        }
    ]
    
    apply_sell_updates(sell_events_to_apply) 