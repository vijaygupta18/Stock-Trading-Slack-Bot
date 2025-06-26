"""
Enhanced Intraday Trading Bot with Gemini AI Integration

STOCK MANAGEMENT SYSTEM:
📊 smart_analytics.json: Contains CURRENT active recommendations (CLEAN + APPEND mode)
📈 trading_history.json: Contains COMPLETED trades (sold stocks with P&L data)

SMART CLEANUP BEHAVIOR:
- NEW RECOMMENDATIONS: Automatically removes old 'not_bought' stocks
- KEEPS BOUGHT STOCKS: Continues monitoring stocks marked as 'bought'  
- CLEAN + APPEND: Fresh recommendations added to cleaned file
- SOLD STOCKS: Automatically moved from smart_analytics.json to trading_history.json
- CLEAN SEPARATION: Active vs completed trades in different files

SLACK ALERT POLICY:
- 📋 RECOMMENDATIONS: Sent once in morning with selected stocks (so you know what to buy)
- 🔔 SELL ALERTS: Sent immediately when any stock is sold (with P&L details)
- 📊 EOD SUMMARY: Sent only if there were actual sells during the day
- ❌ NO ALERTS: Hold decisions, monitoring checks

This keeps you informed of trading plans and results without spam, and maintains a clean file system.
"""

import os
import schedule
import time
import logging
import json
from datetime import datetime, time as dt_time, timedelta
import pytz
import yfinance as yf
import requests
import google.generativeai as genai 
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# --- Configuration ---
# TODO: Fill these in with your actual credentials and preferences
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "YOUR_SLACK_WEBHOOK_URL") # Replaced by SLACK_BOT_TOKEN
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_DEFAULT_CHANNEL = os.environ.get("SLACK_DEFAULT_CHANNEL")
CAPITAL = 500  
DB_FILE = "trading_data.json" # Or use SQLite: "trading_data.db"
SMART_ANALYTICS_FILE = "smart_analytics.json"  # New file for stock tracking
TRADING_HISTORY_FILE = "trading_history.json"  # Separate file for completed trades
LOG_FILE = "trading_log.log"

# --- Trading Parameters (Global Constants) ---
PROFIT_TARGET_PERCENT = 2.0  # Minimum profit percentage to trigger sell (2%)
MINIMUM_HOLD_TIME_MINUTES = 10  # Minimum time to hold a stock before selling (15 minutes)
MINIMUM_PROFIT_PROTECTION_PERCENT = 2.0  # Don't sell if profit is below this threshold
EARLY_TRADING_LOSS_THRESHOLD = -3.0  # During early hours, only sell if loss exceeds this (-3%)
FALLBACK_PROFIT_THRESHOLD = 2.0  # Fallback profit threshold for AI errors (2%)

# --- Timezone Configuration ---
IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

# --- Global Variables ---
portfolio = {} # Stores current holdings: {"STOCK_SYMBOL": {"quantity": X, "buy_price": Y, "timestamp": Z}}
daily_recommendations = [] # Stores today's top 5 recommendations

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import TA-Lib, use fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using manual calculations")

# --- Gemini API Configuration (Placeholder) ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash') # Or your preferred model

# --- Technical Analysis Functions ---
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
        except:
            pass
    
    # Fallback manual calculation
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (Upper, Middle, Lower)"""
    if TALIB_AVAILABLE:
        try:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return pd.Series(upper, index=prices.index), pd.Series(middle, index=prices.index), pd.Series(lower, index=prices.index)
        except:
            pass
    
    # Fallback manual calculation
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_moving_averages(prices: pd.Series) -> Dict[str, pd.Series]:
    """Calculate various moving averages"""
    return {
        'sma_5': prices.rolling(window=5).mean(),
        'sma_10': prices.rolling(window=10).mean(),
        'sma_20': prices.rolling(window=20).mean(),
        'ema_9': prices.ewm(span=9).mean(),
        'ema_21': prices.ewm(span=21).mean()
    }

def analyze_volume_pattern(volume: pd.Series, prices: pd.Series) -> Dict[str, float]:
    """Analyze volume patterns"""
    avg_volume = volume.rolling(window=20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Price-Volume relationship
    price_change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100
    
    return {
        'volume_ratio': volume_ratio,
        'avg_volume_20d': avg_volume,
        'current_volume': current_volume,
        'price_change_pct': price_change,
        'volume_price_correlation': 1 if (volume_ratio > 1.5 and abs(price_change) > 1) else 0
    }

def calculate_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, float]:
    """Calculate support and resistance levels"""
    recent_prices = prices.tail(window)
    support = recent_prices.min()
    resistance = recent_prices.max()
    current_price = prices.iloc[-1]
    
    # Distance from support/resistance as percentage
    support_distance = (current_price - support) / current_price * 100
    resistance_distance = (resistance - current_price) / current_price * 100
    
    return {
        'support': support,
        'resistance': resistance,
        'support_distance_pct': support_distance,
        'resistance_distance_pct': resistance_distance,
        'near_support': support_distance < 2,  # Within 2% of support
        'near_resistance': resistance_distance < 2  # Within 2% of resistance
    }

def apply_technical_filters(stock_data: pd.DataFrame, symbol: str) -> Dict[str, any]:
    """Apply comprehensive technical analysis filters"""
    if stock_data.empty or len(stock_data) < 50:
        return {'symbol': symbol, 'technical_score': 0, 'signals': [], 'reason': 'Insufficient data'}
    
    close_prices = stock_data['Close']
    volume = stock_data['Volume']
    high_prices = stock_data['High']
    low_prices = stock_data['Low']
    
    # Technical indicators
    rsi = calculate_rsi(close_prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
    mas = calculate_moving_averages(close_prices)
    volume_analysis = analyze_volume_pattern(volume, close_prices)
    sr_levels = calculate_support_resistance(close_prices)
    
    current_price = close_prices.iloc[-1]
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    # Scoring system
    technical_score = 0
    signals = []
    
    # RSI Analysis (0-20 points)
    if 30 <= current_rsi <= 70:  # Not overbought/oversold
        technical_score += 10
        signals.append("RSI in neutral zone")
    elif current_rsi < 30:
        technical_score += 15
        signals.append("RSI oversold - potential reversal")
    elif current_rsi > 70:
        technical_score += 5
        signals.append("RSI overbought - caution")
    
    # Moving Average Analysis (0-20 points)
    if current_price > mas['ema_9'].iloc[-1] > mas['ema_21'].iloc[-1]:
        technical_score += 20
        signals.append("Strong upward trend - EMA alignment")
    elif current_price > mas['sma_5'].iloc[-1]:
        technical_score += 10
        signals.append("Short-term bullish trend")
    
    # Bollinger Bands Analysis (0-15 points)
    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    if 0.2 <= bb_position <= 0.8:
        technical_score += 15
        signals.append("Price in BB middle range - good for trading")
    elif bb_position < 0.2:
        technical_score += 10
        signals.append("Near BB lower band - potential bounce")
    
    # Volume Analysis (0-20 points)
    if volume_analysis['volume_ratio'] > 1.5:
        technical_score += 15
        signals.append(f"High volume activity ({volume_analysis['volume_ratio']:.1f}x avg)")
    elif volume_analysis['volume_ratio'] > 1.2:
        technical_score += 10
        signals.append("Above average volume")
    
    if volume_analysis['volume_price_correlation']:
        technical_score += 5
        signals.append("Volume confirms price movement")
    
    # Liquidity Check (0-15 points)
    avg_daily_volume = volume_analysis['avg_volume_20d']
    if avg_daily_volume > 1000000:  # 10 lakh shares daily avg
        technical_score += 15
        signals.append("High liquidity stock")
    elif avg_daily_volume > 500000:  # 5 lakh shares
        technical_score += 10
        signals.append("Good liquidity")
    elif avg_daily_volume > 100000:  # 1 lakh shares
        technical_score += 5
        signals.append("Moderate liquidity")
    else:
        technical_score = 0  # Reject low liquidity stocks
        signals.append("Low liquidity - rejected")
    
    # Support/Resistance Analysis (0-10 points)
    if sr_levels['near_support'] and volume_analysis['volume_ratio'] > 1.2:
        technical_score += 8
        signals.append("Near support with volume - potential bounce")
    elif not sr_levels['near_resistance']:
        technical_score += 5
        signals.append("Room to move up")
    
    # Volatility Check
    price_volatility = close_prices.pct_change().std() * 100
    if 1 <= price_volatility <= 5:  # Good volatility for intraday
        technical_score += 10
        signals.append(f"Good volatility ({price_volatility:.1f}%)")
    elif price_volatility > 5:
        technical_score += 5
        signals.append("High volatility - higher risk/reward")
    
    return {
        'symbol': symbol,
        'technical_score': min(technical_score, 100),  # Cap at 100
        'current_price': current_price,
        'rsi': current_rsi,
        'volume_ratio': volume_analysis['volume_ratio'],
        'avg_volume': avg_daily_volume,
        'volatility': price_volatility,
        'signals': signals,
        'support': sr_levels['support'],
        'resistance': sr_levels['resistance'],
        'bb_position': bb_position,
        'trend_strength': 'Strong' if technical_score >= 70 else 'Moderate' if technical_score >= 50 else 'Weak',
        'liquidity_grade': 'High' if avg_daily_volume > 1000000 else 'Medium' if avg_daily_volume > 500000 else 'Low'
    }

def calculate_position_size(capital: float, stock_price: float, volatility: float, max_risk_per_trade: float = 0.02) -> Dict[str, any]:
    """Calculate position size based on volatility and risk management"""
    # Risk-based position sizing
    risk_amount = capital * max_risk_per_trade  # 2% max risk per trade
    
    # Volatility adjustment - higher volatility = smaller position
    volatility_multiplier = max(0.5, min(1.5, 2 / max(volatility, 0.5)))
    
    # Base quantity calculation
    max_investment = capital * 0.25 * volatility_multiplier  # Max 25% per stock, adjusted for volatility
    base_quantity = int(max_investment // stock_price)
    
    # Stop loss calculation (2-3% or 2x volatility, whichever is larger)
    stop_loss_pct = max(2, volatility * 2)
    stop_loss_price = stock_price * (1 - stop_loss_pct / 100)
    
    # Risk-adjusted quantity
    risk_per_share = stock_price - stop_loss_price
    risk_adjusted_quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else base_quantity
    
    # Final quantity (use smaller of base or risk-adjusted)
    final_quantity = min(base_quantity, risk_adjusted_quantity, int(capital // stock_price))
    
    return {
        'quantity': max(1, final_quantity),  # At least 1 share
        'investment': final_quantity * stock_price,
        'stop_loss_price': stop_loss_price,
        'stop_loss_pct': stop_loss_pct,
        'risk_amount': final_quantity * risk_per_share,
        'position_size_pct': (final_quantity * stock_price / capital) * 100
    }

# --- Slack Notifier Class ---
class SlackNotifier:
    def __init__(self, token, default_channel):
        self.client = WebClient(token=token)
        self.default_channel = default_channel
        if not token or token == "YOUR_SLACK_BOT_TOKEN":
            logger.warning("Slack Bot Token not configured. Slack notifications will be disabled.")
            self.enabled = False
        else:
            self.enabled = True

    def slackify(self, text):
        """
        Placeholder for slackify function.
        You can implement custom markdown formatting here if needed.
        For now, it returns the text as is.
        """
        return text

    def send_message(self, text, channel=None, thread_ts=None):
        """Send a formatted Slack message."""
        if not self.enabled:
            logger.warning("Slack notifications are disabled due to missing token.")
            return None
            
        channel = channel or self.default_channel
        text = self.slackify(text)
        try:
            result = self.client.chat_postMessage(
                channel=channel,
                text=text,  
                thread_ts=thread_ts,
                mrkdwn=True,  
                parse="full"  
            )
            logger.info(f"Slack message sent to {channel}: {text[:50]}...")
            return result
        except SlackApiError as e:
            logger.error(f"❌ Slack API Error: {e.response['error']}")
            return None
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return None

slack_notifier = SlackNotifier(token=SLACK_BOT_TOKEN, default_channel=SLACK_DEFAULT_CHANNEL)

def initialize_db():
    """Initializes the JSON database file if it doesn't exist."""
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump({
                "recommendations": [], 
                "trades": [], 
                "daily_summary": [],
                "active_positions": {},
                "pending_analysis": [],
                "historical_recommendations": []
            }, f, indent=4)
        logger.info(f"Initialized database file: {DB_FILE}")

def save_active_positions():
    """Save current portfolio positions to JSON for persistence"""
    try:
        with open(DB_FILE, 'r+') as f:
            db_content = json.load(f)
            db_content["active_positions"] = portfolio
            f.seek(0)
            json.dump(db_content, f, indent=4)
            f.truncate()
        logger.info(f"Saved active positions to DB: {len(portfolio)} stocks")
    except Exception as e:
        logger.error(f"Error saving active positions: {e}")

def load_active_positions():
    """Load portfolio positions from JSON on startup"""
    global portfolio
    try:
        with open(DB_FILE, 'r') as f:
            db_content = json.load(f)
            if "active_positions" in db_content and db_content["active_positions"]:
                portfolio = db_content["active_positions"]
                logger.info(f"Loaded {len(portfolio)} active positions from DB: {list(portfolio.keys())}")
                return True
            else:
                logger.info("No active positions found in DB")
                return False
    except Exception as e:
        logger.error(f"Error loading active positions: {e}")
        return False

def add_to_pending_analysis(stock_symbol, reason, hold_details):
    """Add stock to pending analysis list for next day"""
    try:
        with open(DB_FILE, 'r+') as f:
            db_content = json.load(f)
            if "pending_analysis" not in db_content:
                db_content["pending_analysis"] = []
            
            # Remove existing entry for same stock if exists
            db_content["pending_analysis"] = [
                item for item in db_content["pending_analysis"] 
                if item.get("symbol") != stock_symbol
            ]
            
            # Add new entry
            db_content["pending_analysis"].append({
                "symbol": stock_symbol,
                "reason": reason,
                "added_date": datetime.now().strftime("%Y-%m-%d"),
                "hold_details": hold_details,
                "analysis_priority": "high" if "overnight" in reason.lower() else "medium"
            })
            
            f.seek(0)
            json.dump(db_content, f, indent=4)
            f.truncate()
        logger.info(f"Added {stock_symbol} to pending analysis: {reason}")
    except Exception as e:
        logger.error(f"Error adding to pending analysis: {e}")

def get_pending_analysis():
    """Get list of stocks pending analysis"""
    try:
        with open(DB_FILE, 'r') as f:
            db_content = json.load(f)
            return db_content.get("pending_analysis", [])
    except Exception as e:
        logger.error(f"Error getting pending analysis: {e}")
        return []

def clear_pending_analysis(stock_symbol=None):
    """Clear pending analysis - specific stock or all"""
    try:
        with open(DB_FILE, 'r+') as f:
            db_content = json.load(f)
            if stock_symbol:
                # Remove specific stock
                db_content["pending_analysis"] = [
                    item for item in db_content.get("pending_analysis", [])
                    if item.get("symbol") != stock_symbol
                ]
                logger.info(f"Cleared pending analysis for {stock_symbol}")
            else:
                # Clear all
                db_content["pending_analysis"] = []
                logger.info("Cleared all pending analysis")
            
            f.seek(0)
            json.dump(db_content, f, indent=4)
            f.truncate()
    except Exception as e:
        logger.error(f"Error clearing pending analysis: {e}")

def save_historical_recommendation(recommendation_data):
    """Save historical recommendation for future reference"""
    try:
        with open(DB_FILE, 'r+') as f:
            db_content = json.load(f)
            if "historical_recommendations" not in db_content:
                db_content["historical_recommendations"] = []
            
            db_content["historical_recommendations"].append(recommendation_data)
            
            # Keep only last 30 days of recommendations
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            db_content["historical_recommendations"] = [
                rec for rec in db_content["historical_recommendations"]
                if rec.get("date", "1900-01-01") >= cutoff_date
            ]
            
            f.seek(0)
            json.dump(db_content, f, indent=4)
            f.truncate()
        logger.info("Saved historical recommendation data")
    except Exception as e:
        logger.error(f"Error saving historical recommendation: {e}")

# --- Smart Analytics Functions ---

def initialize_smart_analytics():
    """Initialize the smart analytics JSON file"""
    if not os.path.exists(SMART_ANALYTICS_FILE):
        with open(SMART_ANALYTICS_FILE, 'w') as f:
            json.dump({
                "current_recommendations": [],
                "last_updated": datetime.now().isoformat()
            }, f, indent=4)
        logger.info(f"Initialized smart analytics file: {SMART_ANALYTICS_FILE}")

def initialize_trading_history():
    """Initialize the trading history JSON file for completed trades"""
    if not os.path.exists(TRADING_HISTORY_FILE):
        with open(TRADING_HISTORY_FILE, 'w') as f:
            json.dump({
                "completed_trades": [],
                "last_updated": datetime.now().isoformat(),
                "total_trades": 0,
                "total_profit_loss": 0.0
            }, f, indent=4)
        logger.info(f"Initialized trading history file: {TRADING_HISTORY_FILE}")

def save_completed_trade_to_history(sold_stock_data):
    """Save a completed trade to the separate trading history file"""
    try:
        # Initialize file if it doesn't exist
        initialize_trading_history()
        
        with open(TRADING_HISTORY_FILE, 'r') as f:
            history_data = json.load(f)
        
        # Add the completed trade
        completed_trades = history_data.get("completed_trades", [])
        completed_trades.append(sold_stock_data)
        
        # Update summary statistics
        total_trades = len(completed_trades)
        total_pl = sum(trade.get("actual_pl", 0) for trade in completed_trades if trade.get("actual_pl") is not None)
        
        # Update the history data
        history_data.update({
            "completed_trades": completed_trades,
            "last_updated": datetime.now().isoformat(),
            "total_trades": total_trades,
            "total_profit_loss": total_pl,
            "summary_stats": {
                "profitable_trades": len([t for t in completed_trades if t.get("actual_pl", 0) > 0]),
                "losing_trades": len([t for t in completed_trades if t.get("actual_pl", 0) < 0]),
                "average_pl": total_pl / total_trades if total_trades > 0 else 0
            }
        })
        
        # Save updated history
        with open(TRADING_HISTORY_FILE, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        logger.info(f"📈 Saved completed trade to history: {sold_stock_data['symbol']} (P&L: ₹{sold_stock_data.get('actual_pl', 0):.2f})")
        
    except Exception as e:
        logger.error(f"Error saving completed trade to history: {e}")

def save_recommendations_to_smart_analytics(recommendations_list):
    """Save new recommendations to smart analytics file (CLEAN + APPEND mode)
    - Removes old 'not_bought' recommendations
    - Keeps 'bought' stocks for continued monitoring  
    - Appends fresh new recommendations
    """
    try:
        # Read existing data
        smart_data = {}
        if os.path.exists(SMART_ANALYTICS_FILE):
            with open(SMART_ANALYTICS_FILE, 'r') as f:
                smart_data = json.load(f)
        
        # Get current date and time for unique IDs
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Clean up sold stocks first (move to separate history file)
        cleanup_sold_stocks_from_analytics()
        
        # Reload after cleanup
        if os.path.exists(SMART_ANALYTICS_FILE):
            with open(SMART_ANALYTICS_FILE, 'r') as f:
                smart_data = json.load(f)
        
        existing_recs = smart_data.get("current_recommendations", [])
        
        # CLEAN UP: Keep only 'bought' stocks, remove old 'not_bought' ones
        bought_stocks = [rec for rec in existing_recs if rec.get("status") == "bought"]
        not_bought_removed = [rec for rec in existing_recs if rec.get("status") == "not_bought"]
        
        if not_bought_removed:
            logger.info(f"🧹 CLEANUP: Removing {len(not_bought_removed)} old 'not_bought' recommendations")
            for removed_rec in not_bought_removed:
                logger.debug(f"   Removed: {removed_rec.get('symbol', 'Unknown')} (not_bought)")
        
        # Keep only bought stocks for continued monitoring
        cleaned_recommendations = bought_stocks
        logger.info(f"📊 KEEPING: {len(bought_stocks)} 'bought' stocks for continued monitoring")
        
        # Get next order number for new recommendations
        max_order = 0
        if cleaned_recommendations:
            max_order = max(rec.get("recommendation_order", 0) for rec in cleaned_recommendations)
        
        # Prepare new recommendations with unique IDs and order numbers
        new_recommendations = []
        for i, rec in enumerate(recommendations_list):
            # Create unique ID with timestamp to avoid duplicates
            unique_id = f"{current_date}_{current_time}_{rec['symbol']}_{i+1}"
            
            new_recommendations.append({
                "id": unique_id,
                "date": current_date,
                "time": current_time,
                "symbol": rec['symbol'],
                "recommended_price": rec['current_price'],
                "quantity": rec['quantity'],
                "ai_reason": rec['reason'],
                "technical_score": rec['technical_score'],
                "stop_loss_price": rec['stop_loss_price'],
                "stop_loss_pct": rec['stop_loss_pct'],
                "signals": rec['signals'][:3],
                "recommendation_order": max_order + i + 1,  # Continue order numbering
                "status": "not_bought",  # Default status
                "actual_buy_price": None,
                "actual_quantity": None,
                "buy_timestamp": None,
                "sell_price": None,
                "sell_timestamp": None,
                "actual_pl": None,
                "notes": "",
                "batch_info": f"Generated at {current_time} on {current_date}"
            })
        
        # FINAL RESULT: Bought stocks + New fresh recommendations
        all_recommendations = cleaned_recommendations + new_recommendations
        smart_data["current_recommendations"] = all_recommendations
        smart_data["last_updated"] = datetime.now().isoformat()
        smart_data["latest_batch_date"] = current_date
        smart_data["latest_batch_time"] = current_time
        smart_data["cleanup_info"] = {
            "last_cleanup": datetime.now().isoformat(),
            "removed_not_bought": len(not_bought_removed),
            "kept_bought": len(bought_stocks),
            "added_new": len(new_recommendations)
        }
        
        # Save back to file
        with open(SMART_ANALYTICS_FILE, 'w') as f:
            json.dump(smart_data, f, indent=4)
        
        logger.info(f"✨ FINAL RESULT: {len(bought_stocks)} bought + {len(new_recommendations)} new = {len(all_recommendations)} total recommendations")
        logger.info(f"📊 New recommendation batch: {current_date} at {current_time}")
        
        # Send enhanced Slack notification
        smart_alert = (
            f"📊 Smart Analytics Updated! (CLEAN + APPEND)\n\n"
            f"🧹 Removed: {len(not_bought_removed)} old 'not_bought' stocks\n"
            f"📊 Kept: {len(bought_stocks)} 'bought' stocks for monitoring\n"
            f"✅ Added: {len(new_recommendations)} fresh recommendations\n"
            f"📋 Total active: {len(all_recommendations)} stocks\n"
            f"🕐 Batch generated: {current_date} at {current_time}\n\n"
            f"📁 File: {SMART_ANALYTICS_FILE}\n"
            f"🔧 Please update 'status' to 'bought' for stocks you purchase\n"
            f"🤖 System monitors only stocks marked as 'bought'\n\n"
            f"📋 NEW Fresh Recommendations:\n"
        )
        
        for rec in new_recommendations:
            smart_alert += f"• {rec['symbol']}: ₹{rec['recommended_price']:.2f} (Score: {rec['technical_score']}/100)\n"
        
        if bought_stocks:
            smart_alert += f"\n📊 Continued Monitoring (Bought Stocks):\n"
            for stock in bought_stocks:
                smart_alert += f"• {stock['symbol']}: ₹{stock.get('recommended_price', 0):.2f} (Status: {stock.get('status', 'unknown')})\n"
        
        smart_alert += f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_slack_alert(smart_alert)
        
    except Exception as e:
        logger.error(f"Error saving to smart analytics: {e}")

def load_bought_stocks_from_analytics():
    """Load only stocks marked as 'bought' from smart analytics"""
    try:
        if not os.path.exists(SMART_ANALYTICS_FILE):
            logger.info("Smart analytics file not found")
            return {}
        
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            smart_data = json.load(f)
        
        bought_stocks = {}
        current_recs = smart_data.get("current_recommendations", [])
        
        for stock in current_recs:
            if stock.get("status") == "bought":
                symbol = stock["symbol"]
                
                # Handle None values properly - if actual value is None, use recommended value
                actual_quantity = stock.get("actual_quantity")
                recommended_quantity = stock.get("quantity", 1)
                final_quantity = actual_quantity if actual_quantity is not None else recommended_quantity
                
                actual_buy_price = stock.get("actual_buy_price")
                recommended_price = stock.get("recommended_price", 0)
                final_buy_price = actual_buy_price if actual_buy_price is not None else recommended_price
                
                actual_timestamp = stock.get("buy_timestamp")
                final_timestamp = actual_timestamp if actual_timestamp is not None else datetime.now().isoformat()
                
                bought_stocks[symbol] = {
                    "quantity": final_quantity,
                    "buy_price": final_buy_price,
                    "stop_loss_price": stock.get("stop_loss_price", 0),
                    "stop_loss_pct": stock.get("stop_loss_pct", 2),
                    "technical_score": stock.get("technical_score", 0),
                    "timestamp": final_timestamp,
                    "recommendation_order": stock.get("recommendation_order", 1),
                    "technical_analysis": True,
                    "reason": stock.get("ai_reason", "Smart analytics"),
                    "signals": stock.get("signals", []),
                    "smart_analytics_id": stock.get("id", ""),
                    "recommended_price": stock.get("recommended_price", 0)
                }
        
        if bought_stocks:
            logger.info(f"📊 Loaded {len(bought_stocks)} bought stocks from smart analytics: {list(bought_stocks.keys())}")
        else:
            logger.info("📊 No stocks marked as 'bought' in smart analytics")
        
        return bought_stocks
        
    except Exception as e:
        logger.error(f"Error loading bought stocks from analytics: {e}")
        return {}

def update_stock_sell_status(symbol, sell_price, sell_reason):
    """Update stock status to 'sold' in smart analytics"""
    try:
        if not os.path.exists(SMART_ANALYTICS_FILE):
            return
        
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            smart_data = json.load(f)
        
        # Find and update the stock
        updated = False
        for stock in smart_data.get("current_recommendations", []):
            if stock["symbol"] == symbol and stock.get("status") == "bought":
                # Calculate actual P&L
                buy_price = stock.get("actual_buy_price", stock.get("recommended_price", 0))
                quantity = stock.get("actual_quantity", stock.get("quantity", 1))
                actual_pl = (sell_price - buy_price) * quantity
                
                # Update stock data
                stock["status"] = "sold"
                stock["sell_price"] = sell_price
                stock["sell_timestamp"] = datetime.now().isoformat()
                stock["actual_pl"] = actual_pl
                stock["sell_reason"] = sell_reason
                stock["notes"] = f"Sold by system: {sell_reason}"
                
                updated = True
                logger.info(f"📊 Updated {symbol} status to 'sold' in smart analytics")
                break
        
        if updated:
            # Save updated data
            smart_data["last_updated"] = datetime.now().isoformat()
            with open(SMART_ANALYTICS_FILE, 'w') as f:
                json.dump(smart_data, f, indent=4)
        
    except Exception as e:
        logger.error(f"Error updating sell status for {symbol}: {e}")

def cleanup_sold_stocks_from_analytics():
    """Remove sold stocks from current recommendations and move them to separate trading history file"""
    try:
        if not os.path.exists(SMART_ANALYTICS_FILE):
            return
        
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            smart_data = json.load(f)
        
        # Separate sold stocks from active ones
        current_recs = smart_data.get("current_recommendations", [])
        sold_stocks = [stock for stock in current_recs if stock.get("status") == "sold"]
        remaining_stocks = [stock for stock in current_recs if stock.get("status") != "sold"]
        
        if sold_stocks:
            # Move each sold stock to the separate trading history file
            for sold_stock in sold_stocks:
                save_completed_trade_to_history(sold_stock)
            
            # Update the smart analytics file with only remaining stocks
            smart_data["current_recommendations"] = remaining_stocks
            smart_data["last_updated"] = datetime.now().isoformat()
            
            # Remove the old trading_history field if it exists (we use separate file now)
            if "trading_history" in smart_data:
                del smart_data["trading_history"]
            
            # Save updated smart analytics
            with open(SMART_ANALYTICS_FILE, 'w') as f:
                json.dump(smart_data, f, indent=4)
            
            logger.info(f"📊 Moved {len(sold_stocks)} sold stocks to separate trading history file")
            logger.info(f"📊 {len(remaining_stocks)} active stocks remain in smart analytics")
        else:
            logger.debug("📊 No sold stocks to clean up")
        
    except Exception as e:
        logger.error(f"Error cleaning up sold stocks: {e}")

def append_new_day_recommendations(new_recommendations):
    """Append new day recommendations to existing ones"""
    try:
        if not os.path.exists(SMART_ANALYTICS_FILE):
            initialize_smart_analytics()
        
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            smart_data = json.load(f)
        
        # Clean up sold stocks first
        cleanup_sold_stocks_from_analytics()
        
        # Reload after cleanup
        with open(SMART_ANALYTICS_FILE, 'r') as f:
            smart_data = json.load(f)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        existing_recs = smart_data.get("current_recommendations", [])
        
        # Check if we already have recommendations for today
        today_recs_exist = any(rec.get("date") == current_date for rec in existing_recs)
        
        if today_recs_exist:
            logger.info("📊 Today's recommendations already exist in smart analytics")
            return
        
        # Prepare new recommendations for today
        new_recs = []
        for i, rec in enumerate(new_recommendations):
            new_recs.append({
                "id": f"{current_date}_{rec['symbol']}",
                "date": current_date,
                "symbol": rec['symbol'],
                "recommended_price": rec['current_price'],
                "quantity": rec['quantity'],
                "ai_reason": rec['reason'],
                "technical_score": rec['technical_score'],
                "stop_loss_price": rec['stop_loss_price'],
                "stop_loss_pct": rec['stop_loss_pct'],
                "signals": rec['signals'][:3],
                "recommendation_order": i + 1,
                "status": "not_bought",
                "actual_buy_price": None,
                "actual_quantity": None,
                "buy_timestamp": None,
                "sell_price": None,
                "sell_timestamp": None,
                "actual_pl": None,
                "notes": ""
            })
        
        # Append to existing recommendations
        all_recommendations = existing_recs + new_recs
        smart_data["current_recommendations"] = all_recommendations
        smart_data["last_updated"] = datetime.now().isoformat()
        
        # Save updated data
        with open(SMART_ANALYTICS_FILE, 'w') as f:
            json.dump(smart_data, f, indent=4)
        
        logger.info(f"📊 Appended {len(new_recs)} new recommendations to smart analytics")
        
    except Exception as e:
        logger.error(f"Error appending new recommendations: {e}")

def save_to_db(data_type, data):
    """Saves data to the JSON database."""
    try:
        with open(DB_FILE, 'r+') as f:
            db_content = json.load(f)
            db_content[data_type].append(data)
            f.seek(0)
            json.dump(db_content, f, indent=4)
            f.truncate()
        logger.info(f"Saved {data_type} to DB: {data}")
    except Exception as e:
        logger.error(f"Error saving to DB: {e}")

def fetch_market_data(stock_symbols, period="1d", interval="1m"):
    """
    Fetches historical and current market data for given stock symbols.
    For same-day analysis, might need more granular data.
    """
    logger.info(f"Fetching market data for: {stock_symbols}")
    data = {}
    for symbol in stock_symbols:
        try:
            # For Indian stocks, append '.NS' for NSE or '.BO' for BSE
            # This is a common convention for Yahoo Finance.
            # Adjust if your market data API uses different symbols.
            ticker_symbol = symbol + ".NS" if "." not in symbol else symbol
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period=period, interval=interval)
            if not hist.empty:
                data[symbol] = hist
                logger.info(f"Successfully fetched data for {symbol}")
            else:
                logger.warning(f"No data returned for {symbol}. It might be delisted or an incorrect symbol.")
                data[symbol] = None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            data[symbol] = None
    return data

def analyze_with_gemini(context_prompt, market_data_str):
    """
    Interacts with the Gemini API to get analysis or decisions.
    (This is a placeholder and needs actual Gemini API integration)
    """
    logger.info(f"Sending data to Gemini for analysis. Prompt prefix: {context_prompt[:100]}...")
    # In a real scenario, you would structure the prompt carefully and send it to the Gemini API
    try:
        response = gemini_model.generate_content(f"{context_prompt}\n\nMarket Data:\n{market_data_str}")
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        # Fallback for now if API call fails
        if "recommend" in context_prompt.lower() and "stock" in context_prompt.lower():
            logger.warning("Gemini API failed for stock recommendations. No fallback - returning error.")
            return json.dumps({"error": "Gemini API unavailable for stock recommendations"})
        elif "should I hold or sell" in context_prompt:
            logger.warning("Gemini API failed for sell/hold decision. Using safe fallback.")
            return json.dumps({"action": "hold", "reason": "Gemini API unavailable, holding for safety."})
        return "Error: Unknown Gemini request and API failed."

def parse_stock_recommendations(text_response):
    """
    Parse stock recommendations with quantities and reasons from Gemini.
    Handles formats like: 
    RELIANCE:5:High volatility expected today
    Or: RELIANCE 5 shares - Strong technical indicators
    Or multi-line format with reasons
    """
    try:
        recommendations = []
        
        # Try to handle multi-line format first
        lines = text_response.strip().split('\n')
        if len(lines) > 1:
            # Multi-line format
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Look for patterns like "1. STOCK:QTY - REASON" or "STOCK QTY - REASON"
                    if '-' in line:
                        parts = line.split('-', 1)
                        stock_qty_part = parts[0].strip()
                        reason = parts[1].strip() if len(parts) > 1 else "Good potential"
                        
                        # Remove numbering like "1.", "2." etc and clean formatting
                        stock_qty_part = stock_qty_part.split('.', 1)[-1].strip()
                        # Remove bold formatting **STOCK** -> STOCK
                        stock_qty_part = stock_qty_part.replace('**', '')
                        
                        # Parse stock and quantity
                        if ':' in stock_qty_part:
                            stock_parts = stock_qty_part.split(':')
                            stock = stock_parts[0].strip().upper()
                            # Handle quantity parsing more robustly
                            qty_part = stock_parts[1].strip()
                            # Extract just the number from the quantity part
                            import re
                            qty_match = re.search(r'\d+', qty_part)
                            quantity = int(qty_match.group()) if qty_match else 1
                        elif ' ' in stock_qty_part:
                            parts = stock_qty_part.split()
                            stock = parts[0].strip().upper()
                            quantity = 1
                            for part in parts[1:]:
                                if part.isdigit():
                                    quantity = int(part)
                                    break
                        else:
                            stock = stock_qty_part.strip().upper()
                            quantity = 1
                            
                        if len(stock) >= 2 and len(stock) <= 15 and stock.isalpha():
                            recommendations.append({
                                'symbol': stock,
                                'quantity': max(1, quantity),
                                'reason': reason
                            })
                            
                except Exception as e:
                    logger.warning(f"Could not parse line '{line}': {e}")
                    continue
        else:
            # Single line format - split by commas
            cleaned = text_response.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
            items = [item.strip() for item in cleaned.split(',') if item.strip()]
            
            for item in items:
                try:
                    reason = "Good trading potential"  # Default reason
                    
                    # Check if reason is included with - separator
                    if '-' in item:
                        parts = item.split('-', 1)
                        item = parts[0].strip()
                        reason = parts[1].strip()
                    
                    # Handle different formats for stock:quantity
                    if ':' in item:
                        # Format: STOCK:QUANTITY or STOCK:QUANTITY:REASON
                        parts = item.split(':')
                        stock = parts[0].strip().upper()
                        quantity = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 1
                        if len(parts) > 2:
                            reason = parts[2].strip()
                    elif ' ' in item:
                        # Format: STOCK QUANTITY
                        parts = item.split()
                        stock = parts[0].strip().upper()
                        quantity = 1
                        for part in parts[1:]:
                            if part.isdigit():
                                quantity = int(part)
                                break
                    else:
                        # Just stock name
                        stock = item.strip().upper()
                        quantity = 1
                    
                    # Validate stock symbol
                    if len(stock) >= 2 and len(stock) <= 15 and stock.isalpha():
                        recommendations.append({
                            'symbol': stock,
                            'quantity': max(1, quantity),
                            'reason': reason
                        })
                except Exception as e:
                    logger.warning(f"Could not parse item '{item}': {e}")
                    continue
        
        return recommendations
    except Exception as e:
        logger.error(f"Error parsing stock recommendations: {e}")
        return []



def analyze_pending_stocks():
    """Analyze stocks that were held overnight and need analysis"""
    pending_stocks = get_pending_analysis()
    if not pending_stocks:
        logger.info("📋 No pending analysis stocks from previous day")
        return
    
    logger.info(f"📋 Found {len(pending_stocks)} stocks pending analysis from previous day")
    
    for stock_info in pending_stocks:
        symbol = stock_info['symbol']
        hold_details = stock_info['hold_details']
        reason = stock_info['reason']
        
        logger.info(f"🔍 Analyzing held stock {symbol}: {reason}")
        
        # Fetch current data for analysis
        try:
            current_data = fetch_market_data([symbol], period="1d", interval="1m")
            if symbol in current_data and current_data[symbol] is not None:
                current_price = current_data[symbol]['Close'].iloc[-1]
                buy_price = hold_details['buy_price']
                quantity = hold_details['quantity']
                
                # Calculate overnight P&L
                overnight_pl = (current_price - buy_price) * quantity
                overnight_pl_pct = ((current_price - buy_price) / buy_price) * 100
                
                # Send analysis update to Slack
                analysis_message = (
                    f"📊 Overnight Stock Analysis: {symbol}\n\n"
                    f"💰 Buy Price: ₹{buy_price:.2f}\n"
                    f"💵 Current Price: ₹{current_price:.2f}\n"
                    f"📦 Quantity: {quantity}\n"
                    f"{'💚' if overnight_pl >= 0 else '❌'} Overnight P&L: ₹{overnight_pl:.2f} ({overnight_pl_pct:.1f}%)\n"
                    f"💡 Hold Reason: {reason}\n"
                    f"⏰ Analysis Time: {datetime.now().strftime('%H:%M:%S')}\n"
                    f"🔄 Stock remains in portfolio for continued monitoring"
                )
                send_slack_alert(analysis_message)
                logger.info(f"📱 Sent overnight analysis for {symbol}")
                
            else:
                logger.warning(f"⚠️ Could not fetch current data for pending stock {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing pending stock {symbol}: {e}")
    
    logger.info("📋 Completed analysis of all pending stocks")

def select_top_stocks():
    """
    Gemini AI-based stock selection with technical analysis validation.
    This function should be run at 10:15 AM after market stabilizes.
    """
    global daily_recommendations, portfolio
    logger.info("🔍 Starting Gemini AI-based stock selection process...")
    
    # First, analyze any stocks held from previous day
    analyze_pending_stocks()
    
    # Load bought stocks from smart analytics instead of active positions
    smart_portfolio = load_bought_stocks_from_analytics()
    if smart_portfolio:
        portfolio.update(smart_portfolio)
        logger.info(f"📊 Loaded {len(smart_portfolio)} bought stocks from smart analytics for monitoring")
        save_active_positions()  # Sync with old system
    
    # Get stock recommendations directly from Gemini AI
    logger.info("🤖 Requesting stock recommendations from Gemini AI...")
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")
    
    # Ask Gemini for stock recommendations
    ai_prompt = (
        f"Today is {current_date} at {current_time}. I need the best 8-12 Indian stock recommendations "
        f"for intraday trading with ₹{CAPITAL} capital. Please recommend liquid, high-volume stocks "
        f"from NSE/BSE that are suitable for same-day trading.\n\n"
        f"CRITICAL REQUIREMENT: Recommend ONLY stocks priced under ₹{CAPITAL} per share so I can afford "
        f"to buy at least 1 share with my ₹{CAPITAL} budget. Focus on affordable, liquid stocks with "
        f"good intraday potential.\n\n"
        f"Consider current market conditions, sector performance, and recent price movements. "
        f"Focus on stocks with good liquidity, intraday potential, and affordable prices.\n\n"
        f"IMPORTANT: Please provide ONLY the stock symbols and reasons in this exact format:\n"
        f"SYMBOL - Brief reason\n"
        f"Example:\n"
        f"Stock Name - Reason\n"
        f"Stock Name - Reason\n"
        f"Provide 8-12 affordable stock recommendations using ONLY the stock symbol (no .NS suffix, no numbering, no asterisks).\n"
        f"Do not include any other text, explanations, or warnings. Just the affordable stock recommendations in the format above."
    )
    
    try:
        ai_response = analyze_with_gemini(ai_prompt, "")
        logger.info(f"🤖 Gemini AI Response: {ai_response}")
        
        # Parse Gemini recommendations
        import re
        gemini_recommendations = []
        for line in ai_response.split('\n'):
            line = line.strip()
            # Look for lines with format "SYMBOL - reason"
            if '-' in line and len(line) > 5:
                parts = line.split('-', 1)
                if len(parts) == 2:
                    symbol_part = parts[0].strip()
                    reason = parts[1].strip()
                    
                    # Extract only alphabetic characters for symbol (remove numbers, asterisks, dots, etc.)
                    symbol = re.sub(r'[^A-Z]', '', symbol_part.upper())
                    
                    # Additional cleaning
                    symbol = symbol.replace('NS', '').replace('BO', '').replace('NSE', '').replace('BSE', '')
                    
                    # Validate symbol (should be 2-15 characters, all letters)
                    if symbol and 2 <= len(symbol) <= 15 and symbol.isalpha():
                        gemini_recommendations.append({
                            'symbol': symbol,
                            'ai_reason': reason
                        })
                        logger.info(f"✅ Parsed recommendation: {symbol} - {reason}")
                    else:
                        logger.debug(f"❌ Invalid symbol parsed: '{symbol}' from line: '{line}'")
        
        if len(gemini_recommendations) < 5:
            logger.error("❌ Insufficient stock recommendations from Gemini AI. Cannot proceed.")
            return
        
        logger.info(f"📊 Gemini recommended {len(gemini_recommendations)} stocks: {[stock['symbol'] for stock in gemini_recommendations]}")
        
        # Fetch historical data for technical analysis of Gemini recommendations
        recommended_symbols = [stock['symbol'] for stock in gemini_recommendations]
        logger.info("📈 Fetching historical data for technical analysis of Gemini recommendations...")
        historical_data = fetch_market_data(recommended_symbols, period="60d", interval="1d")
        
        # Apply technical filters to Gemini recommended stocks
        technical_analysis_results = []
        for stock_info in gemini_recommendations:
            symbol = stock_info['symbol']
            if symbol in historical_data and historical_data[symbol] is not None:
                try:
                    analysis = apply_technical_filters(historical_data[symbol], symbol)
                    analysis['ai_reason'] = stock_info['ai_reason']
                    
                    # Check if stock is affordable (price <= CAPITAL)
                    current_price = analysis['current_price']
                    if current_price <= CAPITAL:
                        technical_analysis_results.append(analysis)
                        logger.info(f"✅ {symbol}: Score {analysis['technical_score']}/100 - {analysis['trend_strength']} trend - Price: ₹{current_price:.2f} (Affordable)")
                    else:
                        logger.warning(f"❌ {symbol}: Price ₹{current_price:.2f} exceeds budget of ₹{CAPITAL} - Skipping")
                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing {symbol}: {e}")
            else:
                logger.warning(f"⚠️ No data available for {symbol}")
        
        # Sort by technical score
        technical_analysis_results.sort(key=lambda x: x['technical_score'], reverse=True)
        
        if len(technical_analysis_results) < 1:
            logger.error("❌ Insufficient affordable Gemini-recommended stocks passed technical screening.")
            logger.error(f"💡 Tip: With ₹{CAPITAL} capital, focus on stocks priced under ₹{CAPITAL} per share.")
            return
        
        # Select top 5 stocks with best technical scores
        top_candidates = technical_analysis_results[:5]
        logger.info(f"🎯 Top 5 technically validated Gemini recommendations: {[stock['symbol'] for stock in top_candidates]}")
        
        # Prepare final selection with AI + Technical analysis
        selected_stocks = []
        for stock in top_candidates:
            selected_stocks.append({
                'symbol': stock['symbol'],
                'technical_data': stock,
                'ai_reason': stock['ai_reason']
            })
    
    except Exception as e:
        logger.error(f"❌ Error getting recommendations from Gemini AI: {e}")
        return
    
    # Calculate position sizes using risk management
    final_recommendations = []
    for stock_info in selected_stocks:
        symbol = stock_info['symbol']
        tech_data = stock_info['technical_data']
        
        # For affordable stocks, calculate how many shares we can buy with allocated capital
        allocated_capital = CAPITAL / len(selected_stocks)
        current_price = tech_data['current_price']
        max_affordable_shares = int(allocated_capital / current_price)
        
        # Ensure we can buy at least 1 share
        if max_affordable_shares < 1:
            max_affordable_shares = 1
        
        # Calculate position size based on volatility and risk management
        position_info = calculate_position_size(
            allocated_capital,
            current_price,
            tech_data['volatility']
        )
        
        # Use the minimum of calculated quantity and affordable quantity
        final_quantity = min(position_info['quantity'], max_affordable_shares)
        if final_quantity < 1:
            final_quantity = 1
        
        # Recalculate stop loss for the final quantity
        position_info['quantity'] = final_quantity
        
        final_recommendations.append({
            'symbol': symbol,
            'quantity': position_info['quantity'],
            'reason': stock_info['ai_reason'],
            'technical_score': tech_data['technical_score'],
            'current_price': tech_data['current_price'],
            'stop_loss_price': position_info['stop_loss_price'],
            'stop_loss_pct': position_info['stop_loss_pct'],
            'risk_amount': position_info['risk_amount'],
            'signals': tech_data['signals']
        })
    
    # Update global variables
    daily_recommendations = [stock['symbol'] for stock in final_recommendations]
    
    logger.info(f"🎯 Final Gemini + Technical selections: {daily_recommendations}")
    
    if not final_recommendations:
        logger.error("❌ No suitable stocks found for trading today.")
        return
        


    if not final_recommendations:
        logger.error("❌ No suitable stocks found for trading today.")
        return

    # Portfolio setup using technical analysis recommendations
    temp_portfolio = {}
    slack_message_stocks = []
    total_investment = 0
    
    for i, stock_rec in enumerate(final_recommendations):
        stock_symbol = stock_rec['symbol']
        quantity = stock_rec['quantity']
        current_price = stock_rec['current_price']
        reason = stock_rec['reason']
        technical_score = stock_rec['technical_score']
        stop_loss_price = stock_rec['stop_loss_price']
        stop_loss_pct = stock_rec['stop_loss_pct']
        
        investment = current_price * quantity
        total_investment += investment
        
        # Create portfolio entry with comprehensive data
        temp_portfolio[stock_symbol] = {
            "quantity": quantity,
            "buy_price": current_price,
            "stop_loss_price": stop_loss_price,
            "stop_loss_pct": stop_loss_pct,
            "technical_score": technical_score,
            "timestamp": datetime.now().isoformat(),
            "recommendation_order": i + 1,
            "technical_analysis": True,
            "reason": reason,
            "signals": stock_rec['signals'][:3],  # Top 3 technical signals
            "risk_amount": stock_rec['risk_amount']
        }
        
        slack_message_stocks.append({
            "name": stock_symbol,
            "price": current_price,
            "quantity": quantity,
            "order": i + 1,
            "reason": reason,
            "technical_score": technical_score,
            "stop_loss_price": stop_loss_price,
            "stop_loss_pct": stop_loss_pct,
            "investment": investment,
            "signals": stock_rec['signals'][:2]  # Top 2 signals for Slack
        })
        
        logger.info(f"✅ Added {stock_symbol}: {quantity} shares @ ₹{current_price:.2f}, "
                   f"SL: ₹{stop_loss_price:.2f} ({stop_loss_pct:.1f}%), Score: {technical_score}/100")

    portfolio = temp_portfolio
    
    # Store comprehensive recommendation data
    recommendation_data = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "capital": CAPITAL,
        "total_investment": total_investment,
        "recommended_stocks": daily_recommendations,
        "portfolio_setup": portfolio,
        "selection_method": "affordable_gemini_ai_with_technical_validation",
        "risk_management": True,
        "detailed_recommendations": final_recommendations
    }
    
    save_to_db("recommendations", recommendation_data)
    save_historical_recommendation(recommendation_data)
    
    # Save to smart analytics for manual control
    save_recommendations_to_smart_analytics(final_recommendations)
    
    # Don't automatically add to portfolio - wait for manual status update
    # save_active_positions()  # Save current positions to JSON

    # Send enhanced Slack alert with Gemini AI + technical analysis
    if slack_message_stocks:
        message = f"🚀 Hi Vijay! Today's Affordable Gemini AI Stock Recommendations:\n\n"
        
        for stock_info in slack_message_stocks:
            message += f"📈 #{stock_info['order']}: {stock_info['name']}\n"
            message += f"   🤖 AI Recommendation: {stock_info['reason']}\n"
            message += f"   📊 Technical Score: {stock_info['technical_score']}/100\n"
            message += f"   💰 Price: ₹{stock_info['price']:.2f} (Affordable!)\n"
            message += f"   📦 Quantity: {stock_info['quantity']} shares\n"
            message += f"   💵 Investment: ₹{stock_info['investment']:.2f}\n"
            message += f"   🛡️ Stop Loss: ₹{stock_info['stop_loss_price']:.2f} ({stock_info['stop_loss_pct']:.1f}%)\n"
            if stock_info['signals']:
                message += f"   📈 Technical Signals: {', '.join(stock_info['signals'])}\n"
            message += "\n\n\n"
        
        message += f"💰 Total Investment: ₹{total_investment:.2f} / ₹{CAPITAL:,}\n"
        message += f"💳 Remaining Capital: ₹{(CAPITAL - total_investment):.2f}\n"
        message += f"🛡️ Risk Management: Stop losses set for all positions\n"
        message += f"📊 Selection Method: Affordable Gemini AI + Technical Validation\n"
        message += f"💡 All stocks priced under ₹{CAPITAL} per share for your budget!\n"
        message += f"⏰ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"🎯 Ready for budget-friendly intraday trading!"
        
        send_slack_alert(message)
        logger.info(f"📱 Sent stock recommendations alert to Slack so you know what to buy")
    else:
        logger.warning("⚠️ No stock information available for Slack alert")


def send_slack_alert(message):
    """Sends a message to Slack via the SlackNotifier instance."""
    slack_notifier.send_message(message)


def sell_decision_logic(stock_symbol, stock_data_hist):
    """
    Enhanced sell/hold decision logic with multiple criteria:
    1. Stop-loss check (automatic sell)
    2. Profit target check (automatic sell)
    3. Technical analysis signals
    4. Gemini AI final decision
    """
    logger.info(f"🔍 Analyzing sell/hold decision for {stock_symbol}...")

    buy_details = portfolio.get(stock_symbol, {})
    if not buy_details:
        logger.error(f"❌ No buy details found for {stock_symbol}")
        return {"action": "hold", "reason": "No buy details available"}

    current_price = stock_data_hist['Close'].iloc[-1]
    buy_price = buy_details.get('buy_price', 0)
    quantity = buy_details.get('quantity', 0)
    stop_loss_price = buy_details.get('stop_loss_price', 0)
    
    # Calculate current P&L
    current_pl = (current_price - buy_price) * quantity
    current_pl_pct = ((current_price - buy_price) / buy_price) * 100
    
    logger.info(f"📊 {stock_symbol}: Buy ₹{buy_price:.2f} → Current ₹{current_price:.2f} | P&L: ₹{current_pl:.2f} ({current_pl_pct:.1f}%)")

    # 0. MINIMUM HOLD TIME CHECK (Hold for at least specified minutes after purchase)
    try:
        buy_timestamp = buy_details.get('timestamp', '')
        if buy_timestamp:
            buy_time = datetime.fromisoformat(buy_timestamp)
            time_held = (datetime.now() - buy_time).total_seconds() / 60  # minutes
            
            if time_held < MINIMUM_HOLD_TIME_MINUTES:
                logger.info(f"⏱️ MINIMUM HOLD TIME: {stock_symbol} held for only {time_held:.1f} min - Need {MINIMUM_HOLD_TIME_MINUTES} min minimum")
                return {
                    "action": "hold",
                    "reason": f"Minimum hold time protection - held only {time_held:.1f}/{MINIMUM_HOLD_TIME_MINUTES} min",
                    "trigger_type": "minimum_hold_time",
                    "pl_amount": current_pl,
                    "pl_percent": current_pl_pct
                }
    except Exception as e:
        logger.debug(f"Could not check hold time for {stock_symbol}: {e}")

    # 1. STOP LOSS CHECK (Automatic Sell)
    if stop_loss_price > 0 and current_price <= stop_loss_price:
        logger.warning(f"🛑 STOP LOSS TRIGGERED for {stock_symbol}: Price ₹{current_price:.2f} <= SL ₹{stop_loss_price:.2f}")
        return {
            "action": "sell",
            "reason": f"Stop loss triggered at ₹{current_price:.2f} (SL: ₹{stop_loss_price:.2f})",
            "trigger_type": "stop_loss",
            "pl_amount": current_pl,
            "pl_percent": current_pl_pct
        }

    # 2. PROFIT TARGET CHECK (Automatic Sell if >= target profit)
    if current_pl_pct >= PROFIT_TARGET_PERCENT:
        logger.info(f"🎯 PROFIT TARGET HIT for {stock_symbol}: {current_pl_pct:.1f}% profit")
        return {
            "action": "sell",
            "reason": f"Profit target achieved: {current_pl_pct:.1f}% gain (₹{current_pl:.2f})",
            "trigger_type": "profit_target",
            "pl_amount": current_pl,
            "pl_percent": current_pl_pct
        }
    
    # 2.5. MINIMUM PROFIT PROTECTION (Don't sell if profit < threshold)
    if 0 < current_pl_pct < MINIMUM_PROFIT_PROTECTION_PERCENT:
        logger.info(f"💰 MINIMUM PROFIT PROTECTION: Holding {stock_symbol} (P&L: {current_pl_pct:.1f}%) - Profit too small to realize")
        return {
            "action": "hold",
            "reason": f"Minimum profit protection - {current_pl_pct:.1f}% profit is below {MINIMUM_PROFIT_PROTECTION_PERCENT}% threshold",
            "trigger_type": "minimum_profit_protection",
            "pl_amount": current_pl,
            "pl_percent": current_pl_pct
        }
    
    # 2.6. EARLY TRADING HOURS PROTECTION (Before 11:30 AM)
    current_time = datetime.now().time()
    if current_time < dt_time(11, 30):
        # During early hours, only sell if significant loss or good profit
        if EARLY_TRADING_LOSS_THRESHOLD < current_pl_pct < PROFIT_TARGET_PERCENT:
            logger.info(f"⏰ EARLY TRADING PROTECTION: Holding {stock_symbol} (P&L: {current_pl_pct:.1f}%) - Too early to exit small positions")
            return {
                "action": "hold",
                "reason": f"Early trading hours protection - small P&L ({current_pl_pct:.1f}%) needs more time",
                "trigger_type": "early_hours_protection",
                "pl_amount": current_pl,
                "pl_percent": current_pl_pct
            }

    # 3. TECHNICAL ANALYSIS CHECK
    try:
        # Quick technical analysis on recent data
        recent_data = stock_data_hist.tail(20)  # Last 20 data points
        if len(recent_data) >= 10:
            rsi = calculate_rsi(recent_data['Close']).iloc[-1]
            current_volume = stock_data_hist['Volume'].iloc[-1]
            avg_volume = stock_data_hist['Volume'].tail(10).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Technical sell signals
            if rsi > 75 and volume_ratio > 1.5:  # Overbought with high volume
                return {
                    "action": "sell",
                    "reason": f"Technical sell signal: RSI {rsi:.1f} (overbought) + high volume {volume_ratio:.1f}x",
                    "trigger_type": "technical",
                    "pl_amount": current_pl,
                    "pl_percent": current_pl_pct
                }
    except Exception as e:
        logger.warning(f"⚠️ Technical analysis failed for {stock_symbol}: {e}")

    # 4. GEMINI AI DECISION (Final check)
    market_data_str = f"--- {stock_symbol} Recent Data ---\n{stock_data_hist.tail(10).to_string()}"
    
    # Get current time for context
    current_time = datetime.now()
    time_str = current_time.strftime("%H:%M")
    
    prompt = (
        f"INTRADAY TRADING DECISION - Time: {time_str}\n"
        f"Stock: {stock_symbol}\n"
        f"Buy Price: ₹{buy_price:.2f} | Current Price: ₹{current_price:.2f}\n"
        f"Current P&L: ₹{current_pl:.2f} ({current_pl_pct:.1f}%)\n"
        f"Stop Loss: ₹{stop_loss_price:.2f}\n"
        f"Quantity: {quantity} shares\n\n"
        f"IMPORTANT CONTEXT:\n"
        f"- This is INTRADAY trading - positions can be held throughout the day\n"
        f"- STRICT RULE: DO NOT sell if profit is less than {PROFIT_TARGET_PERCENT}% - hold for bigger moves\n"
        f"- Small losses/gains (-1% to +1%) are NORMAL in early trading hours\n"
        f"- Only recommend SELL if there's clear technical breakdown or profit >= {PROFIT_TARGET_PERCENT}%\n"
        f"- Consider the TIME: Early hours (before 11:30 AM) need patience for moves to develop\n"
        f"- AVOID selling at breakeven or tiny profits - wait for meaningful moves\n\n"
        f"Based on the market data, P&L status, and time of day, should I SELL or HOLD?\n"
        f"Focus on: Strong technical signals, meaningful profit/loss levels, time remaining in trading day.\n\n"
        f"Respond ONLY in this JSON format (no other text):\n"
        f'{{"action": "sell", "reason": "brief reason"}}\n'
        f'OR\n'
        f'{{"action": "hold", "reason": "brief reason"}}\n\n'
        f"Recent market data:\n{market_data_str}"
    )

    try:
        decision_json = analyze_with_gemini(prompt, market_data_str)
        logger.debug(f"🤖 Gemini response for {stock_symbol}: {decision_json}")
        
        # Clean the response to extract JSON
        decision_json = decision_json.strip()
        if decision_json.startswith('```'):
            decision_json = decision_json.split('\n')[1:-1]
            decision_json = '\n'.join(decision_json)
        
        decision = json.loads(decision_json)
        decision['pl_amount'] = current_pl
        decision['pl_percent'] = current_pl_pct
        decision['trigger_type'] = 'ai_decision'
        
        logger.info(f"🤖 Gemini decision for {stock_symbol}: {decision['action'].upper()} - {decision['reason']}")
        return decision
        
    except Exception as e:
        logger.error(f"❌ Error parsing Gemini decision for {stock_symbol}: {e}")
        logger.error(f"Raw response: {decision_json}")
        
        # Conservative fallback decision based on P&L and time
        current_time = datetime.now().time()
        
        if current_pl_pct < -4.0:  # If losing more than 4%, consider selling
            return {
                "action": "sell", 
                "reason": f"AI error, cutting significant loss at {current_pl_pct:.1f}%",
                "trigger_type": "fallback",
                "pl_amount": current_pl,
                "pl_percent": current_pl_pct
            }
        elif current_pl_pct >= FALLBACK_PROFIT_THRESHOLD:  # If profit >= threshold, sell to secure gains
            return {
                "action": "sell", 
                "reason": f"AI error, securing meaningful profit at {current_pl_pct:.1f}%",
                "trigger_type": "fallback",
                "pl_amount": current_pl,
                "pl_percent": current_pl_pct
            }
        else:  # Small profit/loss or early hours, hold
            return {
                "action": "hold", 
                "reason": f"AI error, conservative hold for P&L ({current_pl_pct:.1f}%) - need more time",
                "trigger_type": "fallback",
                "pl_amount": current_pl,
                "pl_percent": current_pl_pct
            }


def monitor_and_update_portfolio():
    """
    Monitors the active portfolio during market hours (9:10 AM - 3:00 PM).
    Reads from smart analytics file every time to ensure only bought stocks are analyzed.
    """
    global portfolio
    
    # Read fresh data from smart analytics file every time
    logger.info("📊 Reading latest smart analytics data for monitoring...")
    smart_portfolio = load_bought_stocks_from_analytics()
    
    if not smart_portfolio:
        logger.info("📊 No stocks marked as 'bought' in smart analytics. Nothing to monitor.")
        # Clear global portfolio to ensure consistency
        portfolio = {}
        save_active_positions()
        return
    
    # Update global portfolio with fresh data from smart analytics
    portfolio = smart_portfolio
    save_active_positions()  # Sync with old system
    
    logger.info(f"📊 Starting portfolio monitoring for {len(portfolio)} bought stocks: {list(portfolio.keys())}")
    
    # Create a list to iterate over, as portfolio can change during iteration
    active_stocks_to_monitor = list(portfolio.keys()) 

    for stock_symbol in active_stocks_to_monitor:
        if stock_symbol not in portfolio: # Stock might have been sold in a previous iteration
            continue

        logger.info(f"🔍 Monitoring {stock_symbol} (marked as 'bought' in smart analytics)...")
        # Fetch fresh data for the specific stock
        # For intraday, we need very recent data, e.g., last 30 mins or 1 hour with 1-min interval
        stock_latest_data = fetch_market_data([stock_symbol], period="1d", interval="1m") # Fetches up to current minute data for today

        if stock_latest_data.get(stock_symbol) is None or stock_latest_data[stock_symbol].empty:
            logger.warning(f"Could not fetch latest data for {stock_symbol} during monitoring. Skipping decision.")
            continue
        
        current_price_data = stock_latest_data[stock_symbol]
        current_price = current_price_data['Close'].iloc[-1]

        decision = sell_decision_logic(stock_symbol, current_price_data)

        if decision.get("action") == "sell":
            sell_price = current_price # Assume sell at current fetched price
            original_buy_info = portfolio[stock_symbol]
            profit_loss = decision.get('pl_amount', (sell_price - original_buy_info['buy_price']) * original_buy_info['quantity'])
            profit_loss_pct = decision.get('pl_percent', ((sell_price - original_buy_info['buy_price']) / original_buy_info['buy_price']) * 100)
            trigger_type = decision.get('trigger_type', 'unknown')
            
            # Enhanced sell alert with trigger type
            trigger_emoji = {
                'stop_loss': '🛑',
                'profit_target': '🎯',
                'technical': '📈',
                'ai_decision': '🤖',
                'fallback': '⚠️'
            }.get(trigger_type, '🔔')
            
            alert_message = (
                f"{trigger_emoji} Hi Vijay! SELL Alert - {trigger_type.replace('_', ' ').title()}! {trigger_emoji}\n\n"
                f"📊 Stock: {stock_symbol}\n"
                f"📦 Quantity: {original_buy_info['quantity']}\n"
                f"💵 Sell Price: ₹{sell_price:.2f}\n"
                f"💰 Buy Price: ₹{original_buy_info['buy_price']:.2f}\n"
                f"{'💚' if profit_loss >= 0 else '❌'} P&L: ₹{profit_loss:.2f} ({profit_loss_pct:.1f}%)\n"
                f"💡 Trigger: {decision.get('reason', 'N/A')}\n"
                f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
                f"🔄 Action: Execute sell order immediately!"
            )
            send_slack_alert(alert_message)
            logger.info(alert_message)
            
            # Record the enhanced trade
            save_to_db("trades", {
                "timestamp": datetime.now().isoformat(),
                "stock_symbol": stock_symbol,
                "action": "sell",
                "quantity": original_buy_info['quantity'],
                "price": sell_price,
                "buy_price": original_buy_info['buy_price'],
                "reason": decision.get('reason', 'N/A'),
                "trigger_type": trigger_type,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "technical_score": original_buy_info.get('technical_score', 0),
                "stop_loss_price": original_buy_info.get('stop_loss_price', 0)
            })
            # Update smart analytics with sell status
            update_stock_sell_status(stock_symbol, sell_price, decision.get('reason', 'System sell decision'))
            
            del portfolio[stock_symbol] # Remove from active portfolio
            save_active_positions()  # Update JSON after selling
            clear_pending_analysis(stock_symbol)  # Remove from pending analysis if exists
            logger.info(f"Removed {stock_symbol} from active portfolio after sell advice.")
        else:
            profit_loss = decision.get('pl_amount', 0)
            profit_loss_pct = decision.get('pl_percent', 0)
            trigger_type = decision.get('trigger_type', 'unknown')
            
            logger.info(f"📊 HOLDING {stock_symbol}: P&L ₹{profit_loss:.2f} ({profit_loss_pct:.1f}%) - {decision.get('reason', 'No reason')}")
            
            # Log the hold decision with enhanced data
            save_to_db("trades", {
                "timestamp": datetime.now().isoformat(),
                "stock_symbol": stock_symbol,
                "action": "hold_check",
                "price_at_check": current_price,
                "reason": decision.get('reason', 'Periodic check'),
                "trigger_type": trigger_type,
                "current_pl": profit_loss,
                "current_pl_pct": profit_loss_pct
            })

    if not portfolio:
        logger.info("📊 All bought stocks have been processed/sold or none marked as 'bought'.")
        # Potentially stop monitoring if all sold, or let scheduler handle it
    else:
        logger.info(f"📊 Remaining bought stocks in portfolio: {list(portfolio.keys())}")


def end_of_day_assessment():
    """
    Before 3:00 PM (or at EOD), sell all remaining stocks or log hold decision for next day.
    Reads from smart analytics file to ensure only bought stocks are assessed.
    """
    global portfolio
    logger.info("📊 Performing end-of-day assessment for stocks marked as 'bought'...")
    
    # Read fresh data from smart analytics file at EOD
    logger.info("📊 Reading latest smart analytics data for EOD assessment...")
    smart_portfolio = load_bought_stocks_from_analytics()
    
    if not smart_portfolio:
        logger.info("📊 End-of-day: No stocks marked as 'bought' in smart analytics.")
        # Clear global portfolio to ensure consistency
        portfolio = {}
        save_active_positions()
        save_to_db("daily_summary", {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "status": "EOD: No bought stocks to assess.",
            "remaining_holdings": {}
        })
        return
    
    # Update global portfolio with fresh data from smart analytics
    portfolio = smart_portfolio
    save_active_positions()  # Sync with old system
    
    logger.info(f"📊 EOD assessment for {len(portfolio)} bought stocks: {list(portfolio.keys())}")

    stocks_to_assess = list(portfolio.keys())
    for stock_symbol in stocks_to_assess:
        if stock_symbol not in portfolio: continue # Already processed

        logger.info(f"📊 EOD Assessment for {stock_symbol} (marked as 'bought').")
        # Fetch final EOD price
        stock_eod_data = fetch_market_data([stock_symbol], period="1d", interval="1m") # Get latest data
        
        if stock_eod_data.get(stock_symbol) is None or stock_eod_data[stock_symbol].empty:
            logger.warning(f"EOD: Could not fetch final data for {stock_symbol}. Deciding to hold by default to avoid data loss.")
            # Log as held for next day if data is missing
            log_hold_for_next_day(stock_symbol, "Data unavailable at EOD")
            continue

        current_price_data = stock_eod_data[stock_symbol]
        eod_price = current_price_data['Close'].iloc[-1]

        # For EOD, the script could have a simpler rule:
        # e.g., if Gemini didn't say sell, and it's EOD, force sell for intraday,
        # or ask Gemini one last time with urgency.
        # The prompt says: "must have sold all the stocks or confirmed that they should be held for the next day"

        # Let's ask Gemini one last time with "EOD" context
        buy_details = portfolio.get(stock_symbol, {})
        buy_price_info = f"Bought at {buy_details.get('buy_price', 'N/A')}. Current quantity: {buy_details.get('quantity', 'N/A')}. "
        prompt = (
            f"It's near market close. For stock {stock_symbol} ({buy_price_info}), "
            f"should I sell it now to realize any intraday profit/loss, or is it strongly advisable to hold it overnight? "
            f"Prioritize realizing same-day results unless there's a strong case for holding. "
            f"Respond in Text format: {'action': 'sell/hold_overnight', 'reason': '...'} no comments or explanation of yours just Text.\n"
            f"Market data: {current_price_data.to_string()}"
        )
        
        decision_json = analyze_with_gemini(prompt, current_price_data.to_string())
        try:
            decision = json.loads(decision_json)
        except Exception as e:
            logger.error(f"EOD: Error parsing Gemini decision for {stock_symbol}: {e}. Defaulting to sell.")
            decision = {"action": "sell", "reason": f"EOD, Gemini communication error: {e}"}


        if decision.get("action") == "hold_overnight":
            log_hold_for_next_day(stock_symbol, decision.get('reason', 'Gemini advised holding overnight.'), eod_price)
        else: # Default to sell if not 'hold_overnight'
            original_buy_info = portfolio[stock_symbol]
            profit_loss = (eod_price - original_buy_info['buy_price']) * original_buy_info['quantity']
            
            alert_message = (
                f"🔔 Hi Vijay! EOD Auto-Sell Alert 🔔\n\n"
                f"📊 Stock: {stock_symbol}\n"
                f"📦 Quantity: {original_buy_info['quantity']}\n"
                f"💵 EOD Sell Price: ₹{eod_price:.2f}\n"
                f"💰 Original Buy Price: ₹{original_buy_info['buy_price']:.2f}\n"
                f"{'💚' if profit_loss >= 0 else '❌'} Final P/L: ₹{profit_loss:.2f}\n"
                f"💡 Reason: {decision.get('reason', 'End of day forced sell / Gemini advised sell.')}\n"
                f"⏰ EOD Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            send_slack_alert(alert_message)
            logger.info(alert_message)

            save_to_db("trades", {
                "timestamp": datetime.now().isoformat(),
                "stock_symbol": stock_symbol,
                "action": "sell_eod",
                "quantity": original_buy_info['quantity'],
                "price": eod_price,
                "buy_price": original_buy_info['buy_price'],
                "reason": decision.get('reason', 'EOD action'),
                "profit_loss": profit_loss
            })
            # Update smart analytics with sell status
            update_stock_sell_status(stock_symbol, eod_price, decision.get('reason', 'EOD sell decision'))
            
            del portfolio[stock_symbol]
            logger.info(f"EOD: Sold {stock_symbol} and removed from portfolio.")

    # Send daily summary only if there were any actual trades (sells)
    trades_today = []
    try:
        # Get today's trades from database (if any sells happened)
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            if 'trades' in data:
                today = datetime.now().strftime("%Y-%m-%d")
                trades_today = [trade for trade in data['trades'] 
                              if trade.get('timestamp', '').startswith(today) 
                              and trade.get('action') in ['sell', 'sell_eod']]
    except:
        pass
    
    # Only send summary if there were actual sells today
    if trades_today:
        total_pl = sum(trade.get('profit_loss', 0) for trade in trades_today)
        sell_count = len(trades_today)
        
        summary_message = (
            f"📊 End-of-Day Trading Summary 📊\n\n"
            f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"🔄 Total Sells: {sell_count}\n"
            f"{'💚' if total_pl >= 0 else '❌'} Total P&L: ₹{total_pl:.2f}\n"
            f"⏰ Market Close: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"✅ System monitored portfolio throughout the day\n"
            f"🎯 All sell decisions executed automatically"
        )
        send_slack_alert(summary_message)
        logger.info("📱 Sent EOD summary to Slack (sells occurred today)")
    else:
        logger.info("📱 No sells today - no EOD summary sent to Slack")

    logger.info("📊 End-of-day assessment complete.")
    save_to_db("daily_summary", {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "EOD processing complete.",
        "final_portfolio_state_if_any_held": portfolio, # Will be empty if all sold
        "total_sells_today": len(trades_today),
        "total_pl_today": sum(trade.get('profit_loss', 0) for trade in trades_today) if trades_today else 0
    })
    # Reset daily recommendations for the next day
    global daily_recommendations
    daily_recommendations = []


def log_hold_for_next_day(stock_symbol, reason, current_price=None):
    """Logs the decision to hold a stock for the next day."""
    global portfolio
    buy_info = portfolio[stock_symbol]
    log_message = (
        f"➡️ Hi Vijay! HOLDING {stock_symbol} overnight\n\n"
        f"📦 Quantity: {buy_info['quantity']}\n"
        f"💰 Buy Price: ₹{buy_info['buy_price']:.2f}\n"
        f"💡 Reason: {reason}"
    )
    if current_price:
        unrealized_pl = (current_price - buy_info['buy_price']) * buy_info['quantity']
        log_message += f"\n💵 Current Price: ₹{current_price:.2f}"
        log_message += f"\n{'💚' if unrealized_pl >= 0 else '❌'} Unrealized P/L: ₹{unrealized_pl:.2f}"
    log_message += f"\n⏰ Hold Decision Time: {datetime.now().strftime('%H:%M:%S')}"
    
    # Only log to file/console, no Slack alert for holds (only for actual sells)
    logger.info(log_message)
    
    # Add to pending analysis for next day
    hold_details = {
        "buy_price": buy_info['buy_price'],
        "quantity": buy_info['quantity'],
        "current_price": current_price,
        "unrealized_pl": unrealized_pl if current_price else 0,
        "hold_date": datetime.now().strftime("%Y-%m-%d"),
        "technical_score": buy_info.get('technical_score', 0),
        "stop_loss_price": buy_info.get('stop_loss_price', 0),
        "original_reason": buy_info.get('reason', 'N/A')
    }
    
    add_to_pending_analysis(stock_symbol, f"Held overnight: {reason}", hold_details)
    save_active_positions()  # Save updated positions
    
    save_to_db("trades", {
        "timestamp": datetime.now().isoformat(),
        "stock_symbol": stock_symbol,
        "action": "hold_overnight",
        "quantity": buy_info['quantity'],
        "buy_price": buy_info['buy_price'],
        "price_at_decision": current_price,
        "reason": reason
    })
    # The stock remains in 'portfolio' if held overnight.
    # Logic for how this 'portfolio' is handled the *next day* needs to be defined.
    # E.g., does it become part of a longer-term portfolio, or is it prioritized for sale?
    # For this script, we assume it's cleared for the next day's intraday focus,
    # unless specific logic for multi-day holding is added.
    # For simplicity of this daily script, we'll assume 'portfolio' is for the current trading day cycle.
    # If a stock is held overnight, it means it *wasn't* sold within the script's intraday goal.
    # The 'portfolio' variable will be reset or re-evaluated at the start of the next cycle by select_top_stocks.

def append_stocks_to_smart_analytics(recommendations_list):
    """Append additional stocks to smart analytics file (for adding stocks throughout the day)"""
    try:
        logger.info(f"📊 Appending {len(recommendations_list)} additional stocks to smart analytics...")
        
        # Use the main save function which now always appends
        save_recommendations_to_smart_analytics(recommendations_list)
        
        logger.info(f"✅ Successfully appended {len(recommendations_list)} stocks to existing recommendations")
        
    except Exception as e:
        logger.error(f"Error appending stocks to smart analytics: {e}")

def add_manual_stock_to_analytics(symbol, price, quantity, reason="Manual addition"):
    """Add a single stock manually to smart analytics"""
    try:
        # Create a recommendation structure for the manual stock
        manual_recommendation = {
            'symbol': symbol,
            'current_price': price,
            'quantity': quantity,
            'reason': reason,
            'technical_score': 50,  # Default score for manual entries
            'stop_loss_price': price * 0.98,  # 2% stop loss
            'stop_loss_pct': 2.0,
            'signals': ["Manual entry"]
        }
        
        logger.info(f"📝 Adding manual stock: {symbol} @ ₹{price:.2f} (Qty: {quantity})")
        append_stocks_to_smart_analytics([manual_recommendation])
        
    except Exception as e:
        logger.error(f"Error adding manual stock {symbol}: {e}")

if __name__ == "__main__":
    initialize_db()
    initialize_smart_analytics()
    initialize_trading_history()
    
    # Load bought stocks from smart analytics
    smart_portfolio = load_bought_stocks_from_analytics()
    if smart_portfolio:
        portfolio.update(smart_portfolio)
        logger.info(f"📊 Loaded {len(smart_portfolio)} bought stocks from smart analytics")
    
    # Check for pending analysis from previous day
    pending_count = len(get_pending_analysis())
    if pending_count > 0:
        logger.info(f"📋 Found {pending_count} stocks pending analysis from previous day")
    
    select_top_stocks() # comment this out if you want to run the script without selecting stocks
    logger.info("Starting stock trading script. Setting up jobs...")

    # --- Scheduler Setup ---
    # Daily job at 10:15 AM to analyze and select stocks for intraday trading
    # This runs after market stabilizes to get better technical signals
    schedule.every().day.at("10:15").do(select_top_stocks)
    logger.info("Scheduled: Daily stock selection at 10:15 AM (after market stabilizes).")


    # Let's define market hours
    market_open_time = dt_time(9, 10) # 9:10 AM
    market_close_time = dt_time(15, 0) # 3:00 PM



    def calculate_portfolio_volatility():
        """Calculate current portfolio volatility to determine monitoring frequency"""
        # Load bought stocks from smart analytics for volatility calculation
        smart_portfolio = load_bought_stocks_from_analytics()
        if not smart_portfolio:
            return 0.0
        
        total_volatility = 0.0
        stock_count = 0
        
        try:
            for symbol in smart_portfolio.keys():
                # Fetch recent data to calculate volatility
                recent_data = fetch_market_data([symbol], period="1d", interval="5m")
                if symbol in recent_data and recent_data[symbol] is not None:
                    prices = recent_data[symbol]['Close']
                    if len(prices) > 10:
                        # Calculate 1-hour volatility (last 12 data points of 5-min intervals)
                        recent_prices = prices.tail(12)
                        volatility = recent_prices.std() / recent_prices.mean()
                        total_volatility += volatility
                        stock_count += 1
        except Exception as e:
            logger.warning(f"⚠️ Error calculating portfolio volatility: {e}")
            return 0.02  # Default moderate volatility
        
        return total_volatility / stock_count if stock_count > 0 else 0.0

    def get_smart_monitoring_frequency():
        """Determine optimal monitoring frequency based on market conditions"""
        now = datetime.now().time()
        
        # Last 30 minutes before market close - check every minute (critical period)
        if dt_time(14, 30) <= now < dt_time(15, 0):
            return 1, "End-of-day critical period"
        
        # Calculate portfolio volatility
        volatility = calculate_portfolio_volatility()
        
        if volatility > 0.04:  # High volatility (>4%)
            return 2, f"High volatility detected ({volatility:.1%})"
        elif volatility > 0.025:  # Medium volatility (>2.5%)
            return 3, f"Medium volatility detected ({volatility:.1%})"
        else:  # Normal volatility
            return 5, f"Normal market conditions ({volatility:.1%})"

    # Global variable to track last monitoring time
    last_monitoring_time = datetime.min
    
    def smart_monitoring_job():
        """Smart monitoring that adjusts frequency based on market conditions"""
        global last_monitoring_time
        
        now = datetime.now()
        current_time = now.time()
        
        # Only run on weekdays during market hours
        if now.weekday() >= 5 or not (market_open_time <= current_time < market_close_time):
            return
        
        # Check if there are bought stocks in smart analytics (fresh check each time)
        smart_portfolio = load_bought_stocks_from_analytics()
        if not smart_portfolio:
            logger.debug("⏭️ Smart monitoring: No stocks marked as 'bought' in smart analytics")
            return
        
        # Get smart frequency
        frequency_minutes, reason = get_smart_monitoring_frequency()
        
        # Check if enough time has passed since last monitoring
        time_since_last = (now - last_monitoring_time).total_seconds() / 60
        
        if time_since_last >= frequency_minutes:
            logger.info(f"🔍 Smart monitoring: {frequency_minutes}-min frequency ({reason})")
            monitor_and_update_portfolio()
            last_monitoring_time = now
        else:
            logger.debug(f"⏭️ Skipping monitoring: {time_since_last:.1f}/{frequency_minutes} min elapsed")

    # Smart frequency monitoring - starts with 5 minutes, adjusts based on market conditions
    schedule.every(1).minutes.do(smart_monitoring_job)
    logger.info("Scheduled: Smart portfolio monitoring (1-5 min frequency based on market conditions).")

    # End of day assessment job
    # Runs shortly before market close, e.g., 2:55 PM
    schedule.every().day.at("14:55").do(end_of_day_assessment)
    logger.info("Scheduled: End-of-day assessment at 2:55 PM.")
    
    logger.info("--- Waiting for scheduled jobs to run. Press Ctrl+C to exit. ---")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Script stopped by user.")
    finally:
        logger.info("Stock trading script finished.")