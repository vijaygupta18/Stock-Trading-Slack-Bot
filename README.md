# 🤖 AI-Powered Intraday Trading Bot

> **Advanced automated trading system with Gemini AI integration, technical analysis, and intelligent portfolio management**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Trading](https://img.shields.io/badge/Trading-Intraday-orange.svg)](https://github.com)

## 🌟 Overview

This is a sophisticated intraday trading bot that combines AI-powered stock analysis with technical indicators to make intelligent trading decisions. The system uses Google's Gemini AI for market analysis and provides automated stock recommendations with comprehensive risk management.

### ✨ Key Features

- 🧠 **AI-Powered Analysis**: Gemini AI integration for intelligent stock selection and market analysis
- 📊 **Advanced Technical Analysis**: RSI, Bollinger Bands, Moving Averages, Support/Resistance levels
- 📱 **Real-time Slack Integration**: Instant notifications for recommendations, sells, and performance
- 🛡️ **Comprehensive Risk Management**: Stop-loss, profit targets, position sizing, and time-based rules
- 📈 **Smart Portfolio Tracking**: Automated P&L calculation and detailed trade history
- 🔄 **Intelligent File Management**: Clean separation of active positions and completed trades
- ⏰ **Automated Scheduling**: Configurable trading hours and smart monitoring frequency
- 🧹 **Smart Cleanup System**: Automatic organization with CLEAN + APPEND methodology

## 🏗️ System Architecture

### 📁 File Structure

```
Trading Bot/
├── trading.py                 # 🤖 Main trading bot engine (2000+ lines)
├── apply_sell_alerts.py       # 📋 Manual sell alert processor
├── smart_analytics.json       # 📊 Active recommendations (CURRENT trades)
├── trading_history.json       # 📈 Completed trades with P&L statistics
├── trading_data.json          # 💾 System data storage
├── trading_log.log           # 📝 Detailed system logs
├── setup.py                  # 🛠️ Automated setup and verification
├── requirements.txt          # 📦 Python dependencies
├── .env                      # 🔐 Environment variables (CREATE THIS)
├── .gitignore               # 🚫 Git protection for sensitive files
└── README.md                # 📖 This comprehensive documentation
```

### 🗂️ Smart Data Management System

The bot uses a sophisticated **CLEAN + APPEND** file management system:

| File | Purpose | Behavior | Content |
|------|---------|----------|---------|
| `smart_analytics.json` | **ACTIVE Trading** | APPEND new, KEEP bought, REMOVE old | Current recommendations, bought stocks |
| `trading_history.json` | **COMPLETED Trades** | AUTO-POPULATE from sells | P&L data, performance statistics |
| `trading_data.json` | **System Data** | PERSISTENT storage | Bot configurations and metadata |

#### Smart Cleanup Process:
1. **🧹 CLEANUP**: Removes old `"not_bought"` stocks automatically
2. **✅ PRESERVE**: Keeps all `"bought"` stocks for continued monitoring  
3. **🆕 APPEND**: Adds fresh recommendations without losing active positions

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to project
cd "Trading Bot"

# Install dependencies
pip install -r requirements.txt

# Run automated setup
python setup.py
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token

# Configuration
SLACK_DEFAULT_CHANNEL=#trading-alerts
CAPITAL=500
```

**Get API Keys:**
- **Gemini AI**: https://aistudio.google.com/app/apikey
- **Slack Bot**: https://api.slack.com/apps (requires `chat:write` and `channels:read` scopes)

### 3. Run the Bot

```bash
# Start the trading bot
python trading.py

# Apply manual sell alerts (if needed)
python apply_sell_alerts.py

# Monitor real-time logs
tail -f trading_log.log
```

## ⚙️ Configuration

### 🎯 Trading Parameters

All trading parameters are configurable via global constants in `trading.py`:

```python
# --- Trading Parameters (Global Constants) ---
PROFIT_TARGET_PERCENT = 2.0         # Minimum profit % to trigger sell
MINIMUM_HOLD_TIME_MINUTES = 10      # Minimum hold time before selling
MINIMUM_PROFIT_PROTECTION_PERCENT = 2.0  # Profit protection threshold
EARLY_TRADING_LOSS_THRESHOLD = -3.0 # Early hours loss threshold
FALLBACK_PROFIT_THRESHOLD = 2.0     # Fallback profit threshold
CAPITAL = 500                        # Trading capital
```

### 📊 AI Model Configuration

```python
# Gemini AI Model Settings
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
```

## 🎛️ Core Features

### 🧠 AI-Powered Stock Selection

- **Gemini AI Integration**: Advanced market analysis and stock selection
- **Technical Scoring**: Comprehensive technical analysis scoring system
- **Risk Assessment**: Automated risk evaluation for each recommendation
- **Market Context**: Real-time market data analysis

### 📈 Technical Analysis Engine

| Indicator | Purpose | Implementation |
|-----------|---------|----------------|
| **RSI** | Momentum analysis | 14-period RSI with overbought/oversold detection |
| **Bollinger Bands** | Volatility & mean reversion | 20-period with 2 std deviation |
| **Moving Averages** | Trend analysis | SMA (5,10,20) and EMA (9,21) |
| **Support/Resistance** | Key price levels | Dynamic S/R level calculation |
| **Volume Analysis** | Market participation | Volume patterns and price correlation |

### 🛡️ Risk Management

- **Position Sizing**: Dynamic position sizing based on volatility
- **Stop Loss**: Configurable stop-loss percentages
- **Profit Targets**: Automated profit-taking at target levels
- **Time-based Rules**: Minimum hold times and trading hour restrictions
- **Portfolio Limits**: Maximum positions and capital allocation

### 📱 Intelligent Slack Integration

#### 🔔 Alert Types:
- **🌅 Morning Recommendations**: Daily AI-selected stock picks with scores
- **🔔 Immediate Sell Alerts**: Real-time notifications when stocks are sold
- **📊 End-of-Day Summary**: Performance summary only if trades occurred
- **⚠️ System Alerts**: Important bot status and error notifications

#### Sample Slack Notification:
```
📊 Smart Analytics Updated! (CLEAN + APPEND)

🧹 Removed: 3 old 'not_bought' stocks
📊 Kept: 2 'bought' stocks for monitoring  
✅ Added: 5 fresh recommendations
📋 Total active: 7 stocks

📋 NEW Fresh Recommendations:
• ITC: ₹416.80 (Score: 65/100) - High liquidity and consistent volume
• NTPC: ₹330.85 (Score: 55/100) - Stable power sector play
• RVNL: ₹381.65 (Score: 50/100) - Railway sector with momentum

📊 Continued Monitoring (Bought Stocks):
• RELIANCE: ₹2,450.00 (Status: bought, P&L: +₹25.50)
• TCS: ₹3,200.00 (Status: bought, P&L: -₹15.30)
```

## 🔄 Complete System Workflow

### 🌅 Daily Operations

1. **Pre-Market Setup** (9:00 AM): System initialization and file cleanup
2. **Market Analysis** (9:15 AM): AI generates fresh stock recommendations  
3. **Smart Cleanup**: Removes old recommendations, preserves bought stocks
4. **Continuous Monitoring**: Real-time tracking of bought positions
5. **Intelligent Sell Decisions**: AI + technical analysis for exit signals
6. **End-of-Day Summary**: Performance review and file organization

### 📊 Stock Lifecycle Management

```
AI Analysis → Recommendation → Manual Purchase → Active Monitoring
                    ↓               ↓               ↓
             Slack Alert    Update Status    Sell Signal?
                                                ↓
                                        Auto Sell → Trading History
```

### 🔄 Smart File Management Process

**Before Cleanup (mixed old and new):**
```json
{
  "current_recommendations": [
    {"symbol": "ACTIVESTOCK1", "status": "bought"},     // ✅ KEEP - Active position
    {"symbol": "OLDSTOCK1", "status": "not_bought"},    // ❌ REMOVE - Old recommendation
    {"symbol": "OLDSTOCK2", "status": "not_bought"},    // ❌ REMOVE - Old recommendation
    {"symbol": "ACTIVESTOCK2", "status": "bought"}      // ✅ KEEP - Active position
  ]
}
```

**After Cleanup (organized and fresh):**
```json
{
  "current_recommendations": [
    {"symbol": "ACTIVESTOCK1", "status": "bought"},     // ✅ KEPT - Continue monitoring
    {"symbol": "ACTIVESTOCK2", "status": "bought"},     // ✅ KEPT - Continue monitoring
    {"symbol": "FRESHSTOCK1", "status": "not_bought"},  // 🆕 NEW - Fresh recommendation
    {"symbol": "FRESHSTOCK2", "status": "not_bought"}   // 🆕 NEW - Fresh recommendation
  ]
}
```

## 🎮 Complete Usage Guide

### 📋 Daily Trading Workflow

1. **Morning Review** (9:15 AM)
   - Check Slack for fresh AI recommendations
   - Review technical scores and AI reasoning
   - Assess market conditions and risk factors

2. **Manual Purchase Process**
   - Buy recommended stocks through your broker
   - Note actual purchase prices and quantities

3. **System Update**
   - Edit `smart_analytics.json` to mark stocks as bought:
   ```json
   {
       "symbol": "ITC",
       "status": "bought",           // Change from "not_bought"
       "actual_buy_price": 417.00,   // Your actual purchase price
       "actual_quantity": 10,        // Number of shares bought
       "buy_timestamp": "2024-01-15T10:30:00"
   }
   ```

4. **Automated Monitoring**
   - Bot continuously monitors bought stocks
   - Receives instant Slack alerts for sell decisions
   - System handles P&L calculations automatically

5. **Performance Review**
   - Check `trading_history.json` for completed trades
   - Review daily performance summaries
   - Analyze win/loss ratios and average profits

### 🔧 System Administration

```bash
# Monitor system status
python -c "from trading import *; print('System Status: OK')"

# Check active positions
python -c "from trading import load_bought_stocks_from_analytics; print(load_bought_stocks_from_analytics())"

# Manual cleanup (if needed)
python -c "from trading import cleanup_sold_stocks_from_analytics; cleanup_sold_stocks_from_analytics()"

# View real-time logs
tail -f trading_log.log | grep -E "(BUY|SELL|ERROR)"

# Test AI connection
python -c "import google.generativeai as genai; print('AI Status: Connected')"
```

## 📊 Performance Tracking & Analytics

### 💰 Automatic Statistics Tracking

The system maintains comprehensive performance metrics:

```json
{
    "total_trades": 25,
    "total_profit_loss": 450.75,
    "summary_stats": {
        "profitable_trades": 18,
        "losing_trades": 7,
        "average_pl": 18.03,
        "win_rate": 72.0,
        "best_trade": 85.50,
        "worst_trade": -25.30
    }
}
```

### 📈 Real-time Monitoring Features

- **Portfolio Volatility**: Dynamic risk assessment
- **Position Correlation**: Diversification analysis  
- **Market Timing**: Entry/exit timing optimization
- **Sector Exposure**: Industry diversification tracking

## 🔐 Security & Best Practices

### 🛡️ Environment Security

**✅ DO:**
- Store all secrets in `.env` file
- Use environment variables exclusively
- Rotate API keys monthly
- Monitor API usage regularly

**❌ DON'T:**
- Hardcode API keys in source code
- Commit `.env` files to git
- Share trading data publicly
- Use production keys for testing

### 🚫 Automatic Protection

The `.gitignore` file protects:
- ✅ Environment variables (`.env*`)
- ✅ Trading data files (`*.json`)
- ✅ Log files (`*.log`)
- ✅ API keys and secrets
- ✅ Database and backup files

### 🔒 Data Security

```bash
# Create encrypted backups
tar -czf trading_backup_$(date +%Y%m%d).tar.gz *.json
gpg -c trading_backup_$(date +%Y%m%d).tar.gz
rm trading_backup_$(date +%Y%m%d).tar.gz
```

## 🛠️ Advanced Features

### 🎯 Smart Monitoring System

- **Adaptive Frequency**: Monitoring frequency adjusts based on market volatility
- **Selective Tracking**: Only monitors stocks marked as 'bought'
- **Context-Aware Alerts**: Notifications include market context and reasoning

### 🔧 Extensible Architecture

- **Plugin System**: Easy to add new technical indicators
- **Multiple Data Sources**: Yahoo Finance with fallback options
- **Customizable Rules**: Flexible rule engine for trading logic
- **AI Model Switching**: Support for different AI models

### 📊 Advanced Analytics

- **Correlation Analysis**: Portfolio correlation matrix
- **Risk Metrics**: VaR, Sharpe ratio, maximum drawdown
- **Performance Attribution**: Sector and timing analysis

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **No Slack notifications** | Missing alerts | Check `SLACK_BOT_TOKEN` in `.env` |
| **AI analysis fails** | No recommendations | Verify `GEMINI_API_KEY` configuration |
| **Market data errors** | Empty DataFrames | Check internet connection and yfinance |
| **JSON file corruption** | Loading errors | Restore from backup or run `setup.py` |
| **Import errors** | Module not found | Run `pip install -r requirements.txt` |

### 📝 Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Test individual components
from trading import fetch_market_data, analyze_with_gemini
data = fetch_market_data(['ITC'], period="1d")
print(f"Data shape: {data.shape}")

# Test AI connection
response = analyze_with_gemini("Test prompt", "Sample data")
print(f"AI Response: {response[:100]}...")
```

### 🆘 Recovery Procedures

```bash
# Reset data files
python setup.py

# Restore from backup
cp smart_analytics.json.backup smart_analytics.json

# Clear logs
> trading_log.log

# Restart with clean state
rm -f trading_data.json && python trading.py
```

## 🚨 Important Disclaimers

⚠️ **Investment Risk**: This is an automated trading system. All investments carry risk of loss.

⚠️ **Not Financial Advice**: This system is for educational and research purposes only.

⚠️ **Testing Required**: Thoroughly test with paper trading before using real money.

⚠️ **Market Risks**: Automated systems can fail during extreme market conditions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for intelligent market analysis capabilities
- **Yahoo Finance API** for reliable real-time market data
- **Slack API** for seamless communication integration
- **TA-Lib** for advanced technical analysis indicators
- **Python Community** for excellent libraries and tools

---

**⚡ Built with ❤️ for intelligent trading**

> **Remember**: Always trade responsibly, start with paper trading, and never invest more than you can afford to lose. This system is a tool to assist your trading decisions, not replace your judgment. 