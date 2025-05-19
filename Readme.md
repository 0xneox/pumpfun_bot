# Solana Trading Bot (Telegram + Flask)

This project is a pumpfun trading bot with a Telegram interface and a simple Flask web server for status monitoring. It allows users to interact with the bot via Telegram commands and inline buttons.

---

## Features
- Trade Solana tokens via Telegram commands and buttons
- Automatic and manual trading modes
- Web server for status display
- Multi-wallet management

---

## Prerequisites
- Python 3.11+
- Telegram account and bot token (from @BotFather)

---

## Installation & Setup

1. **Clone or Download the Repository**
2. **Install Python dependencies:**
   ```sh
   pip install aiohttp base58 cryptography email-validator flask flask-sqlalchemy gunicorn psycopg2-binary python-dotenv python-telegram-bot solana solders telegram trafilatura websockets
   ```
3. **Configure Environment Variables:**
   - Copy the provided `.env` file and fill in your Telegram bot token and any other required secrets.
   - Example `.env`:
     ```env
     TELEGRAM_TOKEN=your-telegram-bot-token
     # Add other required variables if present
     ```
4. **Set up config.json** (if needed):
   - Edit `config.json` to adjust trading preferences or wallet addresses as required.

---

## Running the App

You must run both the web server and the Telegram bot separately:

1. **Start the Flask Web Server:**
   ```sh
   python main.py
   ```
   - Access the status page at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

2. **Start the Telegram Bot:**
   ```sh
   python bot.py
   ```
   - Interact with your bot via Telegram (search for your bot username).

---

## Telegram Bot Usage & Buttons

This bot is designed to be beginner-friendly! Here’s a detailed guide on every command and button, with practical examples and what you should expect to see.

### Typical User Flow
1. **Start the bot**: Type `/start` in your Telegram chat with the bot.
2. **Configure your preferences** (optional): Use `/configure` or `/settoken`.
3. **Start trading**: Use `/start_trading` or tap the 'Start Trading' button.
4. **Monitor status**: Use `/status` or tap 'Status'.
5. **Pause/stop trading or sell tokens**: Use `/pause_trading`, `/stop_trading`, `/sell_all` or corresponding buttons.
6. **Get help**: Use `/help` at any time.

### Command Explanations & Examples

- **/start**
  - _What it does:_ Initializes the bot and shows the main menu with buttons.
  - _Example:_
    ```
    /start
    ```
  - _Bot reply:_
    "Welcome to the Solana Trading Bot!" + buttons: [Start Trading] [Status] [Configure] [Help]

- **/status**
  - _What it does:_ Shows your current trading status, wallet balances, and active token.
  - _Example:_
    ```
    /status
    ```
  - _Bot reply:_
    "Trading: ACTIVE\nWallet Balance: 2.5 SOL\nCurrent Token: ..." + buttons: [Pause Trading] [Sell All]

- **/start_trading**
  - _What it does:_ Starts automatic trading with your current configuration.
  - _Example:_
    ```
    /start_trading
    ```
  - _Bot reply:_
    "Trading started!" + buttons: [Pause Trading] [Stop Trading]

- **/pause_trading**
  - _What it does:_ Pauses trading without closing positions.
  - _Example:_ `/pause_trading`
  - _Bot reply:_ "Trading paused." + buttons: [Resume Trading] [Stop Trading]

- **/stop_trading**
  - _What it does:_ Stops all trading activity and closes open positions.
  - _Example:_ `/stop_trading`
  - _Bot reply:_ "Trading stopped. All positions closed."

- **/sell_all**
  - _What it does:_ Sells all tokens in your wallet at current market prices.
  - _Example:_ `/sell_all`
  - _Bot reply:_ "All tokens sold. Wallet balance: ..."

- **/configure**
  - _What it does:_ Opens a configuration menu to set trading parameters (like amount, risk, etc.).
  - _Example:_ `/configure`
  - _Bot reply:_ "Configure your trading preferences:" + buttons: [Set Token] [Set Amount] [Auto Trading]

- **/settoken <token_mint>**
  - _What it does:_ Sets the specific Solana token to trade by its mint address.
  - _Example:_
    ```
    /settoken 2n4jz... (your token mint)
    ```
  - _Bot reply:_ "Token set to: 2n4jz..."

- **/auto_trading**
  - _What it does:_ Enables or disables automatic token selection for trading.
  - _Example:_ `/auto_trading`
  - _Bot reply:_ "Auto trading enabled/disabled."

- **/help**
  - _What it does:_ Shows a help message with all commands and usage tips.
  - _Example:_ `/help`
  - _Bot reply:_ "Here are the available commands..."

---

### Button Actions with Examples

Whenever you use a command or the bot sends you a message, you’ll often see buttons below the message. Here’s how to use them:

- **Start Trading**: Tap this button to begin trading (same as `/start_trading`).
- **Pause Trading**: Tap to pause trading (same as `/pause_trading`).
- **Stop Trading**: Tap to fully stop trading and close positions (same as `/stop_trading`).
- **Sell All**: Tap to sell all tokens in your wallet (same as `/sell_all`).
- **Status**: Tap to get an instant update on your trading and wallet status.
- **Configure**: Tap to open a menu for setting your trading preferences.
- **Set Token**: Tap to set a new token for trading (bot will prompt you to enter or select a token mint address).
- **Auto Trading**: Tap to toggle automatic token selection.
- **Help**: Tap to see help and tips.

#### Example Interaction
1. Type `/start` in your Telegram chat with the bot.
2. Tap [Start Trading] to begin.
3. Tap [Pause Trading] if you want to pause.
4. Tap [Sell All] to instantly sell your tokens.
5. Use [Configure] to set your trading preferences at any time.

_The bot will always guide you with buttons and clear messages for every step. If you’re unsure, type `/help` or tap the Help button!_

---

### Configuring the Bot

This section explains how to set up and adjust your trading preferences using the configure commands and buttons.

#### `/configure`
- **What it does:** Opens a menu to set trading parameters like token, amount, and auto-trading.
- **How to use:**
  - Type `/configure` in your chat.
  - The bot will reply with configuration options and buttons: [Set Token], [Set Amount], [Auto Trading], and more.

#### `/settoken <token_mint>`
- **What it does:** Sets the specific Solana token to trade, using its mint address.
- **How to use:**
  - Type `/settoken` followed by your token's mint address, e.g.:
    ```
    /settoken 2n4jz... (replace with your token mint)
    ```
  - The bot will confirm: "Token set to: 2n4jz..."

#### `/auto_trading`
- **What it does:** Enables or disables automatic token selection. When enabled, the bot will choose trending or profitable tokens for you.
- **How to use:**
  - Type `/auto_trading` to toggle.
  - The bot will reply: "Auto trading enabled." or "Auto trading disabled."

#### `/config <parameter> <value>`
- **What it does:** Directly sets a configuration parameter to a specific value.
- **How to use:**
  - Type `/config` followed by the parameter name and the value you want to set, e.g.:
    ```
    /config buy_interval 10
    ```
  - The bot will confirm the update or show an error if the value is invalid.

#### Configuration Parameters
Below are all the available configuration parameters you can set, with explanations and examples:

| Parameter           | Type    | Description                                                      | Example Command                         |
|---------------------|---------|------------------------------------------------------------------|------------------------------------------|
| buy_interval        | int     | Time (in seconds) between buy operations                         | `/config buy_interval 10`                |
| sell_interval       | int     | Time (in seconds) between sell operations                        | `/config sell_interval 15`               |
| buy_amount_min      | float   | Minimum amount of SOL to use for each buy                        | `/config buy_amount_min 0.05`            |
| buy_amount_max      | float   | Maximum amount of SOL to use for each buy                        | `/config buy_amount_max 0.2`             |
| sell_ratio          | float   | Portion of tokens to sell in each operation (0.1 - 1.0)          | `/config sell_ratio 0.75`                |
| slippage            | float   | Slippage tolerance (0.01 - 0.20)                                 | `/config slippage 0.05`                  |
| max_cap             | int     | Maximum market cap in USD for tokens to trade                    | `/config max_cap 20000`                  |
| num_wallets         | int     | Number of proxy wallets to use                                   | `/config num_wallets 5`                  |
| target_multiplier   | float   | Target price multiplier for taking profit                        | `/config target_multiplier 10`           |

**Parameter Details:**
- `buy_interval`: How often the bot will attempt to buy tokens (in seconds).
- `sell_interval`: How often the bot will attempt to sell tokens (in seconds).
- `buy_amount_min` & `buy_amount_max`: The minimum and maximum amount of SOL the bot will use for each buy operation. The bot will randomly choose an amount in this range for each buy.
- `sell_ratio`: The fraction of your token holdings to sell in each operation. 1.0 means sell everything, 0.5 means sell half, etc.
- `slippage`: The maximum acceptable price slippage for trades. Lower values are safer but may cause failed trades.
- `max_cap`: The highest market cap (in USD) of tokens the bot will consider trading.
- `num_wallets`: How many proxy wallets the bot will use for trading. More wallets can help spread risk.
- `target_multiplier`: The price multiplier at which the bot will try to take profit.

#### Configuration Buttons
- **Set Token:** Tap to set a new token for trading. The bot will prompt you to enter or select a token mint address.
- **Set Amount:** Tap to specify how much SOL to use per trade (if available).
- **Auto Trading:** Tap to enable or disable automatic token selection.
- **Other buttons** may appear for advanced parameters.

#### Example Configuration Flow
1. Type `/configure` to open the configuration menu.
2. Tap [Set Token] and enter your token mint address, or use `/settoken <token_mint>` directly.
3. Use `/config` commands (see table above) to fine-tune your trading strategy.
4. Tap [Set Amount] (if available) to set your trading amount.
5. Tap [Auto Trading] to enable or disable automatic token selection.
6. Use `/status` to review your current configuration.

_These commands and buttons make it easy for anyone to set up the bot to their preferences, even with no prior experience!_

---

## Notes
- Make sure both `main.py` and `bot.py` are running for full functionality.
- Ensure your `.env` and `config.json` are correctly set up for your wallets and preferences.
- Logs and wallet data are stored in the `logs/` directory.

---

## Troubleshooting
- If buttons do not work, ensure `bot.py` is running and your Telegram bot token is correct.
- For dependency issues, double-check your Python version and installed packages.
- For Solana RPC or trading issues, check your network connection and wallet configuration.

---

## License
This project is for educational and research purposes only. Use at your own risk.
