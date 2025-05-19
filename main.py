from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return "Solana Trading Bot is running! Use Telegram to interact with it."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)