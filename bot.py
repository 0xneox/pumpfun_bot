import os
import json
import time
import random
import logging
import asyncio
import aiohttp
import base58
from typing import List, Optional, Dict, Tuple, Any, Union
from dotenv import load_dotenv
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction
from solders.instruction import Instruction, AccountMeta
from solders.message import Message
from solana.rpc.types import TxOpts

# Custom functions to replace spl.token functionality
def get_associated_token_address(owner: Pubkey, mint: Pubkey) -> Pubkey:
    """Derive the associated token account address for a given wallet address and token mint."""
    seeds = [
        bytes(owner),
        bytes(TOKEN_PROGRAM_ID),
        bytes(mint)
    ]
    addr, _ = Pubkey.find_program_address(seeds, ASSOCIATED_TOKEN_PROGRAM_ID)
    return addr

def create_associated_token_account(payer: Pubkey, owner: Pubkey, mint: Pubkey, 
                                  program_id=None, token_program_id=None) -> Instruction:
    """Create instruction to initialize an associated token account."""
    if program_id is None:
        program_id = ASSOCIATED_TOKEN_PROGRAM_ID
    if token_program_id is None:
        token_program_id = TOKEN_PROGRAM_ID
        
    # Derive the associated token account address
    account = get_associated_token_address(owner, mint)
    
    # Define account metadata for the instruction
    keys = [
        AccountMeta(pubkey=payer, is_signer=True, is_writable=True),
        AccountMeta(pubkey=account, is_signer=False, is_writable=True),
        AccountMeta(pubkey=owner, is_signer=False, is_writable=False),
        AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
        AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        AccountMeta(pubkey=token_program_id, is_signer=False, is_writable=False),
        AccountMeta(pubkey=RENT_SYSVAR, is_signer=False, is_writable=False),
    ]
    
    # Return the instruction
    return Instruction(program_id=program_id, accounts=keys, data=bytes([1]))
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from cryptography.fernet import Fernet
import websockets
from uuid import uuid4

# Load environment variables
load_dotenv()

# Configuration
RPC_URLS = [
    os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com"),
]
MAIN_WALLET_SECRET = os.getenv("MAIN_WALLET_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID", "")  # Make optional
X_API_KEY = os.getenv("X_API_KEY")

# Only check for required bot token and wallet
if not all([MAIN_WALLET_SECRET, TELEGRAM_BOT_TOKEN]):
    raise ValueError("Missing required environment variables!")

try:
    MAIN_WALLET = Keypair.from_bytes(base58.b58decode(MAIN_WALLET_SECRET))
except Exception as e:
    raise ValueError(f"Invalid MAIN_WALLET_SECRET: {e}")

# Program IDs
if 'PUMP_FUN_PROGRAM' not in globals():
    PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
if 'PUMP_SWAP_PROGRAM' not in globals():
    # Assuming PumpSwap might use the same program ID or a specific one if known.
    # Defaulting to the same as PUMP_FUN_PROGRAM if not specified.
    # If PumpSwap has a distinct, known program ID, it should be used here.
    PUMP_SWAP_PROGRAM = Pubkey.from_string("pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA") 
    # Example if different: PUMP_SWAP_PROGRAM = Pubkey.from_string("pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA")
RAYDIUM_AMM_PROGRAM = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
SYSTEM_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
RENT_SYSVAR = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
WSOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")

# Constants
NUM_WALLET = 5
INITIAL_FUNDING = 0.01 * 1e9  # 0.01 SOL per wallet
TRADING_FEE = 0.0025
MIN_SOL_OUT = 0.00005
DEFAULT_CONFIG = {
    "buy_interval": 10,
    "sell_interval": 15,
    "buy_amount_min": 0.05,
    "buy_amount_max": 0.2,
    "sell_ratio": 0.75,
    "slippage": 0.05,
    "max_cap": 20000,
    "num_wallets": 5,
    "target_multiplier": 10,
    "auto_mode": False,           # Auto token selection mode
    "profit_threshold": 0.2,      # 20% minimum projected profit for auto mode
    "auto_sell_profit": 0.5,      # 50% profit target for auto sell
    "auto_sell_loss": -0.1,       # 10% loss limit for auto sell
    "scan_interval": 300          # 5 minutes between auto scans
}

# Logging Setup
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "bot.log"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Key Encryption
fernet_key = Fernet.generate_key()
cipher = Fernet(fernet_key)

# Global State
client = None
wallets = []
config = DEFAULT_CONFIG.copy()
trading_active = False
paused = False
current_token = None
token_decimals = 9
is_pump_fun = False
trading_task = None
sellall_triggered = False
auto_token_selection_task = None
auto_selected_tokens = []  # List of automatically selected tokens with potential

# Config Persistence
config_file = os.path.join(os.path.dirname(__file__), "config.json")
def save_config():
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

def load_config():
    global config
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config.update(json.load(f))
    else:
        save_config()

# Wallet Management
wallet_keys_file = os.path.join(log_dir, "proxy_wallets.txt")
def load_wallets():
    global wallets
    if os.path.exists(wallet_keys_file):
        with open(wallet_keys_file, "r") as f:
            for line in f:
                try:
                    secret = base58.b58decode(line.strip())
                    wallets.append(Keypair.from_bytes(secret))
                except Exception as e:
                    logging.error(f"Error loading wallet: {e}")
    if len(wallets) < config["num_wallets"]:
        new_wallets = [Keypair() for _ in range(config["num_wallets"] - len(wallets))]
        wallets.extend(new_wallets)
        with open(wallet_keys_file, "w") as f:
            for wallet in wallets:
                f.write(base58.b58encode(wallet.to_bytes()).decode() + "\n")

# RPC Connection
async def init_client():
    global client
    for url in RPC_URLS:
        try:
            client = AsyncClient(url)
            blockhash = await client.get_latest_blockhash()
            if blockhash.value:
                logging.info(f"Connected to RPC: {url}")
                return
        except Exception as e:
            logging.warning(f"Failed to connect to {url}: {e}")
    raise ValueError("Could not connect to any RPC endpoint")

# Helper Functions
async def retry_rpc(func, *args, **kwargs):
    max_attempts = 7
    for attempt in range(max_attempts):
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logging.warning(f"RPC retry ({func.__name__}, attempt {attempt + 1}): {e}")
            if attempt < len(RPC_URLS) - 1:
                new_url = RPC_URLS[(attempt + 1) % len(RPC_URLS)]
                client._provider.endpoint_uri = new_url
            await asyncio.sleep(min(2 ** attempt, 5))
    raise Exception(f"RPC failed after {max_attempts} retries: {func.__name__}")

async def get_account_info(pubkey: Pubkey):
    info = await client.get_account_info(pubkey)
    if info.value:
        logging.info(f"Account {pubkey} found. Owner: {info.value.owner}")
    else:
        logging.info(f"Account {pubkey} not found or has no data.")
    return info.value

async def send_transaction(instructions: List[Instruction], payer: Keypair, *signers: Keypair):
    blockhash = (await client.get_latest_blockhash()).value.blockhash
    message = Message.new_with_blockhash(instructions, payer.pubkey(), blockhash)
    tx = Transaction.new_unsigned(message)
    all_signers = [payer] + list(signers)
    tx.sign(all_signers, blockhash)
    tx_bytes = bytes(tx)
    logging.info(f"Sending transaction with {len(instructions)} instructions")
    tx_sig = await client.send_transaction(
        tx_bytes,
        opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed, max_retries=3)
    )
    await client.confirm_transaction(tx_sig.value, commitment=Confirmed)
    logging.info(f"Transaction confirmed: {tx_sig.value}")
    return tx_sig

async def fetch_token_data(token_mint: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"https://price.jup.ag/v4/price?ids={token_mint}", timeout=10) as resp:
                data = await resp.json()
                price = float(data["data"].get(token_mint, {}).get("price", 0.0001))
                mint_info = await get_account_info(Pubkey.from_string(token_mint))
                decimals = mint_info.data[44] if mint_info else 9
                return {"decimals": decimals, "price": price, "pool": {"quoteReserve": 0, "baseReserve": 0}}
        except Exception as e:
            logging.warning(f"Price fetch error: {e}")
            return {"decimals": 9, "price": 0.0001, "pool": {"quoteReserve": 0, "baseReserve": 0}}

async def fetch_pumpfun_pool_data(token_mint: str) -> Dict:
    try:
        logging.info(f"Fetching PumpFun pool data for token: {token_mint}")
        token_mint_pubkey = Pubkey.from_string(token_mint)
        
        # Try to get Jupiter price data first (most accurate)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://price.jup.ag/v4/price?ids={token_mint}", timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        jup_price = float(data["data"].get(token_mint, {}).get("price", 0))
                        if jup_price > 0:
                            logging.info(f"Found Jupiter price for {token_mint}: {jup_price}")
                            jup_price_available = True
                        else:
                            jup_price_available = False
                    else:
                        jup_price_available = False
        except Exception as e:
            logging.warning(f"Jupiter price fetch error: {e}")
            jup_price_available = False
        
        # Check if it's on bonding curve
        bonding_curve = Pubkey.find_program_address([b"bonding_curve", bytes(token_mint_pubkey)], PUMP_FUN_PROGRAM)[0]
        logging.info(f"Checking bonding curve at {bonding_curve}")
        account_info = await get_account_info(bonding_curve)
        
        if account_info and str(account_info.owner) == str(PUMP_FUN_PROGRAM):
            logging.info("Token is on PumpFun bonding curve")
            data = account_info.data
            sol_reserve = int.from_bytes(data[32:40], "little")
            token_supply = int.from_bytes(data[40:48], "little")
            
            if sol_reserve > 0 and token_supply > 0:
                price = sol_reserve / token_supply
                logging.info(f"Bonding curve price: {price}, SOL reserve: {sol_reserve}, Token supply: {token_supply}")
            else:
                price = jup_price if jup_price_available else 0.0001
                logging.warning(f"Invalid bonding curve data, using fallback price: {price}")
                
            migrated = False
            pool_address = str(bonding_curve)
        else:
            # Check if it's on PumpSwap
            pool_address = Pubkey.find_program_address([b"pool", bytes(token_mint_pubkey)], PUMP_SWAP_PROGRAM)[0]
            logging.info(f"Checking PumpSwap pool at {pool_address}")
            pool_info = await get_account_info(pool_address)
            
            if pool_info and str(pool_info.owner) == str(PUMP_SWAP_PROGRAM):
                logging.info("Token is on PumpSwap")
                # Extract PumpSwap pool data - specific to PumpSwap's format
                data = pool_info.data
                sol_reserve = int.from_bytes(data[32:40], "little") if len(data) >= 40 else 0
                token_supply = int.from_bytes(data[40:48], "little") if len(data) >= 48 else 0
                
                if sol_reserve > 0 and token_supply > 0:
                    price = sol_reserve / token_supply
                    logging.info(f"PumpSwap price: {price}, SOL reserve: {sol_reserve}, Token supply: {token_supply}")
                else:
                    price = jup_price if jup_price_available else 0.0001
                    logging.warning(f"Invalid PumpSwap data, using fallback price: {price}")
            else:
                # Try to get Raydium pool data
                raydium_data = await fetch_raydium_pool_data(token_mint)
                if raydium_data:
                    logging.info("Token is on Raydium")
                    return raydium_data
                else:
                    # If no pool data found, use Jupiter price or default
                    logging.warning(f"No pool data found for {token_mint}")
                    if jup_price_available:
                        price = jup_price
                    else:
                        price = 0.0001
                    sol_reserve = 0
                    token_supply = 0
            
            migrated = True
            pool_address = str(pool_address)
            
        # Get token decimals from mint account
        mint_info = await get_account_info(token_mint_pubkey)
        if mint_info and len(mint_info.data) >= 45:
            decimals = mint_info.data[44]
            logging.info(f"Token decimals: {decimals}")
        else:
            decimals = 9
            logging.warning(f"Could not get decimals, using default: {decimals}")
        
        return {
            "decimals": decimals,
            "price": price,
            "pool": {"quoteReserve": sol_reserve, "baseReserve": token_supply},
            "migrated": migrated,
            "pool_address": pool_address
        }
    except Exception as e:
        logging.error(f"PumpFun pool fetch error: {e}")
        logging.exception(e)  # Log the full traceback
        return {
            "decimals": 9,
            "price": 0.0001,
            "pool": {"quoteReserve": 0, "baseReserve": 0},
            "migrated": False,
            "pool_address": str(Pubkey.find_program_address([b"pool", bytes(Pubkey.from_string(token_mint))], PUMP_SWAP_PROGRAM)[0])
        }

async def fetch_raydium_pool_data(token_mint: str) -> Optional[Dict]:
    try:
        token_mint_pubkey = Pubkey.from_string(token_mint)
        seeds = [b"amm", bytes(token_mint_pubkey), bytes(WSOL_MINT)] if str(token_mint_pubkey) < str(WSOL_MINT) else [b"amm", bytes(WSOL_MINT), bytes(token_mint_pubkey)]
        pool_address = Pubkey.find_program_address(seeds, RAYDIUM_AMM_PROGRAM)[0]
        account_info = await get_account_info(pool_address)
        if not account_info:
            return None
        data = account_info.data
        quote_reserve = int.from_bytes(data[800:808], "little")
        base_reserve = int.from_bytes(data[808:816], "little")
        price = quote_reserve / base_reserve if base_reserve > 0 else 0.0001
        mint_info = await get_account_info(token_mint_pubkey)
        decimals = mint_info.data[44] if mint_info else 9
        return {
            "decimals": decimals,
            "price": price,
            "pool": {"quoteReserve": quote_reserve, "baseReserve": base_reserve},
            "pool_address": str(pool_address)
        }
    except Exception as e:
        logging.warning(f"Raydium pool fetch error: {e}")
        return None

async def fetch_social_sentiment(token_mint: str) -> float:
    if not X_API_KEY:
        return 0.5
    async with aiohttp.ClientSession() as session:
        try:
            headers = {"Authorization": f"Bearer {X_API_KEY}"}
            async with session.get(f"https://api.x.com/2/tweets/search/recent?query={token_mint}", headers=headers) as resp:
                data = await resp.json()
                tweets = data.get("data", [])
                score = sum(1 for tweet in tweets if "pump" in tweet["text"].lower()) / max(len(tweets), 1)
                return min(score, 1.0)
        except Exception as e:
            logging.warning(f"Social sentiment fetch error: {e}")
            return 0.5

async def is_pump_fun_token(token_mint: Pubkey) -> bool:
    try:
        logging.info(f"Checking if {token_mint} is a PumpFun token using enhanced method.")

        # Primary Method: Check via Pump.fun metadata account
        # The metadata PDA for a token on Pump.fun is typically derived with seeds [b'metadata', bytes(token_mint)]
        # from the PUMP_FUN_PROGRAM.
        # This metadata account's data contains the actual bonding_curve_address at offset 8 (after discriminator).
        try:
            metadata_pda_seeds = [b'metadata', bytes(token_mint)]
            metadata_pda, _ = Pubkey.find_program_address(metadata_pda_seeds, PUMP_FUN_PROGRAM)
            logging.info(f"Checking Pump.fun metadata PDA at {metadata_pda}")
            metadata_info = await get_account_info(metadata_pda) # Uses the enhanced get_account_info

            if metadata_info and str(metadata_info.owner) == str(PUMP_FUN_PROGRAM):
                logging.info(f"Found Pump.fun metadata account for {token_mint}.")
                if len(metadata_info.data) >= 40: # Ensure data is long enough for bonding curve (8 + 32)
                    bonding_curve_from_meta = Pubkey(metadata_info.data[8:40])
                    logging.info(f"Bonding curve from metadata: {bonding_curve_from_meta}")
                    
                    # Verify this bonding curve account itself
                    bc_info = await get_account_info(bonding_curve_from_meta)
                    if bc_info and str(bc_info.owner) == str(PUMP_FUN_PROGRAM):
                        logging.info(f"Token {token_mint} IS a PumpFun token (verified via metadata -> bonding curve). Owner: {PUMP_FUN_PROGRAM}")
                        return True
                    else:
                        logging.warning(f"Bonding curve {bonding_curve_from_meta} from metadata not owned by {PUMP_FUN_PROGRAM} or not found.")
                else:
                    logging.warning(f"Pump.fun metadata account {metadata_pda} data too short to extract bonding curve.")
            else:
                logging.info(f"Pump.fun metadata PDA {metadata_pda} not found or not owned by {PUMP_FUN_PROGRAM}.")

        except Exception as e_meta:
            logging.error(f"Error during Pump.fun metadata check for {token_mint}: {e_meta}")
            # Do not return False here, proceed to fallback methods

        logging.info(f"Falling back to original PDA derivation checks for {token_mint}.")
        # Fallback Method 1: Original PumpFun bonding curve PDA check
        # This was the Pubkey.find_program_address([b"bonding_curve", bytes(token_mint)], PUMP_FUN_PROGRAM)[0] check
        try:
            original_bonding_curve_pda_seeds = [b"bonding_curve", bytes(token_mint)]
            original_bonding_curve_pda, _ = Pubkey.find_program_address(original_bonding_curve_pda_seeds, PUMP_FUN_PROGRAM)
            logging.info(f"Checking original bonding curve PDA at {original_bonding_curve_pda}")
            account_info_orig_bc = await get_account_info(original_bonding_curve_pda)
            
            if account_info_orig_bc and str(account_info_orig_bc.owner) == str(PUMP_FUN_PROGRAM):
                logging.info(f"Token {token_mint} IS a PumpFun bonding curve token (original PDA check). Owner: {PUMP_FUN_PROGRAM}")
                return True
        except Exception as e_orig_bc:
            logging.error(f"Error during original bonding curve PDA check for {token_mint}: {e_orig_bc}")

        # Fallback Method 2: PumpSwap pool PDA check
        # This was the Pubkey.find_program_address([b"pool", bytes(token_mint)], PUMP_SWAP_PROGRAM)[0] check
        try:
            pool_pda_seeds = [b"pool", bytes(token_mint)]
            pool_pda, _ = Pubkey.find_program_address(pool_pda_seeds, PUMP_SWAP_PROGRAM)
            logging.info(f"Checking PumpSwap pool PDA at {pool_pda}")
            pool_info = await get_account_info(pool_pda)
            
            if pool_info and str(pool_info.owner) == str(PUMP_SWAP_PROGRAM):
                logging.info(f"Token {token_mint} IS a PumpSwap token (pool PDA check). Owner: {PUMP_SWAP_PROGRAM}")
                return True
        except Exception as e_swap:
            logging.error(f"Error during PumpSwap pool PDA check for {token_mint}: {e_swap}")
            
        logging.info(f"Token {token_mint} is NOT a PumpFun/PumpSwap token after all checks.")
        return False
        
    except Exception as e:
        logging.error(f"Outer error in is_pump_fun_token for {token_mint}: {e}")
        logging.exception(e) # Log full traceback for unexpected errors
        return False

async def get_token_price_jupiter(token_mint: Pubkey, vs_token_mint: Pubkey = WSOL_MINT) -> Optional[Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"https://price.jup.ag/v4/price?ids={token_mint}&vsToken={vs_token_mint}", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data["data"].get(str(token_mint), {}).get("price", 0))
                    return {"price": price}
                else:
                    return None
        except Exception as e:
            logging.warning(f"Error fetching Jupiter price: {e}")
            return None

# Token scanners and auto-selection logic
async def fetch_recent_pumpfun_tokens() -> List[Dict]:
    """Fetch recently created PumpFun tokens"""
    logging.info("Fetching recent PumpFun tokens...")
    tokens = []
    try:
        # First, try to get recent PumpFun tokens from their API
        async with aiohttp.ClientSession() as session:
            async with session.get("https://pump.fun/api/tokens?sort=newest&limit=30", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "tokens" in data:
                        for token in data["tokens"]:
                            if "mint" in token:
                                tokens.append({
                                    "mint": token["mint"],
                                    "name": token.get("name", "Unknown"),
                                    "symbol": token.get("symbol", "UNKNOWN"),
                                    "time": token.get("createdAt", 0)
                                })
                        logging.info(f"Found {len(tokens)} recent tokens from PumpFun API")
                        return tokens
                
        # Fallback to scanning recent transactions on the PumpFun program
        signature_resp = await client.get_signatures_for_address(PUMP_FUN_PROGRAM, limit=50)
        if signature_resp and signature_resp.value:
            for sig_info in signature_resp.value:
                tx_resp = await client.get_transaction(sig_info.signature)
                if tx_resp and tx_resp.value and tx_resp.value.transaction.message.instructions:
                    for account in tx_resp.value.transaction.message.account_keys:
                        # Check if this might be a token mint
                        try:
                            info = await get_account_info(Pubkey.from_string(account))
                            if info and len(info.data) >= 45 and info.data[44] <= 18:  # Check for token mint data structure
                                # This looks like a token mint
                                is_pf = await is_pump_fun_token(Pubkey.from_string(account))
                                if is_pf:
                                    tokens.append({
                                        "mint": account,
                                        "name": "Unknown",
                                        "symbol": "UNKNOWN",
                                        "time": sig_info.block_time
                                    })
                        except Exception as e:
                            continue
            
            logging.info(f"Found {len(tokens)} recent tokens from blockchain scanning")
            return tokens
            
        return []
    except Exception as e:
        logging.error(f"Error fetching recent tokens: {e}")
        logging.exception(e)
        return []

async def analyze_token_potential(token_mint: str) -> Dict:
    """Analyze a token's potential for profitable trading"""
    logging.info(f"Analyzing token potential: {token_mint}")
    try:
        # Get basic token data
        is_pf = await is_pump_fun_token(Pubkey.from_string(token_mint))
        pool_data = await fetch_pumpfun_pool_data(token_mint)
        
        current_price = pool_data["price"]
        sol_reserve = pool_data["pool"]["quoteReserve"]
        token_supply = pool_data["pool"]["baseReserve"]
        decimals = pool_data["decimals"]
        
        # Get market cap
        market_cap = await estimate_market_cap(token_mint, current_price)
        
        # Check if the token is too new (not enough price history)
        too_new = sol_reserve < 0.05 * 1e9  # Less than 0.05 SOL in pool
        
        # Simulate small buy to check price impact
        simulated_buy = 0.05 * 1e9  # 0.05 SOL
        buy_price_impact = 0
        
        if sol_reserve > 0 and token_supply > 0:
            # PumpFun's price formula: price = sol_reserve / token_supply
            # After buy: new_price = (sol_reserve + buy_amount) / token_supply
            new_sol_reserve = sol_reserve + simulated_buy
            new_price = new_sol_reserve / token_supply
            buy_price_impact = (new_price - current_price) / current_price
        
        # Get social sentiment
        sentiment = await fetch_social_sentiment(token_mint)
        
        # Estimate growth potential
        # Lower market cap + higher sentiment = better potential
        if market_cap < 1000:  # Under $1000 market cap
            growth_potential = 0.8
        elif market_cap < 10000:  # Under $10k market cap
            growth_potential = 0.6
        elif market_cap < 100000:  # Under $100k market cap
            growth_potential = 0.4
        elif market_cap < 1000000:  # Under $1M market cap
            growth_potential = 0.2
        else:
            growth_potential = 0.1
            
        # Adjust for sentiment
        growth_potential *= (0.5 + sentiment)
        
        # Adjust for price impact (lower is better)
        if buy_price_impact > 0.5:  # More than 50% impact
            growth_potential *= 0.5
        elif buy_price_impact > 0.2:  # More than 20% impact
            growth_potential *= 0.8
            
        # Calculate overall score (0-100)
        score = growth_potential * 100
        
        return {
            "mint": token_mint,
            "is_pump_fun": is_pf,
            "price": current_price,
            "market_cap": market_cap,
            "growth_potential": growth_potential,
            "sentiment": sentiment,
            "price_impact": buy_price_impact,
            "score": score,
            "too_new": too_new
        }
    except Exception as e:
        logging.error(f"Error analyzing token potential: {e}")
        logging.exception(e)
        return {
            "mint": token_mint,
            "is_pump_fun": False,
            "price": 0,
            "market_cap": 0,
            "growth_potential": 0,
            "sentiment": 0,
            "price_impact": 1,
            "score": 0,
            "too_new": True
        }

async def scan_for_profitable_tokens():
    """Scan for potentially profitable tokens"""
    global auto_selected_tokens
    
    logging.info("Starting token scan for profit opportunities...")
    try:
        # Get recent tokens
        recent_tokens = await fetch_recent_pumpfun_tokens()
        logging.info(f"Found {len(recent_tokens)} recent tokens to analyze")
        
        analyzed_tokens = []
        for token in recent_tokens:
            analysis = await analyze_token_potential(token["mint"])
            if analysis["score"] > 0:
                analysis["name"] = token.get("name", "Unknown")
                analysis["symbol"] = token.get("symbol", "UNKNOWN")
                analyzed_tokens.append(analysis)
                
        # Sort by score (highest first)
        analyzed_tokens.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top 5
        top_tokens = analyzed_tokens[:5]
        auto_selected_tokens = top_tokens
        
        logging.info(f"Found {len(top_tokens)} high-potential tokens")
        for i, token in enumerate(top_tokens):
            logging.info(f"Token #{i+1}: {token['mint']} - Score: {token['score']:.2f}, Market Cap: ${token['market_cap']:.2f}")
            
        return top_tokens
    except Exception as e:
        logging.error(f"Error scanning for profitable tokens: {e}")
        logging.exception(e)
        return []

async def auto_token_selection_loop():
    """Main loop for automatic token selection"""
    global trading_active, paused, current_token, auto_selected_tokens, token_decimals, is_pump_fun
    
    while True:
        try:
            if config["auto_mode"] and not trading_active:
                # Scan for profitable tokens
                top_tokens = await scan_for_profitable_tokens()
                
                if top_tokens and len(top_tokens) > 0:
                    # Pick the highest scoring token that meets our criteria
                    for token in top_tokens:
                        if (token["score"] >= 70 and 
                            token["market_cap"] < config["max_cap"] and
                            not token["too_new"]):
                            
                            # Found a promising token - set it and start trading automatically
                            mint = token["mint"]
                            logging.info(f"Auto-selected token: {mint} with score {token['score']:.2f}")
                            
                            # Set the token
                            token_mint_pubkey = Pubkey.from_string(mint)
                            is_pf = await is_pump_fun_token(token_mint_pubkey)
                            pool_data = await fetch_pumpfun_pool_data(mint)
                            
                            # Update global vars
                            current_token = mint
                            token_decimals = pool_data["decimals"]
                            is_pump_fun = is_pf
                            trading_active = True
                            paused = False
                            
                            # Check if wallets are funded
                            wallet_setup = await setup_wallets()
                            if not wallet_setup:
                                logging.error("Failed to set up wallets for auto trading. Skipping auto selection.")
                                trading_active = False
                                continue
                            
                            # Start trading with this token
                            await start_trading_task()
                            break
            
            # Wait before next scan
            await asyncio.sleep(config["scan_interval"])
        except Exception as e:
            logging.error(f"Error in auto token selection loop: {e}")
            logging.exception(e)
            await asyncio.sleep(60)  # Wait a minute before retrying

async def estimate_market_cap(token_mint: str, price: float) -> float:
    logging.info(f"Estimating market cap for token {token_mint} with price {price}")
    try:
        # Try to get SOL/USD price from Jupiter
        sol_usd_price = 0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://price.jup.ag/v4/price?ids=SOL", timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sol_usd_price = float(data["data"].get("SOL", {}).get("price", 0))
                        logging.info(f"SOL/USD price: ${sol_usd_price}")
        except Exception as e:
            logging.warning(f"Error fetching SOL/USD price: {e}")
        
        # Use a reasonable fallback if we couldn't fetch the price
        if sol_usd_price <= 0:
            sol_usd_price = 110.0  # A reasonable SOL price estimate
            logging.info(f"Using fallback SOL/USD price: ${sol_usd_price}")
            
        # Get token supply from the mint account
        mint_info = await get_account_info(Pubkey.from_string(token_mint))
        if not mint_info or len(mint_info.data) < 12:
            logging.warning("Invalid mint info data, cannot calculate market cap")
            return 0
            
        # Extract decimals from the mint account
        decimals = mint_info.data[44] if len(mint_info.data) >= 45 else 9
        logging.info(f"Token decimals for market cap: {decimals}")
        
        # Calculate total supply accounting for decimals
        supply = int.from_bytes(mint_info.data[4:12], "little") / (10 ** decimals)
        logging.info(f"Token supply: {supply}")
        
        # Calculate market cap in USD
        market_cap = supply * price * sol_usd_price
        logging.info(f"Calculated market cap: ${market_cap}")
        
        return market_cap
    except Exception as e:
        logging.error(f"Error calculating market cap: {e}")
        logging.exception(e)
        return 0

class TransactionManager:
    def __init__(self, token_mint: Pubkey, decimals: int, is_pump_fun: bool):
        self.token_mint = token_mint
        self.decimals = decimals
        self.is_pump_fun = is_pump_fun
        self.bonding_curve = Pubkey.find_program_address([b"bonding_curve", bytes(token_mint)], PUMP_FUN_PROGRAM)[0] if is_pump_fun else None
        self.pool = Pubkey.find_program_address([b"pool", bytes(token_mint)], PUMP_SWAP_PROGRAM)[0] if is_pump_fun else None
        self.migrated = False

    def create_ata_instruction(self, owner: Pubkey) -> Instruction:
        """Create instruction to initialize an associated token account"""
        ata = get_associated_token_address(owner, self.token_mint)
        instruction = create_associated_token_account(
            payer=owner,
            owner=owner,
            mint=self.token_mint,
            program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
            token_program_id=TOKEN_PROGRAM_ID
        )
        return instruction

    async def check_ata_exists(self, owner: Pubkey) -> bool:
        """Check if an associated token account exists for the given owner"""
        ata = get_associated_token_address(owner, self.token_mint)
        account_info = await get_account_info(ata)
        return account_info is not None

    async def get_or_create_ata(self, owner: Keypair) -> Tuple[Pubkey, Optional[Instruction]]:
        """Get an associated token account or create instructions to initialize it"""
        ata = get_associated_token_address(owner.pubkey(), self.token_mint)
        account_info = await get_account_info(ata)
        if account_info:
            return ata, None
        else:
            instruction = self.create_ata_instruction(owner.pubkey())
            return ata, instruction

    async def check_migration_status(self):
        """Check if the PumpFun token is migrated to PumpSwap"""
        if not self.is_pump_fun:
            return False
        
        bonding_info = await get_account_info(self.bonding_curve)
        if bonding_info and str(bonding_info.owner) == str(PUMP_FUN_PROGRAM):
            self.migrated = False
            return False
        
        pool_info = await get_account_info(self.pool)
        if pool_info and str(pool_info.owner) == str(PUMP_SWAP_PROGRAM):
            self.migrated = True
            return True
        
        return False

    async def create_buy_instruction(self, buyer: Pubkey, ata: Pubkey, amount_in: int) -> Instruction:
        """Create a buy instruction for the token"""
        if not self.is_pump_fun:
            raise ValueError("Buy instruction creation only supported for PumpFun tokens")
        
        await self.check_migration_status()
        
        if not self.migrated:
            # Old PumpFun protocol
            return Instruction(
                program_id=PUMP_FUN_PROGRAM,
                accounts=[
                    AccountMeta(pubkey=buyer, is_signer=True, is_writable=True),
                    AccountMeta(pubkey=self.bonding_curve, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=self.token_mint, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=ata, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
                ],
                data=bytes([0]) + amount_in.to_bytes(8, "little")
            )
        else:
            # New PumpSwap protocol
            return Instruction(
                program_id=PUMP_SWAP_PROGRAM,
                accounts=[
                    AccountMeta(pubkey=buyer, is_signer=True, is_writable=True),
                    AccountMeta(pubkey=self.pool, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=self.token_mint, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=ata, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
                ],
                data=bytes([0]) + amount_in.to_bytes(8, "little")
            )

    async def create_sell_instruction(self, seller: Pubkey, ata: Pubkey, amount_in: int) -> Instruction:
        """Create a sell instruction for the token"""
        if not self.is_pump_fun:
            raise ValueError("Sell instruction creation only supported for PumpFun tokens")
        
        await self.check_migration_status()
        
        if not self.migrated:
            # Old PumpFun protocol
            return Instruction(
                program_id=PUMP_FUN_PROGRAM,
                accounts=[
                    AccountMeta(pubkey=seller, is_signer=True, is_writable=True),
                    AccountMeta(pubkey=self.bonding_curve, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=self.token_mint, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=ata, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
                ],
                data=bytes([1]) + amount_in.to_bytes(8, "little")
            )
        else:
            # New PumpSwap protocol
            return Instruction(
                program_id=PUMP_SWAP_PROGRAM,
                accounts=[
                    AccountMeta(pubkey=seller, is_signer=True, is_writable=True),
                    AccountMeta(pubkey=self.pool, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=self.token_mint, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=ata, is_signer=False, is_writable=True),
                    AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
                ],
                data=bytes([1]) + amount_in.to_bytes(8, "little")
            )

    async def get_token_balance(self, owner: Pubkey) -> int:
        """Get token balance for the given owner"""
        ata = get_associated_token_address(owner, self.token_mint)
        try:
            account_info = await get_account_info(ata)
            if not account_info:
                return 0
            # Token account balance is at offset 64
            return int.from_bytes(account_info.data[64:72], "little")
        except Exception as e:
            logging.error(f"Error getting token balance: {e}")
            return 0

    async def get_sol_balance(self, owner: Pubkey) -> int:
        """Get SOL balance for the given owner"""
        try:
            response = await client.get_balance(owner)
            return response.value
        except Exception as e:
            logging.error(f"Error getting SOL balance: {e}")
            return 0

    async def estimate_token_out(self, sol_in: int) -> int:
        """Estimate token output for the given SOL input"""
        try:
            if not self.is_pump_fun:
                return int(sol_in * 0.95 * 1e9)  # Rough estimate for non-PumpFun tokens
            
            pool_data = await fetch_pumpfun_pool_data(str(self.token_mint))
            sol_reserve = pool_data["pool"]["quoteReserve"]
            token_supply = pool_data["pool"]["baseReserve"]
            
            # PumpFun's bonding curve formula: x * y = k
            # New supply = token_supply + token_out
            # New reserve = sol_reserve + sol_in
            # (sol_reserve + sol_in) * (token_supply + token_out) = k
            # k = sol_reserve * token_supply
            # Solve for token_out:
            # token_out = (k / (sol_reserve + sol_in)) - token_supply
            
            k = sol_reserve * token_supply
            new_reserve = sol_reserve + sol_in
            new_supply = k / new_reserve if new_reserve > 0 else 0
            token_out = token_supply - new_supply
            
            # Apply slippage tolerance
            token_out = int(token_out * (1 - config["slippage"]))
            
            return max(0, int(token_out))
        except Exception as e:
            logging.error(f"Error estimating token output: {e}")
            return int(sol_in * 0.95 * 1e9)  # Fallback to a simple estimate

    async def estimate_sol_out(self, token_in: int) -> int:
        """Estimate SOL output for the given token input"""
        try:
            if not self.is_pump_fun:
                return int(token_in * 0.95 / 1e9)  # Rough estimate for non-PumpFun tokens
            
            pool_data = await fetch_pumpfun_pool_data(str(self.token_mint))
            sol_reserve = pool_data["pool"]["quoteReserve"]
            token_supply = pool_data["pool"]["baseReserve"]
            
            # Similar bonding curve calculation for selling
            k = sol_reserve * token_supply
            new_supply = token_supply - token_in
            new_reserve = k / new_supply if new_supply > 0 else 0
            sol_out = sol_reserve - new_reserve
            
            # Apply slippage tolerance
            sol_out = int(sol_out * (1 - config["slippage"]))
            
            return max(0, int(sol_out))
        except Exception as e:
            logging.error(f"Error estimating SOL output: {e}")
            return int(token_in * 0.95 / 1e9)  # Fallback to a simple estimate

# Trading Functions
async def setup_wallets():
    """Fund proxy wallets from the main wallet"""
    logging.info("Setting up wallets...")
    main_balance_response = await retry_rpc(client.get_balance, MAIN_WALLET.pubkey())
    if not main_balance_response or main_balance_response.value is None:
        logging.error(f"Failed to get main wallet balance or balance is None.")
        return False
    
    main_balance = main_balance_response.value

    if main_balance < int(INITIAL_FUNDING) * len(wallets): 
        logging.error(f"Insufficient balance in main wallet: {main_balance / 1e9} SOL. Need at least { (int(INITIAL_FUNDING) * len(wallets)) / 1e9 } SOL")
        return False
    
    all_wallets_funded = True
    for wallet in wallets:
        try:
            balance_response = await retry_rpc(client.get_balance, wallet.pubkey())
            if not balance_response or balance_response.value is None:
                logging.warning(f"Failed to get balance for wallet {str(wallet.pubkey())} or balance is None. Skipping funding.")
                all_wallets_funded = False
                continue

            balance = balance_response.value

            if balance < int(INITIAL_FUNDING) / 2: 
                logging.info(f"Funding wallet {str(wallet.pubkey())}")
                transfer_ix = transfer(
                    TransferParams(
                        from_pubkey=MAIN_WALLET.pubkey(),
                        to_pubkey=wallet.pubkey(),
                        lamports=int(INITIAL_FUNDING)  
                    )
                )
                await retry_rpc(send_transaction, [transfer_ix], MAIN_WALLET)
        except Exception as e:
            logging.error(f"Error funding wallet {str(wallet.pubkey())}: {e}")
            all_wallets_funded = False
    
    return all_wallets_funded

async def buy_token(wallet: Keypair, token_mint: str, amount_sol: float):
    """Buy token using the specified wallet and amount"""
    global token_decimals, is_pump_fun
    
    token_mint_pubkey = Pubkey.from_string(token_mint)
    amount_lamports = int(amount_sol * 1e9)
    
    try:
        # Get token data
        pool_data = await fetch_pumpfun_pool_data(token_mint)
        token_decimals = pool_data["decimals"]
        is_token_pump_fun = await is_pump_fun_token(token_mint_pubkey)
        
        # Initialize transaction manager
        tx_manager = TransactionManager(token_mint_pubkey, token_decimals, is_token_pump_fun)
        
        # Get or create associated token account
        ata, create_ata_ix = await tx_manager.get_or_create_ata(wallet)
        
        instructions = []
        if create_ata_ix:
            instructions.append(create_ata_ix)
        
        if is_token_pump_fun:
            # Add PumpFun buy instruction
            buy_ix = await tx_manager.create_buy_instruction(wallet.pubkey(), ata, amount_lamports)
            instructions.append(buy_ix)
        else:
            # For other tokens, we would need to implement a different exchange mechanism
            # This is a placeholder for Jupiter or Raydium swap instructions
            logging.warning("Non-PumpFun token swaps not fully implemented")
            return False
        
        # Send transaction
        tx_sig = await retry_rpc(send_transaction, instructions, wallet)
        logging.info(f"Buy transaction successful: {tx_sig.value}")
        return True
    except Exception as e:
        logging.error(f"Error buying token: {e}")
        return False

async def sell_token(wallet: Keypair, token_mint: str, ratio: float = 1.0):
    """Sell token using the specified wallet and ratio (0.0-1.0 of holdings)"""
    global token_decimals, is_pump_fun
    
    token_mint_pubkey = Pubkey.from_string(token_mint)
    
    try:
        # Get token data
        pool_data = await fetch_pumpfun_pool_data(token_mint)
        token_decimals = pool_data["decimals"]
        is_token_pump_fun = await is_pump_fun_token(token_mint_pubkey)
        
        # Initialize transaction manager
        tx_manager = TransactionManager(token_mint_pubkey, token_decimals, is_token_pump_fun)
        
        # Get token balance
        ata = get_associated_token_address(wallet.pubkey(), token_mint_pubkey)
        balance = await tx_manager.get_token_balance(wallet.pubkey())
        
        if balance <= 0:
            logging.info(f"No token balance to sell for {str(wallet.pubkey())}")
            return True
        
        # Calculate amount to sell
        amount_to_sell = int(balance * ratio)
        if amount_to_sell <= 0:
            logging.info(f"Amount to sell too small for {str(wallet.pubkey())}")
            return True
        
        # Check if ATA exists
        if not await tx_manager.check_ata_exists(wallet.pubkey()):
            logging.info(f"No ATA found for {str(wallet.pubkey())}")
            return False
        
        if is_token_pump_fun:
            # Add PumpFun sell instruction
            sell_ix = await tx_manager.create_sell_instruction(wallet.pubkey(), ata, amount_to_sell)
            
            # Send transaction
            tx_sig = await retry_rpc(send_transaction, [sell_ix], wallet)
            logging.info(f"Sell transaction successful: {tx_sig.value}")
            return True
        else:
            # For other tokens, we would need to implement a different exchange mechanism
            logging.warning("Non-PumpFun token swaps not fully implemented")
            return False
    except Exception as e:
        logging.error(f"Error selling token: {e}")
        return False

async def check_wallet_token_balance(wallet: Keypair, token_mint: str) -> Tuple[int, float]:
    """Check token balance for the wallet and estimate its value in SOL"""
    token_mint_pubkey = Pubkey.from_string(token_mint)
    tx_manager = TransactionManager(token_mint_pubkey, token_decimals, is_pump_fun)
    
    balance = await tx_manager.get_token_balance(wallet.pubkey())
    if balance <= 0:
        return 0, 0.0
    
    # Estimate SOL value
    sol_value = await tx_manager.estimate_sol_out(balance)
    
    return balance, sol_value / 1e9

async def collect_funds_to_main():
    """Collect remaining SOL from proxy wallets back to main wallet"""
    logging.info("Collecting funds back to main wallet...")
    
    for wallet in wallets:
        try:
            balance = await retry_rpc(client.get_balance, wallet.pubkey())
            # Keep a small amount for transaction fees
            amount_to_transfer = max(0, balance.value - 2000000)
            
            if amount_to_transfer > 0:
                logging.info(f"Transferring {amount_to_transfer / 1e9} SOL from {str(wallet.pubkey())} to main")
                transfer_ix = transfer(
                    TransferParams(
                        from_pubkey=wallet.pubkey(),
                        to_pubkey=MAIN_WALLET.pubkey(),
                        lamports=amount_to_transfer
                    )
                )
                await retry_rpc(send_transaction, [transfer_ix], wallet)
        except Exception as e:
            logging.error(f"Error collecting funds from {str(wallet.pubkey())}: {e}")

async def trading_cycle():
    """Main trading cycle logic"""
    global trading_active, sellall_triggered
    
    if not current_token or not trading_active or paused:
        return
    
    token_mint = current_token
    
    try:
        # Get current market data
        pool_data = await fetch_pumpfun_pool_data(token_mint)
        current_price = pool_data["price"]
        
        # Check if sell all was triggered
        if sellall_triggered:
            logging.info("Sell all triggered, selling all holdings")
            for wallet in wallets:
                await sell_token(wallet, token_mint, 1.0)
            sellall_triggered = False
            trading_active = False
            await collect_funds_to_main()
            return
        
        # Randomly select wallets for buying and selling
        buy_wallets = random.sample(wallets, min(3, len(wallets)))
        sell_wallets = random.sample(wallets, min(2, len(wallets)))
        
        # Perform buys
        for wallet in buy_wallets:
            if not trading_active:
                break
            
            sol_amount = random.uniform(config["buy_amount_min"], config["buy_amount_max"])
            logging.info(f"Buying with {sol_amount} SOL from {str(wallet.pubkey())}")
            await buy_token(wallet, token_mint, sol_amount)
            await asyncio.sleep(random.uniform(1, 3))
        
        # Wait before selling
        await asyncio.sleep(random.uniform(config["buy_interval"], config["buy_interval"] + 5))
        
        # Perform sells
        for wallet in sell_wallets:
            if not trading_active:
                break
            
            # Check token balance and value
            balance, sol_value = await check_wallet_token_balance(wallet, token_mint)
            if balance > 0 and sol_value > 0.001:
                sell_ratio = random.uniform(config["sell_ratio"] - 0.1, config["sell_ratio"] + 0.1)
                logging.info(f"Selling {sell_ratio * 100}% of tokens from {str(wallet.pubkey())}")
                await sell_token(wallet, token_mint, sell_ratio)
                await asyncio.sleep(random.uniform(1, 3))
        
        # Schedule next cycle
        await asyncio.sleep(random.uniform(config["sell_interval"], config["sell_interval"] + 5))
    except Exception as e:
        logging.error(f"Error in trading cycle: {e}")
        await asyncio.sleep(5)

async def start_trading_task():
    """Start the trading task loop"""
    global trading_task
    
    if trading_task and not trading_task.done():
        return
    
    trading_task = asyncio.create_task(trading_loop())

async def trading_loop():
    """Main trading loop"""
    global trading_active
    
    while trading_active:
        await trading_cycle()
        await asyncio.sleep(1)

# Telegram Bot Functions
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Disable auth check for now
    # Log the user ID to help with debugging
    logging.info(f"User ID: {user_id}, TELEGRAM_USER_ID: {TELEGRAM_USER_ID}")
    # if TELEGRAM_USER_ID and TELEGRAM_USER_ID.strip() and str(user_id) != TELEGRAM_USER_ID:
    #     await context.bot.send_message(
    #         chat_id=chat_id,
    #         text="Unauthorized access. This bot is private."
    #     )
    #     return
    
    keyboard = [
        [InlineKeyboardButton("🔄 Status", callback_data="status")],
        [InlineKeyboardButton("🚀 Start Trading", callback_data="start_trading")],
        [InlineKeyboardButton("⏸️ Pause Trading", callback_data="pause_trading")],
        [InlineKeyboardButton("❌ Stop Trading", callback_data="stop_trading")],
        [InlineKeyboardButton("💰 Sell All Tokens", callback_data="sell_all")],
        [InlineKeyboardButton("⚙️ Configure Bot", callback_data="configure")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await context.bot.send_message(
        chat_id=chat_id,
        text="Welcome to the Solana Trading Bot!\n\nUse the buttons below to control the bot.",
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    # Disable auth check for now
    # Log the user ID to help with debugging
    logging.info(f"User ID from button callback: {query.from_user.id}, TELEGRAM_USER_ID: {TELEGRAM_USER_ID}")
    # if TELEGRAM_USER_ID and TELEGRAM_USER_ID.strip() and str(query.from_user.id) != TELEGRAM_USER_ID:
    #     await query.edit_message_text(text="Unauthorized access. This bot is private.")
    #     return
    
    action = query.data
    
    if action == "status":
        await status_command(update, context, from_callback=True)
    elif action == "start_trading":
        await start_trading_command(update, context, from_callback=True)
    elif action == "pause_trading":
        await pause_trading_command(update, context, from_callback=True)
    elif action == "stop_trading":
        await stop_trading_command(update, context, from_callback=True)
    elif action == "sell_all":
        await sell_all_command(update, context, from_callback=True)
    elif action == "configure":
        await configure_command(update, context, from_callback=True)
    elif action == "toggle_auto":
        await auto_trading_command(update, context, from_callback=True)
    elif action.startswith("set_token_"):
        token_mint = action.split("_", 2)[2]
        await set_token_command(update, context, token_mint, from_callback=True)
    elif action.startswith("auto_select_"):
        # Allow selecting from auto-detected tokens
        token_index = int(action.split("_", 2)[2])
        if auto_selected_tokens and len(auto_selected_tokens) > token_index:
            token_mint = auto_selected_tokens[token_index]["mint"]
            await set_token_command(update, context, token_mint, from_callback=True)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Status command handler"""
    global current_token, trading_active, paused
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if not from_callback:
        if False: # Authorization check disabled
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized access. This bot is private."
            )
            return
    
    # Get RPC status
    rpc_status = "Connected" if client else "Disconnected"
    
    # Get wallet information
    main_balance = await retry_rpc(client.get_balance, MAIN_WALLET.pubkey()) if client else None
    main_sol = main_balance.value / 1e9 if main_balance else 0
    
    # Get current token information
    token_info = ""
    total_token_value = 0.0
    
    if current_token:
        pool_data = await fetch_pumpfun_pool_data(current_token)
        price = pool_data["price"]
        token_info = f"Current Token: {current_token}\n"
        token_info += f"Price: {price:.9f} SOL\n"
        
        if trading_active:
            token_info += f"Trading Status: {'Paused' if paused else 'Active'}\n"
            
            # Calculate total holdings and value
            total_tokens = 0
            for wallet in wallets:
                balance, sol_value = await check_wallet_token_balance(wallet, current_token)
                total_tokens += balance
                total_token_value += sol_value
            
            token_info += f"Total Tokens: {total_tokens / (10 ** token_decimals):.6f}\n"
            token_info += f"Estimated Value: {total_token_value:.6f} SOL\n"
    
    # Auto mode status
    auto_mode_info = f"Auto Trading Mode: {'✅ Enabled' if config['auto_mode'] else '❌ Disabled'}\n"
    if config['auto_mode']:
        auto_mode_info += f"Profit Threshold: {config['profit_threshold'] * 100:.0f}%\n"
        auto_mode_info += f"Auto Sell Profit: {config['auto_sell_profit'] * 100:.0f}%\n"
        auto_mode_info += f"Auto Sell Loss: {config['auto_sell_loss'] * 100:.0f}%\n"
        auto_mode_info += f"Scan Interval: {config['scan_interval']} seconds\n"
    
    # Format wallet information
    wallet_info = f"Main Wallet: {str(MAIN_WALLET.pubkey())}\n"
    wallet_info += f"Main Balance: {main_sol:.6f} SOL\n"
    wallet_info += f"Proxy Wallets: {len(wallets)}\n"
    
    proxy_balance_total = 0
    for wallet in wallets:
        balance = await retry_rpc(client.get_balance, wallet.pubkey()) if client else None
        proxy_balance_total += balance.value / 1e9 if balance else 0
    
    wallet_info += f"Proxy Balance Total: {proxy_balance_total:.6f} SOL\n"
    
    # Format configuration
    config_info = "Configuration:\n"
    for key, value in config.items():
        config_info += f"- {key}: {value}\n"
    
    # Add auto-detected tokens if available
    auto_tokens_info = ""
    if config["auto_mode"] and auto_selected_tokens:
        auto_tokens_info = "\nAuto-detected promising tokens:\n"
        for i, token in enumerate(auto_selected_tokens[:3]):
            auto_tokens_info += f"{i+1}. Score: {token['score']:.1f} - {token['mint'][:8]}...{token['mint'][-6:]}\n"
    
    # Combine all information
    message = f"🤖 Bot Status\n\n"
    message += f"RPC: {rpc_status}\n\n"
    message += f"{wallet_info}\n"
    
    if token_info:
        message += f"{token_info}\n"
    
    message += f"Total Value: {(main_sol + proxy_balance_total + total_token_value):.6f} SOL\n\n"
    message += f"{config_info}"
    
    if auto_tokens_info:
        message += f"\n{auto_tokens_info}"
        
    # Add buttons for auto-detected tokens
    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data="status")],
    ]
    
    if config["auto_mode"] and auto_selected_tokens:
        auto_token_buttons = []
        for i, token in enumerate(auto_selected_tokens[:3]):
            button_text = f"Select Token #{i+1}"
            auto_token_buttons.append(InlineKeyboardButton(button_text, callback_data=f"auto_select_{i}"))
        keyboard.append(auto_token_buttons)
        
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if from_callback:
        await update.callback_query.edit_message_text(text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message)

async def start_trading_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Start trading command handler"""
    global trading_active, paused
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if not from_callback:
        if False: # Authorization check disabled
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized access. This bot is private."
            )
            return
    
    if not current_token:
        message = "Error: No token selected. Use /settoken command first."
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        return
    
    # Check if wallets are funded
    wallet_setup = await setup_wallets()
    if not wallet_setup:
        message = "Error: Failed to set up wallets. Check main wallet balance."
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        return
    
    trading_active = True
    paused = False
    
    await start_trading_task()
    
    message = f"✅ Started trading for {current_token}"
    if from_callback:
        await update.callback_query.edit_message_text(text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message)

async def pause_trading_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Pause trading command handler"""
    global paused
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if not from_callback:
        if False: # Authorization check disabled
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized access. This bot is private."
            )
            return
    
    if not trading_active:
        message = "Error: Trading is not active."
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        return
    
    paused = not paused
    
    message = f"{'⏸️ Trading paused' if paused else '▶️ Trading resumed'}"
    if from_callback:
        await update.callback_query.edit_message_text(text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message)

async def stop_trading_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Stop trading command handler"""
    global trading_active, trading_task
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if not from_callback:
        if False: # Authorization check disabled
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized access. This bot is private."
            )
            return
    
    if not trading_active:
        message = "Error: Trading is not active."
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        return
    
    trading_active = False
    if trading_task and not trading_task.done():
        try:
            trading_task.cancel()
        except Exception:
            pass
    
    message = "❌ Trading stopped"
    if from_callback:
        await update.callback_query.edit_message_text(text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message)

async def sell_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Sell all tokens command handler"""
    global sellall_triggered
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if not from_callback:
        if False: # Authorization check disabled
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized access. This bot is private."
            )
            return
    
    if not current_token:
        message = "Error: No token selected."
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        return
    
    sellall_triggered = True
    
    if not trading_active:
        # If trading is not active, execute sell all immediately
        for wallet in wallets:
            await sell_token(wallet, current_token, 1.0)
        sellall_triggered = False
        await collect_funds_to_main()
    
    message = "💰 Sell all triggered. Selling tokens from all wallets..."
    if from_callback:
        await update.callback_query.edit_message_text(text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message)

async def configure_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Configuration command handler"""
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if not from_callback:
        if False: # Authorization check disabled
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized access. This bot is private."
            )
            return
    
    message = "⚙️ Configuration\n\n"
    message += "Use the following commands to configure the bot:\n\n"
    message += "/settoken <token_mint> - Set the token to trade\n"
    message += "/config buy_interval <seconds> - Set the buy interval\n"
    message += "/config sell_interval <seconds> - Set the sell interval\n"
    message += "/config buy_amount_min <sol> - Set the minimum buy amount\n"
    message += "/config buy_amount_max <sol> - Set the maximum buy amount\n"
    message += "/config sell_ratio <ratio> - Set the sell ratio (0.1-1.0)\n"
    message += "/config slippage <percentage> - Set slippage tolerance (0.01-0.20)\n"
    message += "/config max_cap <usd> - Set the maximum market cap (USD)\n"
    message += "/config num_wallets <count> - Set the number of proxy wallets\n"
    message += "/config target_multiplier <multiplier> - Set target price multiplier\n"
    
    if from_callback:
        await update.callback_query.edit_message_text(text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message)

async def settoken_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set token command handler"""
    global current_token, token_decimals, is_pump_fun
    
    chat_id = update.effective_chat.id
    
    if False: # Authorization check disabled
        await context.bot.send_message(
            chat_id=chat_id,
            text="Unauthorized access. This bot is private."
        )
        return
    
    # Check for token mint in args first
    if context.args and len(context.args) >= 1:
        token_mint = context.args[0]
    # If no args, check if there's text after the command
    elif update.message and update.message.text:
        parts = update.message.text.split()
        if len(parts) >= 2:
            # Get everything after the command
            token_mint = " ".join(parts[1:])
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Please provide a token mint address.\nFormat: /settoken YourTokenAddressHere"
            )
            return
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Please provide a token mint address.\nFormat: /settoken YourTokenAddressHere"
        )
        return
    
    # Clean the token mint address (remove any extra whitespace)
    token_mint = token_mint.strip()
    
    logging.info(f"Setting token with mint address: {token_mint}")
    
    try:
        # Validate the token mint
        logging.info("Converting token mint to Pubkey")
        token_mint_pubkey = Pubkey.from_string(token_mint)
        logging.info(f"Getting account info for {token_mint_pubkey}")
        mint_info = await get_account_info(token_mint_pubkey)
        
        if not mint_info:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Error: Invalid token mint address. Token account not found."
            )
            return
        
        # Check if it's a PumpFun token
        is_pump_fun = await is_pump_fun_token(token_mint_pubkey)
        
        # Get token data
        pool_data = await fetch_pumpfun_pool_data(token_mint)
        token_decimals = pool_data["decimals"]
        
        current_token = token_mint
        
        # Fetch token price and market cap
        price = pool_data["price"]
        market_cap = await estimate_market_cap(token_mint, price)
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"✅ Token set to {token_mint}\n"
                f"Decimals: {token_decimals}\n"
                f"PumpFun Token: {'Yes' if is_pump_fun else 'No'}\n"
                f"Price: {price:.9f} SOL\n"
                f"Estimated Market Cap: ${market_cap:.2f}"
        )
    except Exception as e:
        logging.error(f"Error setting token: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Error setting token: {str(e)}"
        )

async def set_token_command(update: Update, context: ContextTypes.DEFAULT_TYPE, token_mint: str, from_callback: bool = False):
    """Set token from callback handler"""
    global current_token, token_decimals, is_pump_fun
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    try:
        # Validate the token mint
        token_mint_pubkey = Pubkey.from_string(token_mint)
        mint_info = await get_account_info(token_mint_pubkey)
        
        if not mint_info:
            message = "Error: Invalid token mint address. Token account not found."
            if from_callback:
                await update.callback_query.edit_message_text(text=message)
            else:
                await context.bot.send_message(chat_id=chat_id, text=message)
            return
        
        # Check if it's a PumpFun token
        is_pump_fun = await is_pump_fun_token(token_mint_pubkey)
        
        # Get token data
        pool_data = await fetch_pumpfun_pool_data(token_mint)
        token_decimals = pool_data["decimals"]
        
        current_token = token_mint
        
        # Fetch token price and market cap
        price = pool_data["price"]
        market_cap = await estimate_market_cap(token_mint, price)
        
        message = f"✅ Token set to {token_mint}\n"
        message += f"Decimals: {token_decimals}\n"
        message += f"PumpFun Token: {'Yes' if is_pump_fun else 'No'}\n"
        message += f"Price: {price:.9f} SOL\n"
        message += f"Estimated Market Cap: ${market_cap:.2f}"
        
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        logging.error(f"Error setting token: {e}")
        message = f"Error setting token: {str(e)}"
        if from_callback:
            await update.callback_query.edit_message_text(text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)

async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Configuration command handler"""
    global config
    
    chat_id = update.effective_chat.id
    
    if False: # Authorization check disabled
        await context.bot.send_message(
            chat_id=chat_id,
            text="Unauthorized access. This bot is private."
        )
        return
    
    if not context.args or len(context.args) < 2:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Please provide a parameter and value."
        )
        return
    
    param = context.args[0]
    value = context.args[1]
    
    if param not in config:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Invalid parameter: {param}"
        )
        return
    
    try:
        if param in ["buy_interval", "sell_interval", "max_cap", "num_wallets"]:
            new_value = int(value)
        else:
            new_value = float(value)
        
        # Validate values
        if param == "sell_ratio" and (new_value < 0.1 or new_value > 1.0):
            await context.bot.send_message(
                chat_id=chat_id,
                text="Sell ratio must be between 0.1 and 1.0"
            )
            return
        
        if param == "slippage" and (new_value < 0.01 or new_value > 0.20):
            await context.bot.send_message(
                chat_id=chat_id,
                text="Slippage must be between 0.01 and 0.20"
            )
            return
        
        if (param == "buy_amount_min" or param == "buy_amount_max") and new_value <= 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Buy amount must be greater than 0"
            )
            return
        
        if param == "buy_amount_min" and "buy_amount_max" in config and new_value > config["buy_amount_max"]:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Minimum buy amount cannot be greater than maximum"
            )
            return
        
        if param == "buy_amount_max" and "buy_amount_min" in config and new_value < config["buy_amount_min"]:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Maximum buy amount cannot be less than minimum"
            )
            return
        
        # Update configuration
        config[param] = new_value
        save_config()
        
        # Update number of wallets if needed
        if param == "num_wallets" and new_value > len(wallets):
            load_wallets()
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"✅ Configuration updated: {param} = {new_value}"
        )
    except ValueError:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Invalid value. Please provide a number."
        )

async def auto_trading_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Auto trading command handler to enable/disable automatic token selection"""
    global config
    
    chat_id = update.effective_chat.id if not from_callback else update.callback_query.message.chat.id
    
    if False: # Authorization check disabled
        await context.bot.send_message(
            chat_id=chat_id,
            text="Unauthorized access. This bot is private."
        )
        return
    
    # Toggle auto mode
    config["auto_mode"] = not config["auto_mode"]
    save_config()
    
    # Create status message
    if config["auto_mode"]:
        message = "🔄 *Auto-trading mode ENABLED*\n\nThe bot will automatically scan for promising tokens and trade them without manual intervention.\n\nUse /auto again to disable."
    else:
        message = "🛑 *Auto-trading mode DISABLED*\n\nYou'll need to manually select tokens using /settoken command.\n\nUse /auto to enable automatic token selection."
    
    # Show auto-selected tokens if available
    if config["auto_mode"] and auto_selected_tokens:
        message += "\n\n*Recently detected promising tokens:*\n"
        for i, token in enumerate(auto_selected_tokens[:3], 1):
            message += f"{i}. Score: {token['score']:.1f} - {token['mint'][:8]}...{token['mint'][-6:]}\n"
    
    # Create keyboard
    keyboard = [
        [InlineKeyboardButton("Toggle Auto-trading", callback_data="toggle_auto")],
        [InlineKeyboardButton("🔄 Refresh Status", callback_data="status")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send message
    if from_callback:
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    chat_id = update.effective_chat.id
    
    if False: # Authorization check disabled
        await context.bot.send_message(
            chat_id=chat_id,
            text="Unauthorized access. This bot is private."
        )
        return
    
    help_text = """
🤖 *Solana Trading Bot Commands* 🤖

/start - Start the bot and show main menu
/status - Check bot status and balances
/start_trading - Start trading the selected token
/pause_trading - Pause or resume active trading
/stop_trading - Stop all trading activities
/sell_all - Sell all token holdings
/settoken {token_address} - Set token to trade
/auto - Toggle automatic token selection mode
/configure - Show configuration options
/settoken <token_mint> - Set the token to trade (e.g., /settoken F9EhG...)
/config <param> <value> - Configure trading parameters
/help - Show this help message

Parameters that can be configured:
- buy_interval: Time between buy operations (seconds)
- sell_interval: Time between sell operations (seconds)
- buy_amount_min: Minimum amount of SOL to use for buys
- buy_amount_max: Maximum amount of SOL to use for buys
- sell_ratio: Portion of tokens to sell in each operation (0.1-1.0)
- slippage: Slippage tolerance (0.01-0.20)
- max_cap: Maximum market cap in USD
- num_wallets: Number of proxy wallets to use
- target_multiplier: Target price multiplier for take profit
    """
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=help_text
    )

async def main():
    """Main function"""
    global auto_token_selection_task
    
    # Load configuration
    load_config()
    
    # Initialize RPC client
    try:
        await init_client()
    except Exception as e:
        logging.error(f"Failed to initialize RPC client: {e}")
        return
    
    # Load wallets
    load_wallets()
    
    # Start the auto token selection task in the background
    auto_token_selection_task = asyncio.create_task(auto_token_selection_loop())
    
    # Initialize Telegram bot
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("start_trading", start_trading_command))
    application.add_handler(CommandHandler("pause_trading", pause_trading_command))
    application.add_handler(CommandHandler("stop_trading", stop_trading_command))
    application.add_handler(CommandHandler("sell_all", sell_all_command))
    application.add_handler(CommandHandler("configure", configure_command))
    application.add_handler(CommandHandler("settoken", settoken_command))
    application.add_handler(CommandHandler("config", config_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("auto", auto_trading_command))
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start the bot
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    logging.info("Bot started")
    
    try:
        # Keep the bot running
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        # Shutdown
        logging.info("Bot stopping...")
        await application.stop()
        await application.updater.stop()
        if client:
            await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        logging.exception(e)
