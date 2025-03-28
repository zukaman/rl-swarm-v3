import json
import logging

from eth_account import Account
from web3 import Web3

ALCHEMY_URL = "https://gensyn-testnet.g.alchemy.com/public"

MAINNET_CHAIN_ID = 685685

SWARM_COORDINATOR_VERSION = "0.2"
SWARM_COORDINATOR_ABI_JSON = f"hivemind_exp/contracts/SwarmCoordinator_{SWARM_COORDINATOR_VERSION}.json"
SWARM_COORDINATOR_CONTRACT = "0xcD1351B125b0ae4f023ADA5D09443087a7d99101"

logger = logging.getLogger(__name__)

def setup_web3() -> Web3:
    # Check testnet connection.
    web3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
    if web3.is_connected():
        logger.info("✅ Connected to Gensyn Testnet")
    else:
        raise Exception("Failed to connect to Gensyn Testnet")
    return web3

def setup_account(web3: Web3, private_key) -> Account:
    # Check wallet balance.
    account = web3.eth.account.from_key(private_key)
    balance = web3.eth.get_balance(account.address)
    eth_balance = web3.from_wei(balance, "ether")
    logger.info(f"💰 Wallet Balance: {eth_balance} ETH")
    return account

def coordinator_contract(web3: Web3):
    with open(SWARM_COORDINATOR_ABI_JSON, "r") as f:
        contract_abi = json.load(f)["abi"]

    return web3.eth.contract(address=SWARM_COORDINATOR_CONTRACT, abi=contract_abi)

def send_chain_txn(web3: Web3, account: Account, txn_factory, chain_id = MAINNET_CHAIN_ID):
    checksummed = Web3.to_checksum_address(account.address)
    txn = txn_factory() | {
        "chainId": chain_id,
        "nonce": web3.eth.get_transaction_count(checksummed),
    }

    # Sign the transaction
    signed_txn = web3.eth.account.sign_transaction(
        txn, private_key=account.key
    )

    # Send the transaction
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    logger.info(f"Sent transaction with hash: {web3.to_hex(tx_hash)}")
