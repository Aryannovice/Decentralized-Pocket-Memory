"""
Blockchain service for crystal proof hash verification on Polygon.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
import os
from web3 import Web3
from eth_account import Account

logger = logging.getLogger(__name__)


class BlockchainService:
    """Service for interacting with Polygon blockchain for crystal verification."""
    
    def __init__(self):
        # Polygon Amoy testnet (Mumbai deprecated) 
        # Change to mainnet: https://polygon-rpc.com
        self.rpc_url = "https://rpc-amoy.polygon.technology/"
        self.chain_id = 80002  # Amoy testnet, use 137 for mainnet
        
        # Demo mode for immediate functionality without requiring real transactions
        self.demo_mode = True
        self.demo_storage = {}  # Mock blockchain storage
        
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.contract_address = None
        self.contract = None
        self.private_key = None
        self.account_address = None
        
        # Simple contract ABI for storing crystal hashes
        self.contract_abi = [
            {
                "inputs": [
                    {"name": "_crystalId", "type": "string"},
                    {"name": "_proofHash", "type": "bytes32"},
                    {"name": "_creator", "type": "address"}
                ],
                "name": "storeCrystalHash",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_crystalId", "type": "string"}],
                "name": "getCrystalProof",
                "outputs": [
                    {"name": "proofHash", "type": "bytes32"},
                    {"name": "creator", "type": "address"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "exists", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        self._initialize()

    def _initialize(self):
        """Initialize blockchain connection."""
        try:
            # Load or generate account
            private_key_file = "blockchain_key.json"
            if os.path.exists(private_key_file):
                with open(private_key_file, 'r') as f:
                    data = json.load(f)
                    self.private_key = data.get("private_key")
                    self.contract_address = data.get("contract_address")
            else:
                # Generate new account for development
                account = Account.create()
                self.private_key = account.key.hex()  # Fixed: privateKey -> key
                
                # Save to file
                with open(private_key_file, 'w') as f:
                    json.dump({
                        "private_key": self.private_key,
                        "contract_address": None  # Will be set after deployment
                    }, f)
                    
                logger.info(f"Generated new blockchain account: {account.address}")
                print(f"🔑 New blockchain account created: {account.address}")
                print(f"🌐 Network: Polygon Amoy Testnet (Chain ID: {self.chain_id})")
                if not self.demo_mode:
                    print("⚠️  Fund this account with MATIC tokens for blockchain operations")
                else:
                    print("🚀 Demo mode enabled - blockchain operations will be simulated")

            if self.private_key:
                account = Account.from_key(self.private_key)
                self.account_address = account.address
                
            # Initialize contract if address exists
            if self.contract_address:
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=self.contract_abi
                )
                
            logger.info(f"Blockchain service initialized. Account: {self.account_address}")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain service: {e}")
            print(f"⚠️  Blockchain initialization error: {e}")
            
    def is_available(self) -> bool:
        """Check if blockchain service is ready."""
        if self.demo_mode:
            return self.account_address is not None
            
        try:
            return (
                self.w3.is_connected() and
                self.private_key is not None and
                self.account_address is not None
            )
        except Exception as e:
            logger.error(f"Blockchain availability check failed: {e}")
            return False
        
    async def store_crystal_hash(
        self, 
        crystal_id: str, 
        proof_hash: str, 
        creator_id: str = "anonymous"
    ) -> Optional[str]:
        """
        Store a crystal proof hash on the blockchain.
        Returns transaction hash if successful.
        """
        if not self.is_available():
            logger.warning("Blockchain service not available")
            return None
            
        try:
            if self.demo_mode:
                # Demo mode - simulate blockchain storage
                import time
                fake_tx_hash = f"0x{hash(f'{crystal_id}{proof_hash}{time.time()}') & 0xffffffff:08x}" + "0" * 56
                self.demo_storage[crystal_id] = {
                    "proof_hash": proof_hash,
                    "creator": self.account_address,
                    "timestamp": int(time.time()),
                    "tx_hash": fake_tx_hash
                }
                logger.info(f"Crystal hash stored in demo mode: {fake_tx_hash}")
                print(f"🔗 Demo: Crystal {crystal_id[:8]}... stored with hash {fake_tx_hash[:10]}...")
                return fake_tx_hash
            
            # Real blockchain mode (when contract is deployed)
            if not self.contract:
                logger.warning("Smart contract not deployed")
                return None
                
            # Convert proof hash to bytes32
            if proof_hash.startswith("0x"):
                hash_bytes = bytes.fromhex(proof_hash[2:])
            else:
                hash_bytes = bytes.fromhex(proof_hash)
                
            if len(hash_bytes) != 32:
                # Pad or truncate to 32 bytes
                hash_bytes = hash_bytes[:32].ljust(32, b'\\0')
                
            # Build transaction
            txn = self.contract.functions.storeCrystalHash(
                crystal_id,
                hash_bytes,
                self.account_address
            ).buildTransaction({
                'chainId': self.chain_id,
                'gas': 100000,
                'gasPrice': self.w3.toWei('20', 'gwei'),
                'nonce': self.w3.eth.getTransactionCount(self.account_address),
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.signTransaction(txn, private_key=self.private_key)
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            logger.info(f"Crystal hash stored on blockchain: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to store crystal hash: {e}")
            return None
            
    def verify_crystal_hash(self, crystal_id: str) -> Optional[Dict[str, Any]]:
        """Verify a crystal hash exists on blockchain."""
        if not self.is_available():
            return None
            
        try:
            if self.demo_mode:
                # Demo mode - check demo storage
                stored = self.demo_storage.get(crystal_id)
                if stored:
                    return {
                        "proof_hash": stored["proof_hash"],
                        "creator": stored["creator"],
                        "timestamp": stored["timestamp"],
                        "tx_hash": stored["tx_hash"],
                        "verified": True,
                        "demo_mode": True
                    }
                return {"verified": False, "demo_mode": True}
            
            # Real blockchain mode
            if not self.contract:
                return {"verified": False, "error": "Contract not deployed"}
                
            result = self.contract.functions.getCrystalProof(crystal_id).call()
            proof_hash, creator, timestamp, exists = result
            
            if exists:
                return {
                    "proof_hash": proof_hash.hex(),
                    "creator": creator,
                    "timestamp": timestamp,
                    "verified": True,
                    "demo_mode": False
                }
            return {"verified": False, "demo_mode": False}
            
        except Exception as e:
            logger.error(f"Failed to verify crystal hash: {e}")
            return {"verified": False, "error": str(e)}
            
    def get_account_info(self) -> Dict[str, Any]:
        """Get blockchain account information."""
        network_name = "Polygon Amoy Testnet" if self.chain_id == 80002 else "Polygon Mainnet"
        
        info = {
            "address": self.account_address,
            "connected": self.is_available(),
            "balance": "0 MATIC",
            "network": network_name,
            "demo_mode": self.demo_mode
        }
        
        if self.is_available():
            try:
                if self.demo_mode:
                    # Demo mode - show fake balance
                    info["balance"] = "1.5000 MATIC (Demo)"
                    info["connected"] = True
                else:
                    # Real blockchain mode
                    balance_wei = self.w3.eth.getBalance(self.account_address)
                    balance_matic = self.w3.fromWei(balance_wei, 'ether')
                    info["balance"] = f"{float(balance_matic):.4f} MATIC"
            except Exception as e:
                logger.error(f"Failed to get balance: {e}")
                info["balance"] = "Error getting balance"
                
        return info


# Demo Smart Contract (Solidity code for reference)
CRYSTAL_STORAGE_CONTRACT = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CrystalStorage {
    struct CrystalProof {
        bytes32 proofHash;
        address creator;
        uint256 timestamp;
        bool exists;
    }
    
    mapping(string => CrystalProof) public crystals;
    
    event CrystalStored(string indexed crystalId, bytes32 proofHash, address creator);
    
    function storeCrystalHash(string memory _crystalId, bytes32 _proofHash, address _creator) public {
        crystals[_crystalId] = CrystalProof({
            proofHash: _proofHash,
            creator: _creator,
            timestamp: block.timestamp,
            exists: true
        });
        
        emit CrystalStored(_crystalId, _proofHash, _creator);
    }
    
    function getCrystalProof(string memory _crystalId) public view returns (
        bytes32 proofHash,
        address creator,
        uint256 timestamp,
        bool exists
    ) {
        CrystalProof memory proof = crystals[_crystalId];
        return (proof.proofHash, proof.creator, proof.timestamp, proof.exists);
    }
}
'''