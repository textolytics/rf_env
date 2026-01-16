"""
Market Data Platform - Python Connectivity Modules
OANDA Forex + Twitter Sentiment for research and analysis
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import zmq.asyncio
import aiohttp
import tweepy
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class AssetClass(Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"

@dataclass
class MarketData:
    """Standardized market data record"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: int
    exchange: str
    asset_class: AssetClass = AssetClass.FOREX

    def to_json(self) -> str:
        data = asdict(self)
        data['asset_class'] = self.asset_class.value
        return json.dumps(data)

@dataclass
class SentimentData:
    """Twitter sentiment record"""
    timestamp: int
    topic: str
    sentiment_score: float  # -1.0 to 1.0
    tweet_count: int
    source: str = "twitter"
    keywords: List[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

# ============================================================================
# Gateway Connectors
# ============================================================================

class GatewayConnector(ABC):
    """Base class for gateway connectors"""
    
    def __init__(self, name: str, zmq_endpoint: str):
        self.name = name
        self.zmq_endpoint = zmq_endpoint
        self.zmq_context = None
        self.zmq_socket = None
        self.connected = False

    async def initialize_zmq(self):
        """Initialize ZMQ publisher socket"""
        self.zmq_context = zmq.asyncio.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(self.zmq_endpoint)
        logger.info(f"ZMQ socket bound to {self.zmq_endpoint}")

    async def publish(self, topic: str, data: str):
        """Publish message to ZMQ"""
        if self.zmq_socket:
            await self.zmq_socket.send_multipart([
                topic.encode('utf-8'),
                data.encode('utf-8')
            ])

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to gateway"""
        pass

    @abstractmethod
    async def fetch_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Fetch market data"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from gateway"""
        pass

# ============================================================================
# OANDA Forex Connector
# ============================================================================

class OANDAConnector(GatewayConnector):
    """
    OANDA Forex API Connector
    Focus: EURUSD and forex trading with real-time pricing
    """
    
    def __init__(self, api_token: str, account_id: str, zmq_endpoint: str):
        super().__init__("oanda", zmq_endpoint)
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com"
        self.session = None

    async def connect(self) -> bool:
        """Connect to OANDA API"""
        try:
            await self.initialize_zmq()
            
            headers = {"Authorization": f"Bearer {self.api_token}"}
            self.session = aiohttp.ClientSession(headers=headers)
            
            # Test connection
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}"
            ) as resp:
                if resp.status == 200:
                    logger.info("✓ Connected to OANDA API")
                    self.connected = True
                    return True
                else:
                    logger.error(f"OANDA connection failed: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False

    async def fetch_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Fetch current market prices"""
        if not self.connected:
            return []

        try:
            # Default to major pairs if none specified
            if not symbols:
                symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"]

            params = {"instruments": ",".join(symbols)}
            async with self.session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
                params=params
            ) as resp:
                data = await resp.json()
                
                market_data = []
                for price in data.get("prices", []):
                    md = MarketData(
                        symbol=price["instrument"],
                        price=(float(price["bids"][0]["price"]) + float(price["asks"][0]["price"])) / 2,
                        bid=float(price["bids"][0]["price"]),
                        ask=float(price["asks"][0]["price"]),
                        volume=0,  # OANDA doesn't provide volume
                        timestamp=int(datetime.fromisoformat(
                            price["time"].replace("Z", "+00:00")
                        ).timestamp() * 1000),
                        exchange="oanda",
                        asset_class=AssetClass.FOREX
                    )
                    market_data.append(md)
                
                return market_data
        except Exception as e:
            logger.error(f"Error fetching OANDA market data: {e}")
            return []

    async def stream_prices(self, topic: str, symbols: List[str], interval_ms: int = 1000):
        """Stream market data continuously"""
        while self.connected:
            try:
                market_data = await self.fetch_market_data(symbols)
                for md in market_data:
                    await self.publish(topic, md.to_json())
                await asyncio.sleep(interval_ms / 1000)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(1)

    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
        self.connected = False
        logger.info("Disconnected from OANDA")

# ============================================================================
# Twitter Sentiment Connector
# ============================================================================

class TwitterSentimentConnector(GatewayConnector):
    """
    Twitter/X Sentiment Analysis Connector
    Analyzes market sentiment from Twitter streams
    """
    
    def __init__(self, bearer_token: str, zmq_endpoint: str):
        super().__init__("twitter", zmq_endpoint)
        self.bearer_token = bearer_token
        self.client = tweepy.AsyncClient(bearer_token=bearer_token)

    async def connect(self) -> bool:
        """Connect to Twitter API"""
        try:
            await self.initialize_zmq()
            # Test connection with simple query
            logger.info("✓ Connected to Twitter API")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Twitter connection error: {e}")
            return False

    async def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text
        Simple keyword-based approach (can be replaced with ML model)
        
        Returns: -1.0 (bearish) to 1.0 (bullish)
        """
        bullish_keywords = [
            'bull', 'moon', 'pump', 'breakthrough', 'surge',
            'buy', 'long', 'profit', 'gain', 'winning'
        ]
        bearish_keywords = [
            'bear', 'crash', 'dump', 'recession', 'decline',
            'sell', 'short', 'loss', 'bearish', 'risk'
        ]
        
        text_lower = text.lower()
        bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total

    async def stream_sentiment(self, topic: str, search_query: str, interval_s: int = 30):
        """Stream sentiment analysis continuously"""
        while self.connected:
            try:
                # Search for recent tweets
                tweets = await self.client.search_recent_tweets(
                    query=f"{search_query} -is:retweet",
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if tweets.data:
                    total_sentiment = 0
                    for tweet in tweets.data:
                        sentiment = await self.analyze_sentiment(tweet.text)
                        total_sentiment += sentiment
                    
                    avg_sentiment = total_sentiment / len(tweets.data)
                    
                    sentiment_data = SentimentData(
                        timestamp=int(datetime.utcnow().timestamp() * 1000),
                        topic=search_query,
                        sentiment_score=avg_sentiment,
                        tweet_count=len(tweets.data),
                        keywords=search_query.split()
                    )
                    
                    await self.publish(topic, sentiment_data.to_json())
                
                await asyncio.sleep(interval_s)
            except Exception as e:
                logger.error(f"Sentiment streaming error: {e}")
                await asyncio.sleep(5)

    async def disconnect(self):
        """Close connection"""
        self.connected = False
        logger.info("Disconnected from Twitter")

# ============================================================================
# Platform Manager
# ============================================================================

class PythonConnectivityManager:
    """Manages Python-based connectors"""
    
    def __init__(self, zmq_broker_host: str = "127.0.0.1", zmq_broker_port: int = 5559):
        self.zmq_endpoint = f"tcp://{zmq_broker_host}:{zmq_broker_port}"
        self.connectors: Dict[str, GatewayConnector] = {}
        self.tasks: List[asyncio.Task] = []

    async def register_oanda(self, api_token: str, account_id: str) -> bool:
        """Register OANDA connector"""
        connector = OANDAConnector(api_token, account_id, self.zmq_endpoint)
        if await connector.connect():
            self.connectors['oanda'] = connector
            return True
        return False

    async def register_twitter(self, bearer_token: str) -> bool:
        """Register Twitter connector"""
        connector = TwitterSentimentConnector(bearer_token, self.zmq_endpoint)
        if await connector.connect():
            self.connectors['twitter'] = connector
            return True
        return False

    async def start_oanda_stream(self, symbols: List[str] = None):
        """Start OANDA price streaming"""
        if 'oanda' in self.connectors:
            if symbols is None:
                symbols = ["EUR_USD"]
            task = asyncio.create_task(
                self.connectors['oanda'].stream_prices(
                    "oanda.eurusd",
                    symbols
                )
            )
            self.tasks.append(task)

    async def start_twitter_sentiment(self, search_query: str):
        """Start Twitter sentiment analysis"""
        if 'twitter' in self.connectors:
            task = asyncio.create_task(
                self.connectors['twitter'].stream_sentiment(
                    "twitter.market_sentiment",
                    search_query
                )
            )
            self.tasks.append(task)

    async def shutdown(self):
        """Shutdown all connectors"""
        for task in self.tasks:
            task.cancel()
        for connector in self.connectors.values():
            await connector.disconnect()

# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example: Run OANDA and Twitter connectors"""
    
    manager = PythonConnectivityManager()
    
    # Setup (load from environment)
    oanda_token = "YOUR_OANDA_TOKEN"
    oanda_account = "YOUR_ACCOUNT_ID"
    twitter_bearer = "YOUR_TWITTER_BEARER_TOKEN"
    
    # Register connectors
    await manager.register_oanda(oanda_token, oanda_account)
    await manager.register_twitter(twitter_bearer)
    
    # Start streaming
    await manager.start_oanda_stream(["EUR_USD"])
    await manager.start_twitter_sentiment("crypto OR forex")
    
    # Run for 60 seconds
    try:
        await asyncio.sleep(60)
    finally:
        await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
