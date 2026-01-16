"""
Market Data Platform - InfluxDB Storage Module
Stores market data, trades, and sentiment data with time-series optimization
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.query_api import QueryApi

logger = logging.getLogger(__name__)

@dataclass
class InfluxDBConfig:
    """InfluxDB connection configuration"""
    url: str
    token: str
    org: str
    bucket: str
    debug: bool = False

class MarketDataStorage:
    """Handles market data storage in InfluxDB"""
    
    def __init__(self, config: InfluxDBConfig):
        self.config = config
        self.client = InfluxDBClient(
            url=config.url,
            token=config.token,
            org=config.org,
            debug=config.debug
        )
        self.write_api = self.client.write_api(write_type=ASYNCHRONOUS)
        self.query_api = self.client.query_api()

    def write_market_tick(self, 
                         symbol: str,
                         price: float,
                         bid: float,
                         ask: float,
                         volume: float,
                         exchange: str,
                         asset_class: str = "forex"):
        """Write market tick to InfluxDB"""
        try:
            point = Point("market_ticks") \
                .tag("symbol", symbol) \
                .tag("exchange", exchange) \
                .tag("asset_class", asset_class) \
                .field("price", price) \
                .field("bid", bid) \
                .field("ask", ask) \
                .field("volume", volume) \
                .field("spread", ask - bid) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket=self.config.bucket, record=point)
        except Exception as e:
            logger.error(f"Error writing market tick: {e}")

    def write_trade(self,
                    symbol: str,
                    quantity: float,
                    price: float,
                    side: str,
                    exchange: str,
                    account_id: str = None):
        """Write trade execution to InfluxDB"""
        try:
            point = Point("trades") \
                .tag("symbol", symbol) \
                .tag("exchange", exchange) \
                .tag("side", side) \
                .tag("account_id", account_id or "unknown") \
                .field("quantity", quantity) \
                .field("price", price) \
                .field("notional", quantity * price) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket=self.config.bucket, record=point)
        except Exception as e:
            logger.error(f"Error writing trade: {e}")

    def write_order_book(self,
                        symbol: str,
                        level: int,
                        bid_price: float,
                        bid_qty: float,
                        ask_price: float,
                        ask_qty: float,
                        exchange: str):
        """Write order book snapshot to InfluxDB"""
        try:
            point = Point("orderbook") \
                .tag("symbol", symbol) \
                .tag("exchange", exchange) \
                .tag("depth_level", str(level)) \
                .field("bid_price", bid_price) \
                .field("bid_qty", bid_qty) \
                .field("ask_price", ask_price) \
                .field("ask_qty", ask_qty) \
                .field("mid_price", (bid_price + ask_price) / 2) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket=self.config.bucket, record=point)
        except Exception as e:
            logger.error(f"Error writing order book: {e}")

    def write_sentiment(self,
                       topic: str,
                       sentiment_score: float,
                       tweet_count: int,
                       source: str = "twitter"):
        """Write sentiment analysis to InfluxDB"""
        try:
            point = Point("sentiment") \
                .tag("topic", topic) \
                .tag("source", source) \
                .field("sentiment_score", sentiment_score) \
                .field("tweet_count", tweet_count) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket=self.config.bucket, record=point)
        except Exception as e:
            logger.error(f"Error writing sentiment: {e}")

    # ========================================================================
    # Query Methods
    # ========================================================================

    def query_latest_price(self, symbol: str, exchange: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            query = f'''
            from(bucket:"{self.config.bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "market_ticks")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: 1)
            '''
            
            result = self.query_api.query(org=self.config.org, query=query)
            
            if result and len(result) > 0:
                return result[0].records[0].values.get("price")
            return None
        except Exception as e:
            logger.error(f"Error querying latest price: {e}")
            return None

    def query_price_history(self,
                           symbol: str,
                           exchange: str,
                           start_time: str = "-24h",
                           aggregation: str = "5m") -> List[Dict]:
        """Get price history with optional aggregation"""
        try:
            query = f'''
            from(bucket:"{self.config.bucket}")
                |> range(start: {start_time})
                |> filter(fn: (r) => r._measurement == "market_ticks")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> aggregateWindow(every: {aggregation}, fn: mean)
            '''
            
            result = self.query_api.query(org=self.config.org, query=query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "timestamp": record.get_time(),
                        "value": record.get_value(),
                        "field": record.get_field()
                    })
            return data
        except Exception as e:
            logger.error(f"Error querying price history: {e}")
            return []

    def query_ohlc(self,
                   symbol: str,
                   exchange: str,
                   interval: str = "1h",
                   start_time: str = "-7d") -> List[Dict]:
        """Get OHLC data"""
        try:
            query = f'''
            from(bucket:"{self.config.bucket}")
                |> range(start: {start_time})
                |> filter(fn: (r) => r._measurement == "market_ticks")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> aggregateWindow(every: {interval}, fn: (tables=<-) => tables
                    |> group(columns: ["_field"])
                    |> reduce(
                        identity: {{open: 0.0, high: 0.0, low: 999999.0, close: 0.0}},
                        fn: (accumulator, row) => ({
                            open: if accumulator.open == 0.0 then row._value else accumulator.open,
                            high: if row._value > accumulator.high then row._value else accumulator.high,
                            low: if row._value < accumulator.low then row._value else accumulator.low,
                            close: row._value
                        })
                    ))
            '''
            
            result = self.query_api.query(org=self.config.org, query=query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "timestamp": record.get_time(),
                        "ohlc": record.values
                    })
            return data
        except Exception as e:
            logger.error(f"Error querying OHLC: {e}")
            return []

    def query_volume_profile(self,
                            symbol: str,
                            exchange: str,
                            time_range: str = "-24h") -> Dict[str, float]:
        """Get volume at price levels"""
        try:
            query = f'''
            from(bucket:"{self.config.bucket}")
                |> range(start: {time_range})
                |> filter(fn: (r) => r._measurement == "market_ticks")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> group(columns: ["price"])
                |> map(fn: (r) => ({{r with volume: r.volume}}))
            '''
            
            result = self.query_api.query(org=self.config.org, query=query)
            
            volume_profile = {}
            for table in result:
                for record in table.records:
                    price = record.values.get("price")
                    volume = record.get_value()
                    if price in volume_profile:
                        volume_profile[price] += volume
                    else:
                        volume_profile[price] = volume
            
            return volume_profile
        except Exception as e:
            logger.error(f"Error querying volume profile: {e}")
            return {}

    def query_sentiment_correlation(self,
                                   symbol: str,
                                   time_range: str = "-7d") -> Dict[str, float]:
        """Correlate sentiment with price movements"""
        try:
            # This is a simplified example
            # In production, would use more sophisticated correlation analysis
            
            sentiment_query = f'''
            from(bucket:"{self.config.bucket}")
                |> range(start: {time_range})
                |> filter(fn: (r) => r._measurement == "sentiment")
            '''
            
            price_query = f'''
            from(bucket:"{self.config.bucket}")
                |> range(start: {time_range})
                |> filter(fn: (r) => r._measurement == "market_ticks")
                |> filter(fn: (r) => r.symbol == "{symbol}")
            '''
            
            return {
                "correlation": "To be implemented with statistical analysis"
            }
        except Exception as e:
            logger.error(f"Error querying sentiment correlation: {e}")
            return {}

    def delete_old_data(self, days: int = 30):
        """Delete data older than specified days"""
        try:
            stop_time = datetime.utcnow() - timedelta(days=days)
            predicate = f'_time < {int(stop_time.timestamp())}000000000'
            
            self.client.delete_api().delete(
                org=self.config.org,
                bucket=self.config.bucket,
                predicate=predicate,
                start="1970-01-01T00:00:00Z",
                stop=stop_time.isoformat() + "Z"
            )
            logger.info(f"Deleted data older than {days} days")
        except Exception as e:
            logger.error(f"Error deleting old data: {e}")

    def close(self):
        """Close connection"""
        self.client.close()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    config = InfluxDBConfig(
        url="http://localhost:8086",
        token="your-token",
        org="market_data",
        bucket="market_data_bucket"
    )
    
    storage = MarketDataStorage(config)
    
    # Write sample data
    storage.write_market_tick(
        symbol="EUR_USD",
        price=1.0850,
        bid=1.0849,
        ask=1.0851,
        volume=1000000,
        exchange="oanda"
    )
    
    # Query data
    price = storage.query_latest_price("EUR_USD", "oanda")
    print(f"Latest EUR/USD price: {price}")
    
    storage.close()
