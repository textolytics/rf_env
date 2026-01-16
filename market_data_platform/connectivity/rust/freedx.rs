// FreeDOM Exchange Connectivity Module (Rust)
// REST API client with ZMQ publisher for market_summary endpoint

use super::{ConnectivityConfig, GatewayConnector, MarketData};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zmq::Context;
use tokio::time::{sleep, Duration};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FreedomTicker {
    pub symbol: String,
    pub last: f64,
    pub lowestAsk: f64,
    pub highestBid: f64,
    pub percentageChange: f64,
    pub volume: f64,
    pub high24Hr: f64,
    pub low24Hr: f64,
    pub base: String,
    pub quote: String,
    pub active: bool,
}

pub struct FreedomExchangeConnector {
    config: ConnectivityConfig,
    zmq_context: Arc<Context>,
    client: Client,
}

impl FreedomExchangeConnector {
    pub fn new(config: ConnectivityConfig, zmq_context: Arc<Context>) -> Self {
        FreedomExchangeConnector {
            config,
            zmq_context,
            client: Client::new(),
        }
    }

    async fn fetch_tickers(&self) -> Result<Vec<FreedomTicker>, String> {
        let url = "https://api.exchange.freedx.com/spot/api/v3.2/market_summary";
        
        match self.client
            .get(url)
            .timeout(Duration::from_secs(self.config.timeout_secs))
            .send()
            .await
        {
            Ok(response) => {
                match response.json::<Vec<FreedomTicker>>().await {
                    Ok(tickers) => Ok(tickers),
                    Err(e) => Err(format!("JSON parse error: {}", e)),
                }
            }
            Err(e) => Err(format!("Request error: {}", e)),
        }
    }

    fn publish_to_zmq(&self, topic: &str, data: &str) -> Result<(), String> {
        let socket = self.zmq_context
            .socket(zmq::PUB)
            .map_err(|e| format!("Socket creation error: {}", e))?;

        socket
            .bind(&self.config.zmq_endpoint)
            .map_err(|e| format!("ZMQ bind error: {}", e))?;

        socket
            .send(topic.as_bytes(), zmq::SNDMORE)
            .map_err(|e| format!("ZMQ send error: {}", e))?;

        socket
            .send(data.as_bytes(), 0)
            .map_err(|e| format!("ZMQ send error: {}", e))?;

        Ok(())
    }
}

#[async_trait::async_trait]
impl GatewayConnector for FreedomExchangeConnector {
    async fn connect(&self) -> Result<(), String> {
        println!("Connecting to FreeDOM Exchange...");
        
        // Test connection
        match self.fetch_tickers().await {
            Ok(tickers) => {
                println!("✓ Connected to FreeDOM Exchange ({} pairs)", tickers.len());
                Ok(())
            }
            Err(e) => {
                eprintln!("✗ Failed to connect: {}", e);
                Err(e)
            }
        }
    }

    async fn fetch_market_data(&self, symbols: Vec<String>) -> Result<Vec<MarketData>, String> {
        let tickers = self.fetch_tickers().await?;
        
        let filtered: Vec<MarketData> = tickers
            .into_iter()
            .filter(|t| symbols.is_empty() || symbols.contains(&t.symbol))
            .map(|t| MarketData {
                symbol: t.symbol,
                price: t.last,
                bid: t.highestBid,
                ask: t.lowestAsk,
                volume: t.volume,
                timestamp: chrono::Utc::now().timestamp_millis(),
                exchange: "freedx".to_string(),
            })
            .collect();

        Ok(filtered)
    }

    async fn start_stream(&self, topic: &str) -> Result<(), String> {
        println!("Starting market data stream on topic: {}", topic);
        
        loop {
            match self.fetch_market_data(vec![]).await {
                Ok(market_data) => {
                    for data in market_data {
                        let json = serde_json::to_string(&data)
                            .map_err(|e| format!("Serialization error: {}", e))?;
                        
                        self.publish_to_zmq(topic, &json)?;
                    }
                }
                Err(e) => eprintln!("Error fetching market data: {}", e),
            }

            sleep(Duration::from_millis(self.config.polling_interval_ms)).await;
        }
    }

    fn disconnect(&self) -> Result<(), String> {
        println!("Disconnected from FreeDOM Exchange");
        Ok(())
    }
}
