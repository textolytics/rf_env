// Market Data Platform - Rust Connectivity Module
// High-performance REST/WebSocket to ZMQ bridge for FreeDOM, Kraken
// Handles: REST polling, WebSocket streaming, ZMQ publishing

use std::sync::Arc;
use tokio::sync::RwLock;
use zmq::Context;

pub mod freedx;
pub mod kraken;
pub mod zmq_publisher;
pub mod metrics;

#[derive(Clone, Debug)]
pub struct ConnectivityConfig {
    pub gateway_name: String,
    pub zmq_endpoint: String,
    pub polling_interval_ms: u64,
    pub max_retries: u32,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub bid: f64,
    pub ask: f64,
    pub volume: f64,
    pub timestamp: i64,
    pub exchange: String,
}

pub trait GatewayConnector: Send + Sync {
    async fn connect(&self) -> Result<(), String>;
    async fn fetch_market_data(&self, symbols: Vec<String>) -> Result<Vec<MarketData>, String>;
    async fn start_stream(&self, topic: &str) -> Result<(), String>;
    fn disconnect(&self) -> Result<(), String>;
}

pub struct PlatformManager {
    zmq_context: Arc<Context>,
    config: ConnectivityConfig,
    metrics: Arc<RwLock<metrics::PlatformMetrics>>,
}

impl PlatformManager {
    pub fn new(config: ConnectivityConfig) -> Self {
        PlatformManager {
            zmq_context: Arc::new(Context::new()),
            config,
            metrics: Arc::new(RwLock::new(metrics::PlatformMetrics::new())),
        }
    }

    pub async fn start_gateway(&self, gateway_type: &str) -> Result<(), String> {
        match gateway_type {
            "freedx" => {
                let connector = freedx::FreedomExchangeConnector::new(
                    self.config.clone(),
                    Arc::clone(&self.zmq_context),
                );
                connector.connect().await
            }
            "kraken" => {
                let connector = kraken::KrakenConnector::new(
                    self.config.clone(),
                    Arc::clone(&self.zmq_context),
                );
                connector.connect().await
            }
            _ => Err(format!("Unknown gateway: {}", gateway_type)),
        }
    }

    pub async fn get_metrics(&self) -> metrics::PlatformMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn update_metrics(&self, data: MarketData) {
        let mut m = self.metrics.write().await;
        m.update_from_market_data(&data);
    }
}

// Example usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConnectivityConfig {
        gateway_name: "freedx".to_string(),
        zmq_endpoint: "tcp://127.0.0.1:5559".to_string(),
        polling_interval_ms: 1000,
        max_retries: 3,
        timeout_secs: 10,
    };

    let platform = PlatformManager::new(config);
    platform.start_gateway("freedx").await?;

    Ok(())
}
