use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use zmq::Context;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketData {
    symbol: String,
    price: f64,
    volume: f64,
    timestamp: i64,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidatedData {
    symbol: String,
    price: f64,
    volume: f64,
    timestamp: i64,
    source: String,
    valid: bool,
    warnings: Vec<String>,
}

struct DataValidator {
    price_threshold: f64,
    volume_threshold: f64,
    processing_count: Arc<Mutex<u64>>,
    error_count: Arc<Mutex<u64>>,
}

impl DataValidator {
    fn new() -> Self {
        DataValidator {
            price_threshold: 1_000_000.0,      // Max reasonable price
            volume_threshold: 1_000_000_000.0, // Max reasonable volume
            processing_count: Arc::new(Mutex::new(0)),
            error_count: Arc::new(Mutex::new(0)),
        }
    }

    fn validate(&self, data: &MarketData) -> ValidatedData {
        let mut valid = true;
        let mut warnings = Vec::new();

        // Validate symbol format
        if data.symbol.is_empty() || !data.symbol.contains('/') {
            valid = false;
            warnings.push("Invalid symbol format".to_string());
        }

        // Validate price
        if data.price <= 0.0 || data.price.is_nan() || data.price.is_infinite() {
            valid = false;
            warnings.push("Invalid price: must be positive number".to_string());
        } else if data.price > self.price_threshold {
            warnings.push(format!("Price exceeds threshold: {}", data.price));
        }

        // Validate volume
        if data.volume < 0.0 || data.volume.is_nan() || data.volume.is_infinite() {
            valid = false;
            warnings.push("Invalid volume: cannot be negative".to_string());
        } else if data.volume > self.volume_threshold {
            warnings.push(format!("Volume exceeds threshold: {}", data.volume));
        }

        // Validate timestamp (should be recent, within 1 day)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        if data.timestamp < now - 86400 || data.timestamp > now + 86400 {
            warnings.push("Timestamp appears invalid (outside ±24h window)".to_string());
        }

        // Validate source
        if data.source.is_empty() {
            valid = false;
            warnings.push("Source missing".to_string());
        }

        ValidatedData {
            symbol: data.symbol.clone(),
            price: data.price,
            volume: data.volume,
            timestamp: data.timestamp,
            source: data.source.clone(),
            valid,
            warnings,
        }
    }

    fn process_batch(&self, items: Vec<MarketData>) -> Vec<ValidatedData> {
        items.iter().map(|item| self.validate(item)).collect()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Market Data Validator - Starting");

    let context = Context::new();
    
    // Subscribe to market data from publisher
    let subscriber = context.socket(zmq::SUB)?;
    subscriber.connect("tcp://127.0.0.1:5555")?;
    subscriber.set_subscribe(b"")?; // Subscribe to all topics

    // Publish validated data
    let publisher = context.socket(zmq::PUB)?;
    publisher.bind("tcp://127.0.0.1:5557")?;

    let validator = DataValidator::new();

    println!("Validator connected to ZMQ:");
    println!("  Subscribing from: tcp://127.0.0.1:5555");
    println!("  Publishing to: tcp://127.0.0.1:5557");
    println!();

    // Give publisher time to bind
    thread::sleep(Duration::from_millis(100));

    let mut items = [
        subscriber.as_poll_item(zmq::POLLIN),
    ];

    loop {
        // Poll with 1 second timeout
        zmq::poll(&mut items, 1000)?;

        if items[0].is_readable() {
            // Receive topic
            if let Ok(topic) = subscriber.recv_string(0) {
                if let Ok(topic_str) = topic {
                    // Receive data
                    if let Ok(data_str) = subscriber.recv_string(0) {
                        if let Ok(data_json) = data_str {
                            // Parse JSON
                            if let Ok(data) = serde_json::from_str::<MarketData>(&data_json) {
                                // Validate
                                let validated = validator.validate(&data);

                                // Send validated data
                                let validated_json = serde_json::to_string(&validated)
                                    .unwrap_or_else(|_| String::new());
                                
                                let validated_topic = format!("validated:{}", topic_str);
                                publisher.send(&validated_topic, zmq::SNDMORE)?;
                                publisher.send(&validated_json, 0)?;

                                // Update counter
                                if let Ok(mut count) = validator.processing_count.lock() {
                                    *count += 1;
                                    if *count % 100 == 0 {
                                        println!("Processed {} records", count);
                                    }
                                }

                                // Log warnings if any
                                if !validated.warnings.is_empty() {
                                    println!(
                                        "[{}] {} - Valid: {} - Warnings: {:?}",
                                        validated.source,
                                        validated.symbol,
                                        validated.valid,
                                        validated.warnings
                                    );
                                } else if *validator.processing_count.lock().unwrap() % 50 == 0 {
                                    println!(
                                        "[{}] {} - Price: {:.2}",
                                        validated.source, validated.symbol, validated.price
                                    );
                                }
                            } else if let Ok(mut errors) = validator.error_count.lock() {
                                *errors += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}
