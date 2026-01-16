// Market Data Platform - C++ Connectivity Module
// Low-latency WebSocket to ZMQ bridge for Gate.io, Betfair
// Handles: WebSocket streaming, high-frequency updates, ZMQ publishing

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

using json = nlohmann::json;
using websocket_client = websocketpp::client<websocketpp::config::asio_client>;

struct MarketData {
    std::string symbol;
    double price;
    double bid;
    double ask;
    double volume;
    long timestamp;
    std::string exchange;
};

class GatewayConnector {
public:
    virtual ~GatewayConnector() = default;
    
    virtual bool connect() = 0;
    virtual bool start_stream(const std::string& topic) = 0;
    virtual bool disconnect() = 0;
    virtual std::vector<MarketData> fetch_market_data(const std::vector<std::string>& symbols) = 0;
};

class WebSocketConnector : public GatewayConnector {
protected:
    std::string ws_url_;
    std::string zmq_endpoint_;
    std::unique_ptr<websocket_client> client_;
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_socket_;
    bool connected_ = false;

public:
    WebSocketConnector(const std::string& ws_url, const std::string& zmq_endpoint)
        : ws_url_(ws_url), zmq_endpoint_(zmq_endpoint) {
        zmq_context_ = std::make_unique<zmq::context_t>(1);
        zmq_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::pub);
    }

    virtual ~WebSocketConnector() {
        disconnect();
    }

    bool connect() override {
        try {
            client_ = std::make_unique<websocket_client>();
            
            // Configure logging
            client_->set_access_channels(websocketpp::log::alevel::all);
            client_->clear_access_channels(websocketpp::log::alevel::frame_payload);
            
            // Initialize ASIO
            client_->init_asio();
            
            // Bind ZMQ socket
            zmq_socket_->bind(zmq_endpoint_);
            
            connected_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            return false;
        }
    }

    bool disconnect() override {
        if (connected_) {
            try {
                if (client_) {
                    client_->stop();
                }
                zmq_socket_->close();
            } catch (const std::exception& e) {
                std::cerr << "Disconnection error: " << e.what() << std::endl;
                return false;
            }
            connected_ = false;
        }
        return true;
    }

    bool start_stream(const std::string& topic) override = 0;
    
    std::vector<MarketData> fetch_market_data(const std::vector<std::string>& symbols) override = 0;

protected:
    void publish_to_zmq(const std::string& topic, const json& data) {
        try {
            std::string topic_msg = topic;
            std::string data_msg = data.dump();

            zmq::message_t msg_topic(topic_msg.begin(), topic_msg.end());
            zmq::message_t msg_data(data_msg.begin(), data_msg.end());

            zmq_socket_->send(msg_topic, zmq::send_flags::sndmore);
            zmq_socket_->send(msg_data);
        } catch (const zmq::error_t& e) {
            std::cerr << "ZMQ publish error: " << e.what() << std::endl;
        }
    }
};

// Gate.io WebSocket Connector
class GateIOConnector : public WebSocketConnector {
public:
    GateIOConnector(const std::string& zmq_endpoint)
        : WebSocketConnector("wss://ws.gate.io/v4", zmq_endpoint) {}

    bool start_stream(const std::string& topic) override {
        if (!connected_) return false;

        // Subscribe to channels
        json subscribe_msg = {
            {"time", 1234567890},
            {"channel", "spot.tickers"},
            {"event", "subscribe"}
        };

        // Send subscription message
        // Implementation depends on WebSocket library specifics
        
        return true;
    }

    std::vector<MarketData> fetch_market_data(const std::vector<std::string>& symbols) override {
        std::vector<MarketData> result;
        // Implementation for fetching REST data
        return result;
    }
};

// Betfair Streaming Connector
class BetfairConnector : public WebSocketConnector {
public:
    BetfairConnector(const std::string& zmq_endpoint)
        : WebSocketConnector("wss://stream-api.betfair.com", zmq_endpoint) {}

    bool start_stream(const std::string& topic) override {
        if (!connected_) return false;
        // Betfair-specific streaming logic
        return true;
    }

    std::vector<MarketData> fetch_market_data(const std::vector<std::string>& symbols) override {
        std::vector<MarketData> result;
        // Implementation for Betfair market data
        return result;
    }
};

class ConnectivityManager {
public:
    static std::unique_ptr<GatewayConnector> create_connector(
        const std::string& gateway_name,
        const std::string& zmq_endpoint) {
        
        if (gateway_name == "gateio") {
            return std::make_unique<GateIOConnector>(zmq_endpoint);
        } else if (gateway_name == "betfair") {
            return std::make_unique<BetfairConnector>(zmq_endpoint);
        }
        return nullptr;
    }
};
