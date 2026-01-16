package main

import (
	"crypto/hmac"
	"crypto/sha512"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-resty/resty/v2"
	jsoniter "github.com/json-iterator/go"
	"github.com/gorilla/websocket"
	zmq "github.com/pebbe/zmq4"
)

// ============================================================================
// Gate.io REST API Client
// ============================================================================

type GateIOClient struct {
	BaseURL    string
	Key        string
	Secret     string
	UserID     string
	httpClient *resty.Client
	mu         sync.RWMutex
}

type OHLCData struct {
	Timestamp int64   `json:"timestamp"`
	Open      float64 `json:"open,string"`
	High      float64 `json:"high,string"`
	Low       float64 `json:"low,string"`
	Close     float64 `json:"close,string"`
	Volume    float64 `json:"volume,string"`
	QuoteVol  float64 `json:"quote_volume,string"`
}

type TickerData struct {
	CurrencyPair string    `json:"currency_pair"`
	Last         float64   `json:"last,string"`
	LowestAsk    float64   `json:"lowest_ask,string"`
	HighestBid   float64   `json:"highest_bid,string"`
	ChangePercent float64  `json:"change_percentage,string"`
	High24h      float64   `json:"high_24h,string"`
	Low24h       float64   `json:"low_24h,string"`
	Volume24h    float64   `json:"volume_24h,string"`
}

func NewGateIOClient(baseURL, key, secret, userID string) *GateIOClient {
	return &GateIOClient{
		BaseURL:    baseURL,
		Key:        key,
		Secret:     secret,
		UserID:     userID,
		httpClient: resty.New().SetTimeout(10 * time.Second),
	}
}

// getSignature generates signature for authenticated requests
func (c *GateIOClient) getSignature(method, path, body, timestamp string) string {
	message := fmt.Sprintf("%s\n%s\n%s\n%s\n%s", method, path, body, c.Key, timestamp)
	h := hmac.New(sha512.New, []byte(c.Secret))
	h.Write([]byte(message))
	return hex.EncodeToString(h.Sum(nil))
}

// GetOHLCData retrieves OHLC data for a symbol
func (c *GateIOClient) GetOHLCData(symbol string, interval string, limit int) ([]OHLCData, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	path := fmt.Sprintf("/api/v4/spot/candlesticks?currency_pair=%s&interval=%s&limit=%d",
		symbol, interval, limit)
	url := c.BaseURL + path

	resp, err := c.httpClient.R().Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get OHLC data: %w", err)
	}

	if resp.StatusCode() != http.StatusOK {
		return nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode(), resp.String())
	}

	var data []OHLCData
	json := jsoniter.ConfigCompatibleWithStandardLibrary
	if err := json.Unmarshal(resp.Body(), &data); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return data, nil
}

// GetTicker retrieves ticker information
func (c *GateIOClient) GetTicker(symbol string) (*TickerData, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	path := fmt.Sprintf("/api/v4/spot/tickers?currency_pair=%s", symbol)
	url := c.BaseURL + path

	resp, err := c.httpClient.R().Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get ticker: %w", err)
	}

	if resp.StatusCode() != http.StatusOK {
		return nil, fmt.Errorf("API error: status %d", resp.StatusCode())
	}

	var data []TickerData
	json := jsoniter.ConfigCompatibleWithStandardLibrary
	if err := json.Unmarshal(resp.Body(), &data); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("no data returned")
	}

	return &data[0], nil
}

// ============================================================================
// WebSocket Connection Handler
// ============================================================================

type WebSocketConnector struct {
	URL               string
	conn              *websocket.Conn
	zmqPublisher      *zmq.Socket
	reconnectInterval time.Duration
	mu                sync.RWMutex
	done              chan struct{}
}

func NewWebSocketConnector(url string, zmqPublisher *zmq.Socket) *WebSocketConnector {
	return &WebSocketConnector{
		URL:               url,
		zmqPublisher:      zmqPublisher,
		reconnectInterval: 5 * time.Second,
		done:              make(chan struct{}),
	}
}

// Connect establishes WebSocket connection
func (wsc *WebSocketConnector) Connect() error {
	wsc.mu.Lock()
	defer wsc.mu.Unlock()

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(wsc.URL, nil)
	if err != nil {
		return fmt.Errorf("websocket dial failed: %w", err)
	}

	wsc.conn = conn
	return nil
}

// Subscribe to market data stream
func (wsc *WebSocketConnector) Subscribe(channel string, symbols []string) error {
	wsc.mu.RLock()
	conn := wsc.conn
	wsc.mu.RUnlock()

	if conn == nil {
		return fmt.Errorf("not connected")
	}

	subscriptions := []interface{}{
		"subscribe",
		channel,
	}

	for _, symbol := range symbols {
		subscriptions = append(subscriptions, symbol)
	}

	return conn.WriteJSON(subscriptions)
}

// Listen processes incoming messages
func (wsc *WebSocketConnector) Listen() error {
	wsc.mu.RLock()
	conn := wsc.conn
	wsc.mu.RUnlock()

	if conn == nil {
		return fmt.Errorf("not connected")
	}

	for {
		select {
		case <-wsc.done:
			return nil
		default:
			_, message, err := conn.ReadMessage()
			if err != nil {
				return fmt.Errorf("read error: %w", err)
			}

			// Publish to ZMQ
			if _, err := wsc.zmqPublisher.Send(string(message), 0); err != nil {
				log.Printf("ZMQ publish error: %v", err)
			}
		}
	}
}

// Close closes the connection
func (wsc *WebSocketConnector) Close() error {
	close(wsc.done)
	wsc.mu.Lock()
	defer wsc.mu.Unlock()

	if wsc.conn != nil {
		return wsc.conn.Close()
	}
	return nil
}

// ============================================================================
// ZMQ Publisher
// ============================================================================

type ZMQPublisher struct {
	address string
	socket  *zmq.Socket
	context *zmq.Context
	mu      sync.RWMutex
}

func NewZMQPublisher(host string, port int) (*ZMQPublisher, error) {
	context, _ := zmq.NewContext()
	socket, _ := context.NewSocket(zmq.PUB)

	address := fmt.Sprintf("tcp://*:%d", port)
	if err := socket.Bind(address); err != nil {
		return nil, fmt.Errorf("bind failed: %w", err)
	}

	return &ZMQPublisher{
		address: address,
		socket:  socket,
		context: context,
	}, nil
}

func (zp *ZMQPublisher) Publish(topic string, message interface{}) error {
	zp.mu.Lock()
	defer zp.mu.Unlock()

	data, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("marshal error: %w", err)
	}

	envelope := fmt.Sprintf("%s:%s", topic, string(data))
	_, err = zp.socket.Send(envelope, 0)
	return err
}

func (zp *ZMQPublisher) Close() error {
	zp.mu.Lock()
	defer zp.mu.Unlock()

	if zp.socket != nil {
		zp.socket.Close()
	}
	if zp.context != nil {
		zp.context.Term()
	}
	return nil
}

// ============================================================================
// Main Gateway Service
// ============================================================================

type GatewayService struct {
	gateioClient *GateIOClient
	wsConnector  *WebSocketConnector
	zmqPublisher *ZMQPublisher
	config       GatewayConfig
	mu           sync.RWMutex
}

type GatewayConfig struct {
	GateIOKey        string
	GateIOSecret     string
	GateIOUserID     string
	GateIORestURL    string
	GateIOWSURL      string
	ZMQHost          string
	ZMQPort          int
	UpdateInterval   time.Duration
	SymbolList       []string
}

func NewGatewayService(config GatewayConfig) (*GatewayService, error) {
	// Initialize Gate.io client
	gateioClient := NewGateIOClient(
		config.GateIORestURL,
		config.GateIOKey,
		config.GateIOSecret,
		config.GateIOUserID,
	)

	// Initialize ZMQ publisher
	zmqPublisher, err := NewZMQPublisher(config.ZMQHost, config.ZMQPort)
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ publisher: %w", err)
	}

	// Initialize WebSocket connector
	wsConnector := NewWebSocketConnector(config.GateIOWSURL, zmqPublisher.socket)

	service := &GatewayService{
		gateioClient: gateioClient,
		wsConnector:  wsConnector,
		zmqPublisher: zmqPublisher,
		config:       config,
	}

	return service, nil
}

// Start initiates the gateway service
func (gs *GatewayService) Start() error {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	// Connect WebSocket
	if err := gs.wsConnector.Connect(); err != nil {
		log.Printf("WebSocket connection failed: %v", err)
	} else {
		log.Println("WebSocket connected")

		// Subscribe to candlestick updates
		for _, symbol := range gs.config.SymbolList {
			gs.wsConnector.Subscribe("spot.candlesticks", []string{symbol})
		}

		go func() {
			if err := gs.wsConnector.Listen(); err != nil {
				log.Printf("WebSocket listen error: %v", err)
			}
		}()
	}

	// Start periodic REST API updates
	go gs.periodicUpdate()

	return nil
}

// periodicUpdate periodically fetches data via REST API
func (gs *GatewayService) periodicUpdate() {
	ticker := time.NewTicker(gs.config.UpdateInterval)
	defer ticker.Stop()

	for range ticker.C {
		for _, symbol := range gs.config.SymbolList {
			// Get ticker data
			ticker, err := gs.gateioClient.GetTicker(symbol)
			if err != nil {
				log.Printf("Failed to get ticker for %s: %v", symbol, err)
				continue
			}

			// Publish to ZMQ
			message := map[string]interface{}{
				"symbol":    symbol,
				"timestamp": time.Now().Unix(),
				"ticker":    ticker,
			}

			if err := gs.zmqPublisher.Publish("gateio.ticker", message); err != nil {
				log.Printf("Failed to publish ticker: %v", err)
			}

			// Get OHLC data
			ohlc, err := gs.gateioClient.GetOHLCData(symbol, "5m", 10)
			if err != nil {
				log.Printf("Failed to get OHLC for %s: %v", symbol, err)
				continue
			}

			ohlcMessage := map[string]interface{}{
				"symbol":    symbol,
				"timestamp": time.Now().Unix(),
				"ohlc":      ohlc,
			}

			if err := gs.zmqPublisher.Publish("gateio.ohlc", ohlcMessage); err != nil {
				log.Printf("Failed to publish OHLC: %v", err)
			}
		}
	}
}

// Stop gracefully shuts down the gateway
func (gs *GatewayService) Stop() error {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	if gs.wsConnector != nil {
		gs.wsConnector.Close()
	}
	if gs.zmqPublisher != nil {
		gs.zmqPublisher.Close()
	}
	return nil
}

// ============================================================================
// Main Entry Point
// ============================================================================

func main() {
	config := GatewayConfig{
		GateIORestURL: "https://api.gateio.ws",
		GateIOWSURL:   "wss://ws.gate.io/v4",
		ZMQHost:       "127.0.0.1",
		ZMQPort:       5555,
		UpdateInterval: 5 * time.Second,
		SymbolList: []string{
			"ETH_USDT",
			"BTC_USDT",
			"BNB_USDT",
		},
	}

	// Create gateway service
	service, err := NewGatewayService(config)
	if err != nil {
		log.Fatalf("Failed to create gateway service: %v", err)
	}

	// Start service
	if err := service.Start(); err != nil {
		log.Fatalf("Failed to start gateway service: %v", err)
	}

	log.Println("Gate.io Gateway started successfully")

	// Keep running
	select {}
}
