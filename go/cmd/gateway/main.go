package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	zmq "github.com/pebbe/zmq4"
)

// MarketData represents price data from gateways
type MarketData struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Volume    float64   `json:"volume"`
	Timestamp time.Time `json:"timestamp"`
	Source    string    `json:"source"`
}

// GatewayManager handles multiple data sources
type GatewayManager struct {
	redisClient *redis.Client
	zmqSocket   *zmq.Socket
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
	mu          sync.RWMutex
	running     bool
}

// NewGatewayManager creates and initializes gateway manager
func NewGatewayManager(ctx context.Context) (*GatewayManager, error) {
	// Redis connection
	redisClient := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})

	if err := redisClient.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis connection failed: %w", err)
	}

	// ZMQ socket for publishing
	socket, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		return nil, fmt.Errorf("zmq socket creation failed: %w", err)
	}

	if err := socket.Connect("tcp://127.0.0.1:5555"); err != nil {
		return nil, fmt.Errorf("zmq connection failed: %w", err)
	}

	cancelCtx, cancel := context.WithCancel(ctx)

	return &GatewayManager{
		redisClient: redisClient,
		zmqSocket:   socket,
		ctx:         cancelCtx,
		cancel:      cancel,
		running:     true,
	}, nil
}

// FetchFreeDXData simulates fetching data from FreeDX gateway
func (gm *GatewayManager) FetchFreeDXData(ctx context.Context) {
	defer gm.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("FreeDX data fetcher stopped")
			return
		case <-ticker.C:
			// Simulate market data
			data := MarketData{
				Symbol:    "BTC/USD",
				Price:     float64(45000 + int((time.Now().Unix() % 1000))),
				Volume:    float64(100),
				Timestamp: time.Now(),
				Source:    "freedx",
			}

			gm.publishData(data)
		}
	}
}

// FetchGateIOData simulates fetching data from Gate.io gateway
func (gm *GatewayManager) FetchGateIOData(ctx context.Context) {
	defer gm.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Gate.io data fetcher stopped")
			return
		case <-ticker.C:
			// Simulate market data
			data := MarketData{
				Symbol:    "ETH/USD",
				Price:     float64(2500 + int((time.Now().Unix() % 500))),
				Volume:    float64(500),
				Timestamp: time.Now(),
				Source:    "gateio",
			}

			gm.publishData(data)
		}
	}
}

// publishData publishes data to Redis and ZMQ
func (gm *GatewayManager) publishData(data MarketData) {
	// Store in Redis with TTL
	key := fmt.Sprintf("market:%s", data.Symbol)
	jsonData, _ := json.Marshal(data)

	if err := gm.redisClient.Set(gm.ctx, key, string(jsonData), 30*time.Second).Err(); err != nil {
		log.Printf("Redis error: %v", err)
	}

	// Store historical data in Redis list (last 1000 records)
	historyKey := fmt.Sprintf("history:%s", data.Symbol)
	gm.redisClient.LPush(gm.ctx, historyKey, string(jsonData))
	gm.redisClient.LTrim(gm.ctx, historyKey, 0, 999)

	// Publish to ZMQ
	topic := fmt.Sprintf("%s:%s", data.Source, data.Symbol)
	if _, err := gm.zmqSocket.Send(topic, zmq.SNDMORE); err != nil {
		log.Printf("ZMQ send topic error: %v", err)
	}
	if _, err := gm.zmqSocket.Send(string(jsonData), 0); err != nil {
		log.Printf("ZMQ send data error: %v", err)
	}

	log.Printf("[%s] %s: %.2f (Vol: %.0f)", data.Source, data.Symbol, data.Price, data.Volume)
}

// Start begins data collection
func (gm *GatewayManager) Start() {
	gm.wg.Add(2)
	go gm.FetchFreeDXData(gm.ctx)
	go gm.FetchGateIOData(gm.ctx)
	log.Println("Gateway manager started - fetching from FreeDX and Gate.io")
}

// Stop gracefully shuts down
func (gm *GatewayManager) Stop() {
	gm.mu.Lock()
	if !gm.running {
		gm.mu.Unlock()
		return
	}
	gm.running = false
	gm.mu.Unlock()

	log.Println("Stopping gateway manager...")
	gm.cancel()
	gm.wg.Wait()

	if gm.zmqSocket != nil {
		gm.zmqSocket.Close()
	}
	if gm.redisClient != nil {
		gm.redisClient.Close()
	}

	log.Println("Gateway manager stopped")
}

// HealthHandler returns health status
func (gm *GatewayManager) HealthHandler(w http.ResponseWriter, r *http.Request) {
	gm.mu.RLock()
	running := gm.running
	gm.mu.RUnlock()

	if !running {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "down",
			"error":  "gateway manager not running",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"uptime": time.Now().Unix(),
	})
}

// MetricsHandler returns current market data
func (gm *GatewayManager) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	keys, err := gm.redisClient.Keys(gm.ctx, "market:*").Result()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	metrics := make(map[string]interface{})
	for _, key := range keys {
		val, err := gm.redisClient.Get(gm.ctx, key).Result()
		if err == nil {
			var data MarketData
			if json.Unmarshal([]byte(val), &data) == nil {
				metrics[key] = data
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create gateway manager
	gm, err := NewGatewayManager(ctx)
	if err != nil {
		log.Fatalf("Failed to create gateway manager: %v", err)
	}

	// Start data collection
	gm.Start()

	// HTTP routes
	http.HandleFunc("/health", gm.HealthHandler)
	http.HandleFunc("/metrics", gm.MetricsHandler)

	// Start HTTP server
	server := &http.Server{
		Addr:         ":8080",
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}

	go func() {
		log.Println("Gateway listening on :8080")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	// Graceful shutdown handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	<-sigChan
	log.Println("\nShutdown signal received")

	// Graceful stop
	gm.Stop()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	server.Shutdown(shutdownCtx)

	log.Println("Gateway shutdown complete")
}
