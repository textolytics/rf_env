// zmq_core/publisher.c - High-performance ZMQ Publisher for Market Data
// Compiled as: gcc -O3 -Wall publisher.c -o publisher -lzmq

#include <zmq.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>

static volatile int should_stop = 0;

void signal_handler(int sig) {
    should_stop = 1;
}

typedef struct {
    char symbol[32];
    double price;
    double volume;
    long timestamp;
} MarketData;

// Publish market data
int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_PUB);
    
    // Set high-water mark for performance
    int hwm = 10000;
    zmq_setsockopt(socket, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    
    // Bind to port
    int rc = zmq_bind(socket, "tcp://127.0.0.1:5555");
    assert(rc == 0);
    printf("[ZMQ Publisher] Bound to tcp://127.0.0.1:5555\n");
    fflush(stdout);
    
    MarketData data;
    struct timespec ts;
    
    while (!should_stop) {
        // Simulate market data
        strcpy(data.symbol, "BTC-USDT");
        data.price = 45000.00 + (rand() % 1000) - 500;
        data.volume = 100.5 + (rand() % 50);
        
        clock_gettime(CLOCK_REALTIME, &ts);
        data.timestamp = (ts.tv_sec * 1000LL) + (ts.tv_nsec / 1000000LL);
        
        // Publish with envelope
        char envelope[256];
        snprintf(envelope, sizeof(envelope), "market.BTC-USDT %ld", data.timestamp);
        zmq_send(socket, envelope, strlen(envelope), ZMQ_SNDMORE);
        zmq_send(socket, (void*)&data, sizeof(data), 0);
        
        usleep(100000); // 100ms
    }
    
    zmq_close(socket);
    zmq_ctx_destroy(context);
    printf("[ZMQ Publisher] Gracefully stopped\n");
    return 0;
}
