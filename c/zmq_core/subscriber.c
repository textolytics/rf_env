// zmq_core/subscriber.c - High-performance ZMQ Subscriber and Router
// Compiled as: gcc -O3 -Wall subscriber.c -o subscriber -lzmq -lpthread

#include <zmq.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <signal.h>

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

// Worker thread for message processing
void *worker_thread(void *arg) {
    void *context = (void*)arg;
    void *socket = zmq_socket(context, ZMQ_DEALER);
    zmq_connect(socket, "tcp://127.0.0.1:5556");
    
    char envelope[256];
    MarketData data;
    
    while (!should_stop) {
        int size = zmq_recv(socket, envelope, 255, ZMQ_DONTWAIT);
        if (size > 0) {
            envelope[size] = '\0';
            size = zmq_recv(socket, (void*)&data, sizeof(data), ZMQ_DONTWAIT);
            if (size > 0) {
                printf("[Router] Market Data: %s | Price: %.2f | Vol: %.2f | TS: %ld\n",
                       data.symbol, data.price, data.volume, data.timestamp);
                fflush(stdout);
            }
        }
        usleep(10000); // 10ms
    }
    
    zmq_close(socket);
    return NULL;
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    void *context = zmq_ctx_new();
    
    // SUB socket for market data
    void *sub_socket = zmq_socket(context, ZMQ_SUB);
    zmq_setsockopt(sub_socket, ZMQ_SUBSCRIBE, "market", 6);
    zmq_connect(sub_socket, "tcp://127.0.0.1:5555");
    printf("[ZMQ Router] Connected to publisher at tcp://127.0.0.1:5555\n");
    
    // ROUTER socket for load distribution
    void *router_socket = zmq_socket(context, ZMQ_ROUTER);
    int hwm = 10000;
    zmq_setsockopt(router_socket, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    zmq_setsockopt(router_socket, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    zmq_bind(router_socket, "tcp://127.0.0.1:5556");
    printf("[ZMQ Router] Bound to tcp://127.0.0.1:5556\n");
    
    // Start worker threads
    pthread_t workers[4];
    for (int i = 0; i < 4; i++) {
        pthread_create(&workers[i], NULL, worker_thread, context);
    }
    
    // Main loop: receive and route
    char envelope[256];
    MarketData data;
    
    while (!should_stop) {
        int size = zmq_recv(sub_socket, envelope, 255, ZMQ_DONTWAIT);
        if (size > 0) {
            envelope[size] = '\0';
            size = zmq_recv(sub_socket, (void*)&data, sizeof(data), ZMQ_DONTWAIT);
            if (size > 0) {
                // Send to clients via ROUTER
                zmq_send(router_socket, "CLIENT", 6, ZMQ_SNDMORE);
                zmq_send(router_socket, envelope, strlen(envelope), ZMQ_SNDMORE);
                zmq_send(router_socket, (void*)&data, sizeof(data), 0);
            }
        }
        usleep(1000); // 1ms
    }
    
    zmq_close(sub_socket);
    zmq_close(router_socket);
    zmq_ctx_destroy(context);
    
    printf("[ZMQ Router] Gracefully stopped\n");
    return 0;
}
