# Go Gateway Service - Multi-stage Build
FROM golang:1.21-alpine AS builder

WORKDIR /build

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    libzmq-dev \
    git

# Copy go modules
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o gateway ./cmd/gateway

# Runtime stage
FROM alpine:3.18

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libzmq \
    postgresql-client

# Copy binary from builder
COPY --from=builder /build/gateway .

# Create non-root user
RUN addgroup -g 1000 appuser && adduser -D -u 1000 -G appuser appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9000

# Default command
CMD ["./gateway"]
