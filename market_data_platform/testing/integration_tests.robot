*** Settings ***
Documentation     Market Data Platform - Integration Test Suite
...               Tests connectivity modules, data storage, and CLI
...               Validates all gateway integrations and data quality

Library           RequestsLibrary
Library           Collections
Library           String
Library           DateTime
Library           JSONLibrary
Library           Process
Library           OperatingSystem

Resource          ../keywords/gateway_keywords.robot
Resource          ../keywords/storage_keywords.robot
Resource          ../keywords/cli_keywords.robot

Suite Setup       Initialize Test Environment
Suite Teardown    Cleanup Test Environment

*** Variables ***
${INFLUXDB_URL}              http://localhost:8086
${INFLUXDB_ORG}              market_data
${INFLUXDB_BUCKET}           market_data_bucket
${INFLUXDB_TOKEN}            market_data_token_change_me

${CLI_EXECUTABLE}            ${CURDIR}/../cli/terminal.py
${GATEWAY_CONFIG}            ${CURDIR}/../config/gateways.yaml

*** Test Cases ***

# =============================================================================
# Gateway Connectivity Tests
# =============================================================================

FreeDOM Exchange Gateway - Connect and Retrieve Data
    [Documentation]    Test FreeDOM Exchange REST API connectivity
    [Tags]    gateway    freedx    connectivity
    Connect To Gateway    freedx
    ${data}=    Fetch Market Data    freedx    limit=10
    Should Not Be Empty    ${data}    msg=No market data returned from FreeDOM
    Validate Market Data Structure    ${data}
    Disconnect From Gateway    freedx

Gate.io Gateway - WebSocket Connectivity
    [Documentation]    Test Gate.io WebSocket streaming
    [Tags]    gateway    gateio    websocket
    Connect To Gateway    gateio
    ${stream_status}=    Start Data Stream    gateio    gateio.tickers
    Should Be Equal    ${stream_status}    success
    Sleep    3s    # Allow stream to collect data
    Stop Data Stream    gateio    gateio.tickers
    Disconnect From Gateway    gateio

OANDA Gateway - Forex API Integration
    [Documentation]    Test OANDA forex connectivity and EURUSD pair
    [Tags]    gateway    oanda    forex
    [Setup]    Skip If    ${OANDA_API_TOKEN} == ${EMPTY}    msg=OANDA token not configured
    Connect To Gateway    oanda
    ${eurusd_data}=    Fetch Market Data    oanda    symbol=EUR_USD
    Should Not Be Empty    ${eurusd_data}
    Validate Forex Data Format    ${eurusd_data}
    Disconnect From Gateway    oanda

Kraken Gateway - Hybrid REST/WebSocket
    [Documentation]    Test Kraken hybrid connectivity
    [Tags]    gateway    kraken    hybrid
    Connect To Gateway    kraken
    ${rest_data}=    Fetch Market Data    kraken    pair=XEUR
    Should Not Be Empty    ${rest_data}
    ${ws_status}=    Start Data Stream    kraken    kraken.eurusd_depth
    Should Be Equal    ${ws_status}    success
    Stop Data Stream    kraken    kraken.eurusd_depth
    Disconnect From Gateway    kraken

Twitter Sentiment Gateway - Stream Analysis
    [Documentation]    Test Twitter sentiment analysis
    [Tags]    gateway    twitter    sentiment
    [Setup]    Skip If    ${TWITTER_BEARER_TOKEN} == ${EMPTY}    msg=Twitter token not configured
    Connect To Gateway    twitter
    ${sentiment_data}=    Analyze Sentiment    crypto
    Should Not Be Empty    ${sentiment_data}
    Validate Sentiment Score    ${sentiment_data}
    Disconnect From Gateway    twitter

Betfair Gateway - Streaming Market Data
    [Documentation]    Test Betfair betting exchange
    [Tags]    gateway    betfair    streaming
    Connect To Gateway    betfair
    ${ws_status}=    Start Data Stream    betfair    betfair.market_books
    Should Be Equal    ${ws_status}    success
    Stop Data Stream    betfair    betfair.market_books
    Disconnect From Gateway    betfair

# =============================================================================
# ZMQ Broker Tests
# =============================================================================

ZMQ Broker - Publish Subscribe Pattern
    [Documentation]    Test ZMQ pub/sub functionality
    [Tags]    zmq    broker    messaging
    [Setup]    Start ZMQ Broker
    Subscribe To Topic    freedx.market_summary
    Publish Message    freedx.market_summary    {"symbol":"BTC-USDT","price":45000}
    ${message}=    Receive Message    2s
    Should Contain    ${message}    symbol
    Unsubscribe From Topic    freedx.market_summary
    [Teardown]    Stop ZMQ Broker

ZMQ Broker - Multiple Topics
    [Documentation]    Test multiple topic subscriptions
    [Tags]    zmq    broker    topics
    [Setup]    Start ZMQ Broker
    Subscribe To Topic    freedx.market_summary
    Subscribe To Topic    gateio.tickers
    Subscribe To Topic    oanda.eurusd
    ${topic1_msg}=    Receive Message From Topic    freedx.market_summary    2s
    ${topic2_msg}=    Receive Message From Topic    gateio.tickers    2s
    Should Not Be Empty    ${topic1_msg}
    Should Not Be Empty    ${topic2_msg}
    [Teardown]    Stop ZMQ Broker

ZMQ Broker - High Throughput
    [Documentation]    Test ZMQ broker with high message volume
    [Tags]    zmq    broker    performance    stress
    [Setup]    Start ZMQ Broker
    Subscribe To Topic    freedx.market_summary
    
    ${start_time}=    Get Current Date    result_format=epoch
    
    FOR    ${i}    IN RANGE    1000
        Publish Message    freedx.market_summary    {"tick":${i},"price":${45000 + i*0.5}}
    END
    
    ${end_time}=    Get Current Date    result_format=epoch
    ${duration}=    Evaluate    ${end_time} - ${start_time}
    
    ${throughput}=    Evaluate    1000 / ${duration}
    Log    Throughput: ${throughput} messages/sec
    Should Be True    ${throughput} > 100    msg=Throughput below 100 msg/s
    
    [Teardown]    Stop ZMQ Broker

# =============================================================================
# InfluxDB Storage Tests
# =============================================================================

InfluxDB - Write Market Tick Data
    [Documentation]    Test market data persistence
    [Tags]    storage    influxdb    market_data
    [Setup]    Connect To InfluxDB
    Write Market Tick    EUR_USD    1.0850    1.0849    1.0851    1000000    oanda
    ${stored_data}=    Query Market Tick    EUR_USD    oanda
    Should Not Be Empty    ${stored_data}
    [Teardown]    Disconnect From InfluxDB

InfluxDB - Write Trade Data
    [Documentation]    Test trade data persistence
    [Tags]    storage    influxdb    trades
    [Setup]    Connect To InfluxDB
    Write Trade Data    EUR_USD    100    1.0850    buy    oanda
    ${trades}=    Query Recent Trades    EUR_USD    oanda
    Should Not Be Empty    ${trades}
    [Teardown]    Disconnect From InfluxDB

InfluxDB - Write Sentiment Data
    [Documentation]    Test sentiment data persistence
    [Tags]    storage    influxdb    sentiment
    [Setup]    Connect To InfluxDB
    Write Sentiment Data    crypto    0.65    2453    twitter
    ${sentiment}=    Query Sentiment    crypto    twitter
    Should Not Be Empty    ${sentiment}
    [Teardown]    Disconnect From InfluxDB

InfluxDB - Query OHLC Data
    [Documentation]    Test OHLC query generation
    [Tags]    storage    influxdb    queries
    [Setup]    Connect To InfluxDB
    ${ohlc_data}=    Query OHLC    EUR_USD    oanda    1h    -7d
    Should Not Be Empty    ${ohlc_data}
    [Teardown]    Disconnect From InfluxDB

InfluxDB - Data Retention Policy
    [Documentation]    Test retention policy enforcement
    [Tags]    storage    influxdb    retention
    [Setup]    Connect To InfluxDB
    Set Retention Policy    30d
    Verify Retention Setting    30d
    [Teardown]    Disconnect From InfluxDB

# =============================================================================
# CLI Tests
# =============================================================================

CLI - Connect Command
    [Documentation]    Test CLI connect functionality
    [Tags]    cli    commands    connectivity
    Start CLI
    Execute CLI Command    connect freedx
    Verify Output Contains    Connected to freedx
    Stop CLI

CLI - Streaming Command
    [Documentation]    Test CLI data streaming
    [Tags]    cli    commands    streaming
    Start CLI
    Execute CLI Command    stream freedx.market_summary
    Verify Output Contains    Streaming
    Execute CLI Command    stop freedx.market_summary
    Stop CLI

CLI - Price Query
    [Documentation]    Test CLI price query
    [Tags]    cli    commands    queries
    Start CLI
    Execute CLI Command    price EURUSD
    Verify Output Contains    Current Price
    Stop CLI

CLI - Sentiment Analysis
    [Documentation]    Test CLI sentiment display
    [Tags]    cli    commands    sentiment
    Start CLI
    Execute CLI Command    sentiment crypto
    Verify Output Contains    Sentiment
    Stop CLI

CLI - Command Auto-Completion
    [Documentation]    Test CLI command completion
    [Tags]    cli    ui    completion
    Start CLI
    Test Auto Completion    connect free    ${GATEWAYS}
    Test Auto Completion    stream gateio    ${ZMQ_TOPICS}
    Stop CLI

CLI - Export Functionality
    [Documentation]    Test data export
    [Tags]    cli    commands    export
    Start CLI
    Execute CLI Command    export json /tmp/market_data.json
    Verify File Exists    /tmp/market_data.json
    Stop CLI

# =============================================================================
# Data Quality Tests
# =============================================================================

Data Quality - Price Range Validation
    [Documentation]    Validate price data is within reasonable ranges
    [Tags]    data_quality    validation
    ${market_data}=    Fetch All Market Data
    FOR    ${tick}    IN    @{market_data}
        ${price}=    Get From Dictionary    ${tick}    price
        ${bid}=    Get From Dictionary    ${tick}    bid
        ${ask}=    Get From Dictionary    ${tick}    ask
        Should Be True    ${price} > 0    msg=Price must be positive
        Should Be True    ${bid} <= ${price} <= ${ask}    msg=Price outside bid/ask
    END

Data Quality - Volume Consistency
    [Documentation]    Validate volume data consistency
    [Tags]    data_quality    volume
    ${trades}=    Query All Recent Trades
    FOR    ${trade}    IN    @{trades}
        ${quantity}=    Get From Dictionary    ${trade}    quantity
        ${price}=    Get From Dictionary    ${trade}    price
        Should Be True    ${quantity} > 0    msg=Quantity must be positive
        Should Be True    ${price} > 0    msg=Price must be positive
    END

Data Quality - No Duplicate Messages
    [Documentation]    Validate no duplicate messages in stream
    [Tags]    data_quality    duplicates
    ${messages}=    Collect Stream Messages    freedx.market_summary    10
    ${unique_messages}=    Get Dictionary Keys    ${messages}
    ${total_messages}=    Get Dictionary Size    ${messages}
    Should Be Equal    ${total_messages}    ${unique_messages}    msg=Duplicate messages detected

# =============================================================================
# Performance Tests
# =============================================================================

Performance - Gateway Latency
    [Documentation]    Measure gateway response times
    [Tags]    performance    latency    stress
    [Setup]    Connect To Gateway    freedx
    
    ${start}=    Get Current Date    result_format=epoch
    ${data}=    Fetch Market Data    freedx
    ${end}=    Get Current Date    result_format=epoch
    
    ${latency_ms}=    Evaluate    (${end} - ${start}) * 1000
    Log    Response latency: ${latency_ms}ms
    Should Be True    ${latency_ms} < 1000    msg=Latency exceeds 1 second
    
    [Teardown]    Disconnect From Gateway    freedx

Performance - ZMQ Throughput
    [Documentation]    Measure ZMQ message throughput
    [Tags]    performance    throughput    zmq
    [Setup]    Start ZMQ Broker
    Subscribe To Topic    freedx.market_summary
    
    ${start}=    Get Current Date    result_format=epoch
    
    FOR    ${i}    IN RANGE    10000
        Publish Message    freedx.market_summary    {"i":${i}}
    END
    
    ${end}=    Get Current Date    result_format=epoch
    ${duration}=    Evaluate    ${end} - ${start}
    ${throughput}=    Evaluate    10000 / ${duration}
    
    Log    Throughput: ${throughput} msg/s
    Should Be True    ${throughput} > 1000    msg=Throughput below target
    
    [Teardown]    Stop ZMQ Broker

Performance - InfluxDB Write Speed
    [Documentation]    Measure InfluxDB write performance
    [Tags]    performance    storage    influxdb
    [Setup]    Connect To InfluxDB
    
    ${start}=    Get Current Date    result_format=epoch
    
    FOR    ${i}    IN RANGE    1000
        Write Market Tick    EUR_USD    1.0850    1.0849    1.0851    1000000    oanda
    END
    
    ${end}=    Get Current Date    result_format=epoch
    ${duration}=    Evaluate    ${end} - ${start}
    ${write_speed}=    Evaluate    1000 / ${duration}
    
    Log    Write speed: ${write_speed} ticks/s
    Should Be True    ${write_speed} > 500    msg=Write speed below target
    
    [Teardown]    Disconnect From InfluxDB

*** Keywords ***

Initialize Test Environment
    [Documentation]    Setup test environment
    Log    Initializing test environment
    Verify InfluxDB Connectivity
    Verify ZMQ Broker Availability
    Load Gateway Configuration

Cleanup Test Environment
    [Documentation]    Teardown test environment
    Log    Cleaning up test environment
    Close All Connections
    Stop All Streams

Verify InfluxDB Connectivity
    [Documentation]    Check InfluxDB is accessible
    ${status}=    Get InfluxDB Health
    Should Be Equal    ${status}    ok

Verify ZMQ Broker Availability
    [Documentation]    Check ZMQ broker is running
    ${available}=    Check ZMQ Broker
    Should Be True    ${available}

Load Gateway Configuration
    [Documentation]    Load gateway settings
    ${config}=    Get File    ${GATEWAY_CONFIG}
    Should Not Be Empty    ${config}
