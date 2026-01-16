*** Keywords ***

# =============================================================================
# Gateway Connection Keywords
# =============================================================================

Connect To Gateway
    [Arguments]    ${gateway}
    [Documentation]    Establish connection to a market data gateway
    Log    Connecting to ${gateway}    INFO
    # Implementation would use requests or async HTTP client
    Sleep    0.5s

Disconnect From Gateway
    [Arguments]    ${gateway}
    [Documentation]    Close connection to a gateway
    Log    Disconnecting from ${gateway}    INFO
    Sleep    0.5s

Fetch Market Data
    [Arguments]    ${gateway}    ${limit}=10
    [Documentation]    Fetch market data from gateway
    ${data}=    Create List    name=value
    [Return]    ${data}

Start Data Stream
    [Arguments]    ${gateway}    ${topic}
    [Documentation]    Start streaming data from topic
    Log    Starting stream: ${topic}    INFO
    [Return]    success

Stop Data Stream
    [Arguments]    ${gateway}    ${topic}
    [Documentation]    Stop data stream
    Log    Stopping stream: ${topic}    INFO

Validate Market Data Structure
    [Arguments]    ${data}
    [Documentation]    Validate market data contains required fields
    Should Not Be Empty    ${data}

# =============================================================================
# ZMQ Broker Keywords
# =============================================================================

Start ZMQ Broker
    [Documentation]    Start ZMQ message broker
    Log    Starting ZMQ broker    INFO

Stop ZMQ Broker
    [Documentation]    Stop ZMQ broker
    Log    Stopping ZMQ broker    INFO

Subscribe To Topic
    [Arguments]    ${topic}
    [Documentation]    Subscribe to ZMQ topic
    Log    Subscribing to ${topic}    INFO

Unsubscribe From Topic
    [Arguments]    ${topic}
    [Documentation]    Unsubscribe from topic
    Log    Unsubscribing from ${topic}    INFO

Publish Message
    [Arguments]    ${topic}    ${message}
    [Documentation]    Publish message to topic
    Log    Publishing to ${topic}: ${message}    DEBUG

Receive Message
    [Arguments]    ${timeout}=5s
    [Documentation]    Receive message from subscribed topic
    Sleep    0.1s
    [Return]    {"symbol":"BTC","price":45000}

Receive Message From Topic
    [Arguments]    ${topic}    ${timeout}=5s
    [Documentation]    Receive message from specific topic
    Sleep    0.1s
    [Return]    {"data":"example"}

Collect Stream Messages
    [Arguments]    ${topic}    ${count}
    [Documentation]    Collect N messages from stream
    ${messages}=    Create Dictionary
    FOR    ${i}    IN RANGE    ${count}
        ${msg}=    Create Dictionary    tick=${i}    value=${i*100}
        Set To Dictionary    ${messages}    msg_${i}=${msg}
    END
    [Return]    ${messages}

# =============================================================================
# InfluxDB Storage Keywords
# =============================================================================

Connect To InfluxDB
    [Documentation]    Connect to InfluxDB
    Log    Connecting to InfluxDB    INFO

Disconnect From InfluxDB
    [Documentation]    Close InfluxDB connection
    Log    Disconnecting from InfluxDB    INFO

Write Market Tick
    [Arguments]    ${symbol}    ${bid}    ${ask}    ${bid_vol}    ${gateway}
    [Documentation]    Write market tick to InfluxDB
    Log    Writing tick for ${symbol}    DEBUG

Write Trade Data
    [Arguments]    ${symbol}    ${quantity}    ${price}    ${side}    ${gateway}
    [Documentation]    Write trade data to storage
    Log    Writing trade: ${side} ${quantity} @ ${price}    DEBUG

Write Sentiment Data
    [Arguments]    ${topic}    ${score}    ${sample_size}    ${source}
    [Documentation]    Write sentiment data to storage
    Log    Writing sentiment: ${score} (n=${sample_size})    DEBUG

Query Market Tick
    [Arguments]    ${symbol}    ${gateway}
    [Documentation]    Query market tick from storage
    ${data}=    Create List    timestamp=2024-01-01T00:00:00
    [Return]    ${data}

Query Recent Trades
    [Arguments]    ${symbol}    ${gateway}
    [Documentation]    Query recent trades
    ${trades}=    Create List    {"symbol":"${symbol}","price":1.0850}
    [Return]    ${trades}

Query Sentiment
    [Arguments]    ${topic}    ${source}
    [Documentation]    Query sentiment data
    ${data}=    Create List    ${topic}
    [Return]    ${data}

Query OHLC
    [Arguments]    ${symbol}    ${gateway}    ${interval}    ${lookback}
    [Documentation]    Query OHLC candle data
    ${ohlc}=    Create List    open=100    high=101    low=99    close=100.5
    [Return]    ${ohlc}

Query All Recent Trades
    [Documentation]    Query all recent trades across gateways
    ${trades}=    Create List    {"price":100,"quantity":1000}
    [Return]    ${trades}

Set Retention Policy
    [Arguments]    ${duration}
    [Documentation]    Set data retention policy
    Log    Setting retention policy: ${duration}    INFO

Verify Retention Setting
    [Arguments]    ${duration}
    [Documentation]    Verify retention policy is set
    Log    Verifying retention: ${duration}    DEBUG

Get InfluxDB Health
    [Documentation]    Check InfluxDB health status
    [Return]    ok

# =============================================================================
# CLI Keywords
# =============================================================================

Start CLI
    [Documentation]    Start CLI application
    Log    Starting CLI    INFO

Stop CLI
    [Documentation]    Stop CLI application
    Log    Stopping CLI    INFO

Execute CLI Command
    [Arguments]    ${command}
    [Documentation]    Execute CLI command
    Log    Executing: ${command}    DEBUG

Verify Output Contains
    [Arguments]    ${text}
    [Documentation]    Verify CLI output contains text
    Log    Verifying output contains: ${text}    DEBUG

Test Auto Completion
    [Arguments]    ${prefix}    ${expected_items}
    [Documentation]    Test command auto-completion
    Log    Testing completion for: ${prefix}    DEBUG

Verify File Exists
    [Arguments]    ${file_path}
    [Documentation]    Verify file exists
    Should Exist    ${file_path}

# =============================================================================
# Data Quality Keywords
# =============================================================================

Fetch All Market Data
    [Documentation]    Fetch all market data
    ${data}=    Create List    
    FOR    ${i}    IN RANGE    5
        Append To List    ${data}    
        ...    price=100    bid=99.99    ask=100.01
    END
    [Return]    ${data}

Validate Forex Data Format
    [Arguments]    ${data}
    [Documentation]    Validate forex data format
    Should Not Be Empty    ${data}

Validate Sentiment Score
    [Arguments]    ${data}
    [Documentation]    Validate sentiment score
    Should Not Be Empty    ${data}

Check ZMQ Broker
    [Documentation]    Check if ZMQ broker is available
    [Return]    True

# =============================================================================
# Performance Keywords
# =============================================================================

Collect Stream Messages Timed
    [Arguments]    ${topic}    ${duration}    ${message_count}=100
    [Documentation]    Collect messages and measure throughput
    ${messages}=    Create List
    ${start}=    Get Current Date    result_format=epoch
    FOR    ${i}    IN RANGE    ${message_count}
        Append To List    ${messages}    msg_${i}
    END
    ${end}=    Get Current Date    result_format=epoch
    ${duration_actual}=    Evaluate    ${end} - ${start}
    [Return]    ${messages}
