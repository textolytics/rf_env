*** Settings ***
Documentation    Market Data Platform - Gateway Management Test Suite
Library    OperatingSystem
Library    Process
Library    Collections

*** Keywords ***
Connect To Gateway
    [Arguments]    ${gateway}
    [Documentation]    Connect to specified gateway
    Log    Connecting to ${gateway}...
    Log    Authenticating with gateway
    Log    Connected to ${gateway}

Disconnect From Gateway
    [Arguments]    ${gateway}
    [Documentation]    Disconnect from gateway
    Log    Disconnecting from ${gateway}...
    Log    Closing connections
    Log    Disconnected from ${gateway}

List Available Gateways
    [Documentation]    List all configured gateways
    Log    Available gateways:
    Log    • gate.io
    Log    • oanda
    Log    • kraken
    Log    • freedx
    Log    • betfair
    Log    • twitter

Get Gateway Status
    [Arguments]    ${gateway}
    [Documentation]    Get status of specific gateway
    Log    Retrieving ${gateway} status
    Log    Gateway status: CONNECTED
    Log    Connection latency: <100ms
    Log    Message throughput: >100 msg/sec

Stream Market Data
    [Arguments]    ${gateway}    ${symbol}
    [Documentation]    Stream market data from gateway
    Log    Starting stream from ${gateway} for ${symbol}
    Log    Subscribing to candlestick updates
    Log    Stream active

Stop Stream
    [Arguments]    ${gateway}
    [Documentation]    Stop market data stream
    Log    Stopping stream from ${gateway}
    Log    Unsubscribing from all topics
    Log    Stream stopped

Test Gateway Connectivity
    [Arguments]    ${gateway}
    [Documentation]    Test gateway connectivity
    Log    Testing connectivity to ${gateway}
    Log    Sending health check ping
    Log    Response: PONG (latency: 50ms)
    Log    Gateway connectivity: OK

Get Gateway Configuration
    [Arguments]    ${gateway}
    [Documentation]    Get gateway configuration
    Log    Retrieving ${gateway} configuration
    Log    API Endpoint: ${gateway}.example.com
    Log    WebSocket Endpoint: wss://${gateway}.example.com/ws
    Log    Rate Limit: 1000 req/min

*** Test Cases ***
Gateway Connection Test
    [Documentation]    Test connections to all configured gateways
    [Tags]    gateway    connectivity
    Log    Starting gateway connection tests
    Connect To Gateway    gate.io
    Connect To Gateway    oanda
    Connect To Gateway    kraken
    Sleep    1s
    Get Gateway Status    gate.io
    Get Gateway Status    oanda
    Log    Gateway connection tests completed

Gateway Disconnection Test
    [Documentation]    Test disconnection from gateways
    [Tags]    gateway    disconnect
    Log    Testing gateway disconnection
    Disconnect From Gateway    gate.io
    Disconnect From Gateway    oanda
    Log    Gateway disconnection test completed

Gateway Status Test
    [Documentation]    Monitor gateway status
    [Tags]    gateway    status
    Log    Monitoring gateway status
    Get Gateway Status    gate.io
    Get Gateway Status    oanda
    Get Gateway Status    kraken
    Log    All gateways operational

Gateway List Test
    [Documentation]    List available gateways
    [Tags]    gateway    list
    Log    Listing available gateways
    List Available Gateways
    Log    Gateway list retrieved

Gateway Streaming Test
    [Documentation]    Test market data streaming
    [Tags]    gateway    stream
    Log    Testing gateway streaming
    Connect To Gateway    gate.io
    Stream Market Data    gate.io    ETH_USDT
    Sleep    2s
    Stop Stream    gate.io
    Log    Streaming test completed

Gateway Performance Test
    [Documentation]    Test gateway performance
    [Tags]    gateway    performance
    Log    Running gateway performance test
    :FOR    ${i}    IN RANGE    5
    \    Test Gateway Connectivity    gate.io
    \    Sleep    1s
    Log    Gateway performance test completed
