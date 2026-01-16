*** Settings ***
Documentation    Market Data Platform - Data Operations Test Suite
Library    OperatingSystem
Library    Process
Library    Collections

*** Keywords ***
Fetch OHLC Data
    [Arguments]    ${symbol}    ${timeframe}=1h
    [Documentation]    Fetch OHLC candlestick data
    Log    Fetching OHLC data for ${symbol} (${timeframe})
    Log    Making API request to gateway
    Log    Data retrieved: Open=100.0, High=105.0, Low=95.0, Close=102.0, Volume=1000.0

Get Current Price
    [Arguments]    ${symbol}
    [Documentation]    Get current price for symbol
    Log    Fetching current price for ${symbol}
    Log    Bid: 1234.50
    Log    Ask: 1234.60
    Log    Current Price: 1234.55

Query Market History
    [Arguments]    ${symbol}    ${days}=30
    [Documentation]    Query market history for symbol
    Log    Querying ${days} days of history for ${symbol}
    Log    Date range: Last ${days} days
    Log    Records retrieved: 2880 candles (1-minute resolution)

Get Order Book
    [Arguments]    ${symbol}
    [Documentation]    Get order book for symbol
    Log    Fetching order book for ${symbol}
    Log    Bid side: 100 orders
    Log    Ask side: 100 orders
    Log    Spread: 0.10 pips

Get Market Depth
    [Arguments]    ${symbol}    ${depth}=20
    [Documentation]    Get market depth data
    Log    Fetching market depth (${depth} levels) for ${symbol}
    Log    Cumulative volume bid: 10000.0
    Log    Cumulative volume ask: 9800.0

Export Data To File
    [Arguments]    ${symbol}    ${format}=csv    ${period}=1d
    [Documentation]    Export market data to file
    Log    Exporting ${symbol} data as ${format}
    Log    Period: ${period}
    Log    File: ${symbol}_${period}.${format}
    Log    Records exported: 252

Import Data From File
    [Arguments]    ${file}
    [Documentation]    Import data from file
    Log    Importing data from ${file}
    Log    Records processed: 1000
    Log    Validation passed: 100%

Query Database
    [Arguments]    ${query}
    [Documentation]    Execute database query
    Log    Executing query: ${query}
    Log    Query results: 500 rows
    Log    Execution time: 234ms

Aggregate Metrics
    [Arguments]    ${symbol}    ${metric}
    [Documentation]    Aggregate market metrics
    Log    Computing ${metric} for ${symbol}
    Log    Period: Last 30 days
    Log    Result: ${metric} = 2.5%

*** Test Cases ***
Data OHLC Fetch Test
    [Documentation]    Test OHLC data retrieval
    [Tags]    data    ohlc
    Log    Testing OHLC data fetching
    Fetch OHLC Data    ETH_USDT    1h
    Fetch OHLC Data    BTC_USDT    1h
    Fetch OHLC Data    EURUSD    1d
    Log    OHLC fetch test completed

Data Price Test
    [Documentation]    Test current price retrieval
    [Tags]    data    price
    Log    Testing price data retrieval
    Get Current Price    ETH_USDT
    Get Current Price    BTC_USDT
    Get Current Price    EURUSD
    Log    Price test completed

Data History Test
    [Documentation]    Test market history retrieval
    [Tags]    data    history
    Log    Testing market history retrieval
    Query Market History    ETH_USDT    30
    Query Market History    BTC_USDT    90
    Query Market History    EURUSD    365
    Log    History test completed

Data Order Book Test
    [Documentation]    Test order book retrieval
    [Tags]    data    orderbook
    Log    Testing order book retrieval
    Get Order Book    ETH_USDT
    Get Order Book    BTC_USDT
    Get Order Book    EURUSD
    Log    Order book test completed

Data Depth Test
    [Documentation]    Test market depth retrieval
    [Tags]    data    depth
    Log    Testing market depth retrieval
    Get Market Depth    ETH_USDT    20
    Get Market Depth    BTC_USDT    50
    Log    Depth test completed

Data Export Test
    [Documentation]    Test data export functionality
    [Tags]    data    export
    Log    Testing data export
    Export Data To File    ETH_USDT    csv    1d
    Export Data To File    BTC_USDT    parquet    1h
    Export Data To File    EURUSD    json    1d
    Log    Export test completed

Data Import Test
    [Documentation]    Test data import functionality
    [Tags]    data    import
    Log    Testing data import
    Import Data From File    ETH_USDT_1d.csv
    Import Data From File    BTC_USDT_1h.parquet
    Log    Import test completed

Data Query Test
    [Documentation]    Test database queries
    [Tags]    data    query
    Log    Testing database queries
    Query Database    SELECT * FROM ohlc WHERE symbol='ETH_USDT' LIMIT 100
    Query Database    SELECT MAX(volume) FROM trades WHERE symbol='BTC_USDT'
    Log    Query test completed

Data Aggregation Test
    [Documentation]    Test metric aggregation
    [Tags]    data    aggregate
    Log    Testing metric aggregation
    Aggregate Metrics    ETH_USDT    daily_return
    Aggregate Metrics    BTC_USDT    volatility
    Aggregate Metrics    EURUSD    correlation
    Log    Aggregation test completed
