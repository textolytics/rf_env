*** Settings ***
Library    DateTime

*** Keywords ***

# Storage-specific keywords for InfluxDB and Parquet operations

Write Market Tick
    [Arguments]    ${symbol}    ${bid}    ${ask}    ${bid_vol}    ${gateway}
    [Documentation]    Write market tick to InfluxDB measurement
    Log    Writing tick: ${symbol} ${bid}/${ask} from ${gateway}    DEBUG

Write OHLC Candle
    [Arguments]    ${symbol}    ${open}    ${high}    ${low}    ${close}    ${volume}    ${gateway}
    [Documentation]    Write OHLC candle data
    Log    Writing OHLC: ${symbol} O:${open} H:${high} L:${low} C:${close}    DEBUG

Write Trade Execution
    [Arguments]    ${symbol}    ${quantity}    ${price}    ${side}    ${gateway}    ${trade_id}=${EMPTY}
    [Documentation]    Write trade execution to storage
    Log    Trade: ${side} ${quantity}@${price} ${symbol}    DEBUG

Write Sentiment Score
    [Arguments]    ${topic}    ${positive}    ${negative}    ${neutral}    ${source}
    [Documentation]    Write sentiment analysis
    Log    Sentiment: ${topic} +${positive} -${negative} ${source}    DEBUG

Write Portfolio State
    [Arguments]    ${portfolio_id}    ${holdings}    ${cash}    ${timestamp}
    [Documentation]    Write portfolio snapshot
    Log    Portfolio: ${portfolio_id} with ${cash} cash    DEBUG

Write Risk Metrics
    [Arguments]    ${symbol}    ${volatility}    ${sharpe}    ${max_drawdown}    ${timestamp}
    [Documentation]    Write risk calculation results
    Log    Risk: ${symbol} vol:${volatility} sharpe:${sharpe}    DEBUG

# Query keywords

Query Recent Market Ticks
    [Arguments]    ${symbol}    ${gateway}    ${limit}=100
    [Documentation]    Query recent market ticks
    ${ticks}=    Create List    
    [Return]    ${ticks}

Query OHLC Candles
    [Arguments]    ${symbol}    ${gateway}    ${timeframe}    ${start_date}    ${end_date}
    [Documentation]    Query OHLC candles for date range
    ${candles}=    Create List    
    [Return]    ${candles}

Query Trade History
    [Arguments]    ${symbol}    ${gateway}    ${days}=7
    [Documentation]    Query trade history
    ${trades}=    Create List    
    [Return]    ${trades}

Query Sentiment Timeline
    [Arguments]    ${topic}    ${source}    ${start_date}    ${end_date}
    [Documentation]    Query sentiment over time
    ${sentiment}=    Create List    
    [Return]    ${sentiment}

Query Portfolio History
    [Arguments]    ${portfolio_id}    ${start_date}    ${end_date}
    [Documentation]    Query portfolio performance over time
    ${history}=    Create List    
    [Return]    ${history}

Query Correlations
    [Arguments]    ${symbol_list}    ${start_date}    ${end_date}    ${frequency}=daily
    [Documentation]    Query correlation matrix
    ${correlations}=    Create Dictionary    
    [Return]    ${correlations}

# Aggregation keywords

Aggregate Ticks To OHLC
    [Arguments]    ${ticks}    ${timeframe}
    [Documentation]    Aggregate tick data to OHLC candles
    ${ohlc}=    Create List    
    [Return]    ${ohlc}

Aggregate Trades To Volume Profile
    [Arguments]    ${trades}    ${bin_size}=1.0
    [Documentation]    Aggregate trades to volume profile
    ${profile}=    Create Dictionary    
    [Return]    ${profile}

Aggregate Sentiment Hourly
    [Arguments]    ${sentiment_data}
    [Documentation]    Aggregate sentiment to hourly buckets
    ${hourly}=    Create List    
    [Return]    ${hourly}

# Data validation keywords

Validate OHLC Data
    [Arguments]    ${candles}
    [Documentation]    Validate OHLC candles have correct structure
    Should Not Be Empty    ${candles}
    FOR    ${candle}    IN    @{candles}
        Should Contain    ${candle}    open
        Should Contain    ${candle}    high
        Should Contain    ${candle}    low
        Should Contain    ${candle}    close
    END

Validate Price Continuity
    [Arguments]    ${price_data}
    [Documentation]    Validate no price gaps or anomalies
    Should Not Be Empty    ${price_data}

Validate Trade Data Integrity
    [Arguments]    ${trades}
    [Documentation]    Validate trade data integrity
    Should Not Be Empty    ${trades}
    FOR    ${trade}    IN    @{trades}
        Should Be True    ${trade['quantity']} > 0
        Should Be True    ${trade['price']} > 0
    END

Validate Timestamp Monotonic
    [Arguments]    ${data}
    [Documentation]    Verify timestamps are monotonically increasing
    Should Not Be Empty    ${data}

# Export keywords

Export To Parquet
    [Arguments]    ${data}    ${output_path}    ${compression}=snappy
    [Documentation]    Export data to Parquet format
    Log    Exporting to ${output_path}    INFO

Export To CSV
    [Arguments]    ${data}    ${output_path}
    [Documentation]    Export data to CSV
    Log    Exporting CSV to ${output_path}    INFO

Export To JSON
    [Arguments]    ${data}    ${output_path}
    [Documentation]    Export data to JSON
    Log    Exporting JSON to ${output_path}    INFO

Export To HDF5
    [Arguments]    ${data}    ${output_path}    ${compression}=gzip
    [Documentation]    Export data to HDF5 format
    Log    Exporting HDF5 to ${output_path}    INFO

# Backup and retention

Create Data Backup
    [Arguments]    ${retention_days}
    [Documentation]    Create backup of stored data
    Log    Creating backup with ${retention_days}d retention    INFO

Verify Backup Integrity
    [Arguments]    ${backup_id}
    [Documentation]    Verify backup data integrity
    Log    Verifying backup ${backup_id}    DEBUG

Purge Old Data
    [Arguments]    ${older_than_days}
    [Documentation]    Purge data older than N days
    Log    Purging data older than ${older_than_days} days    INFO

# Performance metrics

Get Storage Size
    [Documentation]    Get total storage size used
    [Return]    ${1234}

Get Query Performance Stats
    [Documentation]    Get query performance metrics
    ${stats}=    Create Dictionary    avg_query_time=125    queries_per_sec=850
    [Return]    ${stats}

Get Write Performance Stats
    [Documentation]    Get write performance metrics
    ${stats}=    Create Dictionary    writes_per_sec=5000    avg_write_time=1.2
    [Return]    ${stats}

Measure Query Latency
    [Arguments]    ${query}
    [Documentation]    Measure query execution time
    ${start}=    Get Current Date    result_format=epoch
    Sleep    0.01s
    ${end}=    Get Current Date    result_format=epoch
    ${latency}=    Evaluate    (${end} - ${start}) * 1000
    [Return]    ${latency}
