*** Settings ***
Documentation     Regression test suite for FreeDX Exchange Market Summary API
...               Endpoint: https://api.exchange.freedx.com/spot/api/v3.2/market_summary
...               Validates all public trading pairs have non-empty price and volume data

Library           RequestsLibrary
Library           Collections
Library           String
Library           DateTime
Library           JSONLibrary

*** Variables ***
${BASE_URL}                          https://api.exchange.freedx.com
${ENDPOINT}                          /spot/api/v3.2/market_summary
${TIMEOUT}                           10


*** Test Cases ***

# Public Endpoint Validation Tests

Public Endpoint - Market Summary Returns 200
    [Documentation]    Verify public market summary endpoint is accessible
    [Tags]    public    positive    smoke
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    Should Be Equal As Integers    ${response.status_code}    200
    Delete All Sessions

Public Endpoint - Response Is Valid JSON Array
    [Documentation]    Validate response contains valid JSON array with trading pairs
    [Tags]    public    positive    data-structure
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    Should Be True    isinstance(${data}, list)    msg=Response must be a JSON array
    Should Not Be Empty    ${data}    msg=Response must contain trading pairs
    Delete All Sessions

Public Endpoint - Response Body Is Not Empty
    [Documentation]    Verify response body contains meaningful data
    [Tags]    public    positive    data-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    Should Not Be Empty    ${response.text}
    ${length}=    Get Length    ${response.text}
    Should Be True    ${length} > 100    msg=Response should contain substantial data
    Delete All Sessions

Public Endpoint - Content Type Is JSON
    [Documentation]    Verify response content type is JSON
    [Tags]    public    positive    headers
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${content_type}=    Get From Dictionary    ${response.headers}    Content-Type
    Should Contain    ${content_type}    application/json
    Delete All Sessions

# Public Trading Pairs - Price Validation

Public Pairs - All Have Non-Empty Last Price
    [Documentation]    Iterate over all public trading pairs and validate last price is populated
    [Tags]    public    instruments    price-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${last_price}=    Get From Dictionary    ${pair}    last
        
        # Validate last price is not empty/null
        ${is_empty}=    Evaluate    ${last_price} == None or ${last_price} == '' or ${last_price} == 'None' or ${last_price} == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty last price: ${invalid_pairs}
    Delete All Sessions

Public Pairs - All Have Non-Empty Volume
    [Documentation]    Iterate over all public trading pairs and validate volume is populated
    [Tags]    public    instruments    volume-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${volume}=    Get From Dictionary    ${pair}    volume
        
        # Validate volume is not empty/null
        ${is_empty}=    Evaluate    ${volume} == None or ${volume} == '' or ${volume} == 'None' or ${volume} == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty volume: ${invalid_pairs}
    Delete All Sessions

Public Pairs - All Have Non-Empty Ask Prices
    [Documentation]    Validate lowestAsk is populated for all trading pairs
    [Tags]    public    instruments    orderbook-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${ask}=    Get From Dictionary    ${pair}    lowestAsk

        ${is_empty}=    Evaluate    ${ask} == None or ${ask} == '' or ${ask} == 'None' or ${ask} == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty ask price: ${invalid_pairs}
    Delete All Sessions

Public Pairs - All Have Non-Empty Bid Prices
    [Documentation]    Validate highestBid is populated for all trading pairs
    [Tags]    public    instruments    orderbook-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${bid}=    Get From Dictionary    ${pair}    highestBid

        ${is_empty}=    Evaluate    ${bid} == None or ${bid} == '' or ${bid} == 'None' or ${bid} == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty bid price: ${invalid_pairs}
    Delete All Sessions

# Public Trading Pairs - 24h Price Range Validation

Public Pairs - All Have 24Hr High Prices
    [Documentation]    Validate high24Hr is populated for all trading pairs
    [Tags]    public    instruments    price-range-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${high}=    Get From Dictionary    ${pair}    high24Hr

        ${is_empty}=    Evaluate    ${high} == None or ${high} == '' or ${high} == 'None' or ${high} == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty 24hr high: ${invalid_pairs}
    Delete All Sessions

Public Pairs - All Have 24Hr Low Prices
    [Documentation]    Validate low24Hr is populated for all trading pairs
    [Tags]    public    instruments    price-range-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${low}=    Get From Dictionary    ${pair}    low24Hr

        ${is_empty}=    Evaluate    ${low} == None or ${low} == '' or ${low} == 'None' or ${low} == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty 24hr low: ${invalid_pairs}
    Delete All Sessions

# Public Trading Pairs - Metadata Validation

Public Pairs - All Have Base Currency
    [Documentation]    Validate base currency is populated for all pairs
    [Tags]    public    instruments    metadata-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${base}=    Get From Dictionary    ${pair}    base

        ${is_empty}=    Evaluate    '${base}' == None or '${base}' == '' or '${base}' == 'None' or '${base}' == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty base currency: ${invalid_pairs}
    Delete All Sessions

Public Pairs - All Have Quote Currency
    [Documentation]    Validate quote currency is populated for all pairs
    [Tags]    public    instruments    metadata-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${quote}=    Get From Dictionary    ${pair}    quote
        
        ${is_empty}=    Evaluate    '${quote}' == None or '${quote}' == '' or '${quote}' == 'None' or '${quote}' == 'null'
        Run Keyword If    ${is_empty}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with empty quote currency: ${invalid_pairs}
    Delete All Sessions

Public Pairs - All Have Active Status
    [Documentation]    Validate active status is defined for all pairs
    [Tags]    public    instruments    metadata-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${active}=    Get From Dictionary    ${pair}    active
        
        ${is_none}=    Evaluate    ${active} is None
        Run Keyword If    ${is_none}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with undefined active status: ${invalid_pairs}
    Delete All Sessions

# Public Trading Pairs - Data Consistency Validation

Public Pairs - All Prices Are Non-Negative
    [Documentation]    Validate all price values (last, bid, ask, high, low) are >= 0
    [Tags]    public    instruments    data-consistency
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${last}=    Get From Dictionary    ${pair}    last
        ${bid}=    Get From Dictionary    ${pair}    highestBid
        ${ask}=    Get From Dictionary    ${pair}    lowestAsk
        ${high}=    Get From Dictionary    ${pair}    high24Hr
        ${low}=    Get From Dictionary    ${pair}    low24Hr
        
        # Check for negative values
        ${has_negative}=    Evaluate    ${last} < 0 or ${bid} < 0 or ${ask} < 0 or ${high} < 0 or ${low} < 0
        Run Keyword If    ${has_negative}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with negative prices: ${invalid_pairs}
    Delete All Sessions

Public Pairs - High Price Always Greater Than Low Price
    [Documentation]    Validate high24Hr >= low24Hr for all pairs
    [Tags]    public    instruments    data-consistency
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${high}=    Get From Dictionary    ${pair}    high24Hr
        ${low}=    Get From Dictionary    ${pair}    low24Hr
        
        ${is_invalid}=    Evaluate    ${high} < ${low}
        Run Keyword If    ${is_invalid}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with invalid price ranges (high < low): ${invalid_pairs}
    Delete All Sessions

Public Pairs - Last Price Within 24Hr Range
    [Documentation]    Validate last price is between high24Hr and low24Hr
    [Tags]    public    instruments    data-consistency
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${last}=    Get From Dictionary    ${pair}    last
        ${high}=    Get From Dictionary    ${pair}    high24Hr
        ${low}=    Get From Dictionary    ${pair}    low24Hr
        
        ${is_outside_range}=    Evaluate    ${last} < ${low} or ${last} > ${high}
        Run Keyword If    ${is_outside_range}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Not Be Empty    ${invalid_pairs}    msg=Pairs with last price outside 24hr range: ${invalid_pairs}
    Delete All Sessions

Public Pairs - Ask Price Always Higher Than Bid Price
    [Documentation]    Validate lowestAsk >= highestBid for all pairs
    [Tags]    public    instruments    data-consistency
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${ask}=    Get From Dictionary    ${pair}    lowestAsk
        ${bid}=    Get From Dictionary    ${pair}    highestBid
        
        ${is_invalid}=    Evaluate    ${ask} < ${bid}
        Run Keyword If    ${is_invalid}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with ask < bid spread: ${invalid_pairs}
    Delete All Sessions

# Public Pairs - Volume Validation

Public Pairs - All Volumes Are Non-Negative
    [Documentation]    Validate volume is >= 0 for all pairs
    [Tags]    public    instruments    volume-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${volume}=    Get From Dictionary    ${pair}    volume
        
        ${is_negative}=    Evaluate    ${volume} < 0
        Run Keyword If    ${is_negative}    Append To List    ${invalid_pairs}    ${symbol}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with negative volume: ${invalid_pairs}
    Delete All Sessions

# Public Pairs - Percentage Change Validation

Public Pairs - Percentage Change Is In Valid Range
    [Documentation]    Validate percentageChange is between -100 and +1000 for all pairs
    [Tags]    public    instruments    change-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${invalid_pairs}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${change}=    Get From Dictionary    ${pair}    percentageChange
        
        ${is_out_of_range}=    Evaluate    ${change} < -100 or ${change} > 1000
        Run Keyword If    ${is_out_of_range}    Append To List    ${invalid_pairs}    ${symbol}:${change}
    END
    
    Should Be Empty    ${invalid_pairs}    msg=Pairs with out-of-range percentage change: ${invalid_pairs}
    Delete All Sessions

# Public Response Completeness Tests

Public Endpoint - No Duplicate Trading Pairs
    [Documentation]    Verify response contains no duplicate symbols
    [Tags]    public    data-integrity
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${symbols}=    Create List
    ${duplicates}=    Create List
    
    FOR    ${pair}    IN    @{data}
        ${symbol}=    Get From Dictionary    ${pair}    symbol
        ${already_exists}=    Evaluate    '${symbol}' in ${symbols}
        Run Keyword If    ${already_exists}    Append To List    ${duplicates}    ${symbol}
        Append To List    ${symbols}    ${symbol}
    END
    
    Should Be Empty    ${duplicates}    msg=Duplicate symbols found: ${duplicates}
    Delete All Sessions

Public Endpoint - Response Is Complete And Not Truncated
    [Documentation]    Validate response JSON is properly closed (ends with ']')
    [Tags]    public    data-integrity
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${last_char}=    Get Substring    ${response.text}    -1
    Should Be Equal    ${last_char}    ]    msg=Response JSON not properly closed
    Delete All Sessions

Public Endpoint - Minimum Trading Pairs Expected
    [Documentation]    Verify public endpoint returns minimum expected number of trading pairs
    [Tags]    public    data-completeness
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    ${pair_count}=    Evaluate    len(${data})
    Should Be True    ${pair_count} >= 600    msg=Expected at least 600 trading pairs, got ${pair_count}
    Delete All Sessions




*** Keywords ***

Should Be Number
    [Arguments]    ${value}
    [Documentation]    Custom keyword to verify value is numeric
    Should Be True    isinstance(${value}, (int, float))    msg=${value} is not a number
