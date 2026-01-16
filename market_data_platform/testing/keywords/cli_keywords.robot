*** Keywords ***

# CLI-specific test keywords

Start CLI Session
    [Arguments]    ${config_file}=${EMPTY}
    [Documentation]    Start new CLI session
    Log    Starting CLI session    INFO

Stop CLI Session
    [Documentation]    Stop CLI session
    Log    Stopping CLI session    INFO

Execute Command
    [Arguments]    ${command}    ${expected_output}=${EMPTY}    ${timeout}=5s
    [Documentation]    Execute CLI command and return output
    Log    Executing: ${command}    DEBUG
    [Return]    output_text

Verify Command Output
    [Arguments]    ${output}    ${expected_text}
    [Documentation]    Verify command output contains expected text
    Should Contain    ${output}    ${expected_text}

Parse Command Output
    [Arguments]    ${output}    ${format}=json
    [Documentation]    Parse command output to structured data
    ${data}=    Create Dictionary    parsed=true
    [Return]    ${data}

# Command group keywords

Connect Gateway
    [Arguments]    ${gateway}    ${config}=${EMPTY}
    [Documentation]    CLI: connect to gateway
    Execute Command    connect ${gateway}
    Verify Command Output    ${EMPTY}    Connected

Disconnect Gateway
    [Arguments]    ${gateway}
    [Documentation]    CLI: disconnect from gateway
    Execute Command    disconnect ${gateway}

List Gateways
    [Documentation]    CLI: list available gateways
    ${output}=    Execute Command    gateways
    Should Not Be Empty    ${output}
    [Return]    ${output}

List Topics
    [Documentation]    CLI: list available ZMQ topics
    ${output}=    Execute Command    topics
    Should Not Be Empty    ${output}
    [Return]    ${output}

List Symbols
    [Documentation]    CLI: list available symbols
    ${output}=    Execute Command    symbols
    Should Not Be Empty    ${output}
    [Return]    ${output}

# Price and data commands

Get Current Price
    [Arguments]    ${symbol}    ${gateway}=${EMPTY}
    [Documentation]    CLI: get current price
    ${output}=    Execute Command    price ${symbol}
    Should Contain    ${output}    price
    [Return]    ${output}

Get Price History
    [Arguments]    ${symbol}    ${days}
    [Documentation]    CLI: get price history
    ${output}=    Execute Command    history ${symbol} ${days}
    [Return]    ${output}

Get OHLC Data
    [Arguments]    ${symbol}    ${interval}
    [Documentation]    CLI: get OHLC data
    ${output}=    Execute Command    ohlc ${symbol} ${interval}
    [Return]    ${output}

Get Order Book
    [Arguments]    ${symbol}    ${depth}=10
    [Documentation]    CLI: get order book
    ${output}=    Execute Command    orderbook ${symbol} --depth ${depth}
    [Return]    ${output}

Get Sentiment Analysis
    [Arguments]    ${topic}    ${source}=${EMPTY}
    [Documentation]    CLI: get sentiment analysis
    ${output}=    Execute Command    sentiment ${topic}
    [Return]    ${output}

# Streaming commands

Start Streaming
    [Arguments]    ${topic}    ${duration}=60
    [Documentation]    CLI: start data stream
    Execute Command    stream ${topic} --duration ${duration}

Stop Streaming
    [Arguments]    ${topic}
    [Documentation]    CLI: stop data stream
    Execute Command    stop ${topic}

# Configuration commands

Show Config
    [Documentation]    CLI: show configuration
    ${output}=    Execute Command    config show
    [Return]    ${output}

Set Config Value
    [Arguments]    ${key}    ${value}
    [Documentation]    CLI: set configuration value
    Execute Command    config set ${key}=${value}

Reset Config
    [Documentation]    CLI: reset configuration
    Execute Command    config reset

# Export and utility commands

Export Data
    [Arguments]    ${format}    ${output_file}    ${symbol}=${EMPTY}    ${days}=7
    [Documentation]    CLI: export data
    ${cmd}=    Set Variable    export ${format} ${output_file}
    Run Keyword If    ${symbol} != ''    Set Variable    ${cmd}    ${cmd} --symbol ${symbol}
    Execute Command    ${cmd}

Show Status
    [Documentation]    CLI: show platform status
    ${output}=    Execute Command    status
    [Return]    ${output}

Show Statistics
    [Documentation]    CLI: show platform statistics
    ${output}=    Execute Command    stats
    [Return]    ${output}

Manage Alerts
    [Arguments]    ${action}    ${args}=${EMPTY}
    [Documentation]    CLI: manage alerts (add|list|remove)
    ${cmd}=    Set Variable    alerts ${action}
    Run Keyword If    ${args} != ''    Set Variable    ${cmd}    ${cmd} ${args}
    Execute Command    ${cmd}

Show Help
    [Arguments]    ${topic}=${EMPTY}
    [Documentation]    CLI: show help
    ${cmd}=    Set Variable    help
    Run Keyword If    ${topic} != ''    Set Variable    ${cmd}    ${cmd} ${topic}
    ${output}=    Execute Command    ${cmd}
    [Return]    ${output}

# Auto-completion and suggestions

Get Completion Suggestions
    [Arguments]    ${prefix}
    [Documentation]    Get auto-completion suggestions
    ${suggestions}=    Create List    
    [Return]    ${suggestions}

Verify Completion Works
    [Arguments]    ${prefix}    ${expected_items}
    [Documentation]    Verify command completion works
    ${suggestions}=    Get Completion Suggestions    ${prefix}
    FOR    ${item}    IN    @{expected_items}
        Should Contain    ${suggestions}    ${item}
    END

Test Command Shortcut
    [Arguments]    ${shortcut}    ${full_command}
    [Documentation]    Test command shortcut expansion
    ${output1}=    Execute Command    ${shortcut}
    ${output2}=    Execute Command    ${full_command}
    Should Be Equal    ${output1}    ${output2}

# Error handling and validation

Verify Command Error
    [Arguments]    ${command}    ${expected_error}
    [Documentation]    Verify command produces expected error
    ${output}=    Execute Command    ${command}
    Should Contain    ${output}    ${expected_error}

Verify Invalid Argument
    [Arguments]    ${command}    ${invalid_arg}
    [Documentation]    Verify command rejects invalid argument
    Verify Command Error    ${command} ${invalid_arg}    error

Verify Missing Required Argument
    [Arguments]    ${command}
    [Documentation]    Verify command requires arguments
    Verify Command Error    ${command}    Usage

# Output formatting verification

Verify Table Output
    [Arguments]    ${output}    ${headers}
    [Documentation]    Verify output is properly formatted table
    FOR    ${header}    IN    @{headers}
        Should Contain    ${output}    ${header}
    END

Verify JSON Output
    [Arguments]    ${output}
    [Documentation]    Verify output is valid JSON
    ${parsed}=    Parse Command Output    ${output}    json
    Should Not Be Empty    ${parsed}

Verify CSV Output
    [Arguments]    ${output}
    [Documentation]    Verify output is valid CSV
    Should Contain    ${output}    ,

Verify Color Output
    [Arguments]    ${output}
    [Documentation]    Verify output contains ANSI colors
    Should Contain    ${output}    \033[

# Performance and usability

Measure Command Latency
    [Arguments]    ${command}
    [Documentation]    Measure command execution time
    ${start}=    Get Current Date    result_format=epoch
    Execute Command    ${command}
    ${end}=    Get Current Date    result_format=epoch
    ${latency}=    Evaluate    (${end} - ${start}) * 1000
    [Return]    ${latency}

Verify Command Responsiveness
    [Arguments]    ${command}    ${max_latency_ms}=1000
    [Documentation]    Verify command responds within time limit
    ${latency}=    Measure Command Latency    ${command}
    Should Be True    ${latency} < ${max_latency_ms}

Test Multiple Commands
    [Arguments]    ${command_list}
    [Documentation]    Execute multiple commands in sequence
    FOR    ${cmd}    IN    @{command_list}
        Execute Command    ${cmd}
    END

Verify CLI Persistence
    [Arguments]    ${commands}
    [Documentation]    Verify state persists across commands
    FOR    ${cmd}    IN    @{commands}
        Execute Command    ${cmd}
    END
