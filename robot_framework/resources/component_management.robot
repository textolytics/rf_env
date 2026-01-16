*** Settings ***
Documentation    Robot Framework keywords for component management
...              Provides keywords for starting, stopping, and validating
...              service components and connectivity

Library          Process
Library          String
Library          Collections
Library          OperatingSystem
Library          JSON


*** Keywords ***

# ============================================================================
# COMPONENT START/STOP KEYWORDS
# ============================================================================

Start Component
    [Arguments]    ${component}
    [Documentation]    Start a single component by name
    ...    Arguments: component (database, storage, messaging, api, gateway, proxy, etc.)
    [Tags]    component    start    connectivity
    Log    Starting component: ${component}
    ${result}=    Run Process    bash    lib/component_manager.sh    start    ${component}
    ...    cwd=/root/rf_env    shell=True
    Should Be Equal    ${result.rc}    0    Component ${component} failed to start
    Log    Component ${component} started successfully    level=INFO


Start All Components
    [Documentation]    Start all service components in proper order
    [Tags]    component    start    connectivity
    Log    Starting all components...
    ${result}=    Run Process    bash    lib/component_manager.sh    start
    ...    cwd=/root/rf_env    shell=True
    Should Be Equal    ${result.rc}    0    Failed to start all components
    Log    All components started successfully    level=INFO


Start Component Group
    [Arguments]    @{components}
    [Documentation]    Start multiple components
    [Tags]    component    start    connectivity
    Log    Starting component group: ${components}
    FOR    ${component}    IN    @{components}
        Start Component    ${component}
    END


Stop Component
    [Arguments]    ${component}
    [Documentation]    Stop a single component gracefully
    [Tags]    component    stop    connectivity
    Log    Stopping component: ${component}
    ${result}=    Run Process    bash    lib/component_manager.sh    stop    ${component}
    ...    cwd=/root/rf_env    shell=True
    Should Be Equal    ${result.rc}    0    Component ${component} failed to stop
    Log    Component ${component} stopped successfully    level=INFO


Stop All Components
    [Documentation]    Stop all service components in reverse order
    [Tags]    component    stop    connectivity
    Log    Stopping all components...
    ${result}=    Run Process    bash    lib/component_manager.sh    stop
    ...    cwd=/root/rf_env    shell=True
    Should Be Equal    ${result.rc}    0    Failed to stop all components
    Log    All components stopped successfully    level=INFO


Stop Component Group
    [Arguments]    @{components}
    [Documentation]    Stop multiple components
    [Tags]    component    stop    connectivity
    Log    Stopping component group: ${components}
    FOR    ${component}    IN    @{components}
        Stop Component    ${component}
    END


Restart Component
    [Arguments]    ${component}
    [Documentation]    Restart a component (stop then start)
    [Tags]    component    restart    connectivity
    Log    Restarting component: ${component}
    Stop Component    ${component}
    Sleep    2s
    Start Component    ${component}


# ============================================================================
# CONNECTIVITY & HEALTH VALIDATION KEYWORDS
# ============================================================================

Validate Service Connectivity
    [Arguments]    ${service_name}
    [Documentation]    Validate connectivity to a specific service
    ...    Arguments: service_name (database_postgres, api_python, messaging_zmq_publisher, etc.)
    [Tags]    connectivity    validation    health
    Log    Validating connectivity to ${service_name}...
    
    ${cmd}=    Catenate    SEPARATOR=|    
    ...    cd /root/rf_env &&
    ...    python -c "
    ...    import sys
    ...    sys.path.insert(0, '.')
    ...    from market_data_platform.connectivity.validator import ConnectivityValidatorSync
    ...    v = ConnectivityValidatorSync()
    ...    results = v.validate_all()
    ...    health = results.get('${service_name}')
    ...    if health and health.status.value == 'healthy':
    ...        print('HEALTHY')
    ...    else:
    ...        print(f'FAILED: {health.status.value if health else \"NOT FOUND\"}'
    ...        sys.exit(1 if not health or health.status.value != 'healthy' else 0)
    ...    "
    
    ${result}=    Run Process    bash    -c    ${cmd}    shell=True
    Should Be Equal    ${result.rc}    0    Service ${service_name} connectivity validation failed
    Log    ✓ Service ${service_name} connectivity validated    level=INFO


Validate All Services Connectivity
    [Documentation]    Validate connectivity to all services
    [Tags]    connectivity    validation    health
    Log    Validating connectivity to all services...
    
    @{services}=    Create List
    ...    database_postgres
    ...    cache_redis
    ...    storage_influxdb
    ...    monitoring_prometheus
    ...    api_python
    ...    gateway_go
    ...    messaging_zmq_publisher
    
    FOR    ${service}    IN    @{services}
        Validate Service Connectivity    ${service}
    END


Get Service Health Status
    [Arguments]    ${service_name}
    [Documentation]    Get health status for a service
    ...    Returns dictionary with status, response_time, details
    [Tags]    connectivity    validation    health
    
    ${cmd}=    Catenate    SEPARATOR=|
    ...    cd /root/rf_env &&
    ...    python -c "
    ...    import sys, json
    ...    sys.path.insert(0, '.')
    ...    from market_data_platform.connectivity.validator import ConnectivityValidatorSync
    ...    v = ConnectivityValidatorSync()
    ...    results = v.validate_all()
    ...    health = results.get('${service_name}')
    ...    if health:
    ...        print(json.dumps({
    ...            'status': health.status.value,
    ...            'response_time': health.response_time,
    ...            'details': health.details
    ...        }))
    ...    "
    
    ${result}=    Run Process    bash    -c    ${cmd}    shell=True
    Should Not Be Empty    ${result.stdout}    Failed to get health status
    ${health_data}=    Evaluate    json.loads('''${result.stdout}''')    json
    [Return]    ${health_data}


Get Overall Connectivity Status
    [Documentation]    Get overall system connectivity status
    ...    Returns: HEALTHY, DEGRADED, or UNHEALTHY
    [Tags]    connectivity    validation    health
    
    ${cmd}=    Catenate    SEPARATOR=|
    ...    cd /root/rf_env &&
    ...    python -c "
    ...    import sys, json
    ...    sys.path.insert(0, '.')
    ...    from market_data_platform.connectivity.validator import ConnectivityValidatorSync
    ...    v = ConnectivityValidatorSync()
    ...    summary = v.get_summary()
    ...    print(summary['overall_status'])
    ...    "
    
    ${result}=    Run Process    bash    -c    ${cmd}    shell=True
    ${status}=    Get Line    ${result.stdout}    0
    [Return]    ${status}


Assert Service Is Healthy
    [Arguments]    ${service_name}
    [Documentation]    Assert that a service is healthy
    [Tags]    connectivity    assertion    health
    Log    Asserting ${service_name} is healthy...
    
    ${health}=    Get Service Health Status    ${service_name}
    Should Be Equal    ${health}[status]    healthy    Service ${service_name} is not healthy
    Log    ✓ ${service_name} is healthy    level=INFO


Assert All Services Healthy
    [Documentation]    Assert that all services are healthy
    [Tags]    connectivity    assertion    health
    Log    Asserting all services are healthy...
    
    ${status}=    Get Overall Connectivity Status
    Should Be Equal    ${status}    healthy    System is not healthy: ${status}
    Log    ✓ All services are healthy    level=INFO


Wait For Service To Be Ready
    [Arguments]    ${service_name}    ${timeout}=30s
    [Documentation]    Wait for a service to become ready/healthy
    ...    Arguments: service_name, timeout (default: 30s)
    [Tags]    connectivity    wait    health
    
    ${timeout_seconds}=    Convert Time    ${timeout}    result_format=number
    ${start_time}=    Get Time    epoch
    
    WHILE    True
        ${current_time}=    Get Time    epoch
        ${elapsed}=    Evaluate    ${current_time} - ${start_time}
        
        ${status}=    Get Service Health Status    ${service_name}
        Log    Waiting for ${service_name}... Status: ${status}[status] (elapsed: ${elapsed}s)
        
        Exit For Loop If    '${status}[status]' == 'healthy'
        Exit For Loop If    ${elapsed} > ${timeout_seconds}
        
        Sleep    1s
    END
    
    ${status}=    Get Service Health Status    ${service_name}
    Should Be Equal    ${status}[status]    healthy
    ...    Service ${service_name} did not become ready within ${timeout}
    Log    ✓ Service ${service_name} is ready    level=INFO


# ============================================================================
# COMPONENT STATUS KEYWORDS
# ============================================================================

Get Component Status
    [Documentation]    Get current component status (running/stopped)
    [Tags]    component    status
    
    ${result}=    Run Process    bash    lib/component_manager.sh    status
    ...    cwd=/root/rf_env    shell=True
    Log    ${result.stdout}
    [Return]    ${result.stdout}


Assert Component Is Running
    [Arguments]    ${component}
    [Documentation]    Assert that a component is currently running
    [Tags]    component    assertion    status
    
    ${status}=    Get Component Status
    Should Contain    ${status}    ${component}    Component ${component} is not running
    Log    ✓ Component ${component} is running    level=INFO


Assert Component Is Stopped
    [Arguments]    ${component}
    [Documentation]    Assert that a component is stopped
    [Tags]    component    assertion    status
    
    ${status}=    Get Component Status
    Should Not Contain    ${status}    ${component}    Component ${component} is still running
    Log    ✓ Component ${component} is stopped    level=INFO


# ============================================================================
# ZMQ MESSAGING VALIDATION KEYWORDS
# ============================================================================

Validate ZMQ Publisher
    [Documentation]    Validate ZMQ publisher is accessible
    [Tags]    zmq    messaging    connectivity
    Validate Service Connectivity    messaging_zmq_publisher


Validate ZMQ Subscriber
    [Documentation]    Validate ZMQ subscriber is accessible
    [Tags]    zmq    messaging    connectivity
    Validate Service Connectivity    messaging_zmq_subscriber


Validate Messaging Infrastructure
    [Documentation]    Validate entire messaging infrastructure
    [Tags]    zmq    messaging    connectivity
    Start Component    messaging
    Sleep    2s
    Validate ZMQ Publisher
    Validate ZMQ Subscriber
    Log    ✓ Messaging infrastructure validated    level=INFO


# ============================================================================
# DATA WAREHOUSING VALIDATION KEYWORDS
# ============================================================================

Validate Data Storage
    [Documentation]    Validate data warehousing components
    [Tags]    storage    warehousing    connectivity
    
    Log    Validating data storage...
    Validate Service Connectivity    database_postgres
    Validate Service Connectivity    cache_redis
    Validate Service Connectivity    storage_influxdb
    Log    ✓ Data warehousing validated    level=INFO


Validate Database Connection
    [Documentation]    Validate PostgreSQL database connection
    [Tags]    database    storage    connectivity
    Validate Service Connectivity    database_postgres


Validate Cache Connection
    [Documentation]    Validate Redis cache connection
    [Tags]    cache    storage    connectivity
    Validate Service Connectivity    cache_redis


Validate InfluxDB Connection
    [Documentation]    Validate InfluxDB time-series storage
    [Tags]    influxdb    storage    connectivity
    Validate Service Connectivity    storage_influxdb


# ============================================================================
# SYSTEM INITIALIZATION KEYWORDS
# ============================================================================

Initialize System
    [Documentation]    Initialize entire system
    [Tags]    system    initialization    connectivity
    
    Log    Initializing Market Data Platform...
    Start All Components
    Sleep    3s
    Validate All Services Connectivity
    Log    ✓ System initialization complete    level=INFO


Shutdown System
    [Documentation]    Shutdown entire system gracefully
    [Tags]    system    shutdown    connectivity
    
    Log    Shutting down Market Data Platform...
    Stop All Components
    Log    ✓ System shutdown complete    level=INFO


Reinitialize System
    [Documentation]    Restart the entire system
    [Tags]    system    restart    connectivity
    
    Log    Reinitializing system...
    Shutdown System
    Sleep    2s
    Initialize System


# ============================================================================
# DIAGNOSTIC & REPORTING KEYWORDS
# ============================================================================

Print Connectivity Report
    [Documentation]    Print detailed connectivity report for all services
    [Tags]    diagnostics    reporting    connectivity
    
    ${cmd}=    Catenate    SEPARATOR=|
    ...    cd /root/rf_env &&
    ...    python -c "
    ...    import sys, json
    ...    sys.path.insert(0, '.')
    ...    from market_data_platform.connectivity.validator import ConnectivityValidatorSync
    ...    v = ConnectivityValidatorSync()
    ...    summary = v.get_summary()
    ...    print(json.dumps(summary, indent=2, default=str))
    ...    "
    
    ${result}=    Run Process    bash    -c    ${cmd}    shell=True
    Log    ${result.stdout}
    Log Many    ${result.stdout.splitlines()}


Print Component Status Report
    [Documentation]    Print detailed component status report
    [Tags]    diagnostics    reporting    component
    
    ${result}=    Run Process    bash    lib/component_manager.sh    status
    ...    cwd=/root/rf_env    shell=True
    Log    ${result.stdout}


Health Check Summary
    [Documentation]    Generate health check summary
    [Tags]    diagnostics    health    summary
    
    Log    ╔════════════════════════════════════════════════╗
    Log    ║     SYSTEM HEALTH CHECK SUMMARY                ║
    Log    ╚════════════════════════════════════════════════╝
    
    ${overall}=    Get Overall Connectivity Status
    Log    Overall Status: ${overall}
    
    ${components}=    Get Component Status
    Log    Component Status:
    Log    ${components}
    
    Log    ═════════════════════════════════════════════════


# ============================================================================
# CLI TASK KEYWORDS
# ============================================================================

Run Component CLI Task
    [Arguments]    ${task}    @{args}
    [Documentation]    Run a component-related CLI task
    [Tags]    cli    task    component
    
    ${cmd}=    Catenate    SEPARATOR=    bash bin/start.sh    bash bin/stop.sh    bash bin/verify_services.sh    bash lib/component_manager.sh
    Log    Running CLI task: ${task} with args: ${args}
    
    ${result}=    Run Process    bash    -c    ${cmd} ${task} ${args}
    ...    cwd=/root/rf_env    shell=True
    Should Be Equal    ${result.rc}    0    CLI task failed: ${task}
    [Return]    ${result.stdout}


# ============================================================================
# PERFORMANCE & METRICS KEYWORDS
# ============================================================================

Measure Service Response Time
    [Arguments]    ${service_name}
    [Documentation]    Measure response time for a service
    [Tags]    performance    metrics    connectivity
    
    ${health}=    Get Service Health Status    ${service_name}
    ${response_time}=    Get From Dictionary    ${health}    response_time
    Log    Response time for ${service_name}: ${response_time}s
    [Return]    ${response_time}


Benchmark All Services
    [Documentation]    Benchmark response times for all services
    [Tags]    performance    metrics    benchmark
    
    @{services}=    Create List
    ...    database_postgres
    ...    cache_redis
    ...    storage_influxdb
    ...    monitoring_prometheus
    ...    api_python
    ...    gateway_go
    
    FOR    ${service}    IN    @{services}
        ${response_time}=    Measure Service Response Time    ${service}
        Log    ${service}: ${response_time}s
    END
