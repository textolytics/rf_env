*** Settings ***
Documentation    Connectivity Validation Tests
...              Tests for service connectivity, health checks,
...              and network accessibility

Resource    ../resources/component_management.robot
Resource    ../resources/common.robot

Suite Setup       Initialize System
Suite Teardown    Shutdown System


*** Test Cases ***

TC_C001_Validate_PostgreSQL_Connectivity
    [Documentation]    Verify PostgreSQL database connectivity
    [Tags]    connectivity    database    postgres
    Validate Service Connectivity    database_postgres
    Log    ✓ PostgreSQL connectivity validated


TC_C002_Validate_Redis_Connectivity
    [Documentation]    Verify Redis cache connectivity
    [Tags]    connectivity    cache    redis
    Validate Service Connectivity    cache_redis
    Log    ✓ Redis connectivity validated


TC_C003_Validate_InfluxDB_Connectivity
    [Documentation]    Verify InfluxDB time-series database connectivity
    [Tags]    connectivity    storage    influxdb
    Validate Service Connectivity    storage_influxdb
    Log    ✓ InfluxDB connectivity validated


TC_C004_Validate_Prometheus_Connectivity
    [Documentation]    Verify Prometheus metrics endpoint connectivity
    [Tags]    connectivity    monitoring    prometheus
    Validate Service Connectivity    monitoring_prometheus
    Log    ✓ Prometheus connectivity validated


TC_C005_Validate_Python_API_Connectivity
    [Documentation]    Verify Python API service connectivity
    [Tags]    connectivity    api    python
    Validate Service Connectivity    api_python
    Log    ✓ Python API connectivity validated


TC_C006_Validate_Go_Gateway_Connectivity
    [Documentation]    Verify Go Gateway service connectivity
    [Tags]    connectivity    gateway    go
    Validate Service Connectivity    gateway_go
    Log    ✓ Go Gateway connectivity validated


TC_C007_Validate_ZMQ_Publisher_Connectivity
    [Documentation]    Verify ZMQ Publisher endpoint connectivity
    [Tags]    connectivity    zmq    messaging    publisher
    Validate ZMQ Publisher
    Log    ✓ ZMQ Publisher connectivity validated


TC_C008_Validate_ZMQ_Subscriber_Connectivity
    [Documentation]    Verify ZMQ Subscriber endpoint connectivity
    [Tags]    connectivity    zmq    messaging    subscriber
    Validate ZMQ Subscriber
    Log    ✓ ZMQ Subscriber connectivity validated


TC_C009_Validate_All_Services
    [Documentation]    Validate connectivity to all services simultaneously
    [Tags]    connectivity    all    system
    Validate All Services Connectivity
    Log    ✓ All services connectivity validated


TC_C010_Database_Warehousing_Infrastructure
    [Documentation]    Validate complete database warehousing infrastructure
    [Tags]    connectivity    warehousing    storage
    Validate Data Storage
    Log    ✓ Data warehousing infrastructure validated


TC_C011_Messaging_Infrastructure
    [Documentation]    Validate complete messaging infrastructure
    [Tags]    connectivity    messaging    zmq
    Validate Messaging Infrastructure
    Log    ✓ Messaging infrastructure validated


TC_C012_Assert_PostgreSQL_Healthy
    [Documentation]    Assert PostgreSQL service is healthy
    [Tags]    connectivity    assertion    health
    Assert Service Is Healthy    database_postgres
    Log    ✓ PostgreSQL health assertion passed


TC_C013_Assert_Redis_Healthy
    [Documentation]    Assert Redis service is healthy
    [Tags]    connectivity    assertion    health
    Assert Service Is Healthy    cache_redis
    Log    ✓ Redis health assertion passed


TC_C014_Assert_InfluxDB_Healthy
    [Documentation]    Assert InfluxDB service is healthy
    [Tags]    connectivity    assertion    health
    Assert Service Is Healthy    storage_influxdb
    Log    ✓ InfluxDB health assertion passed


TC_C015_Assert_API_Healthy
    [Documentation]    Assert Python API service is healthy
    [Tags]    connectivity    assertion    health
    Assert Service Is Healthy    api_python
    Log    ✓ API health assertion passed


TC_C016_Assert_Gateway_Healthy
    [Documentation]    Assert Go Gateway service is healthy
    [Tags]    connectivity    assertion    health
    Assert Service Is Healthy    gateway_go
    Log    ✓ Gateway health assertion passed


TC_C017_Assert_All_Services_Healthy
    [Documentation]    Assert all services are healthy
    [Tags]    connectivity    assertion    health    system
    Assert All Services Healthy
    Log    ✓ All services health assertion passed


TC_C018_Get_Service_Health_Details
    [Documentation]    Retrieve detailed health information for a service
    [Tags]    connectivity    health    details
    ${health}=    Get Service Health Status    database_postgres
    Log    Service health details: ${health}
    Should Contain    ${health}    status
    Should Contain    ${health}    response_time
    Log    ✓ Service health details retrieved


TC_C019_Service_Response_Time_Measurement
    [Documentation]    Measure service response time
    [Tags]    performance    connectivity    response-time
    ${response_time}=    Measure Service Response Time    api_python
    Log    API response time: ${response_time}s
    Should Be True    ${response_time} >= 0
    Log    ✓ Response time measured: ${response_time}s


TC_C020_Service_Response_Time_Benchmark
    [Documentation]    Benchmark response times across services
    [Tags]    performance    connectivity    benchmark
    Benchmark All Services
    Log    ✓ Service benchmark completed


TC_C021_Wait_For_PostgreSQL_Ready
    [Documentation]    Wait for PostgreSQL to become ready
    [Tags]    connectivity    wait    database
    Stop Component    database
    Start Component    database
    Wait For Service To Be Ready    database_postgres    timeout=60s
    Log    ✓ PostgreSQL ready within timeout


TC_C022_Wait_For_API_Ready
    [Documentation]    Wait for API service to become ready
    [Tags]    connectivity    wait    api
    Stop Component    api
    Start Component    api
    Wait For Service To Be Ready    api_python    timeout=60s
    Log    ✓ API ready within timeout


TC_C023_Overall_System_Connectivity
    [Documentation]    Check overall system connectivity status
    [Tags]    connectivity    system    status
    ${status}=    Get Overall Connectivity Status
    Log    Overall connectivity status: ${status}
    Should Be Equal    ${status}    healthy
    Log    ✓ System connectivity status: ${status}


TC_C024_Connectivity_Report
    [Documentation]    Generate and log connectivity report
    [Tags]    reporting    diagnostics    connectivity
    Print Connectivity Report
    Log    ✓ Connectivity report generated


TC_C025_Health_Check_Summary
    [Documentation]    Generate health check summary
    [Tags]    reporting    diagnostics    health
    Health Check Summary
    Log    ✓ Health check summary generated


*** Keywords ***

Setup All Services For Connectivity Tests
    [Documentation]    Setup all services for connectivity testing
    Log    Setting up services for connectivity tests...
    Start All Components
    Sleep    5s
    Validate All Services Connectivity
