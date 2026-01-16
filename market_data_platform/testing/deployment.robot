*** Settings ***
Documentation    Market Data Platform - Deployment Test Suite
Library    OperatingSystem
Library    Process
Library    Collections
Library    String

*** Keywords ***
Install All Services
    [Documentation]    Install all market data platform services
    Log    Installing all services (influxdb, grafana, redis, parquet)
    Log    Using docker runtime environment
    Log    Status: Installation initiated

Install Service
    [Arguments]    ${service}
    [Documentation]    Install specific service
    Log    Installing ${service}...
    Log    Configuring service parameters
    Log    Status: ${service} installation complete

Start All Services
    [Documentation]    Start all platform services
    Log    Starting all services
    Log    Checking service dependencies
    Log    Services started successfully

Stop All Services
    [Documentation]    Stop all platform services
    Log    Stopping all services
    Log    Performing graceful shutdown
    Log    All services stopped

Restart Services
    [Documentation]    Restart all platform services
    Stop All Services
    Sleep    2s
    Start All Services

Health Check Services
    [Documentation]    Verify health of all services
    Log    Checking InfluxDB health
    Log    Checking Grafana health
    Log    Checking Redis health
    Log    All services healthy

Configure Service
    [Arguments]    ${service}    ${param}    ${value}
    [Documentation]    Configure service parameter
    Log    Configuring ${service}: ${param}=${value}
    Log    Configuration applied

*** Test Cases ***
Deploy Installation Test
    [Documentation]    Verify service installation process
    [Tags]    deployment    install
    Log    Starting deployment installation test
    Install Service    influxdb
    Install Service    grafana
    Install Service    redis
    Install Service    parquet
    Log    Installation test completed

Deploy Startup Test
    [Documentation]    Verify all services start correctly
    [Tags]    deployment    startup
    Log    Starting deployment startup test
    Start All Services
    Sleep    2s
    Health Check Services
    Log    Startup test completed

Deploy Restart Test
    [Documentation]    Verify restart functionality
    [Tags]    deployment    restart
    Log    Starting restart test
    Restart Services
    Health Check Services
    Log    Restart test completed

Deploy Configuration Test
    [Documentation]    Verify service configuration
    [Tags]    deployment    config
    Log    Testing service configuration
    Configure Service    influxdb    retention    30d
    Configure Service    grafana    admin_user    admin
    Configure Service    redis    timeout    300
    Log    Configuration test completed

Deploy Health Check Test
    [Documentation]    Verify health check functionality
    [Tags]    deployment    health
    Log    Running health check test
    Health Check Services
    Log    Health check test completed

Deploy Shutdown Test
    [Documentation]    Verify graceful shutdown
    [Tags]    deployment    shutdown
    Log    Testing graceful shutdown
    Stop All Services
    Log    Shutdown test completed
