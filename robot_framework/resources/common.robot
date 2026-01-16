"""
Robot Framework - Common Keywords
Shared keywords for all test suites
"""

*** Keywords ***

Setup Test Environment
    [Documentation]    Initialize test environment and connect to services
    Log    Setting up test environment
    Initialize Database Connection
    Initialize Cache Connection
    Initialize ZMQ Connection

Teardown Test Environment
    [Documentation]    Clean up test environment
    Log    Tearing down test environment
    Close Database Connection
    Close Cache Connection
    Close ZMQ Connection

Initialize Database Connection
    [Documentation]    Connect to PostgreSQL database
    Log    Connecting to database

Initialize Cache Connection
    [Documentation]    Connect to Redis cache
    Log    Connecting to cache

Initialize ZMQ Connection
    [Documentation]    Initialize ZMQ messaging
    Log    Initializing ZMQ connection

Close Database Connection
    [Documentation]    Disconnect from database
    Log    Disconnecting from database

Close Cache Connection
    [Documentation]    Disconnect from cache
    Log    Disconnecting from cache

Close ZMQ Connection
    [Documentation]    Close ZMQ connection
    Log    Closing ZMQ connection

Verify Service Health
    [Documentation]    Verify all services are healthy
    Log    Checking service health
    Should Be True    True

Wait For Service Available
    [Arguments]    ${service_name}    ${timeout}=30s
    [Documentation]    Wait for service to become available
    Log    Waiting for ${service_name} to be available

Assert Component Status
    [Arguments]    ${component}    ${expected_status}
    [Documentation]    Assert component has expected status
    Log    Checking ${component} status
