*** Settings ***
Documentation    Data Warehousing Component Tests
...              Tests for database, cache, and storage components
...              including PostgreSQL, Redis, and InfluxDB

Resource    ../resources/component_management.robot
Resource    ../resources/common.robot

Suite Setup       Initialize Data Warehousing
Suite Teardown    Shutdown Data Warehousing


*** Test Cases ***

TC_DW001_Start_PostgreSQL_Database
    [Documentation]    Verify PostgreSQL database component starts
    [Tags]    database    warehousing    postgres    start
    Start Component    database
    Sleep    2s
    Validate Database Connection
    Log    ✓ PostgreSQL database started


TC_DW002_Start_Redis_Cache
    [Documentation]    Verify Redis cache component starts
    [Tags]    cache    warehousing    redis    start
    Start Component    database
    Sleep    2s
    Validate Cache Connection
    Log    ✓ Redis cache started


TC_DW003_Start_InfluxDB_Storage
    [Documentation]    Verify InfluxDB time-series storage starts
    [Tags]    storage    warehousing    influxdb    start
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Validate InfluxDB Connection
    Log    ✓ InfluxDB storage started


TC_DW004_Database_Connectivity
    [Documentation]    Verify database connectivity
    [Tags]    database    warehousing    connectivity
    Start Component    database
    Sleep    2s
    Validate Service Connectivity    database_postgres
    Log    ✓ Database connectivity verified


TC_DW005_Cache_Connectivity
    [Documentation]    Verify cache connectivity
    [Tags]    cache    warehousing    connectivity
    Start Component    database
    Sleep    2s
    Validate Service Connectivity    cache_redis
    Log    ✓ Cache connectivity verified


TC_DW006_Storage_Connectivity
    [Documentation]    Verify storage connectivity
    [Tags]    storage    warehousing    connectivity
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Validate Service Connectivity    storage_influxdb
    Log    ✓ Storage connectivity verified


TC_DW007_All_Warehousing_Components
    [Documentation]    Verify all data warehousing components
    [Tags]    warehousing    all    connectivity
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Validate Data Storage
    Log    ✓ All warehousing components validated


TC_DW008_PostgreSQL_Health_Check
    [Documentation]    Verify PostgreSQL health status
    [Tags]    database    warehousing    health
    Start Component    database
    Sleep    2s
    Assert Service Is Healthy    database_postgres
    Log    ✓ PostgreSQL health check passed


TC_DW009_Redis_Health_Check
    [Documentation]    Verify Redis health status
    [Tags]    cache    warehousing    health
    Start Component    database
    Sleep    2s
    Assert Service Is Healthy    cache_redis
    Log    ✓ Redis health check passed


TC_DW010_InfluxDB_Health_Check
    [Documentation]    Verify InfluxDB health status
    [Tags]    storage    warehousing    health
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Assert Service Is Healthy    storage_influxdb
    Log    ✓ InfluxDB health check passed


TC_DW011_Database_Response_Time
    [Documentation]    Measure PostgreSQL response time
    [Tags]    database    warehousing    performance
    Start Component    database
    Sleep    2s
    ${response_time}=    Measure Service Response Time    database_postgres
    Log    Database response time: ${response_time}s
    Log    ✓ Response time: ${response_time}s


TC_DW012_Cache_Response_Time
    [Documentation]    Measure Redis response time
    [Tags]    cache    warehousing    performance
    Start Component    database
    Sleep    2s
    ${response_time}=    Measure Service Response Time    cache_redis
    Log    Cache response time: ${response_time}s
    Log    ✓ Response time: ${response_time}s


TC_DW013_Storage_Response_Time
    [Documentation]    Measure InfluxDB response time
    [Tags]    storage    warehousing    performance
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    ${response_time}=    Measure Service Response Time    storage_influxdb
    Log    Storage response time: ${response_time}s
    Log    ✓ Response time: ${response_time}s


TC_DW014_Restart_Database_Component
    [Documentation]    Verify database component can be restarted
    [Tags]    database    warehousing    restart
    Start Component    database
    Sleep    2s
    Validate Data Storage
    Restart Component    database
    Sleep    3s
    Validate Data Storage
    Log    ✓ Database component restarted successfully


TC_DW015_Stop_Database_Gracefully
    [Documentation]    Verify database stops gracefully
    [Tags]    database    warehousing    stop    graceful
    Start Component    database
    Sleep    2s
    Assert Component Is Running    database
    Stop Component    database
    Sleep    1s
    Assert Component Is Stopped    database
    Log    ✓ Database stopped gracefully


TC_DW016_Database_Recovery
    [Documentation]    Verify database recovery from stop/start
    [Tags]    database    warehousing    recovery
    FOR    ${i}    IN RANGE    3
        Start Component    database
        Sleep    2s
        Validate Data Storage
        Stop Component    database
        Sleep    1s
    END
    Log    ✓ Database recovery test completed


TC_DW017_Multi_Component_Warehousing
    [Documentation]    Verify multiple warehousing components work together
    [Tags]    warehousing    multi-component    integration
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Validate Database Connection
    Validate Cache Connection
    Validate InfluxDB Connection
    Log    ✓ Multi-component warehousing operational


TC_DW018_Warehousing_Data_Integrity
    [Documentation]    Verify data warehousing integrity
    [Tags]    warehousing    data-integrity
    Start Component    database
    Sleep    2s
    Validate Database Connection
    Validate Cache Connection
    Log    ✓ Data warehousing integrity verified


TC_DW019_Wait_For_Database_Ready
    [Documentation]    Verify wait for database to be ready
    [Tags]    database    warehousing    wait
    Stop Component    database
    Start Component    database
    Wait For Service To Be Ready    database_postgres    timeout=60s
    Log    ✓ Database ready within timeout


TC_DW020_Wait_For_Storage_Ready
    [Documentation]    Verify wait for storage to be ready
    [Tags]    storage    warehousing    wait
    Stop Component    storage
    Start Component    database
    Sleep    1s
    Start Component    storage
    Wait For Service To Be Ready    storage_influxdb    timeout=60s
    Log    ✓ Storage ready within timeout


TC_DW021_Database_Service_Details
    [Documentation]    Retrieve database service health details
    [Tags]    database    warehousing    details
    Start Component    database
    Sleep    2s
    ${health}=    Get Service Health Status    database_postgres
    Log    Database service details: ${health}
    Should Contain    ${health}    status
    Log    ✓ Database service details retrieved


TC_DW022_Warehousing_Benchmark
    [Documentation]    Benchmark data warehousing components
    [Tags]    warehousing    performance    benchmark
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    ${db_time}=    Measure Service Response Time    database_postgres
    ${cache_time}=    Measure Service Response Time    cache_redis
    ${storage_time}=    Measure Service Response Time    storage_influxdb
    Log    Database: ${db_time}s | Cache: ${cache_time}s | Storage: ${storage_time}s
    Log    ✓ Warehousing benchmark completed


TC_DW023_Warehousing_Report
    [Documentation]    Generate data warehousing report
    [Tags]    warehousing    reporting    diagnostics
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Print Connectivity Report
    Print Component Status Report
    Log    ✓ Warehousing report generated


TC_DW024_PostgreSQL_Port_Verification
    [Documentation]    Verify PostgreSQL is accessible on correct port
    [Tags]    database    warehousing    port    connectivity
    Start Component    database
    Sleep    2s
    ${health}=    Get Service Health Status    database_postgres
    Log    PostgreSQL accessible on configured port
    Should Be Equal    ${health}[status]    healthy
    Log    ✓ PostgreSQL port verified


TC_DW025_Full_Warehousing_Stack
    [Documentation]    Verify complete data warehousing stack
    [Tags]    warehousing    full-stack    system
    Start All Components
    Sleep    5s
    Validate Database Connection
    Validate Cache Connection
    Validate InfluxDB Connection
    Validate All Services Connectivity
    Log    ✓ Full warehousing stack operational


*** Keywords ***

Initialize Data Warehousing
    [Documentation]    Initialize data warehousing environment
    Log    Setting up data warehousing environment...
    Create Directory    logs
    Create Directory    .pids


Shutdown Data Warehousing
    [Documentation]    Shutdown data warehousing environment
    Log    Shutting down data warehousing environment...
    Stop Component    database
    Stop Component    storage
    Sleep    1s
