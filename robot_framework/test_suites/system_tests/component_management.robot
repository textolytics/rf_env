*** Settings ***
Documentation    Component Management Integration Tests
...              Tests for graceful start/stop of service components
...              and component group management

Resource    ../resources/component_management.robot
Resource    ../resources/common.robot

Suite Setup       Log    Starting Component Management Tests
Suite Teardown    Shutdown System


*** Test Cases ***

TC_001_Start_Database_Component
    [Documentation]    Verify database component (PostgreSQL + Redis) starts correctly
    [Tags]    component    database    start    connectivity
    Start Component    database
    Sleep    2s
    Assert Component Is Running    database
    Validate Service Connectivity    database_postgres
    Validate Service Connectivity    cache_redis


TC_002_Start_Messaging_Component
    [Documentation]    Verify messaging component (ZMQ) starts correctly
    [Tags]    component    messaging    start    connectivity    zmq
    Start Component    messaging
    Sleep    2s
    Assert Component Is Running    messaging
    Validate ZMQ Publisher
    Validate ZMQ Subscriber


TC_003_Start_Storage_Component
    [Documentation]    Verify storage component (InfluxDB) starts correctly
    [Tags]    component    storage    start    connectivity
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Assert Component Is Running    storage
    Validate Service Connectivity    storage_influxdb


TC_004_Start_Monitoring_Component
    [Documentation]    Verify monitoring component (Prometheus + Grafana) starts correctly
    [Tags]    component    monitoring    start    connectivity
    Start Component    monitoring
    Sleep    2s
    Assert Component Is Running    monitoring
    Validate Service Connectivity    monitoring_prometheus


TC_005_Start_API_Component
    [Documentation]    Verify Python API component starts correctly
    [Tags]    component    api    start    connectivity
    Start Component    database
    Sleep    1s
    Start Component    api
    Sleep    3s
    Assert Component Is Running    api
    Validate Service Connectivity    api_python


TC_006_Start_Gateway_Component
    [Documentation]    Verify Go Gateway component starts correctly
    [Tags]    component    gateway    start    connectivity
    Start Component    database
    Sleep    1s
    Start Component    gateway
    Sleep    2s
    Assert Component Is Running    gateway
    Validate Service Connectivity    gateway_go


TC_007_Stop_Component_Gracefully
    [Documentation]    Verify component stops gracefully
    [Tags]    component    stop    graceful    shutdown
    Start Component    database
    Sleep    2s
    Assert Component Is Running    database
    Stop Component    database
    Sleep    1s
    Assert Component Is Stopped    database


TC_008_Start_All_Components_In_Sequence
    [Documentation]    Verify all components start in correct dependency order
    [Tags]    component    start    sequence    connectivity
    Start All Components
    Sleep    5s
    Validate All Services Connectivity
    Health Check Summary


TC_009_Stop_All_Components_In_Sequence
    [Documentation]    Verify all components stop in reverse dependency order
    [Tags]    component    stop    sequence    shutdown
    [Setup]    Start All Components
    Sleep    3s
    Stop All Components
    Sleep    2s
    Get Component Status


TC_010_Restart_Component
    [Documentation]    Verify component restart (stop + start) works correctly
    [Tags]    component    restart    connectivity
    Start Component    database
    Sleep    2s
    Validate Service Connectivity    database_postgres
    Restart Component    database
    Sleep    2s
    Validate Service Connectivity    database_postgres


TC_011_Component_Dependency_Management
    [Documentation]    Verify component dependencies are respected
    [Tags]    component    dependency    start
    Start Component    api
    Sleep    2s
    Assert Component Is Running    database
    Assert Component Is Running    api
    Log    ✓ Dependencies resolved correctly


TC_012_Validate_Database_Connectivity
    [Documentation]    Validate database component connectivity
    [Tags]    connectivity    database    storage
    Start Component    database
    Sleep    2s
    Validate Database Connection
    Validate Cache Connection


TC_013_Validate_ZMQ_Messaging
    [Documentation]    Validate ZMQ messaging component
    [Tags]    connectivity    zmq    messaging
    Start Component    messaging
    Sleep    2s
    Validate Messaging Infrastructure


TC_014_Validate_Data_Warehousing
    [Documentation]    Validate entire data warehousing infrastructure
    [Tags]    connectivity    storage    warehousing
    Start Component    database
    Sleep    1s
    Start Component    storage
    Sleep    2s
    Validate Data Storage


TC_015_Service_Health_Status_Check
    [Documentation]    Verify service health status reporting
    [Tags]    connectivity    health    reporting
    Start Component    database
    Sleep    2s
    ${health}=    Get Service Health Status    database_postgres
    Should Be Equal    ${health}[status]    healthy
    Log    ✓ Service health status: ${health}[status]


TC_016_Overall_System_Health
    [Documentation]    Check overall system connectivity status
    [Tags]    connectivity    health    system
    Start All Components
    Sleep    5s
    ${status}=    Get Overall Connectivity Status
    Should Be Equal    ${status}    healthy
    Log    ✓ Overall system status: ${status}


TC_017_Wait_For_Service_Ready
    [Documentation]    Verify wait for service to be ready functionality
    [Tags]    connectivity    wait    service
    Start Component    database
    Wait For Service To Be Ready    database_postgres    timeout=30s
    Log    ✓ Database service is ready


TC_018_Connectivity_Report_Generation
    [Documentation]    Generate connectivity report for diagnostics
    [Tags]    reporting    diagnostics    connectivity
    Start All Components
    Sleep    3s
    Print Connectivity Report


TC_019_Component_Status_Report
    [Documentation]    Generate component status report
    [Tags]    reporting    diagnostics    component
    Start All Components
    Sleep    2s
    Print Component Status Report


TC_020_Benchmark_Service_Response_Times
    [Documentation]    Benchmark response times of all services
    [Tags]    performance    metrics    benchmark
    Start All Components
    Sleep    3s
    Benchmark All Services


*** Keywords ***

Setup All Components For Testing
    [Documentation]    Setup all components for testing
    Start All Components
    Sleep    5s
    Validate All Services Connectivity


Cleanup Components After Testing
    [Documentation]    Clean up by stopping all components
    Stop All Components
    Sleep    2s
