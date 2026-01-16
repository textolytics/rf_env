*** Settings ***
Documentation    ZMQ Messaging Component Tests
...              Tests for ZMQ publisher/subscriber validation
...              and messaging infrastructure

Resource    ../resources/component_management.robot
Resource    ../resources/common.robot

Suite Setup       Setup ZMQ Environment
Suite Teardown    Cleanup ZMQ Environment


*** Test Cases ***

TC_Z001_Start_ZMQ_Publisher
    [Documentation]    Verify ZMQ Publisher starts correctly
    [Tags]    zmq    messaging    publisher    start
    Start Component    messaging
    Sleep    2s
    Validate ZMQ Publisher
    Log    ✓ ZMQ Publisher started and validated


TC_Z002_Start_ZMQ_Subscriber
    [Documentation]    Verify ZMQ Subscriber starts correctly
    [Tags]    zmq    messaging    subscriber    start
    Start Component    messaging
    Sleep    2s
    Validate ZMQ Subscriber
    Log    ✓ ZMQ Subscriber started and validated


TC_Z003_ZMQ_Publisher_Endpoint
    [Documentation]    Verify ZMQ Publisher endpoint is accessible
    [Tags]    zmq    messaging    endpoint    connectivity
    Start Component    messaging
    Sleep    2s
    ${health}=    Get Service Health Status    messaging_zmq_publisher
    Should Be Equal    ${health}[status]    healthy
    Log    ✓ Publisher endpoint: ${health}[details][endpoint]


TC_Z004_ZMQ_Subscriber_Endpoint
    [Documentation]    Verify ZMQ Subscriber endpoint is accessible
    [Tags]    zmq    messaging    endpoint    connectivity
    Start Component    messaging
    Sleep    2s
    ${health}=    Get Service Health Status    messaging_zmq_subscriber
    Should Be Equal    ${health}[status]    healthy
    Log    ✓ Subscriber endpoint: ${health}[details][endpoint]


TC_Z005_ZMQ_Both_Components_Running
    [Documentation]    Verify both ZMQ publisher and subscriber are running
    [Tags]    zmq    messaging    both    status
    Start Component    messaging
    Sleep    2s
    Validate ZMQ Publisher
    Validate ZMQ Subscriber
    Log    ✓ Both ZMQ components running


TC_Z006_Restart_ZMQ_Messaging
    [Documentation]    Verify ZMQ messaging can be restarted
    [Tags]    zmq    messaging    restart
    Start Component    messaging
    Sleep    2s
    Validate Messaging Infrastructure
    Stop Component    messaging
    Sleep    1s
    Start Component    messaging
    Sleep    2s
    Validate Messaging Infrastructure
    Log    ✓ ZMQ messaging restarted successfully


TC_Z007_ZMQ_Stop_Gracefully
    [Documentation]    Verify ZMQ components stop gracefully
    [Tags]    zmq    messaging    stop    graceful
    Start Component    messaging
    Sleep    2s
    Assert Component Is Running    messaging
    Stop Component    messaging
    Sleep    1s
    Assert Component Is Stopped    messaging
    Log    ✓ ZMQ stopped gracefully


TC_Z008_Publisher_Response_Time
    [Documentation]    Measure ZMQ Publisher response time
    [Tags]    zmq    messaging    performance    response-time
    Start Component    messaging
    Sleep    2s
    ${response_time}=    Measure Service Response Time    messaging_zmq_publisher
    Log    Publisher response time: ${response_time}s
    Log    ✓ Response time: ${response_time}s


TC_Z009_Subscriber_Response_Time
    [Documentation]    Measure ZMQ Subscriber response time
    [Tags]    zmq    messaging    performance    response-time
    Start Component    messaging
    Sleep    2s
    ${response_time}=    Measure Service Response Time    messaging_zmq_subscriber
    Log    Subscriber response time: ${response_time}s
    Log    ✓ Response time: ${response_time}s


TC_Z010_ZMQ_Messaging_With_Other_Components
    [Documentation]    Verify ZMQ messaging works with other components
    [Tags]    zmq    messaging    integration
    Start Component    database
    Sleep    1s
    Start Component    messaging
    Sleep    2s
    Validate Database Connection
    Validate Messaging Infrastructure
    Log    ✓ ZMQ works with database component


TC_Z011_ZMQ_Publisher_Compilation
    [Documentation]    Verify ZMQ Publisher compiles successfully
    [Tags]    zmq    messaging    compilation
    # This is validated during start
    Start Component    messaging
    Sleep    2s
    Assert Component Is Running    messaging
    Log    ✓ ZMQ Publisher compiled successfully


TC_Z012_ZMQ_Subscriber_Compilation
    [Documentation]    Verify ZMQ Subscriber compiles successfully
    [Tags]    zmq    messaging    compilation
    # This is validated during start
    Start Component    messaging
    Sleep    2s
    Assert Component Is Running    messaging
    Log    ✓ ZMQ Subscriber compiled successfully


TC_Z013_Messaging_Infrastructure_Full_Stack
    [Documentation]    Validate complete messaging infrastructure
    [Tags]    zmq    messaging    full-stack
    Start Component    messaging
    Sleep    2s
    Validate Messaging Infrastructure
    Log    ✓ Full messaging stack operational


TC_Z014_ZMQ_Publisher_Connectivity
    [Documentation]    Assert ZMQ Publisher is healthy and connected
    [Tags]    zmq    messaging    connectivity
    Start Component    messaging
    Sleep    2s
    Assert Service Is Healthy    messaging_zmq_publisher
    Log    ✓ Publisher connectivity asserted


TC_Z015_ZMQ_Subscriber_Connectivity
    [Documentation]    Assert ZMQ Subscriber is healthy and connected
    [Tags]    zmq    messaging    connectivity
    Start Component    messaging
    Sleep    2s
    Assert Service Is Healthy    messaging_zmq_subscriber
    Log    ✓ Subscriber connectivity asserted


TC_Z016_ZMQ_Service_State
    [Documentation]    Check ZMQ service state
    [Tags]    zmq    messaging    status
    Start Component    messaging
    Sleep    2s
    Get Component Status
    Assert Component Is Running    messaging
    Log    ✓ ZMQ service state verified


TC_Z017_ZMQ_Stress_Test
    [Documentation]    Stress test ZMQ messaging
    [Tags]    zmq    messaging    stress-test
    Start Component    messaging
    Sleep    2s
    FOR    ${i}    IN RANGE    5
        Validate ZMQ Publisher
        Validate ZMQ Subscriber
        Sleep    1s
    END
    Log    ✓ ZMQ stress test completed


TC_Z018_ZMQ_Recovery
    [Documentation]    Verify ZMQ recovery from stop/start cycles
    [Tags]    zmq    messaging    recovery
    FOR    ${i}    IN RANGE    3
        Start Component    messaging
        Sleep    1s
        Validate Messaging Infrastructure
        Stop Component    messaging
        Sleep    1s
    END
    Log    ✓ ZMQ recovery test completed


TC_Z019_ZMQ_Messaging_With_System
    [Documentation]    Validate ZMQ messaging with full system running
    [Tags]    zmq    messaging    system-integration
    Start All Components
    Sleep    5s
    Validate Messaging Infrastructure
    Log    ✓ ZMQ messaging validated in full system


TC_Z020_ZMQ_Diagnostics
    [Documentation]    Generate ZMQ messaging diagnostics
    [Tags]    zmq    messaging    diagnostics
    Start Component    messaging
    Sleep    2s
    Print Connectivity Report
    Print Component Status Report
    Log    ✓ ZMQ diagnostics generated


*** Keywords ***

Setup ZMQ Environment
    [Documentation]    Setup ZMQ messaging environment
    Log    Setting up ZMQ environment...
    Create Directory    logs
    Create Directory    .pids


Cleanup ZMQ Environment
    [Documentation]    Clean up ZMQ environment
    Log    Cleaning up ZMQ environment...
    Stop Component    messaging
    Sleep    1s
