# RobotMCP AI Agent Instructions

## Project Overview
**RobotMCP** bridges Natural Language Descriptions and Robot Framework automation by providing an MCP (Model Context Protocol) server that translates high-level scenarios into executable RF test suites. The system orchestrates keyword discovery, library recommendations, session management, and test execution.

### Key Architecture
- **FastMCP Server** (`robotmcp.server`): MCP entry point exposing tools for scenario analysis, library recommendation, keyword execution, and test building
- **ExecutionCoordinator** (`robotmcp.components.execution`): Central orchestrator managing sessions, state, and keyword execution across platforms (web, mobile, API, desktop)
- **Dynamic Keyword Discovery** (`robotmcp.core.dynamic_keyword_orchestrator`): Matches natural language to RF keywords via LibDoc integration and fuzzy matching
- **Session Manager** (`robotmcp.core.session_manager`): Maintains isolated test contexts with variables, libraries, and browser/app states
- **NLP Processor** (`robotmcp.components.nlp_processor`): Parses scenario descriptions and creates test execution plans

## Critical Workflows

### Building & Running Tests
1. **Scenario → Execution Flow**: `analyze_scenario()` → `recommend_libraries()` → `manage_session()` → `execute_step()` → `build_test_suite()` → `run_test_suite()`
2. **Session Lifecycle**: Sessions persist across multiple steps; reuse `session_id` to maintain state (variables, imported libraries, open browsers)
3. **Keyword Execution**: `execute_step()` runs keywords with arguments resolved from session variables; supports variable assignment tracking

### Key MCP Tools (in `server.py`)
| Tool | Purpose |
|------|---------|
| `analyze_scenario` | Parse requirements, create session, identify platform type (web/mobile/API/desktop) |
| `recommend_libraries` | Sample 4 Library managers, merge & rank by relevance using LLM consensus |
| `execute_step` | Run RF keyword in active session, capture result/variables, track execution time |
| `manage_session` | Import libraries, set variables, open browsers (Browser/Playwright/Appium) |
| `find_keywords` | Fuzzy-match keywords against loaded libraries + LibDoc cache |
| `get_session_state` | Snapshot: DOM/page_source, screenshot, variables, browser context, execution history |
| `execute_flow` | Execute multi-step plan with conditional branching |
| `build_test_suite` | Compile validated steps into `.robot` file with proper syntax |
| `run_test_suite` | Execute generated suite via Robot Framework CLI |

## Project Patterns & Conventions

### Data Models (in `robotmcp.models.*`)
- **ExecutionSession**: Holds session_id, imported_libraries, variables, browser state, platform_type
- **ExecutionStep**: Tracks keyword, args, status (pending/running/pass/fail), result, assigned_variables
- **SessionType enum**: XML_PROCESSING, WEB_AUTOMATION, API_TESTING, MOBILE_TESTING, etc.
- **PlatformType enum**: WEB, MOBILE, DESKTOP, API

### Library & Keyword Handling
- Libraries loaded via **LibraryManager** (minimal core: BuiltIn, Collections, String)
- Keyword discovery uses **LibDoc** (XML metadata) + fallback to introspection
- Fuzzy matching with score threshold; prefers exact matches and active library keywords
- **Plugin System**: Library managers can be extended via `get_library_plugin_manager()` for custom integrations

### Session & Variable Management
- Variables stored in `session.variables` dict; tracked as `${VARNAME}` in RF syntax
- **Variable Assignment Tracking**: `ExecutionStep.assigned_variables` records which variables a step assigns (for test suite generation)
- Session imports persist (e.g., `import_library("SeleniumLibrary")` stays for subsequent steps)
- Browser contexts isolated per session (no cross-session bleeding)

### Attach Mode (External RF Bridge)
- Environment variables configure bridge: `ROBOTMCP_ATTACH_HOST`, `ROBOTMCP_ATTACH_PORT`, `ROBOTMCP_ATTACH_TOKEN`
- `McpAttach` library exposes HTTP endpoints to forward commands into running RF context
- Mode: "auto" (use if reachable), "force" (always use bridge), "off" (local only)
- Fallback: If bridge unreachable in auto mode, silently fall back to local execution

### Error Handling
- **Keyword Failures**: Return error message + last state snapshot (DOM, variables, screenshot)
- **Library Load Failures**: Log warning, attempt immediate load, verify loading succeeded
- **Session Not Found**: Return clear 404-style error with available session_ids
- **Serialization**: Enhanced serialization system handles complex Python objects (datetime, enums, custom RF objects)

## Integration Points & External Dependencies

### MCP Protocol (via fastmcp)
- FastMCP decorators: `@mcp.tool()`, `@mcp.prompt()`, `@mcp.resource()` define server capabilities
- Tools are async functions; prompts guide multi-step workflows
- Server lifecycle: startup (load libraries), runtime (execute tools), shutdown (close browsers, cleanup)

### Robot Framework Ecosystem
- **Robot Framework 7.4.1**: Core execution engine
- **Browser Library** / **SeleniumLibrary**: Web UI automation
- **Appium Library**: Mobile testing (iOS/Android via Appium server)
- **RequestsLibrary** / **httpx**: API testing
- **BuiltIn / Collections / String**: Core utilities
- **LibDoc**: External tool for extracting keyword metadata (called via `robot --dryrun`)

### Key External Services
- **Appium Server** (optional): Configured via MobileConfig; defaults to `http://127.0.0.1:4723/`
- **Browser Engines**: Playwright (chromium, firefox, webkit), Selenium WebDriver

## Common Debugging Patterns

1. **Keyword Not Found**: Check session.imported_libraries; run `find_keywords()` with library filter
2. **Variable Resolution Fails**: Check `session.variables` dict; log variable names with `${}` syntax
3. **Browser State Lost**: Verify session_id persistence; check if browser was closed unexpectedly
4. **Library Load Timeout**: Increase timeout in LibraryManager; check for circular imports
5. **Attach Mode Failures**: Check `ROBOTMCP_ATTACH_HOST` reachable; test `/diagnostics` endpoint

## Writing New Components

### Adding a Tool
1. Define async function in `server.py` with clear docstring
2. Decorate with `@mcp.tool()` or `@mcp.tool(enabled=False)` if experimental
3. Use `execution_engine` or other singletons to access state
4. Return JSON-serializable dict with "success", "result", "error" keys
5. Log at info level for user-facing actions, debug for internal flow

### Adding a Library Plugin
1. Inherit from library manager interface in `robotmcp.plugins`
2. Implement keyword discovery + library loading
3. Register via `get_library_plugin_manager().register()`
4. Test with `diagnose_library_plugin()` tool

### Session-Specific Logic
1. Fetch session via `execution_engine.session_manager.get_session(session_id)`
2. Update variables: `session.variables[name] = value`
3. Import libraries: `session.import_library(name)`
4. All state auto-persists in the session object

## Testing & Validation
- **Dry Run**: `run_test_suite_dry()` validates syntax without execution
- **State Snapshots**: Use `get_session_state()` to inspect intermediate state during debugging
- **Keyword Info**: `get_keyword_info()` returns signature, documentation, and implementation hints
- **Prompt-Driven Workflows**: Use `@automate()` and `@learn()` prompts to guide multi-step test creation
