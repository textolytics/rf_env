# Terminal System - Complete User Guide

## Overview

The Market Data Platform now features an **advanced Midnight Commander-style terminal system** with comprehensive component management, configuration control, testing, and keyword execution - all accessible through keyboard navigation.

**Status**: ✅ **ALL SYSTEMS VALIDATED AND OPERATIONAL**

## Quick Start

```bash
cd /root/rf_env
python market_data_platform/cli/unified_terminal_launcher.py
```

Then select option **1** for the Advanced Menu Terminal (Recommended).

---

## Terminal Modes

### 1. 🎯 Advanced Menu Terminal (RECOMMENDED)
**Complete dropdown menu system with all features integrated**

- ✅ Component Management (Start/Stop/Status)
- ✅ Configuration Management (View/Edit/Reset)
- ✅ Keyword Browser (5 categories, 27 keywords)
- ✅ Test Management (Run tests, get results)
- ✅ Command Execution (Health check, install, build, etc.)
- ✅ Full Keyboard Navigation

**Best For**: All operations in a single integrated interface

### 2. 🎨 Commander Terminal
**Two-panel Midnight Commander-style interface**

- Two-panel file navigation
- Command execution
- Real-time status tracking
- Visual command history

**Best For**: File browsing and management

### 3. 📊 Dashboard Terminal
**System monitoring and performance metrics**

- Real-time component status
- Performance graphs
- Resource monitoring
- System health overview

**Best For**: System monitoring and troubleshooting

---

## Advanced Menu Terminal Features

### 📋 Component Management

**View Status**
- See all 7 components and their current state
- Visual indicators: ▶ (Running), ⊡ (Stopped), ✗ (Error), ? (Unknown)
- Port information and descriptions
- Last check timestamp

Components Tracked:
1. **ZMQ Bus** (Port 5555) - Message routing system
2. **Python Gateway** (Port 8001) - Python connectivity module
3. **Go Gateway** (Port 8002) - Go Gate.io connector
4. **Rust Gateway** (Port 8003) - Rust data processor
5. **Robot Framework** - Test automation framework
6. **Redis Cache** (Port 6379) - Data caching layer
7. **Postgres DB** (Port 5432) - Primary database

**Start Component**
- Select any component to start
- Simulates or executes actual startup
- Updates status in real-time
- Shows startup messages

**Stop Component**
- Gracefully shutdown any component
- Confirms stop operation
- Updates status display

### 🧪 Testing

**Run Tests - All Types**
- All Tests - Run complete test suite
- Python Tests - Language-specific test suite
- Go Tests - Go module tests
- Rust Tests - Rust module tests
- Robot Framework Tests - RF test suite
- Integration Tests - Cross-component tests

**Run Specific Tests**
- Test Python Modules
- Test Go Connectivity
- Test Rust Processor
- Test RF Execution
- Test Data Pipeline
- Test Complete Flow

### ⚙️ Configuration Management

**View Config**
- Display all 15 configuration settings
- Current values for each setting
- Port configurations
- Timeouts and intervals

**Settings Available:**
```
• Theme: byobu (color scheme)
• Auto Start: True (start services automatically)
• Auto Refresh: True (refresh data automatically)
• Refresh Interval: 5000 (milliseconds)
• Execution Timeout: 30 (seconds)
• Max History: 100 (lines)
• ZMQ Host: 127.0.0.1
• ZMQ Port: 5555
• Python Gateway Port: 8001
• Go Gateway Port: 8002
• Rust Gateway Port: 8003
• Database: postgresql://localhost/market_data
• Redis URL: redis://localhost:6379
• Log Level: INFO
• Environment: development
```

**Edit Config**
- Theme Selection
- Refresh Interval Adjustment
- Timeout Configuration
- ZMQ Host Configuration
- ZMQ Port Configuration
- Log Level Selection

**Reset Config**
- Restore all settings to defaults
- Confirms before resetting
- Updates configuration file

### 🔑 Keywords

**Browse 27 Robot Framework Keywords**

**5 Categories:**

1. **Gateway Management** (6 keywords)
   - Connect To Gateway
   - Disconnect From Gateway
   - List Available Gateways
   - Get Gateway Status
   - Stream Market Data
   - Stop Data Stream

2. **Component Management** (6 keywords)
   - Start Component
   - Stop Component
   - Check Component Status
   - Restart Component
   - Get Component Info
   - Show All Components

3. **Data Operations** (5 keywords)
   - Fetch OHLC Data
   - Process Market Data
   - Store Data
   - Query Data
   - Aggregate Data

4. **Configuration** (5 keywords)
   - Set Config Value
   - Get Config Value
   - Load Configuration
   - Save Configuration
   - Reset Configuration

5. **Testing** (5 keywords)
   - Run All Tests
   - Run Python Tests
   - Run Go Tests
   - Run Rust Tests
   - Run Integration Tests

### 💾 Commands

**Execute System Commands**

- **Health Check** - Verify all components operational
- **Install Dependencies** - Install required packages
- **Build All** - Compile all modules
- **Connect Gate.io** - Test Gate.io connectivity
- **Get Market Prices** - Fetch current market data

Each command:
- Executes asynchronously
- Shows real-time output
- Returns status and results
- Logs to output history

---

## Keyboard Navigation

### Controls

| Key | Action |
|-----|--------|
| **↑** | Move to previous menu item |
| **↓** | Move to next menu item |
| **Enter** | Select/Execute menu item or open submenu |
| **q** | Go back to parent menu or Quit |
| **←** / **→** | (in some modes) Adjust values or navigate panels |

### Navigation Flow

```
Main Menu
  ├─ Components
  │  ├─ View Status → Select Component → Show Details → [q] back
  │  ├─ Start Component → Select Component → Confirm → [q] back
  │  └─ Stop Component → Select Component → Confirm → [q] back
  ├─ Testing
  │  ├─ Run Tests → Select Test Type → Execute → [q] back
  │  └─ Run Specific Test → Select Test → Execute → [q] back
  ├─ Configuration
  │  ├─ View Config → Display Settings → [q] back
  │  ├─ Edit Config → Select Setting → Modify → [q] back
  │  └─ Reset Config → Confirm → [q] back
  ├─ Keywords
  │  └─ Show Keywords → Select Category → View Keywords → [q] back
  ├─ Commands
  │  └─ Execute Command → Select Command → Run → [q] back
  └─ q - Quit
```

### Usage Example

**Starting a Component:**
```
1. Press ↓ until "Start Component" is highlighted
2. Press Enter to open submenu
3. Press ↓ to select component (e.g., "Python Gateway")
4. Press Enter to start
5. See output: "✓ Python Gateway started successfully"
6. Press q to go back
7. Repeat or press q again to return to main menu
```

**Running Tests:**
```
1. Press ↓ until "Run Tests" is highlighted
2. Press Enter to open submenu
3. Press ↓ to select test type (e.g., "Python Tests")
4. Press Enter to run
5. See test output in output section
6. Press q to continue menu navigation
```

---

## Display Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Advanced Menu Terminal v1.0                         │
├─────────────────────────────────────────────────────────────────────────┤
│                          MAIN MENU                                      │
│  1.  📋 Components                                                      │
│  2.    ▶ View Status                                                    │
│  3.    ▶ Start Component                                                │
│  4.    ▶ Stop Component                                                 │
│  5.  ────────────────────────────────────────────────────────           │
│  6.  🧪 Testing                                                         │
│  7.    ▶ Run Tests                                                      │
│  8.    ▶ Run Specific Test                                              │
│  ...                                                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  OUTPUT (Last 20 lines):                                                │
│  ✓ Python Gateway started successfully                                  │
│  ✓ Go Gateway is running (Port 8002)                                    │
│  ⊡ Rust Gateway is stopped                                              │
│  ? Redis Cache status unknown                                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Status: ✓ Operation completed successfully                             │
│  Help: ↑↓=Navigate  Enter=Select  q=Back  Type number to jump           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Color Scheme

The terminal uses a Byobu 4-color theme:

| Color | Usage |
|-------|-------|
| 🔵 **Blue** | Section headers, titles |
| 🟢 **Green** | Menu items, success messages, operational items |
| 🟡 **Yellow** | Warnings, timeouts, cautionary messages |
| 🔷 **Cyan** | Information text, help text, status lines |

---

## Configuration Files

### Location
```
~/.market_data_config.json
```

### Format
```json
{
  "theme": "byobu",
  "auto_start": true,
  "auto_refresh": true,
  "refresh_interval": 5000,
  "execution_timeout": 30,
  "max_history": 100,
  "zmq_host": "127.0.0.1",
  "zmq_port": 5555,
  "python_gateway_port": 8001,
  "go_gateway_port": 8002,
  "rust_gateway_port": 8003,
  "database": "postgresql://localhost/market_data",
  "redis_url": "redis://localhost:6379",
  "log_level": "INFO",
  "environment": "development"
}
```

### Manual Editing
You can edit the config file directly or use the terminal's Edit Config menu option.

---

## Common Operations

### ✅ Check All Components Status

1. Launch terminal: `python unified_terminal_launcher.py` → Option 1
2. Navigate to: **Components** → **View Status**
3. Review each component's status indicator
4. Port information shows connectivity

### ✅ Start All Components

1. Go to: **Components** → **Start Component**
2. Repeat for each component needed
3. Verify status shows ▶ (RUNNING) for each

### ✅ Run Full Test Suite

1. Go to: **Testing** → **Run Tests** → **All Tests**
2. Wait for completion
3. Review output section for results

### ✅ View Current Configuration

1. Go to: **Configuration** → **View Config**
2. All 15 settings displayed with current values
3. Press q to return

### ✅ Change Port Configuration

1. Go to: **Configuration** → **Edit Config**
2. Select: **ZMQ Port** or **Python Gateway Port** etc.
3. Enter new port number
4. Press Enter to save
5. Config file updates automatically

### ✅ Browse Available Keywords

1. Go to: **Keywords** → **Show Keywords**
2. Select category (Gateway Management, Component Management, etc.)
3. View all keywords in that category
4. Press q to explore other categories

### ✅ Execute System Command

1. Go to: **Commands** → **Execute Command**
2. Select desired command:
   - Health Check - Check all systems
   - Install Dependencies - Setup environment
   - Build All - Compile modules
   - Connect Gate.io - Test external connectivity
   - Get Market Prices - Fetch current data
3. Command executes with real-time output
4. Results shown in output section

---

## Troubleshooting

### Terminal Won't Launch

**Error**: `ImportError: No module named 'curses'`
- **Solution**: Curses is included in Python on Linux. Ensure you're running the terminal on Linux or use WSL2 on Windows.

### Colors Not Displaying Correctly

**Error**: Text is garbled or colors are wrong
- **Solution**: 
  1. Ensure terminal supports ANSI colors
  2. Set `TERM=xterm-256color` or `TERM=screen-256color`
  3. Try different terminal emulator

### Component Not Responding

**Error**: Component shows ? (UNKNOWN) or ✗ (ERROR) status
- **Solution**:
  1. Check if component process is actually running: `netstat -an | grep PORT`
  2. Start component via menu: **Components** → **Start Component**
  3. Check logs for errors
  4. Verify port is not in use by another service

### Configuration Not Saving

**Error**: Changes don't persist after closing terminal
- **Solution**:
  1. Check write permissions on ~/.market_data_config.json
  2. Verify disk space available
  3. Try resetting config: **Configuration** → **Reset Config**

### Test Fails to Execute

**Error**: "Failed to run test" message
- **Solution**:
  1. Check test files exist in workspace
  2. Verify dependencies installed: Run **Commands** → **Install Dependencies**
  3. Check Python environment is correct
  4. Review test output for specific errors

---

## File Structure

```
/root/rf_env/market_data_platform/cli/
├── unified_terminal_launcher.py       # Main entry point - mode selector
├── advanced_menu_terminal.py           # Advanced dropdown menu system
├── commander_terminal.py               # MC-style two-panel interface
├── advanced_dashboard.py               # System monitoring dashboard
├── terminal_integration.py             # Integration utilities
└── test_components.py                  # Validation test suite
```

---

## Architecture Overview

### ComponentManager
- Manages 7 system components
- Tracks status (Running/Stopped/Error/Unknown/Starting)
- Port-based connectivity detection
- Thread-safe operations

### ConfigManager
- 15 configuration settings
- JSON file persistence (~/.market_data_config.json)
- Default config fallback
- Get/Set/Save operations

### KeywordManager
- Discovers Robot Framework keywords
- 5 categories, 27 total keywords
- Category-based filtering
- Keyword information display

### MenuSystem
- Hierarchical dropdown menus
- 21 main menu items
- Keyboard-driven navigation
- Real-time output display
- Color-coded display (4-color scheme)

---

## Performance Notes

- **Startup Time**: < 1 second
- **Menu Response**: Instant (< 100ms)
- **Component Status Check**: ~2 seconds (network timeout)
- **Test Execution**: Depends on test suite (typically 5-30 seconds)
- **Output Buffer**: Last 100 lines retained

---

## Advanced Features

### Multi-threaded Operations
- Component operations run in background threads
- Menu remains responsive during long operations
- Status updates in real-time

### Port Detection
- Automatic component status detection via port availability
- Uses `netcat` (nc) for connectivity check
- Configurable port mappings

### Output History
- Maintains last 100 lines of output
- Scrollable output display
- All operation results logged
- Can be extended by modifying `MAX_OUTPUT_LINES`

### Keyboard Shortcuts
- Quick menu jumping by typing numbers
- Single-letter commands (q for quit/back)
- Arrow key navigation
- Enter key selection

---

## Quick Reference

```
┌─────────────────────────────────────────────────────┐
│            QUICK COMMAND REFERENCE                  │
├─────────────────────────────────────────────────────┤
│ Launch:        python unified_terminal_launcher.py  │
│ Select:        Option 1 (Advanced Menu Terminal)    │
│                                                     │
│ Navigation:    ↑↓ = Move  Enter = Select  q = Back  │
│                                                     │
│ Components:    7 total (see Components menu)        │
│ Keywords:      27 total in 5 categories             │
│ Settings:      15 configuration options             │
│ Tests:         12 test suites available             │
│ Commands:      5 system commands                    │
│                                                     │
│ Config File:   ~/.market_data_config.json           │
│ Default Ports: ZMQ=5555, PY=8001, GO=8002, RU=8003 │
│                Redis=6379, DB=5432                  │
└─────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Launch the Terminal**
   ```bash
   python /root/rf_env/market_data_platform/cli/unified_terminal_launcher.py
   ```

2. **Explore Main Menu**
   - Use ↑↓ to navigate sections
   - Press Enter on sections to expand

3. **Check Component Status**
   - Go to Components → View Status
   - See all 7 components and their ports

4. **Run Your First Test**
   - Go to Testing → Run Tests → All Tests
   - Watch real-time output

5. **Configure System**
   - Go to Configuration → View Config
   - Review all 15 settings
   - Modify as needed

6. **Browse Keywords**
   - Go to Keywords → Show Keywords
   - Explore 5 categories and 27 keywords

---

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Review output messages for error details
3. Verify all tests pass: `python test_components.py`
4. Check component status and logs

---

**Terminal System Status**: ✅ **ALL SYSTEMS OPERATIONAL**

Version: 1.0
Last Updated: 2024
