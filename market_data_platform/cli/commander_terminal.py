#!/usr/bin/env python3
"""
Commander Terminal - Midnight Commander & Bloomberg Style Terminal
Multi-panel navigation, command execution, test discovery, and task management
"""

import os
import sys
import curses
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

# Color scheme: Byobu style (4-color modern theme)
class ColorScheme(Enum):
    """Color schemes for terminal UI"""
    PRIMARY = 1      # Blue - Headers, focus
    SUCCESS = 2      # Green - Executable items, success
    WARNING = 3      # Yellow - Important items
    INFO = 4         # Cyan - Information, alternative

    @staticmethod
    def init_colors():
        """Initialize curses color pairs"""
        try:
            # Default terminal background
            curses.init_pair(ColorScheme.PRIMARY.value, curses.COLOR_BLUE, -1)
            curses.init_pair(ColorScheme.SUCCESS.value, curses.COLOR_GREEN, -1)
            curses.init_pair(ColorScheme.WARNING.value, curses.COLOR_YELLOW, -1)
            curses.init_pair(ColorScheme.INFO.value, curses.COLOR_CYAN, -1)
        except:
            pass


@dataclass
class PanelItem:
    """Item in a navigation panel"""
    name: str
    command: str
    item_type: str  # 'test', 'task', 'keyword', 'command'
    description: str = ""
    color: ColorScheme = ColorScheme.INFO


class PanelType(Enum):
    """Types of panels available"""
    COMMANDS = "commands"
    TESTS = "tests"
    TASKS = "tasks"
    KEYWORDS = "keywords"
    NOTEBOOKS = "notebooks"
    EXECUTION = "execution"


class CommanderPanel:
    """Two-panel navigator like Midnight Commander"""
    
    def __init__(self, title: str, panel_type: PanelType, items: List[PanelItem]):
        self.title = title
        self.panel_type = panel_type
        self.items = items
        self.selected_idx = 0
        self.scroll_offset = 0
        self.height = 0
        self.width = 0
        self.y = 0
        self.x = 0
    
    def move_selection(self, direction: int):
        """Move selection up/down"""
        new_idx = self.selected_idx + direction
        if 0 <= new_idx < len(self.items):
            self.selected_idx = new_idx
            # Adjust scroll offset
            if self.selected_idx >= self.scroll_offset + self.height:
                self.scroll_offset = self.selected_idx - self.height + 1
            elif self.selected_idx < self.scroll_offset:
                self.scroll_offset = self.selected_idx
    
    def get_selected_item(self) -> Optional[PanelItem]:
        """Get currently selected item"""
        if 0 <= self.selected_idx < len(self.items):
            return self.items[self.selected_idx]
        return None
    
    def render(self, window) -> int:
        """Render panel to window, return displayed items count"""
        # Clear panel area
        window.attron(curses.color_pair(ColorScheme.PRIMARY.value) | curses.A_BOLD)
        window.addstr(self.y, self.x, self.title.center(self.width))
        window.attroff(curses.color_pair(ColorScheme.PRIMARY.value) | curses.A_BOLD)
        
        # Display items
        displayed = 0
        max_items = self.height - 2
        
        for i in range(max_items):
            item_idx = self.scroll_offset + i
            if item_idx >= len(self.items):
                break
            
            item = self.items[item_idx]
            y_pos = self.y + 1 + i
            
            # Highlight selected item
            is_selected = (item_idx == self.selected_idx)
            if is_selected:
                window.attron(curses.color_pair(item.color.value) | curses.A_REVERSE)
            else:
                window.attron(curses.color_pair(item.color.value))
            
            # Truncate name to fit width
            display_name = item.name[:self.width - 4].ljust(self.width - 4)
            prefix = "▶ " if is_selected else "  "
            window.addstr(y_pos, self.x, prefix + display_name)
            
            window.attroff(curses.color_pair(item.color.value) | 
                          (curses.A_REVERSE if is_selected else 0))
            
            displayed += 1
        
        return displayed


class ExecutionDashboard:
    """Dashboard showing execution results and system status"""
    
    def __init__(self, height: int, width: int, y: int, x: int):
        self.height = height
        self.width = width
        self.y = y
        self.x = x
        self.execution_history: List[Dict[str, Any]] = []
        self.status_message = "Ready"
        self.status_color = ColorScheme.SUCCESS
    
    def add_execution(self, command: str, status: str, duration: float):
        """Record execution result"""
        self.execution_history.append({
            "command": command,
            "status": status,
            "duration": duration,
            "timestamp": datetime.now()
        })
        # Keep last 10 executions
        if len(self.execution_history) > 10:
            self.execution_history = self.execution_history[-10:]
        
        self.status_message = f"{status} in {duration:.2f}s"
        self.status_color = (ColorScheme.SUCCESS if status == "✓" 
                           else ColorScheme.WARNING)
    
    def render(self, window):
        """Render dashboard"""
        # Header
        window.attron(curses.color_pair(ColorScheme.PRIMARY.value) | curses.A_BOLD)
        header = "📊 EXECUTION DASHBOARD".center(self.width)
        window.addstr(self.y, self.x, header)
        window.attroff(curses.color_pair(ColorScheme.PRIMARY.value) | curses.A_BOLD)
        
        # Status line
        window.attron(curses.color_pair(self.status_color.value))
        status_line = f"Status: {self.status_message}".ljust(self.width - 1)
        window.addstr(self.y + 1, self.x, status_line)
        window.attroff(curses.color_pair(self.status_color.value))
        
        # Execution history
        line_y = self.y + 3
        for i, exec_result in enumerate(self.execution_history[-5:]):
            if line_y >= self.y + self.height - 1:
                break
            
            cmd = exec_result["command"][:20]
            status = exec_result["status"]
            duration = exec_result["duration"]
            
            window.attron(curses.color_pair(ColorScheme.INFO.value))
            history_line = f"{status} {cmd:20} {duration:6.2f}s".ljust(self.width - 1)
            window.addstr(line_y, self.x, history_line)
            window.attroff(curses.color_pair(ColorScheme.INFO.value))
            
            line_y += 1


class CommanderTerminal:
    """Main terminal interface with Midnight Commander and Bloomberg style"""
    
    # Function key mappings
    FUNCTION_KEYS = {
        "F1": "help",
        "F2": "refresh",
        "F3": "view",
        "F4": "execute_selected",
        "F5": "copy",
        "F6": "rename",
        "F7": "mkdir",
        "F8": "delete",
        "F9": "exit",
        "F10": "quit"
    }
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        
        # Initialize colors
        ColorScheme.init_colors()
        
        # Setup panels
        self.left_panel = None
        self.right_panel = None
        self.active_panel = 0  # 0 = left, 1 = right
        
        # Setup dashboard
        self.dashboard = None
        
        # Initialize content
        self._setup_panels()
        self._setup_dashboard()
        
        # State
        self.running = True
        self.message = "Ready"
    
    def _setup_panels(self):
        """Initialize left and right panels"""
        panel_width = (self.width - 3) // 2
        panel_height = self.height - 6
        
        # Left panel - Commands
        commands = self._load_commands()
        self.left_panel = CommanderPanel(
            "🚀 COMMANDS",
            PanelType.COMMANDS,
            commands
        )
        self.left_panel.width = panel_width
        self.left_panel.height = panel_height
        self.left_panel.y = 1
        self.left_panel.x = 1
        
        # Right panel - Tests & Tasks
        tests_tasks = self._load_tests_and_tasks()
        self.right_panel = CommanderPanel(
            "🧪 TESTS & TASKS",
            PanelType.TESTS,
            tests_tasks
        )
        self.right_panel.width = panel_width
        self.right_panel.height = panel_height
        self.right_panel.y = 1
        self.right_panel.x = panel_width + 2
    
    def _setup_dashboard(self):
        """Initialize execution dashboard"""
        dashboard_height = self.height - (self.left_panel.height + 3)
        self.dashboard = ExecutionDashboard(
            dashboard_height,
            self.width - 2,
            self.height - dashboard_height,
            1
        )
    
    def _load_commands(self) -> List[PanelItem]:
        """Load commands from CLI"""
        commands = [
            PanelItem("install all", "install all", "command", "Install all services", ColorScheme.SUCCESS),
            PanelItem("start services", "start all", "command", "Start all services", ColorScheme.SUCCESS),
            PanelItem("stop services", "stop all", "command", "Stop all services", ColorScheme.WARNING),
            PanelItem("health check", "health-check", "command", "Check service health", ColorScheme.INFO),
            PanelItem("restart services", "restart all", "command", "Restart all services", ColorScheme.WARNING),
            PanelItem("connect gate.io", "connect gate.io", "command", "Connect to Gate.io", ColorScheme.SUCCESS),
            PanelItem("connect oanda", "connect oanda", "command", "Connect to Oanda", ColorScheme.SUCCESS),
            PanelItem("price EURUSD", "price EURUSD", "command", "Get EURUSD price", ColorScheme.INFO),
            PanelItem("price BTCUSD", "price BTCUSD", "command", "Get BTCUSD price", ColorScheme.INFO),
            PanelItem("ohlc ETH_USDT", "ohlc ETH_USDT 1h", "command", "Get OHLC data", ColorScheme.INFO),
            PanelItem("config show", "config show", "command", "Show configuration", ColorScheme.INFO),
            PanelItem("config set", "config set", "command", "Modify configuration", ColorScheme.WARNING),
        ]
        return commands
    
    def _load_tests_and_tasks(self) -> List[PanelItem]:
        """Load tests from pytest and tasks from Robot Framework"""
        items = []
        
        # Add pytest tests
        pytest_tests = [
            PanelItem("Python Tests", "pytest -m python -v", "test", "Run Python module tests", ColorScheme.SUCCESS),
            PanelItem("Go Tests", "pytest -m go -v", "test", "Run Go module tests", ColorScheme.SUCCESS),
            PanelItem("Rust Tests", "pytest -m rust -v", "test", "Run Rust module tests", ColorScheme.SUCCESS),
            PanelItem("C++ Tests", "pytest -m cpp -v", "test", "Run C++ module tests", ColorScheme.SUCCESS),
            PanelItem("ZMQ Tests", "pytest -m zmq -v", "test", "Run ZMQ integration tests", ColorScheme.INFO),
            PanelItem("Performance Tests", "pytest -m performance -v", "test", "Run performance benchmarks", ColorScheme.WARNING),
        ]
        
        # Add Robot Framework tasks
        rf_tasks = [
            PanelItem("RF: Deployment", "robot deployment.robot", "task", "Run deployment suite", ColorScheme.INFO),
            PanelItem("RF: Gateways", "robot gateways.robot", "task", "Run gateway tests", ColorScheme.INFO),
            PanelItem("RF: Data Ops", "robot data_operations.robot", "task", "Run data operations", ColorScheme.INFO),
            PanelItem("RF: Integration", "robot multilang_integration.robot", "task", "Run integration tests", ColorScheme.INFO),
        ]
        
        items.extend(pytest_tests)
        items.extend(rf_tasks)
        
        return items
    
    def _render_header(self):
        """Render header with title and status"""
        header = "╔" + "═" * (self.width - 2) + "╗"
        self.stdscr.attron(curses.color_pair(ColorScheme.PRIMARY.value) | curses.A_BOLD)
        self.stdscr.addstr(0, 0, header)
        
        title = "MARKET DATA COMMANDER TERMINAL"
        title_centered = title.center(self.width - 2)
        self.stdscr.addstr(0, 1, title_centered)
        
        self.stdscr.attroff(curses.color_pair(ColorScheme.PRIMARY.value) | curses.A_BOLD)
    
    def _render_footer(self):
        """Render footer with function key help"""
        footer_y = self.height - 1
        
        self.stdscr.attron(curses.color_pair(ColorScheme.PRIMARY.value))
        
        # Function key hints
        hints = "F1:Help  F4:Execute  F5:Copy  F9:Exit"
        self.stdscr.addstr(footer_y, 0, hints.ljust(self.width))
        
        self.stdscr.attroff(curses.color_pair(ColorScheme.PRIMARY.value))
    
    def _render_separators(self):
        """Render visual separators between panels"""
        sep_x = self.width // 2
        
        for y in range(1, self.height - 3):
            self.stdscr.attron(curses.color_pair(ColorScheme.INFO.value))
            self.stdscr.addstr(y, sep_x, "│")
            self.stdscr.attroff(curses.color_pair(ColorScheme.INFO.value))
    
    def render(self):
        """Render entire terminal interface"""
        self.stdscr.clear()
        
        # Render structure
        self._render_header()
        self._render_separators()
        
        # Render panels
        self.left_panel.render(self.stdscr)
        self.right_panel.render(self.stdscr)
        
        # Render dashboard
        self.dashboard.render(self.stdscr)
        
        # Render footer
        self._render_footer()
        
        # Render message if any
        if self.message:
            msg_y = self.height - 2
            self.stdscr.attron(curses.color_pair(ColorScheme.INFO.value))
            self.stdscr.addstr(msg_y, 1, self.message[:self.width - 2])
            self.stdscr.attroff(curses.color_pair(ColorScheme.INFO.value))
        
        self.stdscr.refresh()
    
    def handle_input(self):
        """Handle keyboard input"""
        try:
            key = self.stdscr.getch()
            if key == -1:
                return
            
            # Arrow keys
            if key == curses.KEY_UP:
                active = self.left_panel if self.active_panel == 0 else self.right_panel
                active.move_selection(-1)
                self.message = f"Selection moved up in {active.title}"
            
            elif key == curses.KEY_DOWN:
                active = self.left_panel if self.active_panel == 0 else self.right_panel
                active.move_selection(1)
                self.message = f"Selection moved down in {active.title}"
            
            elif key == curses.KEY_LEFT:
                self.active_panel = 0
                self.message = "Switched to left panel (Commands)"
            
            elif key == curses.KEY_RIGHT:
                self.active_panel = 1
                self.message = "Switched to right panel (Tests & Tasks)"
            
            # Tab key - switch panels
            elif key == ord('\t'):
                self.active_panel = 1 - self.active_panel
                panel_name = self.left_panel.title if self.active_panel == 0 else self.right_panel.title
                self.message = f"Active panel: {panel_name}"
            
            # Enter - Execute selected
            elif key == ord('\n'):
                self._execute_selected()
            
            # Function keys
            elif key == curses.KEY_F1:
                self._show_help()
            elif key == curses.KEY_F2:
                self.message = "Refreshing..."
                self._setup_panels()
            elif key == curses.KEY_F4:
                self._execute_selected()
            elif key == curses.KEY_F9 or key == ord('q'):
                self.running = False
                self.message = "Exiting..."
            
            # Search (Ctrl+S)
            elif key == ord('s') - ord('@'):
                self._search_items()
            
            # Refresh (Ctrl+R)
            elif key == ord('r') - ord('@'):
                self._setup_panels()
                self.message = "Panels refreshed"
            
        except Exception as e:
            self.message = f"Error: {str(e)[:50]}"
    
    def _execute_selected(self):
        """Execute selected command or test"""
        active = self.left_panel if self.active_panel == 0 else self.right_panel
        item = active.get_selected_item()
        
        if not item:
            self.message = "No item selected"
            return
        
        self.message = f"Executing: {item.name}..."
        self.stdscr.refresh()
        
        start_time = time.time()
        
        try:
            # Execute command
            if item.item_type == "command":
                result = subprocess.run(
                    item.command,
                    shell=True,
                    capture_output=True,
                    timeout=30,
                    text=True
                )
                duration = time.time() - start_time
                status = "✓" if result.returncode == 0 else "✗"
                self.dashboard.add_execution(item.name, status, duration)
                self.message = f"{item.name}: {status} ({duration:.2f}s)"
            
            elif item.item_type == "test":
                result = subprocess.run(
                    item.command,
                    shell=True,
                    capture_output=True,
                    timeout=60,
                    text=True,
                    cwd=Path(__file__).parent.parent / "testing"
                )
                duration = time.time() - start_time
                status = "✓" if result.returncode == 0 else "✗"
                self.dashboard.add_execution(item.name, status, duration)
                self.message = f"{item.name}: {status} ({duration:.2f}s)"
            
            elif item.item_type == "task":
                result = subprocess.run(
                    item.command,
                    shell=True,
                    capture_output=True,
                    timeout=60,
                    text=True,
                    cwd=Path(__file__).parent.parent / "testing"
                )
                duration = time.time() - start_time
                status = "✓" if result.returncode == 0 else "✗"
                self.dashboard.add_execution(item.name, status, duration)
                self.message = f"{item.name}: {status} ({duration:.2f}s)"
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.dashboard.add_execution(item.name, "⏱", duration)
            self.message = f"{item.name}: Timeout ({duration:.2f}s)"
        except Exception as e:
            duration = time.time() - start_time
            self.dashboard.add_execution(item.name, "✗", duration)
            self.message = f"Error: {str(e)[:40]}"
    
    def _show_help(self):
        """Show help information"""
        self.message = "↑↓=Navigate  ←→/Tab=Switch Panels  Enter=Execute  Ctrl+S=Search  F9=Exit"
    
    def _search_items(self):
        """Search items in active panel"""
        self.message = "Search functionality coming soon..."
    
    def run(self):
        """Main event loop"""
        # Set non-blocking input
        self.stdscr.timeout(50)
        
        try:
            while self.running:
                self.render()
                self.handle_input()
        except KeyboardInterrupt:
            pass


def main(stdscr):
    """Main entry point"""
    # Use default terminal colors
    curses.use_default_colors()
    
    terminal = CommanderTerminal(stdscr)
    terminal.run()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nTerminal closed")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
