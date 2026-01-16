#!/usr/bin/env python3
"""
Advanced Terminal Dashboard - System Monitoring & Real-time Status
Multi-panel execution tracking with color-coded status display
"""

import curses
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import subprocess
from pathlib import Path


class StatusIndicator(Enum):
    """Status indicators for dashboard"""
    RUNNING = "◉"      # Running
    SUCCESS = "✓"      # Success
    FAILED = "✗"       # Failed
    PENDING = "○"      # Pending
    WARNING = "⚠"      # Warning
    INFO = "ℹ"        # Information


@dataclass
class ExecutionMetric:
    """Execution metrics and statistics"""
    total_executed: int = 0
    successful: int = 0
    failed: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    last_execution_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_executed == 0:
            return 0.0
        return (self.successful / self.total_executed) * 100


@dataclass
class SystemStatus:
    """Real-time system status"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_tests: int = 0
    queued_tasks: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class RealTimeMonitor:
    """Monitor system resources and test execution"""
    
    def __init__(self):
        self.system_status = SystemStatus()
        self.metrics = ExecutionMetric()
        self.running = True
        self.lock = threading.Lock()
    
    def update_metrics(self, success: bool, duration: float):
        """Update execution metrics"""
        with self.lock:
            self.metrics.total_executed += 1
            if success:
                self.metrics.successful += 1
            else:
                self.metrics.failed += 1
            
            self.metrics.total_duration += duration
            self.metrics.avg_duration = (
                self.metrics.total_duration / self.metrics.total_executed
            )
            self.metrics.last_execution_time = datetime.now()
    
    def update_system_status(self):
        """Update system resource usage"""
        try:
            import psutil
            with self.lock:
                self.system_status.cpu_usage = psutil.cpu_percent(interval=0.1)
                self.system_status.memory_usage = psutil.virtual_memory().percent
                self.system_status.disk_usage = psutil.disk_usage('/').percent
                self.system_status.timestamp = datetime.now()
        except ImportError:
            pass


class AdvancedDashboard:
    """Multi-panel dashboard with real-time monitoring"""
    
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.monitor = RealTimeMonitor()
        self.execution_log: List[Dict[str, Any]] = []
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def log_execution(self, test_name: str, status: str, duration: float, output: str = ""):
        """Log test execution"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "test": test_name,
            "status": status,
            "duration": duration,
            "output": output[:200]
        }
        self.execution_log.append(entry)
        
        # Keep only last 50 executions
        if len(self.execution_log) > 50:
            self.execution_log = self.execution_log[-50:]
        
        # Update test results
        if test_name not in self.test_results:
            self.test_results[test_name] = {
                "executions": [],
                "success_count": 0,
                "failure_count": 0
            }
        
        self.test_results[test_name]["executions"].append({
            "status": status,
            "duration": duration,
            "timestamp": datetime.now()
        })
        
        if status == "✓":
            self.test_results[test_name]["success_count"] += 1
            self.monitor.update_metrics(True, duration)
        else:
            self.test_results[test_name]["failure_count"] += 1
            self.monitor.update_metrics(False, duration)
    
    def render_system_stats(self, window, y: int, x: int, width: int):
        """Render system statistics panel"""
        self.monitor.update_system_status()
        
        window.attron(curses.color_pair(1) | curses.A_BOLD)
        title = "SYSTEM STATUS"
        window.addstr(y, x, title.center(width))
        window.attroff(curses.color_pair(1) | curses.A_BOLD)
        
        # CPU usage
        cpu_bar = self._render_progress_bar(
            self.monitor.system_status.cpu_usage,
            width - 15
        )
        window.addstr(y + 1, x, f"CPU  [{cpu_bar}] {self.monitor.system_status.cpu_usage:5.1f}%")
        
        # Memory usage
        mem_bar = self._render_progress_bar(
            self.monitor.system_status.memory_usage,
            width - 15
        )
        window.addstr(y + 2, x, f"MEM  [{mem_bar}] {self.monitor.system_status.memory_usage:5.1f}%")
        
        # Disk usage
        disk_bar = self._render_progress_bar(
            self.monitor.system_status.disk_usage,
            width - 15
        )
        window.addstr(y + 3, x, f"DISK [{disk_bar}] {self.monitor.system_status.disk_usage:5.1f}%")
    
    def render_metrics_panel(self, window, y: int, x: int, width: int):
        """Render execution metrics panel"""
        window.attron(curses.color_pair(2) | curses.A_BOLD)
        title = "EXECUTION METRICS"
        window.addstr(y, x, title.center(width))
        window.attroff(curses.color_pair(2) | curses.A_BOLD)
        
        metrics = self.monitor.metrics
        
        window.addstr(y + 1, x, f"Total:     {metrics.total_executed:4d} executions")
        window.addstr(y + 2, x, f"Success:   {metrics.successful:4d} ({metrics.success_rate:.1f}%)")
        window.addstr(y + 3, x, f"Failed:    {metrics.failed:4d}")
        window.addstr(y + 4, x, f"Avg Time:  {metrics.avg_duration:6.2f}s")
    
    def render_recent_executions(self, window, y: int, x: int, width: int, height: int):
        """Render recent execution log"""
        window.attron(curses.color_pair(3) | curses.A_BOLD)
        title = "RECENT EXECUTIONS"
        window.addstr(y, x, title.center(width))
        window.attroff(curses.color_pair(3) | curses.A_BOLD)
        
        line_y = y + 1
        max_lines = height - 2
        
        for entry in self.execution_log[-max_lines:]:
            if line_y >= y + height - 1:
                break
            
            status_icon = entry["status"]
            test_name = entry["test"][:20]
            duration = entry["duration"]
            
            line = f"{status_icon} {test_name:20} {duration:6.2f}s"
            window.addstr(line_y, x, line[:width - 1])
            line_y += 1
    
    def render_test_summary(self, window, y: int, x: int, width: int, height: int):
        """Render test summary"""
        window.attron(curses.color_pair(4) | curses.A_BOLD)
        title = "TEST SUMMARY"
        window.addstr(y, x, title.center(width))
        window.attroff(curses.color_pair(4) | curses.A_BOLD)
        
        line_y = y + 1
        max_lines = height - 2
        
        for test_name, result in list(self.test_results.items())[-max_lines:]:
            if line_y >= y + height - 1:
                break
            
            total = result["success_count"] + result["failure_count"]
            success_rate = (result["success_count"] / total * 100) if total > 0 else 0
            
            status = "✓" if success_rate == 100 else "⚠" if success_rate >= 80 else "✗"
            
            line = f"{status} {test_name[:15]:15} {success_rate:5.1f}%"
            window.addstr(line_y, x, line[:width - 1])
            line_y += 1
    
    @staticmethod
    def _render_progress_bar(value: float, width: int) -> str:
        """Render progress bar"""
        filled = int((value / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar


class InteractiveCommandPalette:
    """Command palette for quick command execution"""
    
    def __init__(self, commands: Dict[str, str]):
        self.commands = commands
        self.filtered_commands: List[Tuple[str, str]] = list(commands.items())
        self.selected_idx = 0
        self.search_text = ""
    
    def update_filter(self, query: str):
        """Filter commands by search query"""
        self.search_text = query.lower()
        if not query:
            self.filtered_commands = list(self.commands.items())
        else:
            self.filtered_commands = [
                (name, cmd) for name, cmd in self.commands.items()
                if query.lower() in name.lower()
            ]
        self.selected_idx = 0
    
    def move_selection(self, direction: int):
        """Move selection"""
        new_idx = self.selected_idx + direction
        if 0 <= new_idx < len(self.filtered_commands):
            self.selected_idx = new_idx
    
    def get_selected(self) -> Optional[Tuple[str, str]]:
        """Get selected command"""
        if 0 <= self.selected_idx < len(self.filtered_commands):
            return self.filtered_commands[self.selected_idx]
        return None
    
    def render(self, window, y: int, x: int, width: int, height: int):
        """Render command palette"""
        window.attron(curses.color_pair(1) | curses.A_BOLD)
        window.addstr(y, x, "COMMAND PALETTE".center(width))
        window.attroff(curses.color_pair(1) | curses.A_BOLD)
        
        # Search box
        search_line = f"Search: {self.search_text}|"
        window.addstr(y + 1, x, search_line[:width - 1])
        
        # Command list
        line_y = y + 2
        max_lines = height - 3
        
        for i, (name, cmd) in enumerate(self.filtered_commands[:max_lines]):
            is_selected = (i == self.selected_idx)
            
            if is_selected:
                window.attron(curses.color_pair(2) | curses.A_REVERSE)
            else:
                window.attron(curses.color_pair(3))
            
            display = f"▶ {name[:30]:30} {cmd[:20]:20}"
            window.addstr(line_y, x, display[:width - 1])
            
            window.attroff(
                (curses.color_pair(2) | curses.A_REVERSE)
                if is_selected else curses.color_pair(3)
            )
            
            line_y += 1


class NotebookCellBrowser:
    """Browse and execute IPython notebook cells"""
    
    def __init__(self, notebook_path: Path):
        self.notebook_path = notebook_path
        self.cells: List[Dict[str, Any]] = []
        self.selected_idx = 0
        self._load_cells()
    
    def _load_cells(self):
        """Load cells from notebook"""
        try:
            with open(self.notebook_path) as f:
                notebook = json.load(f)
                self.cells = notebook.get("cells", [])
        except Exception as e:
            print(f"Error loading notebook: {e}")
    
    def get_selected_cell(self) -> Optional[Dict[str, Any]]:
        """Get selected cell"""
        if 0 <= self.selected_idx < len(self.cells):
            return self.cells[self.selected_idx]
        return None
    
    def move_selection(self, direction: int):
        """Move selection"""
        new_idx = self.selected_idx + direction
        if 0 <= new_idx < len(self.cells):
            self.selected_idx = new_idx
    
    def execute_cell(self) -> Tuple[bool, str]:
        """Execute selected cell"""
        cell = self.get_selected_cell()
        if not cell or cell.get("cell_type") != "code":
            return False, "Not a code cell"
        
        try:
            code = "".join(cell.get("source", []))
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                timeout=10,
                text=True
            )
            return result.returncode == 0, result.stdout or result.stderr
        except Exception as e:
            return False, str(e)
    
    def render(self, window, y: int, x: int, width: int, height: int):
        """Render notebook browser"""
        window.attron(curses.color_pair(1) | curses.A_BOLD)
        window.addstr(y, x, "NOTEBOOK CELLS".center(width))
        window.attroff(curses.color_pair(1) | curses.A_BOLD)
        
        line_y = y + 1
        max_lines = height - 2
        
        for i, cell in enumerate(self.cells[:max_lines]):
            if line_y >= y + height - 1:
                break
            
            is_selected = (i == self.selected_idx)
            cell_type = cell.get("cell_type", "unknown")
            
            # Get cell content preview
            if cell_type == "code":
                source = "".join(cell.get("source", []))[:30]
            else:
                source = "".join(cell.get("source", []))[:30]
            
            if is_selected:
                window.attron(curses.color_pair(2) | curses.A_REVERSE)
            else:
                color = 3 if cell_type == "code" else 4
                window.attron(curses.color_pair(color))
            
            display = f"▶ [{cell_type:6}] {source:30}"
            window.addstr(line_y, x, display[:width - 1])
            
            window.attroff(
                (curses.color_pair(2) | curses.A_REVERSE)
                if is_selected else curses.color_pair(3 if cell_type == "code" else 4)
            )
            
            line_y += 1


class HelpPanel:
    """Context-sensitive help panel"""
    
    HELP_CONTENT = {
        "navigation": [
            "↑/↓ DOWN/UP - Navigate items",
            "←/→ LEFT/RIGHT - Switch panels",
            "TAB - Cycle through panels",
            "ENTER - Execute selected item",
        ],
        "execution": [
            "F4 - Execute selected",
            "Ctrl+E - Execute all tests",
            "Ctrl+X - Stop execution",
            "Ctrl+L - Clear execution log",
        ],
        "search": [
            "Ctrl+F - Open search",
            "Ctrl+S - Search in current panel",
            "ESC - Close search",
        ],
        "panels": [
            "F1 - Commands panel",
            "F2 - Tests & Tasks panel",
            "F3 - System status",
            "F6 - Notebook browser",
        ],
    }
    
    def __init__(self):
        self.current_section = "navigation"
    
    def render(self, window, y: int, x: int, width: int, height: int):
        """Render help panel"""
        window.attron(curses.color_pair(1) | curses.A_BOLD)
        title = f"HELP: {self.current_section.upper()}"
        window.addstr(y, x, title.center(width))
        window.attroff(curses.color_pair(1) | curses.A_BOLD)
        
        help_lines = self.HELP_CONTENT.get(self.current_section, [])
        
        line_y = y + 1
        for help_line in help_lines[:height - 2]:
            if line_y >= y + height - 1:
                break
            
            window.attron(curses.color_pair(3))
            window.addstr(line_y, x, help_line[:width - 1])
            window.attroff(curses.color_pair(3))
            
            line_y += 1
