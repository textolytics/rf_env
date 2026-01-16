#!/usr/bin/env python3
"""
Terminal Integration Module - Unified CLI Interface
Orchestrates all terminal components with Robot Framework integration
"""

import curses
from pathlib import Path
from typing import Dict, List, Optional, Callable
from enum import Enum
import json
import subprocess
import time


class PanelMode(Enum):
    """Terminal panel modes"""
    COMMANDS = "commands"
    TESTS = "tests"
    TASKS = "tasks"
    KEYWORDS = "keywords"
    SYSTEM = "system"
    NOTEBOOKS = "notebooks"
    HELP = "help"


class TerminalConfig:
    """Terminal configuration"""
    
    def __init__(self):
        self.config_file = Path.home() / ".robotmcp_terminal.json"
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict:
        """Load settings from config file"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_settings(self):
        """Save settings to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass
    
    def get(self, key: str, default=None):
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        """Set setting value"""
        self.settings[key] = value


class RobotFrameworkDiscovery:
    """Discover Robot Framework resources"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.tests: Dict[str, List[str]] = {}
        self.keywords: Dict[str, List[Dict]] = {}
        self.resources: Dict[str, List[str]] = {}
        self._discover()
    
    def _discover(self):
        """Discover RF resources"""
        self._discover_tests()
        self._discover_keywords()
        self._discover_resources()
    
    def _discover_tests(self):
        """Discover RF test files"""
        for robot_file in self.workspace_root.glob("**/*.robot"):
            try:
                with open(robot_file) as f:
                    content = f.read()
                    test_names = self._extract_test_names(content)
                    if test_names:
                        self.tests[robot_file.name] = test_names
            except Exception:
                pass
    
    def _discover_keywords(self):
        """Discover custom keywords"""
        for py_file in self.workspace_root.glob("**/*.py"):
            if "test" in py_file.name or "keyword" in py_file.name:
                try:
                    with open(py_file) as f:
                        content = f.read()
                        keywords = self._extract_keywords(content)
                        if keywords:
                            self.keywords[py_file.name] = keywords
                except Exception:
                    pass
    
    def _discover_resources(self):
        """Discover resource files"""
        for resource_file in self.workspace_root.glob("**/*.resource"):
            try:
                with open(resource_file) as f:
                    content = f.read()
                    resource_keywords = self._extract_test_names(content)
                    if resource_keywords:
                        self.resources[resource_file.name] = resource_keywords
            except Exception:
                pass
    
    @staticmethod
    def _extract_test_names(content: str) -> List[str]:
        """Extract test names from RF file"""
        tests = []
        in_test_section = False
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith("*** Test Cases ***"):
                in_test_section = True
                continue
            
            if line.startswith("***"):
                in_test_section = False
                continue
            
            if in_test_section and line and not line.startswith("#"):
                if not line.startswith("    "):
                    tests.append(line)
        
        return tests
    
    @staticmethod
    def _extract_keywords(content: str) -> List[Dict]:
        """Extract keywords from Python file"""
        keywords = []
        import re
        
        # Pattern for def keyword_name or @keyword decorator
        pattern = r"def\s+(\w+)\s*\([^)]*\).*?:(?:\s+\"\"\"([^\"]*)\"\"\")?"
        
        for match in re.finditer(pattern, content, re.DOTALL):
            keywords.append({
                "name": match.group(1),
                "doc": match.group(2) or ""
            })
        
        return keywords


class TestDiscovery:
    """Discover and manage tests"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.pytest_tests: List[Dict] = []
        self.robot_tests: List[Dict] = []
        self._discover()
    
    def _discover(self):
        """Discover tests"""
        self._discover_pytest()
        self._discover_robot()
    
    def _discover_pytest(self):
        """Discover pytest tests"""
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q"],
                cwd=self.workspace_root,
                capture_output=True,
                timeout=10,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if "::" in line:
                    self.pytest_tests.append({
                        "name": line.strip(),
                        "type": "pytest",
                        "file": line.split("::")[0] if "::" in line else ""
                    })
        except Exception:
            pass
    
    def _discover_robot(self):
        """Discover Robot Framework tests"""
        try:
            result = subprocess.run(
                ["robot", "--dryrun", "--collect-only"],
                cwd=self.workspace_root,
                capture_output=True,
                timeout=10,
                text=True
            )
            
            # Parse RF test output
            for line in result.stdout.split('\n'):
                if "test" in line.lower():
                    self.robot_tests.append({
                        "name": line.strip(),
                        "type": "robot",
                    })
        except Exception:
            pass
    
    def get_all_tests(self) -> List[Dict]:
        """Get all discovered tests"""
        return self.pytest_tests + self.robot_tests


class CommandRegistry:
    """Register and manage available commands"""
    
    def __init__(self):
        self.commands: Dict[str, Dict] = {}
        self._load_default_commands()
    
    def _load_default_commands(self):
        """Load default commands"""
        default_commands = {
            "install_all": {
                "description": "Install all dependencies",
                "category": "setup",
                "command": "pip install -r requirements.txt",
            },
            "start_services": {
                "description": "Start all services",
                "category": "services",
                "command": "docker-compose up -d",
            },
            "stop_services": {
                "description": "Stop all services",
                "category": "services",
                "command": "docker-compose down",
            },
            "health_check": {
                "description": "Run health check",
                "category": "monitoring",
                "command": "curl -s http://localhost:8080/health",
            },
            "connect_gateio": {
                "description": "Connect to Gate.io",
                "category": "connectivity",
                "command": "python -c 'from market_data_platform.connectivity import GateIOConnector; GateIOConnector().test_connection()'",
            },
            "connect_oanda": {
                "description": "Connect to OANDA",
                "category": "connectivity",
                "command": "python -c 'from market_data_platform.connectivity import OANDAConnector; OANDAConnector().test_connection()'",
            },
            "run_regression_tests": {
                "description": "Run all regression tests",
                "category": "testing",
                "command": "pytest market_data_platform/testing/regression_tests.py -v",
            },
            "run_robot_tests": {
                "description": "Run Robot Framework tests",
                "category": "testing",
                "command": "robot market_data_platform/testing/",
            },
            "monitor_zmq": {
                "description": "Monitor ZMQ bus",
                "category": "monitoring",
                "command": "python zmq/forwarder_device.py",
            },
            "logs_tail": {
                "description": "Tail application logs",
                "category": "monitoring",
                "command": "tail -f /tmp/market_data.log",
            },
        }
        
        for cmd_name, cmd_info in default_commands.items():
            self.register_command(cmd_name, cmd_info)
    
    def register_command(self, name: str, info: Dict):
        """Register a command"""
        self.commands[name] = info
    
    def get_command(self, name: str) -> Optional[Dict]:
        """Get command info"""
        return self.commands.get(name)
    
    def get_commands_by_category(self, category: str) -> Dict[str, Dict]:
        """Get commands by category"""
        return {
            name: info for name, info in self.commands.items()
            if info.get("category") == category
        }
    
    def execute_command(self, name: str, timeout: int = 30) -> tuple:
        """Execute a command"""
        cmd_info = self.get_command(name)
        if not cmd_info:
            return False, f"Command not found: {name}"
        
        try:
            result = subprocess.run(
                cmd_info["command"],
                shell=True,
                capture_output=True,
                timeout=timeout,
                text=True
            )
            return result.returncode == 0, result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, str(e)


class TerminalState:
    """Manage terminal UI state"""
    
    def __init__(self):
        self.current_panel = PanelMode.COMMANDS
        self.search_active = False
        self.search_query = ""
        self.selected_items = {}
        self.execution_history = []
        self.focus_stack = []
    
    def push_panel(self, panel: PanelMode):
        """Push panel onto focus stack"""
        self.focus_stack.append(self.current_panel)
        self.current_panel = panel
    
    def pop_panel(self):
        """Pop panel from focus stack"""
        if self.focus_stack:
            self.current_panel = self.focus_stack.pop()
    
    def set_search_active(self, active: bool):
        """Set search state"""
        self.search_active = active
        if not active:
            self.search_query = ""
    
    def add_execution_record(self, item: str, status: str, duration: float):
        """Record execution"""
        self.execution_history.append({
            "item": item,
            "status": status,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # Keep only last 100 records
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]


class TerminalIntegration:
    """Main terminal integration layer"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.config = TerminalConfig()
        self.state = TerminalState()
        self.command_registry = CommandRegistry()
        self.rf_discovery = RobotFrameworkDiscovery(workspace_root)
        self.test_discovery = TestDiscovery(workspace_root)
    
    def get_available_items(self, mode: PanelMode) -> List[Dict]:
        """Get available items for a panel mode"""
        if mode == PanelMode.COMMANDS:
            return [
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "category": info.get("category", ""),
                    "command": info.get("command", ""),
                }
                for name, info in self.command_registry.commands.items()
            ]
        
        elif mode == PanelMode.TESTS:
            return self.test_discovery.get_all_tests()
        
        elif mode == PanelMode.KEYWORDS:
            keywords = []
            for file_name, kw_list in self.rf_discovery.keywords.items():
                keywords.extend([
                    {"name": kw["name"], "doc": kw["doc"], "file": file_name}
                    for kw in kw_list
                ])
            return keywords
        
        elif mode == PanelMode.TASKS:
            tasks = []
            for file_name, test_list in self.rf_discovery.tests.items():
                tasks.extend([
                    {"name": test, "file": file_name, "type": "task"}
                    for test in test_list
                ])
            return tasks
        
        return []
    
    def execute_item(self, item: Dict, mode: PanelMode) -> tuple:
        """Execute an item"""
        start_time = time.time()
        
        try:
            if mode == PanelMode.COMMANDS:
                success, output = self.command_registry.execute_command(item.get("name", ""))
            
            elif mode == PanelMode.TESTS:
                test_name = item.get("name", "")
                success, output = self._run_test(test_name)
            
            elif mode == PanelMode.TASKS:
                task_name = item.get("name", "")
                success, output = self._run_task(task_name)
            
            else:
                success, output = False, "Unknown item type"
            
            duration = time.time() - start_time
            status = "✓" if success else "✗"
            
            self.state.add_execution_record(
                item.get("name", "unknown"),
                status,
                duration
            )
            
            return success, output
        
        except Exception as e:
            duration = time.time() - start_time
            self.state.add_execution_record(
                item.get("name", "unknown"),
                "✗",
                duration
            )
            return False, str(e)
    
    def _run_test(self, test_name: str) -> tuple:
        """Run a pytest test"""
        try:
            result = subprocess.run(
                ["pytest", "-v", test_name],
                cwd=self.workspace_root,
                capture_output=True,
                timeout=60,
                text=True
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def _run_task(self, task_name: str) -> tuple:
        """Run a Robot Framework task"""
        try:
            result = subprocess.run(
                ["robot", "-t", task_name, "."],
                cwd=self.workspace_root,
                capture_output=True,
                timeout=60,
                text=True
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get execution history"""
        return self.state.execution_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        history = self.state.execution_history
        
        if not history:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
            }
        
        total = len(history)
        successful = sum(1 for h in history if h["status"] == "✓")
        failed = total - successful
        avg_duration = sum(h["duration"] for h in history) / total
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "avg_duration": avg_duration,
        }
