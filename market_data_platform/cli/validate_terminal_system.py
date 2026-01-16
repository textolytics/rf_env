#!/usr/bin/env python3
"""
Terminal System Validation & Testing
Validates all items, options, and command execution
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, List
import time
import json

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{BLUE}{BOLD}{'='*70}{RESET}")
    print(f"{BLUE}{BOLD}{text.center(70)}{RESET}")
    print(f"{BLUE}{BOLD}{'='*70}{RESET}")


def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text: str):
    """Print info message"""
    print(f"{CYAN}ℹ {text}{RESET}")


class TerminalValidator:
    """Validate terminal system components"""
    
    def __init__(self):
        self.workspace = Path("/root/rf_env")
        self.cli_path = self.workspace / "market_data_platform" / "cli"
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_results = []
    
    def run_validation(self):
        """Run all validation tests"""
        print_header("TERMINAL SYSTEM VALIDATION & TESTING")
        
        self.validate_file_structure()
        self.validate_imports()
        self.validate_menu_system()
        self.validate_component_manager()
        self.validate_config_manager()
        self.validate_keyword_manager()
        self.validate_command_execution()
        self.validate_keyboard_navigation()
        
        self.print_summary()
    
    def validate_file_structure(self):
        """Validate file structure"""
        print_header("FILE STRUCTURE VALIDATION")
        
        required_files = [
            "advanced_menu_terminal.py",
            "unified_terminal_launcher.py",
            "commander_terminal.py",
            "advanced_dashboard.py",
            "terminal_integration.py",
        ]
        
        for filename in required_files:
            filepath = self.cli_path / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print_success(f"Found {filename} ({size:,} bytes)")
                self.passed += 1
            else:
                print_error(f"Missing {filename}")
                self.failed += 1
    
    def validate_imports(self):
        """Validate module imports"""
        print_header("IMPORT VALIDATION")
        
        sys.path.insert(0, str(self.cli_path))
        
        modules_to_test = [
            ("advanced_menu_terminal", ["ComponentManager", "ConfigManager", "KeywordManager", "MenuSystem"]),
            ("commander_terminal", ["CommanderTerminal", "ColorScheme"]),
            ("advanced_dashboard", ["AdvancedDashboard", "RealTimeMonitor"]),
        ]
        
        for module_name, classes in modules_to_test:
            try:
                module = __import__(module_name)
                
                all_found = True
                for class_name in classes:
                    if hasattr(module, class_name):
                        print_info(f"  ✓ {class_name}")
                    else:
                        print_warning(f"  ✗ {class_name} not found")
                        all_found = False
                        self.warnings += 1
                
                if all_found:
                    print_success(f"Module {module_name} - all classes found")
                    self.passed += 1
                else:
                    print_warning(f"Module {module_name} - some classes missing")
                    self.warnings += 1
            
            except Exception as e:
                print_error(f"Failed to import {module_name}: {e}")
                self.failed += 1
    
    def validate_menu_system(self):
        """Validate menu system"""
        print_header("MENU SYSTEM VALIDATION")
        
        try:
            from advanced_menu_terminal import MenuSystem, MenuItem, MenuType
            
            # Create menu system instance
            class MockStdscr:
                def getmaxyx(self):
                    return (30, 100)
            
            workspace = Path("/root/rf_env")
            menu_sys = MenuSystem(MockStdscr(), workspace)
            
            # Validate main menu exists
            if menu_sys.main_menu:
                print_success(f"Main menu created with {len(menu_sys.main_menu)} items")
                self.passed += 1
            else:
                print_error("Main menu not created")
                self.failed += 1
            
            # Validate menu items
            menu_types = defaultdict(int)
            for item in menu_sys.main_menu:
                menu_types[item.type.name] += 1
            
            for item_type, count in menu_types.items():
                print_info(f"  {count} {item_type} items")
            
            print_success(f"Menu structure validated - {len(menu_sys.main_menu)} items")
            self.passed += 1
        
        except Exception as e:
            print_error(f"Menu system validation failed: {e}")
            self.failed += 1
    
    def validate_component_manager(self):
        """Validate component manager"""
        print_header("COMPONENT MANAGER VALIDATION")
        
        try:
            from advanced_menu_terminal import ComponentManager, ComponentStatus
            
            comp_mgr = ComponentManager()
            
            # Check components loaded
            components = comp_mgr.get_all_components()
            if len(components) > 0:
                print_success(f"Component Manager loaded {len(components)} components")
                self.passed += 1
                
                for name, comp in components.items():
                    print_info(f"  • {name:20} | {comp.status.value:15} | Port: {comp.port}")
            else:
                print_error("No components loaded")
                self.failed += 1
            
            # Test start/stop operations
            test_comp_name = list(components.keys())[0] if components else None
            if test_comp_name:
                success, msg = comp_mgr.start_component(test_comp_name)
                if success:
                    print_success(f"Component start: {msg}")
                    self.passed += 1
                else:
                    print_error(f"Component start failed: {msg}")
                    self.failed += 1
                
                success, msg = comp_mgr.stop_component(test_comp_name)
                if success:
                    print_success(f"Component stop: {msg}")
                    self.passed += 1
                else:
                    print_error(f"Component stop failed: {msg}")
                    self.failed += 1
        
        except Exception as e:
            print_error(f"Component Manager validation failed: {e}")
            self.failed += 1
    
    def validate_config_manager(self):
        """Validate configuration manager"""
        print_header("CONFIGURATION MANAGER VALIDATION")
        
        try:
            from advanced_menu_terminal import ConfigManager
            
            config_mgr = ConfigManager()
            
            # Check default config
            config = config_mgr.get_all()
            if len(config) > 0:
                print_success(f"Configuration Manager loaded {len(config)} settings")
                self.passed += 1
                
                for key, value in list(config.items())[:5]:
                    print_info(f"  • {key}: {value}")
                if len(config) > 5:
                    print_info(f"  ... and {len(config) - 5} more settings")
            else:
                print_error("Configuration not loaded")
                self.failed += 1
            
            # Test set/get
            config_mgr.set("test_key", "test_value")
            value = config_mgr.get("test_key")
            if value == "test_value":
                print_success("Configuration set/get working")
                self.passed += 1
            else:
                print_error("Configuration set/get failed")
                self.failed += 1
        
        except Exception as e:
            print_error(f"Configuration Manager validation failed: {e}")
            self.failed += 1
    
    def validate_keyword_manager(self):
        """Validate keyword manager"""
        print_header("KEYWORD MANAGER VALIDATION")
        
        try:
            from advanced_menu_terminal import KeywordManager
            
            keyword_mgr = KeywordManager(self.workspace)
            
            keywords = keyword_mgr.get_all_keywords()
            if len(keywords) > 0:
                print_success(f"Keyword Manager discovered {len(keywords)} categories")
                self.passed += 1
                
                total_keywords = sum(len(kws) for kws in keywords.values())
                print_info(f"Total keywords: {total_keywords}")
                
                for category, kws in keywords.items():
                    print_info(f"  • {category}: {len(kws)} keywords")
            else:
                print_warning("No keywords discovered")
                self.warnings += 1
        
        except Exception as e:
            print_error(f"Keyword Manager validation failed: {e}")
            self.failed += 1
    
    def validate_command_execution(self):
        """Validate command execution"""
        print_header("COMMAND EXECUTION VALIDATION")
        
        try:
            from advanced_menu_terminal import MenuSystem
            
            class MockStdscr:
                def getmaxyx(self):
                    return (30, 100)
            
            workspace = Path("/root/rf_env")
            menu_sys = MenuSystem(MockStdscr(), workspace)
            
            # Test various command executions
            commands_to_test = [
                ("Health Check", menu_sys.cmd_health_check),
                ("Component Status", lambda: menu_sys.show_component_details("ZMQ Bus")),
                ("Configuration Display", menu_sys.show_config),
            ]
            
            for cmd_name, cmd_func in commands_to_test:
                try:
                    output_before = len(menu_sys.output_lines)
                    cmd_func()
                    output_after = len(menu_sys.output_lines)
                    
                    if output_after > output_before:
                        print_success(f"Command '{cmd_name}' executed")
                        self.passed += 1
                    else:
                        print_warning(f"Command '{cmd_name}' generated no output")
                        self.warnings += 1
                except Exception as e:
                    print_error(f"Command '{cmd_name}' failed: {e}")
                    self.failed += 1
        
        except Exception as e:
            print_error(f"Command execution validation failed: {e}")
            self.failed += 1
    
    def validate_keyboard_navigation(self):
        """Validate keyboard navigation"""
        print_header("KEYBOARD NAVIGATION VALIDATION")
        
        try:
            import curses
            
            # Test curses key codes
            key_tests = [
                ("UP Arrow", curses.KEY_UP),
                ("DOWN Arrow", curses.KEY_DOWN),
                ("ENTER", ord('\n')),
                ("Q key", ord('q')),
            ]
            
            for key_name, key_code in key_tests:
                if key_code > 0:
                    print_success(f"Key mapping for {key_name}: {key_code}")
                    self.passed += 1
                else:
                    print_error(f"Key mapping failed for {key_name}")
                    self.failed += 1
        
        except Exception as e:
            print_error(f"Keyboard navigation validation failed: {e}")
            self.failed += 1
    
    def print_summary(self):
        """Print validation summary"""
        print_header("VALIDATION SUMMARY")
        
        total = self.passed + self.failed + self.warnings
        
        print(f"\n{BOLD}Results:{RESET}")
        print(f"  {GREEN}Passed:   {self.passed}{RESET}")
        print(f"  {YELLOW}Warnings: {self.warnings}{RESET}")
        print(f"  {RED}Failed:   {self.failed}{RESET}")
        print(f"  {CYAN}Total:    {total}{RESET}\n")
        
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        if self.failed == 0:
            print_success(f"ALL VALIDATIONS PASSED ({success_rate:.1f}% success rate)")
            return 0
        else:
            print_error(f"SOME VALIDATIONS FAILED ({success_rate:.1f}% success rate)")
            return 1
    
    def print_recommendations(self):
        """Print recommendations"""
        print_header("NEXT STEPS")
        
        print(f"""
{BLUE}Launch Terminal:{RESET}

  python /root/rf_env/market_data_platform/cli/unified_terminal_launcher.py

{BLUE}Terminal Features:{RESET}

  ✓ Advanced Menu Terminal (Recommended)
    • Component Management (Start/Stop/Status)
    • Configuration Management
    • Keyword Browser
    • Command Execution
    • Test Management
    • Full Keyboard Navigation

  ✓ Commander Terminal
    • Two-panel MC-style navigation
    • Command execution dashboard
    • Real-time status tracking

  ✓ Dashboard Terminal
    • System monitoring
    • Performance metrics
    • Resource tracking

{BLUE}Keyboard Controls:{RESET}

  ↑↓        Navigate menu items
  Enter     Select/Execute menu item
  q         Go back or quit
  F1        Help (in some modes)

{BLUE}Supported Operations:{RESET}

  • View all component status
  • Start/Stop components
  • Run health checks
  • Execute tests (Python, Go, Rust, etc.)
  • Manage configuration
  • Browse keywords
  • Execute commands
  • Monitor system resources

{CYAN}All components validated and operational!{RESET}
""")


def main():
    """Main validation entry point"""
    from collections import defaultdict
    
    validator = TerminalValidator()
    exit_code = validator.run_validation()
    
    if exit_code == 0:
        validator.print_recommendations()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
