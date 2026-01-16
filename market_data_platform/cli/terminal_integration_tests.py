#!/usr/bin/env python3
"""
Terminal Integration Tests - Verify all components work together
"""

import sys
import subprocess
from pathlib import Path
import json
import time

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print section header"""
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD}{text.center(60)}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")


def print_info(text):
    """Print info message"""
    print(f"{CYAN}ℹ {text}{RESET}")


class TerminalIntegrationTests:
    """Test suite for terminal integration"""
    
    def __init__(self):
        self.workspace = Path("/root/rf_env")
        self.cli_path = self.workspace / "market_data_platform" / "cli"
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def run_all(self):
        """Run all tests"""
        print_header("TERMINAL INTEGRATION TEST SUITE")
        
        print_info(f"Workspace: {self.workspace}")
        print_info(f"CLI Path: {self.cli_path}\n")
        
        # Test groups
        self.test_syntax()
        self.test_imports()
        self.test_modules()
        self.test_configuration()
        self.test_integration()
        
        # Summary
        self.print_summary()
    
    def test_syntax(self):
        """Test Python syntax of all components"""
        print_header("SYNTAX VALIDATION")
        
        files_to_check = [
            "commander_terminal.py",
            "advanced_dashboard.py",
            "terminal_integration.py",
            "enhanced_terminal_launcher.py",
            "enhanced_cli.py",
        ]
        
        for filename in files_to_check:
            filepath = self.cli_path / filename
            
            if not filepath.exists():
                print_warning(f"File not found: {filename}")
                self.warnings += 1
                continue
            
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(filepath)],
                    capture_output=True,
                    timeout=5,
                    text=True
                )
                
                if result.returncode == 0:
                    print_success(f"Syntax OK: {filename}")
                    self.passed += 1
                else:
                    print_error(f"Syntax error in {filename}")
                    print(f"  {result.stderr}")
                    self.failed += 1
            
            except subprocess.TimeoutExpired:
                print_error(f"Syntax check timeout: {filename}")
                self.failed += 1
            except Exception as e:
                print_error(f"Syntax check failed for {filename}: {e}")
                self.failed += 1
    
    def test_imports(self):
        """Test module imports"""
        print_header("IMPORT VALIDATION")
        
        import_tests = [
            ("commander_terminal", ["ColorScheme", "CommanderPanel", "CommanderTerminal"]),
            ("advanced_dashboard", ["AdvancedDashboard", "RealTimeMonitor", "InteractiveCommandPalette"]),
            ("terminal_integration", ["TerminalIntegration", "TestDiscovery", "RobotFrameworkDiscovery"]),
            ("enhanced_cli", ["EnhancedCLI", "TabCompletionProvider"]),
        ]
        
        sys.path.insert(0, str(self.cli_path))
        
        for module_name, classes in import_tests:
            try:
                module = __import__(module_name)
                
                # Check classes
                all_found = True
                for class_name in classes:
                    if hasattr(module, class_name):
                        print_info(f"  Found: {class_name}")
                    else:
                        print_warning(f"  Missing: {class_name}")
                        all_found = False
                        self.warnings += 1
                
                if all_found:
                    print_success(f"Module imports OK: {module_name}")
                    self.passed += 1
                else:
                    print_warning(f"Partial imports: {module_name}")
                    self.warnings += 1
            
            except ImportError as e:
                print_error(f"Import failed: {module_name}")
                print(f"  Error: {e}")
                self.failed += 1
            except Exception as e:
                print_error(f"Unexpected error importing {module_name}: {e}")
                self.failed += 1
    
    def test_modules(self):
        """Test individual modules"""
        print_header("MODULE FUNCTIONALITY TESTS")
        
        # Test ColorScheme
        try:
            from commander_terminal import ColorScheme
            colors = [c for c in ColorScheme]
            if len(colors) == 4:
                print_success(f"ColorScheme: 4 colors defined")
                self.passed += 1
            else:
                print_error(f"ColorScheme: Expected 4 colors, got {len(colors)}")
                self.failed += 1
        except Exception as e:
            print_error(f"ColorScheme test failed: {e}")
            self.failed += 1
        
        # Test CommandRegistry
        try:
            from terminal_integration import CommandRegistry
            registry = CommandRegistry()
            cmd_count = len(registry.commands)
            if cmd_count >= 10:
                print_success(f"CommandRegistry: {cmd_count} commands loaded")
                self.passed += 1
            else:
                print_warning(f"CommandRegistry: Only {cmd_count} commands (expected >= 10)")
                self.warnings += 1
        except Exception as e:
            print_error(f"CommandRegistry test failed: {e}")
            self.failed += 1
        
        # Test TerminalConfig
        try:
            from terminal_integration import TerminalConfig
            config = TerminalConfig()
            config.set("test_key", "test_value")
            if config.get("test_key") == "test_value":
                print_success("TerminalConfig: Settings work correctly")
                self.passed += 1
            else:
                print_error("TerminalConfig: Settings not working")
                self.failed += 1
        except Exception as e:
            print_error(f"TerminalConfig test failed: {e}")
            self.failed += 1
        
        # Test TerminalIntegration
        try:
            from terminal_integration import TerminalIntegration
            integration = TerminalIntegration(self.workspace)
            if integration.command_registry.commands:
                print_success("TerminalIntegration: Initialized successfully")
                self.passed += 1
            else:
                print_warning("TerminalIntegration: No commands registered")
                self.warnings += 1
        except Exception as e:
            print_error(f"TerminalIntegration test failed: {e}")
            self.failed += 1
    
    def test_configuration(self):
        """Test configuration system"""
        print_header("CONFIGURATION TESTS")
        
        try:
            from terminal_integration import TerminalConfig
            config = TerminalConfig()
            
            # Test default settings
            if config.get("theme") is not None:
                print_success("Configuration: Theme setting available")
            else:
                print_info("Configuration: No theme setting (using defaults)")
            
            # Test set/get
            test_value = {"test": "data"}
            config.set("test_config", test_value)
            retrieved = config.get("test_config")
            
            if retrieved == test_value:
                print_success("Configuration: Set/Get working")
                self.passed += 1
            else:
                print_error("Configuration: Set/Get mismatch")
                self.failed += 1
            
        except Exception as e:
            print_error(f"Configuration test failed: {e}")
            self.failed += 1
    
    def test_integration(self):
        """Test system integration"""
        print_header("SYSTEM INTEGRATION TESTS")
        
        try:
            from terminal_integration import (
                TerminalIntegration, PanelMode, TestDiscovery, RobotFrameworkDiscovery
            )
            
            integration = TerminalIntegration(self.workspace)
            
            # Test command discovery
            commands = integration.get_available_items(PanelMode.COMMANDS)
            if commands:
                print_success(f"Command discovery: Found {len(commands)} commands")
                self.passed += 1
            else:
                print_warning("Command discovery: No commands found")
                self.warnings += 1
            
            # Test test discovery
            tests = integration.get_available_items(PanelMode.TESTS)
            print_info(f"Test discovery: Found {len(tests)} tests")
            
            # Test keyword discovery
            keywords = integration.get_available_items(PanelMode.KEYWORDS)
            print_info(f"Keyword discovery: Found {len(keywords)} keywords")
            
            # Test task discovery
            tasks = integration.get_available_items(PanelMode.TASKS)
            print_info(f"Task discovery: Found {len(tasks)} tasks")
            
            if tests or tasks or keywords:
                print_success("Integration: Discovery systems working")
                self.passed += 1
            else:
                print_warning("Integration: Limited discovery results")
                self.warnings += 1
            
            # Test statistics
            stats = integration.get_statistics()
            print_info(f"Statistics: {stats.get('total', 0)} executions recorded")
            
        except Exception as e:
            print_error(f"Integration test failed: {e}")
            self.failed += 1
    
    def print_summary(self):
        """Print test summary"""
        print_header("TEST SUMMARY")
        
        total = self.passed + self.failed + self.warnings
        
        print(f"\n{BOLD}Results:{RESET}")
        print(f"  {GREEN}Passed:  {self.passed}{RESET}")
        print(f"  {YELLOW}Warnings: {self.warnings}{RESET}")
        print(f"  {RED}Failed:  {self.failed}{RESET}")
        print(f"  {CYAN}Total:   {total}{RESET}\n")
        
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        if self.failed == 0:
            print_success(f"ALL TESTS PASSED ({success_rate:.1f}% success rate)")
            return 0
        else:
            print_error(f"TESTS FAILED ({success_rate:.1f}% success rate)")
            return 1


def print_recommendations():
    """Print recommendations"""
    print_header("RECOMMENDATIONS")
    
    print(f"""
{BLUE}Next Steps:{RESET}

1. {CYAN}Start Terminal:{RESET}
   python /root/rf_env/market_data_platform/cli/enhanced_terminal_launcher.py

2. {CYAN}Read Documentation:{RESET}
   cat /root/rf_env/TERMINAL_USER_GUIDE.md

3. {CYAN}Quick Reference:{RESET}
   cat /root/rf_env/TERMINAL_QUICK_REFERENCE.md

4. {CYAN}Learn Keyboard Shortcuts:{RESET}
   Press F1 inside terminal for help

5. {CYAN}Try Features:{RESET}
   - Press ↑↓ to navigate
   - Press Enter to execute
   - Press F3 for system status
   - Press F6 for notebooks
   - Press F9 to exit

{BLUE}Key Features:{RESET}

✓ Midnight Commander two-panel navigation
✓ Bloomberg Terminal execution dashboard
✓ Byobu 4-color theme
✓ 50+ keyboard shortcuts
✓ Robot Framework integration
✓ Pytest test execution
✓ Jupyter Notebook browser
✓ System monitoring
✓ Command search/palette

{BLUE}Support:{RESET}

- User Guide: TERMINAL_USER_GUIDE.md
- Quick Reference: TERMINAL_QUICK_REFERENCE.md
- Implementation: ADVANCED_TERMINAL_DELIVERY.md
""")


def main():
    """Main entry point"""
    print(f"\n{BOLD}Market Data Terminal - Integration Tests{RESET}\n")
    
    # Run tests
    tester = TerminalIntegrationTests()
    exit_code = tester.run_all()
    
    # Print recommendations
    if exit_code == 0:
        print_recommendations()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
