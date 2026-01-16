#!/usr/bin/env python3
"""
Simple Terminal Component Test - No Curses Required
Tests all components, configuration, and keywords without terminal rendering
"""

import sys
from pathlib import Path
import json
import time

# Add CLI to path
CLI_PATH = Path("/root/rf_env/market_data_platform/cli")
sys.path.insert(0, str(CLI_PATH))

# Colors for console output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BLUE}{BOLD}{'='*70}{RESET}")
    print(f"{BLUE}{BOLD}{text.center(70)}{RESET}")
    print(f"{BLUE}{BOLD}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text):
    print(f"{CYAN}ℹ {text}{RESET}")

def test_components():
    """Test ComponentManager"""
    print_header("TEST 1: COMPONENT MANAGER")
    
    try:
        from advanced_menu_terminal import ComponentManager, ComponentStatus
        
        mgr = ComponentManager()
        components = mgr.get_all_components()
        
        print_success(f"ComponentManager created with {len(components)} components:")
        for name, comp in components.items():
            print_info(f"  • {name}")
            print_info(f"    - Description: {comp.description}")
            print_info(f"    - Port: {comp.port if comp.port > 0 else 'N/A'}")
            print_info(f"    - Status: {comp.status.value}")
        
        # Test start/stop
        comp_name = list(components.keys())[0]
        print_info(f"\nTesting start/stop on '{comp_name}':")
        
        success, msg = mgr.start_component(comp_name)
        print_info(f"  Start result: {msg}")
        
        success, msg = mgr.stop_component(comp_name)
        print_info(f"  Stop result: {msg}")
        
        return True
    except Exception as e:
        print_error(f"ComponentManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test ConfigManager"""
    print_header("TEST 2: CONFIGURATION MANAGER")
    
    try:
        from advanced_menu_terminal import ConfigManager
        
        mgr = ConfigManager()
        config = mgr.get_all()
        
        print_success(f"ConfigManager created with {len(config)} settings:")
        
        # Show first few settings
        for i, (key, value) in enumerate(config.items()):
            if i < 5:
                print_info(f"  • {key}: {value}")
            elif i == 5:
                print_info(f"  ... and {len(config) - 5} more settings")
                break
        
        # Test set/get
        mgr.set("test_key", "test_value")
        value = mgr.get("test_key")
        
        if value == "test_value":
            print_success("Configuration set/get working")
        else:
            print_error("Configuration set/get failed")
            return False
        
        # Test reset
        mgr.reset_config()
        print_success("Configuration reset successful")
        
        return True
    except Exception as e:
        print_error(f"ConfigManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_keywords():
    """Test KeywordManager"""
    print_header("TEST 3: KEYWORD MANAGER")
    
    try:
        from advanced_menu_terminal import KeywordManager
        
        workspace = Path("/root/rf_env")
        mgr = KeywordManager(workspace)
        keywords = mgr.get_all_keywords()
        
        print_success(f"KeywordManager discovered {len(keywords)} categories:")
        
        total_keywords = 0
        for category, kws in keywords.items():
            print_info(f"  • {category}: {len(kws)} keywords")
            total_keywords += len(kws)
            # Show first 2 keywords per category
            for kw in list(kws)[:2]:
                print_info(f"    - {kw}")
            if len(kws) > 2:
                print_info(f"    - ... and {len(kws) - 2} more")
        
        print_success(f"Total: {total_keywords} keywords")
        return True
    except Exception as e:
        print_error(f"KeywordManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_menu_structure():
    """Test MenuSystem"""
    print_header("TEST 4: MENU SYSTEM")
    
    try:
        # Mock curses for testing
        class MockStdscr:
            def getmaxyx(self):
                return (30, 100)
        
        from advanced_menu_terminal import MenuSystem, MenuItem, MenuType
        
        menu_sys = MenuSystem(MockStdscr(), Path("/root/rf_env"))
        
        print_success(f"MenuSystem created with main menu:")
        print_info(f"  • Total items in main menu: {len(menu_sys.main_menu)}")
        
        # Count by type
        type_counts = {}
        for item in menu_sys.main_menu:
            item_type = item.type.name
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        print_info(f"  • Item breakdown:")
        for item_type, count in sorted(type_counts.items()):
            print_info(f"    - {item_type}: {count}")
        
        # List all menu items
        print_info(f"\n  • Menu Items:")
        for i, item in enumerate(menu_sys.main_menu, 1):
            if item.type.name == "HEADER":
                print_info(f"    {i:2}. [{item.type.name:7}] {item.label}")
            elif item.type.name == "DIVIDER":
                print_info(f"    {i:2}. {'-'*60}")
            else:
                print_info(f"    {i:2}. [{item.type.name:7}] {item.label}")
        
        return True
    except Exception as e:
        print_error(f"MenuSystem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_launcher():
    """Test unified launcher exists"""
    print_header("TEST 5: UNIFIED LAUNCHER")
    
    try:
        launcher_path = CLI_PATH / "unified_terminal_launcher.py"
        
        if launcher_path.exists():
            size = launcher_path.stat().st_size
            print_success(f"Unified launcher found ({size:,} bytes)")
            print_info(f"  • Path: {launcher_path}")
            print_info(f"  • Purpose: Integrates all terminal modes")
            return True
        else:
            print_error(f"Unified launcher not found at {launcher_path}")
            return False
    except Exception as e:
        print_error(f"Unified launcher test failed: {e}")
        return False

def test_all_files():
    """Test all required files exist"""
    print_header("TEST 6: FILE STRUCTURE")
    
    required_files = {
        "advanced_menu_terminal.py": "Advanced dropdown menu system",
        "commander_terminal.py": "Midnight Commander two-panel terminal",
        "advanced_dashboard.py": "System dashboard and monitoring",
        "terminal_integration.py": "Integration utilities",
        "unified_terminal_launcher.py": "Mode selector and launcher",
    }
    
    all_found = True
    for filename, description in required_files.items():
        filepath = CLI_PATH / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print_success(f"{filename:35} ({size:6,} bytes) - {description}")
        else:
            print_error(f"{filename:35} MISSING - {description}")
            all_found = False
    
    return all_found

def main():
    """Run all tests"""
    print(f"\n{BOLD}{BLUE}╔════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{BLUE}║           TERMINAL SYSTEM COMPONENT VALIDATION TEST               ║{RESET}")
    print(f"{BOLD}{BLUE}╚════════════════════════════════════════════════════════════════════╝{RESET}")
    
    results = []
    results.append(("File Structure", test_all_files()))
    results.append(("Components", test_components()))
    results.append(("Configuration", test_config()))
    results.append(("Keywords", test_keywords()))
    results.append(("Menu System", test_menu_structure()))
    results.append(("Unified Launcher", test_unified_launcher()))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"  {status} - {test_name}")
    
    print(f"\n{BOLD}Results: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ ALL TESTS PASSED!{RESET}")
        print(f"\n{CYAN}Launch terminal with:{RESET}")
        print(f"  python {CLI_PATH}/unified_terminal_launcher.py\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ SOME TESTS FAILED{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
