#!/usr/bin/env python3
"""
Unified Terminal System Launcher
Integrates all terminal modes with comprehensive feature set
"""

import curses
import sys
import subprocess
from pathlib import Path
from typing import Optional
import time

# Import all terminal modules
sys.path.insert(0, str(Path(__file__).parent))

from advanced_menu_terminal import main as menu_terminal_main


def print_banner():
    """Print welcome banner"""
    print(r"""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║       MARKET DATA TERMINAL - UNIFIED COMPONENT MANAGEMENT SYSTEM      ║
║                                                                        ║
║              Midnight Commander Style with Dropdown Menus             ║
║         Component Management • Testing • Config • Keywords            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

AVAILABLE TERMINAL MODES:

  1. Advanced Menu Terminal (RECOMMENDED)
     • Dropdown menus with component management
     • Start/Stop components
     • Configuration management
     • Keyword browser
     • Execute commands
     • Status monitoring
     • Full keyboard navigation (↑↓ Enter q)

  2. Commander Terminal
     • Two-panel MC-style navigation
     • Command and test execution
     • Quick access dashboard

  3. Dashboard Terminal
     • System monitoring
     • Real-time metrics
     • Performance tracking

  q. Quit

""")


def select_mode() -> str:
    """Interactive mode selection"""
    while True:
        choice = input("Select terminal mode (1-3, q to quit): ").strip().lower()
        
        if choice in ('1', '2', '3', 'q'):
            return choice
        
        print("Invalid selection. Please try again.")


def launch_menu_terminal():
    """Launch advanced menu terminal"""
    print("\nLaunching Advanced Menu Terminal...")
    print("Controls: ↑↓ to navigate, Enter to select, q to go back/quit\n")
    time.sleep(1)
    
    try:
        curses.wrapper(menu_terminal_main)
    except KeyboardInterrupt:
        print("\nTerminal interrupted")
    except Exception as e:
        print(f"Error: {e}")


def launch_commander_terminal():
    """Launch commander terminal"""
    print("\nLaunching Commander Terminal...")
    try:
        result = subprocess.run(
            ["python", str(Path(__file__).parent / "enhanced_terminal_launcher.py"), 
             "--mode", "commander"],
            cwd=Path(__file__).parent.parent.parent
        )
    except Exception as e:
        print(f"Error launching Commander Terminal: {e}")


def launch_dashboard_terminal():
    """Launch dashboard terminal"""
    print("\nLaunching Dashboard Terminal...")
    try:
        result = subprocess.run(
            ["python", str(Path(__file__).parent / "enhanced_terminal_launcher.py"),
             "--mode", "dashboard"],
            cwd=Path(__file__).parent.parent.parent
        )
    except Exception as e:
        print(f"Error launching Dashboard Terminal: {e}")


def main():
    """Main launcher"""
    print_banner()
    
    while True:
        mode = select_mode()
        
        if mode == '1':
            launch_menu_terminal()
        elif mode == '2':
            launch_commander_terminal()
        elif mode == '3':
            launch_dashboard_terminal()
        elif mode == 'q':
            print("Exiting Market Data Terminal...")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTerminal system exited.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
