#!/usr/bin/env python3
"""
Enhanced Terminal Launcher - Unified CLI Experience
Integrates Commander Terminal with advanced dashboard, test discovery, and RF integration
"""

import curses
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import time
from datetime import datetime

# Add imports for integration
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from commander_terminal import (
    ColorScheme, PanelItem, CommanderPanel, ExecutionDashboard, CommanderTerminal
)

try:
    from advanced_dashboard import (
        AdvancedDashboard, RealTimeMonitor, ExecutionMetric, SystemStatus,
        InteractiveCommandPalette, NotebookCellBrowser, HelpPanel
    )
    from terminal_integration import (
        TerminalIntegration, PanelMode, TerminalConfig, TerminalState,
        TestDiscovery, RobotFrameworkDiscovery, CommandRegistry
    )
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Integration modules not fully available: {e}")
    INTEGRATION_AVAILABLE = False


class EnhancedTerminalLauncher:
    """Unified terminal launcher with advanced features"""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.terminal: Optional[CommanderTerminal] = None
        self.dashboard: Optional[AdvancedDashboard] = None
        self.integration: Optional[TerminalIntegration] = None
        self.command_palette: Optional[InteractiveCommandPalette] = None
        self.help_panel = HelpPanel()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all terminal components"""
        if INTEGRATION_AVAILABLE:
            try:
                self.integration = TerminalIntegration(self.workspace_root)
                print(f"✓ Terminal integration initialized")
                print(f"  - Found {len(self.integration.test_discovery.get_all_tests())} tests")
                print(f"  - Found {len(self.integration.rf_discovery.keywords)} keyword files")
                print(f"  - Registered {len(self.integration.command_registry.commands)} commands")
                time.sleep(1.5)
            except Exception as e:
                print(f"⚠ Integration initialization warning: {e}")
                time.sleep(1)
    
    def run(self):
        """Run the enhanced terminal"""
        try:
            curses.wrapper(self._main)
        except KeyboardInterrupt:
            print("\nTerminal interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"Terminal error: {e}")
            sys.exit(1)
    
    def _main(self, stdscr):
        """Main terminal event loop"""
        height, width = stdscr.getmaxyx()
        
        # Initialize colors
        ColorScheme.init_colors()
        
        # Create enhanced terminal instance
        terminal = CommanderTerminal(stdscr, self.workspace_root)
        
        # Inject integration data if available
        if self.integration:
            self._inject_integration_data(terminal)
        
        # Main loop
        running = True
        while running:
            try:
                # Handle input and rendering
                terminal.render()
                terminal.handle_input()
                
                # Small delay for responsiveness
                time.sleep(0.05)
                
            except KeyboardInterrupt:
                running = False
            except curses.error:
                pass
            except Exception as e:
                stdscr.addstr(0, 0, f"Error: {str(e)[:50]}")
    
    def _inject_integration_data(self, terminal: 'CommanderTerminal'):
        """Inject integration data into terminal"""
        try:
            # Update command panel with discovered commands
            command_items = []
            for cmd_name, cmd_info in self.integration.command_registry.commands.items():
                command_items.append(PanelItem(
                    name=cmd_name.replace("_", " ").title(),
                    command=cmd_name,
                    item_type="command",
                    description=cmd_info.get("description", ""),
                    color=ColorScheme.SUCCESS.value
                ))
            
            if command_items:
                terminal.left_panel.items = command_items[:terminal.left_panel.items_height]
            
            # Update test panel with discovered tests
            test_items = []
            
            # Add pytest tests
            for test in self.integration.test_discovery.pytest_tests[:5]:
                test_items.append(PanelItem(
                    name=test.get("name", "")[:30],
                    command=test.get("name", ""),
                    item_type="test",
                    description="Pytest test",
                    color=ColorScheme.SUCCESS.value
                ))
            
            # Add Robot Framework tests
            for file_name, tests in self.integration.rf_discovery.tests.items():
                for test in tests[:3]:
                    test_items.append(PanelItem(
                        name=f"{file_name}: {test}"[:30],
                        command=test,
                        item_type="task",
                        description="Robot Framework test",
                        color=ColorScheme.INFO.value
                    ))
            
            if test_items:
                terminal.right_panel.items = test_items[:terminal.right_panel.items_height]
        
        except Exception as e:
            pass  # Silently fail if integration data injection fails


class TerminalModeSelector:
    """Select terminal mode before launch"""
    
    MODES = {
        "1": {
            "name": "Standard Commander",
            "description": "Two-panel MC-style navigation",
            "launcher": lambda ws: CommanderTerminalLauncher(ws),
        },
        "2": {
            "name": "Advanced Dashboard",
            "description": "System monitoring with metrics",
            "launcher": lambda ws: AdvancedDashboardLauncher(ws),
        },
        "3": {
            "name": "Integrated Terminal",
            "description": "Full features with RF integration",
            "launcher": lambda ws: EnhancedTerminalLauncher(ws),
        },
    }
    
    @staticmethod
    def select():
        """Show mode selection menu"""
        print("\n" + "="*60)
        print("MARKET DATA TERMINAL LAUNCHER".center(60))
        print("="*60)
        print("\nSelect Terminal Mode:\n")
        
        for key, mode_info in TerminalModeSelector.MODES.items():
            print(f"  {key}. {mode_info['name']}")
            print(f"     {mode_info['description']}\n")
        
        print(f"  q. Quit\n")
        
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == "q":
            sys.exit(0)
        
        if choice not in TerminalModeSelector.MODES:
            print("Invalid choice")
            return TerminalModeSelector.select()
        
        return choice
    
    @staticmethod
    def launch(workspace_root: Optional[Path] = None):
        """Launch selected terminal mode"""
        if workspace_root is None:
            workspace_root = Path.cwd()
        
        mode_choice = TerminalModeSelector.select()
        mode_info = TerminalModeSelector.MODES[mode_choice]
        
        print(f"\nLaunching {mode_info['name']}...")
        print("Initializing components...")
        
        launcher = mode_info["launcher"](workspace_root)
        launcher.run()


class CommanderTerminalLauncher:
    """Launcher for standard Commander terminal"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
    
    def run(self):
        """Run standard Commander terminal"""
        try:
            from commander_terminal import main
            curses.wrapper(main)
        except Exception as e:
            print(f"Error running Commander terminal: {e}")
            sys.exit(1)


class AdvancedDashboardLauncher:
    """Launcher for advanced dashboard"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
    
    def run(self):
        """Run advanced dashboard"""
        if not INTEGRATION_AVAILABLE:
            print("Advanced dashboard requires integration modules")
            sys.exit(1)
        
        try:
            curses.wrapper(self._main)
        except KeyboardInterrupt:
            print("\nDashboard interrupted")
        except Exception as e:
            print(f"Dashboard error: {e}")
            sys.exit(1)
    
    def _main(self, stdscr):
        """Main dashboard loop"""
        height, width = stdscr.getmaxyx()
        
        # Initialize colors
        ColorScheme.init_colors()
        
        # Create dashboard
        dashboard = AdvancedDashboard(height, width)
        
        # Simulate some data
        test_names = [
            "test_python_modules",
            "test_cpp_modules",
            "test_rust_modules",
            "test_go_modules",
        ]
        
        # Main loop
        running = True
        iteration = 0
        
        while running:
            try:
                stdscr.clear()
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(0, 0, "ADVANCED MONITORING DASHBOARD".center(width))
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
                
                # Render panels
                left_width = width // 2 - 1
                right_width = width // 2 - 1
                
                dashboard.render_system_stats(stdscr, 2, 0, left_width)
                dashboard.render_metrics_panel(stdscr, 2, width // 2 + 1, right_width)
                dashboard.render_recent_executions(stdscr, 7, 0, width, height - 10)
                
                # Add test data
                if iteration % 20 == 0:
                    test = test_names[iteration % len(test_names)]
                    dashboard.log_execution(test, "✓", 2.3 + (iteration % 5) * 0.5)
                
                stdscr.addstr(height - 2, 0, "Press 'q' to quit, arrow keys to navigate".center(width))
                stdscr.refresh()
                
                # Handle input
                stdscr.timeout(500)
                ch = stdscr.getch()
                
                if ch == ord('q'):
                    running = False
                
                iteration += 1
            
            except curses.error:
                pass
            except KeyboardInterrupt:
                running = False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Market Data Terminal - Advanced CLI Interface"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory"
    )
    parser.add_argument(
        "--mode",
        choices=["commander", "dashboard", "integrated", "select"],
        default="select",
        help="Terminal mode to launch"
    )
    parser.add_argument(
        "--no-integration",
        action="store_true",
        help="Disable integration modules"
    )
    
    args = parser.parse_args()
    
    if args.no_integration:
        global INTEGRATION_AVAILABLE
        INTEGRATION_AVAILABLE = False
    
    # Check workspace
    if not args.workspace.exists():
        print(f"Error: Workspace not found: {args.workspace}")
        sys.exit(1)
    
    # Launch terminal
    if args.mode == "select":
        TerminalModeSelector.launch(args.workspace)
    elif args.mode == "integrated":
        launcher = EnhancedTerminalLauncher(args.workspace)
        launcher.run()
    elif args.mode == "dashboard":
        launcher = AdvancedDashboardLauncher(args.workspace)
        launcher.run()
    elif args.mode == "commander":
        launcher = CommanderTerminalLauncher(args.workspace)
        launcher.run()


if __name__ == "__main__":
    main()
