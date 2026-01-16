#!/usr/bin/env python3
"""
Enhanced CLI with Tab Completion and Keyboard Navigation
Extends market_data_platform CLI with advanced input handling
"""

import cmd
import readline
import sys
from typing import List, Dict, Any, Optional
from enum import Enum


# ============================================================================
# Tab Completion Provider
# ============================================================================

class TabCompletionProvider:
    """Provides tab completion for CLI commands"""
    
    def __init__(self):
        self.command_keywords: Dict[str, List[str]] = {
            # Deployment commands
            "install": ["all", "influxdb", "grafana", "redis", "parquet", "--runtime", "docker", "podman", "lxc"],
            "start": ["all", "influxdb", "grafana", "redis", "parquet", "--runtime"],
            "stop": ["all", "influxdb", "grafana", "redis", "parquet"],
            "restart": ["all", "influxdb", "grafana", "redis", "parquet"],
            "deploy-docker": ["all", "influxdb", "grafana", "redis", "parquet"],
            "deploy-podman": ["all", "influxdb", "grafana", "redis", "parquet"],
            "deploy-lxc": ["all", "influxdb", "grafana", "redis", "parquet"],
            "logs": ["influxdb", "grafana", "redis", "parquet", "--lines"],
            "configure-service": ["influxdb", "grafana", "redis", "parquet"],
            "health-check": ["", "influxdb", "grafana", "redis", "parquet", "all"],
            
            # Gateway commands
            "connect": ["freedx", "gate.io", "oanda", "kraken", "betfair", "twitter"],
            "disconnect": ["freedx", "gate.io", "oanda", "kraken", "betfair", "twitter", "all"],
            "list-gateways": [],
            "gateway-status": ["freedx", "gate.io", "oanda", "kraken", "betfair", "twitter", "all"],
            "stream": ["freedx.eurusd", "gate.io.btcusd", "oanda.eurusd", "kraken.btcusd"],
            "stop-stream": ["all", "freedx", "gate.io", "oanda", "kraken"],
            "test-gateway": ["all", "freedx", "gate.io", "oanda", "kraken"],
            
            # Data commands
            "price": ["EURUSD", "GBPUSD", "BTCUSD", "ETHUSD", "BNBUSD"],
            "ohlc": ["EURUSD", "GBPUSD", "BTCUSD", "--timeframe", "1m", "5m", "1h", "4h", "1d"],
            "history": ["EURUSD", "GBPUSD", "BTCUSD", "--limit"],
            "orderbook": ["EURUSD", "GBPUSD", "BTCUSD"],
            "depth": ["EURUSD", "BTCUSD", "ETHUSD"],
            "export": ["json", "csv", "parquet"],
            "import": ["json", "csv", "parquet"],
            "query": [],
            "aggregate": ["EURUSD", "GBPUSD", "--period", "1d", "--function", "avg", "min", "max"],
            
            # Analytics commands
            "sentiment": ["crypto", "equity", "forex"],
            "correlation": ["EURUSD", "GBPUSD", "BTCUSD"],
            "indicators": ["EURUSD", "GBPUSD", "BTCUSD"],
            "backtest": [],
            "portfolio": [],
            "risk-analysis": [],
            "alert": ["set", "list", "delete"],
            
            # Admin commands
            "config": ["show", "set", "reset"],
            "settings": ["show", "update", "reset"],
            "backup": ["--full", "--database", "--config"],
            "restore": [],
            "upgrade": ["--check"],
            "security": ["status", "audit", "certificate", "firewall"],
            "performance": ["status", "optimize", "profile", "report"],
        }
        
        self.gateway_keywords: Dict[str, List[str]] = {
            "freedx": ["eurusd", "gbpusd", "gold"],
            "gate.io": ["btcusd", "ethusd", "bnbusd"],
            "oanda": ["eurusd", "gbpusd", "usdjpy"],
            "kraken": ["btcusd", "ethusd", "adausd"],
            "betfair": ["soccer", "tennis", "horse_racing"],
            "twitter": ["sentiment", "trends"],
        }
    
    def get_completions(self, text: str, state: int) -> Optional[str]:
        """Get completion for text at given state"""
        # Parse the line buffer to get command and partial argument
        line = readline.get_line_buffer()
        tokens = line.split()
        
        if not tokens:
            # Complete commands
            return self.complete_commands(text, state)
        
        command = tokens[0]
        
        # Get completions for this command
        if command in self.command_keywords:
            keywords = self.command_keywords[command]
            completions = [k for k in keywords if k.startswith(text)]
        else:
            completions = []
        
        if state < len(completions):
            return completions[state]
        
        return None
    
    def complete_commands(self, text: str, state: int) -> Optional[str]:
        """Complete command names"""
        all_commands = list(self.command_keywords.keys())
        matches = [cmd for cmd in all_commands if cmd.startswith(text)]
        
        if state < len(matches):
            return matches[state]
        
        return None
    
    def get_keywords_for_command(self, command: str) -> List[str]:
        """Get keyword suggestions for a command"""
        return self.command_keywords.get(command, [])
    
    def get_command_list(self) -> List[str]:
        """Get all available commands"""
        return list(self.command_keywords.keys())


# ============================================================================
# Keyboard Navigation Handler
# ============================================================================

class KeyboardNavigationHandler:
    """Handles keyboard navigation and shortcuts"""
    
    # Navigation states
    COMMAND_GROUP_DEPLOYMENT = 0
    COMMAND_GROUP_GATEWAYS = 1
    COMMAND_GROUP_DATA = 2
    COMMAND_GROUP_ANALYTICS = 3
    COMMAND_GROUP_ADMIN = 4
    
    def __init__(self):
        self.current_group = self.COMMAND_GROUP_DEPLOYMENT
        self.current_selection = 0
        
        # Define command groups with options
        self.groups = {
            self.COMMAND_GROUP_DEPLOYMENT: {
                "name": "🚀 Deployment & Installation",
                "commands": [
                    "install all",
                    "start all",
                    "stop all",
                    "restart all",
                    "status",
                    "logs",
                    "health-check",
                    "deploy-docker",
                    "deploy-podman",
                    "deploy-lxc",
                    "configure-service",
                ]
            },
            self.COMMAND_GROUP_GATEWAYS: {
                "name": "🔗 Gateway & Connection Management",
                "commands": [
                    "connect",
                    "disconnect",
                    "list-gateways",
                    "gateway-status",
                    "stream",
                    "stop-stream",
                    "test-gateway",
                ]
            },
            self.COMMAND_GROUP_DATA: {
                "name": "📊 Data & Market Operations",
                "commands": [
                    "price",
                    "ohlc",
                    "history",
                    "orderbook",
                    "depth",
                    "export",
                    "import",
                    "query",
                    "aggregate",
                ]
            },
            self.COMMAND_GROUP_ANALYTICS: {
                "name": "📈 Analytics & Analysis",
                "commands": [
                    "sentiment",
                    "correlation",
                    "indicators",
                    "backtest",
                    "portfolio",
                    "risk-analysis",
                    "alert",
                ]
            },
            self.COMMAND_GROUP_ADMIN: {
                "name": "⚙️ Administration & Config",
                "commands": [
                    "config",
                    "settings",
                    "backup",
                    "restore",
                    "upgrade",
                    "security",
                    "performance",
                    "help",
                    "exit",
                ]
            }
        }
    
    def move_group_next(self) -> Dict[str, Any]:
        """Move to next command group (Right arrow)"""
        self.current_group = (self.current_group + 1) % len(self.groups)
        self.current_selection = 0
        return self.get_current_state()
    
    def move_group_previous(self) -> Dict[str, Any]:
        """Move to previous command group (Left arrow)"""
        self.current_group = (self.current_group - 1) % len(self.groups)
        self.current_selection = 0
        return self.get_current_state()
    
    def move_selection_down(self) -> Dict[str, Any]:
        """Move selection down (Down arrow)"""
        group = self.groups[self.current_group]
        max_selection = len(group["commands"]) - 1
        self.current_selection = min(self.current_selection + 1, max_selection)
        return self.get_current_state()
    
    def move_selection_up(self) -> Dict[str, Any]:
        """Move selection up (Up arrow)"""
        self.current_selection = max(self.current_selection - 1, 0)
        return self.get_current_state()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current navigation state"""
        group = self.groups[self.current_group]
        selected_command = group["commands"][self.current_selection] if group["commands"] else ""
        
        return {
            "group": self.current_group,
            "group_name": group["name"],
            "selection": self.current_selection,
            "selected_command": selected_command,
            "total_commands": len(group["commands"]),
        }
    
    def get_navigation_display(self) -> str:
        """Get formatted navigation display"""
        state = self.get_current_state()
        
        group = self.groups[self.current_group]
        commands = group["commands"]
        
        display = f"\n{state['group_name']}\n"
        display += "=" * 50 + "\n"
        
        for i, cmd in enumerate(commands):
            if i == self.current_selection:
                display += f"▶ {cmd}\n"
            else:
                display += f"  {cmd}\n"
        
        display += "\nNavigate: ← → Up Down | Select: Enter | Help: ?\n"
        
        return display
    
    def jump_to_group(self, group_number: int) -> Dict[str, Any]:
        """Jump to specific group (Ctrl+1-5)"""
        if 0 <= group_number < len(self.groups):
            self.current_group = group_number
            self.current_selection = 0
        return self.get_current_state()


# ============================================================================
# Enhanced CLI with Tab Completion
# ============================================================================

class EnhancedCLI(cmd.Cmd):
    """Enhanced CLI with tab completion and keyboard navigation"""
    
    intro = "\n=== Enhanced Market Data Platform CLI ===\nPress Tab for completion, ? for help\n"
    prompt = "MDP> "
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize completion provider
        self.completion_provider = TabCompletionProvider()
        self.navigation_handler = KeyboardNavigationHandler()
        
        # Setup readline for tab completion
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")
        
        # Custom key bindings for arrow navigation
        self._setup_key_bindings()
    
    def _setup_key_bindings(self):
        """Setup custom key bindings"""
        # These would typically be set via .inputrc
        # For now, we'll handle them in the input parsing
        pass
    
    def complete(self, text: str, state: int) -> Optional[str]:
        """Tab completion"""
        return self.completion_provider.get_completions(text, state)
    
    def emptyline(self):
        """Handle empty input - show navigation menu"""
        print(self.navigation_handler.get_navigation_display())
    
    def do_navigate(self, arg):
        """
        navigate [right|left|up|down|1-5]
        Navigate command groups and options
        Examples:
          navigate right      - Next group
          navigate left       - Previous group
          navigate down       - Next command in group
          navigate up         - Previous command in group
          navigate 1          - Jump to Deployment group
          navigate 2          - Jump to Gateways group
        """
        commands = arg.split()
        
        if not commands:
            print(self.navigation_handler.get_navigation_display())
            return
        
        command = commands[0].lower()
        
        if command == "right":
            state = self.navigation_handler.move_group_next()
        elif command == "left":
            state = self.navigation_handler.move_group_previous()
        elif command == "down":
            state = self.navigation_handler.move_selection_down()
        elif command == "up":
            state = self.navigation_handler.move_selection_up()
        elif command in "12345":
            state = self.navigation_handler.jump_to_group(int(command) - 1)
        else:
            print("Invalid navigation command. Use: right, left, up, down, or 1-5")
            return
        
        print(self.navigation_handler.get_navigation_display())
    
    def do_select(self, arg):
        """
        select
        Execute the currently selected command
        """
        state = self.navigation_handler.get_current_state()
        selected = state["selected_command"]
        
        if selected:
            print(f"\nExecuting: {selected}")
            # Execute the command
            self.onecmd(selected)
        else:
            print("No command selected")
    
    def do_list_commands(self, arg):
        """
        list_commands
        List all available commands with descriptions
        """
        print("\n=== Available Commands ===\n")
        
        completion_provider = self.completion_provider
        
        for command in sorted(completion_provider.get_command_list()):
            keywords = completion_provider.get_keywords_for_command(command)
            keywords_str = " ".join(keywords[:3])  # Show first 3 options
            if len(keywords) > 3:
                keywords_str += f" ... (+{len(keywords) - 3} more)"
            
            print(f"  {command:<20} {keywords_str}")
    
    def do_keywords(self, arg):
        """
        keywords <command>
        Show all keyword suggestions for a command
        Example: keywords install
        """
        if not arg:
            print("Usage: keywords <command>")
            return
        
        keywords = self.completion_provider.get_keywords_for_command(arg)
        
        if keywords:
            print(f"\nKeywords for '{arg}':")
            for i, keyword in enumerate(keywords, 1):
                print(f"  {i}. {keyword}")
        else:
            print(f"No keywords found for '{arg}'")
    
    def do_groups(self, arg):
        """
        groups
        List all command groups
        """
        print("\n=== Command Groups ===\n")
        
        for group_id, group_info in self.navigation_handler.groups.items():
            group_num = group_id + 1
            print(f"[{group_num}] {group_info['name']}")
            print(f"    Commands: {len(group_info['commands'])}")
    
    def do_install(self, arg):
        """
        install [all|service] [--runtime docker|podman|lxc]
        Install services
        """
        print(f"Installing: {arg}")
    
    def do_start(self, arg):
        """
        start [all|service]
        Start services
        """
        print(f"Starting: {arg}")
    
    def do_stop(self, arg):
        """
        stop [all|service]
        Stop services
        """
        print(f"Stopping: {arg}")
    
    def do_status(self, arg):
        """
        status
        Show deployment status
        """
        print("Deployment Status:\n")
        print("  Runtime: Docker")
        print("  Services Running: InfluxDB, Grafana, Redis")
        print("  Health: All OK")
    
    def do_help(self, arg):
        """
        help [command]
        Show help for command or list all commands
        """
        if arg:
            super().do_help(arg)
        else:
            print("\n=== Help ===")
            print("Type 'help <command>' for specific help")
            print("Type 'list_commands' to see all commands")
            print("Type 'navigate' to see navigation menu")
            print("Type 'groups' to see command groups")
            print("Type 'keywords <cmd>' to see tab completion options")
            print("\nKeyboard Shortcuts:")
            print("  Tab          - Command/argument completion")
            print("  Type 'navigate right/left/up/down' - Navigate groups")
            print("  Type 'navigate 1-5' - Jump to group")
            print("  Type 'select' - Execute selected command")


# ============================================================================
# Pytest Integration Test
# ============================================================================

def test_tab_completion_provider():
    """Test tab completion provider"""
    provider = TabCompletionProvider()
    
    # Test command completion
    assert "install" in provider.get_command_list()
    assert "start" in provider.get_command_list()
    
    # Test keyword completion
    keywords = provider.get_keywords_for_command("install")
    assert "all" in keywords
    assert "influxdb" in keywords


def test_keyboard_navigation():
    """Test keyboard navigation handler"""
    nav = KeyboardNavigationHandler()
    
    # Test initial state
    state = nav.get_current_state()
    assert state["group"] == 0
    assert state["selection"] == 0
    
    # Test moving groups
    state = nav.move_group_next()
    assert state["group"] == 1
    
    state = nav.move_group_previous()
    assert state["group"] == 0
    
    # Test moving selection
    state = nav.move_selection_down()
    assert state["selection"] == 1
    
    state = nav.move_selection_up()
    assert state["selection"] == 0


if __name__ == "__main__":
    cli = EnhancedCLI()
    cli.cmdloop()
