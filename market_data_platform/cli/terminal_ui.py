#!/usr/bin/env python3
"""
Advanced Terminal UI - Interactive dashboard for component management
Features: Rich status display, real-time monitoring, keyboard navigation
"""

import sys
import os
from typing import Optional
import signal

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.align import Align
    from rich import box
    from rich.live import Live
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
except ImportError:
    print("ERROR: rich library not found. Install with: pip install rich")
    sys.exit(1)

from market_data_platform.cli.rich_status import RichStatusDisplay
from market_data_platform.cli.component_manager import ComponentManager


class AdvancedTerminalUI:
    """Interactive terminal UI for component management"""

    def __init__(self, project_root: str = "."):
        self.console = Console()
        self.project_root = project_root
        self.status_display = RichStatusDisplay(project_root)
        self.component_manager = ComponentManager(project_root)
        self.running = True

    def show_main_menu(self):
        """Display main menu"""
        while self.running:
            self.console.clear()
            
            # Header
            title = "[bold magenta]Market Data Platform - Component Manager[/bold magenta]"
            self.console.print(Panel(Align.center(title), expand=False, style="magenta"))
            self.console.print()

            # Menu options
            menu_table = Table(show_header=False, box=box.ROUNDED)
            menu_table.add_column("Key", style="cyan", width=5)
            menu_table.add_column("Option", width=40)
            menu_table.add_column("Description", width=40)

            menu_items = [
                ("1", "Status", "View rich dashboard with all components"),
                ("2", "Install", "Install components with dependencies"),
                ("3", "Uninstall", "Uninstall components with cleanup"),
                ("4", "Start", "Start installed components"),
                ("5", "Stop", "Stop running components gracefully"),
                ("6", "Restart", "Restart specific components"),
                ("7", "Health", "View comprehensive health report"),
                ("8", "Logs", "View service logs"),
                ("9", "Configuration", "View current configuration"),
                ("0", "Exit", "Exit the application"),
            ]

            for key, option, desc in menu_items:
                menu_table.add_row(key, option, desc)

            self.console.print(menu_table)
            self.console.print()

            choice = self.console.input("[bold cyan]Select option (0-9):[/bold cyan] ").strip()
            self.handle_menu_choice(choice)

    def handle_menu_choice(self, choice: str):
        """Handle menu selection"""
        if choice == "1":
            self.show_status_menu()
        elif choice == "2":
            self.show_install_menu()
        elif choice == "3":
            self.show_uninstall_menu()
        elif choice == "4":
            self.show_start_menu()
        elif choice == "5":
            self.show_stop_menu()
        elif choice == "6":
            self.show_restart_menu()
        elif choice == "7":
            self.show_health_menu()
        elif choice == "8":
            self.show_logs_menu()
        elif choice == "9":
            self.show_config_menu()
        elif choice == "0":
            self.exit_app()
        else:
            self.console.print("[red]Invalid option. Press Enter to continue...[/red]")
            input()

    def show_status_menu(self):
        """Show status dashboard"""
        self.console.clear()
        self.status_display.show_dashboard()
        self.console.print()
        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_install_menu(self):
        """Show install menu"""
        self.console.clear()
        self.console.print("[bold cyan]Install Components[/bold cyan]")
        self.console.print()

        # Get available services
        services = list(self.component_manager.config.get("services", {}).keys())
        
        if not services:
            self.console.print("[red]No services available[/red]")
            self.console.input("[dim]Press Enter to continue...[/dim]")
            return

        # Show available services
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Index", style="cyan")
        table.add_column("Service", width=20)
        table.add_column("Type", width=10)
        table.add_column("Port", width=8)

        for idx, service in enumerate(services):
            config = self.component_manager._get_service_config(service)
            table.add_row(
                str(idx),
                service,
                config.get("type", "unknown"),
                str(config.get("port", "-"))
            )

        self.console.print(table)
        self.console.print()

        choice = self.console.input("[cyan]Enter service index (or 'all'):[/cyan] ").strip()

        if choice.lower() == "all":
            if Confirm.ask("Install all services?"):
                self.component_manager.install_all()
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(services):
                service = services[idx]
                if Confirm.ask(f"Install {service}?"):
                    self.component_manager.install(service)
            else:
                self.console.print("[red]Invalid index[/red]")

        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_uninstall_menu(self):
        """Show uninstall menu"""
        self.console.clear()
        self.console.print("[bold cyan]Uninstall Components[/bold cyan]")
        self.console.print()

        services = list(self.component_manager.config.get("services", {}).keys())
        
        if not services:
            self.console.print("[red]No services available[/red]")
            self.console.input("[dim]Press Enter to continue...[/dim]")
            return

        # Show services
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Index", style="cyan")
        table.add_column("Service", width=20)

        for idx, service in enumerate(services):
            table.add_row(str(idx), service)

        self.console.print(table)
        self.console.print()

        choice = self.console.input("[cyan]Enter service index (or 'all'):[/cyan] ").strip()
        remove_data = Confirm.ask("Remove service data?", default=False)

        if choice.lower() == "all":
            if Confirm.ask("Uninstall all services?"):
                self.component_manager.uninstall_all(remove_data=remove_data)
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(services):
                service = services[idx]
                if Confirm.ask(f"Uninstall {service}?"):
                    self.component_manager.uninstall(service, remove_data=remove_data)

        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_start_menu(self):
        """Show start menu"""
        self.console.clear()
        self.console.print("[bold cyan]Start Components[/bold cyan]")
        self.console.print()

        services = list(self.component_manager.config.get("services", {}).keys())
        
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Index", style="cyan")
        table.add_column("Service", width=20)

        for idx, service in enumerate(services):
            table.add_row(str(idx), service)

        self.console.print(table)
        self.console.print()

        choice = self.console.input("[cyan]Enter service index (or 'all'):[/cyan] ").strip()

        if choice.lower() == "all":
            if Confirm.ask("Start all services?"):
                self._start_all_with_progress()
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(services):
                service = services[idx]
                if Confirm.ask(f"Start {service}?"):
                    self.component_manager.start(service)

        self.console.input("[dim]Press Enter to continue...[/dim]")

    def _start_all_with_progress(self):
        """Start all services with progress display"""
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        services = list(self.component_manager.config.get("services", {}).keys())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Starting services...", total=len(services))
            for service in services:
                progress.update(task, description=f"Starting {service}...")
                self.component_manager.start(service)
                progress.update(task, advance=1)

    def show_stop_menu(self):
        """Show stop menu"""
        self.console.clear()
        self.console.print("[bold cyan]Stop Components[/bold cyan]")
        self.console.print()

        options = [
            ("1", "Graceful shutdown (all)", "Stop all in reverse order"),
            ("2", "Stop specific service", "Stop single service"),
            ("3", "Force stop all", "Force stop all services"),
        ]

        table = Table(show_header=False, box=box.ROUNDED)
        for key, label, desc in options:
            table.add_row(f"[cyan]{key}[/cyan]", label, desc)

        self.console.print(table)
        self.console.print()

        choice = self.console.input("[cyan]Select option:[/cyan] ").strip()

        if choice == "1":
            if Confirm.ask("Graceful shutdown?"):
                self.component_manager.graceful_shutdown()
        elif choice == "2":
            services = list(self.component_manager.config.get("services", {}).keys())
            for idx, svc in enumerate(services):
                self.console.print(f"{idx}: {svc}")
            idx = int(self.console.input("Enter index: "))
            if 0 <= idx < len(services):
                service = services[idx]
                if Confirm.ask(f"Stop {service}?"):
                    self.component_manager.stop(service)

        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_restart_menu(self):
        """Show restart menu"""
        self.console.clear()
        self.console.print("[bold cyan]Restart Components[/bold cyan]")
        self.console.print()

        services = list(self.component_manager.config.get("services", {}).keys())
        
        for idx, service in enumerate(services):
            self.console.print(f"{idx}: {service}")

        choice = self.console.input("[cyan]Enter service index:[/cyan] ").strip()
        
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(services):
                service = services[idx]
                if Confirm.ask(f"Restart {service}?"):
                    self.component_manager.stop(service)
                    import time
                    time.sleep(1)
                    self.component_manager.start(service)

        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_health_menu(self):
        """Show health report"""
        self.console.clear()
        report = self.component_manager.state
        self.status_display.print_health_report()
        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_logs_menu(self):
        """Show logs menu"""
        self.console.clear()
        self.console.print("[bold cyan]View Service Logs[/bold cyan]")
        self.console.print()

        services = list(self.component_manager.config.get("services", {}).keys())
        
        for idx, service in enumerate(services):
            self.console.print(f"{idx}: {service}")

        choice = self.console.input("[cyan]Enter service index:[/cyan] ").strip()
        
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(services):
                service = services[idx]
                lines = int(self.console.input("Number of lines [50]: ") or "50")
                self.status_display.show_service_logs(service, lines)

        self.console.input("[dim]Press Enter to continue...[/dim]")

    def show_config_menu(self):
        """Show configuration"""
        self.console.clear()
        self.console.print("[bold cyan]Configuration[/bold cyan]")
        self.console.print()

        config_info = f"""
[cyan]Project Root:[/cyan] {self.project_root}
[cyan]Services Config:[/cyan] {self.project_root}/config/services.yml
[cyan]State File:[/cyan] {self.project_root}/.component_state.json
[cyan]Logs Directory:[/cyan] {self.project_root}/logs
[cyan]PIDs Directory:[/cyan] {self.project_root}/.pids

[cyan]Services Configured:[/cyan] {len(self.component_manager.config.get('services', {}))}
[cyan]Components:[/cyan] {len(self.component_manager.config.get('components', {}))}
"""
        self.console.print(config_info)
        self.console.input("[dim]Press Enter to continue...[/dim]")

    def exit_app(self):
        """Exit application"""
        if Confirm.ask("Exit application?"):
            self.running = False
            self.console.print("[cyan]Goodbye![/cyan]")
            sys.exit(0)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Terminal UI for Component Management")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    try:
        ui = AdvancedTerminalUI(project_root=args.project_root)
        ui.show_main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
