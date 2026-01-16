#!/usr/bin/env python3
"""
Rich Status Display Module - Terminal UI for component status
Provides real-time monitoring with tables, progress bars, and colors
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import json
import subprocess

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich import box
except ImportError:
    print("ERROR: rich library not found. Install with: pip install rich")
    sys.exit(1)


class ServiceStatus(Enum):
    """Service status enumeration"""
    RUNNING = "✓ Running"
    STOPPED = "✗ Stopped"
    UNKNOWN = "? Unknown"
    STARTING = "⟳ Starting"
    STOPPING = "⟳ Stopping"
    ERROR = "⚠ Error"
    UNHEALTHY = "! Unhealthy"


class ServiceStatusColor(Enum):
    """Status colors for terminal display"""
    RUNNING = "green"
    STOPPED = "dim"
    UNKNOWN = "yellow"
    STARTING = "cyan"
    STOPPING = "yellow"
    ERROR = "red"
    UNHEALTHY = "red"


class RichStatusDisplay:
    """Displays component and service status using rich library"""

    def __init__(self, project_root: str = "."):
        self.console = Console()
        self.project_root = project_root
        self.services_config = {}
        self.load_services_config()

    def load_services_config(self):
        """Load services configuration from YAML"""
        import yaml
        config_file = os.path.join(self.project_root, "config", "services.yml")
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                self.services_config = config.get("services", {})
        except Exception as e:
            self.console.print(f"[red]Warning: Could not load services config: {e}[/red]")

    def get_service_status(self, service: str) -> Tuple[ServiceStatus, str]:
        """Get current status of a service"""
        if service not in self.services_config:
            return ServiceStatus.UNKNOWN, "Configuration not found"

        svc_config = self.services_config[service]
        svc_type = svc_config.get("type", "unknown")

        try:
            if svc_type == "docker":
                container = svc_config.get("container", service)
                result = subprocess.run(
                    ["docker-compose", "ps", "--services", "--filter", f"status=running"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=5
                )
                if container in result.stdout:
                    return ServiceStatus.RUNNING, "Container running"
                else:
                    return ServiceStatus.STOPPED, "Container stopped"

            elif svc_type == "binary":
                port = svc_config.get("port")
                result = subprocess.run(
                    ["timeout", "2", "bash", "-c", f"echo | nc localhost {port}"],
                    capture_output=True,
                    cwd=self.project_root,
                    timeout=5
                )
                if result.returncode == 0:
                    return ServiceStatus.RUNNING, f"Listening on port {port}"
                else:
                    return ServiceStatus.STOPPED, f"Not listening on port {port}"

            # Health check
            health_check = svc_config.get("health_check")
            if health_check:
                result = subprocess.run(
                    health_check,
                    shell=True,
                    capture_output=True,
                    cwd=self.project_root,
                    timeout=5
                )
                if result.returncode != 0:
                    return ServiceStatus.UNHEALTHY, "Health check failed"

            return ServiceStatus.RUNNING, "Status OK"

        except subprocess.TimeoutExpired:
            return ServiceStatus.UNKNOWN, "Status check timeout"
        except Exception as e:
            return ServiceStatus.ERROR, str(e)

    def show_service_table(self, services: Optional[List[str]] = None):
        """Display services in a rich table"""
        if services is None:
            services = list(self.services_config.keys())

        table = Table(title="[bold]Service Status[/bold]", box=box.ROUNDED, show_header=True)
        table.add_column("Service", style="cyan", width=20)
        table.add_column("Status", width=15)
        table.add_column("Type", width=10)
        table.add_column("Port", width=8)
        table.add_column("Details", width=30)

        for service in services:
            if service not in self.services_config:
                continue

            config = self.services_config[service]
            status, details = self.get_service_status(service)
            svc_type = config.get("type", "unknown")
            port = config.get("port", "-")
            indicator = config.get("status_indicator", "")

            color = ServiceStatusColor[status.name].value
            status_text = f"[{color}]{indicator} {status.value}[/{color}]"

            table.add_row(
                service,
                status_text,
                svc_type,
                str(port),
                details
            )

        self.console.print(table)

    def show_component_table(self, components: Optional[List[str]] = None):
        """Display components in a rich table"""
        try:
            import yaml
            config_file = os.path.join(self.project_root, "config", "services.yml")
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                all_components = config.get("components", {})
        except Exception as e:
            self.console.print(f"[red]Error loading components: {e}[/red]")
            return

        if components is None:
            components = list(all_components.keys())

        table = Table(title="[bold]Component Status[/bold]", box=box.ROUNDED, show_header=True)
        table.add_column("Component", style="cyan", width=15)
        table.add_column("Services", width=40)
        table.add_column("Status", width=15)
        table.add_column("Description", width=30)

        for component in components:
            if component not in all_components:
                continue

            comp_config = all_components[component]
            services = comp_config.get("services", [])
            description = comp_config.get("description", "")

            # Determine component status
            statuses = []
            for svc in services:
                status, _ = self.get_service_status(svc)
                statuses.append(status)

            if all(s == ServiceStatus.RUNNING for s in statuses):
                comp_status = "[green]✓ All Running[/green]"
            elif all(s == ServiceStatus.STOPPED for s in statuses):
                comp_status = "[dim]✗ All Stopped[/dim]"
            else:
                running = sum(1 for s in statuses if s == ServiceStatus.RUNNING)
                total = len(statuses)
                comp_status = f"[yellow]⟳ {running}/{total} Running[/yellow]"

            services_text = ", ".join(services)
            table.add_row(component, services_text, comp_status, description)

        self.console.print(table)

    def show_dashboard(self):
        """Show comprehensive dashboard"""
        # Title
        title = "[bold magenta]Market Data Platform - System Dashboard[/bold magenta]"
        self.console.print(Panel(Align.center(title), expand=False))
        self.console.print()

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.print(f"[dim]Last updated: {timestamp}[/dim]")
        self.console.print()

        # Component table
        self.show_component_table()
        self.console.print()

        # Service table
        self.show_service_table()
        self.console.print()

        # Summary
        total_services = len(self.services_config)
        running = sum(
            1 for svc in self.services_config.keys()
            if self.get_service_status(svc)[0] == ServiceStatus.RUNNING
        )
        summary = f"[cyan]Summary:[/cyan] {running}/{total_services} services running"
        self.console.print(Panel(summary, expand=False, style="cyan"))

    def show_service_logs(self, service: str, lines: int = 50):
        """Show logs for a service"""
        if service not in self.services_config:
            self.console.print(f"[red]Error: Service '{service}' not found[/red]")
            return

        config = self.services_config[service]
        svc_type = config.get("type", "unknown")

        self.console.print(f"[bold]Logs for {service} (last {lines} lines)[/bold]")
        self.console.print()

        try:
            if svc_type == "docker":
                container = config.get("container", service)
                result = subprocess.run(
                    ["docker-compose", "logs", "--tail", str(lines), container],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=10
                )
                self.console.print(result.stdout if result.returncode == 0 else result.stderr)
            else:
                self.console.print(f"[yellow]Log viewing not supported for {svc_type} services[/yellow]")

        except subprocess.TimeoutExpired:
            self.console.print("[red]Log retrieval timeout[/red]")
        except Exception as e:
            self.console.print(f"[red]Error retrieving logs: {e}[/red]")

    def show_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_services": len(self.services_config),
            "services": {},
            "components": {},
            "summary": {}
        }

        # Collect service status
        running_count = 0
        stopped_count = 0

        for service in self.services_config:
            status, details = self.get_service_status(service)
            config = self.services_config[service]

            report["services"][service] = {
                "status": status.name,
                "status_text": status.value,
                "type": config.get("type"),
                "port": config.get("port"),
                "details": details
            }

            if status == ServiceStatus.RUNNING:
                running_count += 1
            elif status == ServiceStatus.STOPPED:
                stopped_count += 1

        report["summary"] = {
            "running": running_count,
            "stopped": stopped_count,
            "health": "HEALTHY" if running_count == len(self.services_config) else "DEGRADED"
        }

        return report

    def print_health_report(self, report: Optional[Dict] = None):
        """Print formatted health report"""
        if report is None:
            report = self.show_health_report()

        # Header
        title = "[bold magenta]Health Report[/bold magenta]"
        self.console.print(Panel(Align.center(title), expand=False))
        self.console.print()

        # Timestamp
        self.console.print(f"[dim]Generated: {report['timestamp']}[/dim]")
        self.console.print()

        # Summary
        summary = report["summary"]
        health_color = "green" if summary["health"] == "HEALTHY" else "yellow"
        self.console.print(f"[bold]Health Status:[/bold] [{health_color}]{summary['health']}[/{health_color}]")
        self.console.print(
            f"[bold]Services:[/bold] "
            f"[green]{summary['running']} running[/green], "
            f"[dim]{summary['stopped']} stopped[/dim]"
        )
        self.console.print()

        # Service details
        table = Table(title="Service Details", box=box.ROUNDED, show_header=True)
        table.add_column("Service", style="cyan")
        table.add_column("Status", width=15)
        table.add_column("Type", width=10)
        table.add_column("Details", width=35)

        for service, info in report["services"].items():
            status_color = "green" if info["status"] == "RUNNING" else (
                "dim" if info["status"] == "STOPPED" else "yellow"
            )
            status_text = f"[{status_color}]{info['status_text']}[/{status_color}]"
            table.add_row(service, status_text, info["type"], info["details"])

        self.console.print(table)
        self.console.print()

        # Save report
        self.save_report(report)

    def save_report(self, report: Dict):
        """Save health report to file"""
        try:
            log_dir = os.path.join(self.project_root, "logs")
            os.makedirs(log_dir, exist_ok=True)

            report_file = os.path.join(log_dir, f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            self.console.print(f"[dim]Report saved to: {report_file}[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not save report: {e}[/yellow]")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Rich Status Display for Market Data Platform")
    parser.add_argument("command", nargs="?", default="dashboard", 
                       choices=["dashboard", "services", "components", "health", "logs"])
    parser.add_argument("--service", help="Specific service for logs command")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines to show")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    display = RichStatusDisplay(project_root=args.project_root)

    if args.command == "dashboard":
        display.show_dashboard()
    elif args.command == "services":
        display.show_service_table()
    elif args.command == "components":
        display.show_component_table()
    elif args.command == "health":
        report = display.show_health_report()
        display.print_health_report(report)
    elif args.command == "logs":
        if not args.service:
            display.console.print("[red]Error: --service required for logs command[/red]")
            sys.exit(1)
        display.show_service_logs(args.service, args.lines)


if __name__ == "__main__":
    main()
