#!/usr/bin/env python3
"""
Component Manager - Advanced component lifecycle management
Handles installation, uninstallation, graceful shutdown, and state tracking
"""

import os
import sys
import json
import yaml
import subprocess
import signal
import time
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from pathlib import Path
from datetime import datetime
import logging

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import box
except ImportError:
    print("ERROR: rich library not found. Install with: pip install rich")
    sys.exit(1)


class ComponentState(Enum):
    """Component state enumeration"""
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNINSTALLING = "uninstalling"


class ComponentManager:
    """Manages component lifecycle with dependencies and state tracking"""

    def __init__(self, project_root: str = ".", config_file: str = "config/services.yml"):
        self.project_root = Path(project_root)
        self.config_file = self.project_root / config_file
        self.state_file = self.project_root / ".component_state.json"
        self.console = Console()
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.state = self._load_state()
        self.running_pids: Dict[str, int] = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger("ComponentManager")
        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_dir / "component_manager.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_config(self) -> Dict:
        """Load services configuration"""
        try:
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.console.print(f"[red]Error loading config: {e}[/red]")
            return {}

    def _load_state(self) -> Dict:
        """Load component state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")

        return {
            "components": {},
            "services": {},
            "last_updated": datetime.now().isoformat()
        }

    def _save_state(self):
        """Save component state"""
        try:
            self.state["last_updated"] = datetime.now().isoformat()
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
            self.logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")

    def _get_service_config(self, service: str) -> Optional[Dict]:
        """Get service configuration"""
        return self.config.get("services", {}).get(service)

    def _check_dependencies(self, component: str, operation: str = "start") -> Tuple[bool, List[str]]:
        """Check if dependencies are satisfied"""
        missing = []

        if operation == "start":
            config = self._get_service_config(component)
            if not config:
                return False, ["Service not found in configuration"]

            depends_on = config.get("depends_on", [])
            for dep in depends_on:
                dep_state = self.state.get("services", {}).get(dep, {}).get("state")
                if dep_state != ComponentState.RUNNING.value:
                    missing.append(dep)

        return len(missing) == 0, missing

    def _run_command(self, command: str, timeout: int = 30, shell: bool = True) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Command timeout"
        except Exception as e:
            return 1, "", str(e)

    def install(self, service: str, skip_deps: bool = False) -> bool:
        """Install a service with its dependencies"""
        self.console.print(f"[cyan]→[/cyan] Installing {service}...")
        self.logger.info(f"Installing service: {service}")

        # Check dependencies
        satisfied, missing = self._check_dependencies(service, "install")
        if missing and not skip_deps:
            for dep in missing:
                if not self.install(dep):
                    self.console.print(f"[red]✗[/red] Failed to install dependency: {dep}")
                    return False

        config = self._get_service_config(service)
        if not config:
            self.console.print(f"[red]✗[/red] Service not found: {service}")
            return False

        svc_type = config.get("type", "unknown")
        startup_cmd = config.get("startup_cmd", "")

        if not startup_cmd:
            self.console.print(f"[yellow]⚠[/yellow] No startup command defined")
            return False

        # Run installation
        exit_code, stdout, stderr = self._run_command(startup_cmd, timeout=config.get("startup_timeout", 60))

        if exit_code == 0:
            self.state.setdefault("services", {})[service] = {
                "state": ComponentState.RUNNING.value,
                "installed_at": datetime.now().isoformat(),
                "type": svc_type
            }
            self._save_state()
            self.console.print(f"[green]✓[/green] {service} installed and started")
            self.logger.info(f"Service installed: {service}")
            return True
        else:
            self.state.setdefault("services", {})[service] = {
                "state": ComponentState.FAILED.value,
                "error": stderr or stdout,
                "type": svc_type
            }
            self._save_state()
            self.console.print(f"[red]✗[/red] Failed to install {service}")
            self.console.print(f"[red]{stderr or stdout}[/red]")
            self.logger.error(f"Installation failed for {service}: {stderr or stdout}")
            return False

    def uninstall(self, service: str, remove_data: bool = False) -> bool:
        """Uninstall a service with cleanup"""
        self.console.print(f"[cyan]→[/cyan] Uninstalling {service}...")
        self.logger.info(f"Uninstalling service: {service}, remove_data={remove_data}")

        config = self._get_service_config(service)
        if not config:
            self.console.print(f"[red]✗[/red] Service not found: {service}")
            return False

        # First stop the service
        if not self.stop(service):
            self.console.print(f"[yellow]⚠[/yellow] Could not stop service before uninstall")

        # Run uninstall
        uninstall_cmd = config.get("shutdown_cmd", "")
        if uninstall_cmd:
            exit_code, stdout, stderr = self._run_command(uninstall_cmd, timeout=config.get("shutdown_timeout", 30))
            if exit_code != 0:
                self.console.print(f"[yellow]⚠[/yellow] Uninstall command failed: {stderr}")

        # Data cleanup
        if remove_data:
            self._cleanup_service_data(service)

        # Update state
        if service in self.state.get("services", {}):
            del self.state["services"][service]
        self._save_state()

        self.console.print(f"[green]✓[/green] {service} uninstalled")
        self.logger.info(f"Service uninstalled: {service}")
        return True

    def _cleanup_service_data(self, service: str):
        """Clean up service data files"""
        data_patterns = {
            "postgres": [".pgdata"],
            "redis": [".redis_data"],
            "influxdb": [".influx_data"],
        }

        patterns = data_patterns.get(service, [])
        for pattern in patterns:
            data_path = self.project_root / pattern
            if data_path.exists():
                try:
                    import shutil
                    shutil.rmtree(data_path)
                    self.console.print(f"[dim]Cleaned up data: {pattern}[/dim]")
                except Exception as e:
                    self.console.print(f"[yellow]⚠[/yellow] Could not clean up {pattern}: {e}")

    def start(self, service: str) -> bool:
        """Start a service"""
        self.console.print(f"[cyan]→[/cyan] Starting {service}...")
        self.logger.info(f"Starting service: {service}")

        # Check dependencies
        satisfied, missing = self._check_dependencies(service, "start")
        if not satisfied:
            self.console.print(f"[red]✗[/red] Unsatisfied dependencies: {', '.join(missing)}")
            return False

        config = self._get_service_config(service)
        if not config:
            self.console.print(f"[red]✗[/red] Service not found: {service}")
            return False

        startup_cmd = config.get("startup_cmd", "")
        if not startup_cmd:
            self.console.print(f"[red]✗[/red] No startup command defined")
            return False

        # Check health
        exit_code, stdout, stderr = self._run_command(startup_cmd, timeout=config.get("startup_timeout", 60))

        if exit_code == 0:
            self.state.setdefault("services", {})[service] = {
                "state": ComponentState.RUNNING.value,
                "started_at": datetime.now().isoformat()
            }
            self._save_state()
            self.console.print(f"[green]✓[/green] {service} started")
            self.logger.info(f"Service started: {service}")

            # Verify health
            time.sleep(1)
            if self._check_health(service):
                self.console.print(f"[green]✓[/green] Health check passed")
                return True
            else:
                self.console.print(f"[yellow]⚠[/yellow] Health check failed")
                return False

        else:
            self.state.setdefault("services", {})[service] = {
                "state": ComponentState.FAILED.value,
                "error": stderr or stdout
            }
            self._save_state()
            self.console.print(f"[red]✗[/red] Failed to start {service}: {stderr}")
            return False

    def stop(self, service: str, graceful: bool = True, timeout: int = 30) -> bool:
        """Stop a service gracefully"""
        self.console.print(f"[cyan]→[/cyan] Stopping {service}...")
        self.logger.info(f"Stopping service: {service}, graceful={graceful}")

        config = self._get_service_config(service)
        if not config:
            self.console.print(f"[red]✗[/red] Service not found: {service}")
            return False

        shutdown_cmd = config.get("shutdown_cmd", "")
        if not shutdown_cmd:
            self.console.print(f"[red]✗[/red] No shutdown command defined")
            return False

        # Graceful shutdown with timeout
        shutdown_timeout = config.get("shutdown_timeout", timeout)
        exit_code, stdout, stderr = self._run_command(
            shutdown_cmd,
            timeout=shutdown_timeout
        )

        if exit_code == 0:
            self.state.setdefault("services", {})[service] = {
                "state": ComponentState.STOPPED.value,
                "stopped_at": datetime.now().isoformat()
            }
            self._save_state()
            self.console.print(f"[green]✓[/green] {service} stopped gracefully")
            self.logger.info(f"Service stopped: {service}")
            return True
        else:
            self.console.print(f"[yellow]⚠[/yellow] Graceful stop failed, forcing: {stderr}")
            self.logger.warning(f"Graceful stop failed for {service}: {stderr}")
            # Force kill if graceful stop fails
            return self._force_stop(service)

    def _force_stop(self, service: str) -> bool:
        """Force stop a service"""
        self.console.print(f"[yellow]⚠[/yellow] Force stopping {service}...")

        config = self._get_service_config(service)
        svc_type = config.get("type", "docker")

        if svc_type == "docker":
            cmd = f"docker-compose kill {config.get('container', service)}"
        else:
            cmd = f"pkill -9 -f '{service}'"

        exit_code, _, _ = self._run_command(cmd, timeout=10)

        if exit_code == 0:
            self.state.setdefault("services", {})[service] = {
                "state": ComponentState.STOPPED.value,
                "stopped_at": datetime.now().isoformat()
            }
            self._save_state()
            self.console.print(f"[green]✓[/green] {service} force stopped")
            return True

        return False

    def _check_health(self, service: str) -> bool:
        """Check service health"""
        config = self._get_service_config(service)
        health_check = config.get("health_check")

        if not health_check:
            return True

        try:
            exit_code, _, _ = self._run_command(health_check, timeout=10)
            return exit_code == 0
        except Exception as e:
            self.logger.warning(f"Health check failed for {service}: {e}")
            return False

    def install_all(self) -> bool:
        """Install all services in dependency order"""
        services = self.config.get("services", {}).keys()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Installing all services...", total=len(services))

            for service in services:
                if self.install(service):
                    progress.update(task, advance=1)
                else:
                    self.console.print(f"[red]✗[/red] Installation failed, stopping")
                    return False

        self.console.print("[green]✓[/green] All services installed")
        return True

    def uninstall_all(self, remove_data: bool = False) -> bool:
        """Uninstall all services in reverse order"""
        services = list(reversed(list(self.config.get("services", {}).keys())))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Uninstalling all services...", total=len(services))

            for service in services:
                if self.uninstall(service, remove_data=remove_data):
                    progress.update(task, advance=1)
                else:
                    self.console.print(f"[yellow]⚠[/yellow] Uninstall failed for {service}, continuing")

        self.console.print("[green]✓[/green] All services uninstalled")
        return True

    def graceful_shutdown(self, timeout: int = 60):
        """Graceful system shutdown with signal handling"""
        self.console.print("[cyan]→[/cyan] Initiating graceful shutdown...")
        self.logger.info("Graceful shutdown initiated")

        # Reverse order - stop dependents first
        services = list(reversed(list(self.config.get("services", {}).keys())))
        start_time = time.time()

        for service in services:
            if time.time() - start_time > timeout:
                self.console.print(f"[yellow]⚠[/yellow] Shutdown timeout, force stopping remaining services")
                break

            service_state = self.state.get("services", {}).get(service, {})
            if service_state.get("state") == ComponentState.RUNNING.value:
                self.stop(service)

        self.console.print("[green]✓[/green] Graceful shutdown complete")
        self.logger.info("Graceful shutdown complete")

    def status(self) -> Dict:
        """Get current status of all services"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }

        for service in self.config.get("services", {}).keys():
            service_state = self.state.get("services", {}).get(service, {})
            config = self._get_service_config(service)

            status["services"][service] = {
                "state": service_state.get("state", ComponentState.NOT_INSTALLED.value),
                "type": config.get("type"),
                "port": config.get("port"),
                "depends_on": config.get("depends_on", []),
                "installed_at": service_state.get("installed_at"),
                "started_at": service_state.get("started_at"),
                "stopped_at": service_state.get("stopped_at")
            }

        return status

    def show_status(self):
        """Display status in rich format"""
        status = self.status()

        table = Table(title="[bold]Component Status[/bold]", box=box.ROUNDED, show_header=True)
        table.add_column("Service", style="cyan", width=20)
        table.add_column("State", width=15)
        table.add_column("Type", width=10)
        table.add_column("Port", width=8)
        table.add_column("Dependencies", width=25)

        for service, info in status["services"].items():
            state_color = "green" if info["state"] == "running" else "dim"
            state_text = f"[{state_color}]{info['state']}[/{state_color}]"
            deps = ", ".join(info["depends_on"]) if info["depends_on"] else "-"

            table.add_row(
                service,
                state_text,
                info.get("type", "-"),
                str(info.get("port", "-")),
                deps
            )

        self.console.print(table)
        self.console.print(f"[dim]Last updated: {status['timestamp']}[/dim]")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Component Manager")
    parser.add_argument("command", choices=["install", "uninstall", "start", "stop", "status", "shutdown"])
    parser.add_argument("service", nargs="?", help="Service name (optional for some commands)")
    parser.add_argument("--all", action="store_true", help="Apply to all services")
    parser.add_argument("--remove-data", action="store_true", help="Remove data during uninstall")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()
    manager = ComponentManager(project_root=args.project_root)

    if args.command == "install":
        if args.all:
            manager.install_all()
        elif args.service:
            manager.install(args.service)
        else:
            parser.print_help()

    elif args.command == "uninstall":
        if args.all:
            manager.uninstall_all(remove_data=args.remove_data)
        elif args.service:
            manager.uninstall(args.service, remove_data=args.remove_data)
        else:
            parser.print_help()

    elif args.command == "start":
        if args.service:
            manager.start(args.service)
        else:
            parser.print_help()

    elif args.command == "stop":
        if args.all:
            manager.graceful_shutdown()
        elif args.service:
            manager.stop(args.service)
        else:
            parser.print_help()

    elif args.command == "status":
        manager.show_status()

    elif args.command == "shutdown":
        manager.graceful_shutdown()


if __name__ == "__main__":
    main()
