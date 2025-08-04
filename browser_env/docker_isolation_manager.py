#!/usr/bin/env python3
"""
Docker Isolation Manager for WebArena Parallel Execution

This module provides isolated Docker container environments for each WebArena task,
preventing race conditions and state conflicts during parallel execution.

Key Features:
- Each task gets its own isolated container set
- Unique port allocation per task
- Isolated authentication and database state
- Automatic container lifecycle management
- Robust cleanup and error handling
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for a single container instance"""

    name: str
    image: str
    host_port: int
    container_port: int
    volumes: Dict[str, str] = None
    environment: Dict[str, str] = None
    command: Optional[str] = None

    def __post_init__(self):
        if self.volumes is None:
            self.volumes = {}
        if self.environment is None:
            self.environment = {}


@dataclass
class TaskEnvironment:
    """Complete isolated environment for a single task"""

    task_id: int
    base_port: int
    containers: Dict[str, ContainerConfig]
    auth_dir: str
    temp_dirs: List[str]

    def get_urls(self) -> Dict[str, str]:
        """Get the isolated URLs for this task environment"""
        return {
            "SHOPPING": f"http://localhost:{self.base_port}",
            "SHOPPING_ADMIN": f"http://localhost:{self.base_port + 10}/admin",
            "REDDIT": f"http://localhost:{self.base_port + 20}",
            "GITLAB": f"http://localhost:{self.base_port + 30}",
            "WIKIPEDIA": f"http://localhost:{self.base_port + 40}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
            "MAP": f"http://localhost:{self.base_port + 50}",
            "HOMEPAGE": f"http://localhost:{self.base_port + 60}",
        }


class DockerIsolationManager:
    """Manages isolated Docker environments for WebArena tasks"""

    # WebArena Docker images from environment_docker/README.md
    DOCKER_IMAGES = {
        "shopping": "shopping_final_0712",
        "shopping_admin": "shopping_admin_final_0719",
        "forum": "postmill-populated-exposed-withimg",
        "gitlab": "gitlab-populated-final-port8023",
        "wikipedia": "ghcr.io/kiwix/kiwix-serve:3.3.0",
    }

    # Default port mappings from original setup
    DEFAULT_PORTS = {
        "shopping": 80,  # maps to 7770
        "shopping_admin": 80,  # maps to 7780
        "forum": 80,  # maps to 9999
        "gitlab": 8023,  # maps to 8023
        "wikipedia": 80,  # maps to 8888
    }

    def __init__(
        self, base_port_start: int = 10000, port_range_size: int = 100, max_concurrent_tasks: int = 500
    ):
        """
        Initialize the Docker isolation manager

        Args:
            base_port_start: Starting port for task isolation (default: 10000)
            port_range_size: Port range allocated per task (default: 100)
            max_concurrent_tasks: Maximum concurrent tasks supported (default: 500)
        """
        self.base_port_start = base_port_start
        self.port_range_size = port_range_size
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_environments: Dict[int, TaskEnvironment] = {}
        self.lock = threading.Lock()

    def allocate_ports(self, task_id: int) -> int:
        """
        Allocate a unique port range for a task.
        Uses modulo-based allocation to stay within Docker's valid port range (1-65535).
        
        Args:
            task_id: Task identifier
            
        Returns:
            Base port for the task's port range
        """
        # Use modulo to map task_id to a smaller range that fits within Docker limits
        # This supports up to max_concurrent_tasks unique port ranges
        normalized_task_id = task_id % self.max_concurrent_tasks
        allocated_port = self.base_port_start + (normalized_task_id * self.port_range_size)
        
        # Ensure we don't exceed Docker's port limit (65535)
        max_port = allocated_port + self.port_range_size - 1
        if max_port > 65535:
            raise ValueError(
                f"Port allocation for task {task_id} would exceed Docker's port limit. "
                f"Allocated range: {allocated_port}-{max_port}, max allowed: 65535. "
                f"Consider reducing base_port_start, port_range_size, or max_concurrent_tasks."
            )
        
        return allocated_port

    def create_isolated_auth(self, task_id: int, original_config: Dict) -> str:
        """Create isolated authentication directory for a task"""
        auth_dir = f"./temp_auth_task_{task_id}_{int(time.time())}"
        os.makedirs(auth_dir, exist_ok=True)

        # Copy original auth files if they exist
        if original_config.get("storage_state"):
            original_auth = original_config["storage_state"]
            if os.path.exists(original_auth):
                auth_filename = os.path.basename(original_auth)
                isolated_auth = os.path.join(auth_dir, auth_filename)
                shutil.copy2(original_auth, isolated_auth)
                logger.info(
                    f"Copied auth file for task {task_id}: {original_auth} -> {isolated_auth}"
                )
            else:
                logger.warning(
                    f"Original auth file not found: {original_auth}"
                )

        return auth_dir

    def create_task_environment(
        self, task_id: int, sites: List[str]
    ) -> TaskEnvironment:
        """Create isolated environment for a specific task"""
        base_port = self.allocate_ports(task_id)
        auth_dir = self.create_isolated_auth(
            task_id, {}
        )  # Will be updated later

        containers = {}
        temp_dirs = [auth_dir]

        # Create container configs for required sites
        for site in sites:
            if site == "shopping":
                containers["shopping"] = ContainerConfig(
                    name=f"webarena_shopping_task_{task_id}",
                    image=self.DOCKER_IMAGES["shopping"],
                    host_port=base_port,
                    container_port=self.DEFAULT_PORTS["shopping"],
                )
            elif site == "shopping_admin":
                containers["shopping_admin"] = ContainerConfig(
                    name=f"webarena_shopping_admin_task_{task_id}",
                    image=self.DOCKER_IMAGES["shopping_admin"],
                    host_port=base_port + 10,
                    container_port=self.DEFAULT_PORTS["shopping_admin"],
                )
            elif site == "reddit":
                containers["forum"] = ContainerConfig(
                    name=f"webarena_forum_task_{task_id}",
                    image=self.DOCKER_IMAGES["forum"],
                    host_port=base_port + 20,
                    container_port=self.DEFAULT_PORTS["forum"],
                )
            elif site == "gitlab":
                containers["gitlab"] = ContainerConfig(
                    name=f"webarena_gitlab_task_{task_id}",
                    image=self.DOCKER_IMAGES["gitlab"],
                    host_port=base_port + 30,
                    container_port=self.DEFAULT_PORTS["gitlab"],
                    command="/opt/gitlab/embedded/bin/runsvdir-start",
                )
            elif site == "wikipedia":
                # Wikipedia needs special volume mount for data
                wiki_data_dir = f"./temp_wiki_task_{task_id}"
                os.makedirs(wiki_data_dir, exist_ok=True)
                temp_dirs.append(wiki_data_dir)

                containers["wikipedia"] = ContainerConfig(
                    name=f"webarena_wikipedia_task_{task_id}",
                    image=self.DOCKER_IMAGES["wikipedia"],
                    host_port=base_port + 40,
                    container_port=self.DEFAULT_PORTS["wikipedia"],
                    volumes={wiki_data_dir: "/data"},
                    command="wikipedia_en_all_maxi_2022-05.zim",
                )

        return TaskEnvironment(
            task_id=task_id,
            base_port=base_port,
            containers=containers,
            auth_dir=auth_dir,
            temp_dirs=temp_dirs,
        )

    def start_container(self, container: ContainerConfig) -> bool:
        """Start a single Docker container"""
        try:
            # Check if container already exists and remove it
            subprocess.run(
                ["docker", "rm", "-f", container.name],
                capture_output=True,
                check=False,
            )

            # Build docker run command
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                container.name,
                "-p",
                f"{container.host_port}:{container.container_port}",
            ]

            # Add volume mounts
            for host_path, container_path in container.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

            # Add environment variables
            for key, value in container.environment.items():
                cmd.extend(["-e", f"{key}={value}"])

            # Add image
            cmd.append(container.image)

            # Add command if specified
            if container.command:
                cmd.extend(container.command.split())

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            logger.info(
                f"Started container {container.name} on port {container.host_port}"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to start container {container.name}: {e.stderr}"
            )
            return False

    def wait_for_container_ready(
        self, container: ContainerConfig, timeout: int = 120
    ) -> bool:
        """Wait for container to be ready to accept connections"""
        import socket
        import time

        # For shopping_admin containers, give extra time for Magento to start
        if "shopping_admin" in container.name:
            timeout = 180  # 3 minutes for Magento startup

        start_time = time.time()
        logger.info(
            f"Waiting for container {container.name} to be ready (timeout: {timeout}s)..."
        )

        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex(
                        ("localhost", container.host_port)
                    )
                    if result == 0:
                        logger.info(
                            f"Container {container.name} is ready on port {container.host_port}"
                        )
                        # Additional wait for Magento services to fully initialize
                        if "shopping_admin" in container.name:
                            logger.info(
                                f"Giving Magento additional 30s to fully initialize..."
                            )
                            time.sleep(30)
                        return True
            except Exception:
                pass
            time.sleep(5)  # Check every 5 seconds instead of 2

        logger.warning(
            f"Container {container.name} not ready after {timeout}s"
        )
        return False

    def configure_container_urls(self, env: TaskEnvironment) -> None:
        """Configure container internal URLs for proper operation"""
        urls = env.get_urls()

        # Configure shopping container
        if "shopping" in env.containers:
            container = env.containers["shopping"]
            shopping_url = urls["SHOPPING"]

            commands = [
                f'/var/www/magento2/bin/magento setup:store-config:set --base-url="{shopping_url}"',
                f"mysql -u magentouser -pMyPassword magentodb -e \"UPDATE core_config_data SET value='{shopping_url}/' WHERE path = 'web/secure/base_url';\"",
                "/var/www/magento2/bin/magento cache:flush",
            ]

            for cmd in commands:
                try:
                    subprocess.run(
                        ["docker", "exec", container.name, "bash", "-c", cmd],
                        capture_output=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"Failed to configure shopping container: {e}"
                    )

        # Configure shopping_admin container
        if "shopping_admin" in env.containers:
            container = env.containers["shopping_admin"]
            admin_url = urls["SHOPPING_ADMIN"].replace("/admin", "")

            commands = [
                f'/var/www/magento2/bin/magento setup:store-config:set --base-url="{admin_url}"',
                f"mysql -u magentouser -pMyPassword magentodb -e \"UPDATE core_config_data SET value='{admin_url}/' WHERE path = 'web/secure/base_url';\"",
                "php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0",
                "php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0",
                "/var/www/magento2/bin/magento cache:flush",
            ]

            for cmd in commands:
                try:
                    subprocess.run(
                        ["docker", "exec", container.name, "bash", "-c", cmd],
                        capture_output=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"Failed to configure shopping_admin container: {e}"
                    )

        # Configure gitlab container
        if "gitlab" in env.containers:
            container = env.containers["gitlab"]
            gitlab_url = urls["GITLAB"]

            commands = [
                f"sed -i \"s|^external_url.*|external_url '{gitlab_url}'|\" /etc/gitlab/gitlab.rb",
                "gitlab-ctl reconfigure",
            ]

            for cmd in commands:
                try:
                    subprocess.run(
                        ["docker", "exec", container.name, "bash", "-c", cmd],
                        capture_output=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"Failed to configure gitlab container: {e}"
                    )

    def start_environment(
        self, task_id: int, sites: List[str]
    ) -> Optional[TaskEnvironment]:
        """Start complete isolated environment for a task"""
        with self.lock:
            if task_id in self.active_environments:
                logger.warning(
                    f"Environment for task {task_id} already exists"
                )
                return self.active_environments[task_id]

            env = self.create_task_environment(task_id, sites)

            # Start all containers
            success = True
            for container in env.containers.values():
                if not self.start_container(container):
                    success = False
                    break

            if not success:
                self.cleanup_environment(task_id)
                return None

            # Wait for containers to be ready
            for container in env.containers.values():
                if not self.wait_for_container_ready(container):
                    logger.warning(
                        f"Container {container.name} may not be fully ready"
                    )

            # Configure container URLs
            self.configure_container_urls(env)

            # Generate authentication for isolated environment if needed
            if sites:
                self._generate_isolated_auth(task_id, sites, env)

            self.active_environments[task_id] = env
            logger.info(
                f"Started isolated environment for task {task_id} on ports {env.base_port}-{env.base_port + 99}"
            )
            return env

    def stop_environment(self, task_id: int) -> None:
        """Stop and cleanup environment for a task"""
        with self.lock:
            if task_id not in self.active_environments:
                return

            env = self.active_environments[task_id]

            # Stop all containers
            for container in env.containers.values():
                try:
                    subprocess.run(
                        ["docker", "stop", container.name],
                        capture_output=True,
                        check=False,
                    )
                    subprocess.run(
                        ["docker", "rm", container.name],
                        capture_output=True,
                        check=False,
                    )
                    logger.info(f"Stopped container {container.name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to stop container {container.name}: {e}"
                    )

            self.cleanup_environment(task_id)

    def _remove_directory_with_retry(
        self, dir_path: str, max_retries: int = 3
    ) -> bool:
        """Remove directory with retry logic for robustness"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(dir_path):
                    # Try to make all files writable first (in case of permission issues)
                    for root, dirs, files in os.walk(dir_path):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o755)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o644)

                    shutil.rmtree(dir_path)
                    logger.info(f"Successfully removed directory: {dir_path}")
                    return True
                else:
                    logger.debug(f"Directory already removed: {dir_path}")
                    return True
            except PermissionError as e:
                logger.warning(
                    f"Permission error removing {dir_path} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief delay before retry
            except Exception as e:
                logger.warning(
                    f"Error removing {dir_path} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        logger.error(
            f"Failed to remove directory after {max_retries} attempts: {dir_path}"
        )
        return False

    def cleanup_environment(self, task_id: int) -> None:
        """Clean up temporary files and directories for a task"""
        cleanup_success = True
        temp_dirs_to_clean = []

        # Get temp directories from active environment if it exists
        if task_id in self.active_environments:
            env = self.active_environments[task_id]
            temp_dirs_to_clean = env.temp_dirs.copy()

        # Also check for any orphaned directories for this task
        orphaned_dirs = self._find_orphaned_directories(task_id)
        temp_dirs_to_clean.extend(orphaned_dirs)

        # Remove duplicates
        temp_dirs_to_clean = list(set(temp_dirs_to_clean))

        # Clean up all temporary directories
        for temp_dir in temp_dirs_to_clean:
            success = self._remove_directory_with_retry(temp_dir)
            if not success:
                cleanup_success = False

        # Remove from active environments (do this even if cleanup partially failed)
        if task_id in self.active_environments:
            del self.active_environments[task_id]

        if cleanup_success:
            logger.info(
                f"Successfully cleaned up all resources for task {task_id}"
            )
        else:
            logger.warning(
                f"Cleanup completed with some failures for task {task_id}"
            )

    def _find_orphaned_directories(self, task_id: int = None) -> List[str]:
        """Find orphaned temporary directories that may have been left behind"""
        orphaned_dirs = []
        current_dir = os.getcwd()

        # Pattern matching for temp directories
        patterns = [
            f"temp_auth_task_{task_id}_*"
            if task_id is not None
            else "temp_auth_task_*",
            f"temp_wiki_task_{task_id}"
            if task_id is not None
            else "temp_wiki_task_*",
        ]

        for pattern in patterns:
            import glob

            matching_dirs = glob.glob(os.path.join(current_dir, pattern))
            orphaned_dirs.extend(matching_dirs)

        return orphaned_dirs

    def cleanup_orphaned_directories(self) -> int:
        """Clean up all orphaned temporary directories from previous runs"""
        orphaned_dirs = self._find_orphaned_directories()
        cleaned_count = 0

        logger.info(
            f"Found {len(orphaned_dirs)} orphaned directories to clean up"
        )

        for dir_path in orphaned_dirs:
            if self._remove_directory_with_retry(dir_path):
                cleaned_count += 1

        logger.info(
            f"Cleaned up {cleaned_count}/{len(orphaned_dirs)} orphaned directories"
        )
        return cleaned_count

    def _generate_isolated_auth(
        self, task_id: int, sites: List[str], env: TaskEnvironment
    ) -> None:
        """Generate authentication files for isolated environment with dynamic ports"""
        try:
            import os
            import subprocess

            # Get URLs for this environment
            urls = env.get_urls()

            # Build environment variables with dynamic ports
            env_vars = os.environ.copy()

            # Update environment variables with isolated URLs
            for site in sites:
                if site == "shopping":
                    env_vars["SHOPPING"] = urls["SHOPPING"]
                elif site == "shopping_admin":
                    env_vars["SHOPPING_ADMIN"] = urls["SHOPPING_ADMIN"]
                elif site == "reddit":
                    env_vars["REDDIT"] = urls["REDDIT"]
                elif site == "gitlab":
                    env_vars["GITLAB"] = urls["GITLAB"]

            # Run auto_login.py with modified environment for each site
            for site in sites:
                logger.info(f"Generating auth for {site} on isolated ports...")
                result = subprocess.run(
                    [
                        "python",
                        "browser_env/auto_login.py",
                        "--site_list",
                        site,
                        "--auth_folder",
                        env.auth_dir,
                    ],
                    env=env_vars,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    logger.info(f"✓ Generated auth for {site}")
                else:
                    logger.warning(
                        f"Failed to generate auth for {site}: {result.stderr}"
                    )

        except Exception as e:
            logger.warning(
                f"Error generating isolated auth for task {task_id}: {e}"
            )

    def cleanup_all(self) -> None:
        """Clean up all active environments"""
        task_ids = list(self.active_environments.keys())
        for task_id in task_ids:
            self.stop_environment(task_id)

    def get_environment_config(
        self, task_id: int, original_config: Dict
    ) -> Dict:
        """Get modified config for isolated environment"""
        if task_id not in self.active_environments:
            raise ValueError(f"No environment found for task {task_id}")

        env = self.active_environments[task_id]
        urls = env.get_urls()

        # Create modified config with isolated URLs
        modified_config = original_config.copy()

        # Update URLs based on sites required
        if "shopping" in original_config.get("sites", []):
            modified_config["start_url"] = modified_config[
                "start_url"
            ].replace("localhost:7770", f"localhost:{env.base_port}")
        elif "shopping_admin" in original_config.get("sites", []):
            modified_config["start_url"] = modified_config[
                "start_url"
            ].replace("localhost:7780", f"localhost:{env.base_port + 10}")

        # Handle auth state path - use dynamically generated auth for isolated environment
        if original_config.get("storage_state"):
            # Get the site name from the original auth filename
            original_auth = original_config["storage_state"]
            auth_filename = os.path.basename(original_auth)

            # Look for the isolated auth file generated for this environment
            isolated_auth_path = os.path.join(env.auth_dir, auth_filename)

            if os.path.exists(isolated_auth_path):
                logger.info(
                    f"Using isolated auth file for task {task_id}: {isolated_auth_path}"
                )
                modified_config["storage_state"] = isolated_auth_path
            else:
                logger.warning(
                    f"Isolated auth file not found for task {task_id}, auth may fail"
                )
                # Keep the original path as fallback
                modified_config["storage_state"] = original_auth

        return modified_config


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the Docker isolation manager
    manager = DockerIsolationManager()

    try:
        # Start environment for task 78 (shopping_admin)
        env = manager.start_environment(78, ["shopping_admin"])
        if env:
            print(f"Environment started for task 78:")
            print(f"URLs: {env.get_urls()}")

            # Test config modification
            original_config = {
                "sites": ["shopping_admin"],
                "start_url": "http://localhost:7780/admin",
                "storage_state": "./.auth/shopping_admin_state.json",
            }

            modified_config = manager.get_environment_config(
                78, original_config
            )
            print(f"Modified config: {modified_config}")

            input("Press Enter to cleanup...")

    finally:
        manager.cleanup_all()
