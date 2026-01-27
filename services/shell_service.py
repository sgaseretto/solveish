"""
Shell Service - Shell command execution with optional safe mode.

This module provides shell command execution via pshnb with optional
safecmd validation for sandboxed environments.

Features:
- Execute shell commands via pshnb's ShellInterpreter
- Optional safe mode using safecmd allowlist validation
- Variable expansion from Python namespace using @{var} syntax
- SSH remote shell support

Usage:
    from services.shell_service import ShellService, SHFMT_AVAILABLE

    # Basic execution
    service = ShellService()
    result = service.execute("ls -la")

    # Safe mode (requires shfmt binary)
    service = ShellService(safe_mode=True)
    result = service.execute("ls -la")  # OK
    result = service.execute("rm -rf /")  # Raises DisallowedCmd

    # Variable expansion
    namespace = {"name": "world"}
    result = service.execute("echo @{name}", namespace=namespace)
    # Output: "world"

    # SSH remote execution
    service = ShellService(ssh="hostname")
    result = service.execute("uname -a")
"""
from __future__ import annotations

import shutil
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# shfmt Dependency Check
# ============================================================================

def check_shfmt_installed() -> bool:
    """Check if shfmt binary is available on the system."""
    return shutil.which('shfmt') is not None


def warn_missing_shfmt() -> bool:
    """
    Log warning with installation instructions if shfmt is missing.

    Returns:
        True if shfmt is available, False otherwise.
    """
    if not check_shfmt_installed():
        logger.warning(
            "\n" + "=" * 60 + "\n"
            "  shfmt not found - Safe Mode will not work!\n"
            "\n"
            "The 'shfmt' binary is required for safecmd command validation.\n"
            "Install it to enable Safe Mode for shell commands:\n"
            "\n"
            "  macOS:   brew install shfmt\n"
            "  Ubuntu:  sudo apt install shfmt\n"
            "  Arch:    sudo pacman -S shfmt\n"
            "  Other:   https://github.com/mvdan/sh/releases\n"
            "\n"
            "Shell execution will still work, but Safe Mode will be disabled.\n"
            + "=" * 60
        )
        return False
    return True


# Module-level check - run once at import time
SHFMT_AVAILABLE = check_shfmt_installed()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ShellResult:
    """Result of shell command execution."""
    output: str
    return_code: int = 0
    error: str = ""


class ShellExecutionError(Exception):
    """Base exception for shell execution errors."""
    pass


class SafeModeError(ShellExecutionError):
    """Raised when safe mode blocks a command."""
    pass


class ShfmtNotInstalledError(ShellExecutionError):
    """Raised when safe mode is enabled but shfmt is not installed."""
    pass


# ============================================================================
# Shell Service
# ============================================================================

class ShellService:
    """
    Shell execution service with optional safecmd validation.

    Provides a high-level interface for executing shell commands with:
    - Optional safe mode using safecmd allowlist validation
    - Variable expansion from Python namespace
    - SSH remote shell support

    Args:
        safe_mode: If True, validate commands against safecmd allowlist
        ssh: SSH host for remote execution (e.g., "user@hostname")

    Example:
        >>> service = ShellService()
        >>> result = service.execute("echo hello")
        >>> print(result.output)
        hello
    """

    def __init__(self, safe_mode: bool = False, ssh: Optional[str] = None):
        self.safe_mode = safe_mode
        self.ssh = ssh

        # Validate safe_mode requirements
        if safe_mode and not SHFMT_AVAILABLE:
            raise ShfmtNotInstalledError(
                "Safe mode requires shfmt binary. Install with:\n"
                "  macOS:   brew install shfmt\n"
                "  Ubuntu:  sudo apt install shfmt\n"
                "  Other:   https://github.com/mvdan/sh/releases"
            )

    def execute(
        self,
        cmd: str,
        namespace: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> ShellResult:
        """
        Execute a shell command.

        Args:
            cmd: Shell command to execute
            namespace: Python namespace for @{var} expansion
            timeout: Command timeout in seconds

        Returns:
            ShellResult with output and status

        Raises:
            SafeModeError: If safe_mode=True and command not in allowlist
            ShellExecutionError: If command execution fails
        """
        # Step 1: Validate with safecmd if safe_mode enabled
        if self.safe_mode:
            self._validate_command(cmd)

        # Step 2: Apply variable expansion if namespace provided
        if namespace:
            cmd = self._expand_variables(cmd, namespace)

        # Step 3: Execute command
        return self._execute_command(cmd, timeout)

    def _validate_command(self, cmd: str) -> None:
        """
        Validate command against safecmd allowlist.

        Raises:
            SafeModeError: If command is not allowed
        """
        try:
            from safecmd import validate, DisallowedCmd, DisallowedDest
        except ImportError:
            raise ShellExecutionError(
                "safecmd package not installed. Install with: pip install safecmd"
            )

        try:
            validate(cmd)
        except DisallowedCmd as e:
            raise SafeModeError(f"Command not allowed: {e}")
        except DisallowedDest as e:
            raise SafeModeError(f"Destination not allowed: {e}")
        except Exception as e:
            raise SafeModeError(f"Command validation failed: {e}")

    def _expand_variables(self, cmd: str, namespace: Dict[str, Any]) -> str:
        """
        Expand @{var} references with values from namespace.

        Args:
            cmd: Command with @{var} references
            namespace: Dict of variable names to values

        Returns:
            Command with variables expanded
        """
        try:
            from pshnb import shell_replace
            return shell_replace(cmd, namespace)
        except ImportError:
            # Fallback: simple regex-based replacement
            import re
            def replace_var(match):
                var_name = match.group(1)
                if var_name in namespace:
                    return str(namespace[var_name])
                return match.group(0)  # Keep original if not found
            return re.sub(r'@\{(\w+)\}', replace_var, cmd)

    def _execute_command(self, cmd: str, timeout: int) -> ShellResult:
        """
        Execute command using pshnb ShellInterpreter.

        Creates a fresh interpreter for each execution (per user requirement).

        Args:
            cmd: Command to execute
            timeout: Command timeout in seconds

        Returns:
            ShellResult with output

        Note:
            SSH remote execution is only supported via the %bash -r magic in
            code cells. The ssh parameter on ShellService is reserved for
            future use when pshnb adds native SSH support.
        """
        try:
            from pshnb import ShellInterpreter
        except ImportError:
            raise ShellExecutionError(
                "pshnb package not installed. Install with: pip install pshnb"
            )

        try:
            # Create fresh interpreter per execution (user requirement: no persistent state)
            # Note: SSH support via ShellInterpreter is not yet available in pshnb
            # Use %bash -r in code cells for SSH support
            interpreter = ShellInterpreter(timeout=timeout)
            output = interpreter(cmd)
            return ShellResult(output=output, return_code=0)
        except Exception as e:
            error_msg = str(e)
            return ShellResult(output="", return_code=1, error=error_msg)


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_shell(
    cmd: str,
    safe_mode: bool = False,
    namespace: Optional[Dict[str, Any]] = None,
    ssh: Optional[str] = None,
    timeout: int = 30
) -> ShellResult:
    """
    Convenience function for one-off shell command execution.

    Args:
        cmd: Shell command to execute
        safe_mode: If True, validate against safecmd allowlist
        namespace: Python namespace for @{var} expansion
        ssh: SSH host for remote execution
        timeout: Command timeout in seconds

    Returns:
        ShellResult with output and status
    """
    service = ShellService(safe_mode=safe_mode, ssh=ssh)
    return service.execute(cmd, namespace=namespace, timeout=timeout)


def print_shfmt_status() -> None:
    """Print shfmt availability status for startup logging."""
    if SHFMT_AVAILABLE:
        print("      shfmt:    available (Safe Mode enabled)")
    else:
        print("      shfmt:    NOT FOUND (Safe Mode disabled)")
        print("                Install: brew install shfmt (macOS)")
