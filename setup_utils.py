import os
import sys
from typing import Dict, Optional

from .executors import CommandExecutor, FinallyCallback, LineCallback


_COMMAND_EXECUTOR: Optional[CommandExecutor] = None


def _get_command_executor() -> CommandExecutor:
    global _COMMAND_EXECUTOR  # pylint: disable=global-statement
    if _COMMAND_EXECUTOR is None:
        _COMMAND_EXECUTOR = CommandExecutor()

    return _COMMAND_EXECUTOR


def is_installer_running() -> bool:
    return _get_command_executor().is_running


def get_installer_exit_code() -> int:
    return _get_command_executor().exit_code


def get_required_modules() -> Dict[str, bool]:
    import pkgutil  # pylint: disable=import-outside-toplevel

    modules = {
        'torch': False,
        'numpy': False,
        'smplx': False,
        'scipy': False,
        'chumpy': False,
        'clip': False,
    }

    modules.update(
        (m.name, True) for m in pkgutil.iter_modules() if m.name in modules
    )

    return modules


def install_python_modules(use_gpu=False, line_callback: Optional[LineCallback] = None, finally_callback: Optional[FinallyCallback] = None):
    site_packages_path = next((p for p in sys.path if p.endswith('site-packages')), None)
    target_option = ['--target', site_packages_path] if site_packages_path else []

    _get_command_executor().exec_command(
        # ensurepip
        sys.executable, '-m', 'ensurepip',
        line_callback=line_callback,
        finally_callback=lambda e: e.exec_command(
            # force install setuptools
            sys.executable, '-m', 'pip', 'install',
            *target_option,
            '--disable-pip-version-check',
            '--no-input',
            '--ignore-installed',
            'setuptools',
            line_callback=line_callback,
            finally_callback=lambda e: e.exec_command(
                # and then install depending modules
                sys.executable, '-m', 'pip', 'install',
                *target_option,
                '--disable-pip-version-check',
                '--no-input',
                # '--upgrade',
                # '--upgrade-strategy', 'only-if-needed',
                # '--no-cache-dir',
                '--exists-action', 'i',
                # '--ignore-installed',
                '--extra-index-url', 'https://download.pytorch.org/whl/cu116' if use_gpu else 'https://download.pytorch.org/whl/cpu',
                'torch',
                'numpy<1.24.0',
                'git+https://github.com/openai/CLIP.git',
                'smplx',
                'scipy',
                'chumpy',
                line_callback=line_callback, finally_callback=finally_callback
            )
        )
    )


def uninstall_python_modules(line_callback: Optional[LineCallback] = None, finally_callback: Optional[FinallyCallback] = None):
    _get_command_executor().exec_command(
        sys.executable, '-m', 'pip', 'uninstall',
        '--yes',
        'torch',
        'typing-extensions',
        # 'certifi',
        'chumpy',
        'clip',
        'ftfy',
        # 'idna',
        # 'numpy',
        'pillow',
        'regex',
        'scipy',
        'six',
        'smplx',
        'torchvision',
        'tqdm',
        # 'urllib3',
        'wcwidth',
        line_callback=line_callback, finally_callback=finally_callback
    )


def list_python_modules(line_callback: Optional[LineCallback] = None, finally_callback: Optional[FinallyCallback] = None):
    _get_command_executor().exec_command(
        sys.executable, '-m', 'pip', 'list', '-v',
        line_callback=line_callback, finally_callback=finally_callback
    )
