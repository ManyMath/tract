#!/usr/bin/env python3
"""Fail before hooks can install a Rust toolchain or target on a shared host."""
import re
import shutil
import subprocess
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parent.parent
native = root / 'packages/tract/native'
config = tomllib.loads((native / 'rust-toolchain.toml').read_text())['toolchain']
channel = config['channel']
if not re.fullmatch(r'\d+\.\d+\.\d+', channel):
    raise SystemExit('The hook requires an exact provisioned Rust version')
rustup = shutil.which('rustup')
if not rustup:
    raise SystemExit('Required Rust toolchain manager not found; provision rustup before running hooks')

def run(*args):
    return subprocess.check_output([rustup, *args], text=True, cwd=root)

installed = run('toolchain', 'list').splitlines()
if not any(line.split()[0].startswith(channel + '-') for line in installed):
    raise SystemExit(f'Required Rust {channel} is not provisioned; no installation attempted')
targets = set(run('target', 'list', '--installed', '--toolchain', channel).splitlines())
missing = set(config.get('targets', [])) - targets
if missing:
    raise SystemExit(f'Required Rust targets are not provisioned: {sorted(missing)}; no installation attempted')
components = run('component', 'list', '--installed', '--toolchain', channel).splitlines()
for required in ('rustfmt-', 'clippy-'):
    if not any(item.startswith(required) for item in components):
        raise SystemExit(f'Required Rust component {required[:-1]} is not provisioned')
package = tomllib.loads((native / 'Cargo.toml').read_text())
if package['package'].get('publish') is not False:
    raise SystemExit('Internal tract_ffi crate must remain publish = false')
print(run('run', channel, 'rustc', '--version', '--verbose').strip())
print(f'Provisioned toolchain and all {len(config["targets"])} declared hook targets verified; internal crate publication disabled')
