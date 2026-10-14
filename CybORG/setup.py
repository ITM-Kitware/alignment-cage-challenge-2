import sys
from pathlib import Path
from setuptools import find_packages, setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 10, \
    "This CybORG release requires Python 3.10 or newer." \
    + "Please install it before proceeding."

PACKAGE_ROOT = Path(__file__).resolve().parent

with open(PACKAGE_ROOT / 'Requirements.txt') as f:
    requirements = f.read().splitlines()

with open(PACKAGE_ROOT / 'CybORG/version.txt') as f:
    CYBORG_VERSION = f.read().strip()


setup(
    name="CybORG",
    version=CYBORG_VERSION,
    python_requires=">=3.10",
    packages=find_packages(exclude=["CybORG.Tests", "CybORG.Tests.*"]),
    package_data={"CybORG": [
        "version.txt", "Shared/Config/*.ini", "Shared/Scenarios/*.yaml",
        "Shared/Scenarios/**/*.yaml", "Shared/Scenarios/**/*.json",
    ]},
    install_requires=requirements,
    description="A Cyber Security Research Environment",
)
