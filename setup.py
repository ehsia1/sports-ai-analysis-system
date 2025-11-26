"""Setup script for quick installation."""

from setuptools import find_packages, setup

setup(
    name="sports-betting",
    version="0.1.0",
    description="AI/ML powered sports betting analysis system",
    author="Sports Betting AI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.25.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "torch>=2.0.0",
        "nfl-data-py>=0.3.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
        "click>=8.1.0",
        "rich>=13.5.0",
        "pydantic>=2.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "sports-betting-analyze=sports_betting.cli.analyzer:main",
        ],
    },
)