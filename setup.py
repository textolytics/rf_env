#!/usr/bin/env python3
"""
Market Data Platform - Python Setup Configuration
Defines package metadata, dependencies, and installation instructions.
"""

from setuptools import setup, find_packages
import os

# Read version from file
def get_version():
    """Read version from market_data_platform/__init__.py"""
    init_file = os.path.join(
        os.path.dirname(__file__),
        'market_data_platform',
        '__init__.py'
    )
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split("'")[1]
    return '0.1.0'

# Read long description from README
def get_long_description():
    """Read long description from README.md"""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Market Data Platform - Real-time trading data aggregation and analytics'

# Core dependencies (always required)
INSTALL_REQUIRES = [
    'python-dotenv>=0.19.0',
    'PyYAML>=5.4.0',
    'click>=8.0.0',
    'requests>=2.27.0',
    'aiohttp>=3.8.0',
    'websockets>=10.0',
]

# API and web framework dependencies
API_REQUIRES = [
    'Flask>=2.0.0',
    'FastAPI>=0.79.0',
    'uvicorn>=0.18.0',
    'pydantic>=1.9.0',
    'flask-cors>=3.0.10',
]

# Database and persistence dependencies
DATABASE_REQUIRES = [
    'SQLAlchemy>=1.4.0',
    'psycopg2-binary>=2.9.0',
    'redis>=4.0.0',
    'sqlalchemy-utils>=0.37.0',
]

# Data processing and analytics dependencies
DATA_REQUIRES = [
    'pandas>=1.3.0',
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
]

# ZMQ and messaging dependencies
MESSAGING_REQUIRES = [
    'pyzmq>=22.0.0',
    'msgpack>=1.0.0',
]

# Monitoring and observability dependencies
MONITORING_REQUIRES = [
    'prometheus-client>=0.12.0',
    'structlog>=21.1.0',
]

# Testing dependencies
TEST_REQUIRES = [
    'pytest>=6.2.0',
    'pytest-cov>=2.12.0',
    'pytest-asyncio>=0.15.0',
    'pytest-mock>=3.6.0',
    'responses>=0.13.0',
]

# Development dependencies
DEV_REQUIRES = [
    'black>=21.6b0',
    'isort>=5.9.0',
    'flake8>=3.9.0',
    'pylint>=2.9.0',
    'mypy>=0.910',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
]

setup(
    name='market-data-platform',
    version=get_version(),
    author='Market Data Team',
    author_email='team@marketdata.local',
    description='Real-time market data aggregation, processing, and API platform',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/marketdata/platform',
    license='MIT',
    python_requires='>=3.9',
    
    packages=find_packages(
        exclude=['tests', 'tests.*', 'docs', 'build', 'dist']
    ),
    
    package_data={
        'market_data_platform': [
            'config/*.yaml',
            'resources/*.json',
        ],
    },
    
    install_requires=INSTALL_REQUIRES,
    
    extras_require={
        'api': API_REQUIRES,
        'database': DATABASE_REQUIRES,
        'data': DATA_REQUIRES,
        'messaging': MESSAGING_REQUIRES,
        'monitoring': MONITORING_REQUIRES,
        'test': TEST_REQUIRES,
        'dev': DEV_REQUIRES + TEST_REQUIRES,
        'all': (
            API_REQUIRES +
            DATABASE_REQUIRES +
            DATA_REQUIRES +
            MESSAGING_REQUIRES +
            MONITORING_REQUIRES +
            TEST_REQUIRES
        ),
    },
    
    entry_points={
        'console_scripts': [
            'marketdata=market_data_platform.cli:main',
            'marketdata-terminal=market_data_platform.cli:terminal',
            'marketdata-api=market_data_platform.api:serve',
            'marketdata-gateway=market_data_platform.gateway:serve',
        ],
    },
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    keywords='market-data trading finance API gateway analytics',
    
    project_urls={
        'Documentation': 'https://docs.marketdata.local',
        'Source': 'https://github.com/marketdata/platform',
        'Issues': 'https://github.com/marketdata/platform/issues',
    },
    
    zip_safe=False,
    include_package_data=True,
)
