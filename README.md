# FreeDOM Exchange Market Summary API - Test Suite

## Overview

This repository contains a comprehensive regression test suite for the **FreeDOM Exchange Market Summary API**. The test suite is built using [Robot Framework](https://robotframework.org/) and validates all public trading pairs to ensure non-empty price and volume data, data consistency, and API reliability.

**API Endpoint:** `https://api.exchange.freedx.com/spot/api/v3.2/market_summary`

---

## Prerequisites

- **Python 3.8+** (Recommended: Python 3.13)
- **Robot Framework 7.4.1+**
- **pip** (Python package manager)
- **Virtual Environment** (recommended)

---

## Setup Instructions

### 1. Clone/Navigate to Project Directory

```bash
cd /root/rf_env
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python3 -m venv rf_env

# Activate virtual environment
source rf_env/bin/activate  # On Windows: rf_env\Scripts\activate
```

### 3. Install Required Dependencies

The project includes a Python virtual environment with pre-installed dependencies. If needed, install dependencies manually:

```bash
pip install robot-framework==7.4.1
pip install requests-library
pip install robotframework-jsonlibrary
```

Or install from requirements (if available):

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
robot --version
python --version
```

---

## Test Suite Structure

### Test File Location
- **File:** `market_summary_api_tests.robot`
- **Lines:** 400+ test steps across 18 test cases

### Test Categories

#### 1. **Public Endpoint Validation** (4 tests)
- Endpoint accessibility (HTTP 200)
- Response is valid JSON array
- Response body is not empty
- Content-Type is application/json

#### 2. **Price Validation** (3 tests)
- All trading pairs have non-empty last price
- All trading pairs have non-empty ask prices (lowestAsk)
- All trading pairs have non-empty bid prices (highestBid)

#### 3. **24-Hour Price Range Validation** (2 tests)
- All pairs have 24-hour high prices
- All pairs have 24-hour low prices

#### 4. **Metadata Validation** (4 tests)
- All pairs have base currency
- All pairs have quote currency
- All pairs have active status defined
- All volumes are non-empty

#### 5. **Data Consistency Validation** (4 tests)
- All prices are non-negative
- High price ≥ Low price (no inverted ranges)
- Last price within 24-hour range
- Ask price ≥ Bid price (valid spread)

#### 6. **Volume Validation** (1 test)
- All volumes are non-negative

#### 7. **Percentage Change Validation** (1 test)
- Percentage change within valid range (-100% to +1000%)

#### 8. **Data Integrity & Completeness** (3 tests)
- No duplicate trading pairs
- Response is complete (not truncated)
- Minimum expected trading pairs (≥600)

---

## Running the Test Suite

### Run All Tests

```bash
robot market_summary_api_tests.robot
```

### Run Specific Test by Name

```bash
robot -t "Public Endpoint - Market Summary Returns 200" market_summary_api_tests.robot
```

### Run Tests by Tag

```bash
# Run only smoke tests
robot --include smoke market_summary_api_tests.robot

# Run only price validation tests
robot --include price-validation market_summary_api_tests.robot

# Run all public endpoint tests
robot --include public market_summary_api_tests.robot

# Exclude certain tests
robot --exclude data-consistency market_summary_api_tests.robot
```

### Generate Custom Reports

```bash
# Specify output directory
robot --outputdir results/ market_summary_api_tests.robot

# Generate report with custom name
robot --output results/execution.xml --log results/log.html market_summary_api_tests.robot

# Disable report generation
robot --output NONE --log NONE market_summary_api_tests.robot
```

### Run with Different Verbosity Levels

```bash
# Verbose output
robot --loglevel DEBUG market_summary_api_tests.robot

# Less verbose
robot --loglevel INFO market_summary_api_tests.robot
```

---

## Test Execution Variables

### Configurable Variables

Edit `*** Variables ***` section in `market_summary_api_tests.robot`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `${BASE_URL}` | `https://api.exchange.freedx.com` | API base URL |
| `${ENDPOINT}` | `/spot/api/v3.2/market_summary` | Market summary endpoint |
| `${TIMEOUT}` | `10` | HTTP request timeout (seconds) |

### Modify Variables from Command Line

```bash
robot --variable BASE_URL:https://staging-api.exchange.freedx.com market_summary_api_tests.robot
robot --variable TIMEOUT:20 market_summary_api_tests.robot
```

---

## Understanding Test Results

### Report Files Generated

After execution, the following files are created:

- **`report.html`** - Executive summary with overall statistics
- **`log.html`** - Detailed execution log with keyword-level information
- **`output.xml`** - Machine-readable test results (used for reporting tools)

### Opening Reports

```bash
# Open in default browser (on Linux)
xdg-open report.html

# Or use direct file path
firefox report.html
```

### Exit Codes

- **0** - All tests passed ✓
- **1** - One or more tests failed ✗
- **2** - Critical error during execution (e.g., suite not found)

### Interpreting Test Output

```
Public Endpoint - Market Summary Returns 200           PASS
Public Pairs - All Have Non-Empty Last Price          PASS
Public Pairs - All Prices Are Non-Negative            FAIL
    AssertionError: Pairs with negative prices: [BTC_USD]
```

---

## Required Libraries

### Core Robot Framework Libraries
- **BuiltIn** - Standard Robot Framework keywords
- **Collections** - Dictionary/List manipulation
- **String** - String operations
- **DateTime** - Date/time operations

### Third-Party Libraries
- **RequestsLibrary** - HTTP request handling
- **JSONLibrary** - JSON parsing and validation

### Library Import Statements

```robot
Library           RequestsLibrary
Library           Collections
Library           String
Library           DateTime
Library           JSONLibrary
```

---

## Troubleshooting

### Issue: Connection Timeout

**Problem:** Test fails with `Connection timeout`

**Solution:**
```bash
# Increase timeout value
robot --variable TIMEOUT:30 market_summary_api_tests.robot
```

### Issue: No Module Named 'RequestsLibrary'

**Problem:** `ModuleNotFoundError: No module named 'RequestsLibrary'`

**Solution:**
```bash
pip install requests-library
```

### Issue: API Endpoint Unreachable

**Problem:** All tests fail with connection errors

**Solution:**
```bash
# Test connectivity
curl -I https://api.exchange.freedx.com/spot/api/v3.2/market_summary

# Check network/firewall
ping api.exchange.freedx.com
```

### Issue: JSON Parsing Errors

**Problem:** `JSONLibrary` keyword errors

**Solution:**
```bash
pip install robotframework-jsonlibrary
```

### Issue: Test Not Found

**Problem:** `Execution aborted: Suite 'market_summary_api_tests' not found`

**Solution:**
```bash
# Verify file exists
ls -la market_summary_api_tests.robot

# Run from correct directory
cd /root/rf_env && robot market_summary_api_tests.robot
```

---

## Example: Complete Execution Workflow

```bash
# 1. Navigate to project
cd /root/rf_env

# 2. Activate virtual environment (if applicable)
source bin/activate

# 3. Run full test suite
robot market_summary_api_tests.robot

# 4. View results
firefox report.html

# 5. Run specific tag subset
robot --include smoke market_summary_api_tests.robot

# 6. Export results for CI/CD
robot --outputdir artifacts/ market_summary_api_tests.robot
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: API Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.13
      - name: Install dependencies
        run: |
          pip install robot-framework
          pip install requests-library
          pip install robotframework-jsonlibrary
      - name: Run tests
        run: robot market_summary_api_tests.robot
      - name: Upload reports
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: test-results
          path: |
            report.html
            log.html
            output.xml
```

---

## Test Tags Reference

| Tag | Description | Count |
|-----|-------------|-------|
| `smoke` | Quick smoke tests | 1 |
| `public` | Public endpoint tests | 18 |
| `positive` | Expected behavior tests | 4 |
| `price-validation` | Price field validation | 3 |
| `price-range-validation` | 24h high/low validation | 2 |
| `metadata-validation` | Pair metadata tests | 4 |
| `orderbook-validation` | Bid/ask spread validation | 2 |
| `data-consistency` | Price/volume logic validation | 4 |
| `volume-validation` | Volume field validation | 2 |
| `change-validation` | Percentage change validation | 1 |
| `data-integrity` | Duplicate/completeness validation | 2 |
| `data-completeness` | Minimum pairs count validation | 1 |
| `headers` | HTTP header validation | 1 |
| `instruments` | Trading pair validation | 12 |

---

## API Response Schema

### Sample Response Structure

```json
[
  {
    "symbol": "BTC_USDT",
    "last": 42500.50,
    "lowestAsk": 42501.00,
    "highestBid": 42500.00,
    "volume": 1250000.00,
    "high24Hr": 43200.00,
    "low24Hr": 41800.00,
    "base": "BTC",
    "quote": "USDT",
    "active": true,
    "percentageChange": 2.50
  },
  ...
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | String | Trading pair identifier (e.g., BTC_USDT) |
| `last` | Number | Last trade price |
| `lowestAsk` | Number | Lowest asking price in order book |
| `highestBid` | Number | Highest bidding price in order book |
| `volume` | Number | 24-hour trading volume |
| `high24Hr` | Number | Highest price in last 24 hours |
| `low24Hr` | Number | Lowest price in last 24 hours |
| `base` | String | Base currency code |
| `quote` | String | Quote currency code |
| `active` | Boolean | Whether pair is actively traded |
| `percentageChange` | Number | 24-hour price change percentage |

---

## Performance Metrics

### Expected Execution Time
- **Full Suite:** ~30-60 seconds (depends on API response time)
- **Per Test:** ~2-4 seconds average

### Typical Test Results
- **Pass Rate:** 95%+ (depends on API data quality)
- **Critical Tests:** 8 (endpoint validation, basic data structure)

---

## Contributing

To add new tests:

1. Open `market_summary_api_tests.robot`
2. Add new test case in appropriate section
3. Use existing keywords and patterns
4. Add meaningful tags for categorization
5. Document the test purpose

### Example New Test

```robot
Public Pairs - All Have Minimum 6 Decimal Places
    [Documentation]    Validate price precision for accurate trading
    [Tags]    public    instruments    precision-validation
    Create Session    market_api    ${BASE_URL}    timeout=${TIMEOUT}
    ${response}=    GET On Session    market_api    ${ENDPOINT}
    ${data}=    Evaluate    json.loads(r'''${response.text}''')    json
    # Add validation logic
    Delete All Sessions
```

---

## References

- [Robot Framework Documentation](https://robotframework.org/)
- [RequestsLibrary](https://github.com/MarketSquare/robotframework-requests)
- [Robot Framework JSON Library](https://github.com/robotframework-thailand/robotframework-jsonlibrary)
- [FreeDOM Exchange API Docs](https://api.exchange.freedx.com)

---

## Support & Questions

For issues or questions:
1. Check the **Troubleshooting** section
2. Review test output in `log.html`
3. Verify API endpoint is accessible: `curl https://api.exchange.freedx.com/spot/api/v3.2/market_summary`

---

## License

This test suite is provided for testing and quality assurance purposes.

---

**Last Updated:** December 30, 2025  
**Robot Framework Version:** 7.4.1  
**API Version:** v3.2
