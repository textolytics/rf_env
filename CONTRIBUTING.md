# Contributing to Market Data Platform

Thank you for your interest in contributing to the Market Data Platform! We welcome contributions from the community and are grateful for every pull request, bug report, and suggestion.

## 📋 Code of Conduct

This project is committed to fostering an inclusive and respectful community. Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 How to Contribute

### 1. Reporting Bugs

Before creating a bug report, please check the [issue list](../../issues) to avoid duplicates.

When creating a bug report, include:
- **Clear description** of what happened
- **Expected behavior** vs. actual behavior
- **Steps to reproduce** the issue
- **Your environment** (OS, Python version, etc.)
- **Relevant logs or screenshots** if applicable

### 2. Suggesting Features

We welcome feature suggestions! When proposing a feature:
- Use a clear and descriptive title
- Provide a detailed description of the feature
- Explain why you think it would be useful
- List possible use cases

### 3. Submitting Pull Requests

Before submitting a pull request:

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/market_data_platform.git
   cd market_data_platform
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Set up development environment**
   ```bash
   make install-dev
   ```

4. **Make your changes**
   - Write clean, readable code
   - Follow the code style guidelines (see below)
   - Add tests for new functionality
   - Update documentation

5. **Run tests and linting**
   ```bash
   make lint
   make format
   make test
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

8. **Create a Pull Request**
   - Use a clear title and description
   - Reference related issues
   - Wait for review and address feedback

## 📝 Code Style Guidelines

### Python

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use:
- **Black** for code formatting
- **isort** for import sorting
- **pylint** for code analysis
- **mypy** for type checking

```bash
# Format code
black market_data_platform tests
isort market_data_platform tests

# Run linters
pylint market_data_platform
mypy market_data_platform

# Auto-fix formatting
autopep8 --in-place --aggressive --recursive market_data_platform
```

### Go

We follow [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments):

```bash
# Format code
cd go && go fmt ./...
go vet ./...

# Run linter
golangci-lint run ./...
```

### Rust

We follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

```bash
# Format code
cd rust && cargo fmt --all

# Run clippy
cargo clippy --all -- -D warnings
```

## 🧪 Testing

### Python Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_models.py -v

# Run with coverage
pytest tests/ --cov=market_data_platform --cov-report=html

# Run specific test function
pytest tests/unit/test_models.py::test_market_data_creation -v
```

### Go Tests

```bash
# Run all tests
cd go && go test ./... -v

# Run with race detection
go test ./... -race

# Generate coverage report
go test ./... -cover -coverprofile=coverage.out
```

### Rust Tests

```bash
# Run all tests
cd rust && cargo test --release

# Run specific test
cargo test --release processor_test

# Run with backtrace
RUST_BACKTRACE=1 cargo test --release
```

## 📚 Documentation

When adding features, please include:
- **Docstrings** for functions and classes
- **Type hints** for function parameters and returns
- **README updates** if applicable
- **API documentation** for new endpoints
- **Comments** for complex logic

### Docstring Format (Python)

```python
def process_market_data(symbol: str, data: dict) -> dict:
    """
    Process market data for a given symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USDT')
        data: Raw market data dictionary
    
    Returns:
        Processed market data dictionary
    
    Raises:
        ValueError: If symbol or data is invalid
    """
```

## 🔄 Development Workflow

1. **Create a branch** from `main` or `develop`
2. **Make changes** following code style guidelines
3. **Write tests** for new functionality
4. **Run all checks**:
   ```bash
   make lint
   make format
   make test
   ```
5. **Submit PR** with clear description
6. **Address feedback** from reviewers
7. **Merge** when approved and CI passes

## 📦 Release Process

We use [semantic versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Updating Version

```bash
# Update version number
bump2version patch  # 1.0.0 -> 1.0.1
bump2version minor  # 1.0.0 -> 1.1.0
bump2version major  # 1.0.0 -> 2.0.0
```

## 🐛 Common Issues

### Issue: Tests fail locally but pass on CI

**Solution**: Make sure you have the same dependencies:
```bash
pip install -r requirements.txt -r requirements-dev.txt
docker-compose up -d  # Start services
```

### Issue: Black formatting conflicts with pylint

**Solution**: Black takes precedence. Format with Black first, then fix other issues:
```bash
black market_data_platform
pylint market_data_platform
```

### Issue: Import errors after changes

**Solution**: Reinstall package in editable mode:
```bash
pip install -e .
```

## 🎯 What We're Looking For

- **Quality code** with tests
- **Clear commit messages**
- **Documentation updates**
- **Discussion** of design decisions
- **Adherence** to existing patterns

## ⚠️ What We Won't Accept

- Code without tests
- Breaking changes without discussion
- Committing secrets or credentials
- Unformatted code
- Incomplete documentation

## 🏆 Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project homepage

## 📞 Questions?

- **GitHub Issues**: Ask questions in issues
- **Discussions**: Use GitHub Discussions for broader topics
- **Email**: team@marketdata.local

---

Thank you for contributing to make Market Data Platform better! 🎉
