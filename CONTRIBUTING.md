# Contributing to Plant Disease Detection System

We welcome contributions to improve the Plant Disease Detection System. This document provides guidelines for contributing to the project.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Describe the issue in detail
- Include steps to reproduce
- Mention the environment (OS, Python version, etc.)

### Feature Requests

- Open an issue describing the feature
- Explain the use case and benefits
- Discuss implementation approach if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes following our coding standards
4. Write or update tests as needed
5. Update documentation
6. Commit changes (`git commit -m 'Add feature'`)
7. Push to branch (`git push origin feature/improvement`)
8. Create a Pull Request

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage

### Documentation

- Update README.md if adding new features
- Document API changes
- Include examples where helpful

## Project Structure

```
plant-disease-detection/
├── src/              # Source code modules
├── tests/            # Test files
├── scripts/          # Utility scripts
├── docs/             # Documentation
└── weights/          # Model weights
```

## Development Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest tests/`

## Questions?

Feel free to open an issue for any questions about contributing.