# Contributing to QSAR Toolbox Assistant

Thank you for considering contributing to QSAR Toolbox Assistant! This document provides guidelines and instructions for contributing.

## Ways to Contribute

1. **Bug Reports**: Report bugs and issues through the GitHub issue tracker
2. **Feature Requests**: Suggest new features or improvements
3. **Documentation**: Improve or add documentation
4. **Code Contributions**: Submit pull requests with bug fixes or new features

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```
   git clone https://github.com/your-username/streamlined_qsar_app.git
   cd streamlined_qsar_app
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set up the environment variables:
   ```
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

## Pull Request Process

1. Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and add tests if applicable
3. Run the cleanup script to remove any unnecessary files:
   ```
   python cleanup.py
   ```
4. Commit your changes with clear commit messages:
   ```
   git commit -m "Add feature: description of your changes"
   ```
5. Push to your branch:
   ```
   git push origin feature/your-feature-name
   ```
6. Submit a pull request to the main repository

## Code Style

- Follow PEP 8 style guidelines for Python code
- Include docstrings for functions, classes, and modules
- Write clear commit messages

## Testing

- Add tests for new features when applicable
- Ensure all tests pass before submitting pull requests

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
