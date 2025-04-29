# QSAR Toolbox Assistant Testing Documentation

This document outlines the testing approach for the QSAR Toolbox Assistant application, including test organization, coverage, and recommendations for future test development.

## Test Structure

The test suite is organized into the following categories:

1. **Unit Tests**:
   - `test_data_formatter.py`: Tests for the data formatting utilities
   - `test_qsar_api.py`: Tests for the QSAR Toolbox API client

2. **API Behavior Tests**:
   - `test_api_behavior.py`: Tests that validate the expected behavior of the API functions without mocking internal implementations

3. **Integration Tests**:
   - `test_integration.py`: Tests that verify components work together correctly
   - `test_app.py`: Tests for the Streamlit application itself

## Current Test Coverage

The current test suite provides coverage for:

- **Data Formatting**: Testing the format_chemical_data, format_calculator_result, and clean_response_data functions
- **API Error Classes**: Validating the error hierarchy for QSAR API interactions
- **JSON Serialization**: Testing the safe_json utility with various data types
- **Agent Function Signatures**: Verifying that agent functions exist and have the expected signatures
- **App Structure**: Testing key app components are accessible and properly structured
- **Session State**: Testing initialization of session state variables
- **Component Integration**: Testing that data flows correctly between formatting components

## Testing Challenges

### LLM Component Testing

Testing the LangChain-based LLM components presents unique challenges:

1. **Complex Chain Construction**: LangChain uses a piping syntax (`prompt | llm | parser`) that is difficult to mock effectively
2. **Global Instances**: The application uses globally defined LLM instances
3. **External API Dependencies**: LLM functions rely on external OpenAI API calls
4. **Non-deterministic Results**: LLM responses can vary, making assertions challenging

Rather than attempting to mock the internal implementation details of the LangChain chains, we adopted a more pragmatic approach that focuses on:

1. Verifying the existence and signatures of agent functions
2. Testing input/output data structures
3. Integration testing of components that don't rely on LLM processing
4. Skipping complex Streamlit-integrated functionality that's challenging to test in isolation

## Future Test Enhancements

To improve test coverage in the future:

1. **Mock OpenAI Responses**: Use tools like `openai-mock` to provide deterministic responses for LLM calls
2. **Snapshot Testing**: Compare LLM responses against recorded "snapshot" responses for regression testing
3. **End-to-End Testing**: Use tools like Playwright or Selenium to test the full application
4. **Property-Based Testing**: Use tools like Hypothesis to test with a variety of input data
5. **Test Environment**: Create a dedicated test environment with controlled LLM responses

## Running Tests

Run all tests with:

```bash
python -m pytest streamlined_qsar_app/tests
```

Run specific test files with:

```bash
python -m pytest streamlined_qsar_app/tests/test_api_behavior.py
```

Run tests with verbose output:

```bash
python -m pytest -v streamlined_qsar_app/tests
```

## Test Dependencies

The test suite depends on:

- pytest
- pytest-asyncio
- unittest.mock

These are all included in the main project requirements.txt file.
