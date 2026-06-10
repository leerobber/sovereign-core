```markdown
# sovereign-core Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill provides guidance on contributing to the `sovereign-core` Python codebase. It covers the project's coding conventions, commit patterns, and testing strategies. By following these patterns, contributors can ensure code consistency and maintainability across the repository.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - **Example:** `user_manager.py`, `data_processor.py`

### Import Style
- Use **relative imports** within the package.
  - **Example:**
    ```python
    from .utils import calculate_hash
    from ..models import User
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - **Example:**
    ```python
    def process_data(data):
        # processing logic
        return result

    __all__ = ['process_data']
    ```

### Commit Messages
- Follow **conventional commit** format.
- Use the `feat` prefix for new features.
- Keep commit messages concise (average ~66 characters).
  - **Example:**  
    ```
    feat: add user authentication middleware
    ```

## Workflows

### Feature Development
**Trigger:** When adding a new feature  
**Command:** `/feature-development`

1. Create a new branch for your feature.
2. Implement the feature following coding conventions.
3. Add or update relevant tests.
4. Commit changes using the `feat` prefix.
    - Example: `feat: implement data export functionality`
5. Push your branch and open a pull request.

### Testing
**Trigger:** When validating code changes  
**Command:** `/run-tests`

1. Identify or create test files using the `*.test.*` naming pattern.
2. Run all tests using your preferred Python test runner (e.g., `pytest`, `unittest`).
3. Ensure all tests pass before submitting code.

## Testing Patterns

- Test files follow the `*.test.*` naming convention.
  - **Example:** `user_manager.test.py`, `api_client.test.py`
- The specific testing framework is not enforced; use your preferred Python test runner.
- Write tests close to the code they validate, using the same import style as the main codebase.

  **Example test file:**
  ```python
  from ..user_manager import process_data

  def test_process_data():
      assert process_data([1, 2, 3]) == [1, 4, 9]
  ```

## Commands
| Command              | Purpose                                   |
|----------------------|-------------------------------------------|
| /feature-development | Start a new feature development workflow  |
| /run-tests           | Run all tests in the codebase             |
```
