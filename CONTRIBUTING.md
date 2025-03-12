# Contributing to Yoke

Thank you for considering contributing to Yoke!

### Reporting Bugs

If you find a bug, please report it by opening an issue on our GitHub repository. Include as much detail as possible to help us understand and reproduce the issue.

### Suggesting Features

Please open an issue on our GitHub repository and describe the feature you would like to see, why you need it, and how it should work.

### Submitting Changes

1. Fork the repository.
2. Create a new branch for your changes (`git checkout -b my-feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin my-feature-branch`).
6. Open a pull request.

### Code Style & Other Rules

Please follow the coding style used in the project. This includes indentation, naming conventions, and commenting. The following list includes a few specific examples of rules to follow:

1. Make sure the PR title and description are used properly. The idea is that without looking at the code a reviewer should be able to understand what was done, how it was done, and why it was done.
    1. The title should describe the issue in a few words. Describe the change itself. For example, "Fixing the badges on the README.md", instead of "Changing README.md". The idea is that the title should tell a reviewer what the PR is doing without having to open the PR.
    2. The description should include more detail as to what changes are being made. It doesn't have to be long, but it should clearly describe the PR. Any other information required to assist reviewers should be included here.
2. Make sure the setting for squashing commits is turned on at merge time. This keeps the history clean and makes it easier to follow changes. Instead of seeing a commit, you will see the PR listed (with it's title and description), so that you get an idea of why that change was made.
3. Please ensure you are using a robust solution when dealing with directories! The `+` operator is not robust and easily breaks. If you are trying to access `~/Yoke`, then you can either end up with `~//Yoke` or `~Yoke`, which are both wrong. Here are two examples of robust solutions:

    1.
    ```python
    from pathlib import Path
    p = Path("~")
    yokeDir = p / "Yoke"
    ```
    2.
    ```python
    import os
    yokeDir = os.path.join("~", "Yoke")
    ```

### Running Tests

Pytest is used for unit tests, please ensure all run before submitting PR.

## Getting Help

If you need help, feel free to open an issue or reach out to the maintainers.

Thank you for contributing!
