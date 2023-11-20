# Makefile

# Commands
ISORT = isort
BLACK = black
PYTEST = pytest

# Directories
SRC_DIR = ./mia
TEST_DIR = ./tests

# Targets
.PHONY: check-format fix-format test

check-format:
	@echo "Checking code format..."
	@$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)
	@$(BLACK) --check $(SRC_DIR) $(TEST_DIR)

fix-format:
	@echo "Fixing code format..."
	@$(ISORT) $(SRC_DIR) $(TEST_DIR)
	@$(BLACK) $(SRC_DIR) $(TEST_DIR)

test:
	@echo "Running tests..."
	@$(PYTEST) $(TEST_DIR)