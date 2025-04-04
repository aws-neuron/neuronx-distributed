"""Unit tests for the JSON utility functions.

This module contains test cases for the JSON file handling utilities, specifically
focusing on the get_dict_from_json function which reads and parses JSON files
into Python dictionaries.

The tests cover various scenarios including:

- Valid JSON files with different structures
- Empty JSON files
- Invalid JSON content
- File system related errors
- Encoding issues
"""

import pytest
import json
import os
from typing import Dict, Any
from neuronx_distributed.utils.utils import get_dict_from_json


class TestGetDictFromJson:
    """Test suite for get_dict_from_json utility function.

    This class contains comprehensive tests for the get_dict_from_json function,
    verifying its behavior with various types of input files and error conditions.
    It ensures the function correctly handles both valid and invalid JSON files,
    proper error reporting, and support for different file path types.
    """

    def test_valid_json(self, tmp_path):
        """Test reading a valid JSON file."""
        test_data = {"key": "value", "number": 42}
        json_file = tmp_path / "valid.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        result = get_dict_from_json(json_file)

        assert result == test_data

    def test_empty_json(self, tmp_path):
        """Test reading an empty JSON file (valid JSON)."""
        test_data = {}
        json_file = tmp_path / "empty.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        result = get_dict_from_json(json_file)

        assert result == test_data

    def test_invalid_json(self, tmp_path):
        """Test reading an invalid JSON file."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w", encoding="utf-8") as f:
            f.write("{invalid json")

        with pytest.raises(ValueError) as exc_info:
            get_dict_from_json(json_file)
        assert "Failed to parse JSON file" in str(exc_info.value)

    def test_file_not_found(self):
        """Test reading a non-existent file."""
        non_existent_file = "does_not_exist.json"

        with pytest.raises(ValueError) as exc_info:
            get_dict_from_json(non_existent_file)
        assert "JSON file not found" in str(exc_info.value)

    def test_non_utf8_file(self, tmp_path):
        """Test reading a file with non-UTF-8 encoding."""
        json_file = tmp_path / "non_utf8.json"
        with open(json_file, "wb") as f:
            f.write(b'\xff\xfe{"key": "value"}')  # Invalid UTF-8

        with pytest.raises(ValueError) as exc_info:
            get_dict_from_json(json_file)
        assert "Failed to parse JSON file" in str(exc_info.value)

    def test_nested_structure(self, tmp_path):
        """Test reading a JSON file with nested structure."""
        test_data = {"level1": {"level2": {"level3": ["item1", "item2"], "number": 42}}}
        json_file = tmp_path / "nested.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        result = get_dict_from_json(json_file)

        assert result == test_data
        assert result["level1"]["level2"]["level3"] == ["item1", "item2"]

    def test_pathlib_path(self, tmp_path):
        """Test reading a JSON file using PathLib path."""
        test_data = {"key": "value"}
        json_file = tmp_path / "pathlib.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        result = get_dict_from_json(json_file)

        assert result == test_data
