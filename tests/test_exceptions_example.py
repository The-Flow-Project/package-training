"""
Example tests to demonstrate exception handling for TrOCR training.

Run with: pytest tests/test_exceptions_example.py -v
"""

import pytest
from PIL import Image, UnidentifiedImageError
from flow_training.train import Trainer


class TestConvertToRgb:
    """Test exception handling in _convert_to_rgb method."""

    def test_valid_pil_image_returns_rgb(self):
        """Test that a valid PIL Image is converted to RGB."""
        # Create a test image in grayscale
        img = Image.new("L", (100, 100), color=128)
        result = Trainer._convert_to_rgb(img)

        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_invalid_path_raises_file_not_found(self, tmp_path):
        """Test that invalid file path raises FileNotFoundError."""
        invalid_path = tmp_path / "nonexistent_image.jpg"

        with pytest.raises(FileNotFoundError):
            Trainer._convert_to_rgb(str(invalid_path))

    def test_invalid_image_format_raises_unidentified_error(self, tmp_path):
        """Test that invalid image format raises UnidentifiedImageError."""
        # Create a file with .jpg extension but text content
        fake_image = tmp_path / "fake_image.jpg"
        fake_image.write_text("This is not an image")

        with pytest.raises(UnidentifiedImageError):
            Trainer._convert_to_rgb(str(fake_image))

    def test_empty_file_raises_error(self, tmp_path):
        """Test that empty file raises an exception."""
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_bytes(b"")

        with pytest.raises((UnidentifiedImageError, OSError)):
            Trainer._convert_to_rgb(str(empty_file))


class TestExceptionTypes:
    """Demonstrate different exception types that can occur."""

    def test_key_error_example(self):
        """Show how KeyError is raised when field is missing."""
        example = {"text": "hello"}  # Missing "image" field

        with pytest.raises(KeyError):
            _ = example["image"]

    def test_type_error_example(self):
        """Show how TypeError occurs with wrong argument types."""
        with pytest.raises((TypeError, AttributeError)):
            # Image.open expects str, bytes, or PathLike
            Image.open(12345)  # Invalid type

    def test_attribute_error_example(self):
        """Show how AttributeError occurs with wrong attributes."""
        example = {"data": "not an image"}

        with pytest.raises(AttributeError):
            # strings don't have .height attribute
            _ = example["data"].height


class TestExceptionInspection:
    """Show how to inspect exceptions in detail."""

    def test_inspect_exception_details(self, tmp_path):
        """Demonstrate how to inspect exception details."""
        fake_image = tmp_path / "fake.jpg"
        fake_image.write_text("not an image")

        try:
            Trainer._convert_to_rgb(str(fake_image))
        except UnidentifiedImageError as e:
            # You can inspect the exception
            assert isinstance(e, Exception)
            assert hasattr(e, "args")
            assert len(e.args) >= 0

            # The exception message
            assert str(e)  # Non-empty message


def test_exception_hierarchy():
    """Show exception hierarchy in Python."""
    # All exceptions inherit from BaseException
    assert issubclass(FileNotFoundError, Exception)
    assert issubclass(KeyError, Exception)
    assert issubclass(UnidentifiedImageError, Exception)

    # Some exceptions inherit from OSError
    assert issubclass(FileNotFoundError, OSError)

    # This is why we can catch multiple with:
    # except (FileNotFoundError, UnidentifiedImageError):
    #     pass


class TestCommonExceptionPatterns:
    """Show common exception handling patterns."""

    def test_multiple_specific_exceptions(self):
        """Pattern: Catch multiple specific exceptions."""
        example = {}

        # Try to access missing key
        with pytest.raises((KeyError, ValueError)):
            try:
                _ = example["image"]  # Raises KeyError
            except KeyError:
                # Convert to ValueError with better message
                raise ValueError("'image' field is required") from None

    def test_exception_with_context(self):
        """Pattern: Preserve exception context with 'from'."""
        try:
            try:
                raise FileNotFoundError("Image not found")
            except FileNotFoundError as e:
                # This preserves the original exception in __cause__
                raise ValueError("Cannot process image") from e
        except ValueError as e:
            assert e.__cause__.__class__.__name__ == "FileNotFoundError"


# Test-Lauf Beispiel:
#
# pytest tests/test_exceptions_example.py -v
#
# Output:
# tests/test_exceptions_example.py::TestConvertToRgb::test_valid_pil_image_returns_rgb PASSED
# tests/test_exceptions_example.py::TestConvertToRgb::test_invalid_path_raises_file_not_found PASSED
# tests/test_exceptions_example.py::TestConvertToRgb::test_invalid_image_format_raises_unidentified_error PASSED
# ...

if __name__ == "__main__":
    # Run with: python tests/test_exceptions_example.py
    pytest.main([__file__, "-v", "--tb=short"])
