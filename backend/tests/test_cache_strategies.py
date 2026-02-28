"""
Tests for Cache Strategies - Strategy pattern for ledger caching.
"""

import pytest
import os
import tempfile
from persistence import (
    FileCacheStrategy,
    NoCacheStrategy,
    get_cache_strategy,
    get_available_strategies,
)


class TestBaseCacheStrategy:
    """Test the ABC and discovery mechanism."""

    def test_get_available_strategies_returns_list(self):
        """Should return a list of strategy names."""
        strategies = get_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) >= 2  # At least file and none

    def test_file_strategy_in_available(self):
        """File strategy should be discoverable."""
        assert "file" in get_available_strategies()

    def test_none_strategy_in_available(self):
        """None strategy should be discoverable."""
        assert "none" in get_available_strategies()

    def test_get_strategy_by_name(self):
        """Should return correct strategy instance by name."""
        strategy = get_cache_strategy("file")
        assert isinstance(strategy, FileCacheStrategy)

    def test_get_none_strategy_by_name(self):
        """Should return NoCacheStrategy for 'none'."""
        strategy = get_cache_strategy("none")
        assert isinstance(strategy, NoCacheStrategy)

    def test_unknown_strategy_defaults_to_file(self):
        """Unknown strategy name should default to file."""
        strategy = get_cache_strategy("unknown_xyz")
        assert isinstance(strategy, FileCacheStrategy)


class TestFileCacheStrategy:
    """Test FileCacheStrategy load/save operations."""

    @pytest.fixture
    def temp_cache_file(self):
        """Create a temporary cache file path."""
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_strategy_name(self):
        """Strategy name should be 'file'."""
        strategy = FileCacheStrategy()
        assert strategy.strategy_name == "file"

    def test_save_and_load(self, temp_cache_file):
        """Should save and load data correctly."""
        strategy = FileCacheStrategy(cache_path=temp_cache_file)

        test_data = {"key": "value", "number": 42}
        strategy.save(test_data)

        loaded = strategy.load()
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_load_nonexistent_file(self, temp_cache_file):
        """Should return None for nonexistent file."""
        os.remove(temp_cache_file)  # Ensure file doesn't exist
        strategy = FileCacheStrategy(cache_path=temp_cache_file)

        result = strategy.load()
        assert result is None

    def test_save_with_md5(self, temp_cache_file):
        """Should store MD5 metadata when provided."""
        strategy = FileCacheStrategy(cache_path=temp_cache_file)

        test_data = {"key": "value"}
        strategy.save(test_data, source_md5="abc123")

        loaded = strategy.load()
        assert loaded["__metadata__"]["md5"] == "abc123"

    def test_load_with_matching_md5(self, temp_cache_file):
        """Should return data when MD5 matches."""
        strategy = FileCacheStrategy(cache_path=temp_cache_file)

        test_data = {"key": "value"}
        strategy.save(test_data, source_md5="abc123")

        loaded = strategy.load(expected_md5="abc123")
        assert loaded is not None
        assert loaded["key"] == "value"

    def test_load_with_mismatched_md5(self, temp_cache_file):
        """Should return None and delete cache when MD5 mismatches."""
        strategy = FileCacheStrategy(cache_path=temp_cache_file)

        test_data = {"key": "value"}
        strategy.save(test_data, source_md5="abc123")

        loaded = strategy.load(expected_md5="different_md5")
        assert loaded is None
        assert not os.path.exists(temp_cache_file)


class TestNoCacheStrategy:
    """Test NoCacheStrategy (no-op implementation)."""

    def test_strategy_name(self):
        """Strategy name should be 'none'."""
        strategy = NoCacheStrategy()
        assert strategy.strategy_name == "none"

    def test_load_always_returns_none(self):
        """Load should always return None."""
        strategy = NoCacheStrategy()
        assert strategy.load() is None
        assert strategy.load(expected_md5="anything") is None

    def test_save_does_nothing(self):
        """Save should not raise and do nothing."""
        strategy = NoCacheStrategy()
        # Should not raise
        strategy.save({"data": "test"})
        strategy.save({"data": "test"}, source_md5="abc123")


class TestCacheStrategyProtocol:
    """Test that strategies satisfy the CacheStrategy protocol."""

    def test_file_strategy_has_required_methods(self):
        """FileCacheStrategy should have load and save."""
        strategy = FileCacheStrategy()
        assert hasattr(strategy, "load")
        assert hasattr(strategy, "save")
        assert callable(strategy.load)
        assert callable(strategy.save)

    def test_none_strategy_has_required_methods(self):
        """NoCacheStrategy should have load and save."""
        strategy = NoCacheStrategy()
        assert hasattr(strategy, "load")
        assert hasattr(strategy, "save")
        assert callable(strategy.load)
        assert callable(strategy.save)
