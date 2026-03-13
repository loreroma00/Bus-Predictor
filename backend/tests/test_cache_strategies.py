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
from application.domain.ledgers import TopologyLedger, ScheduleLedger


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
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)

    def test_strategy_name(self):
        """Strategy name should be 'file'."""
        strategy = FileCacheStrategy()
        assert strategy.strategy_name == "file"

    def test_save_and_load_topology(self, temp_cache_dir):
        """Should save and load TopologyLedger correctly."""
        strategy = FileCacheStrategy(cache_dir=temp_cache_dir)

        topology = TopologyLedger(routes={}, stops={"s1": {"name": "Stop 1"}}, shapes={}, trips={})
        strategy.save_topology(topology)

        loaded = strategy.load_topology()
        assert loaded is not None
        assert loaded.stops["s1"]["name"] == "Stop 1"

    def test_load_nonexistent_file(self, temp_cache_dir):
        """Should return None for nonexistent file."""
        strategy = FileCacheStrategy(cache_dir=temp_cache_dir)

        result = strategy.load_topology()
        assert result is None

    def test_save_with_md5(self, temp_cache_dir):
        """Should store MD5 metadata when provided."""
        strategy = FileCacheStrategy(cache_dir=temp_cache_dir)

        topology = TopologyLedger()
        strategy.save_topology(topology, source_md5="abc123")

        loaded = strategy.load_topology()
        assert loaded.source_md5 == "abc123"

    def test_load_with_matching_md5(self, temp_cache_dir):
        """Should return data when MD5 matches."""
        strategy = FileCacheStrategy(cache_dir=temp_cache_dir)

        topology = TopologyLedger()
        strategy.save_topology(topology, source_md5="abc123")

        loaded = strategy.load_topology(expected_md5="abc123")
        assert loaded is not None

    def test_load_with_mismatched_md5(self, temp_cache_dir):
        """Should return None and delete cache when MD5 mismatches."""
        strategy = FileCacheStrategy(cache_dir=temp_cache_dir)

        topology = TopologyLedger()
        strategy.save_topology(topology, source_md5="abc123")

        loaded = strategy.load_topology(expected_md5="different_md5")
        assert loaded is None
        assert not os.path.exists(os.path.join(temp_cache_dir, "topology_cache.pkl"))


class TestNoCacheStrategy:
    """Test NoCacheStrategy (no-op implementation)."""

    def test_strategy_name(self):
        """Strategy name should be 'none'."""
        strategy = NoCacheStrategy()
        assert strategy.strategy_name == "none"

    def test_load_always_returns_none(self):
        """Load should always return None."""
        strategy = NoCacheStrategy()
        assert strategy.load_topology() is None
        assert strategy.load_topology(expected_md5="anything") is None
        assert strategy.load_schedule() is None

    def test_save_does_nothing(self):
        """Save should not raise and do nothing."""
        strategy = NoCacheStrategy()
        topology = TopologyLedger()
        # Should not raise
        strategy.save_topology(topology)
        strategy.save_topology(topology, source_md5="abc123")


class TestCacheStrategyProtocol:
    """Test that strategies satisfy the CacheStrategy protocol."""

    def test_file_strategy_has_required_methods(self):
        """FileCacheStrategy should have load_topology and save_topology."""
        strategy = FileCacheStrategy()
        assert hasattr(strategy, "load_topology")
        assert hasattr(strategy, "save_topology")
        assert hasattr(strategy, "load_schedule")
        assert hasattr(strategy, "save_schedule")
        assert callable(strategy.load_topology)
        assert callable(strategy.save_topology)

    def test_none_strategy_has_required_methods(self):
        """NoCacheStrategy should have load_topology and save_topology."""
        strategy = NoCacheStrategy()
        assert hasattr(strategy, "load_topology")
        assert hasattr(strategy, "save_topology")
        assert hasattr(strategy, "load_schedule")
        assert hasattr(strategy, "save_schedule")
        assert callable(strategy.load_topology)
        assert callable(strategy.save_topology)
