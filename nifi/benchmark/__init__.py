"""Benchmark datasets and protocol utilities for NiFi.

This package keeps imports lazy to avoid pulling optional heavy dependencies
when callers only need lightweight registry/download helpers.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "copy_local_tree": ("nifi.benchmark.download", "copy_local_tree"),
    "download_mipnerf360": ("nifi.benchmark.download", "download_mipnerf360"),
    "download_tandt_deepblending_bundle": ("nifi.benchmark.download", "download_tandt_deepblending_bundle"),
    "PAPER_NIFI_RESULTS": ("nifi.benchmark.evaluation_protocol", "PAPER_NIFI_RESULTS"),
    "aggregate_benchmark_records": ("nifi.benchmark.evaluation_protocol", "aggregate_benchmark_records"),
    "compare_with_paper": ("nifi.benchmark.evaluation_protocol", "compare_with_paper"),
    "BenchmarkPairPreparationConfig": ("nifi.benchmark.pair_preprocessing", "BenchmarkPairPreparationConfig"),
    "discover_scene_dirs": ("nifi.benchmark.pair_preprocessing", "discover_scene_dirs"),
    "prepare_benchmark_pairs": ("nifi.benchmark.pair_preprocessing", "prepare_benchmark_pairs"),
    "NiFiBenchmarkPairDataset": ("nifi.benchmark.paired_dataset", "NiFiBenchmarkPairDataset"),
    "BENCHMARK_DATASETS": ("nifi.benchmark.registry", "BENCHMARK_DATASETS"),
    "PAPER_EVAL_METRICS": ("nifi.benchmark.registry", "PAPER_EVAL_METRICS"),
    "PAPER_EXTREME_RATES_LAMBDA": ("nifi.benchmark.registry", "PAPER_EXTREME_RATES_LAMBDA"),
    "list_supported_datasets": ("nifi.benchmark.registry", "list_supported_datasets"),
    "rate_folder_name": ("nifi.benchmark.registry", "rate_folder_name"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

