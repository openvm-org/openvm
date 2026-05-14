| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 1,907 |  4,000,051 |  538 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 13,563 |  14,365,133 |  2,244 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 9,485 |  11,167,961 |  1,420 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 1,589 |  4,090,656 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 645 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 754 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-e6af7e918a2edbe80d2fe20f866a27bbcddec470.md) | 2,029 |  1,979,971 |  427 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e6af7e918a2edbe80d2fe20f866a27bbcddec470

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25854449786)
