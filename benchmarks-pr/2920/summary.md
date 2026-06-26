| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/fibonacci-586f6e427dc13030bf0504f5a491313db8345f70.md) | 1,024 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/keccak-586f6e427dc13030bf0504f5a491313db8345f70.md) | 15,716 |  14,365,133 |  3,019 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/sha2_bench-586f6e427dc13030bf0504f5a491313db8345f70.md) | 8,252 |  11,167,961 |  1,014 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/regex-586f6e427dc13030bf0504f5a491313db8345f70.md) | 1,167 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/ecrecover-586f6e427dc13030bf0504f5a491313db8345f70.md) | 429 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/pairing-586f6e427dc13030bf0504f5a491313db8345f70.md) | 602 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/kitchen_sink-586f6e427dc13030bf0504f5a491313db8345f70.md) | 3,877 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/586f6e427dc13030bf0504f5a491313db8345f70

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28250551898)
