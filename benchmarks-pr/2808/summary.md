| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 1,567 |  4,000,051 |  438 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 14,065 |  14,365,133 |  2,411 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 9,414 |  11,167,961 |  1,440 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 1,477 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 474 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 598 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-eef7099822b8720ac4d6cfdae6dfd683a472b54d.md) | 1,818 |  1,979,971 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eef7099822b8720ac4d6cfdae6dfd683a472b54d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26312537637)
