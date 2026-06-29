| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/fibonacci-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 1,036 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/keccak-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 16,043 |  14,365,133 |  3,090 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/sha2_bench-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 8,238 |  11,167,961 |  1,006 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/regex-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 1,185 |  4,090,656 |  365 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/ecrecover-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 428 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/pairing-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 590 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2936/kitchen_sink-62582280ac6822fd15d0fda3f96bcc9f31cd2cde.md) | 3,857 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/62582280ac6822fd15d0fda3f96bcc9f31cd2cde

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28400945674)
