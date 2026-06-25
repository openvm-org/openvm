| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 1,023 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 16,274 |  14,365,133 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 8,207 |  11,167,961 |  1,002 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 1,174 |  4,090,656 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 433 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 590 |  592,827 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-34b8e6fef5098786fc8d20e7adbd753a6071c7bb.md) | 3,906 |  1,979,971 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/34b8e6fef5098786fc8d20e7adbd753a6071c7bb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28176727079)
