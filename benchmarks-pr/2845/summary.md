| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/fibonacci-bc9552e36d25717f2bb655281894094b83231e07.md) | 3,731 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/keccak-bc9552e36d25717f2bb655281894094b83231e07.md) | 18,210 |  18,655,329 |  3,310 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/sha2_bench-bc9552e36d25717f2bb655281894094b83231e07.md) | 10,063 |  14,793,960 |  1,479 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/regex-bc9552e36d25717f2bb655281894094b83231e07.md) | 1,381 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/ecrecover-bc9552e36d25717f2bb655281894094b83231e07.md) | 593 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/pairing-bc9552e36d25717f2bb655281894094b83231e07.md) | 901 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/kitchen_sink-bc9552e36d25717f2bb655281894094b83231e07.md) | 3,883 |  2,579,903 |  964 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc9552e36d25717f2bb655281894094b83231e07

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27052982379)
