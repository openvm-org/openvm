| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/fibonacci-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 3,846 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/keccak-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 18,659 |  18,655,329 |  3,336 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/sha2_bench-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 8,963 |  14,793,960 |  1,390 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/regex-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 1,431 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/ecrecover-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 649 |  123,583 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/pairing-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 905 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/kitchen_sink-aa2b8118e05e778789f3328c06010f641a05aca7.md) | 2,082 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aa2b8118e05e778789f3328c06010f641a05aca7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24473439313)
