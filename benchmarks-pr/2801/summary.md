| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/fibonacci-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 1,555 |  4,000,051 |  440 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/keccak-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 13,867 |  14,365,133 |  2,357 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/sha2_bench-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 9,086 |  11,167,961 |  1,404 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/regex-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 1,568 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/ecrecover-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 479 |  112,210 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/pairing-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 611 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/kitchen_sink-aa9c756ec4edb30b63349a545dca0fe70175f8ba.md) | 2,004 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aa9c756ec4edb30b63349a545dca0fe70175f8ba

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26901057667)
