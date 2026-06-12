| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/fibonacci-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 3,965 |  12,000,265 |  1,146 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/keccak-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 21,384 |  18,655,329 |  4,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/sha2_bench-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 9,722 |  14,793,960 |  1,863 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/regex-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 1,507 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/ecrecover-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 602 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/pairing-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 949 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2885/kitchen_sink-4aa773156825930ad595d2c5d41e0a7f118f6cab.md) | 4,150 |  2,579,903 |  893 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4aa773156825930ad595d2c5d41e0a7f118f6cab

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27425932776)
