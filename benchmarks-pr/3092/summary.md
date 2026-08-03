| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/fibonacci-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 1,578 |  12,000,265 |  361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/keccak-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 9,345 |  18,655,329 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/sha2_bench-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 4,950 |  14,793,960 |  578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/regex-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 660 |  4,137,067 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/ecrecover-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 431 |  123,583 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/pairing-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 556 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3092/kitchen_sink-e0f88506c350711ec40d26a6277e2594c2776dd8.md) | 2,200 |  2,579,903 |  481 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e0f88506c350711ec40d26a6277e2594c2776dd8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30832587582)
