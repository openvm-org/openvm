| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/fibonacci-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 3,820 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/keccak-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 18,575 |  18,655,329 |  3,314 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/sha2_bench-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 8,913 |  14,793,960 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/regex-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 1,401 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/ecrecover-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 637 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/pairing-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 897 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/kitchen_sink-7c6124158cbb6711cdafdbd046d5839f3faf7d95.md) | 2,089 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7c6124158cbb6711cdafdbd046d5839f3faf7d95

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25238600824)
