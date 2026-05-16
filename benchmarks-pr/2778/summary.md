| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-33351177259f527fd5d6266774a23367ac2fa27b.md) | 1,407 |  4,000,051 |  433 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-33351177259f527fd5d6266774a23367ac2fa27b.md) | 13,206 |  14,365,133 |  2,175 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-33351177259f527fd5d6266774a23367ac2fa27b.md) | 8,918 |  11,167,961 |  1,392 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-33351177259f527fd5d6266774a23367ac2fa27b.md) | 1,349 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-33351177259f527fd5d6266774a23367ac2fa27b.md) | 469 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-33351177259f527fd5d6266774a23367ac2fa27b.md) | 591 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-33351177259f527fd5d6266774a23367ac2fa27b.md) | 1,797 |  1,979,971 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/33351177259f527fd5d6266774a23367ac2fa27b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25963003063)
