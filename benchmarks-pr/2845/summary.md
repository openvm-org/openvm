| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/fibonacci-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 3,732 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/keccak-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 18,487 |  18,655,329 |  3,252 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/sha2_bench-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 10,277 |  14,793,960 |  1,465 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/regex-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 1,392 |  4,137,067 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/ecrecover-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 611 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/pairing-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 893 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/kitchen_sink-ec750de6137b26a1ce3bde28f997fc1e0a183765.md) | 1,902 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ec750de6137b26a1ce3bde28f997fc1e0a183765

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26987521680)
