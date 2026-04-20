| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/fibonacci-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 3,815 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/keccak-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 18,314 |  18,655,329 |  3,263 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/sha2_bench-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 8,911 |  14,793,960 |  1,389 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/regex-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 1,423 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/ecrecover-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 638 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/pairing-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 914 |  1,745,757 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/kitchen_sink-16376e945f68a2d2e2d8e6f3c576464ac92eb0ff.md) | 2,088 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/16376e945f68a2d2e2d8e6f3c576464ac92eb0ff

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24682792120)
