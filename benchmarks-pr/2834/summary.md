| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/fibonacci-6950753b5334fa13928a17bfa411a007819a9d44.md) | 3,767 |  12,000,265 |  927 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/keccak-6950753b5334fa13928a17bfa411a007819a9d44.md) | 18,530 |  18,655,329 |  3,258 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/sha2_bench-6950753b5334fa13928a17bfa411a007819a9d44.md) | 10,398 |  14,793,960 |  1,484 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/regex-6950753b5334fa13928a17bfa411a007819a9d44.md) | 1,400 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/ecrecover-6950753b5334fa13928a17bfa411a007819a9d44.md) | 597 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/pairing-6950753b5334fa13928a17bfa411a007819a9d44.md) | 883 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/kitchen_sink-6950753b5334fa13928a17bfa411a007819a9d44.md) | 1,901 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6950753b5334fa13928a17bfa411a007819a9d44

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26800105425)
