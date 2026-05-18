| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 3,761 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 18,755 |  18,655,329 |  3,299 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 10,214 |  14,793,960 |  1,462 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 1,398 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 603 |  123,583 |  244 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 899 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-e556dc6df3968ff163aa5da97fa58b9665ca4e7f.md) | 1,915 |  2,579,903 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e556dc6df3968ff163aa5da97fa58b9665ca4e7f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26053987245)
