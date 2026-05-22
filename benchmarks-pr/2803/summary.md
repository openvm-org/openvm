| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 3,748 |  12,000,265 |  918 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 18,590 |  18,655,329 |  3,275 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 10,158 |  14,793,960 |  1,451 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 1,390 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 600 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 894 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-69da03b49f8a96a080f33189016b583c4dfe1923.md) | 1,897 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/69da03b49f8a96a080f33189016b583c4dfe1923

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26311494533)
