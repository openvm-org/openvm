| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/fibonacci-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 3,846 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/keccak-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 19,207 |  18,655,329 |  3,412 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/sha2_bench-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 9,004 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/regex-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 1,408 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/ecrecover-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 641 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/pairing-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 911 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/kitchen_sink-3af323fedc9d72677b615d51ec98ba5dff007b2b.md) | 2,095 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3af323fedc9d72677b615d51ec98ba5dff007b2b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25234053332)
