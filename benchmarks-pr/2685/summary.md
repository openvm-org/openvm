| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/fibonacci-e548958bace91e23fa498a65a579cb9b7df77270.md) | 3,890 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/keccak-e548958bace91e23fa498a65a579cb9b7df77270.md) | 18,604 |  18,655,329 |  3,334 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/regex-e548958bace91e23fa498a65a579cb9b7df77270.md) | 1,428 |  4,137,067 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/ecrecover-e548958bace91e23fa498a65a579cb9b7df77270.md) | 645 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/pairing-e548958bace91e23fa498a65a579cb9b7df77270.md) | 905 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/kitchen_sink-e548958bace91e23fa498a65a579cb9b7df77270.md) | 2,171 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e548958bace91e23fa498a65a579cb9b7df77270

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24209866849)
