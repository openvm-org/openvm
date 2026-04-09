| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-f4e1f8d5902f2198d7062984e3678232c3aec014.md) | 3,770 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-f4e1f8d5902f2198d7062984e3678232c3aec014.md) | 18,635 |  18,655,329 |  3,354 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-f4e1f8d5902f2198d7062984e3678232c3aec014.md) | 1,405 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-f4e1f8d5902f2198d7062984e3678232c3aec014.md) | 650 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-f4e1f8d5902f2198d7062984e3678232c3aec014.md) | 914 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-f4e1f8d5902f2198d7062984e3678232c3aec014.md) | 2,152 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f4e1f8d5902f2198d7062984e3678232c3aec014

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24212816632)
