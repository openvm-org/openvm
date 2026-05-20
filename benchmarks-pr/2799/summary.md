| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/fibonacci-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 3,770 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/keccak-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 18,992 |  18,655,329 |  3,344 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/sha2_bench-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 10,324 |  14,793,960 |  1,473 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/regex-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 1,414 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/ecrecover-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 601 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/pairing-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 887 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/kitchen_sink-6a4de1ad2b07837e7c4557a2406c74e7b980b97d.md) | 1,907 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6a4de1ad2b07837e7c4557a2406c74e7b980b97d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26189933971)
