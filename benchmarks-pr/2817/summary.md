| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/fibonacci-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 3,854 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/keccak-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 18,777 |  18,655,329 |  3,314 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/sha2_bench-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 10,192 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/regex-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 1,401 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/ecrecover-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 605 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/pairing-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 894 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/kitchen_sink-10c7c9d76ef765c095bcea380a1f5dfe4bb640e5.md) | 1,907 |  2,579,903 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/10c7c9d76ef765c095bcea380a1f5dfe4bb640e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26517723567)
