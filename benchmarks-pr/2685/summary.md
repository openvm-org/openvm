| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/fibonacci-afd52c2f9664f217a73e055416d6d43e3eb23933.md) | 3,914 |  12,000,265 |  969 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/keccak-afd52c2f9664f217a73e055416d6d43e3eb23933.md) | 18,575 |  18,655,329 |  3,331 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/regex-afd52c2f9664f217a73e055416d6d43e3eb23933.md) | 1,435 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/ecrecover-afd52c2f9664f217a73e055416d6d43e3eb23933.md) | 641 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/pairing-afd52c2f9664f217a73e055416d6d43e3eb23933.md) | 908 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/kitchen_sink-afd52c2f9664f217a73e055416d6d43e3eb23933.md) | 2,155 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/afd52c2f9664f217a73e055416d6d43e3eb23933

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24244684786)
