| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/fibonacci-3f794b664f644e9b188bb74133afabc831db392a.md) | 3,936 |  12,000,265 |  1,137 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/keccak-3f794b664f644e9b188bb74133afabc831db392a.md) | 21,920 |  18,655,329 |  4,623 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/sha2_bench-3f794b664f644e9b188bb74133afabc831db392a.md) | 9,689 |  14,793,960 |  1,860 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/regex-3f794b664f644e9b188bb74133afabc831db392a.md) | 1,506 |  4,137,067 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/ecrecover-3f794b664f644e9b188bb74133afabc831db392a.md) | 605 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/pairing-3f794b664f644e9b188bb74133afabc831db392a.md) | 944 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2870/kitchen_sink-3f794b664f644e9b188bb74133afabc831db392a.md) | 4,101 |  2,579,903 |  875 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3f794b664f644e9b188bb74133afabc831db392a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27312918364)
