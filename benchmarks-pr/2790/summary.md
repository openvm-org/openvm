| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-7a4158c785ef6def35d83b498fd7c614df711119.md) | 3,745 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-7a4158c785ef6def35d83b498fd7c614df711119.md) | 18,126 |  18,655,329 |  3,296 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-7a4158c785ef6def35d83b498fd7c614df711119.md) | 10,071 |  14,793,960 |  1,469 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-7a4158c785ef6def35d83b498fd7c614df711119.md) | 1,375 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-7a4158c785ef6def35d83b498fd7c614df711119.md) | 598 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-7a4158c785ef6def35d83b498fd7c614df711119.md) | 881 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-7a4158c785ef6def35d83b498fd7c614df711119.md) | 3,877 |  2,579,903 |  953 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-7a4158c785ef6def35d83b498fd7c614df711119.md) | 1,619 |  12,000,265 |  409 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-7a4158c785ef6def35d83b498fd7c614df711119.md) | 668 |  4,137,067 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-7a4158c785ef6def35d83b498fd7c614df711119.md) | 361 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-7a4158c785ef6def35d83b498fd7c614df711119.md) | 479 |  1,745,757 |  133 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-7a4158c785ef6def35d83b498fd7c614df711119.md) | 1,823 |  2,579,903 |  401 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a4158c785ef6def35d83b498fd7c614df711119

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27091285424)
