| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/fibonacci-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 3,747 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/keccak-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 18,546 |  18,655,329 |  3,265 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/sha2_bench-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 10,133 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/regex-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 1,394 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/ecrecover-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 597 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/pairing-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 875 |  1,745,757 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/kitchen_sink-2a8e6c89e136dded0e61d1289ad418b232feccbe.md) | 1,898 |  2,579,903 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2a8e6c89e136dded0e61d1289ad418b232feccbe

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26239357650)
