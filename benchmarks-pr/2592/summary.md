| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3.md) | 3,852 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3.md) | 18,744 |  18,655,329 |  3,344 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3.md) | 1,439 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3.md) | 661 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3.md) | 898 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3.md) | 2,278 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b7e2663cfe3d4abd7ab49f599fdfd1ec9a08b0b3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23812020564)
