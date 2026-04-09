| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/fibonacci-cd702c49cc898f14b5192fba42ae43bd6dc3e168.md) | 3,847 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/keccak-cd702c49cc898f14b5192fba42ae43bd6dc3e168.md) | 18,690 |  18,655,329 |  3,354 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/regex-cd702c49cc898f14b5192fba42ae43bd6dc3e168.md) | 1,439 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/ecrecover-cd702c49cc898f14b5192fba42ae43bd6dc3e168.md) | 643 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/pairing-cd702c49cc898f14b5192fba42ae43bd6dc3e168.md) | 908 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/kitchen_sink-cd702c49cc898f14b5192fba42ae43bd6dc3e168.md) | 2,181 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cd702c49cc898f14b5192fba42ae43bd6dc3e168

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24195289828)
