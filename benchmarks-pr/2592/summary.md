| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.md) | 3,826 |  12,000,265 |  934 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.md) | 18,723 |  18,655,329 |  3,292 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.md) | 1,407 |  4,137,067 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.md) | 640 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.md) | 907 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-2ea2f43a828fac2702e871f5c8249e1ef11ae7b9.md) | 2,264 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ea2f43a828fac2702e871f5c8249e1ef11ae7b9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23567637984)
