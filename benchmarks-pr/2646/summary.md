| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-eb2cbdf67728beabf73c2acf074e533683e1b1c0.md) | 3,839 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-eb2cbdf67728beabf73c2acf074e533683e1b1c0.md) | 18,669 |  18,655,329 |  3,316 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-eb2cbdf67728beabf73c2acf074e533683e1b1c0.md) | 1,439 |  4,137,067 |  383 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-eb2cbdf67728beabf73c2acf074e533683e1b1c0.md) | 734 |  317,792 |  343 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-eb2cbdf67728beabf73c2acf074e533683e1b1c0.md) | 924 |  1,745,757 |  314 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-eb2cbdf67728beabf73c2acf074e533683e1b1c0.md) | 2,509 |  2,580,026 |  543 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eb2cbdf67728beabf73c2acf074e533683e1b1c0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23900682094)
