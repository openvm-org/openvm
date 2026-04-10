| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/fibonacci-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 3,871 |  12,000,265 |  962 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/keccak-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 15,736 |  1,235,218 |  2,215 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/regex-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 1,418 |  4,136,694 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/ecrecover-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 633 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/pairing-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 916 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/kitchen_sink-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 2,364 |  154,763 |  412 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/fibonacci_e2e-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 1,722 |  12,000,265 |  457 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/regex_e2e-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 879 |  4,136,694 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/ecrecover_e2e-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 403 |  122,348 |  149 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/pairing_e2e-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 541 |  1,745,757 |  152 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2697/kitchen_sink_e2e-be72a3942dc3b5d9c7aa168150003790456b559d.md) | 2,375 |  154,763 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/be72a3942dc3b5d9c7aa168150003790456b559d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24265267425)
