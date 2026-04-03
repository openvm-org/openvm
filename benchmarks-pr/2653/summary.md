| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/fibonacci-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 3,843 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/keccak-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 15,680 |  1,235,218 |  2,200 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/regex-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 1,411 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/ecrecover-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 634 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/pairing-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 922 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/kitchen_sink-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 2,375 |  154,763 |  416 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/fibonacci_e2e-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 1,716 |  12,000,265 |  455 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/regex_e2e-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 881 |  4,136,694 |  188 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/ecrecover_e2e-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 397 |  122,348 |  147 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/pairing_e2e-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 544 |  1,745,757 |  151 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/kitchen_sink_e2e-78166c6c31b7f8da316bfaae010b5f22ae9cdf94.md) | 2,378 |  154,763 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/78166c6c31b7f8da316bfaae010b5f22ae9cdf94

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23954757292)
