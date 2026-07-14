| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 464 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/keccak-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 8,673 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/sha2_bench-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 4,097 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 558 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 216 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 286 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 1,946 |  1,979,971 |  461 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci_e2e-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 493 |  4,000,051 |  219 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex_e2e-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 674 |  4,090,656 |  207 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover_e2e-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 214 |  112,210 |  172 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing_e2e-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 317 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink_e2e-2c29e5f80726e1d35757484cb86d444b6d7ed076.md) | 2,310 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c29e5f80726e1d35757484cb86d444b6d7ed076

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29342225620)
