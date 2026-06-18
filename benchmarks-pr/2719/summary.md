| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-ead16304615e6dbd386c4323496a924c7c145061.md) | 1,511 |  4,000,051 |  533 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-ead16304615e6dbd386c4323496a924c7c145061.md) | 16,259 |  14,365,133 |  3,041 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-ead16304615e6dbd386c4323496a924c7c145061.md) | 10,384 |  11,167,961 |  1,926 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-ead16304615e6dbd386c4323496a924c7c145061.md) | 1,501 |  4,090,656 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-ead16304615e6dbd386c4323496a924c7c145061.md) | 438 |  112,210 |  309 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-ead16304615e6dbd386c4323496a924c7c145061.md) | 600 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-ead16304615e6dbd386c4323496a924c7c145061.md) | 3,888 |  1,979,971 |  852 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-ead16304615e6dbd386c4323496a924c7c145061.md) | 834 |  4,000,051 |  234 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-ead16304615e6dbd386c4323496a924c7c145061.md) | 833 |  4,090,656 |  197 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-ead16304615e6dbd386c4323496a924c7c145061.md) | 320 |  112,210 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-ead16304615e6dbd386c4323496a924c7c145061.md) | 406 |  592,827 |  145 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-ead16304615e6dbd386c4323496a924c7c145061.md) | 1,951 |  1,979,971 |  374 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ead16304615e6dbd386c4323496a924c7c145061

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27789312144)
