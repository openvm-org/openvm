| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-316c9145b822e3f4542e6650729871e9a0236447.md) | 1,399 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-316c9145b822e3f4542e6650729871e9a0236447.md) | 16,321 |  14,365,133 |  3,040 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-316c9145b822e3f4542e6650729871e9a0236447.md) | 10,021 |  11,167,961 |  1,007 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-316c9145b822e3f4542e6650729871e9a0236447.md) | 1,574 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-316c9145b822e3f4542e6650729871e9a0236447.md) | 440 |  112,210 |  307 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-316c9145b822e3f4542e6650729871e9a0236447.md) | 596 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-316c9145b822e3f4542e6650729871e9a0236447.md) | 3,895 |  1,979,971 |  859 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-316c9145b822e3f4542e6650729871e9a0236447.md) | 728 |  4,000,051 |  179 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-316c9145b822e3f4542e6650729871e9a0236447.md) | 990 |  4,090,656 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-316c9145b822e3f4542e6650729871e9a0236447.md) | 318 |  112,210 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-316c9145b822e3f4542e6650729871e9a0236447.md) | 412 |  592,827 |  143 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-316c9145b822e3f4542e6650729871e9a0236447.md) | 1,946 |  1,979,971 |  375 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/316c9145b822e3f4542e6650729871e9a0236447

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27816546392)
