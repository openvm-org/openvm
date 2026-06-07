| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 1,556 |  4,000,051 |  433 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 13,774 |  14,365,133 |  2,376 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 8,818 |  11,167,961 |  1,400 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 1,457 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 473 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 596 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 3,730 |  1,979,971 |  927 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 810 |  4,000,051 |  197 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 863 |  4,090,656 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 324 |  112,210 |  134 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 397 |  592,827 |  128 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-dac2c10a6c4338056c88144c3c00104254aad91e.md) | 2,050 |  1,979,971 |  395 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dac2c10a6c4338056c88144c3c00104254aad91e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27092919798)
