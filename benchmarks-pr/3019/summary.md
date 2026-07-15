| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-12883ba039e942442f82a2b109af4e6125a25156.md) | 470 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-12883ba039e942442f82a2b109af4e6125a25156.md) | 8,737 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-12883ba039e942442f82a2b109af4e6125a25156.md) | 3,923 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-12883ba039e942442f82a2b109af4e6125a25156.md) | 506 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-12883ba039e942442f82a2b109af4e6125a25156.md) | 216 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-12883ba039e942442f82a2b109af4e6125a25156.md) | 279 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-12883ba039e942442f82a2b109af4e6125a25156.md) | 1,914 |  1,979,971 |  461 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-12883ba039e942442f82a2b109af4e6125a25156.md) | 489 |  4,000,051 |  223 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-12883ba039e942442f82a2b109af4e6125a25156.md) | 582 |  4,090,656 |  180 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-12883ba039e942442f82a2b109af4e6125a25156.md) | 220 |  112,210 |  175 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-12883ba039e942442f82a2b109af4e6125a25156.md) | 287 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-12883ba039e942442f82a2b109af4e6125a25156.md) | 2,284 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/12883ba039e942442f82a2b109af4e6125a25156

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29446983204)
