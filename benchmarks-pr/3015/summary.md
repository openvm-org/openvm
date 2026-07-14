| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 461 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/keccak-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 8,683 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/sha2_bench-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 4,061 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 558 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 223 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 278 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 1,942 |  1,979,971 |  457 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci_e2e-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 476 |  4,000,051 |  216 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex_e2e-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 679 |  4,090,656 |  207 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover_e2e-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 212 |  112,210 |  170 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing_e2e-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 300 |  592,827 |  176 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink_e2e-aa63ef79e1582d73d0b5232c70f16b5e5ed1329c.md) | 2,294 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aa63ef79e1582d73d0b5232c70f16b5e5ed1329c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29357459392)
