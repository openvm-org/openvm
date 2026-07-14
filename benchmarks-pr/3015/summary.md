| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 472 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/keccak-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 8,717 |  14,365,133 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/sha2_bench-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 4,081 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 569 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 218 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 284 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 1,949 |  1,979,971 |  458 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci_e2e-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 478 |  4,000,051 |  219 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex_e2e-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 666 |  4,090,656 |  202 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover_e2e-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 217 |  112,210 |  171 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing_e2e-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 301 |  592,827 |  172 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink_e2e-c5ce09321d1a9916885f84ea3c77965190611c1c.md) | 2,303 |  1,979,971 |  451 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c5ce09321d1a9916885f84ea3c77965190611c1c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29356600048)
