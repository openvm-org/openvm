| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 485 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 8,779 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 3,897 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 503 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 218 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 272 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 1,928 |  1,979,971 |  465 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 479 |  4,000,051 |  216 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 587 |  4,090,656 |  181 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 216 |  112,210 |  174 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 291 |  592,827 |  176 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-c35008eba3c87dd37271c5227e2d81b299b054f7.md) | 2,289 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c35008eba3c87dd37271c5227e2d81b299b054f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29402827924)
