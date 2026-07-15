| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 464 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 8,931 |  14,365,133 |  1,560 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 3,906 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 508 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 216 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 273 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 1,910 |  1,979,971 |  462 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 477 |  4,000,051 |  220 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 587 |  4,090,656 |  188 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 215 |  112,210 |  175 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 276 |  592,827 |  173 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-2835a08c855c728341dd2180b166c2afe5bb98c1.md) | 2,290 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2835a08c855c728341dd2180b166c2afe5bb98c1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29440929774)
