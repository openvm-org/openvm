| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 441 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 7,184 |  14,365,133 |  1,602 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 4,073 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 709 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 206 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 236 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-078531f2d3520ffac19a8c9dd46652ddcaed5066.md) | 2,178 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/078531f2d3520ffac19a8c9dd46652ddcaed5066

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31201361889)
