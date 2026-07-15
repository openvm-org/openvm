| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 468 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 8,896 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 3,973 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 508 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 220 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 266 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 1,919 |  1,979,971 |  462 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 478 |  4,000,051 |  221 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 585 |  4,090,656 |  181 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 214 |  112,210 |  172 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 288 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 2,293 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9702d702ff6897c40b485af4bc2077f739b5b949

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29399633171)
