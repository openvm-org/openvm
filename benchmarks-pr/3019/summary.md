| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 473 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 8,825 |  14,365,133 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 3,951 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 500 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 263 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-9702d702ff6897c40b485af4bc2077f739b5b949.md) | 1,923 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9702d702ff6897c40b485af4bc2077f739b5b949

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29374283717)
