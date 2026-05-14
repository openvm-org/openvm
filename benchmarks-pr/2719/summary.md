| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 1,900 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 13,737 |  14,365,133 |  2,256 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 9,372 |  11,167,961 |  1,404 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 1,602 |  4,090,656 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 644 |  112,210 |  295 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 754 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-5eaac821d715bc18aa96b2756d288cfcd44b2ddd.md) | 2,038 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5eaac821d715bc18aa96b2756d288cfcd44b2ddd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25885138404)
