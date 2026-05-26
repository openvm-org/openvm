| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/fibonacci-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 1,880 |  4,000,051 |  515 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/keccak-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 13,561 |  14,365,133 |  2,216 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/sha2_bench-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 9,532 |  11,167,961 |  1,419 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/regex-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 1,560 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/ecrecover-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 602 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/pairing-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 730 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/kitchen_sink-d3079628a847d9ffedf8752ccaac8f2f38b23516.md) | 1,874 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d3079628a847d9ffedf8752ccaac8f2f38b23516

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26468811447)
