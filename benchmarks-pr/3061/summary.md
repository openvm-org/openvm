| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 469 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 7,325 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 4,686 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 661 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 231 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 321 |  592,827 |  192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-effc449caefb629f964fc0525479f7832dbfbbd1.md) | 2,683 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/effc449caefb629f964fc0525479f7832dbfbbd1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30036425987)
