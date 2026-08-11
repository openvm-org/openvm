| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 470 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 7,339 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 4,111 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 673 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 224 |  112,210 |  200 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 233 |  592,827 |  200 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-59880b9bdcce34652ddf4b95416c112569c7ea1f.md) | 2,026 |  1,979,971 |  521 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/59880b9bdcce34652ddf4b95416c112569c7ea1f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31501315709)
