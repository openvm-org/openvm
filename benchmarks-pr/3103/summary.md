| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-880525213760c482ca4cd2c34bfc25f16df23009.md) | 473 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-880525213760c482ca4cd2c34bfc25f16df23009.md) | 7,300 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-880525213760c482ca4cd2c34bfc25f16df23009.md) | 4,161 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-880525213760c482ca4cd2c34bfc25f16df23009.md) | 684 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-880525213760c482ca4cd2c34bfc25f16df23009.md) | 229 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-880525213760c482ca4cd2c34bfc25f16df23009.md) | 238 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-880525213760c482ca4cd2c34bfc25f16df23009.md) | 2,026 |  1,979,971 |  524 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/880525213760c482ca4cd2c34bfc25f16df23009

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31049517650)
