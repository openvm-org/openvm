| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/fibonacci-516acf2b605bae3d10df65329981fd27dcceb188.md) | 1,871 |  4,000,051 |  530 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/keccak-516acf2b605bae3d10df65329981fd27dcceb188.md) | 13,405 |  14,365,133 |  2,201 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/sha2_bench-516acf2b605bae3d10df65329981fd27dcceb188.md) | 9,285 |  11,167,961 |  1,244 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/regex-516acf2b605bae3d10df65329981fd27dcceb188.md) | 1,578 |  4,090,656 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/ecrecover-516acf2b605bae3d10df65329981fd27dcceb188.md) | 645 |  112,210 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/pairing-516acf2b605bae3d10df65329981fd27dcceb188.md) | 764 |  592,827 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2762/kitchen_sink-516acf2b605bae3d10df65329981fd27dcceb188.md) | 2,070 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/516acf2b605bae3d10df65329981fd27dcceb188

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25065706258)
