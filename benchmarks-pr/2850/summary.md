| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 5,244 |  4,000,051 |  430 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 18,573 |  14,365,133 |  2,364 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 12,531 |  11,167,961 |  1,420 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 3,668 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 1,945 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 2,090 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3.md) | 6,063 |  1,979,971 |  953 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/acb1ddf9dcd0c3a0baa17ffa57dbf73b46f8ecd3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27226295727)
