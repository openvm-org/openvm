| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 552 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 7,526 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 4,440 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 594 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 223 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 254 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-e6691e4e708c6c4dc282abae3f4fe4993a2fbe57.md) | 2,735 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e6691e4e708c6c4dc282abae3f4fe4993a2fbe57

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29392568651)
