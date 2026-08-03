| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/fibonacci-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 470 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/keccak-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 7,282 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/sha2_bench-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 4,117 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/regex-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 666 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/ecrecover-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 234 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/pairing-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 239 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3096/kitchen_sink-bf71f7dcd3f672d0792652a03cdc8966d6cfef0a.md) | 2,060 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bf71f7dcd3f672d0792652a03cdc8966d6cfef0a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30852318383)
