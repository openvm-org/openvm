| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 5,608 |  4,000,051 |  553 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 20,396 |  14,365,133 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 14,072 |  11,167,961 |  1,924 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 3,806 |  4,090,656 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 1,962 |  112,210 |  312 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 2,086 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-e5f8e9c036de5d533631242be98751119cd0f8c1.md) | 5,589 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e5f8e9c036de5d533631242be98751119cd0f8c1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27788242554)
