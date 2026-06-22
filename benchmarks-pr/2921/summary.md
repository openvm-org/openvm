| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-ee7ff2320539430237771041ade044976b56c1e2.md) | 1,035 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-ee7ff2320539430237771041ade044976b56c1e2.md) | 16,325 |  14,365,133 |  3,026 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-ee7ff2320539430237771041ade044976b56c1e2.md) | 8,286 |  11,167,961 |  1,014 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-ee7ff2320539430237771041ade044976b56c1e2.md) | 1,210 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-ee7ff2320539430237771041ade044976b56c1e2.md) | 436 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-ee7ff2320539430237771041ade044976b56c1e2.md) | 597 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-ee7ff2320539430237771041ade044976b56c1e2.md) | 3,879 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ee7ff2320539430237771041ade044976b56c1e2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27974066028)
