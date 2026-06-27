| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 1,026 |  4,000,051 |  391 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 15,681 |  14,365,133 |  3,013 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 8,267 |  11,167,961 |  1,022 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 1,168 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 439 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 590 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-e57f30293759e274b31cc1830a3c725fe023a57b.md) | 3,858 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e57f30293759e274b31cc1830a3c725fe023a57b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28274036395)
