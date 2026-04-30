| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 1,894 |  4,000,051 |  536 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 13,608 |  14,365,133 |  2,220 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 9,406 |  11,167,961 |  1,266 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 1,574 |  4,090,656 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 637 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 744 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c.md) | 2,069 |  1,979,971 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e8e2e8f96bbb28cc2ad70d1efab1cd1a82e0d95c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25192112797)
