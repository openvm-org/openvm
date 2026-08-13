| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 462 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 7,303 |  14,365,133 |  1,503 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 4,187 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 662 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 197 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 238 |  592,827 |  201 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8.md) | 2,037 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a98ed3afa7c50aeec280f25c3aebe7903bbfd3e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31670572384)
