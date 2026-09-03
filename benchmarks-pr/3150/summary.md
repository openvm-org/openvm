| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/fibonacci-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 486 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/keccak-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 7,758 |  14,365,133 |  1,653 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/sha2_bench-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 4,396 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/regex-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 758 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/ecrecover-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 208 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/pairing-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 251 |  592,827 |  174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/kitchen_sink-c66c3e8c976d16a0353e12c6c2be547dea36e1d5.md) | 2,263 |  1,979,971 |  480 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c66c3e8c976d16a0353e12c6c2be547dea36e1d5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33817421627)
