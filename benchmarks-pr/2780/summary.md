| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/fibonacci-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 1,887 |  4,000,051 |  536 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/keccak-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 13,475 |  14,365,133 |  2,218 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/sha2_bench-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 9,409 |  11,167,961 |  1,400 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/regex-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 1,609 |  4,090,656 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/ecrecover-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 649 |  112,210 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/pairing-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 749 |  592,827 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2780/kitchen_sink-a5eb4e79039e437cd733baad4479fbdb013767f0.md) | 2,039 |  1,979,971 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a5eb4e79039e437cd733baad4479fbdb013767f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25880067516)
