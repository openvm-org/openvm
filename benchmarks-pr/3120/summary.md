| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 466 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 7,610 |  14,365,133 |  1,608 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 4,319 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 741 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 211 |  112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 247 |  592,827 |  172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-608824a311cb9b39bf1416ef2dee931c639da9b7.md) | 2,225 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/608824a311cb9b39bf1416ef2dee931c639da9b7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33189651842)
