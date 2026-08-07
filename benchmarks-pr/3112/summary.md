| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 456 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 7,387 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 4,152 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 657 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 196 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 228 |  592,827 |  180 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-88b365638da75251c5b2397ab4b2d2ad0094b614.md) | 2,023 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/88b365638da75251c5b2397ab4b2d2ad0094b614

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31220685291)
