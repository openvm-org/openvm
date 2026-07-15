| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/fibonacci-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 413 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/keccak-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 8,364 |  14,365,133 |  1,507 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/sha2_bench-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 3,992 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/regex-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 565 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/ecrecover-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 220 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/pairing-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 272 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/kitchen_sink-222bbd9bfa90869f1eade7a82d48726728ec4972.md) | 1,911 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/222bbd9bfa90869f1eade7a82d48726728ec4972

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29452360305)
