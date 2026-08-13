| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-07cd0afdca95106297204a679f7af681e6440b2b.md) | 464 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-07cd0afdca95106297204a679f7af681e6440b2b.md) | 7,361 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-07cd0afdca95106297204a679f7af681e6440b2b.md) | 4,197 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-07cd0afdca95106297204a679f7af681e6440b2b.md) | 666 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-07cd0afdca95106297204a679f7af681e6440b2b.md) | 196 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-07cd0afdca95106297204a679f7af681e6440b2b.md) | 238 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-07cd0afdca95106297204a679f7af681e6440b2b.md) | 2,032 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/07cd0afdca95106297204a679f7af681e6440b2b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31666511932)
