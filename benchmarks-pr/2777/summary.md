| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 1,827 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 14,139 |  14,365,133 |  2,392 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 8,132 |  11,167,961 |  892 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 1,577 |  4,090,656 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 602 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 738 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-455f911c9b373c1c4f03f0a3802e3a3b3c439d6f.md) | 1,890 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/455f911c9b373c1c4f03f0a3802e3a3b3c439d6f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25934700822)
