| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 464 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 7,368 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 4,187 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 668 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 197 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 234 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-6f4a6bd389e546b3c14375d67904e043083d8472.md) | 2,034 |  1,979,971 |  529 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6f4a6bd389e546b3c14375d67904e043083d8472

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31664478788)
