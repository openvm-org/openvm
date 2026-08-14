| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 457 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 7,564 |  14,365,133 |  1,565 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 4,193 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 654 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 202 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 238 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-8ad03c56def9baccc71c34496bbc6078a86e5bcf.md) | 2,024 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8ad03c56def9baccc71c34496bbc6078a86e5bcf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31841264609)
