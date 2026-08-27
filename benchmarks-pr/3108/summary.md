| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 445 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 7,269 |  14,365,133 |  1,570 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 4,037 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 759 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 207 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 238 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-9f33138fdcc08d8a3ba11f4f6853e595b64a0a13.md) | 2,142 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9f33138fdcc08d8a3ba11f4f6853e595b64a0a13

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33106445891)
