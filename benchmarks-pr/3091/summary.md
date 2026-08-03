| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/fibonacci-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 473 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/keccak-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 7,501 |  14,365,133 |  1,542 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/sha2_bench-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 4,176 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/regex-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 644 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/ecrecover-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 230 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/pairing-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 242 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3091/kitchen_sink-544a63f3de8a04a4f3787d7cf5004733aeb5403a.md) | 2,032 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/544a63f3de8a04a4f3787d7cf5004733aeb5403a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30827365473)
