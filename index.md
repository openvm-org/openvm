| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 3,092 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 16,425 |  18,655,329 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 9,249 |  14,793,960 |  1,128 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 1,182 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 602 |  123,583 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 934 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 4,130 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28976260289)
