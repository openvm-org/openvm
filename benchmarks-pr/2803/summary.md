| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 3,758 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 18,407 |  18,655,329 |  3,244 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 10,076 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 1,382 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 603 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 892 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-ccd2dfad935c08db92fc95aff95822fa93a0b0a3.md) | 1,900 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ccd2dfad935c08db92fc95aff95822fa93a0b0a3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26251152868)
