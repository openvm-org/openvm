| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 1,695 |  12,000,265 |  372 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 9,538 |  18,655,329 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 5,243 |  14,793,960 |  586 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 709 |  4,137,067 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 442 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 590 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-f0461583b76731fa88bd4332e32c3ac6904aed3f.md) | 2,290 |  2,579,903 |  497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f0461583b76731fa88bd4332e32c3ac6904aed3f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33190467967)
