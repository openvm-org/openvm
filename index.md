| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 1,604 |  12,000,265 |  364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 9,404 |  18,655,329 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 4,854 |  14,793,960 |  574 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 653 |  4,137,067 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 435 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 598 |  1,745,757 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-b820b25baab6c5d9b055f64e0286b6b1058e707c.md) | 2,217 |  2,579,903 |  479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b820b25baab6c5d9b055f64e0286b6b1058e707c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29510200632)
