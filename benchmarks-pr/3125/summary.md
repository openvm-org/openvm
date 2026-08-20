| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/fibonacci-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 1,585 |  12,000,265 |  362 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/keccak-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 9,204 |  18,655,329 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/sha2_bench-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 4,915 |  14,793,960 |  574 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/regex-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 669 |  4,137,067 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/ecrecover-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 435 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/pairing-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 577 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3125/kitchen_sink-586d800fbeb9b7d1f08d57289050025ffed23a31.md) | 2,195 |  2,579,903 |  479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/586d800fbeb9b7d1f08d57289050025ffed23a31

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32398992791)
