| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/fibonacci-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 3,766 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/keccak-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 18,827 |  18,655,329 |  3,321 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/sha2_bench-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 10,332 |  14,793,960 |  1,480 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/regex-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 1,399 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/ecrecover-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 597 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/pairing-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 897 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/kitchen_sink-a13b1497d75b72fe5e09d851dd956871fd3d68b2.md) | 1,898 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a13b1497d75b72fe5e09d851dd956871fd3d68b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26178731433)
