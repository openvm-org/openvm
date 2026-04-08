| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/fibonacci-ee03d35f307a138ea86221b1f96df6dc6c6b5107.md) | 3,818 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/keccak-ee03d35f307a138ea86221b1f96df6dc6c6b5107.md) | 18,531 |  18,655,329 |  3,306 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/regex-ee03d35f307a138ea86221b1f96df6dc6c6b5107.md) | 1,422 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/ecrecover-ee03d35f307a138ea86221b1f96df6dc6c6b5107.md) | 649 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/pairing-ee03d35f307a138ea86221b1f96df6dc6c6b5107.md) | 906 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/kitchen_sink-ee03d35f307a138ea86221b1f96df6dc6c6b5107.md) | 2,278 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ee03d35f307a138ea86221b1f96df6dc6c6b5107

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24141587534)
