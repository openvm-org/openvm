| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-db07bcaff018f973c682f6d2bf326333980b046a.md) | 3,845 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-db07bcaff018f973c682f6d2bf326333980b046a.md) | 18,861 |  18,655,329 |  3,361 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-db07bcaff018f973c682f6d2bf326333980b046a.md) | 10,033 |  14,793,960 |  1,417 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-db07bcaff018f973c682f6d2bf326333980b046a.md) | 1,427 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-db07bcaff018f973c682f6d2bf326333980b046a.md) | 643 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-db07bcaff018f973c682f6d2bf326333980b046a.md) | 904 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-db07bcaff018f973c682f6d2bf326333980b046a.md) | 2,165 |  2,579,903 |  436 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-db07bcaff018f973c682f6d2bf326333980b046a.md) | 1,863 |  12,000,265 |  452 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-db07bcaff018f973c682f6d2bf326333980b046a.md) | 854 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-db07bcaff018f973c682f6d2bf326333980b046a.md) | 554 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-db07bcaff018f973c682f6d2bf326333980b046a.md) | 659 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-db07bcaff018f973c682f6d2bf326333980b046a.md) | 2,271 |  2,579,903 |  425 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db07bcaff018f973c682f6d2bf326333980b046a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24417196671)
