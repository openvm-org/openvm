| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/fibonacci-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 3,126 |  12,000,265 |  679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/keccak-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 16,875 |  18,655,329 |  3,080 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/sha2_bench-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 9,753 |  14,793,960 |  1,154 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/regex-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 1,299 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/ecrecover-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 584 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/pairing-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 911 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/kitchen_sink-bcd4d0318e1da0b9d679bea80bcd89105c31ea2c.md) | 4,590 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bcd4d0318e1da0b9d679bea80bcd89105c31ea2c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29058131705)
