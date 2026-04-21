| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/fibonacci-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 3,811 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/keccak-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 18,616 |  18,655,329 |  3,324 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/sha2_bench-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 8,971 |  14,793,960 |  1,382 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/regex-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 1,396 |  4,137,067 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/ecrecover-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 639 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/pairing-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 912 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/kitchen_sink-5d25ac9791ee17e4c2908a4341dac4153c0ce3a0.md) | 2,082 |  2,579,903 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5d25ac9791ee17e4c2908a4341dac4153c0ce3a0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24732393420)
