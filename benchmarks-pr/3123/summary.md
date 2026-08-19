| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/fibonacci-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 1,549 |  12,000,265 |  357 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/keccak-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 9,220 |  18,655,329 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/sha2_bench-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 4,895 |  14,793,960 |  578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/regex-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 669 |  4,137,067 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/ecrecover-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 427 |  123,583 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/pairing-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 570 |  1,745,757 |  192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/kitchen_sink-6c3ec0deb8174e281a05df6873d2bd800e63767c.md) | 2,178 |  2,579,903 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6c3ec0deb8174e281a05df6873d2bd800e63767c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32290040765)
