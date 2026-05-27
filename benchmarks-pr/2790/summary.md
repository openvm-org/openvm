| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 3,751 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 18,562 |  18,655,329 |  3,277 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 10,350 |  14,793,960 |  1,489 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 1,377 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 614 |  123,583 |  258 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 901 |  1,745,757 |  270 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 1,896 |  2,579,903 |  413 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 1,780 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 812 |  4,137,067 |  168 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 515 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 632 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-9ede114de0b5acfe760fc635ed68c032db0b33f7.md) | 2,020 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9ede114de0b5acfe760fc635ed68c032db0b33f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26538071161)
