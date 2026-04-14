| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 3,845 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 18,695 |  18,655,329 |  3,334 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 9,110 |  14,793,960 |  1,405 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 1,440 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 642 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 914 |  1,745,757 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 2,089 |  2,579,903 |  431 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 1,866 |  12,000,265 |  456 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 850 |  4,137,067 |  193 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 555 |  123,583 |  154 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 654 |  1,745,757 |  154 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-a6a9bc150ae63e3486e1aa311db1545ee71df4e1.md) | 2,219 |  2,579,903 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a6a9bc150ae63e3486e1aa311db1545ee71df4e1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24422847389)
