| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 3,887 |  12,000,265 |  964 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 18,949 |  18,655,329 |  3,375 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 9,165 |  14,793,960 |  1,417 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 1,405 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 638 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 908 |  1,745,757 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 2,092 |  2,579,903 |  434 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 1,856 |  12,000,265 |  458 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 853 |  4,137,067 |  195 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 547 |  123,583 |  153 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 645 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-337ec13588d914317f65bdecb6865dd3c3c38441.md) | 2,204 |  2,579,903 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/337ec13588d914317f65bdecb6865dd3c3c38441

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25341240214)
