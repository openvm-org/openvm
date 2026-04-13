| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-43fd22683c1eaba623e58b354d04002639eec179.md) | 3,866 |  12,000,265 |  963 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-43fd22683c1eaba623e58b354d04002639eec179.md) | 18,757 |  18,655,329 |  3,348 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-43fd22683c1eaba623e58b354d04002639eec179.md) | 10,082 |  14,793,960 |  1,427 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-43fd22683c1eaba623e58b354d04002639eec179.md) | 1,453 |  4,137,067 |  385 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-43fd22683c1eaba623e58b354d04002639eec179.md) | 642 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-43fd22683c1eaba623e58b354d04002639eec179.md) | 906 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-43fd22683c1eaba623e58b354d04002639eec179.md) | 2,153 |  2,579,903 |  433 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-43fd22683c1eaba623e58b354d04002639eec179.md) | 1,862 |  12,000,265 |  456 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-43fd22683c1eaba623e58b354d04002639eec179.md) | 861 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-43fd22683c1eaba623e58b354d04002639eec179.md) | 554 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-43fd22683c1eaba623e58b354d04002639eec179.md) | 657 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-43fd22683c1eaba623e58b354d04002639eec179.md) | 2,269 |  2,579,903 |  426 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/43fd22683c1eaba623e58b354d04002639eec179

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24365025565)
