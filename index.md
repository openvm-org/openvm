| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-c65f9fa74c74f57bc94262314b75178f43202757.md) | 1,570 |  12,000,265 |  359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-c65f9fa74c74f57bc94262314b75178f43202757.md) | 9,322 |  18,655,329 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-c65f9fa74c74f57bc94262314b75178f43202757.md) | 4,891 |  14,793,960 |  579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-c65f9fa74c74f57bc94262314b75178f43202757.md) | 656 |  4,137,067 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-c65f9fa74c74f57bc94262314b75178f43202757.md) | 435 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-c65f9fa74c74f57bc94262314b75178f43202757.md) | 561 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-c65f9fa74c74f57bc94262314b75178f43202757.md) | 2,193 |  2,579,903 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c65f9fa74c74f57bc94262314b75178f43202757

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31526321774)
