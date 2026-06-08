| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/fibonacci-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 3,776 |  12,000,265 |  934 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/keccak-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 18,030 |  18,655,329 |  3,284 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/sha2_bench-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 10,109 |  14,793,960 |  1,475 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/regex-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 1,383 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/ecrecover-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 602 |  123,583 |  256 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/pairing-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 881 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/kitchen_sink-4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09.md) | 3,892 |  2,579,903 |  960 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4cdf911143b4fe2c2f8ed0a6a409ab2799ad5a09

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27148448232)
