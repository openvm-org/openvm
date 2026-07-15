| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/fibonacci-b886348adf2805b223b87be0064865a5feed0828.md) | 414 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/keccak-b886348adf2805b223b87be0064865a5feed0828.md) | 8,432 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/sha2_bench-b886348adf2805b223b87be0064865a5feed0828.md) | 3,939 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/regex-b886348adf2805b223b87be0064865a5feed0828.md) | 573 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/ecrecover-b886348adf2805b223b87be0064865a5feed0828.md) | 219 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/pairing-b886348adf2805b223b87be0064865a5feed0828.md) | 269 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/kitchen_sink-b886348adf2805b223b87be0064865a5feed0828.md) | 1,881 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b886348adf2805b223b87be0064865a5feed0828

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29450153204)
