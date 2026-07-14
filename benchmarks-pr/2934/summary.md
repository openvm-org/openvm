| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/fibonacci-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 462 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/keccak-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 8,741 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/sha2_bench-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 3,963 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/regex-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 507 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/ecrecover-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 217 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/pairing-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 263 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/kitchen_sink-809d9b031d7b0d40d9d566942f7b08ec2c91ca1c.md) | 1,924 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/809d9b031d7b0d40d9d566942f7b08ec2c91ca1c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29371993679)
