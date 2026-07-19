| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 417 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 8,758 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 4,163 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 571 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 223 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 292 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-bfb19800bbfb23cb4e923bbc80d30ec2a3b37720.md) | 1,915 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bfb19800bbfb23cb4e923bbc80d30ec2a3b37720

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29684197207)
