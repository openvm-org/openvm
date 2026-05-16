| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 1,412 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 13,224 |  14,365,133 |  2,191 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 8,969 |  11,167,961 |  1,409 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 1,351 |  4,090,656 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 468 |  112,210 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 591 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-80f197b5ed0efa972279de31b1fa6a6681ec2e7a.md) | 2,197 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80f197b5ed0efa972279de31b1fa6a6681ec2e7a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25966923792)
