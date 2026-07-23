| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 481 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 7,321 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 4,714 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 680 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 229 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 279 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-91b9b5e42c139101a3a2248b65ac668f33dced02.md) | 2,745 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/91b9b5e42c139101a3a2248b65ac668f33dced02

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29982280660)
