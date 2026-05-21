| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 1,877 |  4,000,051 |  518 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 13,501 |  14,365,133 |  2,200 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 9,336 |  11,167,961 |  1,384 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 1,551 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 602 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 735 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-391e4f0dc0fc8ae643fa030654b9fecee0f519ea.md) | 1,868 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/391e4f0dc0fc8ae643fa030654b9fecee0f519ea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26230597998)
