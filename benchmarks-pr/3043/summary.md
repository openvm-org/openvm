| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 415 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 8,500 |  14,365,133 |  1,505 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 4,267 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 572 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 216 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 293 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7.md) | 1,932 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ed5fc27640a5824464d6302adc1fdf5dbbaf7aa7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29656172387)
