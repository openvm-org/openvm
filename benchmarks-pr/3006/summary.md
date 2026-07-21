| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 474 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 7,230 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 4,661 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 674 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 231 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 325 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-1e3f6759db8a60ffbd7499f7aa5d19d14076f477.md) | 2,596 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1e3f6759db8a60ffbd7499f7aa5d19d14076f477

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29830155724)
