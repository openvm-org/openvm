| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-9535938601e0cf55038b552094adfb2410349595.md) | 468 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-9535938601e0cf55038b552094adfb2410349595.md) | 7,244 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-9535938601e0cf55038b552094adfb2410349595.md) | 4,716 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-9535938601e0cf55038b552094adfb2410349595.md) | 674 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-9535938601e0cf55038b552094adfb2410349595.md) | 226 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-9535938601e0cf55038b552094adfb2410349595.md) | 315 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-9535938601e0cf55038b552094adfb2410349595.md) | 2,684 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9535938601e0cf55038b552094adfb2410349595

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29860048499)
