| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 486 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 7,323 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 4,158 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 658 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 232 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 239 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-46f4e3f0277048492549eb0cca384474e6c3b17a.md) | 2,032 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/46f4e3f0277048492549eb0cca384474e6c3b17a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30888323443)
