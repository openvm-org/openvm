| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 479 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 10,259 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 4,672 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 682 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 228 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 275 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-1c628f124c06ea57f0fb33035d11f8e814bb3416.md) | 2,393 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1c628f124c06ea57f0fb33035d11f8e814bb3416

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30138057825)
