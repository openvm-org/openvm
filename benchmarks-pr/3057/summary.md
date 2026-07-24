| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/fibonacci-bd2f706b3959809bc595edc92191f51981b60595.md) | 462 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/keccak-bd2f706b3959809bc595edc92191f51981b60595.md) | 7,257 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/sha2_bench-bd2f706b3959809bc595edc92191f51981b60595.md) | 4,677 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/regex-bd2f706b3959809bc595edc92191f51981b60595.md) | 654 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/ecrecover-bd2f706b3959809bc595edc92191f51981b60595.md) | 278 |  78,475 |  226 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/pairing-bd2f706b3959809bc595edc92191f51981b60595.md) | 307 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/kitchen_sink-bd2f706b3959809bc595edc92191f51981b60595.md) | 2,934 |  2,341,811 |  551 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bd2f706b3959809bc595edc92191f51981b60595

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30132975707)
