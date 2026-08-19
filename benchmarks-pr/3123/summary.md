| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/fibonacci-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 1,561 |  12,000,265 |  359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/keccak-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 9,340 |  18,655,329 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/sha2_bench-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 4,951 |  14,793,960 |  577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/regex-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 675 |  4,137,067 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/ecrecover-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 420 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/pairing-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 578 |  1,745,757 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3123/kitchen_sink-80d4cdf2055edc333e0f76ab42412b68c2a5cb4a.md) | 2,220 |  2,579,903 |  480 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80d4cdf2055edc333e0f76ab42412b68c2a5cb4a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32289520205)
