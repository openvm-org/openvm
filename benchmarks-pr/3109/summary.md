| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 479 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 7,407 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 4,141 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 660 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 229 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 231 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-802dadd6e40c14384be79ff2ee7c3fab9686af33.md) | 2,027 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/802dadd6e40c14384be79ff2ee7c3fab9686af33

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31046534230)
