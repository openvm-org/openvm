| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 410 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 8,748 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 4,263 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 576 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 216 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 284 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-a1725ff177678b64cb5aa7cfd5c2471b6cb03966.md) | 1,937 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a1725ff177678b64cb5aa7cfd5c2471b6cb03966

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29652957612)
