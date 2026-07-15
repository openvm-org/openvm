| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 469 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 7,226 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 4,450 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 687 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 230 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 253 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-89750e0dbde71bba3bf107bb4b6adda13be44b2b.md) | 2,663 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/89750e0dbde71bba3bf107bb4b6adda13be44b2b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29431366753)
