| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 408 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 8,609 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 4,172 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 496 |  4,090,656 |  187 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 220 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 264 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-f23c0435916286a410bfd29c523af42d5f559c5d.md) | 1,881 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f23c0435916286a410bfd29c523af42d5f559c5d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29433783036)
