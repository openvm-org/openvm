| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 1,422 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 13,146 |  14,365,133 |  2,189 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 9,096 |  11,167,961 |  1,434 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 1,340 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 469 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 589 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f.md) | 1,775 |  1,979,971 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e0c12c4ffdaa2ed6a8aa6adfb500c9615579176f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25927612721)
