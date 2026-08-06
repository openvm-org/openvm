| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 478 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 7,326 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 4,145 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 653 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 224 |  112,210 |  195 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 235 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-72a1ca0e97c8ce77c6ac81bc311ee33840638198.md) | 2,036 |  1,979,971 |  528 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/72a1ca0e97c8ce77c6ac81bc311ee33840638198

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31130698290)
