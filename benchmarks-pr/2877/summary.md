| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/fibonacci-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 4,044 |  12,000,265 |  1,173 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/keccak-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 21,669 |  18,655,329 |  4,607 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/sha2_bench-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 9,594 |  14,793,960 |  1,841 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/regex-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 1,524 |  4,137,067 |  436 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/ecrecover-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 608 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/pairing-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 936 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/kitchen_sink-e2ead72f5ef291c81a6249869216c2458e145df4.md) | 4,143 |  2,579,903 |  875 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e2ead72f5ef291c81a6249869216c2458e145df4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27432581854)
