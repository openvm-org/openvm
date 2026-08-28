| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 1,669 |  12,000,265 |  369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 9,533 |  18,655,329 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 5,348 |  14,793,960 |  595 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 704 |  4,137,067 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 438 |  123,583 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 570 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-37db4afed6e0556991d9ebfc069c1a519ba88697.md) | 2,296 |  2,579,903 |  499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/37db4afed6e0556991d9ebfc069c1a519ba88697

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33212366638)
