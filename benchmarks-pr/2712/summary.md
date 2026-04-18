| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 3,873 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 18,638 |  18,655,329 |  3,315 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 9,073 |  14,793,960 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 1,429 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 653 |  123,583 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 910 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-9341ec7eea2252abd173ab6f33cb400cf418aea0.md) | 2,084 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9341ec7eea2252abd173ab6f33cb400cf418aea0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24592681567)
