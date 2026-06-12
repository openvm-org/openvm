| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/fibonacci-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 3,966 |  12,000,265 |  1,139 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/keccak-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 21,766 |  18,655,329 |  4,611 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/sha2_bench-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 9,673 |  14,793,960 |  1,858 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/regex-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 1,493 |  4,137,067 |  426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/ecrecover-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 602 |  123,583 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/pairing-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 940 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/kitchen_sink-092f0499faa2f701f4a44cee819edbd3674ab49d.md) | 4,128 |  2,579,903 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/092f0499faa2f701f4a44cee819edbd3674ab49d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27430735502)
