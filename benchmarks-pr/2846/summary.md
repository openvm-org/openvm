| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 3,086 |  12,000,265 | <span style='color: green'>(-3807 [-84.9%])</span> 679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 16,174 |  18,655,329 |  3,003 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 9,292 |  14,793,960 |  1,131 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 1,195 |  4,137,067 | <span style='color: green'>(-11638 [-97.0%])</span> 359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 600 |  123,583 | <span style='color: green'>(-5575 [-95.2%])</span> 281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 938 |  1,745,757 | <span style='color: green'>(-6071 [-95.2%])</span> 309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 4,119 |  2,579,903 |  882 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 1,374 |  12,000,265 |  284 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 619 |  4,137,067 |  166 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 369 |  123,583 |  144 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 505 |  1,745,757 |  150 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-191cfb8de2e6cee00f440f6490082f62ac3099bf.md) | 2,172 |  2,579,903 |  383 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/191cfb8de2e6cee00f440f6490082f62ac3099bf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27652122588)
