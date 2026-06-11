| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 3,957 |  12,000,265 | <span style='color: green'>(-3342 [-74.5%])</span> 1,144 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 21,928 |  18,655,329 |  4,659 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 9,442 |  14,793,960 |  1,828 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 1,481 |  4,137,067 | <span style='color: green'>(-11574 [-96.5%])</span> 423 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 612 |  123,583 | <span style='color: green'>(-5573 [-95.2%])</span> 283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 935 |  1,745,757 | <span style='color: green'>(-6073 [-95.2%])</span> 307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 4,116 |  2,579,903 |  880 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 1,711 |  12,000,265 |  493 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 723 |  4,137,067 |  195 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 368 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 509 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-28122a9e79f224f63b39b508f0785f8e7a720880.md) | 2,159 |  2,579,903 |  385 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/28122a9e79f224f63b39b508f0785f8e7a720880

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27313958471)
