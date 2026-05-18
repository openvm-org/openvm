| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 3,770 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 18,360 |  18,655,329 |  3,240 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 10,359 |  14,793,960 |  1,485 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 1,405 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 608 |  123,583 |  243 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 884 |  1,745,757 |  269 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 1,890 |  2,579,903 |  410 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 1,771 |  12,000,265 |  409 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 826 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 512 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 631 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-0110850a7864e278b0be4198b804faa6adb87c5f.md) | 2,034 |  2,579,903 |  402 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0110850a7864e278b0be4198b804faa6adb87c5f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26053725461)
