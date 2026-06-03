| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 3,775 |  12,000,265 |  929 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 18,439 |  18,655,329 |  3,252 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 10,274 |  14,793,960 |  1,474 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 1,403 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 595 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 898 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-2c88f51e165e826938845ea49ed2496dd02dd895.md) | 1,902 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c88f51e165e826938845ea49ed2496dd02dd895

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26903281307)
