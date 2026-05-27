| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/fibonacci-779f183d468b10bf9f7dec020564c82a50416450.md) | 3,805 |  12,000,265 |  929 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/keccak-779f183d468b10bf9f7dec020564c82a50416450.md) | 18,730 |  18,655,329 |  3,300 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/sha2_bench-779f183d468b10bf9f7dec020564c82a50416450.md) | 10,136 |  14,793,960 |  1,456 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/regex-779f183d468b10bf9f7dec020564c82a50416450.md) | 1,391 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/ecrecover-779f183d468b10bf9f7dec020564c82a50416450.md) | 606 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/pairing-779f183d468b10bf9f7dec020564c82a50416450.md) | 892 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/kitchen_sink-779f183d468b10bf9f7dec020564c82a50416450.md) | 1,905 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/779f183d468b10bf9f7dec020564c82a50416450

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26528436005)
