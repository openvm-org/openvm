| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/fibonacci-e06248452f30f73ee2676d3f649147a2f9489310.md) | 3,739 |  12,000,265 |  914 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/keccak-e06248452f30f73ee2676d3f649147a2f9489310.md) | 19,160 |  18,655,329 |  3,355 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/sha2_bench-e06248452f30f73ee2676d3f649147a2f9489310.md) | 10,158 |  14,793,960 |  1,450 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/regex-e06248452f30f73ee2676d3f649147a2f9489310.md) | 1,386 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/ecrecover-e06248452f30f73ee2676d3f649147a2f9489310.md) | 604 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/pairing-e06248452f30f73ee2676d3f649147a2f9489310.md) | 892 |  1,745,757 |  269 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/kitchen_sink-e06248452f30f73ee2676d3f649147a2f9489310.md) | 1,899 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e06248452f30f73ee2676d3f649147a2f9489310

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26649863575)
