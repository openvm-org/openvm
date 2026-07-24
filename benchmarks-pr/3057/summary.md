| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/fibonacci-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 470 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/keccak-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 7,340 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/sha2_bench-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 4,702 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/regex-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 678 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/ecrecover-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 275 |  78,475 |  226 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/pairing-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 326 |  592,827 |  201 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3057/kitchen_sink-a17b3cdb0a970c07a3f22dea69c27d3b22ae7955.md) | 2,989 |  2,341,811 |  554 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a17b3cdb0a970c07a3f22dea69c27d3b22ae7955

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30121051721)
