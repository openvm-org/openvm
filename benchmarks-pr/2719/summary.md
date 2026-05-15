| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 1,886 |  4,000,051 |  518 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 13,618 |  14,365,133 |  2,227 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 9,486 |  11,167,961 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 1,586 |  4,090,656 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 598 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 748 |  592,827 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-16f040497b957c12a30d2fd0f5cf41e020b16580.md) | 1,883 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/16f040497b957c12a30d2fd0f5cf41e020b16580

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25938142669)
