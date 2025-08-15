#pragma once

#include "fp.h"

namespace poseidon2 {

static __device__ __constant__ Fp INITIAL_ROUND_CONSTANTS[64] = {
    1774958255, 1185780729, 1621102414, 1796380621, 588815102,  1932426223, 1925334750, 747903232,
    89648862,   360728943,  977184635,  1425273457, 256487465,  1200041953, 572403254,  448208942,
    1215789478, 944884184,  953948096,  547326025,  646827752,  889997530,  1536873262, 86189867,
    1065944411, 32019634,   333311454,  456061748,  1963448500, 1827584334, 1391160226, 1348741381,
    88424255,   104111868,  1763866748, 79691676,   1988915530, 1050669594, 359890076,  573163527,
    222820492,  159256268,  669703072,  763177444,  889367200,  256335831,  704371273,  25886717,
    51754520,   1833211857, 454499742,  1384520381, 777848065,  1053320300, 1851729162, 344647910,
    401996362,  1046925956, 5351995,    1212119315, 754867989,  36972490,   751272725,  506915399
};
static __device__ __constant__ Fp TERMINAL_ROUND_CONSTANTS[64] = {
    1922082829, 1870549801, 1502529704, 1990744480, 1700391016, 1702593455, 321330495,  528965731,
    183414327,  1886297254, 1178602734, 1923111974, 744004766,  549271463,  1781349648, 542259047,
    1536158148, 715456982,  503426110,  340311124,  1558555932, 1226350925, 742828095,  1338992758,
    1641600456, 1843351545, 301835475,  43203215,   386838401,  1520185679, 1235297680, 904680097,
    1491801617, 1581784677, 913384905,  247083962,  532844013,  107190701,  213827818,  1979521776,
    1358282574, 1681743681, 1867507480, 1530706910, 507181886,  695185447,  1172395131, 1250800299,
    1503161625, 817684387,  498481458,  494676004,  1404253825, 108246855,  59414691,   744214112,
    890862029,  1342765939, 1417398904, 1897591937, 1066647396, 1682806907, 1015795079, 1619482808
};
static __device__ __constant__ Fp INTERNAL_ROUND_CONSTANTS[13] = {
    1518359488,
    1765533241,
    945325693,
    422793067,
    311365592,
    1311448267,
    1629555936,
    1009879353,
    190525218,
    786108885,
    557776863,
    212616710,
    605745517
};

// V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
static __device__ __constant__ Fp internal_diag16[16] = {
    -Fp::one().doubled().doubled(),
    Fp::one(),
    Fp::one().doubled(),
    Fp::one().mul_2exp_neg_n(1),
    Fp::one().doubled() + Fp::one(),
    Fp::one().doubled().doubled(),
    -(Fp::one().mul_2exp_neg_n(1)),
    -(Fp::one().doubled() + Fp::one()),
    -(Fp::one().doubled().doubled()),
    Fp::one().mul_2exp_neg_n(8),
    Fp::one().mul_2exp_neg_n(2),
    Fp::one().mul_2exp_neg_n(3),
    Fp::one().mul_2exp_neg_n(27),
    -(Fp::one().mul_2exp_neg_n(8)),
    -(Fp::one().mul_2exp_neg_n(4)),
    -(Fp::one().mul_2exp_neg_n(27)),
};


#define CELLS 16
#define CELLS_RATE 8
#define CELLS_OUT 8

#define ROUNDS_FULL 8
#define ROUNDS_HALF_FULL (ROUNDS_FULL / 2)
#define ROUNDS_PARTIAL 13

static __device__ Fp sbox_d7(Fp x) {
    Fp x2 = x * x;
    Fp x4 = x2 * x2;
    Fp x6 = x4 * x2;
    return x6 * x;
}

static __device__ void do_full_sboxes(Fp *cells) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = sbox_d7(cells[i]);
    }
}

static __device__ void do_partial_sboxes(Fp *cells) { 
    cells[0] = sbox_d7(cells[0]); 
}

// Plonky3 version
// Multiply a 4-element vector x by:
// [ 2 3 1 1 ]
// [ 1 2 3 1 ]
// [ 1 1 2 3 ]
// [ 3 1 1 2 ].
static __device__ void multiply_by_4x4_circulant(Fp *x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];

    x[3] = t01233 + Fp(2) * x[0];
    x[1] = t01123 + Fp(2) * x[2];
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

static __device__ void multiply_by_m_ext(Fp *old_cells) {
    // Optimized method for multiplication by M_EXT.
    // See appendix B of Poseidon2 paper for additional details.
    Fp cells[CELLS];
    for (uint i = 0; i < CELLS; i++) {
        cells[0] = 0;
    }
    Fp tmp_sums[4];
    for (uint i = 0; i < 4; i++) {
        tmp_sums[i] = 0;
    }
    for (uint i = 0; i < CELLS / 4; i++) {
        multiply_by_4x4_circulant(old_cells + i * 4);
        for (uint j = 0; j < 4; j++) {
            Fp to_add = old_cells[i * 4 + j];
            tmp_sums[j] += to_add;
            cells[i * 4 + j] += to_add;
        }
    }
    for (uint i = 0; i < CELLS; i++) {
        old_cells[i] = cells[i] + tmp_sums[i % 4];
    }
}

static __device__ void add_round_constants_full(const Fp *ROUND_CONSTANTS_PLONKY3, Fp *cells, uint round) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] += ROUND_CONSTANTS_PLONKY3[round * CELLS + i];
    }
}

static __device__ void add_round_constants_partial(
    const Fp *PARTIAL_ROUND_CONSTANTS_PLONKY3,
    Fp *cells,
    uint round
) {
    cells[0] += PARTIAL_ROUND_CONSTANTS_PLONKY3[round];
}

static __device__ __forceinline__ void internal_layer_mat_mul(Fp* cells, Fp sum) {
    cells[1] += sum;
#pragma unroll
    for (int i = 2; i < CELLS; i++) {
        cells[i] = sum + cells[i] * internal_diag16[i];
    }
}

static __device__ void full_round_half(const Fp *ROUND_CONSTANTS, Fp *cells, uint round) {
    add_round_constants_full(ROUND_CONSTANTS, cells, round);
    do_full_sboxes(cells);
    multiply_by_m_ext(cells);
}

static __device__ void partial_round(const Fp *PARTIAL_ROUND_CONSTANTS, Fp *cells, uint round) {
    add_round_constants_partial(PARTIAL_ROUND_CONSTANTS, cells, round);
    do_partial_sboxes(cells);
    Fp part_sum = Fp(0);
    for (uint i = 1; i < CELLS; i++) {
        part_sum += cells[i];
    }
    Fp full_sum = part_sum + cells[0];
    cells[0] = part_sum - cells[0];
    internal_layer_mat_mul(cells, full_sum);
}

static __device__ void poseidon2_mix(Fp *cells) {
    // First linear layer.
    multiply_by_m_ext(cells);

    // perform initial full rounds (external)
    for (uint i = 0; i < ROUNDS_HALF_FULL; i++) {
        full_round_half(INITIAL_ROUND_CONSTANTS, cells, i);
    }

    // perform partial rounds (internal)
    for (uint i = 0; i < ROUNDS_PARTIAL; i++) {
        partial_round(INTERNAL_ROUND_CONSTANTS, cells, i);
    }

    // perform terminal full rounds (external)
    for (uint r = 0; r < ROUNDS_HALF_FULL; r++) {
        full_round_half(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
}
} // namespace poseidon2
