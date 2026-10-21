// Lean compiler output
// Module: Mathlib.Logic.Equiv.Sum
// Imports: public import Init public meta import Init public import Mathlib.Data.Option.Defs public import Mathlib.Data.Sigma.Basic public import Mathlib.Logic.Equiv.Prod public import Mathlib.Tactic.Coe
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* l_Sum_swap(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Sum_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Option_elim_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
lean_object* l_Sum_elim___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumProdDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_psumEquivSum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psumEquivSum___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psumEquivSum___closed__0 = (const lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_psumEquivSum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psumEquivSum___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psumEquivSum___closed__1 = (const lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_psumEquivSum___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_psumEquivSum___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_psumEquivSum___closed__2 = (const lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_psumEquivSum___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__1_value),((lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_psumEquivSum___closed__3 = (const lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_psumEquivSum___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__0_value),((lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__3_value)}};
static const lean_object* lp_mathlib_Equiv_psumEquivSum___closed__4 = (const lean_object*)&lp_mathlib_Equiv_psumEquivSum___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_psumSum___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_psumSum___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeSum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeSum___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeSum___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeSum___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeSum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeSum___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeSum___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeSum___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeSum___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__2 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__1_value),((lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__3 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__3_value)}};
static const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__4 = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit = (const lean_object*)&lp_mathlib_Equiv_boolEquivPUnitSumPUnit___closed__4_value;
static const lean_closure_object lp_mathlib_Equiv_sumComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_swap, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_sumComm___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumComm___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_sumComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumComm___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumComm___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_sumComm___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumComm___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumComm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__5(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumAssoc___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumAssoc___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumAssoc___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumAssoc___lam__3, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__3_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumAssoc___lam__4, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__4_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumAssoc___lam__5, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__5 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__5_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__6 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__6_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__6_value),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__3_value)} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__7 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__7_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__5_value),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__1_value)} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__8 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__8_value;
static const lean_closure_object lp_mathlib_Equiv_sumAssoc___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__4_value),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__8_value)} };
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__9 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__9_value;
static const lean_ctor_object lp_mathlib_Equiv_sumAssoc___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__7_value),((lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__9_value)}};
static const lean_object* lp_mathlib_Equiv_sumAssoc___closed__10 = (const lean_object*)&lp_mathlib_Equiv_sumAssoc___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumSumSumComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumSumSumComm___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__4;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__5;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__6;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__7;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__8;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__9;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__10;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__11;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__12;
static lean_once_cell_t lp_mathlib_Equiv_sumSumSumComm___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumEmpty___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumEmpty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumEmpty___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumEmpty___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sumEmpty___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_sumEmpty___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sumEmpty___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__2_value),((lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__0_value)} };
static const lean_object* lp_mathlib_Equiv_sumEmpty___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_sumEmpty___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__3_value),((lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sumEmpty___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sumEmpty___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_emptySum___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_emptySum___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_emptySum___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_emptySum___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptySum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumEquivSigmaBool___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumEquivSigmaBool___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumEquivSigmaBool___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumEquivSigmaBool___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sumEquivSigmaBool___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumEquivSigmaBool___lam__2, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__1_value)} };
static const lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sumEquivSigmaBool___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumEquivSigmaBool___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_sumEquivSigmaBool___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__2_value),((lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__3_value)}};
static const lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sumEquivSigmaBool___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaFiberEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaFiberEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumCompl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Equiv_sumCompl___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumCompl___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_prodSumDistrib___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_prodSumDistrib___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_prodSumDistrib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_prodSumDistrib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_prodSumDistrib___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodSumDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaSumDistrib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaSumDistrib___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaSumDistrib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaSumDistrib___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaSumDistrib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaSumDistrib___lam__3___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaSumDistrib___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sigma_map, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumSumSumComm___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__1_value)} };
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__3_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaSumDistrib___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sigma_map, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumSumSumComm___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__4_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaSumDistrib___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__3_value),((lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__4_value)} };
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__5 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__5_value;
static const lean_ctor_object lp_mathlib_Equiv_sigmaSumDistrib___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__5_value)}};
static const lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__6 = (const lean_object*)&lp_mathlib_Equiv_sigmaSumDistrib___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumSigmaDistrib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumSigmaDistrib___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumSigmaDistrib___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumSigmaDistrib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumSigmaDistrib___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumSigmaDistrib___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sumSigmaDistrib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumSigmaDistrib___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumSigmaDistrib___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sumSigmaDistrib___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__1_value),((lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_sumSigmaDistrib___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_sumSigmaDistrib___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__3_value)}};
static const lean_object* lp_mathlib_Equiv_sumSigmaDistrib___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sumSigmaDistrib___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__0(lean_object* v_s_1_){
_start:
{
if (lean_obj_tag(v_s_1_) == 0)
{
lean_object* v_val_2_; lean_object* v___x_4_; uint8_t v_isShared_5_; uint8_t v_isSharedCheck_9_; 
v_val_2_ = lean_ctor_get(v_s_1_, 0);
v_isSharedCheck_9_ = !lean_is_exclusive(v_s_1_);
if (v_isSharedCheck_9_ == 0)
{
v___x_4_ = v_s_1_;
v_isShared_5_ = v_isSharedCheck_9_;
goto v_resetjp_3_;
}
else
{
lean_inc(v_val_2_);
lean_dec(v_s_1_);
v___x_4_ = lean_box(0);
v_isShared_5_ = v_isSharedCheck_9_;
goto v_resetjp_3_;
}
v_resetjp_3_:
{
lean_object* v___x_7_; 
if (v_isShared_5_ == 0)
{
v___x_7_ = v___x_4_;
goto v_reusejp_6_;
}
else
{
lean_object* v_reuseFailAlloc_8_; 
v_reuseFailAlloc_8_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_8_, 0, v_val_2_);
v___x_7_ = v_reuseFailAlloc_8_;
goto v_reusejp_6_;
}
v_reusejp_6_:
{
return v___x_7_;
}
}
}
else
{
lean_object* v_val_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_17_; 
v_val_10_ = lean_ctor_get(v_s_1_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v_s_1_);
if (v_isSharedCheck_17_ == 0)
{
v___x_12_ = v_s_1_;
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_val_10_);
lean_dec(v_s_1_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_15_; 
if (v_isShared_13_ == 0)
{
v___x_15_ = v___x_12_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v_val_10_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__1(lean_object* v_val_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_19_, 0, v_val_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__2(lean_object* v_val_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_21_, 0, v_val_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = ((lean_object*)(lp_mathlib_Equiv_psumEquivSum___closed__4));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__0(lean_object* v_ea_34_, lean_object* v___y_35_){
_start:
{
lean_object* v_toFun_36_; lean_object* v___x_37_; 
v_toFun_36_ = lean_ctor_get(v_ea_34_, 0);
lean_inc(v_toFun_36_);
lean_dec_ref(v_ea_34_);
v___x_37_ = lean_apply_1(v_toFun_36_, v___y_35_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__1(lean_object* v_eb_38_, lean_object* v___y_39_){
_start:
{
lean_object* v_toFun_40_; lean_object* v___x_41_; 
v_toFun_40_ = lean_ctor_get(v_eb_38_, 0);
lean_inc(v_toFun_40_);
lean_dec_ref(v_eb_38_);
v___x_41_ = lean_apply_1(v_toFun_40_, v___y_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__2(lean_object* v___x_42_, lean_object* v___y_43_){
_start:
{
lean_object* v_toFun_44_; lean_object* v___x_45_; 
v_toFun_44_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_toFun_44_);
lean_dec_ref(v___x_42_);
v___x_45_ = lean_apply_1(v_toFun_44_, v___y_43_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object* v_ea_46_, lean_object* v_eb_47_){
_start:
{
lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___f_52_; lean_object* v___x_53_; lean_object* v___f_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
lean_inc_ref(v_ea_46_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_48_, 0, v_ea_46_);
lean_inc_ref(v_eb_47_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_49_, 0, v_eb_47_);
v___x_50_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_50_, 0, lean_box(0));
lean_closure_set(v___x_50_, 1, lean_box(0));
lean_closure_set(v___x_50_, 2, lean_box(0));
lean_closure_set(v___x_50_, 3, lean_box(0));
lean_closure_set(v___x_50_, 4, v___f_48_);
lean_closure_set(v___x_50_, 5, v___f_49_);
v___x_51_ = lp_mathlib_Equiv_symm___redArg(v_ea_46_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_52_, 0, v___x_51_);
v___x_53_ = lp_mathlib_Equiv_symm___redArg(v_eb_47_);
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_54_, 0, v___x_53_);
v___x_55_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, lean_box(0));
lean_closure_set(v___x_55_, 2, lean_box(0));
lean_closure_set(v___x_55_, 3, lean_box(0));
lean_closure_set(v___x_55_, 4, v___f_52_);
lean_closure_set(v___x_55_, 5, v___f_54_);
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_50_);
lean_ctor_set(v___x_56_, 1, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr(lean_object* v_00_u03b1_u2081_57_, lean_object* v_00_u03b1_u2082_58_, lean_object* v_00_u03b2_u2081_59_, lean_object* v_00_u03b2_u2082_60_, lean_object* v_ea_61_, lean_object* v_eb_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_Equiv_sumCongr___redArg(v_ea_61_, v_eb_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__0(lean_object* v_e_u2081_64_, lean_object* v_e_u2082_65_, lean_object* v_x_66_){
_start:
{
if (lean_obj_tag(v_x_66_) == 0)
{
lean_object* v_a_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_76_; 
lean_dec_ref(v_e_u2082_65_);
v_a_67_ = lean_ctor_get(v_x_66_, 0);
v_isSharedCheck_76_ = !lean_is_exclusive(v_x_66_);
if (v_isSharedCheck_76_ == 0)
{
v___x_69_ = v_x_66_;
v_isShared_70_ = v_isSharedCheck_76_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_a_67_);
lean_dec(v_x_66_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_76_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v_toFun_71_; lean_object* v___x_72_; lean_object* v___x_74_; 
v_toFun_71_ = lean_ctor_get(v_e_u2081_64_, 0);
lean_inc(v_toFun_71_);
lean_dec_ref(v_e_u2081_64_);
v___x_72_ = lean_apply_1(v_toFun_71_, v_a_67_);
if (v_isShared_70_ == 0)
{
lean_ctor_set(v___x_69_, 0, v___x_72_);
v___x_74_ = v___x_69_;
goto v_reusejp_73_;
}
else
{
lean_object* v_reuseFailAlloc_75_; 
v_reuseFailAlloc_75_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_75_, 0, v___x_72_);
v___x_74_ = v_reuseFailAlloc_75_;
goto v_reusejp_73_;
}
v_reusejp_73_:
{
return v___x_74_;
}
}
}
else
{
lean_object* v_a_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_86_; 
lean_dec_ref(v_e_u2081_64_);
v_a_77_ = lean_ctor_get(v_x_66_, 0);
v_isSharedCheck_86_ = !lean_is_exclusive(v_x_66_);
if (v_isSharedCheck_86_ == 0)
{
v___x_79_ = v_x_66_;
v_isShared_80_ = v_isSharedCheck_86_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_a_77_);
lean_dec(v_x_66_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_86_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v_toFun_81_; lean_object* v___x_82_; lean_object* v___x_84_; 
v_toFun_81_ = lean_ctor_get(v_e_u2082_65_, 0);
lean_inc(v_toFun_81_);
lean_dec_ref(v_e_u2082_65_);
v___x_82_ = lean_apply_1(v_toFun_81_, v_a_77_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 0, v___x_82_);
v___x_84_ = v___x_79_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v___x_82_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__1(lean_object* v_e_u2081_87_, lean_object* v_e_u2082_88_, lean_object* v_x_89_){
_start:
{
if (lean_obj_tag(v_x_89_) == 0)
{
lean_object* v_a_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_100_; 
lean_dec_ref(v_e_u2082_88_);
v_a_90_ = lean_ctor_get(v_x_89_, 0);
v_isSharedCheck_100_ = !lean_is_exclusive(v_x_89_);
if (v_isSharedCheck_100_ == 0)
{
v___x_92_ = v_x_89_;
v_isShared_93_ = v_isSharedCheck_100_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_a_90_);
lean_dec(v_x_89_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_100_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_94_; lean_object* v_toFun_95_; lean_object* v___x_96_; lean_object* v___x_98_; 
v___x_94_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_87_);
v_toFun_95_ = lean_ctor_get(v___x_94_, 0);
lean_inc(v_toFun_95_);
lean_dec_ref(v___x_94_);
v___x_96_ = lean_apply_1(v_toFun_95_, v_a_90_);
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 0, v___x_96_);
v___x_98_ = v___x_92_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v___x_96_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
else
{
lean_object* v_a_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_111_; 
lean_dec_ref(v_e_u2081_87_);
v_a_101_ = lean_ctor_get(v_x_89_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v_x_89_);
if (v_isSharedCheck_111_ == 0)
{
v___x_103_ = v_x_89_;
v_isShared_104_ = v_isSharedCheck_111_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_a_101_);
lean_dec(v_x_89_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_111_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_105_; lean_object* v_toFun_106_; lean_object* v___x_107_; lean_object* v___x_109_; 
v___x_105_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_88_);
v_toFun_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc(v_toFun_106_);
lean_dec_ref(v___x_105_);
v___x_107_ = lean_apply_1(v_toFun_106_, v_a_101_);
if (v_isShared_104_ == 0)
{
lean_ctor_set(v___x_103_, 0, v___x_107_);
v___x_109_ = v___x_103_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v___x_107_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg(lean_object* v_e_u2081_112_, lean_object* v_e_u2082_113_){
_start:
{
lean_object* v___f_114_; lean_object* v___f_115_; lean_object* v___x_116_; 
lean_inc_ref(v_e_u2082_113_);
lean_inc_ref(v_e_u2081_112_);
v___f_114_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumCongr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_114_, 0, v_e_u2081_112_);
lean_closure_set(v___f_114_, 1, v_e_u2082_113_);
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumCongr___redArg___lam__1), 3, 2);
lean_closure_set(v___f_115_, 0, v_e_u2081_112_);
lean_closure_set(v___f_115_, 1, v_e_u2082_113_);
v___x_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_116_, 0, v___f_114_);
lean_ctor_set(v___x_116_, 1, v___f_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr(lean_object* v_00_u03b1_117_, lean_object* v_00_u03b2_118_, lean_object* v_00_u03b3_119_, lean_object* v_00_u03b4_120_, lean_object* v_e_u2081_121_, lean_object* v_e_u2082_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Equiv_psumCongr___redArg(v_e_u2081_121_, v_e_u2082_122_);
return v___x_123_;
}
}
static lean_object* _init_lp_mathlib_Equiv_psumSum___redArg___closed__0(void){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_Equiv_psumEquivSum(lean_box(0), lean_box(0));
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum___redArg(lean_object* v_ea_125_, lean_object* v_eb_126_){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = lp_mathlib_Equiv_psumCongr___redArg(v_ea_125_, v_eb_126_);
v___x_128_ = lean_obj_once(&lp_mathlib_Equiv_psumSum___redArg___closed__0, &lp_mathlib_Equiv_psumSum___redArg___closed__0_once, _init_lp_mathlib_Equiv_psumSum___redArg___closed__0);
v___x_129_ = lp_mathlib_Equiv_trans___redArg(v___x_127_, v___x_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum(lean_object* v_00_u03b1_u2081_130_, lean_object* v_00_u03b2_u2081_131_, lean_object* v_00_u03b1_u2082_132_, lean_object* v_00_u03b2_u2082_133_, lean_object* v_ea_134_, lean_object* v_eb_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_Equiv_psumSum___redArg(v_ea_134_, v_eb_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum___redArg(lean_object* v_ea_137_, lean_object* v_eb_138_){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_139_ = lp_mathlib_Equiv_symm___redArg(v_ea_137_);
v___x_140_ = lp_mathlib_Equiv_symm___redArg(v_eb_138_);
v___x_141_ = lp_mathlib_Equiv_psumSum___redArg(v___x_139_, v___x_140_);
v___x_142_ = lp_mathlib_Equiv_symm___redArg(v___x_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum(lean_object* v_00_u03b1_u2082_143_, lean_object* v_00_u03b2_u2082_144_, lean_object* v_00_u03b1_u2081_145_, lean_object* v_00_u03b2_u2081_146_, lean_object* v_ea_147_, lean_object* v_eb_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_Equiv_sumPSum___redArg(v_ea_147_, v_eb_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum___lam__0(lean_object* v_x_150_){
_start:
{
if (lean_obj_tag(v_x_150_) == 0)
{
lean_object* v_val_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_158_; 
v_val_151_ = lean_ctor_get(v_x_150_, 0);
v_isSharedCheck_158_ = !lean_is_exclusive(v_x_150_);
if (v_isSharedCheck_158_ == 0)
{
v___x_153_ = v_x_150_;
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_val_151_);
lean_dec(v_x_150_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_156_; 
if (v_isShared_154_ == 0)
{
v___x_156_ = v___x_153_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_val_151_);
v___x_156_ = v_reuseFailAlloc_157_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
return v___x_156_;
}
}
}
else
{
lean_object* v_val_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_166_; 
v_val_159_ = lean_ctor_get(v_x_150_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v_x_150_);
if (v_isSharedCheck_166_ == 0)
{
v___x_161_ = v_x_150_;
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_val_159_);
lean_dec(v_x_150_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_164_; 
if (v_isShared_162_ == 0)
{
v___x_164_ = v___x_161_;
goto v_reusejp_163_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v_val_159_);
v___x_164_ = v_reuseFailAlloc_165_;
goto v_reusejp_163_;
}
v_reusejp_163_:
{
return v___x_164_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum(lean_object* v_00_u03b1_170_, lean_object* v_00_u03b2_171_, lean_object* v_p_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = ((lean_object*)(lp_mathlib_Equiv_subtypeSum___closed__1));
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr___redArg(lean_object* v_ea_174_, lean_object* v_eb_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Equiv_sumCongr___redArg(v_ea_174_, v_eb_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr(lean_object* v_00_u03b1_177_, lean_object* v_00_u03b2_178_, lean_object* v_ea_179_, lean_object* v_eb_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Equiv_sumCongr___redArg(v_ea_179_, v_eb_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0(uint8_t v_b_186_){
_start:
{
if (v_b_186_ == 0)
{
lean_object* v___x_187_; 
v___x_187_ = ((lean_object*)(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0));
return v___x_187_;
}
else
{
lean_object* v___x_188_; 
v___x_188_ = ((lean_object*)(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1));
return v___x_188_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___boxed(lean_object* v_b_189_){
_start:
{
uint8_t v_b_boxed_190_; lean_object* v_res_191_; 
v_b_boxed_190_ = lean_unbox(v_b_189_);
v_res_191_ = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0(v_b_boxed_190_);
return v_res_191_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1(lean_object* v_x_192_){
_start:
{
uint8_t v___x_193_; 
v___x_193_ = 0;
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1___boxed(lean_object* v_x_194_){
_start:
{
uint8_t v_res_195_; lean_object* v_r_196_; 
v_res_195_ = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1(v_x_194_);
v_r_196_ = lean_box(v_res_195_);
return v_r_196_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2(lean_object* v_x_197_){
_start:
{
uint8_t v___x_198_; 
v___x_198_ = 1;
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2___boxed(lean_object* v_x_199_){
_start:
{
uint8_t v_res_200_; lean_object* v_r_201_; 
v_res_200_ = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2(v_x_199_);
v_r_201_ = lean_box(v_res_200_);
return v_r_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumComm(lean_object* v_00_u03b1_215_, lean_object* v_00_u03b2_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = ((lean_object*)(lp_mathlib_Equiv_sumComm___closed__1));
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__0(lean_object* v_val_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_219_, 0, v_val_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__1(lean_object* v_val_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_221_, 0, v_val_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__2(lean_object* v___y_222_){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_223_, 0, v___y_222_);
v___x_224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__3(lean_object* v___y_225_){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_226_, 0, v___y_225_);
v___x_227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_227_, 0, v___x_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__4(lean_object* v___y_228_){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_229_, 0, v___y_228_);
v___x_230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__5(lean_object* v___y_231_){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_232_, 0, v___y_231_);
v___x_233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc(lean_object* v_00_u03b1_255_, lean_object* v_00_u03b2_256_, lean_object* v_00_u03b3_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = ((lean_object*)(lp_mathlib_Equiv_sumAssoc___closed__10));
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0(lean_object* v___y_259_){
_start:
{
lean_inc(v___y_259_);
return v___y_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed(lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Equiv_sumSumSumComm___lam__0(v___y_260_);
lean_dec(v___y_260_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__2(lean_object* v___x_262_, lean_object* v___y_263_){
_start:
{
lean_object* v_toFun_264_; lean_object* v___x_265_; 
v_toFun_264_ = lean_ctor_get(v___x_262_, 0);
lean_inc(v_toFun_264_);
lean_dec_ref(v___x_262_);
v___x_265_ = lean_apply_1(v_toFun_264_, v___y_263_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__1(lean_object* v___x_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_toFun_268_; lean_object* v___x_269_; 
v_toFun_268_ = lean_ctor_get(v___x_266_, 0);
lean_inc(v_toFun_268_);
lean_dec_ref(v___x_266_);
v___x_269_ = lean_apply_1(v_toFun_268_, v___y_267_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__3(lean_object* v___x_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_toFun_272_; lean_object* v___x_273_; 
v_toFun_272_ = lean_ctor_get(v___x_270_, 0);
lean_inc(v_toFun_272_);
lean_dec_ref(v___x_270_);
v___x_273_ = lean_apply_1(v_toFun_272_, v___y_271_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__5(lean_object* v___x_274_, lean_object* v___f_275_, lean_object* v___f_276_, lean_object* v___x_277_, lean_object* v___x_278_, lean_object* v___f_279_, lean_object* v___y_280_){
_start:
{
lean_object* v_toFun_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v_toFun_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v_toFun_281_ = lean_ctor_get(v___x_274_, 0);
lean_inc(v_toFun_281_);
lean_dec_ref(v___x_274_);
v___x_282_ = lean_apply_1(v_toFun_281_, v___y_280_);
lean_inc_n(v___f_276_, 2);
v___x_283_ = l_Sum_map___redArg(v___f_275_, v___f_276_, v___x_282_);
v_toFun_284_ = lean_ctor_get(v___x_277_, 0);
lean_inc(v_toFun_284_);
lean_dec_ref(v___x_277_);
v___x_285_ = l_Sum_map___redArg(v___x_278_, v___f_276_, v___x_283_);
v___x_286_ = l_Sum_map___redArg(v___f_279_, v___f_276_, v___x_285_);
v___x_287_ = lean_apply_1(v_toFun_284_, v___x_286_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__1(void){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_mathlib_Equiv_sumAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_289_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__2(void){
_start:
{
lean_object* v___x_290_; lean_object* v___f_291_; 
v___x_290_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__1, &lp_mathlib_Equiv_sumSumSumComm___closed__1_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__1);
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__2), 2, 1);
lean_closure_set(v___f_291_, 0, v___x_290_);
return v___f_291_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__3(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_292_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__1, &lp_mathlib_Equiv_sumSumSumComm___closed__1_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__1);
v___x_293_ = lp_mathlib_Equiv_symm___redArg(v___x_292_);
return v___x_293_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__4(void){
_start:
{
lean_object* v___x_294_; lean_object* v___f_295_; 
v___x_294_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__3, &lp_mathlib_Equiv_sumSumSumComm___closed__3_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__3);
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__1), 2, 1);
lean_closure_set(v___f_295_, 0, v___x_294_);
return v___f_295_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__5(void){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return v___x_296_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__6(void){
_start:
{
lean_object* v___x_297_; lean_object* v___f_298_; 
v___x_297_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__5, &lp_mathlib_Equiv_sumSumSumComm___closed__5_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__5);
v___f_298_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__3), 2, 1);
lean_closure_set(v___f_298_, 0, v___x_297_);
return v___f_298_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__7(void){
_start:
{
lean_object* v___f_299_; lean_object* v___f_300_; lean_object* v___x_301_; 
v___f_299_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__6, &lp_mathlib_Equiv_sumSumSumComm___closed__6_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__6);
v___f_300_ = ((lean_object*)(lp_mathlib_Equiv_sumSumSumComm___closed__0));
v___x_301_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_301_, 0, lean_box(0));
lean_closure_set(v___x_301_, 1, lean_box(0));
lean_closure_set(v___x_301_, 2, lean_box(0));
lean_closure_set(v___x_301_, 3, lean_box(0));
lean_closure_set(v___x_301_, 4, v___f_300_);
lean_closure_set(v___x_301_, 5, v___f_299_);
return v___x_301_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__8(void){
_start:
{
lean_object* v___f_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___f_305_; lean_object* v___f_306_; lean_object* v___x_307_; lean_object* v___f_308_; 
v___f_302_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__4, &lp_mathlib_Equiv_sumSumSumComm___closed__4_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__4);
v___x_303_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__7, &lp_mathlib_Equiv_sumSumSumComm___closed__7_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__7);
v___x_304_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__1, &lp_mathlib_Equiv_sumSumSumComm___closed__1_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__1);
v___f_305_ = ((lean_object*)(lp_mathlib_Equiv_sumSumSumComm___closed__0));
v___f_306_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__2, &lp_mathlib_Equiv_sumSumSumComm___closed__2_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__2);
v___x_307_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__3, &lp_mathlib_Equiv_sumSumSumComm___closed__3_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__3);
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__5), 7, 6);
lean_closure_set(v___f_308_, 0, v___x_307_);
lean_closure_set(v___f_308_, 1, v___f_306_);
lean_closure_set(v___f_308_, 2, v___f_305_);
lean_closure_set(v___f_308_, 3, v___x_304_);
lean_closure_set(v___f_308_, 4, v___x_303_);
lean_closure_set(v___f_308_, 5, v___f_302_);
return v___f_308_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__9(void){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_309_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__5, &lp_mathlib_Equiv_sumSumSumComm___closed__5_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__5);
v___x_310_ = lp_mathlib_Equiv_symm___redArg(v___x_309_);
return v___x_310_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__10(void){
_start:
{
lean_object* v___x_311_; lean_object* v___f_312_; 
v___x_311_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__9, &lp_mathlib_Equiv_sumSumSumComm___closed__9_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__9);
v___f_312_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__3), 2, 1);
lean_closure_set(v___f_312_, 0, v___x_311_);
return v___f_312_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__11(void){
_start:
{
lean_object* v___f_313_; lean_object* v___f_314_; lean_object* v___x_315_; 
v___f_313_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__10, &lp_mathlib_Equiv_sumSumSumComm___closed__10_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__10);
v___f_314_ = ((lean_object*)(lp_mathlib_Equiv_sumSumSumComm___closed__0));
v___x_315_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_315_, 0, lean_box(0));
lean_closure_set(v___x_315_, 1, lean_box(0));
lean_closure_set(v___x_315_, 2, lean_box(0));
lean_closure_set(v___x_315_, 3, lean_box(0));
lean_closure_set(v___x_315_, 4, v___f_314_);
lean_closure_set(v___x_315_, 5, v___f_313_);
return v___x_315_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__12(void){
_start:
{
lean_object* v___f_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___f_319_; lean_object* v___f_320_; lean_object* v___x_321_; lean_object* v___f_322_; 
v___f_316_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__4, &lp_mathlib_Equiv_sumSumSumComm___closed__4_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__4);
v___x_317_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__11, &lp_mathlib_Equiv_sumSumSumComm___closed__11_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__11);
v___x_318_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__1, &lp_mathlib_Equiv_sumSumSumComm___closed__1_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__1);
v___f_319_ = ((lean_object*)(lp_mathlib_Equiv_sumSumSumComm___closed__0));
v___f_320_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__2, &lp_mathlib_Equiv_sumSumSumComm___closed__2_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__2);
v___x_321_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__3, &lp_mathlib_Equiv_sumSumSumComm___closed__3_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__3);
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__5), 7, 6);
lean_closure_set(v___f_322_, 0, v___x_321_);
lean_closure_set(v___f_322_, 1, v___f_320_);
lean_closure_set(v___f_322_, 2, v___f_319_);
lean_closure_set(v___f_322_, 3, v___x_318_);
lean_closure_set(v___f_322_, 4, v___x_317_);
lean_closure_set(v___f_322_, 5, v___f_316_);
return v___f_322_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__13(void){
_start:
{
lean_object* v___f_323_; lean_object* v___f_324_; lean_object* v___x_325_; 
v___f_323_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__12, &lp_mathlib_Equiv_sumSumSumComm___closed__12_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__12);
v___f_324_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__8, &lp_mathlib_Equiv_sumSumSumComm___closed__8_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__8);
v___x_325_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_325_, 0, v___f_324_);
lean_ctor_set(v___x_325_, 1, v___f_323_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm(lean_object* v_00_u03b1_326_, lean_object* v_00_u03b2_327_, lean_object* v_00_u03b3_328_, lean_object* v_00_u03b4_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__13, &lp_mathlib_Equiv_sumSumSumComm___closed__13_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__13);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0(lean_object* v_a_331_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0___boxed(lean_object* v_a_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_Equiv_sumEmpty___lam__0(v_a_332_);
lean_dec(v_a_332_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__1(lean_object* v_val_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_335_, 0, v_val_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty(lean_object* v_00_u03b1_345_, lean_object* v_00_u03b2_346_, lean_object* v_inst_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = ((lean_object*)(lp_mathlib_Equiv_sumEmpty___closed__4));
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib_Equiv_emptySum___closed__0(void){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lp_mathlib_Equiv_sumEmpty(lean_box(0), lean_box(0), lean_box(0));
return v___x_349_;
}
}
static lean_object* _init_lp_mathlib_Equiv_emptySum___closed__1(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_350_ = lean_obj_once(&lp_mathlib_Equiv_emptySum___closed__0, &lp_mathlib_Equiv_emptySum___closed__0_once, _init_lp_mathlib_Equiv_emptySum___closed__0);
v___x_351_ = lean_obj_once(&lp_mathlib_Equiv_sumSumSumComm___closed__5, &lp_mathlib_Equiv_sumSumSumComm___closed__5_once, _init_lp_mathlib_Equiv_sumSumSumComm___closed__5);
v___x_352_ = lp_mathlib_Equiv_trans___redArg(v___x_351_, v___x_350_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptySum(lean_object* v_00_u03b1_353_, lean_object* v_00_u03b2_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lean_obj_once(&lp_mathlib_Equiv_emptySum___closed__1, &lp_mathlib_Equiv_emptySum___closed__1_once, _init_lp_mathlib_Equiv_emptySum___closed__1);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__0(lean_object* v_x_357_){
_start:
{
uint8_t v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = 0;
v___x_359_ = lean_box(v___x_358_);
v___x_360_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set(v___x_360_, 1, v_x_357_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__1(lean_object* v_x_361_){
_start:
{
uint8_t v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_362_ = 1;
v___x_363_ = lean_box(v___x_362_);
v___x_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v_x_361_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__2(lean_object* v___f_365_, lean_object* v___f_366_, lean_object* v_s_367_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = l_Sum_elim___redArg(v___f_365_, v___f_366_, v_s_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3(lean_object* v_s_369_){
_start:
{
lean_object* v_fst_370_; uint8_t v___x_371_; 
v_fst_370_ = lean_ctor_get(v_s_369_, 0);
v___x_371_ = lean_unbox(v_fst_370_);
if (v___x_371_ == 0)
{
lean_object* v_snd_372_; lean_object* v___x_373_; 
v_snd_372_ = lean_ctor_get(v_s_369_, 1);
lean_inc(v_snd_372_);
v___x_373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_373_, 0, v_snd_372_);
return v___x_373_;
}
else
{
lean_object* v_snd_374_; lean_object* v___x_375_; 
v_snd_374_ = lean_ctor_get(v_s_369_, 1);
lean_inc(v_snd_374_);
v___x_375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_375_, 0, v_snd_374_);
return v___x_375_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3___boxed(lean_object* v_s_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_Equiv_sumEquivSigmaBool___lam__3(v_s_376_);
lean_dec_ref(v_s_376_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool(lean_object* v_00_u03b1_387_, lean_object* v_00_u03b2_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = ((lean_object*)(lp_mathlib_Equiv_sumEquivSigmaBool___closed__4));
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0(lean_object* v_x_390_){
_start:
{
lean_object* v_snd_391_; 
v_snd_391_ = lean_ctor_get(v_x_390_, 1);
lean_inc(v_snd_391_);
return v_snd_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0___boxed(lean_object* v_x_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0(v_x_392_);
lean_dec_ref(v_x_392_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__1(lean_object* v_f_394_, lean_object* v_x_395_){
_start:
{
lean_object* v___x_396_; lean_object* v___x_397_; 
lean_inc(v_x_395_);
v___x_396_ = lean_apply_1(v_f_394_, v_x_395_);
v___x_397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_397_, 0, v___x_396_);
lean_ctor_set(v___x_397_, 1, v_x_395_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object* v_f_399_){
_start:
{
lean_object* v___f_400_; lean_object* v___f_401_; lean_object* v___x_402_; 
v___f_400_ = ((lean_object*)(lp_mathlib_Equiv_sigmaFiberEquiv___redArg___closed__0));
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_401_, 0, v_f_399_);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v___f_400_);
lean_ctor_set(v___x_402_, 1, v___f_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b2_404_, lean_object* v_f_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__0(lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_a_409_){
_start:
{
lean_object* v___x_410_; uint8_t v___x_411_; 
lean_inc(v_a_409_);
v___x_410_ = lean_apply_2(v_inst_407_, v_a_409_, v_inst_408_);
v___x_411_ = lean_unbox(v___x_410_);
if (v___x_411_ == 0)
{
lean_object* v___x_412_; 
v___x_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_412_, 0, v_a_409_);
return v___x_412_;
}
else
{
lean_object* v___x_413_; 
lean_dec(v_a_409_);
v___x_413_ = lean_box(0);
return v___x_413_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1(lean_object* v_self_414_){
_start:
{
lean_inc(v_self_414_);
return v_self_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed(lean_object* v_self_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1(v_self_415_);
lean_dec(v_self_415_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg(lean_object* v_inst_418_, lean_object* v_inst_419_){
_start:
{
lean_object* v___f_420_; lean_object* v___f_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
lean_inc(v_inst_418_);
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__0), 3, 2);
lean_closure_set(v___f_420_, 0, v_inst_419_);
lean_closure_set(v___f_420_, 1, v_inst_418_);
v___f_421_ = ((lean_object*)(lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___closed__0));
v___x_422_ = lean_alloc_closure((void*)(lp_mathlib_Option_elim_x27___boxed), 5, 4);
lean_closure_set(v___x_422_, 0, lean_box(0));
lean_closure_set(v___x_422_, 1, lean_box(0));
lean_closure_set(v___x_422_, 2, v_inst_418_);
lean_closure_set(v___x_422_, 3, v___f_421_);
v___x_423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_423_, 0, v___f_420_);
lean_ctor_set(v___x_423_, 1, v___x_422_);
v___x_424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_424_, 0, lean_box(0));
lean_ctor_set(v___x_424_, 1, v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited(lean_object* v_00_u03b1_425_, lean_object* v_inst_426_, lean_object* v_inst_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg(v_inst_426_, v_inst_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg___lam__2(lean_object* v_inst_429_, lean_object* v_a_430_){
_start:
{
lean_object* v___x_431_; uint8_t v___x_432_; 
lean_inc(v_a_430_);
v___x_431_ = lean_apply_1(v_inst_429_, v_a_430_);
v___x_432_ = lean_unbox(v___x_431_);
if (v___x_432_ == 0)
{
lean_object* v___x_433_; 
v___x_433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_433_, 0, v_a_430_);
return v___x_433_;
}
else
{
lean_object* v___x_434_; 
v___x_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_434_, 0, v_a_430_);
return v___x_434_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg(lean_object* v_inst_437_){
_start:
{
lean_object* v___f_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v___f_438_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCompl___redArg___lam__2), 2, 1);
lean_closure_set(v___f_438_, 0, v_inst_437_);
v___x_439_ = ((lean_object*)(lp_mathlib_Equiv_sumCompl___redArg___closed__0));
v___x_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_440_, 0, v___x_439_);
lean_ctor_set(v___x_440_, 1, v___f_438_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl(lean_object* v_00_u03b1_441_, lean_object* v_p_442_, lean_object* v_inst_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_443_);
return v___x_444_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__0(void){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return v___x_445_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__1(void){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lp_mathlib_Equiv_sumProdDistrib(lean_box(0), lean_box(0), lean_box(0));
return v___x_446_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__2(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_447_ = lean_obj_once(&lp_mathlib_Equiv_prodSumDistrib___closed__1, &lp_mathlib_Equiv_prodSumDistrib___closed__1_once, _init_lp_mathlib_Equiv_prodSumDistrib___closed__1);
v___x_448_ = lean_obj_once(&lp_mathlib_Equiv_prodSumDistrib___closed__0, &lp_mathlib_Equiv_prodSumDistrib___closed__0_once, _init_lp_mathlib_Equiv_prodSumDistrib___closed__0);
v___x_449_ = lp_mathlib_Equiv_trans___redArg(v___x_448_, v___x_447_);
return v___x_449_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__3(void){
_start:
{
lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_450_ = lean_obj_once(&lp_mathlib_Equiv_prodSumDistrib___closed__0, &lp_mathlib_Equiv_prodSumDistrib___closed__0_once, _init_lp_mathlib_Equiv_prodSumDistrib___closed__0);
v___x_451_ = lp_mathlib_Equiv_sumCongr___redArg(v___x_450_, v___x_450_);
return v___x_451_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__4(void){
_start:
{
lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_452_ = lean_obj_once(&lp_mathlib_Equiv_prodSumDistrib___closed__3, &lp_mathlib_Equiv_prodSumDistrib___closed__3_once, _init_lp_mathlib_Equiv_prodSumDistrib___closed__3);
v___x_453_ = lean_obj_once(&lp_mathlib_Equiv_prodSumDistrib___closed__2, &lp_mathlib_Equiv_prodSumDistrib___closed__2_once, _init_lp_mathlib_Equiv_prodSumDistrib___closed__2);
v___x_454_ = lp_mathlib_Equiv_trans___redArg(v___x_453_, v___x_452_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodSumDistrib(lean_object* v_00_u03b1_455_, lean_object* v_00_u03b2_456_, lean_object* v_00_u03b3_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lean_obj_once(&lp_mathlib_Equiv_prodSumDistrib___closed__4, &lp_mathlib_Equiv_prodSumDistrib___closed__4_once, _init_lp_mathlib_Equiv_prodSumDistrib___closed__4);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__0(lean_object* v_fst_459_, lean_object* v_snd_460_){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_461_, 0, v_fst_459_);
lean_ctor_set(v___x_461_, 1, v_snd_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__2(lean_object* v_p_462_){
_start:
{
lean_object* v_fst_463_; lean_object* v_snd_464_; lean_object* v___f_465_; lean_object* v___x_466_; 
v_fst_463_ = lean_ctor_get(v_p_462_, 0);
lean_inc(v_fst_463_);
v_snd_464_ = lean_ctor_get(v_p_462_, 1);
lean_inc(v_snd_464_);
lean_dec_ref(v_p_462_);
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSumDistrib___lam__0), 2, 1);
lean_closure_set(v___f_465_, 0, v_fst_463_);
lean_inc_ref(v___f_465_);
v___x_466_ = l_Sum_map___redArg(v___f_465_, v___f_465_, v_snd_464_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1(lean_object* v_x_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_469_, 0, v___y_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1___boxed(lean_object* v_x_470_, lean_object* v___y_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_Equiv_sigmaSumDistrib___lam__1(v_x_470_, v___y_471_);
lean_dec(v_x_470_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3(lean_object* v_x_473_, lean_object* v___y_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_475_, 0, v___y_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3___boxed(lean_object* v_x_476_, lean_object* v___y_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_Equiv_sigmaSumDistrib___lam__3(v_x_476_, v___y_477_);
lean_dec(v_x_476_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib(lean_object* v_00_u03b9_494_, lean_object* v_00_u03b1_495_, lean_object* v_00_u03b2_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = ((lean_object*)(lp_mathlib_Equiv_sigmaSumDistrib___closed__6));
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__0(lean_object* v_x_498_){
_start:
{
lean_object* v_fst_499_; 
v_fst_499_ = lean_ctor_get(v_x_498_, 0);
lean_inc(v_fst_499_);
if (lean_obj_tag(v_fst_499_) == 0)
{
lean_object* v_snd_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_515_; 
v_snd_500_ = lean_ctor_get(v_x_498_, 1);
v_isSharedCheck_515_ = !lean_is_exclusive(v_x_498_);
if (v_isSharedCheck_515_ == 0)
{
lean_object* v_unused_516_; 
v_unused_516_ = lean_ctor_get(v_x_498_, 0);
lean_dec(v_unused_516_);
v___x_502_ = v_x_498_;
v_isShared_503_ = v_isSharedCheck_515_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_snd_500_);
lean_dec(v_x_498_);
v___x_502_ = lean_box(0);
v_isShared_503_ = v_isSharedCheck_515_;
goto v_resetjp_501_;
}
v_resetjp_501_:
{
lean_object* v_val_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_514_; 
v_val_504_ = lean_ctor_get(v_fst_499_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v_fst_499_);
if (v_isSharedCheck_514_ == 0)
{
v___x_506_ = v_fst_499_;
v_isShared_507_ = v_isSharedCheck_514_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_val_504_);
lean_dec(v_fst_499_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_514_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_509_; 
if (v_isShared_503_ == 0)
{
lean_ctor_set(v___x_502_, 0, v_val_504_);
v___x_509_ = v___x_502_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_val_504_);
lean_ctor_set(v_reuseFailAlloc_513_, 1, v_snd_500_);
v___x_509_ = v_reuseFailAlloc_513_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
lean_object* v___x_511_; 
if (v_isShared_507_ == 0)
{
lean_ctor_set(v___x_506_, 0, v___x_509_);
v___x_511_ = v___x_506_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_512_; 
v_reuseFailAlloc_512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_512_, 0, v___x_509_);
v___x_511_ = v_reuseFailAlloc_512_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
return v___x_511_;
}
}
}
}
}
else
{
lean_object* v_snd_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_532_; 
v_snd_517_ = lean_ctor_get(v_x_498_, 1);
v_isSharedCheck_532_ = !lean_is_exclusive(v_x_498_);
if (v_isSharedCheck_532_ == 0)
{
lean_object* v_unused_533_; 
v_unused_533_ = lean_ctor_get(v_x_498_, 0);
lean_dec(v_unused_533_);
v___x_519_ = v_x_498_;
v_isShared_520_ = v_isSharedCheck_532_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_snd_517_);
lean_dec(v_x_498_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_532_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v_val_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_531_; 
v_val_521_ = lean_ctor_get(v_fst_499_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v_fst_499_);
if (v_isSharedCheck_531_ == 0)
{
v___x_523_ = v_fst_499_;
v_isShared_524_ = v_isSharedCheck_531_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_val_521_);
lean_dec(v_fst_499_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_531_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___x_526_; 
if (v_isShared_520_ == 0)
{
lean_ctor_set(v___x_519_, 0, v_val_521_);
v___x_526_ = v___x_519_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_val_521_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v_snd_517_);
v___x_526_ = v_reuseFailAlloc_530_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
lean_object* v___x_528_; 
if (v_isShared_524_ == 0)
{
lean_ctor_set(v___x_523_, 0, v___x_526_);
v___x_528_ = v___x_523_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v___x_526_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__1(lean_object* v_a_534_){
_start:
{
lean_object* v_fst_535_; lean_object* v_snd_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_544_; 
v_fst_535_ = lean_ctor_get(v_a_534_, 0);
v_snd_536_ = lean_ctor_get(v_a_534_, 1);
v_isSharedCheck_544_ = !lean_is_exclusive(v_a_534_);
if (v_isSharedCheck_544_ == 0)
{
v___x_538_ = v_a_534_;
v_isShared_539_ = v_isSharedCheck_544_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_snd_536_);
lean_inc(v_fst_535_);
lean_dec(v_a_534_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_544_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_540_; lean_object* v___x_542_; 
v___x_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_540_, 0, v_fst_535_);
if (v_isShared_539_ == 0)
{
lean_ctor_set(v___x_538_, 0, v___x_540_);
v___x_542_ = v___x_538_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v___x_540_);
lean_ctor_set(v_reuseFailAlloc_543_, 1, v_snd_536_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__2(lean_object* v_b_545_){
_start:
{
lean_object* v_fst_546_; lean_object* v_snd_547_; lean_object* v___x_549_; uint8_t v_isShared_550_; uint8_t v_isSharedCheck_555_; 
v_fst_546_ = lean_ctor_get(v_b_545_, 0);
v_snd_547_ = lean_ctor_get(v_b_545_, 1);
v_isSharedCheck_555_ = !lean_is_exclusive(v_b_545_);
if (v_isSharedCheck_555_ == 0)
{
v___x_549_ = v_b_545_;
v_isShared_550_ = v_isSharedCheck_555_;
goto v_resetjp_548_;
}
else
{
lean_inc(v_snd_547_);
lean_inc(v_fst_546_);
lean_dec(v_b_545_);
v___x_549_ = lean_box(0);
v_isShared_550_ = v_isSharedCheck_555_;
goto v_resetjp_548_;
}
v_resetjp_548_:
{
lean_object* v___x_551_; lean_object* v___x_553_; 
v___x_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_551_, 0, v_fst_546_);
if (v_isShared_550_ == 0)
{
lean_ctor_set(v___x_549_, 0, v___x_551_);
v___x_553_ = v___x_549_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___x_551_);
lean_ctor_set(v_reuseFailAlloc_554_, 1, v_snd_547_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib(lean_object* v_00_u03b1_565_, lean_object* v_00_u03b2_566_, lean_object* v_t_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = ((lean_object*)(lp_mathlib_Equiv_sumSigmaDistrib___closed__4));
return v___x_568_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Sum(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Sum(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Sum(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Sum(builtin);
}
#ifdef __cplusplus
}
#endif
