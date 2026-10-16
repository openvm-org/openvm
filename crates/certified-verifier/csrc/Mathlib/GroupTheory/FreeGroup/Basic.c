// Lean compiler output
// Module: Mathlib.GroupTheory.FreeGroup.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Algebra.Group.Subgroup.Ker public import Mathlib.Data.List.Chain public import Mathlib.Algebra.Group.Int.Defs public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Algebra.BigOperators.Group.List.Defs
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
extern lean_object* lp_mathlib_Int_instAddCommGroup;
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_Quot_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* l_List_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instOne(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instZero(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instUniqueOfIsEmpty(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_instMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_appendTR___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMul___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_instMul___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMul(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAdd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_invRev___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_invRev(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_negRev___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_negRev(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_instInv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_invRev, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeGroup_instInv___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_instInv___closed__0_value;
static const lean_closure_object lp_mathlib_FreeGroup_instInv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quot_map, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FreeGroup_instInv___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeGroup_instInv___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_instInv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instInv(lean_object*);
static const lean_closure_object lp_mathlib_FreeAddGroup_instNeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_negRev, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeAddGroup_instNeg___closed__0 = (const lean_object*)&lp_mathlib_FreeAddGroup_instNeg___closed__0_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instNeg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quot_map, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FreeAddGroup_instNeg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeAddGroup_instNeg___closed__1 = (const lean_object*)&lp_mathlib_FreeAddGroup_instNeg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instNeg(lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_instGroup___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowBinRecAuto___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FreeGroup_instMul___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeGroup_instGroup___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_instGroup___closed__0_value;
static const lean_ctor_object lp_mathlib_FreeGroup_instGroup___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FreeGroup_instMul___closed__0_value),((lean_object*)&lp_mathlib_FreeGroup_instGroup___closed__0_value)}};
static const lean_object* lp_mathlib_FreeGroup_instGroup___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_instGroup___closed__1_value;
static lean_once_cell_t lp_mathlib_FreeGroup_instGroup___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instGroup___closed__2;
static lean_once_cell_t lp_mathlib_FreeGroup_instGroup___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instGroup___closed__3;
static const lean_closure_object lp_mathlib_FreeGroup_instGroup___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_npowRec___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FreeGroup_instMul___closed__0_value)} };
static const lean_object* lp_mathlib_FreeGroup_instGroup___closed__4 = (const lean_object*)&lp_mathlib_FreeGroup_instGroup___closed__4_value;
static lean_once_cell_t lp_mathlib_FreeGroup_instGroup___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instGroup___closed__5;
static lean_once_cell_t lp_mathlib_FreeGroup_instGroup___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instGroup___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instGroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___at___00FreeAddGroup_instAddGroup_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___at___00FreeAddGroup_instAddGroup_spec__1(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAddGroup___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAddGroup___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0___redArg(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAddGroup_instAddGroup___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__0 = (const lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__0_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instAddGroup___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubNegMonoid_sub_x27___at___00FreeAddGroup_instAddGroup_spec__1___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__1 = (const lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__1_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instAddGroup___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__2 = (const lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__2_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instAddGroup___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instAddGroup___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__2_value)} };
static const lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__3 = (const lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__3_value;
static const lean_ctor_object lp_mathlib_FreeAddGroup_instAddGroup___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FreeGroup_instMul___closed__0_value),((lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__0_value)}};
static const lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__4 = (const lean_object*)&lp_mathlib_FreeAddGroup_instAddGroup___closed__4_value;
static lean_once_cell_t lp_mathlib_FreeAddGroup_instAddGroup___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__5;
static lean_once_cell_t lp_mathlib_FreeAddGroup_instAddGroup___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddGroup_instAddGroup___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAddGroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_of(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_of(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_lift___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_lift___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_lift___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_lift___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAddGroup_lift___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_lift___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_lift___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeAddGroup_lift___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_map_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_map___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_map(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_map_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_map(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_freeAddGroupCongr(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_prod___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeGroup_prod___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_prod___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_sum___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_sum(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_sum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_sum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1(lean_object*);
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__0_value;
static const lean_closure_object lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__1_value;
static const lean_ctor_object lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__1_value),((lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__0_value)}};
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__2 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___lam__0___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__0;
static const lean_closure_object lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__0_value;
static const lean_closure_object lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__0_value)} };
static const lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__1_value;
static const lean_closure_object lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__2 = (const lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__2_value;
static const lean_closure_object lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__2, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__2_value)} };
static const lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__3 = (const lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__3_value;
static const lean_ctor_object lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__3_value),((lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__1_value)}};
static const lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__4 = (const lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt = (const lean_object*)&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__2(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__0;
static lean_once_cell_t lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__2(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeGroup_instMonad___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instMonad___lam__3___closed__0;
static lean_once_cell_t lp_mathlib_FreeGroup_instMonad___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instMonad___lam__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeGroup_instMonad___lam__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeGroup_instMonad___lam__4___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__9(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__1_value;
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__2 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__2_value;
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_instMonad___lam__4, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__3 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_instMonad___lam__7, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__3_value)} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__4 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__4_value;
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_instMonad___lam__9, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__5 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__5_value;
static const lean_ctor_object lp_mathlib_FreeGroup_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__0_value),((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__1_value)}};
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__6 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__6_value;
static const lean_closure_object lp_mathlib_FreeGroup_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_of, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__7 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__7_value;
static const lean_ctor_object lp_mathlib_FreeGroup_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__6_value),((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__7_value),((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__2_value),((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__4_value),((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__5_value)}};
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__8 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_FreeGroup_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__8_value),((lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__3_value)}};
static const lean_object* lp_mathlib_FreeGroup_instMonad___closed__9 = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_FreeGroup_instMonad = (const lean_object*)&lp_mathlib_FreeGroup_instMonad___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0___redArg(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__0 = (const lean_object*)&lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__0_value),((lean_object*)&lp_mathlib_FreeAddGroup_lift___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__1 = (const lean_object*)&lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAddGroup_instMonad___lam__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__4___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__0 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instMonad___lam__2, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__1 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__1_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instMonad___lam__4, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__2 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__2_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instMonad___lam__5, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__3 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instMonad___lam__8, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__3_value)} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__4 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__4_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_instMonad___lam__10, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__5 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__5_value;
static const lean_ctor_object lp_mathlib_FreeAddGroup_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__0_value),((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__1_value)}};
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__6 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__6_value;
static const lean_closure_object lp_mathlib_FreeAddGroup_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddGroup_of, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__7 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__7_value;
static const lean_ctor_object lp_mathlib_FreeAddGroup_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__6_value),((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__7_value),((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__2_value),((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__4_value),((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__5_value)}};
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__8 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_FreeAddGroup_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__8_value),((lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__3_value)}};
static const lean_object* lp_mathlib_FreeAddGroup_instMonad___closed__9 = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_FreeAddGroup_instMonad = (const lean_object*)&lp_mathlib_FreeAddGroup_instMonad___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk___redArg(lean_object* v_L_1_){
_start:
{
lean_inc(v_L_1_);
return v_L_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk___redArg___boxed(lean_object* v_L_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_FreeGroup_mk___redArg(v_L_2_);
lean_dec(v_L_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk(lean_object* v_00_u03b1_4_, lean_object* v_L_5_){
_start:
{
lean_inc(v_L_5_);
return v_L_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mk___boxed(lean_object* v_00_u03b1_6_, lean_object* v_L_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_FreeGroup_mk(v_00_u03b1_6_, v_L_7_);
lean_dec(v_L_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk___redArg(lean_object* v_L_9_){
_start:
{
lean_inc(v_L_9_);
return v_L_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk___redArg___boxed(lean_object* v_L_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_FreeAddGroup_mk___redArg(v_L_10_);
lean_dec(v_L_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk(lean_object* v_00_u03b1_12_, lean_object* v_L_13_){
_start:
{
lean_inc(v_L_13_);
return v_L_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_mk___boxed(lean_object* v_00_u03b1_14_, lean_object* v_L_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_FreeAddGroup_mk(v_00_u03b1_14_, v_L_15_);
lean_dec(v_L_15_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instOne(lean_object* v_00_u03b1_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instZero(lean_object* v_00_u03b1_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_box(0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instInhabited(lean_object* v_00_u03b1_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_box(0);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instInhabited(lean_object* v_00_u03b1_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_box(0);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instUniqueOfIsEmpty(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_box(0);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instUniqueOfIsEmpty(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_box(0);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMul(lean_object* v_00_u03b1_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = ((lean_object*)(lp_mathlib_FreeGroup_instMul___closed__0));
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAdd(lean_object* v_00_u03b1_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = ((lean_object*)(lp_mathlib_FreeGroup_instMul___closed__0));
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0___redArg(lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
if (lean_obj_tag(v_a_36_) == 0)
{
lean_object* v___x_38_; 
v___x_38_ = l_List_reverse___redArg(v_a_37_);
return v___x_38_;
}
else
{
lean_object* v_head_39_; lean_object* v_tail_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_74_; 
v_head_39_ = lean_ctor_get(v_a_36_, 0);
v_tail_40_ = lean_ctor_get(v_a_36_, 1);
v_isSharedCheck_74_ = !lean_is_exclusive(v_a_36_);
if (v_isSharedCheck_74_ == 0)
{
v___x_42_ = v_a_36_;
v_isShared_43_ = v_isSharedCheck_74_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_tail_40_);
lean_inc(v_head_39_);
lean_dec(v_a_36_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_74_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v___y_45_; lean_object* v_snd_50_; uint8_t v___x_51_; 
v_snd_50_ = lean_ctor_get(v_head_39_, 1);
v___x_51_ = lean_unbox(v_snd_50_);
if (v___x_51_ == 0)
{
lean_object* v_fst_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_61_; 
v_fst_52_ = lean_ctor_get(v_head_39_, 0);
v_isSharedCheck_61_ = !lean_is_exclusive(v_head_39_);
if (v_isSharedCheck_61_ == 0)
{
lean_object* v_unused_62_; 
v_unused_62_ = lean_ctor_get(v_head_39_, 1);
lean_dec(v_unused_62_);
v___x_54_ = v_head_39_;
v_isShared_55_ = v_isSharedCheck_61_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_fst_52_);
lean_dec(v_head_39_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_61_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
uint8_t v___x_56_; lean_object* v___x_57_; lean_object* v___x_59_; 
v___x_56_ = 1;
v___x_57_ = lean_box(v___x_56_);
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 1, v___x_57_);
v___x_59_ = v___x_54_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v_fst_52_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v___x_57_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
v___y_45_ = v___x_59_;
goto v___jp_44_;
}
}
}
else
{
lean_object* v_fst_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_72_; 
v_fst_63_ = lean_ctor_get(v_head_39_, 0);
v_isSharedCheck_72_ = !lean_is_exclusive(v_head_39_);
if (v_isSharedCheck_72_ == 0)
{
lean_object* v_unused_73_; 
v_unused_73_ = lean_ctor_get(v_head_39_, 1);
lean_dec(v_unused_73_);
v___x_65_ = v_head_39_;
v_isShared_66_ = v_isSharedCheck_72_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_fst_63_);
lean_dec(v_head_39_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_72_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_70_; 
v___x_67_ = 0;
v___x_68_ = lean_box(v___x_67_);
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 1, v___x_68_);
v___x_70_ = v___x_65_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_71_; 
v_reuseFailAlloc_71_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_71_, 0, v_fst_63_);
lean_ctor_set(v_reuseFailAlloc_71_, 1, v___x_68_);
v___x_70_ = v_reuseFailAlloc_71_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
v___y_45_ = v___x_70_;
goto v___jp_44_;
}
}
}
v___jp_44_:
{
lean_object* v___x_47_; 
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 1, v_a_37_);
lean_ctor_set(v___x_42_, 0, v___y_45_);
v___x_47_ = v___x_42_;
goto v_reusejp_46_;
}
else
{
lean_object* v_reuseFailAlloc_49_; 
v_reuseFailAlloc_49_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_49_, 0, v___y_45_);
lean_ctor_set(v_reuseFailAlloc_49_, 1, v_a_37_);
v___x_47_ = v_reuseFailAlloc_49_;
goto v_reusejp_46_;
}
v_reusejp_46_:
{
v_a_36_ = v_tail_40_;
v_a_37_ = v___x_47_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_invRev___redArg(lean_object* v_w_75_){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_76_ = lean_box(0);
v___x_77_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0___redArg(v_w_75_, v___x_76_);
v___x_78_ = l_List_reverse___redArg(v___x_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_invRev(lean_object* v_00_u03b1_79_, lean_object* v_w_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_FreeGroup_invRev___redArg(v_w_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0(lean_object* v_00_u03b1_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0___redArg(v_a_83_, v_a_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_negRev___redArg(lean_object* v_w_86_){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_87_ = lean_box(0);
v___x_88_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_invRev_spec__0___redArg(v_w_86_, v___x_87_);
v___x_89_ = l_List_reverse___redArg(v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_negRev(lean_object* v_00_u03b1_90_, lean_object* v_w_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_FreeAddGroup_negRev___redArg(v_w_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instInv(lean_object* v_00_u03b1_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = ((lean_object*)(lp_mathlib_FreeGroup_instInv___closed__1));
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instNeg(lean_object* v_00_u03b1_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = ((lean_object*)(lp_mathlib_FreeAddGroup_instNeg___closed__1));
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instGroup___closed__2(void){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_FreeGroup_instInv(lean_box(0));
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instGroup___closed__3(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_111_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__2, &lp_mathlib_FreeGroup_instGroup___closed__2_once, _init_lp_mathlib_FreeGroup_instGroup___closed__2);
v___x_112_ = ((lean_object*)(lp_mathlib_FreeGroup_instGroup___closed__1));
v___x_113_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_113_, 0, lean_box(0));
lean_closure_set(v___x_113_, 1, v___x_112_);
lean_closure_set(v___x_113_, 2, v___x_111_);
return v___x_113_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instGroup___closed__5(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___f_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_117_ = ((lean_object*)(lp_mathlib_FreeGroup_instGroup___closed__4));
v___x_118_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__2, &lp_mathlib_FreeGroup_instGroup___closed__2_once, _init_lp_mathlib_FreeGroup_instGroup___closed__2);
v___f_119_ = ((lean_object*)(lp_mathlib_FreeGroup_instMul___closed__0));
v___x_120_ = lean_box(0);
v___x_121_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_121_, 0, lean_box(0));
lean_closure_set(v___x_121_, 1, v___x_120_);
lean_closure_set(v___x_121_, 2, v___f_119_);
lean_closure_set(v___x_121_, 3, v___x_118_);
lean_closure_set(v___x_121_, 4, v___x_117_);
return v___x_121_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instGroup___closed__6(void){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_122_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__5, &lp_mathlib_FreeGroup_instGroup___closed__5_once, _init_lp_mathlib_FreeGroup_instGroup___closed__5);
v___x_123_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__3, &lp_mathlib_FreeGroup_instGroup___closed__3_once, _init_lp_mathlib_FreeGroup_instGroup___closed__3);
v___x_124_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__2, &lp_mathlib_FreeGroup_instGroup___closed__2_once, _init_lp_mathlib_FreeGroup_instGroup___closed__2);
v___x_125_ = ((lean_object*)(lp_mathlib_FreeGroup_instGroup___closed__1));
v___x_126_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_126_, 0, v___x_125_);
lean_ctor_set(v___x_126_, 1, v___x_124_);
lean_ctor_set(v___x_126_, 2, v___x_123_);
lean_ctor_set(v___x_126_, 3, v___x_122_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instGroup(lean_object* v_00_u03b1_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__6, &lp_mathlib_FreeGroup_instGroup___closed__6_once, _init_lp_mathlib_FreeGroup_instGroup___closed__6);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___at___00FreeAddGroup_instAddGroup_spec__1___redArg(lean_object* v_a_129_, lean_object* v_b_130_){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_131_ = lp_mathlib_FreeAddGroup_negRev___redArg(v_b_130_);
v___x_132_ = l_List_appendTR___redArg(v_a_129_, v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubNegMonoid_sub_x27___at___00FreeAddGroup_instAddGroup_spec__1(lean_object* v_00_u03b1_133_, lean_object* v_a_134_, lean_object* v_b_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_SubNegMonoid_sub_x27___at___00FreeAddGroup_instAddGroup_spec__1___redArg(v_a_134_, v_b_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v_natZero_137_; lean_object* v_intZero_138_; 
v_natZero_137_ = lean_unsigned_to_nat(0u);
v_intZero_138_ = lean_nat_to_int(v_natZero_137_);
return v_intZero_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg(lean_object* v_nsmul_139_, lean_object* v_x_140_, lean_object* v_x_141_){
_start:
{
lean_object* v_intZero_142_; uint8_t v_isNeg_143_; 
v_intZero_142_ = lean_obj_once(&lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0, &lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0_once, _init_lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0);
v_isNeg_143_ = lean_int_dec_lt(v_x_140_, v_intZero_142_);
if (v_isNeg_143_ == 0)
{
lean_object* v_a_144_; lean_object* v___x_145_; 
v_a_144_ = lean_nat_abs(v_x_140_);
v___x_145_ = lean_apply_2(v_nsmul_139_, v_a_144_, v_x_141_);
return v___x_145_;
}
else
{
lean_object* v_abs_146_; lean_object* v_one_147_; lean_object* v_a_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_abs_146_ = lean_nat_abs(v_x_140_);
v_one_147_ = lean_unsigned_to_nat(1u);
v_a_148_ = lean_nat_sub(v_abs_146_, v_one_147_);
lean_dec(v_abs_146_);
v___x_149_ = lean_nat_add(v_a_148_, v_one_147_);
lean_dec(v_a_148_);
v___x_150_ = lean_apply_2(v_nsmul_139_, v___x_149_, v_x_141_);
v___x_151_ = lp_mathlib_FreeAddGroup_negRev___redArg(v___x_150_);
return v___x_151_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___boxed(lean_object* v_nsmul_152_, lean_object* v_x_153_, lean_object* v_x_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg(v_nsmul_152_, v_x_153_, v_x_154_);
lean_dec(v_x_153_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3(lean_object* v_00_u03b1_156_, lean_object* v_nsmul_157_, lean_object* v_x_158_, lean_object* v_x_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg(v_nsmul_157_, v_x_158_, v_x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___boxed(lean_object* v_00_u03b1_161_, lean_object* v_nsmul_162_, lean_object* v_x_163_, lean_object* v_x_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3(v_00_u03b1_161_, v_nsmul_162_, v_x_163_, v_x_164_);
lean_dec(v_x_163_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAddGroup___lam__0(lean_object* v___f_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg(v___f_166_, v___y_167_, v___y_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAddGroup___lam__0___boxed(lean_object* v___f_170_, lean_object* v___y_171_, lean_object* v___y_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_FreeAddGroup_instAddGroup___lam__0(v___f_170_, v___y_171_, v___y_172_);
lean_dec(v___y_171_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg(lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
lean_object* v_zero_176_; uint8_t v_isZero_177_; 
v_zero_176_ = lean_unsigned_to_nat(0u);
v_isZero_177_ = lean_nat_dec_eq(v_x_174_, v_zero_176_);
if (v_isZero_177_ == 1)
{
lean_object* v___x_178_; 
lean_dec(v_x_175_);
v___x_178_ = lean_box(0);
return v___x_178_;
}
else
{
lean_object* v_one_179_; lean_object* v_n_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v_one_179_ = lean_unsigned_to_nat(1u);
v_n_180_ = lean_nat_sub(v_x_174_, v_one_179_);
lean_inc(v_x_175_);
v___x_181_ = lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg(v_n_180_, v_x_175_);
lean_dec(v_n_180_);
v___x_182_ = l_List_appendTR___redArg(v___x_181_, v_x_175_);
return v___x_182_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg___boxed(lean_object* v_x_183_, lean_object* v_x_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg(v_x_183_, v_x_184_);
lean_dec(v_x_183_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___lam__0(lean_object* v_y_186_, lean_object* v_x_187_){
_start:
{
lean_inc(v_y_186_);
return v_y_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___lam__0___boxed(lean_object* v_y_188_, lean_object* v_x_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___lam__0(v_y_188_, v_x_189_);
lean_dec(v_x_189_);
lean_dec(v_y_188_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3_spec__5___redArg(lean_object* v_zero_191_, lean_object* v_n_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___x_200_; uint8_t v___x_201_; 
v___x_200_ = lean_unsigned_to_nat(0u);
v___x_201_ = lean_nat_dec_eq(v_n_192_, v___x_200_);
if (v___x_201_ == 0)
{
lean_object* v___x_202_; lean_object* v___x_206_; uint8_t v___x_207_; 
v___x_202_ = lean_unsigned_to_nat(1u);
v___x_206_ = lean_nat_land(v___x_202_, v_n_192_);
v___x_207_ = lean_nat_dec_eq(v___x_206_, v___x_200_);
lean_dec(v___x_206_);
if (v___x_207_ == 0)
{
goto v___jp_203_;
}
else
{
if (v___x_201_ == 0)
{
lean_object* v___x_208_; 
v___x_208_ = lean_nat_shiftr(v_n_192_, v___x_202_);
lean_dec(v_n_192_);
v___y_196_ = v___x_208_;
v___y_197_ = v___y_193_;
goto v___jp_195_;
}
else
{
goto v___jp_203_;
}
}
v___jp_203_:
{
lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_204_ = lean_nat_shiftr(v_n_192_, v___x_202_);
lean_dec(v_n_192_);
lean_inc(v___y_194_);
v___x_205_ = l_List_appendTR___redArg(v___y_193_, v___y_194_);
v___y_196_ = v___x_204_;
v___y_197_ = v___x_205_;
goto v___jp_195_;
}
}
else
{
lean_object* v___x_209_; 
lean_dec(v_n_192_);
v___x_209_ = lean_apply_2(v_zero_191_, v___y_193_, v___y_194_);
return v___x_209_;
}
v___jp_195_:
{
lean_object* v___x_198_; 
lean_inc(v___y_194_);
v___x_198_ = l_List_appendTR___redArg(v___y_194_, v___y_194_);
v_n_192_ = v___y_196_;
v___y_193_ = v___y_197_;
v___y_194_ = v___x_198_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg(lean_object* v_k_211_, lean_object* v_a_212_, lean_object* v_a_213_){
_start:
{
lean_object* v___f_214_; lean_object* v___x_215_; 
v___f_214_ = ((lean_object*)(lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg___closed__0));
v___x_215_ = lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3_spec__5___redArg(v___f_214_, v_k_211_, v_a_212_, v_a_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0___redArg(lean_object* v_k_216_, lean_object* v_a_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = lean_box(0);
v___x_219_ = lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg(v_k_216_, v___x_218_, v_a_217_);
return v___x_219_;
}
}
static lean_object* _init_lp_mathlib_FreeAddGroup_instAddGroup___closed__5(void){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_FreeAddGroup_instNeg(lean_box(0));
return v___x_229_;
}
}
static lean_object* _init_lp_mathlib_FreeAddGroup_instAddGroup___closed__6(void){
_start:
{
lean_object* v___f_230_; lean_object* v___f_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___f_230_ = ((lean_object*)(lp_mathlib_FreeAddGroup_instAddGroup___closed__3));
v___f_231_ = ((lean_object*)(lp_mathlib_FreeAddGroup_instAddGroup___closed__1));
v___x_232_ = lean_obj_once(&lp_mathlib_FreeAddGroup_instAddGroup___closed__5, &lp_mathlib_FreeAddGroup_instAddGroup___closed__5_once, _init_lp_mathlib_FreeAddGroup_instAddGroup___closed__5);
v___x_233_ = ((lean_object*)(lp_mathlib_FreeAddGroup_instAddGroup___closed__4));
v___x_234_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
lean_ctor_set(v___x_234_, 2, v___f_231_);
lean_ctor_set(v___x_234_, 3, v___f_230_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instAddGroup(lean_object* v_00_u03b1_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lean_obj_once(&lp_mathlib_FreeAddGroup_instAddGroup___closed__6, &lp_mathlib_FreeAddGroup_instAddGroup___closed__6_once, _init_lp_mathlib_FreeAddGroup_instAddGroup___closed__6);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0___redArg(lean_object* v_k_237_, lean_object* v_m_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0___redArg(v_k_237_, v_m_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0(lean_object* v_00_u03b1_240_, lean_object* v_k_241_, lean_object* v_m_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0___redArg(v_k_241_, v_m_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2(lean_object* v_00_u03b1_244_, lean_object* v_x_245_, lean_object* v_x_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___redArg(v_x_245_, v_x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2___boxed(lean_object* v_00_u03b1_248_, lean_object* v_x_249_, lean_object* v_x_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_nsmulRec___at___00FreeAddGroup_instAddGroup_spec__2(v_00_u03b1_248_, v_x_249_, v_x_250_);
lean_dec(v_x_249_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0(lean_object* v_00_u03b1_252_, lean_object* v_k_253_, lean_object* v_a_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0___redArg(v_k_253_, v_a_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3(lean_object* v_00_u03b1_256_, lean_object* v_k_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3___redArg(v_k_257_, v_a_258_, v_a_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3_spec__5(lean_object* v_00_u03b1_261_, lean_object* v_zero_262_, lean_object* v_n_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddGroup_instAddGroup_spec__0_spec__0_spec__3_spec__5___redArg(v_zero_262_, v_n_263_, v___y_264_, v___y_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_of___redArg(lean_object* v_x_267_){
_start:
{
uint8_t v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_268_ = 1;
v___x_269_ = lean_box(v___x_268_);
v___x_270_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_270_, 0, v_x_267_);
lean_ctor_set(v___x_270_, 1, v___x_269_);
v___x_271_ = lean_box(0);
v___x_272_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_270_);
lean_ctor_set(v___x_272_, 1, v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_of(lean_object* v_00_u03b1_273_, lean_object* v_x_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_FreeGroup_of___redArg(v_x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_of___redArg(lean_object* v_x_276_){
_start:
{
uint8_t v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_277_ = 1;
v___x_278_ = lean_box(v___x_277_);
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v_x_276_);
lean_ctor_set(v___x_279_, 1, v___x_278_);
v___x_280_ = lean_box(0);
v___x_281_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_279_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_of(lean_object* v_00_u03b1_282_, lean_object* v_x_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_FreeAddGroup_of___redArg(v_x_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___redArg___lam__0(lean_object* v_f_285_, lean_object* v_toInv_286_, lean_object* v_x_287_){
_start:
{
lean_object* v_snd_288_; uint8_t v___x_289_; 
v_snd_288_ = lean_ctor_get(v_x_287_, 1);
v___x_289_ = lean_unbox(v_snd_288_);
if (v___x_289_ == 0)
{
lean_object* v_fst_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v_fst_290_ = lean_ctor_get(v_x_287_, 0);
lean_inc(v_fst_290_);
lean_dec_ref(v_x_287_);
v___x_291_ = lean_apply_1(v_f_285_, v_fst_290_);
v___x_292_ = lean_apply_1(v_toInv_286_, v___x_291_);
return v___x_292_;
}
else
{
lean_object* v_fst_293_; lean_object* v___x_294_; 
lean_dec(v_toInv_286_);
v_fst_293_ = lean_ctor_get(v_x_287_, 0);
lean_inc(v_fst_293_);
lean_dec_ref(v_x_287_);
v___x_294_ = lean_apply_1(v_f_285_, v_fst_293_);
return v___x_294_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___redArg(lean_object* v_inst_295_, lean_object* v_f_296_, lean_object* v_L_297_){
_start:
{
lean_object* v_toMonoid_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v_toMul_301_; lean_object* v___x_302_; lean_object* v_toOne_303_; lean_object* v_toInv_304_; lean_object* v___f_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; 
v_toMonoid_298_ = lean_ctor_get(v_inst_295_, 0);
v___x_299_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_298_);
v___x_300_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_299_);
v_toMul_301_ = lean_ctor_get(v___x_300_, 1);
lean_inc(v_toMul_301_);
lean_dec_ref(v___x_300_);
v___x_302_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_295_);
v_toOne_303_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_toOne_303_);
v_toInv_304_ = lean_ctor_get(v___x_302_, 1);
lean_inc(v_toInv_304_);
lean_dec_ref(v___x_302_);
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_Lift_aux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_305_, 0, v_f_296_);
lean_closure_set(v___f_305_, 1, v_toInv_304_);
v___x_306_ = lean_box(0);
v___x_307_ = l_List_mapTR_loop___redArg(v___f_305_, v_L_297_, v___x_306_);
v___x_308_ = l_List_prod___redArg(v_toMul_301_, v_toOne_303_, v___x_307_);
lean_dec(v_toOne_303_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___redArg___boxed(lean_object* v_inst_309_, lean_object* v_f_310_, lean_object* v_L_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_FreeGroup_Lift_aux___redArg(v_inst_309_, v_f_310_, v_L_311_);
lean_dec_ref(v_inst_309_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux(lean_object* v_00_u03b1_313_, lean_object* v_00_u03b2_314_, lean_object* v_inst_315_, lean_object* v_f_316_, lean_object* v_L_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_FreeGroup_Lift_aux___redArg(v_inst_315_, v_f_316_, v_L_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___boxed(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_inst_321_, lean_object* v_f_322_, lean_object* v_L_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_FreeGroup_Lift_aux(v_00_u03b1_319_, v_00_u03b2_320_, v_inst_321_, v_f_322_, v_L_323_);
lean_dec_ref(v_inst_321_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___redArg___lam__0(lean_object* v_f_325_, lean_object* v_toNeg_326_, lean_object* v_x_327_){
_start:
{
lean_object* v_snd_328_; uint8_t v___x_329_; 
v_snd_328_ = lean_ctor_get(v_x_327_, 1);
v___x_329_ = lean_unbox(v_snd_328_);
if (v___x_329_ == 0)
{
lean_object* v_fst_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v_fst_330_ = lean_ctor_get(v_x_327_, 0);
lean_inc(v_fst_330_);
lean_dec_ref(v_x_327_);
v___x_331_ = lean_apply_1(v_f_325_, v_fst_330_);
v___x_332_ = lean_apply_1(v_toNeg_326_, v___x_331_);
return v___x_332_;
}
else
{
lean_object* v_fst_333_; lean_object* v___x_334_; 
lean_dec(v_toNeg_326_);
v_fst_333_ = lean_ctor_get(v_x_327_, 0);
lean_inc(v_fst_333_);
lean_dec_ref(v_x_327_);
v___x_334_ = lean_apply_1(v_f_325_, v_fst_333_);
return v___x_334_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___redArg(lean_object* v_inst_335_, lean_object* v_f_336_, lean_object* v_L_337_){
_start:
{
lean_object* v_toAddMonoid_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v_toAdd_341_; lean_object* v___x_342_; lean_object* v_toZero_343_; lean_object* v_toNeg_344_; lean_object* v___f_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v_toAddMonoid_338_ = lean_ctor_get(v_inst_335_, 0);
v___x_339_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_338_);
v___x_340_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_339_);
v_toAdd_341_ = lean_ctor_get(v___x_340_, 1);
lean_inc(v_toAdd_341_);
lean_dec_ref(v___x_340_);
v___x_342_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_335_);
v_toZero_343_ = lean_ctor_get(v___x_342_, 0);
lean_inc(v_toZero_343_);
v_toNeg_344_ = lean_ctor_get(v___x_342_, 1);
lean_inc(v_toNeg_344_);
lean_dec_ref(v___x_342_);
v___f_345_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_Lift_aux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_345_, 0, v_f_336_);
lean_closure_set(v___f_345_, 1, v_toNeg_344_);
v___x_346_ = lean_box(0);
v___x_347_ = l_List_mapTR_loop___redArg(v___f_345_, v_L_337_, v___x_346_);
v___x_348_ = l_List_sum___redArg(v_toAdd_341_, v_toZero_343_, v___x_347_);
lean_dec(v_toZero_343_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___redArg___boxed(lean_object* v_inst_349_, lean_object* v_f_350_, lean_object* v_L_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_FreeAddGroup_Lift_aux___redArg(v_inst_349_, v_f_350_, v_L_351_);
lean_dec_ref(v_inst_349_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux(lean_object* v_00_u03b1_353_, lean_object* v_00_u03b2_354_, lean_object* v_inst_355_, lean_object* v_f_356_, lean_object* v_L_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_mathlib_FreeAddGroup_Lift_aux___redArg(v_inst_355_, v_f_356_, v_L_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___boxed(lean_object* v_00_u03b1_359_, lean_object* v_00_u03b2_360_, lean_object* v_inst_361_, lean_object* v_f_362_, lean_object* v_L_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_mathlib_FreeAddGroup_Lift_aux(v_00_u03b1_359_, v_00_u03b2_360_, v_inst_361_, v_f_362_, v_L_363_);
lean_dec_ref(v_inst_361_);
return v_res_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg___lam__0(lean_object* v_g_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_367_ = lp_mathlib_FreeGroup_of___redArg(v___y_366_);
v___x_368_ = lean_apply_1(v_g_365_, v___x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg___lam__1(lean_object* v_inst_369_, lean_object* v_f_370_, lean_object* v___y_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_FreeGroup_Lift_aux___redArg(v_inst_369_, v_f_370_, v___y_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg___lam__1___boxed(lean_object* v_inst_373_, lean_object* v_f_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_FreeGroup_lift___redArg___lam__1(v_inst_373_, v_f_374_, v___y_375_);
lean_dec_ref(v_inst_373_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___redArg(lean_object* v_inst_378_){
_start:
{
lean_object* v___f_379_; lean_object* v___f_380_; lean_object* v___x_381_; 
v___f_379_ = ((lean_object*)(lp_mathlib_FreeGroup_lift___redArg___closed__0));
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_lift___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_380_, 0, v_inst_378_);
v___x_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_381_, 0, v___f_380_);
lean_ctor_set(v___x_381_, 1, v___f_379_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift(lean_object* v_00_u03b1_382_, lean_object* v_00_u03b2_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_FreeGroup_lift___redArg(v_inst_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg___lam__0(lean_object* v_g_386_, lean_object* v___y_387_){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_388_ = lp_mathlib_FreeAddGroup_of___redArg(v___y_387_);
v___x_389_ = lean_apply_1(v_g_386_, v___x_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg___lam__1(lean_object* v_inst_390_, lean_object* v_f_391_, lean_object* v___y_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_FreeAddGroup_Lift_aux___redArg(v_inst_390_, v_f_391_, v___y_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg___lam__1___boxed(lean_object* v_inst_394_, lean_object* v_f_395_, lean_object* v___y_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_FreeAddGroup_lift___redArg___lam__1(v_inst_394_, v_f_395_, v___y_396_);
lean_dec_ref(v_inst_394_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___redArg(lean_object* v_inst_399_){
_start:
{
lean_object* v___f_400_; lean_object* v___f_401_; lean_object* v___x_402_; 
v___f_400_ = ((lean_object*)(lp_mathlib_FreeAddGroup_lift___redArg___closed__0));
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_lift___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_401_, 0, v_inst_399_);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v___f_401_);
lean_ctor_set(v___x_402_, 1, v___f_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b2_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_FreeAddGroup_lift___redArg(v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___redArg(lean_object* v_f_407_){
_start:
{
lean_inc(v_f_407_);
return v_f_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___redArg___boxed(lean_object* v_f_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___redArg(v_f_408_);
lean_dec(v_f_408_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1(lean_object* v_00_u03b2_410_, lean_object* v_00_u03b1_411_, lean_object* v_f_412_, lean_object* v_map__mul_413_){
_start:
{
lean_inc(v_f_412_);
return v_f_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1___boxed(lean_object* v_00_u03b2_414_, lean_object* v_00_u03b1_415_, lean_object* v_f_416_, lean_object* v_map__mul_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_map_spec__1(v_00_u03b2_414_, v_00_u03b1_415_, v_f_416_, v_map__mul_417_);
lean_dec(v_f_416_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_map_spec__0___redArg(lean_object* v_f_419_, lean_object* v_a_420_, lean_object* v_a_421_){
_start:
{
if (lean_obj_tag(v_a_420_) == 0)
{
lean_object* v___x_422_; 
lean_dec(v_f_419_);
v___x_422_ = l_List_reverse___redArg(v_a_421_);
return v___x_422_;
}
else
{
lean_object* v_head_423_; lean_object* v_tail_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_442_; 
v_head_423_ = lean_ctor_get(v_a_420_, 0);
v_tail_424_ = lean_ctor_get(v_a_420_, 1);
v_isSharedCheck_442_ = !lean_is_exclusive(v_a_420_);
if (v_isSharedCheck_442_ == 0)
{
v___x_426_ = v_a_420_;
v_isShared_427_ = v_isSharedCheck_442_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_tail_424_);
lean_inc(v_head_423_);
lean_dec(v_a_420_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_442_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v_fst_428_; lean_object* v_snd_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_441_; 
v_fst_428_ = lean_ctor_get(v_head_423_, 0);
v_snd_429_ = lean_ctor_get(v_head_423_, 1);
v_isSharedCheck_441_ = !lean_is_exclusive(v_head_423_);
if (v_isSharedCheck_441_ == 0)
{
v___x_431_ = v_head_423_;
v_isShared_432_ = v_isSharedCheck_441_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_snd_429_);
lean_inc(v_fst_428_);
lean_dec(v_head_423_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_441_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___x_433_; lean_object* v___x_435_; 
lean_inc(v_f_419_);
v___x_433_ = lean_apply_1(v_f_419_, v_fst_428_);
if (v_isShared_432_ == 0)
{
lean_ctor_set(v___x_431_, 0, v___x_433_);
v___x_435_ = v___x_431_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v___x_433_);
lean_ctor_set(v_reuseFailAlloc_440_, 1, v_snd_429_);
v___x_435_ = v_reuseFailAlloc_440_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
lean_object* v___x_437_; 
if (v_isShared_427_ == 0)
{
lean_ctor_set(v___x_426_, 1, v_a_421_);
lean_ctor_set(v___x_426_, 0, v___x_435_);
v___x_437_ = v___x_426_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v___x_435_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v_a_421_);
v___x_437_ = v_reuseFailAlloc_439_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
v_a_420_ = v_tail_424_;
v_a_421_ = v___x_437_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_map___redArg___lam__0(lean_object* v_f_443_, lean_object* v___y_444_){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_445_ = lean_box(0);
v___x_446_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_map_spec__0___redArg(v_f_443_, v___y_444_, v___x_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_map___redArg(lean_object* v_f_447_){
_start:
{
lean_object* v___f_448_; lean_object* v___x_449_; 
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_448_, 0, v_f_447_);
v___x_449_ = lean_alloc_closure((void*)(lp_mathlib_Quot_map), 7, 6);
lean_closure_set(v___x_449_, 0, lean_box(0));
lean_closure_set(v___x_449_, 1, lean_box(0));
lean_closure_set(v___x_449_, 2, lean_box(0));
lean_closure_set(v___x_449_, 3, lean_box(0));
lean_closure_set(v___x_449_, 4, v___f_448_);
lean_closure_set(v___x_449_, 5, lean_box(0));
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_map(lean_object* v_00_u03b1_450_, lean_object* v_00_u03b2_451_, lean_object* v_f_452_){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = lp_mathlib_FreeGroup_map___redArg(v_f_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_map_spec__0(lean_object* v_00_u03b1_454_, lean_object* v_00_u03b2_455_, lean_object* v_f_456_, lean_object* v_a_457_, lean_object* v_a_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_map_spec__0___redArg(v_f_456_, v_a_457_, v_a_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___redArg(lean_object* v_f_460_){
_start:
{
lean_inc(v_f_460_);
return v_f_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___redArg___boxed(lean_object* v_f_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___redArg(v_f_461_);
lean_dec(v_f_461_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0(lean_object* v_00_u03b2_463_, lean_object* v_00_u03b1_464_, lean_object* v_f_465_, lean_object* v_map__mul_466_){
_start:
{
lean_inc(v_f_465_);
return v_f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0___boxed(lean_object* v_00_u03b2_467_, lean_object* v_00_u03b1_468_, lean_object* v_f_469_, lean_object* v_map__mul_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_map_spec__0(v_00_u03b2_467_, v_00_u03b1_468_, v_f_469_, v_map__mul_470_);
lean_dec(v_f_469_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_map___redArg(lean_object* v_f_472_){
_start:
{
lean_object* v___f_473_; lean_object* v___x_474_; 
v___f_473_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_473_, 0, v_f_472_);
v___x_474_ = lean_alloc_closure((void*)(lp_mathlib_Quot_map), 7, 6);
lean_closure_set(v___x_474_, 0, lean_box(0));
lean_closure_set(v___x_474_, 1, lean_box(0));
lean_closure_set(v___x_474_, 2, lean_box(0));
lean_closure_set(v___x_474_, 3, lean_box(0));
lean_closure_set(v___x_474_, 4, v___f_473_);
lean_closure_set(v___x_474_, 5, lean_box(0));
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_map(lean_object* v_00_u03b1_475_, lean_object* v_00_u03b2_476_, lean_object* v_f_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_mathlib_FreeAddGroup_map___redArg(v_f_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__0(lean_object* v_e_479_, lean_object* v___y_480_){
_start:
{
lean_object* v_toFun_481_; lean_object* v___x_482_; 
v_toFun_481_ = lean_ctor_get(v_e_479_, 0);
lean_inc(v_toFun_481_);
lean_dec_ref(v_e_479_);
v___x_482_ = lean_apply_1(v_toFun_481_, v___y_480_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__1(lean_object* v___f_483_, lean_object* v___y_484_){
_start:
{
lean_object* v___x_76__overap_485_; lean_object* v___x_486_; 
v___x_76__overap_485_ = lp_mathlib_FreeGroup_map___redArg(v___f_483_);
v___x_486_ = lean_apply_1(v___x_76__overap_485_, v___y_484_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__2(lean_object* v___x_487_, lean_object* v___y_488_){
_start:
{
lean_object* v_toFun_489_; lean_object* v___x_490_; 
v_toFun_489_ = lean_ctor_get(v___x_487_, 0);
lean_inc(v_toFun_489_);
lean_dec_ref(v___x_487_);
v___x_490_ = lean_apply_1(v_toFun_489_, v___y_488_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr___redArg(lean_object* v_e_491_){
_start:
{
lean_object* v___f_492_; lean_object* v___f_493_; lean_object* v___x_494_; lean_object* v___f_495_; lean_object* v___f_496_; lean_object* v___x_497_; 
lean_inc_ref(v_e_491_);
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_492_, 0, v_e_491_);
v___f_493_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_493_, 0, v___f_492_);
v___x_494_ = lp_mathlib_Equiv_symm___redArg(v_e_491_);
v___f_495_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_495_, 0, v___x_494_);
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_496_, 0, v___f_495_);
v___x_497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_497_, 0, v___f_493_);
lean_ctor_set(v___x_497_, 1, v___f_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupCongr(lean_object* v_00_u03b1_498_, lean_object* v_00_u03b2_499_, lean_object* v_e_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lp_mathlib_FreeGroup_freeGroupCongr___redArg(v_e_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg___lam__1(lean_object* v___f_502_, lean_object* v___y_503_){
_start:
{
lean_object* v___x_76__overap_504_; lean_object* v___x_505_; 
v___x_76__overap_504_ = lp_mathlib_FreeAddGroup_map___redArg(v___f_502_);
v___x_505_ = lean_apply_1(v___x_76__overap_504_, v___y_503_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg(lean_object* v_e_506_){
_start:
{
lean_object* v___f_507_; lean_object* v___f_508_; lean_object* v___x_509_; lean_object* v___f_510_; lean_object* v___f_511_; lean_object* v___x_512_; 
lean_inc_ref(v_e_506_);
v___f_507_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_507_, 0, v_e_506_);
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_508_, 0, v___f_507_);
v___x_509_ = lp_mathlib_Equiv_symm___redArg(v_e_506_);
v___f_510_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_freeGroupCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_510_, 0, v___x_509_);
v___f_511_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_511_, 0, v___f_510_);
v___x_512_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_512_, 0, v___f_508_);
lean_ctor_set(v___x_512_, 1, v___f_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_freeAddGroupCongr(lean_object* v_00_u03b1_513_, lean_object* v_00_u03b2_514_, lean_object* v_e_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_mathlib_FreeAddGroup_freeAddGroupCongr___redArg(v_e_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___redArg(lean_object* v_inst_518_){
_start:
{
lean_object* v___x_519_; lean_object* v_toFun_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v___x_519_ = lp_mathlib_FreeGroup_lift___redArg(v_inst_518_);
v_toFun_520_ = lean_ctor_get(v___x_519_, 0);
lean_inc(v_toFun_520_);
lean_dec_ref(v___x_519_);
v___x_521_ = ((lean_object*)(lp_mathlib_FreeGroup_prod___redArg___closed__0));
v___x_522_ = lean_apply_1(v_toFun_520_, v___x_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod(lean_object* v_00_u03b1_523_, lean_object* v_inst_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_FreeGroup_prod___redArg(v_inst_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_sum___redArg(lean_object* v_inst_526_){
_start:
{
lean_object* v___x_527_; lean_object* v_toFun_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_527_ = lp_mathlib_FreeAddGroup_lift___redArg(v_inst_526_);
v_toFun_528_ = lean_ctor_get(v___x_527_, 0);
lean_inc(v_toFun_528_);
lean_dec_ref(v___x_527_);
v___x_529_ = ((lean_object*)(lp_mathlib_FreeGroup_prod___redArg___closed__0));
v___x_530_ = lean_apply_1(v_toFun_528_, v___x_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_sum(lean_object* v_00_u03b1_531_, lean_object* v_inst_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_FreeAddGroup_sum___redArg(v_inst_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_sum___redArg(lean_object* v_inst_534_, lean_object* v_x_535_){
_start:
{
lean_object* v___x_536_; lean_object* v___x_15__overap_537_; lean_object* v___x_538_; 
v___x_536_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_534_);
v___x_15__overap_537_ = lp_mathlib_FreeGroup_prod___redArg(v___x_536_);
v___x_538_ = lean_apply_1(v___x_15__overap_537_, v_x_535_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_sum(lean_object* v_00_u03b1_539_, lean_object* v_inst_540_, lean_object* v_x_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lp_mathlib_FreeGroup_sum___redArg(v_inst_540_, v_x_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__1(lean_object* v_npow_543_, lean_object* v_x_544_, lean_object* v_x_545_){
_start:
{
lean_object* v_intZero_546_; uint8_t v_isNeg_547_; 
v_intZero_546_ = lean_obj_once(&lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0, &lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0_once, _init_lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0);
v_isNeg_547_ = lean_int_dec_lt(v_x_544_, v_intZero_546_);
if (v_isNeg_547_ == 0)
{
lean_object* v_a_548_; lean_object* v___x_549_; 
v_a_548_ = lean_nat_abs(v_x_544_);
v___x_549_ = lean_apply_2(v_npow_543_, v_a_548_, v_x_545_);
return v___x_549_;
}
else
{
lean_object* v_abs_550_; lean_object* v_one_551_; lean_object* v_a_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; 
v_abs_550_ = lean_nat_abs(v_x_544_);
v_one_551_ = lean_unsigned_to_nat(1u);
v_a_552_ = lean_nat_sub(v_abs_550_, v_one_551_);
lean_dec(v_abs_550_);
v___x_553_ = lean_nat_add(v_a_552_, v_one_551_);
lean_dec(v_a_552_);
v___x_554_ = lean_apply_2(v_npow_543_, v___x_553_, v_x_545_);
v___x_555_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_554_);
return v___x_555_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__1___boxed(lean_object* v_npow_556_, lean_object* v_x_557_, lean_object* v_x_558_){
_start:
{
lean_object* v_res_559_; 
v_res_559_ = lp_mathlib_zpowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__1(v_npow_556_, v_x_557_, v_x_558_);
lean_dec(v_x_557_);
return v_res_559_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___closed__0(void){
_start:
{
lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_560_ = lean_box(0);
v___x_561_ = lp_mathlib_FreeGroup_of___redArg(v___x_560_);
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0(lean_object* v___f_562_, lean_object* v_x_563_){
_start:
{
lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_564_ = lean_obj_once(&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___closed__0, &lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___closed__0_once, _init_lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___closed__0);
v___x_565_ = lp_mathlib_zpowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__1(v___f_562_, v_x_563_, v___x_564_);
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0___boxed(lean_object* v___f_566_, lean_object* v_x_567_){
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__0(v___f_566_, v_x_567_);
lean_dec(v_x_567_);
return v_res_568_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0(void){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_569_ = lean_unsigned_to_nat(1u);
v___x_570_ = lean_nat_to_int(v___x_569_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1(lean_object* v_x_571_){
_start:
{
lean_object* v___x_572_; 
v___x_572_ = lean_obj_once(&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0, &lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0_once, _init_lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0);
return v___x_572_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_573_; 
v___x_573_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_573_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg(lean_object* v_f_575_, lean_object* v_a_576_, lean_object* v_a_577_){
_start:
{
if (lean_obj_tag(v_a_576_) == 0)
{
lean_object* v___x_578_; 
lean_dec_ref(v_f_575_);
v___x_578_ = l_List_reverse___redArg(v_a_577_);
return v___x_578_;
}
else
{
lean_object* v_head_579_; lean_object* v_tail_580_; lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_603_; 
v_head_579_ = lean_ctor_get(v_a_576_, 0);
v_tail_580_ = lean_ctor_get(v_a_576_, 1);
v_isSharedCheck_603_ = !lean_is_exclusive(v_a_576_);
if (v_isSharedCheck_603_ == 0)
{
v___x_582_ = v_a_576_;
v_isShared_583_ = v_isSharedCheck_603_;
goto v_resetjp_581_;
}
else
{
lean_inc(v_tail_580_);
lean_inc(v_head_579_);
lean_dec(v_a_576_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_603_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v___y_585_; lean_object* v_snd_590_; uint8_t v___x_591_; 
v_snd_590_ = lean_ctor_get(v_head_579_, 1);
v___x_591_ = lean_unbox(v_snd_590_);
if (v___x_591_ == 0)
{
lean_object* v_fst_592_; lean_object* v___x_593_; lean_object* v_toFun_594_; lean_object* v___x_595_; lean_object* v_toFun_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
v_fst_592_ = lean_ctor_get(v_head_579_, 0);
lean_inc(v_fst_592_);
lean_dec(v_head_579_);
v___x_593_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0);
v_toFun_594_ = lean_ctor_get(v___x_593_, 0);
v___x_595_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__1);
v_toFun_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc_ref(v_f_575_);
v___x_597_ = lean_apply_1(v_f_575_, v_fst_592_);
lean_inc(v_toFun_594_);
v___x_598_ = lean_apply_1(v_toFun_594_, v___x_597_);
v___x_599_ = lean_int_neg(v___x_598_);
lean_dec(v___x_598_);
lean_inc(v_toFun_596_);
v___x_600_ = lean_apply_1(v_toFun_596_, v___x_599_);
v___y_585_ = v___x_600_;
goto v___jp_584_;
}
else
{
lean_object* v_fst_601_; lean_object* v___x_602_; 
v_fst_601_ = lean_ctor_get(v_head_579_, 0);
lean_inc(v_fst_601_);
lean_dec(v_head_579_);
lean_inc_ref(v_f_575_);
v___x_602_ = lean_apply_1(v_f_575_, v_fst_601_);
v___y_585_ = v___x_602_;
goto v___jp_584_;
}
v___jp_584_:
{
lean_object* v___x_587_; 
if (v_isShared_583_ == 0)
{
lean_ctor_set(v___x_582_, 1, v_a_577_);
lean_ctor_set(v___x_582_, 0, v___y_585_);
v___x_587_ = v___x_582_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v___y_585_);
lean_ctor_set(v_reuseFailAlloc_589_, 1, v_a_577_);
v___x_587_ = v_reuseFailAlloc_589_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
v_a_576_ = v_tail_580_;
v_a_577_ = v___x_587_;
goto _start;
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0(void){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(lean_object* v_init_605_, lean_object* v_x_606_){
_start:
{
if (lean_obj_tag(v_x_606_) == 0)
{
lean_inc(v_init_605_);
return v_init_605_;
}
else
{
lean_object* v_head_607_; lean_object* v_tail_608_; lean_object* v___x_609_; lean_object* v_toFun_610_; lean_object* v___x_611_; lean_object* v_toFun_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v_head_607_ = lean_ctor_get(v_x_606_, 0);
lean_inc(v_head_607_);
v_tail_608_ = lean_ctor_get(v_x_606_, 1);
lean_inc(v_tail_608_);
lean_dec_ref_known(v_x_606_, 2);
v___x_609_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0);
v_toFun_610_ = lean_ctor_get(v___x_609_, 0);
v___x_611_ = lean_obj_once(&lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0, &lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0_once, _init_lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0);
v_toFun_612_ = lean_ctor_get(v___x_611_, 0);
lean_inc_n(v_toFun_610_, 2);
v___x_613_ = lean_apply_1(v_toFun_610_, v_head_607_);
v___x_614_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(v_init_605_, v_tail_608_);
v___x_615_ = lean_apply_1(v_toFun_610_, v___x_614_);
v___x_616_ = lean_int_add(v___x_613_, v___x_615_);
lean_dec(v___x_615_);
lean_dec(v___x_613_);
lean_inc(v_toFun_612_);
v___x_617_ = lean_apply_1(v_toFun_612_, v___x_616_);
return v___x_617_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___boxed(lean_object* v_init_618_, lean_object* v_x_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(v_init_618_, v_x_619_);
lean_dec(v_init_618_);
return v_res_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6(lean_object* v_l_621_){
_start:
{
lean_object* v___x_622_; lean_object* v_toFun_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_622_ = lean_obj_once(&lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0, &lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0_once, _init_lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0);
v_toFun_623_ = lean_ctor_get(v___x_622_, 0);
v___x_624_ = lean_obj_once(&lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0, &lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0_once, _init_lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg___closed__0);
lean_inc(v_toFun_623_);
v___x_625_ = lean_apply_1(v_toFun_623_, v___x_624_);
v___x_626_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(v___x_625_, v_l_621_);
lean_dec(v___x_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4___redArg(lean_object* v_f_627_, lean_object* v_L_628_){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_629_ = lean_box(0);
v___x_630_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg(v_f_627_, v_L_628_, v___x_629_);
v___x_631_ = lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6(v___x_630_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___lam__0(lean_object* v_g_632_, lean_object* v___y_633_){
_start:
{
lean_object* v___x_634_; lean_object* v___x_635_; 
v___x_634_ = lp_mathlib_FreeGroup_of___redArg(v___y_633_);
v___x_635_ = lean_apply_1(v_g_632_, v___x_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3(lean_object* v_00_u03b1_641_){
_start:
{
lean_object* v___x_642_; 
v___x_642_ = ((lean_object*)(lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3___closed__2));
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___lam__0(lean_object* v___y_643_){
_start:
{
lean_inc(v___y_643_);
return v___y_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___lam__0___boxed(lean_object* v___y_644_){
_start:
{
lean_object* v_res_645_; 
v_res_645_ = lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___lam__0(v___y_644_);
lean_dec(v___y_644_);
return v_res_645_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__0(void){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lp_mathlib_FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3(lean_box(0));
return v___x_646_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2(void){
_start:
{
lean_object* v___x_648_; lean_object* v_toFun_649_; lean_object* v___f_650_; lean_object* v___x_651_; 
v___x_648_ = lean_obj_once(&lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__0, &lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__0_once, _init_lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__0);
v_toFun_649_ = lean_ctor_get(v___x_648_, 0);
v___f_650_ = ((lean_object*)(lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2___closed__1));
lean_inc(v_toFun_649_);
v___x_651_ = lean_apply_1(v_toFun_649_, v___f_650_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2(lean_object* v_x_652_){
_start:
{
lean_object* v___x_134__overap_653_; lean_object* v___x_654_; 
v___x_134__overap_653_ = lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2;
v___x_654_ = lean_apply_1(v___x_134__overap_653_, v_x_652_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__2(lean_object* v___f_655_, lean_object* v_x_656_){
_start:
{
lean_object* v___x_642__overap_657_; lean_object* v___x_658_; lean_object* v___x_824__overap_659_; lean_object* v___x_660_; 
v___x_642__overap_657_ = lp_mathlib_FreeGroup_map___redArg(v___f_655_);
v___x_658_ = lean_apply_1(v___x_642__overap_657_, v_x_656_);
v___x_824__overap_659_ = lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2;
v___x_660_ = lean_apply_1(v___x_824__overap_659_, v___x_658_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0(lean_object* v_x_661_, lean_object* v_x_662_){
_start:
{
lean_object* v_zero_663_; uint8_t v_isZero_664_; 
v_zero_663_ = lean_unsigned_to_nat(0u);
v_isZero_664_ = lean_nat_dec_eq(v_x_661_, v_zero_663_);
if (v_isZero_664_ == 1)
{
lean_object* v___x_665_; 
lean_dec(v_x_662_);
v___x_665_ = lean_box(0);
return v___x_665_;
}
else
{
lean_object* v_one_666_; lean_object* v_n_667_; lean_object* v___x_668_; lean_object* v___x_669_; 
v_one_666_ = lean_unsigned_to_nat(1u);
v_n_667_ = lean_nat_sub(v_x_661_, v_one_666_);
lean_inc(v_x_662_);
v___x_668_ = lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0(v_n_667_, v_x_662_);
lean_dec(v_n_667_);
v___x_669_ = l_List_appendTR___redArg(v___x_668_, v_x_662_);
return v___x_669_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0___boxed(lean_object* v_x_670_, lean_object* v_x_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_npowRec___at___00FreeGroup_freeGroupUnitEquivInt_spec__0(v_x_670_, v_x_671_);
lean_dec(v_x_670_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___redArg(lean_object* v_f_683_){
_start:
{
lean_inc_ref(v_f_683_);
return v_f_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_f_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___redArg(v_f_684_);
lean_dec_ref(v_f_684_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5(lean_object* v_00_u03b1_686_, lean_object* v_f_687_, lean_object* v_map__mul_688_){
_start:
{
lean_inc_ref(v_f_687_);
return v_f_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b1_689_, lean_object* v_f_690_, lean_object* v_map__mul_691_){
_start:
{
lean_object* v_res_692_; 
v_res_692_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__5(v_00_u03b1_689_, v_f_690_, v_map__mul_691_);
lean_dec_ref(v_f_690_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4(lean_object* v_00_u03b1_693_, lean_object* v_f_694_, lean_object* v_L_695_){
_start:
{
lean_object* v___x_696_; 
v___x_696_ = lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4___redArg(v_f_694_, v_L_695_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5(lean_object* v_00_u03b1_697_, lean_object* v_f_698_, lean_object* v_a_699_, lean_object* v_a_700_){
_start:
{
lean_object* v___x_701_; 
v___x_701_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg(v_f_698_, v_a_699_, v_a_700_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__0(lean_object* v_inst_702_, lean_object* v_x_703_){
_start:
{
lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_704_ = lp_mathlib_FreeGroup_of___redArg(v_inst_702_);
v___x_705_ = lean_obj_once(&lp_mathlib_FreeGroup_instGroup___closed__2, &lp_mathlib_FreeGroup_instGroup___closed__2_once, _init_lp_mathlib_FreeGroup_instGroup___closed__2);
v___x_706_ = ((lean_object*)(lp_mathlib_FreeGroup_instGroup___closed__4));
v___x_707_ = lp_mathlib_zpowRec___redArg(v___x_705_, v___x_706_, v_x_703_, v___x_704_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__0___boxed(lean_object* v_inst_708_, lean_object* v_x_709_){
_start:
{
lean_object* v_res_710_; 
v_res_710_ = lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__0(v_inst_708_, v_x_709_);
lean_dec(v_x_709_);
return v_res_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__1(lean_object* v_x_711_){
_start:
{
lean_object* v___x_712_; 
v___x_712_ = lean_obj_once(&lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0, &lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0_once, _init_lp_mathlib_FreeGroup_freeGroupUnitEquivInt___lam__1___closed__0);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__1___boxed(lean_object* v_x_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__1(v_x_713_);
lean_dec(v_x_713_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__2(lean_object* v___f_715_, lean_object* v___x_716_, lean_object* v_x_717_){
_start:
{
lean_object* v___x_117__overap_718_; lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_117__overap_718_ = lp_mathlib_FreeGroup_map___redArg(v___f_715_);
v___x_719_ = lean_apply_1(v___x_117__overap_718_, v_x_717_);
v___x_720_ = lp_mathlib_FreeGroup_sum___redArg(v___x_716_, v___x_719_);
return v___x_720_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__1(void){
_start:
{
lean_object* v___x_722_; lean_object* v___f_723_; lean_object* v___f_724_; 
v___x_722_ = lp_mathlib_Int_instAddCommGroup;
v___f_723_ = ((lean_object*)(lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__0));
v___f_724_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__2), 3, 2);
lean_closure_set(v___f_724_, 0, v___f_723_);
lean_closure_set(v___f_724_, 1, v___x_722_);
return v___f_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique___redArg(lean_object* v_inst_725_){
_start:
{
lean_object* v___f_726_; lean_object* v___f_727_; lean_object* v___x_728_; 
v___f_726_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_equivIntOfUnique___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_726_, 0, v_inst_725_);
v___f_727_ = lean_obj_once(&lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__1, &lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__1_once, _init_lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__1);
v___x_728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_728_, 0, v___f_727_);
lean_ctor_set(v___x_728_, 1, v___f_726_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_equivIntOfUnique(lean_object* v_00_u03b1_729_, lean_object* v_inst_730_){
_start:
{
lean_object* v___x_731_; 
v___x_731_ = lp_mathlib_FreeGroup_equivIntOfUnique___redArg(v_inst_730_);
return v___x_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__0(lean_object* v___x_732_, lean_object* v___y_733_){
_start:
{
lean_object* v_toFun_734_; lean_object* v___x_735_; 
v_toFun_734_ = lean_ctor_get(v___x_732_, 0);
lean_inc(v_toFun_734_);
lean_dec_ref(v___x_732_);
v___x_735_ = lean_apply_1(v_toFun_734_, v___y_733_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__1(lean_object* v___x_736_, lean_object* v___y_737_){
_start:
{
lean_object* v_toFun_738_; lean_object* v___x_739_; 
v_toFun_738_ = lean_ctor_get(v___x_736_, 0);
lean_inc(v_toFun_738_);
lean_dec_ref(v___x_736_);
v___x_739_ = lean_apply_1(v_toFun_738_, v___y_737_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__2(lean_object* v___x_740_, lean_object* v___y_741_){
_start:
{
lean_object* v_toFun_742_; lean_object* v___x_743_; 
v_toFun_742_ = lean_ctor_get(v___x_740_, 0);
lean_inc(v_toFun_742_);
lean_dec_ref(v___x_740_);
v___x_743_ = lean_apply_1(v_toFun_742_, v___y_741_);
return v___x_743_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__0(void){
_start:
{
lean_object* v___x_744_; lean_object* v___f_745_; 
v___x_744_ = lean_obj_once(&lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0, &lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0_once, _init_lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___closed__0);
v___f_745_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__0), 2, 1);
lean_closure_set(v___f_745_, 0, v___x_744_);
return v___f_745_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__1(void){
_start:
{
lean_object* v___x_746_; lean_object* v___f_747_; 
v___x_746_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2_spec__3_spec__4_spec__5___redArg___closed__0);
v___f_747_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__0), 2, 1);
lean_closure_set(v___f_747_, 0, v___x_746_);
return v___f_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg(lean_object* v_inst_748_){
_start:
{
lean_object* v___f_749_; lean_object* v___x_750_; lean_object* v___f_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___f_754_; lean_object* v___f_755_; lean_object* v___x_756_; lean_object* v___x_757_; 
v___f_749_ = lean_obj_once(&lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__0, &lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__0_once, _init_lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__0);
v___x_750_ = lp_mathlib_FreeGroup_equivIntOfUnique___redArg(v_inst_748_);
lean_inc_ref(v___x_750_);
v___f_751_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__1), 2, 1);
lean_closure_set(v___f_751_, 0, v___x_750_);
v___x_752_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_752_, 0, lean_box(0));
lean_closure_set(v___x_752_, 1, lean_box(0));
lean_closure_set(v___x_752_, 2, lean_box(0));
lean_closure_set(v___x_752_, 3, v___f_749_);
lean_closure_set(v___x_752_, 4, v___f_751_);
v___x_753_ = lp_mathlib_Equiv_symm___redArg(v___x_750_);
v___f_754_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___lam__2), 2, 1);
lean_closure_set(v___f_754_, 0, v___x_753_);
v___f_755_ = lean_obj_once(&lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__1, &lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__1_once, _init_lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg___closed__1);
v___x_756_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_756_, 0, lean_box(0));
lean_closure_set(v___x_756_, 1, lean_box(0));
lean_closure_set(v___x_756_, 2, lean_box(0));
lean_closure_set(v___x_756_, 3, v___f_754_);
lean_closure_set(v___x_756_, 4, v___f_755_);
v___x_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_757_, 0, v___x_752_);
lean_ctor_set(v___x_757_, 1, v___x_756_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_mulEquivIntOfUnique(lean_object* v_00_u03b1_758_, lean_object* v_inst_759_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_mathlib_FreeGroup_mulEquivIntOfUnique___redArg(v_inst_759_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__0(lean_object* v_inst_761_, lean_object* v___f_762_, lean_object* v_x_763_){
_start:
{
lean_object* v___x_764_; lean_object* v___x_765_; 
v___x_764_ = lp_mathlib_FreeAddGroup_of___redArg(v_inst_761_);
v___x_765_ = lp_mathlib_zsmulRec___at___00FreeAddGroup_instAddGroup_spec__3___redArg(v___f_762_, v_x_763_, v___x_764_);
return v___x_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__0___boxed(lean_object* v_inst_766_, lean_object* v___f_767_, lean_object* v_x_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__0(v_inst_766_, v___f_767_, v_x_768_);
lean_dec(v_x_768_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__2(lean_object* v___f_770_, lean_object* v___x_771_, lean_object* v_x_772_){
_start:
{
lean_object* v___x_136__overap_773_; lean_object* v___x_774_; lean_object* v___x_137__overap_775_; lean_object* v___x_776_; 
v___x_136__overap_773_ = lp_mathlib_FreeAddGroup_map___redArg(v___f_770_);
v___x_774_ = lean_apply_1(v___x_136__overap_773_, v_x_772_);
v___x_137__overap_775_ = lp_mathlib_FreeAddGroup_sum___redArg(v___x_771_);
v___x_776_ = lean_apply_1(v___x_137__overap_775_, v___x_774_);
return v___x_776_;
}
}
static lean_object* _init_lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___closed__0(void){
_start:
{
lean_object* v___x_777_; lean_object* v___f_778_; lean_object* v___f_779_; 
v___x_777_ = lp_mathlib_Int_instAddCommGroup;
v___f_778_ = ((lean_object*)(lp_mathlib_FreeGroup_equivIntOfUnique___redArg___closed__0));
v___f_779_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__2), 3, 2);
lean_closure_set(v___f_779_, 0, v___f_778_);
lean_closure_set(v___f_779_, 1, v___x_777_);
return v___f_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg(lean_object* v_inst_780_){
_start:
{
lean_object* v___f_781_; lean_object* v___f_782_; lean_object* v___f_783_; lean_object* v___x_784_; 
v___f_781_ = ((lean_object*)(lp_mathlib_FreeAddGroup_instAddGroup___closed__2));
v___f_782_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_782_, 0, v_inst_780_);
lean_closure_set(v___f_782_, 1, v___f_781_);
v___f_783_ = lean_obj_once(&lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___closed__0, &lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___closed__0_once, _init_lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg___closed__0);
v___x_784_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_784_, 0, v___f_783_);
lean_ctor_set(v___x_784_, 1, v___f_782_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_addEquivIntOfUnique(lean_object* v_00_u03b1_785_, lean_object* v_inst_786_){
_start:
{
lean_object* v___x_787_; 
v___x_787_ = lp_mathlib_FreeAddGroup_addEquivIntOfUnique___redArg(v_inst_786_);
return v___x_787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__0(lean_object* v_00___u03b1_788_, lean_object* v_00___u03b2_789_, lean_object* v_f_790_, lean_object* v___y_791_){
_start:
{
lean_object* v___x_224__overap_792_; lean_object* v___x_793_; 
v___x_224__overap_792_ = lp_mathlib_FreeGroup_map___redArg(v_f_790_);
v___x_793_ = lean_apply_1(v___x_224__overap_792_, v___y_791_);
return v___x_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__1(lean_object* v_00_u03b1_794_, lean_object* v_00_u03b2_795_, lean_object* v___y_796_, lean_object* v___y_797_){
_start:
{
lean_object* v___x_798_; lean_object* v___x_230__overap_799_; lean_object* v___x_800_; 
v___x_798_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_798_, 0, lean_box(0));
lean_closure_set(v___x_798_, 1, lean_box(0));
lean_closure_set(v___x_798_, 2, v___y_796_);
v___x_230__overap_799_ = lp_mathlib_FreeGroup_map___redArg(v___x_798_);
v___x_800_ = lean_apply_1(v___x_230__overap_799_, v___y_797_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__2(lean_object* v_x_801_, lean_object* v_y_802_){
_start:
{
lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_233__overap_805_; lean_object* v___x_806_; 
v___x_803_ = lean_box(0);
v___x_804_ = lean_apply_1(v_x_801_, v___x_803_);
v___x_233__overap_805_ = lp_mathlib_FreeGroup_map___redArg(v_y_802_);
v___x_806_ = lean_apply_1(v___x_233__overap_805_, v___x_804_);
return v___x_806_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instMonad___lam__3___closed__0(void){
_start:
{
lean_object* v___x_807_; 
v___x_807_ = lp_mathlib_FreeGroup_instGroup(lean_box(0));
return v___x_807_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instMonad___lam__3___closed__1(void){
_start:
{
lean_object* v___x_808_; lean_object* v___x_809_; 
v___x_808_ = lean_obj_once(&lp_mathlib_FreeGroup_instMonad___lam__3___closed__0, &lp_mathlib_FreeGroup_instMonad___lam__3___closed__0_once, _init_lp_mathlib_FreeGroup_instMonad___lam__3___closed__0);
v___x_809_ = lp_mathlib_FreeGroup_lift___redArg(v___x_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__3(lean_object* v_00_u03b1_810_, lean_object* v_00_u03b2_811_, lean_object* v_f_812_, lean_object* v_x_813_){
_start:
{
lean_object* v___x_814_; lean_object* v_toFun_815_; lean_object* v___f_816_; lean_object* v___x_817_; 
v___x_814_ = lean_obj_once(&lp_mathlib_FreeGroup_instMonad___lam__3___closed__1, &lp_mathlib_FreeGroup_instMonad___lam__3___closed__1_once, _init_lp_mathlib_FreeGroup_instMonad___lam__3___closed__1);
v_toFun_815_ = lean_ctor_get(v___x_814_, 0);
v___f_816_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_instMonad___lam__2), 2, 1);
lean_closure_set(v___f_816_, 0, v_x_813_);
lean_inc(v_toFun_815_);
v___x_817_ = lean_apply_2(v_toFun_815_, v___f_816_, v_f_812_);
return v___x_817_;
}
}
static lean_object* _init_lp_mathlib_FreeGroup_instMonad___lam__4___closed__0(void){
_start:
{
lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_818_ = lean_obj_once(&lp_mathlib_FreeGroup_instMonad___lam__3___closed__0, &lp_mathlib_FreeGroup_instMonad___lam__3___closed__0_once, _init_lp_mathlib_FreeGroup_instMonad___lam__3___closed__0);
v___x_819_ = lp_mathlib_FreeGroup_lift___redArg(v___x_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__4(lean_object* v_00___u03b1_820_, lean_object* v_00___u03b2_821_, lean_object* v_x_822_, lean_object* v_f_823_){
_start:
{
lean_object* v___x_824_; lean_object* v_toFun_825_; lean_object* v___x_826_; 
v___x_824_ = lean_obj_once(&lp_mathlib_FreeGroup_instMonad___lam__4___closed__0, &lp_mathlib_FreeGroup_instMonad___lam__4___closed__0_once, _init_lp_mathlib_FreeGroup_instMonad___lam__4___closed__0);
v_toFun_825_ = lean_ctor_get(v___x_824_, 0);
lean_inc(v_toFun_825_);
v___x_826_ = lean_apply_2(v_toFun_825_, v_f_823_, v_x_822_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__5(lean_object* v_a_827_, lean_object* v_x_828_){
_start:
{
lean_object* v___x_829_; 
v___x_829_ = lp_mathlib_FreeGroup_of___redArg(v_a_827_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__5___boxed(lean_object* v_a_830_, lean_object* v_x_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_FreeGroup_instMonad___lam__5(v_a_830_, v_x_831_);
lean_dec(v_x_831_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__6(lean_object* v_y_833_, lean_object* v___f_834_, lean_object* v_a_835_){
_start:
{
lean_object* v___f_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; 
v___f_836_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_instMonad___lam__5___boxed), 2, 1);
lean_closure_set(v___f_836_, 0, v_a_835_);
v___x_837_ = lean_box(0);
v___x_838_ = lean_apply_1(v_y_833_, v___x_837_);
v___x_839_ = lean_apply_4(v___f_834_, lean_box(0), lean_box(0), v___x_838_, v___f_836_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__7(lean_object* v___f_840_, lean_object* v_00_u03b1_841_, lean_object* v_00_u03b2_842_, lean_object* v_x_843_, lean_object* v_y_844_){
_start:
{
lean_object* v___f_845_; lean_object* v___x_846_; 
lean_inc(v___f_840_);
v___f_845_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_instMonad___lam__6), 3, 2);
lean_closure_set(v___f_845_, 0, v_y_844_);
lean_closure_set(v___f_845_, 1, v___f_840_);
v___x_846_ = lean_apply_4(v___f_840_, lean_box(0), lean_box(0), v_x_843_, v___f_845_);
return v___x_846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__8(lean_object* v_y_847_, lean_object* v_x_848_){
_start:
{
lean_object* v___x_849_; lean_object* v___x_850_; 
v___x_849_ = lean_box(0);
v___x_850_ = lean_apply_1(v_y_847_, v___x_849_);
return v___x_850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__8___boxed(lean_object* v_y_851_, lean_object* v_x_852_){
_start:
{
lean_object* v_res_853_; 
v_res_853_ = lp_mathlib_FreeGroup_instMonad___lam__8(v_y_851_, v_x_852_);
lean_dec(v_x_852_);
return v_res_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_instMonad___lam__9(lean_object* v_00_u03b1_854_, lean_object* v_00_u03b2_855_, lean_object* v_x_856_, lean_object* v_y_857_){
_start:
{
lean_object* v___x_858_; lean_object* v_toFun_859_; lean_object* v___f_860_; lean_object* v___x_861_; 
v___x_858_ = lean_obj_once(&lp_mathlib_FreeGroup_instMonad___lam__4___closed__0, &lp_mathlib_FreeGroup_instMonad___lam__4___closed__0_once, _init_lp_mathlib_FreeGroup_instMonad___lam__4___closed__0);
v_toFun_859_ = lean_ctor_get(v___x_858_, 0);
v___f_860_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_instMonad___lam__8___boxed), 2, 1);
lean_closure_set(v___f_860_, 0, v_y_857_);
lean_inc(v_toFun_859_);
v___x_861_ = lean_apply_2(v_toFun_859_, v___f_860_, v_x_856_);
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__0(lean_object* v_00___u03b1_883_, lean_object* v_00___u03b2_884_, lean_object* v_f_885_, lean_object* v___y_886_){
_start:
{
lean_object* v___x_554__overap_887_; lean_object* v___x_888_; 
v___x_554__overap_887_ = lp_mathlib_FreeAddGroup_map___redArg(v_f_885_);
v___x_888_ = lean_apply_1(v___x_554__overap_887_, v___y_886_);
return v___x_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__1(lean_object* v___y_889_, lean_object* v___y_890_){
_start:
{
lean_inc(v___y_889_);
return v___y_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__1___boxed(lean_object* v___y_891_, lean_object* v___y_892_){
_start:
{
lean_object* v_res_893_; 
v_res_893_ = lp_mathlib_FreeAddGroup_instMonad___lam__1(v___y_891_, v___y_892_);
lean_dec(v___y_892_);
lean_dec(v___y_891_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__2(lean_object* v_00_u03b1_894_, lean_object* v_00_u03b2_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
lean_object* v___f_898_; lean_object* v___x_562__overap_899_; lean_object* v___x_900_; 
v___f_898_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_instMonad___lam__1___boxed), 2, 1);
lean_closure_set(v___f_898_, 0, v___y_896_);
v___x_562__overap_899_ = lp_mathlib_FreeAddGroup_map___redArg(v___f_898_);
v___x_900_ = lean_apply_1(v___x_562__overap_899_, v___y_897_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__3(lean_object* v_x_901_, lean_object* v_y_902_){
_start:
{
lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_565__overap_905_; lean_object* v___x_906_; 
v___x_903_ = lean_box(0);
v___x_904_ = lean_apply_1(v_x_901_, v___x_903_);
v___x_565__overap_905_ = lp_mathlib_FreeAddGroup_map___redArg(v_y_902_);
v___x_906_ = lean_apply_1(v___x_565__overap_905_, v___x_904_);
return v___x_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg(lean_object* v_init_907_, lean_object* v_x_908_){
_start:
{
if (lean_obj_tag(v_x_908_) == 0)
{
lean_inc(v_init_907_);
return v_init_907_;
}
else
{
lean_object* v_head_909_; lean_object* v_tail_910_; lean_object* v___x_911_; lean_object* v___x_912_; 
v_head_909_ = lean_ctor_get(v_x_908_, 0);
lean_inc(v_head_909_);
v_tail_910_ = lean_ctor_get(v_x_908_, 1);
lean_inc(v_tail_910_);
lean_dec_ref_known(v_x_908_, 2);
v___x_911_ = lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg(v_init_907_, v_tail_910_);
v___x_912_ = l_List_appendTR___redArg(v_head_909_, v___x_911_);
return v___x_912_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object* v_init_913_, lean_object* v_x_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg(v_init_913_, v_x_914_);
lean_dec(v_init_913_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2___redArg(lean_object* v_l_916_){
_start:
{
lean_object* v___x_917_; lean_object* v___x_918_; 
v___x_917_ = lean_box(0);
v___x_918_ = lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg(v___x_917_, v_l_916_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__1___redArg(lean_object* v_f_919_, lean_object* v_a_920_, lean_object* v_a_921_){
_start:
{
if (lean_obj_tag(v_a_920_) == 0)
{
lean_object* v___x_922_; 
lean_dec(v_f_919_);
v___x_922_ = l_List_reverse___redArg(v_a_921_);
return v___x_922_;
}
else
{
lean_object* v_head_923_; lean_object* v_tail_924_; lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_941_; 
v_head_923_ = lean_ctor_get(v_a_920_, 0);
v_tail_924_ = lean_ctor_get(v_a_920_, 1);
v_isSharedCheck_941_ = !lean_is_exclusive(v_a_920_);
if (v_isSharedCheck_941_ == 0)
{
v___x_926_ = v_a_920_;
v_isShared_927_ = v_isSharedCheck_941_;
goto v_resetjp_925_;
}
else
{
lean_inc(v_tail_924_);
lean_inc(v_head_923_);
lean_dec(v_a_920_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_941_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
lean_object* v___y_929_; lean_object* v_snd_934_; uint8_t v___x_935_; 
v_snd_934_ = lean_ctor_get(v_head_923_, 1);
v___x_935_ = lean_unbox(v_snd_934_);
if (v___x_935_ == 0)
{
lean_object* v_fst_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
v_fst_936_ = lean_ctor_get(v_head_923_, 0);
lean_inc(v_fst_936_);
lean_dec(v_head_923_);
lean_inc(v_f_919_);
v___x_937_ = lean_apply_1(v_f_919_, v_fst_936_);
v___x_938_ = lp_mathlib_FreeAddGroup_negRev___redArg(v___x_937_);
v___y_929_ = v___x_938_;
goto v___jp_928_;
}
else
{
lean_object* v_fst_939_; lean_object* v___x_940_; 
v_fst_939_ = lean_ctor_get(v_head_923_, 0);
lean_inc(v_fst_939_);
lean_dec(v_head_923_);
lean_inc(v_f_919_);
v___x_940_ = lean_apply_1(v_f_919_, v_fst_939_);
v___y_929_ = v___x_940_;
goto v___jp_928_;
}
v___jp_928_:
{
lean_object* v___x_931_; 
if (v_isShared_927_ == 0)
{
lean_ctor_set(v___x_926_, 1, v_a_921_);
lean_ctor_set(v___x_926_, 0, v___y_929_);
v___x_931_ = v___x_926_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v___y_929_);
lean_ctor_set(v_reuseFailAlloc_933_, 1, v_a_921_);
v___x_931_ = v_reuseFailAlloc_933_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
v_a_920_ = v_tail_924_;
v_a_921_ = v___x_931_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0___redArg(lean_object* v_f_942_, lean_object* v_L_943_){
_start:
{
lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; 
v___x_944_ = lean_box(0);
v___x_945_ = lp_mathlib_List_mapTR_loop___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__1___redArg(v_f_942_, v_L_943_, v___x_944_);
v___x_946_ = lp_mathlib_List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2___redArg(v___x_945_);
return v___x_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0(lean_object* v_00_u03b2_951_, lean_object* v_00_u03b1_952_){
_start:
{
lean_object* v___x_953_; 
v___x_953_ = ((lean_object*)(lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0___closed__1));
return v___x_953_;
}
}
static lean_object* _init_lp_mathlib_FreeAddGroup_instMonad___lam__4___closed__0(void){
_start:
{
lean_object* v___x_954_; 
v___x_954_ = lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0(lean_box(0), lean_box(0));
return v___x_954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__4(lean_object* v_00_u03b1_955_, lean_object* v_00_u03b2_956_, lean_object* v_f_957_, lean_object* v_x_958_){
_start:
{
lean_object* v___x_959_; lean_object* v_toFun_960_; lean_object* v___f_961_; lean_object* v___x_962_; 
v___x_959_ = lean_obj_once(&lp_mathlib_FreeAddGroup_instMonad___lam__4___closed__0, &lp_mathlib_FreeAddGroup_instMonad___lam__4___closed__0_once, _init_lp_mathlib_FreeAddGroup_instMonad___lam__4___closed__0);
v_toFun_960_ = lean_ctor_get(v___x_959_, 0);
v___f_961_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_instMonad___lam__3), 2, 1);
lean_closure_set(v___f_961_, 0, v_x_958_);
lean_inc(v_toFun_960_);
v___x_962_ = lean_apply_2(v_toFun_960_, v___f_961_, v_f_957_);
return v___x_962_;
}
}
static lean_object* _init_lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0(void){
_start:
{
lean_object* v___x_963_; 
v___x_963_ = lp_mathlib_FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0(lean_box(0), lean_box(0));
return v___x_963_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__5(lean_object* v_00___u03b1_964_, lean_object* v_00___u03b2_965_, lean_object* v_x_966_, lean_object* v_f_967_){
_start:
{
lean_object* v___x_968_; lean_object* v_toFun_969_; lean_object* v___x_970_; 
v___x_968_ = lean_obj_once(&lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0, &lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0_once, _init_lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0);
v_toFun_969_ = lean_ctor_get(v___x_968_, 0);
lean_inc(v_toFun_969_);
v___x_970_ = lean_apply_2(v_toFun_969_, v_f_967_, v_x_966_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__6(lean_object* v_a_971_, lean_object* v_x_972_){
_start:
{
lean_object* v___x_973_; 
v___x_973_ = lp_mathlib_FreeAddGroup_of___redArg(v_a_971_);
return v___x_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__6___boxed(lean_object* v_a_974_, lean_object* v_x_975_){
_start:
{
lean_object* v_res_976_; 
v_res_976_ = lp_mathlib_FreeAddGroup_instMonad___lam__6(v_a_974_, v_x_975_);
lean_dec(v_x_975_);
return v_res_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__7(lean_object* v_y_977_, lean_object* v___f_978_, lean_object* v_a_979_){
_start:
{
lean_object* v___f_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___f_980_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_instMonad___lam__6___boxed), 2, 1);
lean_closure_set(v___f_980_, 0, v_a_979_);
v___x_981_ = lean_box(0);
v___x_982_ = lean_apply_1(v_y_977_, v___x_981_);
v___x_983_ = lean_apply_4(v___f_978_, lean_box(0), lean_box(0), v___x_982_, v___f_980_);
return v___x_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__8(lean_object* v___f_984_, lean_object* v_00_u03b1_985_, lean_object* v_00_u03b2_986_, lean_object* v_x_987_, lean_object* v_y_988_){
_start:
{
lean_object* v___f_989_; lean_object* v___x_990_; 
lean_inc(v___f_984_);
v___f_989_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddGroup_instMonad___lam__7), 3, 2);
lean_closure_set(v___f_989_, 0, v_y_988_);
lean_closure_set(v___f_989_, 1, v___f_984_);
v___x_990_ = lean_apply_4(v___f_984_, lean_box(0), lean_box(0), v_x_987_, v___f_989_);
return v___x_990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_instMonad___lam__10(lean_object* v_00_u03b1_991_, lean_object* v_00_u03b2_992_, lean_object* v_x_993_, lean_object* v_y_994_){
_start:
{
lean_object* v___x_995_; lean_object* v_toFun_996_; lean_object* v___f_997_; lean_object* v___x_998_; 
v___x_995_ = lean_obj_once(&lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0, &lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0_once, _init_lp_mathlib_FreeAddGroup_instMonad___lam__5___closed__0);
v_toFun_996_ = lean_ctor_get(v___x_995_, 0);
v___f_997_ = lean_alloc_closure((void*)(lp_mathlib_FreeGroup_instMonad___lam__8___boxed), 2, 1);
lean_closure_set(v___f_997_, 0, v_y_994_);
lean_inc(v_toFun_996_);
v___x_998_ = lean_apply_2(v_toFun_996_, v___f_997_, v_x_993_);
return v___x_998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___redArg(lean_object* v_f_1020_){
_start:
{
lean_inc(v_f_1020_);
return v_f_1020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___redArg___boxed(lean_object* v_f_1021_){
_start:
{
lean_object* v_res_1022_; 
v_res_1022_ = lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___redArg(v_f_1021_);
lean_dec(v_f_1021_);
return v_res_1022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1(lean_object* v_00_u03b2_1023_, lean_object* v_00_u03b1_1024_, lean_object* v_f_1025_, lean_object* v_map__mul_1026_){
_start:
{
lean_inc(v_f_1025_);
return v_f_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1027_, lean_object* v_00_u03b1_1028_, lean_object* v_f_1029_, lean_object* v_map__mul_1030_){
_start:
{
lean_object* v_res_1031_; 
v_res_1031_ = lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__1(v_00_u03b2_1027_, v_00_u03b1_1028_, v_f_1029_, v_map__mul_1030_);
lean_dec(v_f_1029_);
return v_res_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0(lean_object* v_00_u03b2_1032_, lean_object* v_00_u03b1_1033_, lean_object* v_f_1034_, lean_object* v_L_1035_){
_start:
{
lean_object* v___x_1036_; 
v___x_1036_ = lp_mathlib_FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0___redArg(v_f_1034_, v_L_1035_);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_1037_, lean_object* v_00_u03b2_1038_, lean_object* v_f_1039_, lean_object* v_a_1040_, lean_object* v_a_1041_){
_start:
{
lean_object* v___x_1042_; 
v___x_1042_ = lp_mathlib_List_mapTR_loop___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__1___redArg(v_f_1039_, v_a_1040_, v_a_1041_);
return v___x_1042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_1043_, lean_object* v_l_1044_){
_start:
{
lean_object* v___x_1045_; 
v___x_1045_ = lp_mathlib_List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2___redArg(v_l_1044_);
return v___x_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4(lean_object* v_00_u03b2_1046_, lean_object* v_init_1047_, lean_object* v_x_1048_){
_start:
{
lean_object* v___x_1049_; 
v___x_1049_ = lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___redArg(v_init_1047_, v_x_1048_);
return v___x_1049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_00_u03b2_1050_, lean_object* v_init_1051_, lean_object* v_x_1052_){
_start:
{
lean_object* v_res_1053_; 
v_res_1053_ = lp_mathlib_List_foldr___at___00List_sum___at___00FreeAddGroup_Lift_aux___at___00FreeAddGroup_lift___at___00FreeAddGroup_instMonad_spec__0_spec__0_spec__2_spec__4(v_00_u03b2_1050_, v_init_1051_, v_x_1052_);
lean_dec(v_init_1051_);
return v_res_1053_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Chain(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2 = _init_lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2();
lean_mark_persistent(lp_mathlib_FreeGroup_prod___at___00FreeGroup_sum___at___00FreeGroup_freeGroupUnitEquivInt_spec__2_spec__2);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Chain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
