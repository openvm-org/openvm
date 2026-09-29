// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.SubMulAction
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Actions public import Mathlib.Algebra.Module.Defs public import Mathlib.Data.SetLike.Basic public import Mathlib.Data.Setoid.Basic public import Mathlib.GroupTheory.GroupAction.Defs public import Mathlib.GroupTheory.GroupAction.Hom
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
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSetLike___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSetLike___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_SubMulAction_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SubMulAction_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_SubAddAction_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SubAddAction_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMax___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SubMulAction_instMax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubMulAction_instMax___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubMulAction_instMax___closed__0 = (const lean_object*)&lp_mathlib_SubMulAction_instMax___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMax(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMax___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMax___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SubAddAction_instMax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubAddAction_instMax___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubAddAction_instMax___closed__0 = (const lean_object*)&lp_mathlib_SubAddAction_instMax___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMax(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMax___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSupSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_SubMulAction_instSupSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubMulAction_instSupSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubMulAction_instSupSet___closed__0 = (const lean_object*)&lp_mathlib_SubMulAction_instSupSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSupSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSupSet___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSupSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_SubAddAction_instSupSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubAddAction_instSupSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubAddAction_instSupSet___closed__0 = (const lean_object*)&lp_mathlib_SubAddAction_instSupSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSupSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSupSet___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInfSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInfSet___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInfSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInfSet___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubMulAction_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubAddAction_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instVAddSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instVAddSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SubMulAction_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubMulAction_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubMulAction_subtype___closed__0 = (const lean_object*)&lp_mathlib_SubMulAction_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_subtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_toMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_toMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_toMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_toAddAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_toAddAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_toAddAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_smul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_smul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_smul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_vadd_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_vadd_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_vadd_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompl___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_SubMulAction_instCompl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubMulAction_instCompl___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubMulAction_instCompl___closed__0 = (const lean_object*)&lp_mathlib_SubMulAction_instCompl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompl___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_SubAddAction_instCompl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubAddAction_instCompl___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubAddAction_instCompl___closed__0 = (const lean_object*)&lp_mathlib_SubAddAction_instCompl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_nonZeroSubMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_nonZeroSubMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubMulOfNormal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubMulOfNormal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubAddOfNormal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubAddOfNormal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_r_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_r_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul(lean_object* v_S_7_, lean_object* v_R_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_hS_12_, lean_object* v_s_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_inst_10_);
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul___boxed(lean_object* v_S_15_, lean_object* v_R_16_, lean_object* v_M_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_hS_20_, lean_object* v_s_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_SetLike_smul(v_S_15_, v_R_16_, v_M_17_, v_inst_18_, v_inst_19_, v_hS_20_, v_s_21_);
lean_dec(v_s_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd(lean_object* v_S_25_, lean_object* v_R_26_, lean_object* v_M_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_hS_30_, lean_object* v_s_31_){
_start:
{
lean_object* v___f_32_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_32_, 0, v_inst_28_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd___boxed(lean_object* v_S_33_, lean_object* v_R_34_, lean_object* v_M_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_hS_38_, lean_object* v_s_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_SetLike_vadd(v_S_33_, v_R_34_, v_M_35_, v_inst_36_, v_inst_37_, v_hS_38_, v_s_39_);
lean_dec(v_s_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul_x27___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul_x27(lean_object* v_S_43_, lean_object* v_M_44_, lean_object* v_N_45_, lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_s_54_){
_start:
{
lean_object* v___f_55_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_55_, 0, v_inst_49_);
return v___f_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_smul_x27___boxed(lean_object* v_S_56_, lean_object* v_M_57_, lean_object* v_N_58_, lean_object* v_00_u03b1_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_s_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_SetLike_smul_x27(v_S_56_, v_M_57_, v_N_58_, v_00_u03b1_59_, v_inst_60_, v_inst_61_, v_inst_62_, v_inst_63_, v_inst_64_, v_inst_65_, v_inst_66_, v_s_67_);
lean_dec(v_s_67_);
lean_dec(v_inst_64_);
lean_dec_ref(v_inst_63_);
lean_dec(v_inst_61_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd_x27___redArg(lean_object* v_inst_69_){
_start:
{
lean_object* v___f_70_; 
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_70_, 0, v_inst_69_);
return v___f_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd_x27(lean_object* v_S_71_, lean_object* v_M_72_, lean_object* v_N_73_, lean_object* v_00_u03b1_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_s_82_){
_start:
{
lean_object* v___f_83_; 
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_83_, 0, v_inst_77_);
return v___f_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_vadd_x27___boxed(lean_object* v_S_84_, lean_object* v_M_85_, lean_object* v_N_86_, lean_object* v_00_u03b1_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_s_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_SetLike_vadd_x27(v_S_84_, v_M_85_, v_N_86_, v_00_u03b1_87_, v_inst_88_, v_inst_89_, v_inst_90_, v_inst_91_, v_inst_92_, v_inst_93_, v_inst_94_, v_s_95_);
lean_dec(v_s_95_);
lean_dec(v_inst_92_);
lean_dec_ref(v_inst_91_);
lean_dec(v_inst_89_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSetLike(lean_object* v_R_97_, lean_object* v_M_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_box(0);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSetLike___boxed(lean_object* v_R_101_, lean_object* v_M_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_SubMulAction_instSetLike(v_R_101_, v_M_102_, v_inst_103_);
lean_dec(v_inst_103_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSetLike(lean_object* v_R_105_, lean_object* v_M_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lean_box(0);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSetLike___boxed(lean_object* v_R_109_, lean_object* v_M_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_SubAddAction_instSetLike(v_R_109_, v_M_110_, v_inst_111_);
lean_dec(v_inst_111_);
return v_res_112_;
}
}
static lean_object* _init_lp_mathlib_SubMulAction_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = lean_box(0);
v___x_114_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instPartialOrder(lean_object* v_R_115_, lean_object* v_M_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_obj_once(&lp_mathlib_SubMulAction_instPartialOrder___closed__0, &lp_mathlib_SubMulAction_instPartialOrder___closed__0_once, _init_lp_mathlib_SubMulAction_instPartialOrder___closed__0);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instPartialOrder___boxed(lean_object* v_R_119_, lean_object* v_M_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_SubMulAction_instPartialOrder(v_R_119_, v_M_120_, v_inst_121_);
lean_dec(v_inst_121_);
return v_res_122_;
}
}
static lean_object* _init_lp_mathlib_SubAddAction_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_123_ = lean_box(0);
v___x_124_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instPartialOrder(lean_object* v_R_125_, lean_object* v_M_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lean_obj_once(&lp_mathlib_SubAddAction_instPartialOrder___closed__0, &lp_mathlib_SubAddAction_instPartialOrder___closed__0_once, _init_lp_mathlib_SubAddAction_instPartialOrder___closed__0);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instPartialOrder___boxed(lean_object* v_R_129_, lean_object* v_M_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_SubAddAction_instPartialOrder(v_R_129_, v_M_130_, v_inst_131_);
lean_dec(v_inst_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_copy(lean_object* v_R_133_, lean_object* v_M_134_, lean_object* v_inst_135_, lean_object* v_p_136_, lean_object* v_s_137_, lean_object* v_hs_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lean_box(0);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_copy___boxed(lean_object* v_R_140_, lean_object* v_M_141_, lean_object* v_inst_142_, lean_object* v_p_143_, lean_object* v_s_144_, lean_object* v_hs_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_SubMulAction_copy(v_R_140_, v_M_141_, v_inst_142_, v_p_143_, v_s_144_, v_hs_145_);
lean_dec(v_inst_142_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_copy(lean_object* v_R_147_, lean_object* v_M_148_, lean_object* v_inst_149_, lean_object* v_p_150_, lean_object* v_s_151_, lean_object* v_hs_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_box(0);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_copy___boxed(lean_object* v_R_154_, lean_object* v_M_155_, lean_object* v_inst_156_, lean_object* v_p_157_, lean_object* v_s_158_, lean_object* v_hs_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_SubAddAction_copy(v_R_154_, v_M_155_, v_inst_156_, v_p_157_, v_s_158_, v_hs_159_);
lean_dec(v_inst_156_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instBot(lean_object* v_R_161_, lean_object* v_M_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lean_box(0);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instBot___boxed(lean_object* v_R_165_, lean_object* v_M_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_SubMulAction_instBot(v_R_165_, v_M_166_, v_inst_167_);
lean_dec(v_inst_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instBot(lean_object* v_R_169_, lean_object* v_M_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lean_box(0);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instBot___boxed(lean_object* v_R_173_, lean_object* v_M_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_SubAddAction_instBot(v_R_173_, v_M_174_, v_inst_175_);
lean_dec(v_inst_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInhabited(lean_object* v_R_177_, lean_object* v_M_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lean_box(0);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInhabited___boxed(lean_object* v_R_181_, lean_object* v_M_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_SubMulAction_instInhabited(v_R_181_, v_M_182_, v_inst_183_);
lean_dec(v_inst_183_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInhabited(lean_object* v_R_185_, lean_object* v_M_186_, lean_object* v_inst_187_){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lean_box(0);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInhabited___boxed(lean_object* v_R_189_, lean_object* v_M_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_SubAddAction_instInhabited(v_R_189_, v_M_190_, v_inst_191_);
lean_dec(v_inst_191_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instTop(lean_object* v_R_193_, lean_object* v_M_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lean_box(0);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instTop___boxed(lean_object* v_R_197_, lean_object* v_M_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_SubMulAction_instTop(v_R_197_, v_M_198_, v_inst_199_);
lean_dec(v_inst_199_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instTop(lean_object* v_R_201_, lean_object* v_M_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lean_box(0);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instTop___boxed(lean_object* v_R_205_, lean_object* v_M_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_SubAddAction_instTop(v_R_205_, v_M_206_, v_inst_207_);
lean_dec(v_inst_207_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMax___lam__0(lean_object* v_s_209_, lean_object* v_t_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lean_box(0);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMax(lean_object* v_R_213_, lean_object* v_M_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___f_216_; 
v___f_216_ = ((lean_object*)(lp_mathlib_SubMulAction_instMax___closed__0));
return v___f_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMax___boxed(lean_object* v_R_217_, lean_object* v_M_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_SubMulAction_instMax(v_R_217_, v_M_218_, v_inst_219_);
lean_dec(v_inst_219_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMax___lam__0(lean_object* v_s_221_, lean_object* v_t_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_box(0);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMax(lean_object* v_R_225_, lean_object* v_M_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___f_228_; 
v___f_228_ = ((lean_object*)(lp_mathlib_SubAddAction_instMax___closed__0));
return v___f_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMax___boxed(lean_object* v_R_229_, lean_object* v_M_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_SubAddAction_instMax(v_R_229_, v_M_230_, v_inst_231_);
lean_dec(v_inst_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMin(lean_object* v_R_233_, lean_object* v_M_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v___f_236_; 
v___f_236_ = ((lean_object*)(lp_mathlib_SubMulAction_instMax___closed__0));
return v___f_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instMin___boxed(lean_object* v_R_237_, lean_object* v_M_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_SubMulAction_instMin(v_R_237_, v_M_238_, v_inst_239_);
lean_dec(v_inst_239_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMin(lean_object* v_R_241_, lean_object* v_M_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v___f_244_; 
v___f_244_ = ((lean_object*)(lp_mathlib_SubAddAction_instMax___closed__0));
return v___f_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instMin___boxed(lean_object* v_R_245_, lean_object* v_M_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_SubAddAction_instMin(v_R_245_, v_M_246_, v_inst_247_);
lean_dec(v_inst_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSupSet___lam__0(lean_object* v_S_249_){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = lean_box(0);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSupSet(lean_object* v_R_252_, lean_object* v_M_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v___f_255_; 
v___f_255_ = ((lean_object*)(lp_mathlib_SubMulAction_instSupSet___closed__0));
return v___f_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSupSet___boxed(lean_object* v_R_256_, lean_object* v_M_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_SubMulAction_instSupSet(v_R_256_, v_M_257_, v_inst_258_);
lean_dec(v_inst_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSupSet___lam__0(lean_object* v_S_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_box(0);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSupSet(lean_object* v_R_263_, lean_object* v_M_264_, lean_object* v_inst_265_){
_start:
{
lean_object* v___f_266_; 
v___f_266_ = ((lean_object*)(lp_mathlib_SubAddAction_instSupSet___closed__0));
return v___f_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instSupSet___boxed(lean_object* v_R_267_, lean_object* v_M_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_SubAddAction_instSupSet(v_R_267_, v_M_268_, v_inst_269_);
lean_dec(v_inst_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInfSet(lean_object* v_R_271_, lean_object* v_M_272_, lean_object* v_inst_273_){
_start:
{
lean_object* v___f_274_; 
v___f_274_ = ((lean_object*)(lp_mathlib_SubMulAction_instSupSet___closed__0));
return v___f_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instInfSet___boxed(lean_object* v_R_275_, lean_object* v_M_276_, lean_object* v_inst_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_SubMulAction_instInfSet(v_R_275_, v_M_276_, v_inst_277_);
lean_dec(v_inst_277_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInfSet(lean_object* v_R_279_, lean_object* v_M_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___f_282_; 
v___f_282_ = ((lean_object*)(lp_mathlib_SubAddAction_instSupSet___closed__0));
return v___f_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instInfSet___boxed(lean_object* v_R_283_, lean_object* v_M_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_SubAddAction_instInfSet(v_R_283_, v_M_284_, v_inst_285_);
lean_dec(v_inst_285_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg___lam__0(lean_object* v_a_287_, lean_object* v_b_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lean_box(0);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg(lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; lean_object* v_toLE_295_; lean_object* v_toLT_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_309_; 
v___x_294_ = lp_mathlib_SubMulAction_instPartialOrder(lean_box(0), lean_box(0), v_inst_293_);
v_toLE_295_ = lean_ctor_get(v___x_294_, 0);
v_toLT_296_ = lean_ctor_get(v___x_294_, 1);
v_isSharedCheck_309_ = !lean_is_exclusive(v___x_294_);
if (v_isSharedCheck_309_ == 0)
{
v___x_298_ = v___x_294_;
v_isShared_299_ = v_isSharedCheck_309_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_toLT_296_);
lean_inc(v_toLE_295_);
lean_dec(v___x_294_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_309_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v___f_300_; lean_object* v___f_301_; lean_object* v___x_303_; 
v___f_300_ = ((lean_object*)(lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__0));
v___f_301_ = ((lean_object*)(lp_mathlib_SubMulAction_instSupSet___closed__0));
if (v_isShared_299_ == 0)
{
v___x_303_ = v___x_298_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v_toLE_295_);
lean_ctor_set(v_reuseFailAlloc_308_, 1, v_toLT_296_);
v___x_303_ = v_reuseFailAlloc_308_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v___x_303_);
lean_ctor_set(v___x_304_, 1, v___f_300_);
v___x_305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v___f_300_);
v___x_306_ = ((lean_object*)(lp_mathlib_SubMulAction_instCompleteLattice___redArg___closed__1));
v___x_307_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_307_, 0, v___x_305_);
lean_ctor_set(v___x_307_, 1, v___f_301_);
lean_ctor_set(v___x_307_, 2, v___f_301_);
lean_ctor_set(v___x_307_, 3, v___x_306_);
return v___x_307_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___redArg___boxed(lean_object* v_inst_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_SubMulAction_instCompleteLattice___redArg(v_inst_310_);
lean_dec(v_inst_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice(lean_object* v_R_312_, lean_object* v_M_313_, lean_object* v_inst_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_SubMulAction_instCompleteLattice___redArg(v_inst_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompleteLattice___boxed(lean_object* v_R_316_, lean_object* v_M_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_SubMulAction_instCompleteLattice(v_R_316_, v_M_317_, v_inst_318_);
lean_dec(v_inst_318_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg___lam__0(lean_object* v_a_320_, lean_object* v_b_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lean_box(0);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg(lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; lean_object* v_toLE_328_; lean_object* v_toLT_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_342_; 
v___x_327_ = lp_mathlib_SubAddAction_instPartialOrder(lean_box(0), lean_box(0), v_inst_326_);
v_toLE_328_ = lean_ctor_get(v___x_327_, 0);
v_toLT_329_ = lean_ctor_get(v___x_327_, 1);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_327_);
if (v_isSharedCheck_342_ == 0)
{
v___x_331_ = v___x_327_;
v_isShared_332_ = v_isSharedCheck_342_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_toLT_329_);
lean_inc(v_toLE_328_);
lean_dec(v___x_327_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_342_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___f_333_; lean_object* v___f_334_; lean_object* v___x_336_; 
v___f_333_ = ((lean_object*)(lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__0));
v___f_334_ = ((lean_object*)(lp_mathlib_SubAddAction_instSupSet___closed__0));
if (v_isShared_332_ == 0)
{
v___x_336_ = v___x_331_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_toLE_328_);
lean_ctor_set(v_reuseFailAlloc_341_, 1, v_toLT_329_);
v___x_336_ = v_reuseFailAlloc_341_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_336_);
lean_ctor_set(v___x_337_, 1, v___f_333_);
v___x_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v___f_333_);
v___x_339_ = ((lean_object*)(lp_mathlib_SubAddAction_instCompleteLattice___redArg___closed__1));
v___x_340_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_340_, 0, v___x_338_);
lean_ctor_set(v___x_340_, 1, v___f_334_);
lean_ctor_set(v___x_340_, 2, v___f_334_);
lean_ctor_set(v___x_340_, 3, v___x_339_);
return v___x_340_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___redArg___boxed(lean_object* v_inst_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_SubAddAction_instCompleteLattice___redArg(v_inst_343_);
lean_dec(v_inst_343_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice(lean_object* v_R_345_, lean_object* v_M_346_, lean_object* v_inst_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_SubAddAction_instCompleteLattice___redArg(v_inst_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompleteLattice___boxed(lean_object* v_R_349_, lean_object* v_M_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_SubAddAction_instCompleteLattice(v_R_349_, v_M_350_, v_inst_351_);
lean_dec(v_inst_351_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0(lean_object* v_inst_353_, lean_object* v_c_354_, lean_object* v_x_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lean_apply_2(v_inst_353_, v_c_354_, v_x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg(lean_object* v_inst_357_){
_start:
{
lean_object* v___f_358_; 
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_358_, 0, v_inst_357_);
return v___f_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instSMulSubtypeMem(lean_object* v_R_359_, lean_object* v_M_360_, lean_object* v_inst_361_, lean_object* v_p_362_){
_start:
{
lean_object* v___f_363_; 
v___f_363_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_363_, 0, v_inst_361_);
return v___f_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instVAddSubtypeMem___redArg(lean_object* v_inst_364_){
_start:
{
lean_object* v___f_365_; 
v___f_365_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_365_, 0, v_inst_364_);
return v___f_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instVAddSubtypeMem(lean_object* v_R_366_, lean_object* v_M_367_, lean_object* v_inst_368_, lean_object* v_p_369_){
_start:
{
lean_object* v___f_370_; 
v___f_370_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_370_, 0, v_inst_368_);
return v___f_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype___lam__0(lean_object* v_self_371_){
_start:
{
lean_inc(v_self_371_);
return v_self_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype___lam__0___boxed(lean_object* v_self_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_SubMulAction_subtype___lam__0(v_self_372_);
lean_dec(v_self_372_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype(lean_object* v_R_375_, lean_object* v_M_376_, lean_object* v_inst_377_, lean_object* v_p_378_){
_start:
{
lean_object* v___f_379_; 
v___f_379_ = ((lean_object*)(lp_mathlib_SubMulAction_subtype___closed__0));
return v___f_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_subtype___boxed(lean_object* v_R_380_, lean_object* v_M_381_, lean_object* v_inst_382_, lean_object* v_p_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_SubMulAction_subtype(v_R_380_, v_M_381_, v_inst_382_, v_p_383_);
lean_dec(v_inst_382_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_subtype(lean_object* v_R_385_, lean_object* v_M_386_, lean_object* v_inst_387_, lean_object* v_p_388_){
_start:
{
lean_object* v___f_389_; 
v___f_389_ = ((lean_object*)(lp_mathlib_SubMulAction_subtype___closed__0));
return v___f_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_subtype___boxed(lean_object* v_R_390_, lean_object* v_M_391_, lean_object* v_inst_392_, lean_object* v_p_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_SubAddAction_subtype(v_R_390_, v_M_391_, v_inst_392_, v_p_393_);
lean_dec(v_inst_392_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_toMulAction___redArg(lean_object* v_inst_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_396_, 0, v_inst_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_toMulAction(lean_object* v_R_397_, lean_object* v_M_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_A_401_, lean_object* v_inst_402_, lean_object* v_hA_403_, lean_object* v_S_x27_404_){
_start:
{
lean_object* v___f_405_; 
v___f_405_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_405_, 0, v_inst_400_);
return v___f_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_toMulAction___boxed(lean_object* v_R_406_, lean_object* v_M_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_A_410_, lean_object* v_inst_411_, lean_object* v_hA_412_, lean_object* v_S_x27_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_SubMulAction_SMulMemClass_toMulAction(v_R_406_, v_M_407_, v_inst_408_, v_inst_409_, v_A_410_, v_inst_411_, v_hA_412_, v_S_x27_413_);
lean_dec(v_S_x27_413_);
lean_dec_ref(v_inst_408_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_toAddAction___redArg(lean_object* v_inst_415_){
_start:
{
lean_object* v___f_416_; 
v___f_416_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_416_, 0, v_inst_415_);
return v___f_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_toAddAction(lean_object* v_R_417_, lean_object* v_M_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_A_421_, lean_object* v_inst_422_, lean_object* v_hA_423_, lean_object* v_S_x27_424_){
_start:
{
lean_object* v___f_425_; 
v___f_425_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_425_, 0, v_inst_420_);
return v___f_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_toAddAction___boxed(lean_object* v_R_426_, lean_object* v_M_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_A_430_, lean_object* v_inst_431_, lean_object* v_hA_432_, lean_object* v_S_x27_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_SubAddAction_SMulMemClass_toAddAction(v_R_426_, v_M_427_, v_inst_428_, v_inst_429_, v_A_430_, v_inst_431_, v_hA_432_, v_S_x27_433_);
lean_dec(v_S_x27_433_);
lean_dec_ref(v_inst_428_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_subtype(lean_object* v_R_435_, lean_object* v_M_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_A_439_, lean_object* v_inst_440_, lean_object* v_hA_441_, lean_object* v_S_x27_442_){
_start:
{
lean_object* v___f_443_; 
v___f_443_ = ((lean_object*)(lp_mathlib_SubMulAction_subtype___closed__0));
return v___f_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_SMulMemClass_subtype___boxed(lean_object* v_R_444_, lean_object* v_M_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_A_448_, lean_object* v_inst_449_, lean_object* v_hA_450_, lean_object* v_S_x27_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_mathlib_SubMulAction_SMulMemClass_subtype(v_R_444_, v_M_445_, v_inst_446_, v_inst_447_, v_A_448_, v_inst_449_, v_hA_450_, v_S_x27_451_);
lean_dec(v_S_x27_451_);
lean_dec(v_inst_447_);
lean_dec_ref(v_inst_446_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_subtype(lean_object* v_R_453_, lean_object* v_M_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_A_457_, lean_object* v_inst_458_, lean_object* v_hA_459_, lean_object* v_S_x27_460_){
_start:
{
lean_object* v___f_461_; 
v___f_461_ = ((lean_object*)(lp_mathlib_SubMulAction_subtype___closed__0));
return v___f_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_SMulMemClass_subtype___boxed(lean_object* v_R_462_, lean_object* v_M_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_A_466_, lean_object* v_inst_467_, lean_object* v_hA_468_, lean_object* v_S_x27_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_SubAddAction_SMulMemClass_subtype(v_R_462_, v_M_463_, v_inst_464_, v_inst_465_, v_A_466_, v_inst_467_, v_hA_468_, v_S_x27_469_);
lean_dec(v_S_x27_469_);
lean_dec(v_inst_465_);
lean_dec_ref(v_inst_464_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_smul_x27___redArg(lean_object* v_inst_471_){
_start:
{
lean_object* v___f_472_; 
v___f_472_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_472_, 0, v_inst_471_);
return v___f_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_smul_x27(lean_object* v_S_473_, lean_object* v_R_474_, lean_object* v_M_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_p_481_){
_start:
{
lean_object* v___f_482_; 
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_482_, 0, v_inst_479_);
return v___f_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_smul_x27___boxed(lean_object* v_S_483_, lean_object* v_R_484_, lean_object* v_M_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_p_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_SubMulAction_smul_x27(v_S_483_, v_R_484_, v_M_485_, v_inst_486_, v_inst_487_, v_inst_488_, v_inst_489_, v_inst_490_, v_p_491_);
lean_dec(v_inst_488_);
lean_dec(v_inst_487_);
lean_dec_ref(v_inst_486_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_vadd_x27___redArg(lean_object* v_inst_493_){
_start:
{
lean_object* v___f_494_; 
v___f_494_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_494_, 0, v_inst_493_);
return v___f_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_vadd_x27(lean_object* v_S_495_, lean_object* v_R_496_, lean_object* v_M_497_, lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_p_503_){
_start:
{
lean_object* v___f_504_; 
v___f_504_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_504_, 0, v_inst_501_);
return v___f_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_vadd_x27___boxed(lean_object* v_S_505_, lean_object* v_R_506_, lean_object* v_M_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_p_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_SubAddAction_vadd_x27(v_S_505_, v_R_506_, v_M_507_, v_inst_508_, v_inst_509_, v_inst_510_, v_inst_511_, v_inst_512_, v_p_513_);
lean_dec(v_inst_510_);
lean_dec(v_inst_509_);
lean_dec_ref(v_inst_508_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction_x27___redArg(lean_object* v_inst_515_){
_start:
{
lean_object* v___f_516_; 
v___f_516_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_516_, 0, v_inst_515_);
return v___f_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction_x27(lean_object* v_S_517_, lean_object* v_R_518_, lean_object* v_M_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_p_526_){
_start:
{
lean_object* v___f_527_; 
v___f_527_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_527_, 0, v_inst_524_);
return v___f_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction_x27___boxed(lean_object* v_S_528_, lean_object* v_R_529_, lean_object* v_M_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_p_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_SubMulAction_mulAction_x27(v_S_528_, v_R_529_, v_M_530_, v_inst_531_, v_inst_532_, v_inst_533_, v_inst_534_, v_inst_535_, v_inst_536_, v_p_537_);
lean_dec(v_inst_534_);
lean_dec_ref(v_inst_533_);
lean_dec(v_inst_532_);
lean_dec_ref(v_inst_531_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction_x27___redArg(lean_object* v_inst_539_){
_start:
{
lean_object* v___f_540_; 
v___f_540_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_540_, 0, v_inst_539_);
return v___f_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction_x27(lean_object* v_S_541_, lean_object* v_R_542_, lean_object* v_M_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_p_550_){
_start:
{
lean_object* v___f_551_; 
v___f_551_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_551_, 0, v_inst_548_);
return v___f_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction_x27___boxed(lean_object* v_S_552_, lean_object* v_R_553_, lean_object* v_M_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_inst_560_, lean_object* v_p_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_mathlib_SubAddAction_addAction_x27(v_S_552_, v_R_553_, v_M_554_, v_inst_555_, v_inst_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_inst_560_, v_p_561_);
lean_dec(v_inst_558_);
lean_dec_ref(v_inst_557_);
lean_dec(v_inst_556_);
lean_dec_ref(v_inst_555_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction___redArg(lean_object* v_inst_563_){
_start:
{
lean_object* v___f_564_; 
v___f_564_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_564_, 0, v_inst_563_);
return v___f_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction(lean_object* v_R_565_, lean_object* v_M_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_p_569_){
_start:
{
lean_object* v___f_570_; 
v___f_570_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_570_, 0, v_inst_568_);
return v___f_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_mulAction___boxed(lean_object* v_R_571_, lean_object* v_M_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_p_575_){
_start:
{
lean_object* v_res_576_; 
v_res_576_ = lp_mathlib_SubMulAction_mulAction(v_R_571_, v_M_572_, v_inst_573_, v_inst_574_, v_p_575_);
lean_dec_ref(v_inst_573_);
return v_res_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction___redArg(lean_object* v_inst_577_){
_start:
{
lean_object* v___f_578_; 
v___f_578_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_578_, 0, v_inst_577_);
return v___f_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction(lean_object* v_R_579_, lean_object* v_M_580_, lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_p_583_){
_start:
{
lean_object* v___f_584_; 
v___f_584_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_584_, 0, v_inst_582_);
return v___f_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_addAction___boxed(lean_object* v_R_585_, lean_object* v_M_586_, lean_object* v_inst_587_, lean_object* v_inst_588_, lean_object* v_p_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_SubAddAction_addAction(v_R_585_, v_M_586_, v_inst_587_, v_inst_588_, v_p_589_);
lean_dec_ref(v_inst_587_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompl___lam__0(lean_object* v_s_591_){
_start:
{
lean_object* v___x_592_; 
v___x_592_ = lean_box(0);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompl(lean_object* v_R_594_, lean_object* v_M_595_, lean_object* v_inst_596_, lean_object* v_inst_597_){
_start:
{
lean_object* v___f_598_; 
v___f_598_ = ((lean_object*)(lp_mathlib_SubMulAction_instCompl___closed__0));
return v___f_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instCompl___boxed(lean_object* v_R_599_, lean_object* v_M_600_, lean_object* v_inst_601_, lean_object* v_inst_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_mathlib_SubMulAction_instCompl(v_R_599_, v_M_600_, v_inst_601_, v_inst_602_);
lean_dec(v_inst_602_);
lean_dec_ref(v_inst_601_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompl___lam__0(lean_object* v_s_604_){
_start:
{
lean_object* v___x_605_; 
v___x_605_ = lean_box(0);
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompl(lean_object* v_R_607_, lean_object* v_M_608_, lean_object* v_inst_609_, lean_object* v_inst_610_){
_start:
{
lean_object* v___f_611_; 
v___f_611_ = ((lean_object*)(lp_mathlib_SubAddAction_instCompl___closed__0));
return v___f_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_instCompl___boxed(lean_object* v_R_612_, lean_object* v_M_613_, lean_object* v_inst_614_, lean_object* v_inst_615_){
_start:
{
lean_object* v_res_616_; 
v_res_616_ = lp_mathlib_SubAddAction_instCompl(v_R_612_, v_M_613_, v_inst_614_, v_inst_615_);
lean_dec(v_inst_615_);
lean_dec_ref(v_inst_614_);
return v_res_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___redArg(lean_object* v_inst_617_){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v_toZero_620_; 
v___x_618_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_617_);
v___x_619_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_618_);
v_toZero_620_ = lean_ctor_get(v___x_619_, 0);
lean_inc(v_toZero_620_);
lean_dec_ref(v___x_619_);
return v_toZero_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___redArg___boxed(lean_object* v_inst_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___redArg(v_inst_621_);
lean_dec_ref(v_inst_621_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty(lean_object* v_R_623_, lean_object* v_M_624_, lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_p_628_, lean_object* v_n__empty_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___redArg(v_inst_626_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty___boxed(lean_object* v_R_631_, lean_object* v_M_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_p_636_, lean_object* v_n__empty_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_mathlib_SubMulAction_instZeroSubtypeMemOfNonempty(v_R_631_, v_M_632_, v_inst_633_, v_inst_634_, v_inst_635_, v_p_636_, v_n__empty_637_);
lean_dec(v_inst_635_);
lean_dec_ref(v_inst_634_);
lean_dec_ref(v_inst_633_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___redArg___lam__0(lean_object* v_toNeg_639_, lean_object* v_x_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lean_apply_1(v_toNeg_639_, v_x_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___redArg(lean_object* v_inst_642_){
_start:
{
lean_object* v___x_643_; lean_object* v_toNeg_644_; lean_object* v___f_645_; 
v___x_643_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_642_);
v_toNeg_644_ = lean_ctor_get(v___x_643_, 1);
lean_inc(v_toNeg_644_);
lean_dec_ref(v___x_643_);
v___f_645_ = lean_alloc_closure((void*)(lp_mathlib_SubMulAction_instNegSubtypeMem___redArg___lam__0), 2, 1);
lean_closure_set(v___f_645_, 0, v_toNeg_644_);
return v___f_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___redArg___boxed(lean_object* v_inst_646_){
_start:
{
lean_object* v_res_647_; 
v_res_647_ = lp_mathlib_SubMulAction_instNegSubtypeMem___redArg(v_inst_646_);
lean_dec_ref(v_inst_646_);
return v_res_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem(lean_object* v_R_648_, lean_object* v_M_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_p_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_SubMulAction_instNegSubtypeMem___redArg(v_inst_651_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_instNegSubtypeMem___boxed(lean_object* v_R_655_, lean_object* v_M_656_, lean_object* v_inst_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_p_660_){
_start:
{
lean_object* v_res_661_; 
v_res_661_ = lp_mathlib_SubMulAction_instNegSubtypeMem(v_R_655_, v_M_656_, v_inst_657_, v_inst_658_, v_inst_659_, v_p_660_);
lean_dec(v_inst_659_);
lean_dec_ref(v_inst_658_);
lean_dec_ref(v_inst_657_);
return v_res_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_inclusion(lean_object* v_M_662_, lean_object* v_00_u03b1_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_s_666_){
_start:
{
lean_object* v___f_667_; 
v___f_667_ = ((lean_object*)(lp_mathlib_SubMulAction_subtype___closed__0));
return v___f_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubMulAction_inclusion___boxed(lean_object* v_M_668_, lean_object* v_00_u03b1_669_, lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_s_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_SubMulAction_inclusion(v_M_668_, v_00_u03b1_669_, v_inst_670_, v_inst_671_, v_s_672_);
lean_dec(v_inst_671_);
lean_dec_ref(v_inst_670_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_inclusion(lean_object* v_M_674_, lean_object* v_00_u03b1_675_, lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_s_678_){
_start:
{
lean_object* v___f_679_; 
v___f_679_ = ((lean_object*)(lp_mathlib_SubMulAction_subtype___closed__0));
return v___f_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubAddAction_inclusion___boxed(lean_object* v_M_680_, lean_object* v_00_u03b1_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_s_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_mathlib_SubAddAction_inclusion(v_M_680_, v_00_u03b1_681_, v_inst_682_, v_inst_683_, v_s_684_);
lean_dec(v_inst_683_);
lean_dec_ref(v_inst_682_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_nonZeroSubMul(lean_object* v_R_686_, lean_object* v_M_687_, lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_inst_690_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = lean_box(0);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_nonZeroSubMul___boxed(lean_object* v_R_692_, lean_object* v_M_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_inst_696_){
_start:
{
lean_object* v_res_697_; 
v_res_697_ = lp_mathlib_Units_nonZeroSubMul(v_R_692_, v_M_693_, v_inst_694_, v_inst_695_, v_inst_696_);
lean_dec(v_inst_696_);
lean_dec_ref(v_inst_695_);
lean_dec_ref(v_inst_694_);
return v_res_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1___redArg(lean_object* v_inst_698_, lean_object* v_c_699_, lean_object* v_x_700_){
_start:
{
lean_object* v_val_701_; lean_object* v___x_702_; 
v_val_701_ = lean_ctor_get(v_c_699_, 0);
lean_inc(v_val_701_);
lean_dec_ref(v_c_699_);
v___x_702_ = lean_apply_2(v_inst_698_, v_val_701_, v_x_700_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1(lean_object* v_R_703_, lean_object* v_M_704_, lean_object* v_inst_705_, lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_c_708_, lean_object* v_x_709_){
_start:
{
lean_object* v_val_710_; lean_object* v___x_711_; 
v_val_710_ = lean_ctor_get(v_c_708_, 0);
lean_inc(v_val_710_);
lean_dec_ref(v_c_708_);
v___x_711_ = lean_apply_2(v_inst_707_, v_val_710_, v_x_709_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1___boxed(lean_object* v_R_712_, lean_object* v_M_713_, lean_object* v_inst_714_, lean_object* v_inst_715_, lean_object* v_inst_716_, lean_object* v_c_717_, lean_object* v_x_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1(v_R_712_, v_M_713_, v_inst_714_, v_inst_715_, v_inst_716_, v_c_717_, v_x_718_);
lean_dec_ref(v_inst_715_);
lean_dec_ref(v_inst_714_);
return v_res_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat___redArg(lean_object* v_inst_720_, lean_object* v_inst_721_, lean_object* v_inst_722_){
_start:
{
lean_object* v___x_723_; 
v___x_723_ = lean_alloc_closure((void*)(lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1___boxed), 7, 5);
lean_closure_set(v___x_723_, 0, lean_box(0));
lean_closure_set(v___x_723_, 1, lean_box(0));
lean_closure_set(v___x_723_, 2, v_inst_720_);
lean_closure_set(v___x_723_, 3, v_inst_721_);
lean_closure_set(v___x_723_, 4, v_inst_722_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_instMulActionSubtypeNeOfNat(lean_object* v_R_724_, lean_object* v_M_725_, lean_object* v_inst_726_, lean_object* v_inst_727_, lean_object* v_inst_728_){
_start:
{
lean_object* v___x_729_; 
v___x_729_ = lean_alloc_closure((void*)(lp_mathlib_Units_instMulActionSubtypeNeOfNat___aux__1___boxed), 7, 5);
lean_closure_set(v___x_729_, 0, lean_box(0));
lean_closure_set(v___x_729_, 1, lean_box(0));
lean_closure_set(v___x_729_, 2, v_inst_726_);
lean_closure_set(v___x_729_, 3, v_inst_727_);
lean_closure_set(v___x_729_, 4, v_inst_728_);
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubMulOfNormal(lean_object* v_G_730_, lean_object* v_inst_731_, lean_object* v_00_u03b1_732_, lean_object* v_inst_733_, lean_object* v_H_734_, lean_object* v_hH_735_){
_start:
{
lean_object* v___x_736_; 
v___x_736_ = lean_box(0);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubMulOfNormal___boxed(lean_object* v_G_737_, lean_object* v_inst_738_, lean_object* v_00_u03b1_739_, lean_object* v_inst_740_, lean_object* v_H_741_, lean_object* v_hH_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_mathlib_fixedPointsSubMulOfNormal(v_G_737_, v_inst_738_, v_00_u03b1_739_, v_inst_740_, v_H_741_, v_hH_742_);
lean_dec(v_inst_740_);
lean_dec_ref(v_inst_738_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubAddOfNormal(lean_object* v_G_744_, lean_object* v_inst_745_, lean_object* v_00_u03b1_746_, lean_object* v_inst_747_, lean_object* v_H_748_, lean_object* v_hH_749_){
_start:
{
lean_object* v___x_750_; 
v___x_750_ = lean_box(0);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fixedPointsSubAddOfNormal___boxed(lean_object* v_G_751_, lean_object* v_inst_752_, lean_object* v_00_u03b1_753_, lean_object* v_inst_754_, lean_object* v_H_755_, lean_object* v_hH_756_){
_start:
{
lean_object* v_res_757_; 
v_res_757_ = lp_mathlib_fixedPointsSubAddOfNormal(v_G_751_, v_inst_752_, v_00_u03b1_753_, v_inst_754_, v_H_755_, v_hH_756_);
lean_dec(v_inst_754_);
lean_dec_ref(v_inst_752_);
return v_res_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1___redArg(lean_object* v_inst_758_, lean_object* v_c_759_, lean_object* v_x_760_){
_start:
{
lean_object* v___x_761_; 
v___x_761_ = lean_apply_2(v_inst_758_, v_c_759_, v_x_760_);
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1(lean_object* v_G_762_, lean_object* v_inst_763_, lean_object* v_00_u03b1_764_, lean_object* v_inst_765_, lean_object* v_H_766_, lean_object* v_hH_767_, lean_object* v_c_768_, lean_object* v_x_769_){
_start:
{
lean_object* v___x_770_; 
v___x_770_ = lean_apply_2(v_inst_765_, v_c_768_, v_x_769_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1___boxed(lean_object* v_G_771_, lean_object* v_inst_772_, lean_object* v_00_u03b1_773_, lean_object* v_inst_774_, lean_object* v_H_775_, lean_object* v_hH_776_, lean_object* v_c_777_, lean_object* v_x_778_){
_start:
{
lean_object* v_res_779_; 
v_res_779_ = lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1(v_G_771_, v_inst_772_, v_00_u03b1_773_, v_inst_774_, v_H_775_, v_hH_776_, v_c_777_, v_x_778_);
lean_dec_ref(v_inst_772_);
return v_res_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___redArg(lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_H_782_){
_start:
{
lean_object* v___x_783_; 
v___x_783_ = lean_alloc_closure((void*)(lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1___boxed), 8, 6);
lean_closure_set(v___x_783_, 0, lean_box(0));
lean_closure_set(v___x_783_, 1, v_inst_780_);
lean_closure_set(v___x_783_, 2, lean_box(0));
lean_closure_set(v___x_783_, 3, v_inst_781_);
lean_closure_set(v___x_783_, 4, v_H_782_);
lean_closure_set(v___x_783_, 5, lean_box(0));
return v___x_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal(lean_object* v_G_784_, lean_object* v_inst_785_, lean_object* v_00_u03b1_786_, lean_object* v_inst_787_, lean_object* v_H_788_, lean_object* v_hH_789_){
_start:
{
lean_object* v___x_790_; 
v___x_790_ = lean_alloc_closure((void*)(lp_mathlib_instMulActionElemFixedPointsSubtypeMemSubgroupOfNormal___aux__1___boxed), 8, 6);
lean_closure_set(v___x_790_, 0, lean_box(0));
lean_closure_set(v___x_790_, 1, v_inst_785_);
lean_closure_set(v___x_790_, 2, lean_box(0));
lean_closure_set(v___x_790_, 3, v_inst_787_);
lean_closure_set(v___x_790_, 4, v_H_788_);
lean_closure_set(v___x_790_, 5, lean_box(0));
return v___x_790_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
}
#ifdef __cplusplus
}
#endif
