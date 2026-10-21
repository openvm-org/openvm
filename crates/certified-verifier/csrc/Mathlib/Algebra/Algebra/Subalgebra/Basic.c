// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Subalgebra.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Equiv public import Mathlib.Algebra.Algebra.NonUnitalSubalgebra public import Mathlib.Algebra.Module.Submodule.EqLocus public import Mathlib.RingTheory.SimpleRing.Basic
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
lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_SetLike_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Set_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemiring_toSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_ofSubsemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSetLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSetLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subalgebra_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subalgebra_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAddSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAddSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instInhabitedSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instInhabitedSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instInhabitedSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmodule___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Subalgebra_toSubmodule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subalgebra_toSubmodule___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subalgebra_toSubmodule___closed__0 = (const lean_object*)&lp_mathlib_Subalgebra_toSubmodule___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subalgebra_val___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subalgebra_val___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subalgebra_val___closed__0 = (const lean_object*)&lp_mathlib_Subalgebra_val___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_val(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_val___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_ofLeftInverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_ofLeftInverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_subalgebraMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_subalgebraMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_subalgebraMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subalgebra_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_inclusion___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Subalgebra_inclusion___closed__0 = (const lean_object*)&lp_mathlib_Subalgebra_inclusion___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subalgebra_equivOfEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subalgebra_equivOfEq___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subalgebra_equivOfEq___closed__0 = (const lean_object*)&lp_mathlib_Subalgebra_equivOfEq___closed__0_value;
static const lean_ctor_object lp_mathlib_Subalgebra_equivOfEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subalgebra_equivOfEq___closed__0_value),((lean_object*)&lp_mathlib_Subalgebra_equivOfEq___closed__0_value)}};
static const lean_object* lp_mathlib_Subalgebra_equivOfEq___closed__1 = (const lean_object*)&lp_mathlib_Subalgebra_equivOfEq___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_subalgebraMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_subalgebraMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_subalgebraMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_moduleLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_moduleLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_moduleLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_center(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_center___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_centralizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_centralizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubsemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubsemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubsemiring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_equalizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_equalizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSetLike(lean_object* v_R_1_, lean_object* v_A_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSetLike___boxed(lean_object* v_R_7_, lean_object* v_A_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Subalgebra_instSetLike(v_R_7_, v_A_8_, v_inst_9_, v_inst_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
return v_res_12_;
}
}
static lean_object* _init_lp_mathlib_Subalgebra_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_box(0);
v___x_14_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instPartialOrder(lean_object* v_R_15_, lean_object* v_A_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_obj_once(&lp_mathlib_Subalgebra_instPartialOrder___closed__0, &lp_mathlib_Subalgebra_instPartialOrder___closed__0_once, _init_lp_mathlib_Subalgebra_instPartialOrder___closed__0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instPartialOrder___boxed(lean_object* v_R_21_, lean_object* v_A_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Subalgebra_instPartialOrder(v_R_21_, v_A_22_, v_inst_23_, v_inst_24_, v_inst_25_);
lean_dec_ref(v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_ofClass(lean_object* v_S_27_, lean_object* v_R_28_, lean_object* v_A_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_s_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lean_box(0);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_ofClass___boxed(lean_object* v_S_38_, lean_object* v_R_39_, lean_object* v_A_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_s_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Subalgebra_ofClass(v_S_38_, v_R_39_, v_A_40_, v_inst_41_, v_inst_42_, v_inst_43_, v_inst_44_, v_inst_45_, v_inst_46_, v_s_47_);
lean_dec(v_s_47_);
lean_dec_ref(v_inst_43_);
lean_dec_ref(v_inst_42_);
lean_dec_ref(v_inst_41_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_copy(lean_object* v_R_49_, lean_object* v_A_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_S_54_, lean_object* v_s_55_, lean_object* v_hs_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_box(0);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_copy___boxed(lean_object* v_R_58_, lean_object* v_A_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_S_63_, lean_object* v_s_64_, lean_object* v_hs_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Subalgebra_copy(v_R_58_, v_A_59_, v_inst_60_, v_inst_61_, v_inst_62_, v_S_63_, v_s_64_, v_hs_65_);
lean_dec_ref(v_inst_62_);
lean_dec_ref(v_inst_61_);
lean_dec_ref(v_inst_60_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebra(lean_object* v_R_67_, lean_object* v_A_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_S_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_box(0);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebra___boxed(lean_object* v_R_74_, lean_object* v_A_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_S_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_Subalgebra_toNonUnitalSubalgebra(v_R_74_, v_A_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_S_79_);
lean_dec_ref(v_inst_78_);
lean_dec_ref(v_inst_77_);
lean_dec_ref(v_inst_76_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAddSubmonoid(lean_object* v_R_81_, lean_object* v_A_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_S_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lean_box(0);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAddSubmonoid___boxed(lean_object* v_R_88_, lean_object* v_A_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_S_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Subalgebra_toAddSubmonoid(v_R_88_, v_A_89_, v_inst_90_, v_inst_91_, v_inst_92_, v_S_93_);
lean_dec_ref(v_inst_92_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubring___redArg(lean_object* v_S_95_){
_start:
{
return v_S_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubring(lean_object* v_R_96_, lean_object* v_A_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_S_101_){
_start:
{
return v_S_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubring___boxed(lean_object* v_R_102_, lean_object* v_A_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_S_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Subalgebra_toSubring(v_R_102_, v_A_103_, v_inst_104_, v_inst_105_, v_inst_106_, v_S_107_);
lean_dec_ref(v_inst_106_);
lean_dec_ref(v_inst_105_);
lean_dec_ref(v_inst_104_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instInhabitedSubtypeMem___redArg(lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; lean_object* v_toZero_111_; 
v___x_110_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_109_);
v_toZero_111_ = lean_ctor_get(v___x_110_, 1);
lean_inc(v_toZero_111_);
lean_dec_ref(v___x_110_);
return v_toZero_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instInhabitedSubtypeMem(lean_object* v_R_112_, lean_object* v_A_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_S_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_Subalgebra_instInhabitedSubtypeMem___redArg(v_inst_115_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instInhabitedSubtypeMem___boxed(lean_object* v_R_119_, lean_object* v_A_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_S_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Subalgebra_instInhabitedSubtypeMem(v_R_119_, v_A_120_, v_inst_121_, v_inst_122_, v_inst_123_, v_S_124_);
lean_dec_ref(v_inst_123_);
lean_dec_ref(v_inst_121_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSemiring___redArg(lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSemiring(lean_object* v_R_128_, lean_object* v_A_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_S_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_131_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSemiring___boxed(lean_object* v_R_135_, lean_object* v_A_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_S_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Subalgebra_toSemiring(v_R_135_, v_A_136_, v_inst_137_, v_inst_138_, v_inst_139_, v_S_140_);
lean_dec_ref(v_inst_139_);
lean_dec_ref(v_inst_137_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommSemiring___redArg(lean_object* v_inst_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommSemiring(lean_object* v_R_144_, lean_object* v_A_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_S_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_147_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommSemiring___boxed(lean_object* v_R_151_, lean_object* v_A_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_S_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Subalgebra_toCommSemiring(v_R_151_, v_A_152_, v_inst_153_, v_inst_154_, v_inst_155_, v_S_156_);
lean_dec_ref(v_inst_155_);
lean_dec_ref(v_inst_153_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toRing___redArg(lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toRing(lean_object* v_R_160_, lean_object* v_A_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_S_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_163_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toRing___boxed(lean_object* v_R_167_, lean_object* v_A_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_S_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_Subalgebra_toRing(v_R_167_, v_A_168_, v_inst_169_, v_inst_170_, v_inst_171_, v_S_172_);
lean_dec_ref(v_inst_171_);
lean_dec_ref(v_inst_169_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommRing___redArg(lean_object* v_inst_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommRing(lean_object* v_R_176_, lean_object* v_A_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_S_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_179_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toCommRing___boxed(lean_object* v_R_183_, lean_object* v_A_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_S_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Subalgebra_toCommRing(v_R_183_, v_A_184_, v_inst_185_, v_inst_186_, v_inst_187_, v_S_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_185_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmodule___lam__0(lean_object* v_S_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lean_box(0);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmodule(lean_object* v_R_193_, lean_object* v_A_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v___f_198_; 
v___f_198_ = ((lean_object*)(lp_mathlib_Subalgebra_toSubmodule___closed__0));
return v___f_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmodule___boxed(lean_object* v_R_199_, lean_object* v_A_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Subalgebra_toSubmodule(v_R_199_, v_A_200_, v_inst_201_, v_inst_202_, v_inst_203_);
lean_dec_ref(v_inst_203_);
lean_dec_ref(v_inst_202_);
lean_dec_ref(v_inst_201_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra_x27___redArg(lean_object* v_inst_205_){
_start:
{
lean_object* v_toSMul_206_; lean_object* v_algebraMap_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_216_; 
v_toSMul_206_ = lean_ctor_get(v_inst_205_, 0);
v_algebraMap_207_ = lean_ctor_get(v_inst_205_, 1);
v_isSharedCheck_216_ = !lean_is_exclusive(v_inst_205_);
if (v_isSharedCheck_216_ == 0)
{
v___x_209_ = v_inst_205_;
v_isShared_210_ = v_isSharedCheck_216_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_algebraMap_207_);
lean_inc(v_toSMul_206_);
lean_dec(v_inst_205_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_216_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___f_211_; lean_object* v___f_212_; lean_object* v___x_214_; 
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_211_, 0, v_toSMul_206_);
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_212_, 0, v_algebraMap_207_);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 1, v___f_212_);
lean_ctor_set(v___x_209_, 0, v___f_211_);
v___x_214_ = v___x_209_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v___f_211_);
lean_ctor_set(v_reuseFailAlloc_215_, 1, v___f_212_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra_x27(lean_object* v_R_x27_217_, lean_object* v_R_218_, lean_object* v_A_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_S_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_Subalgebra_algebra_x27___redArg(v_inst_226_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra_x27___boxed(lean_object* v_R_x27_229_, lean_object* v_R_230_, lean_object* v_A_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_S_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_Subalgebra_algebra_x27(v_R_x27_229_, v_R_230_, v_A_231_, v_inst_232_, v_inst_233_, v_inst_234_, v_S_235_, v_inst_236_, v_inst_237_, v_inst_238_, v_inst_239_);
lean_dec(v_inst_237_);
lean_dec_ref(v_inst_236_);
lean_dec_ref(v_inst_234_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra___redArg(lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_Subalgebra_algebra_x27___redArg(v_inst_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra(lean_object* v_R_243_, lean_object* v_A_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_S_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_Subalgebra_algebra_x27___redArg(v_inst_247_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_algebra___boxed(lean_object* v_R_250_, lean_object* v_A_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_S_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Subalgebra_algebra(v_R_250_, v_A_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_S_255_);
lean_dec_ref(v_inst_253_);
lean_dec_ref(v_inst_252_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val___lam__0(lean_object* v_self_257_){
_start:
{
lean_inc(v_self_257_);
return v_self_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val___lam__0___boxed(lean_object* v_self_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_Subalgebra_val___lam__0(v_self_258_);
lean_dec(v_self_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val(lean_object* v_R_261_, lean_object* v_A_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_S_266_){
_start:
{
lean_object* v___f_267_; 
v___f_267_ = ((lean_object*)(lp_mathlib_Subalgebra_val___closed__0));
return v___f_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_val___boxed(lean_object* v_R_268_, lean_object* v_A_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_S_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_Subalgebra_val(v_R_268_, v_A_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_S_273_);
lean_dec_ref(v_inst_272_);
lean_dec_ref(v_inst_271_);
lean_dec_ref(v_inst_270_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv___redArg(lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_){
_start:
{
lean_object* v_toAddCommMonoid_278_; lean_object* v_toSMul_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v_toAddCommMonoid_278_ = lean_ctor_get(v_inst_276_, 0);
v_toSMul_279_ = lean_ctor_get(v_inst_277_, 0);
v___x_280_ = lean_box(0);
v___x_281_ = lp_mathlib_LinearEquiv_ofEq(lean_box(0), lean_box(0), v_inst_275_, v_toAddCommMonoid_278_, v_toSMul_279_, v___x_280_, v___x_280_, lean_box(0));
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv___redArg___boxed(lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_Subalgebra_toSubmoduleEquiv___redArg(v_inst_282_, v_inst_283_, v_inst_284_);
lean_dec_ref(v_inst_284_);
lean_dec_ref(v_inst_283_);
lean_dec_ref(v_inst_282_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv(lean_object* v_R_286_, lean_object* v_A_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_S_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Subalgebra_toSubmoduleEquiv___redArg(v_inst_288_, v_inst_289_, v_inst_290_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toSubmoduleEquiv___boxed(lean_object* v_R_293_, lean_object* v_A_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_S_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Subalgebra_toSubmoduleEquiv(v_R_293_, v_A_294_, v_inst_295_, v_inst_296_, v_inst_297_, v_S_298_);
lean_dec_ref(v_inst_297_);
lean_dec_ref(v_inst_296_);
lean_dec_ref(v_inst_295_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_map(lean_object* v_R_300_, lean_object* v_A_301_, lean_object* v_B_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_f_308_, lean_object* v_S_309_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lean_box(0);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_map___boxed(lean_object* v_R_311_, lean_object* v_A_312_, lean_object* v_B_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_f_319_, lean_object* v_S_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_Subalgebra_map(v_R_311_, v_A_312_, v_B_313_, v_inst_314_, v_inst_315_, v_inst_316_, v_inst_317_, v_inst_318_, v_f_319_, v_S_320_);
lean_dec(v_f_319_);
lean_dec_ref(v_inst_318_);
lean_dec_ref(v_inst_317_);
lean_dec_ref(v_inst_316_);
lean_dec_ref(v_inst_315_);
lean_dec_ref(v_inst_314_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_comap(lean_object* v_R_322_, lean_object* v_A_323_, lean_object* v_B_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_f_330_, lean_object* v_S_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lean_box(0);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_comap___boxed(lean_object* v_R_333_, lean_object* v_A_334_, lean_object* v_B_335_, lean_object* v_inst_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_f_341_, lean_object* v_S_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_Subalgebra_comap(v_R_333_, v_A_334_, v_B_335_, v_inst_336_, v_inst_337_, v_inst_338_, v_inst_339_, v_inst_340_, v_f_341_, v_S_342_);
lean_dec(v_f_341_);
lean_dec_ref(v_inst_340_);
lean_dec_ref(v_inst_339_);
lean_dec_ref(v_inst_338_);
lean_dec_ref(v_inst_337_);
lean_dec_ref(v_inst_336_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra___redArg___lam__0(lean_object* v_algebraMap_344_, lean_object* v_r_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lean_apply_1(v_algebraMap_344_, v_r_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra___redArg(lean_object* v_inst_347_){
_start:
{
lean_object* v_toSMul_348_; lean_object* v_algebraMap_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_358_; 
v_toSMul_348_ = lean_ctor_get(v_inst_347_, 0);
v_algebraMap_349_ = lean_ctor_get(v_inst_347_, 1);
v_isSharedCheck_358_ = !lean_is_exclusive(v_inst_347_);
if (v_isSharedCheck_358_ == 0)
{
v___x_351_ = v_inst_347_;
v_isShared_352_ = v_isSharedCheck_358_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_algebraMap_349_);
lean_inc(v_toSMul_348_);
lean_dec(v_inst_347_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_358_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
lean_object* v___f_353_; lean_object* v___f_354_; lean_object* v___x_356_; 
v___f_353_ = lean_alloc_closure((void*)(lp_mathlib_SubalgebraClass_toAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_353_, 0, v_algebraMap_349_);
v___f_354_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_354_, 0, v_toSMul_348_);
if (v_isShared_352_ == 0)
{
lean_ctor_set(v___x_351_, 1, v___f_353_);
lean_ctor_set(v___x_351_, 0, v___f_354_);
v___x_356_ = v___x_351_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v___f_354_);
lean_ctor_set(v_reuseFailAlloc_357_, 1, v___f_353_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra(lean_object* v_S_359_, lean_object* v_R_360_, lean_object* v_A_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_hSR_367_, lean_object* v_s_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lp_mathlib_SubalgebraClass_toAlgebra___redArg(v_inst_364_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_toAlgebra___boxed(lean_object* v_S_370_, lean_object* v_R_371_, lean_object* v_A_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_hSR_378_, lean_object* v_s_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_mathlib_SubalgebraClass_toAlgebra(v_S_370_, v_R_371_, v_A_372_, v_inst_373_, v_inst_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_hSR_378_, v_s_379_);
lean_dec(v_s_379_);
lean_dec_ref(v_inst_374_);
lean_dec_ref(v_inst_373_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_val(lean_object* v_S_381_, lean_object* v_R_382_, lean_object* v_A_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_hSR_389_, lean_object* v_s_390_){
_start:
{
lean_object* v___f_391_; 
v___f_391_ = ((lean_object*)(lp_mathlib_Subalgebra_val___closed__0));
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubalgebraClass_val___boxed(lean_object* v_S_392_, lean_object* v_R_393_, lean_object* v_A_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_hSR_400_, lean_object* v_s_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_mathlib_SubalgebraClass_val(v_S_392_, v_R_393_, v_A_394_, v_inst_395_, v_inst_396_, v_inst_397_, v_inst_398_, v_inst_399_, v_hSR_400_, v_s_401_);
lean_dec(v_s_401_);
lean_dec_ref(v_inst_397_);
lean_dec_ref(v_inst_396_);
lean_dec_ref(v_inst_395_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubalgebra(lean_object* v_R_403_, lean_object* v_A_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_p_408_, lean_object* v_h__one_409_, lean_object* v_h__mul_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lean_box(0);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubalgebra___boxed(lean_object* v_R_412_, lean_object* v_A_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_p_417_, lean_object* v_h__one_418_, lean_object* v_h__mul_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_Submodule_toSubalgebra(v_R_412_, v_A_413_, v_inst_414_, v_inst_415_, v_inst_416_, v_p_417_, v_h__one_418_, v_h__mul_419_);
lean_dec_ref(v_inst_416_);
lean_dec_ref(v_inst_415_);
lean_dec_ref(v_inst_414_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_range(lean_object* v_R_421_, lean_object* v_A_422_, lean_object* v_B_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_00_u03c6_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lean_box(0);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_range___boxed(lean_object* v_R_431_, lean_object* v_A_432_, lean_object* v_B_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_00_u03c6_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_mathlib_AlgHom_range(v_R_431_, v_A_432_, v_B_433_, v_inst_434_, v_inst_435_, v_inst_436_, v_inst_437_, v_inst_438_, v_00_u03c6_439_);
lean_dec(v_00_u03c6_439_);
lean_dec_ref(v_inst_438_);
lean_dec_ref(v_inst_437_);
lean_dec_ref(v_inst_436_);
lean_dec_ref(v_inst_435_);
lean_dec_ref(v_inst_434_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict___redArg___lam__0(lean_object* v_f_441_, lean_object* v___y_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lean_apply_1(v_f_441_, v___y_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict___redArg(lean_object* v_f_444_){
_start:
{
lean_object* v___f_445_; lean_object* v___f_446_; 
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_445_, 0, v_f_444_);
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_446_, 0, v___f_445_);
return v___f_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict(lean_object* v_R_447_, lean_object* v_A_448_, lean_object* v_B_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_f_455_, lean_object* v_S_456_, lean_object* v_hf_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_mathlib_AlgHom_codRestrict___redArg(v_f_455_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_codRestrict___boxed(lean_object* v_R_459_, lean_object* v_A_460_, lean_object* v_B_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_f_467_, lean_object* v_S_468_, lean_object* v_hf_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_AlgHom_codRestrict(v_R_459_, v_A_460_, v_B_461_, v_inst_462_, v_inst_463_, v_inst_464_, v_inst_465_, v_inst_466_, v_f_467_, v_S_468_, v_hf_469_);
lean_dec_ref(v_inst_466_);
lean_dec_ref(v_inst_465_);
lean_dec_ref(v_inst_464_);
lean_dec_ref(v_inst_463_);
lean_dec_ref(v_inst_462_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_rangeRestrict___redArg(lean_object* v_f_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lp_mathlib_AlgHom_codRestrict___redArg(v_f_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_rangeRestrict(lean_object* v_R_473_, lean_object* v_A_474_, lean_object* v_B_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_f_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib_AlgHom_codRestrict___redArg(v_f_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_rangeRestrict___boxed(lean_object* v_R_483_, lean_object* v_A_484_, lean_object* v_B_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_f_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_AlgHom_rangeRestrict(v_R_483_, v_A_484_, v_B_485_, v_inst_486_, v_inst_487_, v_inst_488_, v_inst_489_, v_inst_490_, v_f_491_);
lean_dec_ref(v_inst_490_);
lean_dec_ref(v_inst_489_);
lean_dec_ref(v_inst_488_);
lean_dec_ref(v_inst_487_);
lean_dec_ref(v_inst_486_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange___redArg___lam__0(lean_object* v_00_u03c6_493_, lean_object* v___y_494_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lean_apply_1(v_00_u03c6_493_, v___y_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange___redArg(lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_00_u03c6_498_){
_start:
{
lean_object* v___f_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___f_499_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_fintypeRange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_499_, 0, v_00_u03c6_498_);
v___x_500_ = lp_mathlib_PLift_fintype___redArg(v_inst_496_);
v___x_501_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_497_, v___f_499_, v___x_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange(lean_object* v_R_502_, lean_object* v_A_503_, lean_object* v_B_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_00_u03c6_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib_AlgHom_fintypeRange___redArg(v_inst_510_, v_inst_511_, v_00_u03c6_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fintypeRange___boxed(lean_object* v_R_514_, lean_object* v_A_515_, lean_object* v_B_516_, lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_00_u03c6_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_mathlib_AlgHom_fintypeRange(v_R_514_, v_A_515_, v_B_516_, v_inst_517_, v_inst_518_, v_inst_519_, v_inst_520_, v_inst_521_, v_inst_522_, v_inst_523_, v_00_u03c6_524_);
lean_dec_ref(v_inst_521_);
lean_dec_ref(v_inst_520_);
lean_dec_ref(v_inst_519_);
lean_dec_ref(v_inst_518_);
lean_dec_ref(v_inst_517_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__0(lean_object* v_f_526_, lean_object* v___y_527_){
_start:
{
lean_object* v___x_65__overap_528_; lean_object* v___x_529_; 
v___x_65__overap_528_ = lp_mathlib_AlgHom_codRestrict___redArg(v_f_526_);
v___x_529_ = lean_apply_1(v___x_65__overap_528_, v___y_527_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__1(lean_object* v___y_530_){
_start:
{
lean_inc(v___y_530_);
return v___y_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__1___boxed(lean_object* v___y_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__1(v___y_531_);
lean_dec(v___y_531_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___redArg(lean_object* v_g_534_, lean_object* v_f_535_){
_start:
{
lean_object* v___f_536_; lean_object* v___f_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___f_536_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_ofLeftInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_536_, 0, v_f_535_);
v___f_537_ = ((lean_object*)(lp_mathlib_AlgEquiv_ofLeftInverse___redArg___closed__0));
v___x_538_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_538_, 0, lean_box(0));
lean_closure_set(v___x_538_, 1, lean_box(0));
lean_closure_set(v___x_538_, 2, lean_box(0));
lean_closure_set(v___x_538_, 3, v_g_534_);
lean_closure_set(v___x_538_, 4, v___f_537_);
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v___f_536_);
lean_ctor_set(v___x_539_, 1, v___x_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse(lean_object* v_R_540_, lean_object* v_A_541_, lean_object* v_B_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_g_548_, lean_object* v_f_549_, lean_object* v_h_550_){
_start:
{
lean_object* v___x_551_; 
v___x_551_ = lp_mathlib_AlgEquiv_ofLeftInverse___redArg(v_g_548_, v_f_549_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLeftInverse___boxed(lean_object* v_R_552_, lean_object* v_A_553_, lean_object* v_B_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_g_560_, lean_object* v_f_561_, lean_object* v_h_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_AlgEquiv_ofLeftInverse(v_R_552_, v_A_553_, v_B_554_, v_inst_555_, v_inst_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_g_560_, v_f_561_, v_h_562_);
lean_dec_ref(v_inst_559_);
lean_dec_ref(v_inst_558_);
lean_dec_ref(v_inst_557_);
lean_dec_ref(v_inst_556_);
lean_dec_ref(v_inst_555_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_subalgebraMap___redArg(lean_object* v_e_564_){
_start:
{
lean_object* v___x_565_; 
v___x_565_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_564_);
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_subalgebraMap(lean_object* v_R_566_, lean_object* v_A_567_, lean_object* v_B_568_, lean_object* v_inst_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_e_574_, lean_object* v_S_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_574_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_subalgebraMap___boxed(lean_object* v_R_577_, lean_object* v_A_578_, lean_object* v_B_579_, lean_object* v_inst_580_, lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_e_585_, lean_object* v_S_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib_AlgEquiv_subalgebraMap(v_R_577_, v_A_578_, v_B_579_, v_inst_580_, v_inst_581_, v_inst_582_, v_inst_583_, v_inst_584_, v_e_585_, v_S_586_);
lean_dec_ref(v_inst_584_);
lean_dec_ref(v_inst_583_);
lean_dec_ref(v_inst_582_);
lean_dec_ref(v_inst_581_);
lean_dec_ref(v_inst_580_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_inclusion(lean_object* v_R_589_, lean_object* v_A_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_S_594_, lean_object* v_T_595_, lean_object* v_h_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = ((lean_object*)(lp_mathlib_Subalgebra_inclusion___closed__0));
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_inclusion___boxed(lean_object* v_R_598_, lean_object* v_A_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_S_603_, lean_object* v_T_604_, lean_object* v_h_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_Subalgebra_inclusion(v_R_598_, v_A_599_, v_inst_600_, v_inst_601_, v_inst_602_, v_S_603_, v_T_604_, v_h_605_);
lean_dec_ref(v_inst_602_);
lean_dec_ref(v_inst_601_);
lean_dec_ref(v_inst_600_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq___lam__0(lean_object* v_x_607_){
_start:
{
lean_inc(v_x_607_);
return v_x_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq___lam__0___boxed(lean_object* v_x_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_Subalgebra_equivOfEq___lam__0(v_x_608_);
lean_dec(v_x_608_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq(lean_object* v_R_613_, lean_object* v_A_614_, lean_object* v_inst_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_S_618_, lean_object* v_T_619_, lean_object* v_h_620_){
_start:
{
lean_object* v___x_621_; 
v___x_621_ = ((lean_object*)(lp_mathlib_Subalgebra_equivOfEq___closed__1));
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_equivOfEq___boxed(lean_object* v_R_622_, lean_object* v_A_623_, lean_object* v_inst_624_, lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_S_627_, lean_object* v_T_628_, lean_object* v_h_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_Subalgebra_equivOfEq(v_R_622_, v_A_623_, v_inst_624_, v_inst_625_, v_inst_626_, v_S_627_, v_T_628_, v_h_629_);
lean_dec_ref(v_inst_626_);
lean_dec_ref(v_inst_625_);
lean_dec_ref(v_inst_624_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_subalgebraMap___redArg(lean_object* v_f_631_){
_start:
{
lean_object* v___f_632_; lean_object* v___x_633_; lean_object* v___x_634_; 
v___f_632_ = ((lean_object*)(lp_mathlib_Subalgebra_val___closed__0));
v___x_633_ = lp_mathlib_AlgHom_comp___redArg(v_f_631_, v___f_632_);
v___x_634_ = lp_mathlib_AlgHom_codRestrict___redArg(v___x_633_);
return v___x_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_subalgebraMap(lean_object* v_R_635_, lean_object* v_A_636_, lean_object* v_B_637_, lean_object* v_inst_638_, lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_S_643_, lean_object* v_f_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lp_mathlib_AlgHom_subalgebraMap___redArg(v_f_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_subalgebraMap___boxed(lean_object* v_R_646_, lean_object* v_A_647_, lean_object* v_B_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_S_654_, lean_object* v_f_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_mathlib_AlgHom_subalgebraMap(v_R_646_, v_A_647_, v_B_648_, v_inst_649_, v_inst_650_, v_inst_651_, v_inst_652_, v_inst_653_, v_S_654_, v_f_655_);
lean_dec_ref(v_inst_653_);
lean_dec_ref(v_inst_652_);
lean_dec_ref(v_inst_651_);
lean_dec_ref(v_inst_650_);
lean_dec_ref(v_inst_649_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem___redArg___lam__0(lean_object* v_inst_657_, lean_object* v_m_658_, lean_object* v_a_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lean_apply_2(v_inst_657_, v_m_658_, v_a_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem___redArg(lean_object* v_inst_661_){
_start:
{
lean_object* v___f_662_; 
v___f_662_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_662_, 0, v_inst_661_);
return v___f_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem(lean_object* v_R_663_, lean_object* v_A_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_00_u03b1_668_, lean_object* v_inst_669_, lean_object* v_S_670_){
_start:
{
lean_object* v___f_671_; 
v___f_671_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_671_, 0, v_inst_669_);
return v___f_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulSubtypeMem___boxed(lean_object* v_R_672_, lean_object* v_A_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_00_u03b1_677_, lean_object* v_inst_678_, lean_object* v_S_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_mathlib_Subalgebra_instSMulSubtypeMem(v_R_672_, v_A_673_, v_inst_674_, v_inst_675_, v_inst_676_, v_00_u03b1_677_, v_inst_678_, v_S_679_);
lean_dec_ref(v_inst_676_);
lean_dec_ref(v_inst_675_);
lean_dec_ref(v_inst_674_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionSubtypeMem___redArg(lean_object* v_inst_681_){
_start:
{
lean_object* v___f_682_; 
v___f_682_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_682_, 0, v_inst_681_);
return v___f_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionSubtypeMem(lean_object* v_R_683_, lean_object* v_A_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_00_u03b1_688_, lean_object* v_inst_689_, lean_object* v_S_690_){
_start:
{
lean_object* v___f_691_; 
v___f_691_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_691_, 0, v_inst_689_);
return v___f_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionSubtypeMem___boxed(lean_object* v_R_692_, lean_object* v_A_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_00_u03b1_697_, lean_object* v_inst_698_, lean_object* v_S_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_mathlib_Subalgebra_instMulActionSubtypeMem(v_R_692_, v_A_693_, v_inst_694_, v_inst_695_, v_inst_696_, v_00_u03b1_697_, v_inst_698_, v_S_699_);
lean_dec_ref(v_inst_696_);
lean_dec_ref(v_inst_695_);
lean_dec_ref(v_inst_694_);
return v_res_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem___redArg(lean_object* v_inst_701_){
_start:
{
lean_object* v___f_702_; 
v___f_702_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_702_, 0, v_inst_701_);
return v___f_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem(lean_object* v_R_703_, lean_object* v_A_704_, lean_object* v_inst_705_, lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_00_u03b1_708_, lean_object* v_inst_709_, lean_object* v_inst_710_, lean_object* v_S_711_){
_start:
{
lean_object* v___f_712_; 
v___f_712_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_712_, 0, v_inst_710_);
return v___f_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem___boxed(lean_object* v_R_713_, lean_object* v_A_714_, lean_object* v_inst_715_, lean_object* v_inst_716_, lean_object* v_inst_717_, lean_object* v_00_u03b1_718_, lean_object* v_inst_719_, lean_object* v_inst_720_, lean_object* v_S_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_Subalgebra_instDistribMulActionSubtypeMem(v_R_713_, v_A_714_, v_inst_715_, v_inst_716_, v_inst_717_, v_00_u03b1_718_, v_inst_719_, v_inst_720_, v_S_721_);
lean_dec_ref(v_inst_719_);
lean_dec_ref(v_inst_717_);
lean_dec_ref(v_inst_716_);
lean_dec_ref(v_inst_715_);
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem___redArg(lean_object* v_inst_723_){
_start:
{
lean_object* v___f_724_; 
v___f_724_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_724_, 0, v_inst_723_);
return v___f_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem(lean_object* v_R_725_, lean_object* v_A_726_, lean_object* v_inst_727_, lean_object* v_inst_728_, lean_object* v_inst_729_, lean_object* v_00_u03b1_730_, lean_object* v_inst_731_, lean_object* v_inst_732_, lean_object* v_S_733_){
_start:
{
lean_object* v___f_734_; 
v___f_734_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_734_, 0, v_inst_732_);
return v___f_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem___boxed(lean_object* v_R_735_, lean_object* v_A_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_00_u03b1_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_S_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_mathlib_Subalgebra_instSMulWithZeroSubtypeMem(v_R_735_, v_A_736_, v_inst_737_, v_inst_738_, v_inst_739_, v_00_u03b1_740_, v_inst_741_, v_inst_742_, v_S_743_);
lean_dec(v_inst_741_);
lean_dec_ref(v_inst_739_);
lean_dec_ref(v_inst_738_);
lean_dec_ref(v_inst_737_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem___redArg(lean_object* v_inst_745_){
_start:
{
lean_object* v___f_746_; 
v___f_746_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_746_, 0, v_inst_745_);
return v___f_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem(lean_object* v_R_747_, lean_object* v_A_748_, lean_object* v_inst_749_, lean_object* v_inst_750_, lean_object* v_inst_751_, lean_object* v_00_u03b1_752_, lean_object* v_inst_753_, lean_object* v_inst_754_, lean_object* v_S_755_){
_start:
{
lean_object* v___f_756_; 
v___f_756_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_756_, 0, v_inst_754_);
return v___f_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem___boxed(lean_object* v_R_757_, lean_object* v_A_758_, lean_object* v_inst_759_, lean_object* v_inst_760_, lean_object* v_inst_761_, lean_object* v_00_u03b1_762_, lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_S_765_){
_start:
{
lean_object* v_res_766_; 
v_res_766_ = lp_mathlib_Subalgebra_instMulActionWithZeroSubtypeMem(v_R_757_, v_A_758_, v_inst_759_, v_inst_760_, v_inst_761_, v_00_u03b1_762_, v_inst_763_, v_inst_764_, v_S_765_);
lean_dec(v_inst_763_);
lean_dec_ref(v_inst_761_);
lean_dec_ref(v_inst_760_);
lean_dec_ref(v_inst_759_);
return v_res_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_moduleLeft___redArg(lean_object* v_inst_767_){
_start:
{
lean_object* v___f_768_; 
v___f_768_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_768_, 0, v_inst_767_);
return v___f_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_moduleLeft(lean_object* v_R_769_, lean_object* v_A_770_, lean_object* v_inst_771_, lean_object* v_inst_772_, lean_object* v_inst_773_, lean_object* v_00_u03b1_774_, lean_object* v_inst_775_, lean_object* v_inst_776_, lean_object* v_S_777_){
_start:
{
lean_object* v___f_778_; 
v___f_778_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_778_, 0, v_inst_776_);
return v___f_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_moduleLeft___boxed(lean_object* v_R_779_, lean_object* v_A_780_, lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_00_u03b1_784_, lean_object* v_inst_785_, lean_object* v_inst_786_, lean_object* v_S_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_mathlib_Subalgebra_moduleLeft(v_R_779_, v_A_780_, v_inst_781_, v_inst_782_, v_inst_783_, v_00_u03b1_784_, v_inst_785_, v_inst_786_, v_S_787_);
lean_dec_ref(v_inst_785_);
lean_dec_ref(v_inst_783_);
lean_dec_ref(v_inst_782_);
lean_dec_ref(v_inst_781_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAlgebra___redArg(lean_object* v_inst_789_){
_start:
{
lean_object* v___x_790_; 
v___x_790_ = lp_mathlib_Algebra_ofSubsemiring___redArg(v_inst_789_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAlgebra(lean_object* v_00_u03b1_791_, lean_object* v_R_792_, lean_object* v_A_793_, lean_object* v_inst_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_inst_798_, lean_object* v_S_799_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = lp_mathlib_Algebra_ofSubsemiring___redArg(v_inst_798_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toAlgebra___boxed(lean_object* v_00_u03b1_801_, lean_object* v_R_802_, lean_object* v_A_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_, lean_object* v_S_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_mathlib_Subalgebra_toAlgebra(v_00_u03b1_801_, v_R_802_, v_A_803_, v_inst_804_, v_inst_805_, v_inst_806_, v_inst_807_, v_inst_808_, v_S_809_);
lean_dec_ref(v_inst_807_);
lean_dec_ref(v_inst_806_);
lean_dec_ref(v_inst_805_);
lean_dec_ref(v_inst_804_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_center(lean_object* v_R_811_, lean_object* v_A_812_, lean_object* v_inst_813_, lean_object* v_inst_814_, lean_object* v_inst_815_){
_start:
{
lean_object* v___x_816_; 
v___x_816_ = lean_box(0);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_center___boxed(lean_object* v_R_817_, lean_object* v_A_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_Subalgebra_center(v_R_817_, v_A_818_, v_inst_819_, v_inst_820_, v_inst_821_);
lean_dec_ref(v_inst_821_);
lean_dec_ref(v_inst_820_);
lean_dec_ref(v_inst_819_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter___redArg(lean_object* v_inst_823_){
_start:
{
lean_object* v___x_824_; 
v___x_824_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_823_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter(lean_object* v_R_825_, lean_object* v_A_826_, lean_object* v_inst_827_, lean_object* v_inst_828_, lean_object* v_inst_829_){
_start:
{
lean_object* v___x_830_; 
v___x_830_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_828_);
return v___x_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter___boxed(lean_object* v_R_831_, lean_object* v_A_832_, lean_object* v_inst_833_, lean_object* v_inst_834_, lean_object* v_inst_835_){
_start:
{
lean_object* v_res_836_; 
v_res_836_ = lp_mathlib_Subalgebra_instCommSemiringSubtypeMemCenter(v_R_831_, v_A_832_, v_inst_833_, v_inst_834_, v_inst_835_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_833_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___redArg(lean_object* v_inst_837_, lean_object* v_a_838_){
_start:
{
lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v_toNeg_841_; lean_object* v___x_842_; 
v___x_839_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_837_);
v___x_840_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_839_);
lean_dec_ref(v___x_839_);
v_toNeg_841_ = lean_ctor_get(v___x_840_, 1);
lean_inc(v_toNeg_841_);
lean_dec_ref(v___x_840_);
v___x_842_ = lean_apply_1(v_toNeg_841_, v_a_838_);
return v___x_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___redArg___boxed(lean_object* v_inst_843_, lean_object* v_a_844_){
_start:
{
lean_object* v_res_845_; 
v_res_845_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___redArg(v_inst_843_, v_a_844_);
lean_dec_ref(v_inst_843_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1(lean_object* v_R_846_, lean_object* v_inst_847_, lean_object* v_A_848_, lean_object* v_inst_849_, lean_object* v_inst_850_, lean_object* v_a_851_){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v_toNeg_854_; lean_object* v___x_855_; 
v___x_852_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_849_);
v___x_853_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_852_);
lean_dec_ref(v___x_852_);
v_toNeg_854_ = lean_ctor_get(v___x_853_, 1);
lean_inc(v_toNeg_854_);
lean_dec_ref(v___x_853_);
v___x_855_ = lean_apply_1(v_toNeg_854_, v_a_851_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___boxed(lean_object* v_R_856_, lean_object* v_inst_857_, lean_object* v_A_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_a_861_){
_start:
{
lean_object* v_res_862_; 
v_res_862_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1(v_R_856_, v_inst_857_, v_A_858_, v_inst_859_, v_inst_860_, v_a_861_);
lean_dec_ref(v_inst_860_);
lean_dec_ref(v_inst_859_);
lean_dec_ref(v_inst_857_);
return v_res_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3___redArg(lean_object* v_inst_863_, lean_object* v_a_864_, lean_object* v_b_865_){
_start:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v_toSub_868_; lean_object* v___x_869_; 
v___x_866_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_863_);
v___x_867_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_866_);
lean_dec_ref(v___x_866_);
v_toSub_868_ = lean_ctor_get(v___x_867_, 2);
lean_inc(v_toSub_868_);
lean_dec_ref(v___x_867_);
v___x_869_ = lean_apply_2(v_toSub_868_, v_a_864_, v_b_865_);
return v___x_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3(lean_object* v_R_870_, lean_object* v_inst_871_, lean_object* v_A_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_a_875_, lean_object* v_b_876_){
_start:
{
lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v_toSub_879_; lean_object* v___x_880_; 
v___x_877_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_873_);
v___x_878_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_877_);
lean_dec_ref(v___x_877_);
v_toSub_879_ = lean_ctor_get(v___x_878_, 2);
lean_inc(v_toSub_879_);
lean_dec_ref(v___x_878_);
v___x_880_ = lean_apply_2(v_toSub_879_, v_a_875_, v_b_876_);
return v___x_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3___boxed(lean_object* v_R_881_, lean_object* v_inst_882_, lean_object* v_A_883_, lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_a_886_, lean_object* v_b_887_){
_start:
{
lean_object* v_res_888_; 
v_res_888_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3(v_R_881_, v_inst_882_, v_A_883_, v_inst_884_, v_inst_885_, v_a_886_, v_b_887_);
lean_dec_ref(v_inst_885_);
lean_dec_ref(v_inst_882_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___redArg(lean_object* v_inst_889_, lean_object* v_n_890_, lean_object* v_x_891_){
_start:
{
lean_object* v___x_892_; lean_object* v_toNonUnitalNonAssocRing_893_; lean_object* v_toAddCommGroup_894_; lean_object* v_toZSMul_895_; lean_object* v___x_896_; 
v___x_892_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_889_);
v_toNonUnitalNonAssocRing_893_ = lean_ctor_get(v___x_892_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_893_);
lean_dec_ref(v___x_892_);
v_toAddCommGroup_894_ = lean_ctor_get(v_toNonUnitalNonAssocRing_893_, 0);
lean_inc_ref(v_toAddCommGroup_894_);
lean_dec_ref(v_toNonUnitalNonAssocRing_893_);
v_toZSMul_895_ = lean_ctor_get(v_toAddCommGroup_894_, 3);
lean_inc(v_toZSMul_895_);
lean_dec_ref(v_toAddCommGroup_894_);
v___x_896_ = lean_apply_2(v_toZSMul_895_, v_n_890_, v_x_891_);
return v___x_896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___redArg___boxed(lean_object* v_inst_897_, lean_object* v_n_898_, lean_object* v_x_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___redArg(v_inst_897_, v_n_898_, v_x_899_);
lean_dec_ref(v_inst_897_);
return v_res_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5(lean_object* v_R_901_, lean_object* v_inst_902_, lean_object* v_A_903_, lean_object* v_inst_904_, lean_object* v_inst_905_, lean_object* v_n_906_, lean_object* v_x_907_){
_start:
{
lean_object* v___x_908_; lean_object* v_toNonUnitalNonAssocRing_909_; lean_object* v_toAddCommGroup_910_; lean_object* v_toZSMul_911_; lean_object* v___x_912_; 
v___x_908_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_904_);
v_toNonUnitalNonAssocRing_909_ = lean_ctor_get(v___x_908_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_909_);
lean_dec_ref(v___x_908_);
v_toAddCommGroup_910_ = lean_ctor_get(v_toNonUnitalNonAssocRing_909_, 0);
lean_inc_ref(v_toAddCommGroup_910_);
lean_dec_ref(v_toNonUnitalNonAssocRing_909_);
v_toZSMul_911_ = lean_ctor_get(v_toAddCommGroup_910_, 3);
lean_inc(v_toZSMul_911_);
lean_dec_ref(v_toAddCommGroup_910_);
v___x_912_ = lean_apply_2(v_toZSMul_911_, v_n_906_, v_x_907_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___boxed(lean_object* v_R_913_, lean_object* v_inst_914_, lean_object* v_A_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_n_918_, lean_object* v_x_919_){
_start:
{
lean_object* v_res_920_; 
v_res_920_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5(v_R_913_, v_inst_914_, v_A_915_, v_inst_916_, v_inst_917_, v_n_918_, v_x_919_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec_ref(v_inst_914_);
return v_res_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___redArg(lean_object* v_inst_921_, lean_object* v_n_922_){
_start:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v_toIntCast_926_; lean_object* v___x_927_; 
v___x_923_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_921_);
v___x_924_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_923_);
v___x_925_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_924_);
lean_dec_ref(v___x_924_);
v_toIntCast_926_ = lean_ctor_get(v___x_925_, 0);
lean_inc(v_toIntCast_926_);
lean_dec_ref(v___x_925_);
v___x_927_ = lean_apply_1(v_toIntCast_926_, v_n_922_);
return v___x_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___redArg___boxed(lean_object* v_inst_928_, lean_object* v_n_929_){
_start:
{
lean_object* v_res_930_; 
v_res_930_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___redArg(v_inst_928_, v_n_929_);
lean_dec_ref(v_inst_928_);
return v_res_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12(lean_object* v_R_931_, lean_object* v_inst_932_, lean_object* v_A_933_, lean_object* v_inst_934_, lean_object* v_inst_935_, lean_object* v_n_936_){
_start:
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v_toIntCast_940_; lean_object* v___x_941_; 
v___x_937_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_934_);
v___x_938_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_937_);
v___x_939_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_938_);
lean_dec_ref(v___x_938_);
v_toIntCast_940_ = lean_ctor_get(v___x_939_, 0);
lean_inc(v_toIntCast_940_);
lean_dec_ref(v___x_939_);
v___x_941_ = lean_apply_1(v_toIntCast_940_, v_n_936_);
return v___x_941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___boxed(lean_object* v_R_942_, lean_object* v_inst_943_, lean_object* v_A_944_, lean_object* v_inst_945_, lean_object* v_inst_946_, lean_object* v_n_947_){
_start:
{
lean_object* v_res_948_; 
v_res_948_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12(v_R_942_, v_inst_943_, v_A_944_, v_inst_945_, v_inst_946_, v_n_947_);
lean_dec_ref(v_inst_946_);
lean_dec_ref(v_inst_945_);
lean_dec_ref(v_inst_943_);
return v_res_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___redArg(lean_object* v_inst_949_, lean_object* v_inst_950_, lean_object* v_inst_951_){
_start:
{
lean_object* v_toSemiring_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; 
v_toSemiring_952_ = lean_ctor_get(v_inst_950_, 0);
lean_inc_ref(v_toSemiring_952_);
v___x_953_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_toSemiring_952_);
lean_inc_ref_n(v_inst_951_, 3);
lean_inc_ref_n(v_inst_950_, 3);
lean_inc_ref_n(v_inst_949_, 3);
v___x_954_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__1___boxed), 6, 5);
lean_closure_set(v___x_954_, 0, lean_box(0));
lean_closure_set(v___x_954_, 1, v_inst_949_);
lean_closure_set(v___x_954_, 2, lean_box(0));
lean_closure_set(v___x_954_, 3, v_inst_950_);
lean_closure_set(v___x_954_, 4, v_inst_951_);
v___x_955_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__3___boxed), 7, 5);
lean_closure_set(v___x_955_, 0, lean_box(0));
lean_closure_set(v___x_955_, 1, v_inst_949_);
lean_closure_set(v___x_955_, 2, lean_box(0));
lean_closure_set(v___x_955_, 3, v_inst_950_);
lean_closure_set(v___x_955_, 4, v_inst_951_);
v___x_956_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__5___boxed), 7, 5);
lean_closure_set(v___x_956_, 0, lean_box(0));
lean_closure_set(v___x_956_, 1, v_inst_949_);
lean_closure_set(v___x_956_, 2, lean_box(0));
lean_closure_set(v___x_956_, 3, v_inst_950_);
lean_closure_set(v___x_956_, 4, v_inst_951_);
v___x_957_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___aux__12___boxed), 6, 5);
lean_closure_set(v___x_957_, 0, lean_box(0));
lean_closure_set(v___x_957_, 1, v_inst_949_);
lean_closure_set(v___x_957_, 2, lean_box(0));
lean_closure_set(v___x_957_, 3, v_inst_950_);
lean_closure_set(v___x_957_, 4, v_inst_951_);
v___x_958_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_958_, 0, v___x_953_);
lean_ctor_set(v___x_958_, 1, v___x_954_);
lean_ctor_set(v___x_958_, 2, v___x_955_);
lean_ctor_set(v___x_958_, 3, v___x_956_);
lean_ctor_set(v___x_958_, 4, v___x_957_);
return v___x_958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter(lean_object* v_R_959_, lean_object* v_inst_960_, lean_object* v_A_961_, lean_object* v_inst_962_, lean_object* v_inst_963_){
_start:
{
lean_object* v___x_964_; 
v___x_964_ = lp_mathlib_Subalgebra_instCommRingSubtypeMemCenter___redArg(v_inst_960_, v_inst_962_, v_inst_963_);
return v___x_964_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_centralizer(lean_object* v_R_965_, lean_object* v_A_966_, lean_object* v_inst_967_, lean_object* v_inst_968_, lean_object* v_inst_969_, lean_object* v_s_970_){
_start:
{
lean_object* v___x_971_; 
v___x_971_ = lean_box(0);
return v___x_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_centralizer___boxed(lean_object* v_R_972_, lean_object* v_A_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_inst_976_, lean_object* v_s_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_Subalgebra_centralizer(v_R_972_, v_A_973_, v_inst_974_, v_inst_975_, v_inst_976_, v_s_977_);
lean_dec_ref(v_inst_976_);
lean_dec_ref(v_inst_975_);
lean_dec_ref(v_inst_974_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubsemiring___redArg(lean_object* v_S_979_){
_start:
{
return v_S_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubsemiring(lean_object* v_R_980_, lean_object* v_inst_981_, lean_object* v_S_982_){
_start:
{
return v_S_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubsemiring___boxed(lean_object* v_R_983_, lean_object* v_inst_984_, lean_object* v_S_985_){
_start:
{
lean_object* v_res_986_; 
v_res_986_ = lp_mathlib_subalgebraOfSubsemiring(v_R_983_, v_inst_984_, v_S_985_);
lean_dec_ref(v_inst_984_);
return v_res_986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubring___redArg(lean_object* v_S_987_){
_start:
{
return v_S_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubring(lean_object* v_R_988_, lean_object* v_inst_989_, lean_object* v_S_990_){
_start:
{
return v_S_990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subalgebraOfSubring___boxed(lean_object* v_R_991_, lean_object* v_inst_992_, lean_object* v_S_993_){
_start:
{
lean_object* v_res_994_; 
v_res_994_ = lp_mathlib_subalgebraOfSubring(v_R_991_, v_inst_992_, v_S_993_);
lean_dec_ref(v_inst_992_);
return v_res_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_equalizer(lean_object* v_R_995_, lean_object* v_A_996_, lean_object* v_B_997_, lean_object* v_inst_998_, lean_object* v_inst_999_, lean_object* v_inst_1000_, lean_object* v_inst_1001_, lean_object* v_inst_1002_, lean_object* v_00_u03d5_1003_, lean_object* v_00_u03c8_1004_){
_start:
{
lean_object* v___x_1005_; 
v___x_1005_ = lean_box(0);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_equalizer___boxed(lean_object* v_R_1006_, lean_object* v_A_1007_, lean_object* v_B_1008_, lean_object* v_inst_1009_, lean_object* v_inst_1010_, lean_object* v_inst_1011_, lean_object* v_inst_1012_, lean_object* v_inst_1013_, lean_object* v_00_u03d5_1014_, lean_object* v_00_u03c8_1015_){
_start:
{
lean_object* v_res_1016_; 
v_res_1016_ = lp_mathlib_AlgHom_equalizer(v_R_1006_, v_A_1007_, v_B_1008_, v_inst_1009_, v_inst_1010_, v_inst_1011_, v_inst_1012_, v_inst_1013_, v_00_u03d5_1014_, v_00_u03c8_1015_);
lean_dec(v_00_u03c8_1015_);
lean_dec(v_00_u03d5_1014_);
lean_dec_ref(v_inst_1013_);
lean_dec_ref(v_inst_1012_);
lean_dec_ref(v_inst_1011_);
lean_dec_ref(v_inst_1010_);
lean_dec_ref(v_inst_1009_);
return v_res_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubalgebra(lean_object* v_R_1017_, lean_object* v_A_1018_, lean_object* v_inst_1019_, lean_object* v_inst_1020_, lean_object* v_inst_1021_, lean_object* v_S_1022_, lean_object* v_h1_1023_){
_start:
{
lean_object* v___x_1024_; 
v___x_1024_ = lean_box(0);
return v___x_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubalgebra___boxed(lean_object* v_R_1025_, lean_object* v_A_1026_, lean_object* v_inst_1027_, lean_object* v_inst_1028_, lean_object* v_inst_1029_, lean_object* v_S_1030_, lean_object* v_h1_1031_){
_start:
{
lean_object* v_res_1032_; 
v_res_1032_ = lp_mathlib_NonUnitalSubalgebra_toSubalgebra(v_R_1025_, v_A_1026_, v_inst_1027_, v_inst_1028_, v_inst_1029_, v_S_1030_, v_h1_1031_);
lean_dec_ref(v_inst_1029_);
lean_dec_ref(v_inst_1028_);
lean_dec_ref(v_inst_1027_);
return v_res_1032_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
