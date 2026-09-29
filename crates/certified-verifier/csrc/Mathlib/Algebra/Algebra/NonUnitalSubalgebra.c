// Lean compiler output
// Module: Mathlib.Algebra.Algebra.NonUnitalSubalgebra
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.NonUnitalHom public import Mathlib.Data.Set.UnionLift public import Mathlib.LinearAlgebra.Span.Basic public import Mathlib.RingTheory.NonUnitalSubring.Basic
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
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_SetLike_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Set_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubalgebraClass_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubalgebraClass_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instSetLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instSetLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_NonUnitalSubalgebra_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_NonUnitalSubalgebra_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubsemiring_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubsemiring_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toNonUnitalSubalgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toNonUnitalSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toNonUnitalSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_equalizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_equalizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgebra_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgebra_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgebra_gi___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgebra_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_gi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_gi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__1 = (const lean_object*)&lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__2 = (const lean_object*)&lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instInhabitedNonUnitalSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instInhabitedNonUnitalSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalAlgebra_toTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_NonUnitalAlgebra_toTop___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalAlgebra_toTop___closed__0_value;
static lean_once_cell_t lp_mathlib_NonUnitalAlgebra_toTop___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_NonUnitalAlgebra_toTop___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_toTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_toTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubalgebra_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_inclusion___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_NonUnitalSubalgebra_inclusion___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubalgebra_inclusion___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_centralizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_centralizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_NonUnitalSubalgebraClass_subtype___lam__0(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype(lean_object* v_S_5_, lean_object* v_R_6_, lean_object* v_A_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_hSR_13_, lean_object* v_s_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = ((lean_object*)(lp_mathlib_NonUnitalSubalgebraClass_subtype___closed__0));
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebraClass_subtype___boxed(lean_object* v_S_16_, lean_object* v_R_17_, lean_object* v_A_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_hSR_24_, lean_object* v_s_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_NonUnitalSubalgebraClass_subtype(v_S_16_, v_R_17_, v_A_18_, v_inst_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_hSR_24_, v_s_25_);
lean_dec(v_s_25_);
lean_dec(v_inst_21_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_19_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule___redArg(lean_object* v_self_27_){
_start:
{
return v_self_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule(lean_object* v_R_28_, lean_object* v_A_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_self_33_){
_start:
{
return v_self_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule___boxed(lean_object* v_R_34_, lean_object* v_A_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_self_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_NonUnitalSubalgebra_toSubmodule(v_R_34_, v_A_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_self_39_);
lean_dec(v_inst_38_);
lean_dec_ref(v_inst_37_);
lean_dec_ref(v_inst_36_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instSetLike(lean_object* v_R_41_, lean_object* v_A_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instSetLike___boxed(lean_object* v_R_47_, lean_object* v_A_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_NonUnitalSubalgebra_instSetLike(v_R_47_, v_A_48_, v_inst_49_, v_inst_50_, v_inst_51_);
lean_dec(v_inst_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_52_;
}
}
static lean_object* _init_lp_mathlib_NonUnitalSubalgebra_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = lean_box(0);
v___x_54_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instPartialOrder(lean_object* v_R_55_, lean_object* v_A_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_obj_once(&lp_mathlib_NonUnitalSubalgebra_instPartialOrder___closed__0, &lp_mathlib_NonUnitalSubalgebra_instPartialOrder___closed__0_once, _init_lp_mathlib_NonUnitalSubalgebra_instPartialOrder___closed__0);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instPartialOrder___boxed(lean_object* v_R_61_, lean_object* v_A_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_NonUnitalSubalgebra_instPartialOrder(v_R_61_, v_A_62_, v_inst_63_, v_inst_64_, v_inst_65_);
lean_dec(v_inst_65_);
lean_dec_ref(v_inst_64_);
lean_dec_ref(v_inst_63_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_ofClass(lean_object* v_S_67_, lean_object* v_R_68_, lean_object* v_A_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_s_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lean_box(0);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_ofClass___boxed(lean_object* v_S_78_, lean_object* v_R_79_, lean_object* v_A_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_s_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_NonUnitalSubalgebra_ofClass(v_S_78_, v_R_79_, v_A_80_, v_inst_81_, v_inst_82_, v_inst_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_s_87_);
lean_dec(v_s_87_);
lean_dec(v_inst_83_);
lean_dec_ref(v_inst_82_);
lean_dec_ref(v_inst_81_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_copy(lean_object* v_R_89_, lean_object* v_A_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_S_94_, lean_object* v_s_95_, lean_object* v_hs_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_box(0);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_copy___boxed(lean_object* v_R_98_, lean_object* v_A_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_S_103_, lean_object* v_s_104_, lean_object* v_hs_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_NonUnitalSubalgebra_copy(v_R_98_, v_A_99_, v_inst_100_, v_inst_101_, v_inst_102_, v_S_103_, v_s_104_, v_hs_105_);
lean_dec(v_inst_102_);
lean_dec_ref(v_inst_101_);
lean_dec_ref(v_inst_100_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem___redArg(lean_object* v_inst_107_){
_start:
{
lean_object* v___x_108_; lean_object* v_toZero_109_; 
v___x_108_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_inst_107_);
v_toZero_109_ = lean_ctor_get(v___x_108_, 1);
lean_inc(v_toZero_109_);
lean_dec_ref(v___x_108_);
return v_toZero_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem(lean_object* v_R_110_, lean_object* v_A_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_S_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem___redArg(v_inst_113_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem___boxed(lean_object* v_R_117_, lean_object* v_A_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_S_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_NonUnitalSubalgebra_instInhabitedSubtypeMem(v_R_117_, v_A_118_, v_inst_119_, v_inst_120_, v_inst_121_, v_S_122_);
lean_dec(v_inst_121_);
lean_dec_ref(v_inst_119_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring___redArg(lean_object* v_S_124_){
_start:
{
return v_S_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring(lean_object* v_R_125_, lean_object* v_A_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_S_130_){
_start:
{
return v_S_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring___boxed(lean_object* v_R_131_, lean_object* v_A_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_S_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring(v_R_131_, v_A_132_, v_inst_133_, v_inst_134_, v_inst_135_, v_S_136_);
lean_dec(v_inst_135_);
lean_dec_ref(v_inst_134_);
lean_dec_ref(v_inst_133_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring(lean_object* v_R_140_, lean_object* v_A_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_S_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_143_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring___boxed(lean_object* v_R_147_, lean_object* v_A_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_S_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocSemiring(v_R_147_, v_A_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_S_152_);
lean_dec(v_inst_151_);
lean_dec_ref(v_inst_149_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring___redArg(lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring(lean_object* v_R_156_, lean_object* v_A_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_S_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_159_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring___boxed(lean_object* v_R_163_, lean_object* v_A_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_S_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalSemiring(v_R_163_, v_A_164_, v_inst_165_, v_inst_166_, v_inst_167_, v_S_168_);
lean_dec(v_inst_167_);
lean_dec_ref(v_inst_165_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring___redArg(lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring(lean_object* v_R_172_, lean_object* v_A_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_S_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_175_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring___boxed(lean_object* v_R_179_, lean_object* v_A_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_S_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommSemiring(v_R_179_, v_A_180_, v_inst_181_, v_inst_182_, v_inst_183_, v_S_184_);
lean_dec(v_inst_183_);
lean_dec_ref(v_inst_181_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing___redArg(lean_object* v_inst_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing(lean_object* v_R_188_, lean_object* v_A_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_S_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_191_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing___boxed(lean_object* v_R_195_, lean_object* v_A_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_S_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalNonAssocRing(v_R_195_, v_A_196_, v_inst_197_, v_inst_198_, v_inst_199_, v_S_200_);
lean_dec(v_inst_199_);
lean_dec_ref(v_inst_197_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing___redArg(lean_object* v_inst_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing(lean_object* v_R_204_, lean_object* v_A_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_S_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_207_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing___boxed(lean_object* v_R_211_, lean_object* v_A_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_S_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalRing(v_R_211_, v_A_212_, v_inst_213_, v_inst_214_, v_inst_215_, v_S_216_);
lean_dec(v_inst_215_);
lean_dec_ref(v_inst_213_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing___redArg(lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing(lean_object* v_R_220_, lean_object* v_A_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_S_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_223_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing___boxed(lean_object* v_R_227_, lean_object* v_A_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_S_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalCommRing(v_R_227_, v_A_228_, v_inst_229_, v_inst_230_, v_inst_231_, v_S_232_);
lean_dec(v_inst_231_);
lean_dec_ref(v_inst_229_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___lam__0(lean_object* v_S_234_){
_start:
{
return v_S_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27(lean_object* v_R_236_, lean_object* v_A_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___f_241_; 
v___f_241_ = ((lean_object*)(lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___closed__0));
return v___f_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___boxed(lean_object* v_R_242_, lean_object* v_A_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27(v_R_242_, v_A_243_, v_inst_244_, v_inst_245_, v_inst_246_);
lean_dec(v_inst_246_);
lean_dec_ref(v_inst_245_);
lean_dec_ref(v_inst_244_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubsemiring_x27(lean_object* v_R_248_, lean_object* v_A_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_){
_start:
{
lean_object* v___f_253_; 
v___f_253_ = ((lean_object*)(lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___closed__0));
return v___f_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubsemiring_x27___boxed(lean_object* v_R_254_, lean_object* v_A_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubsemiring_x27(v_R_254_, v_A_255_, v_inst_256_, v_inst_257_, v_inst_258_);
lean_dec(v_inst_258_);
lean_dec_ref(v_inst_257_);
lean_dec_ref(v_inst_256_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring_x27(lean_object* v_R_260_, lean_object* v_A_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___f_265_; 
v___f_265_ = ((lean_object*)(lp_mathlib_NonUnitalSubalgebra_toSubmodule_x27___closed__0));
return v___f_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring_x27___boxed(lean_object* v_R_266_, lean_object* v_A_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_NonUnitalSubalgebra_toNonUnitalSubring_x27(v_R_266_, v_A_267_, v_inst_268_, v_inst_269_, v_inst_270_);
lean_dec(v_inst_270_);
lean_dec_ref(v_inst_269_);
lean_dec_ref(v_inst_268_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule_x27___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v___f_273_; 
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_273_, 0, v_inst_272_);
return v___f_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule_x27(lean_object* v_R_x27_274_, lean_object* v_R_275_, lean_object* v_A_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_S_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v___f_285_; 
v___f_285_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_285_, 0, v_inst_283_);
return v___f_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule_x27___boxed(lean_object* v_R_x27_286_, lean_object* v_R_287_, lean_object* v_A_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_S_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_NonUnitalSubalgebra_instModule_x27(v_R_x27_286_, v_R_287_, v_A_288_, v_inst_289_, v_inst_290_, v_inst_291_, v_S_292_, v_inst_293_, v_inst_294_, v_inst_295_, v_inst_296_);
lean_dec(v_inst_294_);
lean_dec_ref(v_inst_293_);
lean_dec(v_inst_291_);
lean_dec_ref(v_inst_290_);
lean_dec_ref(v_inst_289_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule___redArg(lean_object* v_inst_298_){
_start:
{
lean_object* v___f_299_; 
v___f_299_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_299_, 0, v_inst_298_);
return v___f_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule(lean_object* v_R_300_, lean_object* v_A_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_S_305_){
_start:
{
lean_object* v___f_306_; 
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_306_, 0, v_inst_304_);
return v___f_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_instModule___boxed(lean_object* v_R_307_, lean_object* v_A_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_S_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib_NonUnitalSubalgebra_instModule(v_R_307_, v_A_308_, v_inst_309_, v_inst_310_, v_inst_311_, v_S_312_);
lean_dec_ref(v_inst_310_);
lean_dec_ref(v_inst_309_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___redArg(lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_S_317_){
_start:
{
lean_object* v_toAddCommMonoid_318_; lean_object* v___x_319_; 
v_toAddCommMonoid_318_ = lean_ctor_get(v_inst_315_, 0);
v___x_319_ = lp_mathlib_LinearEquiv_ofEq(lean_box(0), lean_box(0), v_inst_314_, v_toAddCommMonoid_318_, v_inst_316_, v_S_317_, v_S_317_, lean_box(0));
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___redArg___boxed(lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_S_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___redArg(v_inst_320_, v_inst_321_, v_inst_322_, v_S_323_);
lean_dec(v_inst_322_);
lean_dec_ref(v_inst_321_);
lean_dec_ref(v_inst_320_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv(lean_object* v_R_325_, lean_object* v_A_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_S_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___redArg(v_inst_327_, v_inst_328_, v_inst_329_, v_S_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv___boxed(lean_object* v_R_332_, lean_object* v_A_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_S_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_NonUnitalSubalgebra_toSubmoduleEquiv(v_R_332_, v_A_333_, v_inst_334_, v_inst_335_, v_inst_336_, v_S_337_);
lean_dec(v_inst_336_);
lean_dec_ref(v_inst_335_);
lean_dec_ref(v_inst_334_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_map(lean_object* v_R_339_, lean_object* v_A_340_, lean_object* v_B_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_f_347_, lean_object* v_S_348_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lean_box(0);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_map___boxed(lean_object* v_R_350_, lean_object* v_A_351_, lean_object* v_B_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_f_358_, lean_object* v_S_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_NonUnitalSubalgebra_map(v_R_350_, v_A_351_, v_B_352_, v_inst_353_, v_inst_354_, v_inst_355_, v_inst_356_, v_inst_357_, v_f_358_, v_S_359_);
lean_dec(v_f_358_);
lean_dec(v_inst_357_);
lean_dec(v_inst_356_);
lean_dec_ref(v_inst_355_);
lean_dec_ref(v_inst_354_);
lean_dec_ref(v_inst_353_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_comap(lean_object* v_R_361_, lean_object* v_A_362_, lean_object* v_B_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_f_369_, lean_object* v_S_370_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lean_box(0);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_comap___boxed(lean_object* v_R_372_, lean_object* v_A_373_, lean_object* v_B_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_f_380_, lean_object* v_S_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_NonUnitalSubalgebra_comap(v_R_372_, v_A_373_, v_B_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_inst_379_, v_f_380_, v_S_381_);
lean_dec(v_f_380_);
lean_dec(v_inst_379_);
lean_dec(v_inst_378_);
lean_dec_ref(v_inst_377_);
lean_dec_ref(v_inst_376_);
lean_dec_ref(v_inst_375_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toNonUnitalSubalgebra___redArg(lean_object* v_p_383_){
_start:
{
return v_p_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toNonUnitalSubalgebra(lean_object* v_R_384_, lean_object* v_A_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_p_389_, lean_object* v_h__mul_390_){
_start:
{
return v_p_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toNonUnitalSubalgebra___boxed(lean_object* v_R_391_, lean_object* v_A_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_p_396_, lean_object* v_h__mul_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_Submodule_toNonUnitalSubalgebra(v_R_391_, v_A_392_, v_inst_393_, v_inst_394_, v_inst_395_, v_p_396_, v_h__mul_397_);
lean_dec(v_inst_395_);
lean_dec_ref(v_inst_394_);
lean_dec_ref(v_inst_393_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_range(lean_object* v_R_399_, lean_object* v_A_400_, lean_object* v_B_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_00_u03c6_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lean_box(0);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_range___boxed(lean_object* v_R_409_, lean_object* v_A_410_, lean_object* v_B_411_, lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_00_u03c6_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_NonUnitalAlgHom_range(v_R_409_, v_A_410_, v_B_411_, v_inst_412_, v_inst_413_, v_inst_414_, v_inst_415_, v_inst_416_, v_00_u03c6_417_);
lean_dec(v_00_u03c6_417_);
lean_dec(v_inst_416_);
lean_dec_ref(v_inst_415_);
lean_dec(v_inst_414_);
lean_dec_ref(v_inst_413_);
lean_dec_ref(v_inst_412_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___lam__0(lean_object* v_f_419_, lean_object* v___y_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lean_apply_1(v_f_419_, v___y_420_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___redArg(lean_object* v_f_423_){
_start:
{
lean_object* v___f_424_; lean_object* v___f_425_; lean_object* v___f_426_; 
v___f_424_ = ((lean_object*)(lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___closed__0));
v___f_425_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_425_, 0, v_f_423_);
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0), 3, 2);
lean_closure_set(v___f_426_, 0, v___f_424_);
lean_closure_set(v___f_426_, 1, v___f_425_);
return v___f_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict(lean_object* v_R_427_, lean_object* v_A_428_, lean_object* v_B_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_f_435_, lean_object* v_S_436_, lean_object* v_hf_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_mathlib_NonUnitalAlgHom_codRestrict___redArg(v_f_435_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_codRestrict___boxed(lean_object* v_R_439_, lean_object* v_A_440_, lean_object* v_B_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_f_447_, lean_object* v_S_448_, lean_object* v_hf_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_NonUnitalAlgHom_codRestrict(v_R_439_, v_A_440_, v_B_441_, v_inst_442_, v_inst_443_, v_inst_444_, v_inst_445_, v_inst_446_, v_f_447_, v_S_448_, v_hf_449_);
lean_dec(v_inst_446_);
lean_dec_ref(v_inst_445_);
lean_dec(v_inst_444_);
lean_dec_ref(v_inst_443_);
lean_dec_ref(v_inst_442_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_rangeRestrict___redArg(lean_object* v_f_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_mathlib_NonUnitalAlgHom_codRestrict___redArg(v_f_451_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_rangeRestrict(lean_object* v_R_453_, lean_object* v_A_454_, lean_object* v_B_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_f_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_mathlib_NonUnitalAlgHom_codRestrict___redArg(v_f_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_rangeRestrict___boxed(lean_object* v_R_463_, lean_object* v_A_464_, lean_object* v_B_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_f_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_NonUnitalAlgHom_rangeRestrict(v_R_463_, v_A_464_, v_B_465_, v_inst_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_, v_f_471_);
lean_dec(v_inst_470_);
lean_dec_ref(v_inst_469_);
lean_dec(v_inst_468_);
lean_dec_ref(v_inst_467_);
lean_dec_ref(v_inst_466_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_equalizer(lean_object* v_R_473_, lean_object* v_A_474_, lean_object* v_B_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_00_u03d5_481_, lean_object* v_00_u03c8_482_){
_start:
{
lean_object* v___x_483_; 
v___x_483_ = lean_box(0);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_equalizer___boxed(lean_object* v_R_484_, lean_object* v_A_485_, lean_object* v_B_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_00_u03d5_492_, lean_object* v_00_u03c8_493_){
_start:
{
lean_object* v_res_494_; 
v_res_494_ = lp_mathlib_NonUnitalAlgHom_equalizer(v_R_484_, v_A_485_, v_B_486_, v_inst_487_, v_inst_488_, v_inst_489_, v_inst_490_, v_inst_491_, v_00_u03d5_492_, v_00_u03c8_493_);
lean_dec(v_00_u03c8_493_);
lean_dec(v_00_u03d5_492_);
lean_dec(v_inst_491_);
lean_dec_ref(v_inst_490_);
lean_dec(v_inst_489_);
lean_dec_ref(v_inst_488_);
lean_dec_ref(v_inst_487_);
return v_res_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange___redArg___lam__0(lean_object* v_00_u03c6_495_, lean_object* v___y_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lean_apply_1(v_00_u03c6_495_, v___y_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange___redArg(lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_00_u03c6_500_){
_start:
{
lean_object* v___f_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v___f_501_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalAlgHom_fintypeRange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_501_, 0, v_00_u03c6_500_);
v___x_502_ = lp_mathlib_PLift_fintype___redArg(v_inst_498_);
v___x_503_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_499_, v___f_501_, v___x_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange(lean_object* v_R_504_, lean_object* v_A_505_, lean_object* v_B_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_00_u03c6_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_mathlib_NonUnitalAlgHom_fintypeRange___redArg(v_inst_512_, v_inst_513_, v_00_u03c6_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgHom_fintypeRange___boxed(lean_object* v_R_516_, lean_object* v_A_517_, lean_object* v_B_518_, lean_object* v_inst_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_00_u03c6_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_NonUnitalAlgHom_fintypeRange(v_R_516_, v_A_517_, v_B_518_, v_inst_519_, v_inst_520_, v_inst_521_, v_inst_522_, v_inst_523_, v_inst_524_, v_inst_525_, v_00_u03c6_526_);
lean_dec(v_inst_523_);
lean_dec_ref(v_inst_522_);
lean_dec(v_inst_521_);
lean_dec_ref(v_inst_520_);
lean_dec_ref(v_inst_519_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoin(lean_object* v_R_528_, lean_object* v_A_529_, lean_object* v_inst_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_s_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lean_box(0);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoin___boxed(lean_object* v_R_537_, lean_object* v_A_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_s_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_mathlib_NonUnitalAlgebra_adjoin(v_R_537_, v_A_538_, v_inst_539_, v_inst_540_, v_inst_541_, v_inst_542_, v_inst_543_, v_s_544_);
lean_dec(v_inst_541_);
lean_dec_ref(v_inst_540_);
lean_dec_ref(v_inst_539_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_gi___lam__0(lean_object* v_s_546_, lean_object* v_hs_547_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lean_box(0);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_gi(lean_object* v_R_550_, lean_object* v_A_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_inst_555_, lean_object* v_inst_556_){
_start:
{
lean_object* v___f_557_; 
v___f_557_ = ((lean_object*)(lp_mathlib_NonUnitalAlgebra_gi___closed__0));
return v___f_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_gi___boxed(lean_object* v_R_558_, lean_object* v_A_559_, lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_mathlib_NonUnitalAlgebra_gi(v_R_558_, v_A_559_, v_inst_560_, v_inst_561_, v_inst_562_, v_inst_563_, v_inst_564_);
lean_dec(v_inst_562_);
lean_dec_ref(v_inst_561_);
lean_dec_ref(v_inst_560_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___lam__0(lean_object* v_s_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lean_box(0);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___lam__1(lean_object* v_a_568_, lean_object* v_b_569_){
_start:
{
lean_object* v___x_570_; 
v___x_570_ = lean_box(0);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg(lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_){
_start:
{
lean_object* v___f_578_; lean_object* v___f_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___f_578_ = ((lean_object*)(lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__0));
v___f_579_ = ((lean_object*)(lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__1));
v___x_580_ = lp_mathlib_NonUnitalSubalgebra_instPartialOrder(lean_box(0), lean_box(0), v_inst_575_, v_inst_576_, v_inst_577_);
v___x_581_ = ((lean_object*)(lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___closed__2));
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_580_);
lean_ctor_set(v___x_582_, 1, v___f_579_);
v___x_583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_583_, 0, v___x_582_);
lean_ctor_set(v___x_583_, 1, v___f_579_);
v___x_584_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
lean_ctor_set(v___x_584_, 1, v___f_578_);
lean_ctor_set(v___x_584_, 2, v___f_578_);
lean_ctor_set(v___x_584_, 3, v___x_581_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg___boxed(lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_inst_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg(v_inst_585_, v_inst_586_, v_inst_587_);
lean_dec(v_inst_587_);
lean_dec_ref(v_inst_586_);
lean_dec_ref(v_inst_585_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra(lean_object* v_R_589_, lean_object* v_A_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_){
_start:
{
lean_object* v___x_596_; 
v___x_596_ = lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___redArg(v_inst_591_, v_inst_592_, v_inst_593_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra___boxed(lean_object* v_R_597_, lean_object* v_A_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_inst_603_){
_start:
{
lean_object* v_res_604_; 
v_res_604_ = lp_mathlib_NonUnitalAlgebra_instCompleteLatticeNonUnitalSubalgebra(v_R_597_, v_A_598_, v_inst_599_, v_inst_600_, v_inst_601_, v_inst_602_, v_inst_603_);
lean_dec(v_inst_601_);
lean_dec_ref(v_inst_600_);
lean_dec_ref(v_inst_599_);
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instInhabitedNonUnitalSubalgebra(lean_object* v_R_605_, lean_object* v_A_606_, lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_inst_609_, lean_object* v_inst_610_, lean_object* v_inst_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lean_box(0);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_instInhabitedNonUnitalSubalgebra___boxed(lean_object* v_R_613_, lean_object* v_A_614_, lean_object* v_inst_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_inst_618_, lean_object* v_inst_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib_NonUnitalAlgebra_instInhabitedNonUnitalSubalgebra(v_R_613_, v_A_614_, v_inst_615_, v_inst_616_, v_inst_617_, v_inst_618_, v_inst_619_);
lean_dec(v_inst_617_);
lean_dec_ref(v_inst_616_);
lean_dec_ref(v_inst_615_);
return v_res_620_;
}
}
static lean_object* _init_lp_mathlib_NonUnitalAlgebra_toTop___closed__1(void){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_622_ = ((lean_object*)(lp_mathlib_NonUnitalAlgebra_toTop___closed__0));
v___x_623_ = lp_mathlib_NonUnitalAlgHom_codRestrict___redArg(v___x_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_toTop(lean_object* v_R_624_, lean_object* v_A_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_inst_628_, lean_object* v_inst_629_, lean_object* v_inst_630_){
_start:
{
lean_object* v___x_631_; 
v___x_631_ = lean_obj_once(&lp_mathlib_NonUnitalAlgebra_toTop___closed__1, &lp_mathlib_NonUnitalAlgebra_toTop___closed__1_once, _init_lp_mathlib_NonUnitalAlgebra_toTop___closed__1);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_toTop___boxed(lean_object* v_R_632_, lean_object* v_A_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_inst_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib_NonUnitalAlgebra_toTop(v_R_632_, v_A_633_, v_inst_634_, v_inst_635_, v_inst_636_, v_inst_637_, v_inst_638_);
lean_dec(v_inst_636_);
lean_dec_ref(v_inst_635_);
lean_dec_ref(v_inst_634_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_prod(lean_object* v_R_640_, lean_object* v_A_641_, lean_object* v_B_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_S_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_S_u2081_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lean_box(0);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_prod___boxed(lean_object* v_R_651_, lean_object* v_A_652_, lean_object* v_B_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_S_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_S_u2081_660_){
_start:
{
lean_object* v_res_661_; 
v_res_661_ = lp_mathlib_NonUnitalSubalgebra_prod(v_R_651_, v_A_652_, v_B_653_, v_inst_654_, v_inst_655_, v_inst_656_, v_S_657_, v_inst_658_, v_inst_659_, v_S_u2081_660_);
lean_dec(v_inst_659_);
lean_dec_ref(v_inst_658_);
lean_dec(v_inst_656_);
lean_dec_ref(v_inst_655_);
lean_dec_ref(v_inst_654_);
return v_res_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_inclusion(lean_object* v_R_663_, lean_object* v_A_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_S_668_, lean_object* v_T_669_, lean_object* v_h_670_){
_start:
{
lean_object* v___x_671_; 
v___x_671_ = ((lean_object*)(lp_mathlib_NonUnitalSubalgebra_inclusion___closed__0));
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_inclusion___boxed(lean_object* v_R_672_, lean_object* v_A_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_S_677_, lean_object* v_T_678_, lean_object* v_h_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_mathlib_NonUnitalSubalgebra_inclusion(v_R_672_, v_A_673_, v_inst_674_, v_inst_675_, v_inst_676_, v_S_677_, v_T_678_, v_h_679_);
lean_dec(v_inst_676_);
lean_dec_ref(v_inst_675_);
lean_dec_ref(v_inst_674_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center(lean_object* v_R_681_, lean_object* v_A_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_){
_start:
{
lean_object* v___x_688_; 
v___x_688_ = lean_box(0);
return v___x_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center___boxed(lean_object* v_R_689_, lean_object* v_A_690_, lean_object* v_inst_691_, lean_object* v_inst_692_, lean_object* v_inst_693_, lean_object* v_inst_694_, lean_object* v_inst_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_mathlib_NonUnitalSubalgebra_center(v_R_689_, v_A_690_, v_inst_691_, v_inst_692_, v_inst_693_, v_inst_694_, v_inst_695_);
lean_dec(v_inst_693_);
lean_dec_ref(v_inst_692_);
lean_dec_ref(v_inst_691_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring___redArg(lean_object* v_inst_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_697_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring(lean_object* v_R_699_, lean_object* v_A_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_inst_704_, lean_object* v_inst_705_){
_start:
{
lean_object* v___x_706_; 
v___x_706_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_702_);
return v___x_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring___boxed(lean_object* v_R_707_, lean_object* v_A_708_, lean_object* v_inst_709_, lean_object* v_inst_710_, lean_object* v_inst_711_, lean_object* v_inst_712_, lean_object* v_inst_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommSemiring(v_R_707_, v_A_708_, v_inst_709_, v_inst_710_, v_inst_711_, v_inst_712_, v_inst_713_);
lean_dec(v_inst_711_);
lean_dec_ref(v_inst_709_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___redArg(lean_object* v_inst_715_, lean_object* v_a_716_){
_start:
{
lean_object* v_toAddCommGroup_717_; lean_object* v___x_718_; lean_object* v_toNeg_719_; lean_object* v___x_720_; 
v_toAddCommGroup_717_ = lean_ctor_get(v_inst_715_, 0);
v___x_718_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_toAddCommGroup_717_);
v_toNeg_719_ = lean_ctor_get(v___x_718_, 1);
lean_inc(v_toNeg_719_);
lean_dec_ref(v___x_718_);
v___x_720_ = lean_apply_1(v_toNeg_719_, v_a_716_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___redArg___boxed(lean_object* v_inst_721_, lean_object* v_a_722_){
_start:
{
lean_object* v_res_723_; 
v_res_723_ = lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___redArg(v_inst_721_, v_a_722_);
lean_dec_ref(v_inst_721_);
return v_res_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1(lean_object* v_R_724_, lean_object* v_inst_725_, lean_object* v_A_726_, lean_object* v_inst_727_, lean_object* v_inst_728_, lean_object* v_inst_729_, lean_object* v_inst_730_, lean_object* v_a_731_){
_start:
{
lean_object* v_toAddCommGroup_732_; lean_object* v___x_733_; lean_object* v_toNeg_734_; lean_object* v___x_735_; 
v_toAddCommGroup_732_ = lean_ctor_get(v_inst_727_, 0);
v___x_733_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_toAddCommGroup_732_);
v_toNeg_734_ = lean_ctor_get(v___x_733_, 1);
lean_inc(v_toNeg_734_);
lean_dec_ref(v___x_733_);
v___x_735_ = lean_apply_1(v_toNeg_734_, v_a_731_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___boxed(lean_object* v_R_736_, lean_object* v_inst_737_, lean_object* v_A_738_, lean_object* v_inst_739_, lean_object* v_inst_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_a_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1(v_R_736_, v_inst_737_, v_A_738_, v_inst_739_, v_inst_740_, v_inst_741_, v_inst_742_, v_a_743_);
lean_dec(v_inst_740_);
lean_dec_ref(v_inst_739_);
lean_dec_ref(v_inst_737_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3___redArg(lean_object* v_inst_745_, lean_object* v_a_746_, lean_object* v_b_747_){
_start:
{
lean_object* v_toAddCommGroup_748_; lean_object* v_toSub_749_; lean_object* v___x_750_; 
v_toAddCommGroup_748_ = lean_ctor_get(v_inst_745_, 0);
lean_inc_ref(v_toAddCommGroup_748_);
lean_dec_ref(v_inst_745_);
v_toSub_749_ = lean_ctor_get(v_toAddCommGroup_748_, 2);
lean_inc(v_toSub_749_);
lean_dec_ref(v_toAddCommGroup_748_);
v___x_750_ = lean_apply_2(v_toSub_749_, v_a_746_, v_b_747_);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3(lean_object* v_R_751_, lean_object* v_inst_752_, lean_object* v_A_753_, lean_object* v_inst_754_, lean_object* v_inst_755_, lean_object* v_inst_756_, lean_object* v_inst_757_, lean_object* v_a_758_, lean_object* v_b_759_){
_start:
{
lean_object* v_toAddCommGroup_760_; lean_object* v_toSub_761_; lean_object* v___x_762_; 
v_toAddCommGroup_760_ = lean_ctor_get(v_inst_754_, 0);
lean_inc_ref(v_toAddCommGroup_760_);
lean_dec_ref(v_inst_754_);
v_toSub_761_ = lean_ctor_get(v_toAddCommGroup_760_, 2);
lean_inc(v_toSub_761_);
lean_dec_ref(v_toAddCommGroup_760_);
v___x_762_ = lean_apply_2(v_toSub_761_, v_a_758_, v_b_759_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3___boxed(lean_object* v_R_763_, lean_object* v_inst_764_, lean_object* v_A_765_, lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_a_770_, lean_object* v_b_771_){
_start:
{
lean_object* v_res_772_; 
v_res_772_ = lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3(v_R_763_, v_inst_764_, v_A_765_, v_inst_766_, v_inst_767_, v_inst_768_, v_inst_769_, v_a_770_, v_b_771_);
lean_dec(v_inst_767_);
lean_dec_ref(v_inst_764_);
return v_res_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5___redArg(lean_object* v_inst_773_, lean_object* v_n_774_, lean_object* v_x_775_){
_start:
{
lean_object* v_toAddCommGroup_776_; lean_object* v_toZSMul_777_; lean_object* v___x_778_; 
v_toAddCommGroup_776_ = lean_ctor_get(v_inst_773_, 0);
lean_inc_ref(v_toAddCommGroup_776_);
lean_dec_ref(v_inst_773_);
v_toZSMul_777_ = lean_ctor_get(v_toAddCommGroup_776_, 3);
lean_inc(v_toZSMul_777_);
lean_dec_ref(v_toAddCommGroup_776_);
v___x_778_ = lean_apply_2(v_toZSMul_777_, v_n_774_, v_x_775_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5(lean_object* v_R_779_, lean_object* v_inst_780_, lean_object* v_A_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_inst_784_, lean_object* v_inst_785_, lean_object* v_n_786_, lean_object* v_x_787_){
_start:
{
lean_object* v_toAddCommGroup_788_; lean_object* v_toZSMul_789_; lean_object* v___x_790_; 
v_toAddCommGroup_788_ = lean_ctor_get(v_inst_782_, 0);
lean_inc_ref(v_toAddCommGroup_788_);
lean_dec_ref(v_inst_782_);
v_toZSMul_789_ = lean_ctor_get(v_toAddCommGroup_788_, 3);
lean_inc(v_toZSMul_789_);
lean_dec_ref(v_toAddCommGroup_788_);
v___x_790_ = lean_apply_2(v_toZSMul_789_, v_n_786_, v_x_787_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5___boxed(lean_object* v_R_791_, lean_object* v_inst_792_, lean_object* v_A_793_, lean_object* v_inst_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_n_798_, lean_object* v_x_799_){
_start:
{
lean_object* v_res_800_; 
v_res_800_ = lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5(v_R_791_, v_inst_792_, v_A_793_, v_inst_794_, v_inst_795_, v_inst_796_, v_inst_797_, v_n_798_, v_x_799_);
lean_dec(v_inst_795_);
lean_dec_ref(v_inst_792_);
return v_res_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___redArg(lean_object* v_inst_801_, lean_object* v_inst_802_, lean_object* v_inst_803_){
_start:
{
lean_object* v_toAddCommGroup_804_; lean_object* v_toAddMonoid_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_828_; 
v_toAddCommGroup_804_ = lean_ctor_get(v_inst_802_, 0);
lean_inc_ref(v_toAddCommGroup_804_);
v_toAddMonoid_805_ = lean_ctor_get(v_toAddCommGroup_804_, 0);
v_isSharedCheck_828_ = !lean_is_exclusive(v_toAddCommGroup_804_);
if (v_isSharedCheck_828_ == 0)
{
lean_object* v_unused_829_; lean_object* v_unused_830_; lean_object* v_unused_831_; 
v_unused_829_ = lean_ctor_get(v_toAddCommGroup_804_, 3);
lean_dec(v_unused_829_);
v_unused_830_ = lean_ctor_get(v_toAddCommGroup_804_, 2);
lean_dec(v_unused_830_);
v_unused_831_ = lean_ctor_get(v_toAddCommGroup_804_, 1);
lean_dec(v_unused_831_);
v___x_807_ = v_toAddCommGroup_804_;
v_isShared_808_ = v_isSharedCheck_828_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_toAddMonoid_805_);
lean_dec(v_toAddCommGroup_804_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_828_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_815_; 
lean_inc_ref_n(v_inst_802_, 3);
v___x_809_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_802_);
v___x_810_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_toAddMonoid_805_);
lean_inc_n(v_inst_803_, 2);
lean_inc_ref_n(v_inst_801_, 2);
v___x_811_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__1___boxed), 8, 7);
lean_closure_set(v___x_811_, 0, lean_box(0));
lean_closure_set(v___x_811_, 1, v_inst_801_);
lean_closure_set(v___x_811_, 2, lean_box(0));
lean_closure_set(v___x_811_, 3, v_inst_802_);
lean_closure_set(v___x_811_, 4, v_inst_803_);
lean_closure_set(v___x_811_, 5, lean_box(0));
lean_closure_set(v___x_811_, 6, lean_box(0));
v___x_812_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__3___boxed), 9, 7);
lean_closure_set(v___x_812_, 0, lean_box(0));
lean_closure_set(v___x_812_, 1, v_inst_801_);
lean_closure_set(v___x_812_, 2, lean_box(0));
lean_closure_set(v___x_812_, 3, v_inst_802_);
lean_closure_set(v___x_812_, 4, v_inst_803_);
lean_closure_set(v___x_812_, 5, lean_box(0));
lean_closure_set(v___x_812_, 6, lean_box(0));
v___x_813_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___aux__5___boxed), 9, 7);
lean_closure_set(v___x_813_, 0, lean_box(0));
lean_closure_set(v___x_813_, 1, v_inst_801_);
lean_closure_set(v___x_813_, 2, lean_box(0));
lean_closure_set(v___x_813_, 3, v_inst_802_);
lean_closure_set(v___x_813_, 4, v_inst_803_);
lean_closure_set(v___x_813_, 5, lean_box(0));
lean_closure_set(v___x_813_, 6, lean_box(0));
if (v_isShared_808_ == 0)
{
lean_ctor_set(v___x_807_, 3, v___x_813_);
lean_ctor_set(v___x_807_, 2, v___x_812_);
lean_ctor_set(v___x_807_, 1, v___x_811_);
lean_ctor_set(v___x_807_, 0, v___x_810_);
v___x_815_ = v___x_807_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v___x_810_);
lean_ctor_set(v_reuseFailAlloc_827_, 1, v___x_811_);
lean_ctor_set(v_reuseFailAlloc_827_, 2, v___x_812_);
lean_ctor_set(v_reuseFailAlloc_827_, 3, v___x_813_);
v___x_815_ = v_reuseFailAlloc_827_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v_toMul_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_825_; 
v___x_816_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v___x_809_);
v___x_817_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_816_);
v_toMul_818_ = lean_ctor_get(v___x_817_, 0);
v_isSharedCheck_825_ = !lean_is_exclusive(v___x_817_);
if (v_isSharedCheck_825_ == 0)
{
lean_object* v_unused_826_; 
v_unused_826_ = lean_ctor_get(v___x_817_, 1);
lean_dec(v_unused_826_);
v___x_820_ = v___x_817_;
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_toMul_818_);
lean_dec(v___x_817_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_823_; 
if (v_isShared_821_ == 0)
{
lean_ctor_set(v___x_820_, 1, v_toMul_818_);
lean_ctor_set(v___x_820_, 0, v___x_815_);
v___x_823_ = v___x_820_;
goto v_reusejp_822_;
}
else
{
lean_object* v_reuseFailAlloc_824_; 
v_reuseFailAlloc_824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_824_, 0, v___x_815_);
lean_ctor_set(v_reuseFailAlloc_824_, 1, v_toMul_818_);
v___x_823_ = v_reuseFailAlloc_824_;
goto v_reusejp_822_;
}
v_reusejp_822_:
{
return v___x_823_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing(lean_object* v_R_832_, lean_object* v_inst_833_, lean_object* v_A_834_, lean_object* v_inst_835_, lean_object* v_inst_836_, lean_object* v_inst_837_, lean_object* v_inst_838_){
_start:
{
lean_object* v___x_839_; 
v___x_839_ = lp_mathlib_NonUnitalSubalgebra_center_instNonUnitalCommRing___redArg(v_inst_833_, v_inst_835_, v_inst_836_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_centralizer(lean_object* v_R_840_, lean_object* v_A_841_, lean_object* v_inst_842_, lean_object* v_inst_843_, lean_object* v_inst_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_s_847_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = lean_box(0);
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubalgebra_centralizer___boxed(lean_object* v_R_849_, lean_object* v_A_850_, lean_object* v_inst_851_, lean_object* v_inst_852_, lean_object* v_inst_853_, lean_object* v_inst_854_, lean_object* v_inst_855_, lean_object* v_s_856_){
_start:
{
lean_object* v_res_857_; 
v_res_857_ = lp_mathlib_NonUnitalSubalgebra_centralizer(v_R_849_, v_A_850_, v_inst_851_, v_inst_852_, v_inst_853_, v_inst_854_, v_inst_855_, v_s_856_);
lean_dec(v_inst_853_);
lean_dec_ref(v_inst_852_);
lean_dec_ref(v_inst_851_);
return v_res_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm___redArg(lean_object* v_inst_858_){
_start:
{
lean_object* v___x_859_; 
v___x_859_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_858_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm(lean_object* v_R_860_, lean_object* v_A_861_, lean_object* v_inst_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_s_867_, lean_object* v_hcomm_868_){
_start:
{
lean_object* v___x_869_; 
v___x_869_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_863_);
return v___x_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm___boxed(lean_object* v_R_870_, lean_object* v_A_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_s_877_, lean_object* v_hcomm_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommSemiringOfComm(v_R_870_, v_A_871_, v_inst_872_, v_inst_873_, v_inst_874_, v_inst_875_, v_inst_876_, v_s_877_, v_hcomm_878_);
lean_dec(v_inst_874_);
lean_dec_ref(v_inst_872_);
return v_res_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm___redArg(lean_object* v_inst_880_){
_start:
{
lean_object* v___x_881_; 
v___x_881_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_880_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm(lean_object* v_R_882_, lean_object* v_A_883_, lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_inst_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_s_889_, lean_object* v_hcomm_890_){
_start:
{
lean_object* v___x_891_; 
v___x_891_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_885_);
return v___x_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm___boxed(lean_object* v_R_892_, lean_object* v_A_893_, lean_object* v_inst_894_, lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_inst_898_, lean_object* v_s_899_, lean_object* v_hcomm_900_){
_start:
{
lean_object* v_res_901_; 
v_res_901_ = lp_mathlib_NonUnitalAlgebra_adjoinNonUnitalCommRingOfComm(v_R_892_, v_A_893_, v_inst_894_, v_inst_895_, v_inst_896_, v_inst_897_, v_inst_898_, v_s_899_, v_hcomm_900_);
lean_dec(v_inst_896_);
lean_dec_ref(v_inst_894_);
return v_res_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring___redArg(lean_object* v_S_902_){
_start:
{
return v_S_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring(lean_object* v_R_903_, lean_object* v_inst_904_, lean_object* v_S_905_){
_start:
{
return v_S_905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring___boxed(lean_object* v_R_906_, lean_object* v_inst_907_, lean_object* v_S_908_){
_start:
{
lean_object* v_res_909_; 
v_res_909_ = lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubsemiring(v_R_906_, v_inst_907_, v_S_908_);
lean_dec_ref(v_inst_907_);
return v_res_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring___redArg(lean_object* v_S_910_){
_start:
{
return v_S_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring(lean_object* v_R_911_, lean_object* v_inst_912_, lean_object* v_S_913_){
_start:
{
return v_S_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring___boxed(lean_object* v_R_914_, lean_object* v_inst_915_, lean_object* v_S_916_){
_start:
{
lean_object* v_res_917_; 
v_res_917_ = lp_mathlib_nonUnitalSubalgebraOfNonUnitalSubring(v_R_914_, v_inst_915_, v_S_916_);
lean_dec_ref(v_inst_915_);
return v_res_917_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_UnionLift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_UnionLift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_UnionLift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_UnionLift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_NonUnitalSubalgebra(builtin);
}
#ifdef __cplusplus
}
#endif
