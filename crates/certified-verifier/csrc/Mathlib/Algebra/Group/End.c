// Lean compiler output
// Module: Mathlib.Algebra.Group.End
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.TypeTags public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Algebra.Group.Prod public import Mathlib.Algebra.Group.Units.Equiv public import Mathlib.Data.Set.Basic public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Register
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_Perm_extendDomain___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Perm_subtypeCongr___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddEquiv_toMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulEquiv_toAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_toHomUnits___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidEnd___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidEnd___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instMonoidEnd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instMonoidEnd___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instMonoidEnd___closed__0 = (const lean_object*)&lp_mathlib_instMonoidEnd___closed__0_value;
static const lean_closure_object lp_mathlib_instMonoidEnd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instMonoidEnd___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instMonoidEnd___closed__1 = (const lean_object*)&lp_mathlib_instMonoidEnd___closed__1_value;
static const lean_closure_object lp_mathlib_instMonoidEnd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_instMonoidEnd___closed__2 = (const lean_object*)&lp_mathlib_instMonoidEnd___closed__2_value;
static const lean_ctor_object lp_mathlib_instMonoidEnd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instMonoidEnd___closed__2_value),((lean_object*)&lp_mathlib_instMonoidEnd___closed__0_value),((lean_object*)&lp_mathlib_instMonoidEnd___closed__1_value)}};
static const lean_object* lp_mathlib_instMonoidEnd___closed__3 = (const lean_object*)&lp_mathlib_instMonoidEnd___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_instMonoidEnd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedEnd(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Perm_instOne___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Perm_instOne___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instOne(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instMul___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_instMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_instMul___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_instMul___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_instMul___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instMul(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_instInv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_symm, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_Perm_instInv___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_instInv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instInv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instPowNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instPowNat___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_instPowNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_instPowNat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_instPowNat___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_instPowNat___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_Perm_instPowNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_instPowNat___lam__1, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Perm_instPowNat___closed__0_value)} };
static const lean_object* lp_mathlib_Equiv_Perm_instPowNat___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Perm_instPowNat___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instPowNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_permGroup___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_permGroup___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_permGroup___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_permGroup___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_Perm_permGroup___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Perm_permGroup___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_Perm_permGroup___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Perm_permGroup___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_Perm_permGroup___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Perm_permGroup___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_Perm_permGroup___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Perm_permGroup___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_equivUnitsEnd___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_equivUnitsEnd___lam__2, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Perm_instPowNat___closed__0_value)} };
static const lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__1_value),((lean_object*)&lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__2 = (const lean_object*)&lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd(lean_object*);
static lean_once_cell_t lp_mathlib_MonoidHom_toHomPerm___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidHom_toHomPerm___redArg___closed__0;
static lean_once_cell_t lp_mathlib_MonoidHom_toHomPerm___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidHom_toHomPerm___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongrHom___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_sumCongrHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_sumCongrHom___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_sumCongrHom___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_sumCongrHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongrHom(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_sigmaCongrRightHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaCongrRight___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRightHom___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_sigmaCongrRightHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRightHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongrHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongrHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongrHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongrHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongrHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomainHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomainHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomainHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSubtype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSubtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Perm_subtypePerm___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instGroup___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulAut_instGroup___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulAut_instGroup___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulAut_instGroup___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulAut_instGroup___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_MulAut_instGroup___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAut_instGroup___redArg___closed__1;
static lean_once_cell_t lp_mathlib_MulAut_instGroup___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAut_instGroup___redArg___closed__2;
static lean_once_cell_t lp_mathlib_MulAut_instGroup___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAut_instGroup___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instGroup(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddAut_instAddGroup___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAut_instAddGroup___redArg___closed__0;
static lean_once_cell_t lp_mathlib_AddAut_instAddGroup___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAut_instAddGroup___redArg___closed__1;
static lean_once_cell_t lp_mathlib_AddAut_instAddGroup___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddAut_instAddGroup___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instAddGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulAut_toPerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulAut_toPerm___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulAut_toPerm___closed__0 = (const lean_object*)&lp_mathlib_MulAut_toPerm___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_toPerm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_toPerm___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMonoidEnd___lam__0(lean_object* v_x1_1_, lean_object* v_x2_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_apply_1(v_x2_2_, v___y_3_);
v___x_5_ = lean_apply_1(v_x1_1_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidEnd___lam__1(lean_object* v_n_6_, lean_object* v_f_7_, lean_object* v___y_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Nat_iterate___redArg(v_f_7_, v_n_6_, v___y_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMonoidEnd(lean_object* v_00_u03b1_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = ((lean_object*)(lp_mathlib_instMonoidEnd___closed__3));
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedEnd(lean_object* v_00_u03b1_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = ((lean_object*)(lp_mathlib_instMonoidEnd___closed__2));
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_instOne___closed__0(void){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instOne(lean_object* v_00_u03b1_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instMul___lam__0(lean_object* v_f_24_, lean_object* v_g_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Equiv_trans___redArg(v_g_25_, v_f_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instMul(lean_object* v_00_u03b1_28_){
_start:
{
lean_object* v___f_29_; 
v___f_29_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instMul___closed__0));
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instInv(lean_object* v_00_u03b1_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instInv___closed__0));
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instPowNat___lam__0(lean_object* v_self_33_, lean_object* v___y_34_){
_start:
{
lean_object* v_toFun_35_; lean_object* v___x_36_; 
v_toFun_35_ = lean_ctor_get(v_self_33_, 0);
lean_inc(v_toFun_35_);
lean_dec_ref(v_self_33_);
v___x_36_ = lean_apply_1(v_toFun_35_, v___y_34_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instPowNat___lam__1(lean_object* v___f_37_, lean_object* v_f_38_, lean_object* v_n_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
lean_inc(v___f_37_);
lean_inc_ref(v_f_38_);
v___x_40_ = lean_apply_1(v___f_37_, v_f_38_);
lean_inc(v_n_39_);
v___x_41_ = lean_alloc_closure((void*)(lp_mathlib_Nat_iterate), 4, 3);
lean_closure_set(v___x_41_, 0, lean_box(0));
lean_closure_set(v___x_41_, 1, v___x_40_);
lean_closure_set(v___x_41_, 2, v_n_39_);
v___x_42_ = lp_mathlib_Equiv_symm___redArg(v_f_38_);
v___x_43_ = lean_apply_1(v___f_37_, v___x_42_);
v___x_44_ = lean_alloc_closure((void*)(lp_mathlib_Nat_iterate), 4, 3);
lean_closure_set(v___x_44_, 0, lean_box(0));
lean_closure_set(v___x_44_, 1, v___x_43_);
lean_closure_set(v___x_44_, 2, v_n_39_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_41_);
lean_ctor_set(v___x_45_, 1, v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instPowNat(lean_object* v_00_u03b1_49_){
_start:
{
lean_object* v___f_50_; 
v___f_50_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instPowNat___closed__1));
return v___f_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup___lam__0(lean_object* v_f_51_, lean_object* v___y_52_){
_start:
{
lean_object* v_toFun_53_; lean_object* v___x_54_; 
v_toFun_53_ = lean_ctor_get(v_f_51_, 0);
lean_inc(v_toFun_53_);
lean_dec_ref(v_f_51_);
v___x_54_ = lean_apply_1(v_toFun_53_, v___y_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup___lam__1(lean_object* v___x_55_, lean_object* v___y_56_){
_start:
{
lean_object* v_toFun_57_; lean_object* v___x_58_; 
v_toFun_57_ = lean_ctor_get(v___x_55_, 0);
lean_inc(v_toFun_57_);
lean_dec_ref(v___x_55_);
v___x_58_ = lean_apply_1(v_toFun_57_, v___y_56_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup___lam__2(lean_object* v_n_59_, lean_object* v_f_60_){
_start:
{
lean_object* v___f_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___f_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
lean_inc_ref(v_f_60_);
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_permGroup___lam__0), 2, 1);
lean_closure_set(v___f_61_, 0, v_f_60_);
lean_inc(v_n_59_);
v___x_62_ = lean_alloc_closure((void*)(lp_mathlib_Nat_iterate), 4, 3);
lean_closure_set(v___x_62_, 0, lean_box(0));
lean_closure_set(v___x_62_, 1, v___f_61_);
lean_closure_set(v___x_62_, 2, v_n_59_);
v___x_63_ = lp_mathlib_Equiv_symm___redArg(v_f_60_);
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_permGroup___lam__1), 2, 1);
lean_closure_set(v___f_64_, 0, v___x_63_);
v___x_65_ = lean_alloc_closure((void*)(lp_mathlib_Nat_iterate), 4, 3);
lean_closure_set(v___x_65_, 0, lean_box(0));
lean_closure_set(v___x_65_, 1, v___f_64_);
lean_closure_set(v___x_65_, 2, v_n_59_);
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_62_);
lean_ctor_set(v___x_66_, 1, v___x_65_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_permGroup___closed__1(void){
_start:
{
lean_object* v___f_68_; lean_object* v___f_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___f_68_ = ((lean_object*)(lp_mathlib_Equiv_Perm_permGroup___closed__0));
v___f_69_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instMul___closed__0));
v___x_70_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_71_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v___f_69_);
lean_ctor_set(v___x_71_, 2, v___f_68_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_permGroup___closed__2(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_72_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instInv___closed__0));
v___x_73_ = lean_obj_once(&lp_mathlib_Equiv_Perm_permGroup___closed__1, &lp_mathlib_Equiv_Perm_permGroup___closed__1_once, _init_lp_mathlib_Equiv_Perm_permGroup___closed__1);
v___x_74_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_74_, 0, lean_box(0));
lean_closure_set(v___x_74_, 1, v___x_73_);
lean_closure_set(v___x_74_, 2, v___x_72_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_permGroup___closed__3(void){
_start:
{
lean_object* v___f_75_; lean_object* v___x_76_; lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___f_75_ = ((lean_object*)(lp_mathlib_Equiv_Perm_permGroup___closed__0));
v___x_76_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instInv___closed__0));
v___f_77_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instMul___closed__0));
v___x_78_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_79_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_79_, 0, lean_box(0));
lean_closure_set(v___x_79_, 1, v___x_78_);
lean_closure_set(v___x_79_, 2, v___f_77_);
lean_closure_set(v___x_79_, 3, v___x_76_);
lean_closure_set(v___x_79_, 4, v___f_75_);
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Perm_permGroup___closed__4(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_80_ = lean_obj_once(&lp_mathlib_Equiv_Perm_permGroup___closed__3, &lp_mathlib_Equiv_Perm_permGroup___closed__3_once, _init_lp_mathlib_Equiv_Perm_permGroup___closed__3);
v___x_81_ = lean_obj_once(&lp_mathlib_Equiv_Perm_permGroup___closed__2, &lp_mathlib_Equiv_Perm_permGroup___closed__2_once, _init_lp_mathlib_Equiv_Perm_permGroup___closed__2);
v___x_82_ = ((lean_object*)(lp_mathlib_Equiv_Perm_instInv___closed__0));
v___x_83_ = lean_obj_once(&lp_mathlib_Equiv_Perm_permGroup___closed__1, &lp_mathlib_Equiv_Perm_permGroup___closed__1_once, _init_lp_mathlib_Equiv_Perm_permGroup___closed__1);
v___x_84_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v___x_82_);
lean_ctor_set(v___x_84_, 2, v___x_81_);
lean_ctor_set(v___x_84_, 3, v___x_80_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_permGroup(lean_object* v_00_u03b1_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_obj_once(&lp_mathlib_Equiv_Perm_permGroup___closed__4, &lp_mathlib_Equiv_Perm_permGroup___closed__4_once, _init_lp_mathlib_Equiv_Perm_permGroup___closed__4);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___lam__0(lean_object* v_u_87_){
_start:
{
lean_object* v_val_88_; lean_object* v_inv_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_96_; 
v_val_88_ = lean_ctor_get(v_u_87_, 0);
v_inv_89_ = lean_ctor_get(v_u_87_, 1);
v_isSharedCheck_96_ = !lean_is_exclusive(v_u_87_);
if (v_isSharedCheck_96_ == 0)
{
v___x_91_ = v_u_87_;
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_inv_89_);
lean_inc(v_val_88_);
lean_dec(v_u_87_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_val_88_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_inv_89_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd___lam__2(lean_object* v___f_97_, lean_object* v_e_98_){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
lean_inc(v___f_97_);
lean_inc_ref(v_e_98_);
v___x_99_ = lean_apply_1(v___f_97_, v_e_98_);
v___x_100_ = lp_mathlib_Equiv_symm___redArg(v_e_98_);
v___x_101_ = lean_apply_1(v___f_97_, v___x_100_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_99_);
lean_ctor_set(v___x_102_, 1, v___x_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_equivUnitsEnd(lean_object* v_00_u03b1_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = ((lean_object*)(lp_mathlib_Equiv_Perm_equivUnitsEnd___closed__2));
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_MonoidHom_toHomPerm___redArg___closed__0(void){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_Equiv_Perm_equivUnitsEnd(lean_box(0));
return v___x_111_;
}
}
static lean_object* _init_lp_mathlib_MonoidHom_toHomPerm___redArg___closed__1(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_112_ = lean_obj_once(&lp_mathlib_MonoidHom_toHomPerm___redArg___closed__0, &lp_mathlib_MonoidHom_toHomPerm___redArg___closed__0_once, _init_lp_mathlib_MonoidHom_toHomPerm___redArg___closed__0);
v___x_113_ = lp_mathlib_Equiv_symm___redArg(v___x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm___redArg(lean_object* v_inst_114_, lean_object* v_f_115_){
_start:
{
lean_object* v___x_116_; lean_object* v_toFun_117_; lean_object* v___x_118_; lean_object* v___f_119_; 
v___x_116_ = lean_obj_once(&lp_mathlib_MonoidHom_toHomPerm___redArg___closed__1, &lp_mathlib_MonoidHom_toHomPerm___redArg___closed__1_once, _init_lp_mathlib_MonoidHom_toHomPerm___redArg___closed__1);
v_toFun_117_ = lean_ctor_get(v___x_116_, 0);
v___x_118_ = lp_mathlib_MonoidHom_toHomUnits___redArg(v_inst_114_, v_f_115_);
lean_inc(v_toFun_117_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_119_, 0, v___x_118_);
lean_closure_set(v___f_119_, 1, v_toFun_117_);
return v___f_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm___redArg___boxed(lean_object* v_inst_120_, lean_object* v_f_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_MonoidHom_toHomPerm___redArg(v_inst_120_, v_f_121_);
lean_dec_ref(v_inst_120_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm(lean_object* v_00_u03b1_123_, lean_object* v_G_124_, lean_object* v_inst_125_, lean_object* v_f_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_MonoidHom_toHomPerm___redArg(v_inst_125_, v_f_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toHomPerm___boxed(lean_object* v_00_u03b1_128_, lean_object* v_G_129_, lean_object* v_inst_130_, lean_object* v_f_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_MonoidHom_toHomPerm(v_00_u03b1_128_, v_G_129_, v_inst_130_, v_f_131_);
lean_dec_ref(v_inst_130_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongrHom___lam__0(lean_object* v_a_133_){
_start:
{
lean_object* v_fst_134_; lean_object* v_snd_135_; lean_object* v___x_136_; 
v_fst_134_ = lean_ctor_get(v_a_133_, 0);
lean_inc(v_fst_134_);
v_snd_135_ = lean_ctor_get(v_a_133_, 1);
lean_inc(v_snd_135_);
lean_dec_ref(v_a_133_);
v___x_136_ = lp_mathlib_Equiv_sumCongr___redArg(v_fst_134_, v_snd_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongrHom(lean_object* v_00_u03b1_138_, lean_object* v_00_u03b2_139_){
_start:
{
lean_object* v___f_140_; 
v___f_140_ = ((lean_object*)(lp_mathlib_Equiv_Perm_sumCongrHom___closed__0));
return v___f_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sigmaCongrRightHom(lean_object* v_00_u03b1_142_, lean_object* v_00_u03b2_143_){
_start:
{
lean_object* v___f_144_; 
v___f_144_ = ((lean_object*)(lp_mathlib_Equiv_Perm_sigmaCongrRightHom___closed__0));
return v___f_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongrHom___redArg___lam__0(lean_object* v_inst_145_, lean_object* v_pair_146_){
_start:
{
lean_object* v_fst_147_; lean_object* v_snd_148_; lean_object* v___x_149_; 
v_fst_147_ = lean_ctor_get(v_pair_146_, 0);
lean_inc(v_fst_147_);
v_snd_148_ = lean_ctor_get(v_pair_146_, 1);
lean_inc(v_snd_148_);
lean_dec_ref(v_pair_146_);
v___x_149_ = lp_mathlib_Equiv_Perm_subtypeCongr___redArg(v_inst_145_, v_fst_147_, v_snd_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongrHom___redArg(lean_object* v_inst_150_){
_start:
{
lean_object* v___f_151_; 
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_subtypeCongrHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_151_, 0, v_inst_150_);
return v___f_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongrHom(lean_object* v_00_u03b1_152_, lean_object* v_p_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___f_155_; 
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_subtypeCongrHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_155_, 0, v_inst_154_);
return v___f_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongrHom___redArg(lean_object* v_e_156_){
_start:
{
lean_object* v___x_157_; 
lean_inc_ref(v_e_156_);
v___x_157_ = lp_mathlib_Equiv_equivCongr___redArg(v_e_156_, v_e_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_permCongrHom(lean_object* v_00_u03b1_158_, lean_object* v_00_u03b2_159_, lean_object* v_e_160_){
_start:
{
lean_object* v___x_161_; 
lean_inc_ref(v_e_160_);
v___x_161_ = lp_mathlib_Equiv_equivCongr___redArg(v_e_160_, v_e_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomainHom___redArg___lam__0(lean_object* v_inst_162_, lean_object* v_f_163_, lean_object* v_e_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Equiv_Perm_extendDomain___redArg(v_e_164_, v_inst_162_, v_f_163_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomainHom___redArg(lean_object* v_inst_166_, lean_object* v_f_167_){
_start:
{
lean_object* v___f_168_; 
v___f_168_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_extendDomainHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_168_, 0, v_inst_166_);
lean_closure_set(v___f_168_, 1, v_f_167_);
return v___f_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomainHom(lean_object* v_00_u03b1_169_, lean_object* v_00_u03b2_170_, lean_object* v_p_171_, lean_object* v_inst_172_, lean_object* v_f_173_){
_start:
{
lean_object* v___f_174_; 
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_extendDomainHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_174_, 0, v_inst_172_);
lean_closure_set(v___f_174_, 1, v_f_173_);
return v___f_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg___lam__0(lean_object* v_f_175_, lean_object* v_x_176_){
_start:
{
lean_object* v_toFun_177_; lean_object* v___x_178_; 
v_toFun_177_ = lean_ctor_get(v_f_175_, 0);
lean_inc(v_toFun_177_);
lean_dec_ref(v_f_175_);
v___x_178_ = lean_apply_1(v_toFun_177_, v_x_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg___lam__1(lean_object* v_f_179_, lean_object* v_x_180_){
_start:
{
lean_object* v___x_181_; lean_object* v_toFun_182_; lean_object* v___x_183_; 
v___x_181_ = lp_mathlib_Equiv_symm___redArg(v_f_179_);
v_toFun_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_toFun_182_);
lean_dec_ref(v___x_181_);
v___x_183_ = lean_apply_1(v_toFun_182_, v_x_180_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg(lean_object* v_f_184_){
_start:
{
lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___x_187_; 
lean_inc_ref(v_f_184_);
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_subtypePerm___redArg___lam__0), 2, 1);
lean_closure_set(v___f_185_, 0, v_f_184_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_subtypePerm___redArg___lam__1), 2, 1);
lean_closure_set(v___f_186_, 0, v_f_184_);
v___x_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_187_, 0, v___f_185_);
lean_ctor_set(v___x_187_, 1, v___f_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypePerm(lean_object* v_00_u03b1_188_, lean_object* v_p_189_, lean_object* v_f_190_, lean_object* v_h_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_Equiv_Perm_subtypePerm___redArg(v_f_190_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSubtype___redArg___lam__0(lean_object* v_inst_193_, lean_object* v_f_194_){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_195_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_196_ = lp_mathlib_Equiv_Perm_extendDomain___redArg(v_f_194_, v_inst_193_, v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSubtype___redArg(lean_object* v_inst_197_){
_start:
{
lean_object* v___f_198_; 
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_ofSubtype___redArg___lam__0), 2, 1);
lean_closure_set(v___f_198_, 0, v_inst_197_);
return v___f_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_ofSubtype(lean_object* v_00_u03b1_199_, lean_object* v_p_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v___f_202_; 
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_ofSubtype___redArg___lam__0), 2, 1);
lean_closure_set(v___f_202_, 0, v_inst_201_);
return v___f_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___lam__0(lean_object* v_inst_203_, lean_object* v_f_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Equiv_Perm_ofSubtype___redArg___lam__0(v_inst_203_, v_f_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg(lean_object* v_inst_207_){
_start:
{
lean_object* v___f_208_; lean_object* v___f_209_; lean_object* v___x_210_; 
v___f_208_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___lam__0), 2, 1);
lean_closure_set(v___f_208_, 0, v_inst_207_);
v___f_209_ = ((lean_object*)(lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg___closed__0));
v___x_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_210_, 0, v___f_208_);
lean_ctor_set(v___x_210_, 1, v___f_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm(lean_object* v_00_u03b1_211_, lean_object* v_p_212_, lean_object* v_inst_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_Equiv_Perm_subtypeEquivSubtypePerm___redArg(v_inst_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instGroup___redArg___lam__0(lean_object* v_g_215_, lean_object* v_h_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lp_mathlib_Equiv_trans___redArg(v_h_216_, v_g_215_);
return v___x_217_;
}
}
static lean_object* _init_lp_mathlib_MulAut_instGroup___redArg___closed__1(void){
_start:
{
lean_object* v___x_219_; lean_object* v___f_220_; lean_object* v___x_221_; 
v___x_219_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___f_220_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_221_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_221_, 0, lean_box(0));
lean_closure_set(v___x_221_, 1, v___f_220_);
lean_closure_set(v___x_221_, 2, v___x_219_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_MulAut_instGroup___redArg___closed__2(void){
_start:
{
lean_object* v___x_222_; lean_object* v___f_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_222_ = lean_obj_once(&lp_mathlib_MulAut_instGroup___redArg___closed__1, &lp_mathlib_MulAut_instGroup___redArg___closed__1_once, _init_lp_mathlib_MulAut_instGroup___redArg___closed__1);
v___f_223_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_224_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_225_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v___f_223_);
lean_ctor_set(v___x_225_, 2, v___x_222_);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib_MulAut_instGroup___redArg___closed__3(void){
_start:
{
lean_object* v___f_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___f_226_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_227_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_228_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_228_, 0, lean_box(0));
lean_closure_set(v___x_228_, 1, v___x_227_);
lean_closure_set(v___x_228_, 2, v___f_226_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instGroup___redArg(lean_object* v_inst_229_){
_start:
{
lean_object* v___f_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___f_230_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_231_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_232_ = lean_obj_once(&lp_mathlib_MulAut_instGroup___redArg___closed__2, &lp_mathlib_MulAut_instGroup___redArg___closed__2_once, _init_lp_mathlib_MulAut_instGroup___redArg___closed__2);
lean_inc(v_inst_229_);
v___x_233_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_symm___boxed), 5, 4);
lean_closure_set(v___x_233_, 0, lean_box(0));
lean_closure_set(v___x_233_, 1, lean_box(0));
lean_closure_set(v___x_233_, 2, v_inst_229_);
lean_closure_set(v___x_233_, 3, v_inst_229_);
lean_inc_ref_n(v___x_233_, 2);
v___x_234_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_234_, 0, lean_box(0));
lean_closure_set(v___x_234_, 1, v___x_232_);
lean_closure_set(v___x_234_, 2, v___x_233_);
v___x_235_ = lean_obj_once(&lp_mathlib_MulAut_instGroup___redArg___closed__3, &lp_mathlib_MulAut_instGroup___redArg___closed__3_once, _init_lp_mathlib_MulAut_instGroup___redArg___closed__3);
v___x_236_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_236_, 0, lean_box(0));
lean_closure_set(v___x_236_, 1, v___x_231_);
lean_closure_set(v___x_236_, 2, v___f_230_);
lean_closure_set(v___x_236_, 3, v___x_233_);
lean_closure_set(v___x_236_, 4, v___x_235_);
v___x_237_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_237_, 0, v___x_232_);
lean_ctor_set(v___x_237_, 1, v___x_233_);
lean_ctor_set(v___x_237_, 2, v___x_234_);
lean_ctor_set(v___x_237_, 3, v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instGroup(lean_object* v_M_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lp_mathlib_MulAut_instGroup___redArg(v_inst_239_);
return v___x_240_;
}
}
static lean_object* _init_lp_mathlib_AddAut_instAddGroup___redArg___closed__0(void){
_start:
{
lean_object* v___x_241_; lean_object* v___f_242_; lean_object* v___x_243_; 
v___x_241_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___f_242_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_243_ = lean_alloc_closure((void*)(lp_mathlib_nsmulBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_243_, 0, lean_box(0));
lean_closure_set(v___x_243_, 1, v___f_242_);
lean_closure_set(v___x_243_, 2, v___x_241_);
return v___x_243_;
}
}
static lean_object* _init_lp_mathlib_AddAut_instAddGroup___redArg___closed__1(void){
_start:
{
lean_object* v___x_244_; lean_object* v___f_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_244_ = lean_obj_once(&lp_mathlib_AddAut_instAddGroup___redArg___closed__0, &lp_mathlib_AddAut_instAddGroup___redArg___closed__0_once, _init_lp_mathlib_AddAut_instAddGroup___redArg___closed__0);
v___f_245_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_246_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_247_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_247_, 0, v___x_246_);
lean_ctor_set(v___x_247_, 1, v___f_245_);
lean_ctor_set(v___x_247_, 2, v___x_244_);
return v___x_247_;
}
}
static lean_object* _init_lp_mathlib_AddAut_instAddGroup___redArg___closed__2(void){
_start:
{
lean_object* v___f_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v___f_248_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_249_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_250_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_250_, 0, lean_box(0));
lean_closure_set(v___x_250_, 1, v___x_249_);
lean_closure_set(v___x_250_, 2, v___f_248_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instAddGroup___redArg(lean_object* v_inst_251_){
_start:
{
lean_object* v___f_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; 
v___f_252_ = ((lean_object*)(lp_mathlib_MulAut_instGroup___redArg___closed__0));
v___x_253_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
v___x_254_ = lean_obj_once(&lp_mathlib_AddAut_instAddGroup___redArg___closed__1, &lp_mathlib_AddAut_instAddGroup___redArg___closed__1_once, _init_lp_mathlib_AddAut_instAddGroup___redArg___closed__1);
lean_inc(v_inst_251_);
v___x_255_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_symm___boxed), 5, 4);
lean_closure_set(v___x_255_, 0, lean_box(0));
lean_closure_set(v___x_255_, 1, lean_box(0));
lean_closure_set(v___x_255_, 2, v_inst_251_);
lean_closure_set(v___x_255_, 3, v_inst_251_);
lean_inc_ref_n(v___x_255_, 2);
v___x_256_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_256_, 0, lean_box(0));
lean_closure_set(v___x_256_, 1, v___x_254_);
lean_closure_set(v___x_256_, 2, v___x_255_);
v___x_257_ = lean_obj_once(&lp_mathlib_AddAut_instAddGroup___redArg___closed__2, &lp_mathlib_AddAut_instAddGroup___redArg___closed__2_once, _init_lp_mathlib_AddAut_instAddGroup___redArg___closed__2);
v___x_258_ = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(v___x_258_, 0, lean_box(0));
lean_closure_set(v___x_258_, 1, v___x_253_);
lean_closure_set(v___x_258_, 2, v___f_252_);
lean_closure_set(v___x_258_, 3, v___x_255_);
lean_closure_set(v___x_258_, 4, v___x_257_);
v___x_259_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_259_, 0, v___x_254_);
lean_ctor_set(v___x_259_, 1, v___x_255_);
lean_ctor_set(v___x_259_, 2, v___x_256_);
lean_ctor_set(v___x_259_, 3, v___x_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instAddGroup(lean_object* v_M_260_, lean_object* v_inst_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_mathlib_AddAut_instAddGroup___redArg(v_inst_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instInhabited(lean_object* v_M_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_instInhabited___boxed(lean_object* v_M_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_MulAut_instInhabited(v_M_266_, v_inst_267_);
lean_dec(v_inst_267_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instInhabited(lean_object* v_M_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lean_obj_once(&lp_mathlib_Equiv_Perm_instOne___closed__0, &lp_mathlib_Equiv_Perm_instOne___closed__0_once, _init_lp_mathlib_Equiv_Perm_instOne___closed__0);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_instInhabited___boxed(lean_object* v_M_272_, lean_object* v_inst_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_AddAut_instInhabited(v_M_272_, v_inst_273_);
lean_dec(v_inst_273_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm___lam__0(lean_object* v_self_275_){
_start:
{
lean_inc_ref(v_self_275_);
return v_self_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm___lam__0___boxed(lean_object* v_self_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_MulAut_toPerm___lam__0(v_self_276_);
lean_dec_ref(v_self_276_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm(lean_object* v_M_279_, lean_object* v_inst_280_){
_start:
{
lean_object* v___f_281_; 
v___f_281_ = ((lean_object*)(lp_mathlib_MulAut_toPerm___closed__0));
return v___f_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_toPerm___boxed(lean_object* v_M_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib_MulAut_toPerm(v_M_282_, v_inst_283_);
lean_dec(v_inst_283_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___lam__0(lean_object* v_toMul_285_, lean_object* v_g_286_, lean_object* v_toInv_287_, lean_object* v_h_288_){
_start:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
lean_inc(v_toMul_285_);
lean_inc(v_g_286_);
v___x_289_ = lean_apply_2(v_toMul_285_, v_g_286_, v_h_288_);
v___x_290_ = lean_apply_1(v_toInv_287_, v_g_286_);
v___x_291_ = lean_apply_2(v_toMul_285_, v___x_289_, v___x_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___lam__1(lean_object* v_toInv_292_, lean_object* v_g_293_, lean_object* v_toMul_294_, lean_object* v_h_295_){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
lean_inc(v_g_293_);
v___x_296_ = lean_apply_1(v_toInv_292_, v_g_293_);
lean_inc(v_toMul_294_);
v___x_297_ = lean_apply_2(v_toMul_294_, v___x_296_, v_h_295_);
v___x_298_ = lean_apply_2(v_toMul_294_, v___x_297_, v_g_293_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___lam__2(lean_object* v_toMul_299_, lean_object* v_toInv_300_, lean_object* v_g_301_){
_start:
{
lean_object* v___f_302_; lean_object* v___f_303_; lean_object* v___x_304_; 
lean_inc(v_toInv_300_);
lean_inc(v_g_301_);
lean_inc(v_toMul_299_);
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_conj___redArg___lam__0), 4, 3);
lean_closure_set(v___f_302_, 0, v_toMul_299_);
lean_closure_set(v___f_302_, 1, v_g_301_);
lean_closure_set(v___f_302_, 2, v_toInv_300_);
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_conj___redArg___lam__1), 4, 3);
lean_closure_set(v___f_303_, 0, v_toInv_300_);
lean_closure_set(v___f_303_, 1, v_g_301_);
lean_closure_set(v___f_303_, 2, v_toMul_299_);
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v___f_302_);
lean_ctor_set(v___x_304_, 1, v___f_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg(lean_object* v_inst_305_){
_start:
{
lean_object* v_toMonoid_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v_toMul_309_; lean_object* v___x_310_; lean_object* v_toInv_311_; lean_object* v___f_312_; 
v_toMonoid_306_ = lean_ctor_get(v_inst_305_, 0);
v___x_307_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_306_);
v___x_308_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_307_);
v_toMul_309_ = lean_ctor_get(v___x_308_, 1);
lean_inc(v_toMul_309_);
lean_dec_ref(v___x_308_);
v___x_310_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_305_);
v_toInv_311_ = lean_ctor_get(v___x_310_, 1);
lean_inc(v_toInv_311_);
lean_dec_ref(v___x_310_);
v___f_312_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_conj___redArg___lam__2), 3, 2);
lean_closure_set(v___f_312_, 0, v_toMul_309_);
lean_closure_set(v___f_312_, 1, v_toInv_311_);
return v___f_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___redArg___boxed(lean_object* v_inst_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_MulAut_conj___redArg(v_inst_313_);
lean_dec_ref(v_inst_313_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj(lean_object* v_G_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_MulAut_conj___redArg(v_inst_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conj___boxed(lean_object* v_G_318_, lean_object* v_inst_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_MulAut_conj(v_G_318_, v_inst_319_);
lean_dec_ref(v_inst_319_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___lam__0(lean_object* v_toAdd_321_, lean_object* v_g_322_, lean_object* v_toNeg_323_, lean_object* v_h_324_){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
lean_inc(v_toAdd_321_);
lean_inc(v_g_322_);
v___x_325_ = lean_apply_2(v_toAdd_321_, v_g_322_, v_h_324_);
v___x_326_ = lean_apply_1(v_toNeg_323_, v_g_322_);
v___x_327_ = lean_apply_2(v_toAdd_321_, v___x_325_, v___x_326_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___lam__1(lean_object* v_toNeg_328_, lean_object* v_g_329_, lean_object* v_toAdd_330_, lean_object* v_h_331_){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; 
lean_inc(v_g_329_);
v___x_332_ = lean_apply_1(v_toNeg_328_, v_g_329_);
lean_inc(v_toAdd_330_);
v___x_333_ = lean_apply_2(v_toAdd_330_, v___x_332_, v_h_331_);
v___x_334_ = lean_apply_2(v_toAdd_330_, v___x_333_, v_g_329_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___lam__2(lean_object* v_toAdd_335_, lean_object* v_toNeg_336_, lean_object* v_g_337_){
_start:
{
lean_object* v___f_338_; lean_object* v___f_339_; lean_object* v___x_340_; 
lean_inc(v_toNeg_336_);
lean_inc(v_g_337_);
lean_inc(v_toAdd_335_);
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_AddAut_addConj___redArg___lam__0), 4, 3);
lean_closure_set(v___f_338_, 0, v_toAdd_335_);
lean_closure_set(v___f_338_, 1, v_g_337_);
lean_closure_set(v___f_338_, 2, v_toNeg_336_);
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_AddAut_addConj___redArg___lam__1), 4, 3);
lean_closure_set(v___f_339_, 0, v_toNeg_336_);
lean_closure_set(v___f_339_, 1, v_g_337_);
lean_closure_set(v___f_339_, 2, v_toAdd_335_);
v___x_340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_340_, 0, v___f_338_);
lean_ctor_set(v___x_340_, 1, v___f_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg(lean_object* v_inst_341_){
_start:
{
lean_object* v_toAddMonoid_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v_toAdd_345_; lean_object* v___x_346_; lean_object* v_toNeg_347_; lean_object* v___f_348_; 
v_toAddMonoid_342_ = lean_ctor_get(v_inst_341_, 0);
v___x_343_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_342_);
v___x_344_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_343_);
v_toAdd_345_ = lean_ctor_get(v___x_344_, 1);
lean_inc(v_toAdd_345_);
lean_dec_ref(v___x_344_);
v___x_346_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_341_);
v_toNeg_347_ = lean_ctor_get(v___x_346_, 1);
lean_inc(v_toNeg_347_);
lean_dec_ref(v___x_346_);
v___f_348_ = lean_alloc_closure((void*)(lp_mathlib_AddAut_addConj___redArg___lam__2), 3, 2);
lean_closure_set(v___f_348_, 0, v_toAdd_345_);
lean_closure_set(v___f_348_, 1, v_toNeg_347_);
return v___f_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___redArg___boxed(lean_object* v_inst_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_AddAut_addConj___redArg(v_inst_349_);
lean_dec_ref(v_inst_349_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj(lean_object* v_G_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_mathlib_AddAut_addConj___redArg(v_inst_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_addConj___boxed(lean_object* v_G_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_AddAut_addConj(v_G_354_, v_inst_355_);
lean_dec_ref(v_inst_355_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___redArg___lam__0(lean_object* v_00_u03d5_357_, lean_object* v_f_358_){
_start:
{
lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; 
lean_inc_ref(v_00_u03d5_357_);
v___x_359_ = lp_mathlib_Equiv_symm___redArg(v_00_u03d5_357_);
v___x_360_ = lp_mathlib_Equiv_trans___redArg(v_f_358_, v___x_359_);
v___x_361_ = lp_mathlib_Equiv_trans___redArg(v_00_u03d5_357_, v___x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___redArg___lam__1(lean_object* v_00_u03d5_362_, lean_object* v_f_363_){
_start:
{
lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; 
lean_inc_ref(v_00_u03d5_362_);
v___x_364_ = lp_mathlib_Equiv_symm___redArg(v_00_u03d5_362_);
v___x_365_ = lp_mathlib_Equiv_trans___redArg(v_f_363_, v_00_u03d5_362_);
v___x_366_ = lp_mathlib_Equiv_trans___redArg(v___x_364_, v___x_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___redArg(lean_object* v_00_u03d5_367_){
_start:
{
lean_object* v___f_368_; lean_object* v___f_369_; lean_object* v___x_370_; 
lean_inc_ref(v_00_u03d5_367_);
v___f_368_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_congr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_368_, 0, v_00_u03d5_367_);
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_congr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_369_, 0, v_00_u03d5_367_);
v___x_370_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_370_, 0, v___f_369_);
lean_ctor_set(v___x_370_, 1, v___f_368_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr(lean_object* v_G_371_, lean_object* v_inst_372_, lean_object* v_H_373_, lean_object* v_inst_374_, lean_object* v_00_u03d5_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_mathlib_MulAut_congr___redArg(v_00_u03d5_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_congr___boxed(lean_object* v_G_377_, lean_object* v_inst_378_, lean_object* v_H_379_, lean_object* v_inst_380_, lean_object* v_00_u03d5_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_MulAut_congr(v_G_377_, v_inst_378_, v_H_379_, v_inst_380_, v_00_u03d5_381_);
lean_dec_ref(v_inst_380_);
lean_dec_ref(v_inst_378_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_congr___redArg(lean_object* v_00_u03d5_383_){
_start:
{
lean_object* v___f_384_; lean_object* v___f_385_; lean_object* v___x_386_; 
lean_inc_ref(v_00_u03d5_383_);
v___f_384_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_congr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_384_, 0, v_00_u03d5_383_);
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_congr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_385_, 0, v_00_u03d5_383_);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v___f_385_);
lean_ctor_set(v___x_386_, 1, v___f_384_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_congr(lean_object* v_G_387_, lean_object* v_inst_388_, lean_object* v_H_389_, lean_object* v_inst_390_, lean_object* v_00_u03d5_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lp_mathlib_AddAut_congr___redArg(v_00_u03d5_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_congr___boxed(lean_object* v_G_393_, lean_object* v_inst_394_, lean_object* v_H_395_, lean_object* v_inst_396_, lean_object* v_00_u03d5_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_AddAut_congr(v_G_393_, v_inst_394_, v_H_395_, v_inst_396_, v_00_u03d5_397_);
lean_dec_ref(v_inst_396_);
lean_dec_ref(v_inst_394_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_toPerm(lean_object* v_A_399_, lean_object* v_inst_400_){
_start:
{
lean_object* v___f_401_; 
v___f_401_ = ((lean_object*)(lp_mathlib_MulAut_toPerm___closed__0));
return v___f_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_toPerm___boxed(lean_object* v_A_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_AddAut_toPerm(v_A_402_, v_inst_403_);
lean_dec(v_inst_403_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj___redArg(lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_AddAut_addConj___redArg(v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj___redArg___boxed(lean_object* v_inst_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_AddAut_conj___redArg(v_inst_407_);
lean_dec_ref(v_inst_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj(lean_object* v_G_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lp_mathlib_AddAut_addConj___redArg(v_inst_410_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_conj___boxed(lean_object* v_G_412_, lean_object* v_inst_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_AddAut_conj(v_G_412_, v_inst_413_);
lean_dec_ref(v_inst_413_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative___redArg(lean_object* v_inst_415_){
_start:
{
lean_object* v_toAddMonoid_416_; lean_object* v_toAdd_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v_toAddMonoid_416_ = lean_ctor_get(v_inst_415_, 0);
v_toAdd_417_ = lean_ctor_get(v_toAddMonoid_416_, 1);
v___x_418_ = lp_mathlib_AddEquiv_toMultiplicative(lean_box(0), lean_box(0), v_toAdd_417_, v_toAdd_417_);
v___x_419_ = lp_mathlib_Equiv_symm___redArg(v___x_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative___redArg___boxed(lean_object* v_inst_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib_MulAutMultiplicative___redArg(v_inst_420_);
lean_dec_ref(v_inst_420_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative(lean_object* v_G_422_, lean_object* v_inst_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_mathlib_MulAutMultiplicative___redArg(v_inst_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAutMultiplicative___boxed(lean_object* v_G_425_, lean_object* v_inst_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_MulAutMultiplicative(v_G_425_, v_inst_426_);
lean_dec_ref(v_inst_426_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive___redArg(lean_object* v_inst_428_){
_start:
{
lean_object* v_toMonoid_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v_toMul_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v_toMonoid_429_ = lean_ctor_get(v_inst_428_, 0);
v___x_430_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_429_);
v___x_431_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_430_);
v_toMul_432_ = lean_ctor_get(v___x_431_, 1);
lean_inc(v_toMul_432_);
lean_dec_ref(v___x_431_);
v___x_433_ = lp_mathlib_MulEquiv_toAdditive(lean_box(0), lean_box(0), v_toMul_432_, v_toMul_432_);
lean_dec(v_toMul_432_);
v___x_434_ = lp_mathlib_Equiv_symm___redArg(v___x_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive___redArg___boxed(lean_object* v_inst_435_){
_start:
{
lean_object* v_res_436_; 
v_res_436_ = lp_mathlib_AddAutAdditive___redArg(v_inst_435_);
lean_dec_ref(v_inst_435_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive(lean_object* v_G_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_AddAutAdditive___redArg(v_inst_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAutAdditive___boxed(lean_object* v_G_440_, lean_object* v_inst_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_AddAutAdditive(v_G_440_, v_inst_441_);
lean_dec_ref(v_inst_441_);
return v_res_442_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
}
#ifdef __cplusplus
}
#endif
