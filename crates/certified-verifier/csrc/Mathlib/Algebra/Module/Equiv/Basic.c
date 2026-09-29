// Lean compiler output
// Module: Mathlib.Algebra.Module.Equiv.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.GroupWithZero.Action.Basic public import Mathlib.Algebra.GroupWithZero.Action.Units public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.Hom public import Mathlib.Algebra.Module.LinearMap.Basic public import Mathlib.Algebra.Module.LinearMap.End public import Mathlib.Algebra.Module.Pi public import Mathlib.Algebra.Module.Prod
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
lean_object* lp_mathlib_Units_mulLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_refl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instSemiring;
lean_object* lp_mathlib_instMulActionNatOfAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_smulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_mulRight___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemiring;
lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Units_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_curry(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumPiEquivProdPi(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Units_mk0___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_restrictScalars___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_symm___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_automorphismGroup___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv_applyDistribMulAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_applyDistribMulAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_applyDistribMulAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleAut___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleAut(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toNatLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toNatLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toNatLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toIntLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toIntLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toIntLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piApply___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_piApply___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_piApply___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_piApply___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_piApply___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piApply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piApply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_addMonoidHomLequivNat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_addMonoidHomLequivNat___redArg___closed__0 = (const lean_object*)&lp_mathlib_addMonoidHomLequivNat___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivInt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndRingEquivInt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndRingEquivInt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_curry___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_curry___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_curry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_curry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinear___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinear___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjRingEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfUnit___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfUnit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfUnit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongr___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_sumPiEquivProdPi___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_sumPiEquivProdPi___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumPiEquivProdPi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumPiEquivProdPi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars___redArg___lam__0(lean_object* v___x_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_toLinearMap_3_; lean_object* v___x_4_; 
v_toLinearMap_3_ = lean_ctor_get(v___x_1_, 0);
lean_inc(v_toLinearMap_3_);
lean_dec_ref(v___x_1_);
v___x_4_ = lean_apply_1(v_toLinearMap_3_, v___y_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars___redArg(lean_object* v_f_5_){
_start:
{
lean_object* v_toLinearMap_6_; lean_object* v___f_7_; lean_object* v___x_8_; lean_object* v___f_9_; lean_object* v___x_10_; 
v_toLinearMap_6_ = lean_ctor_get(v_f_5_, 0);
lean_inc(v_toLinearMap_6_);
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_7_, 0, v_toLinearMap_6_);
v___x_8_ = lp_mathlib_LinearEquiv_symm___redArg(v_f_5_);
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_9_, 0, v___x_8_);
v___x_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_10_, 0, v___f_7_);
lean_ctor_set(v___x_10_, 1, v___f_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars(lean_object* v_R_11_, lean_object* v_S_12_, lean_object* v_M_13_, lean_object* v_M_u2082_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_f_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_LinearEquiv_restrictScalars___redArg(v_f_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_restrictScalars___boxed(lean_object* v_R_26_, lean_object* v_S_27_, lean_object* v_M_28_, lean_object* v_M_u2082_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_f_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_LinearEquiv_restrictScalars(v_R_26_, v_S_27_, v_M_28_, v_M_u2082_29_, v_inst_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_f_39_);
lean_dec(v_inst_37_);
lean_dec(v_inst_36_);
lean_dec(v_inst_35_);
lean_dec(v_inst_34_);
lean_dec_ref(v_inst_33_);
lean_dec_ref(v_inst_32_);
lean_dec_ref(v_inst_31_);
lean_dec_ref(v_inst_30_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg___lam__0(lean_object* v_f_41_, lean_object* v_g_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_LinearEquiv_trans___redArg(v_g_42_, v_f_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg(lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___f_49_; lean_object* v___f_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___f_49_ = ((lean_object*)(lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__0));
v___f_50_ = ((lean_object*)(lp_mathlib_LinearEquiv_automorphismGroup___redArg___closed__1));
v___x_51_ = lp_mathlib_LinearEquiv_refl(lean_box(0), lean_box(0), v_inst_46_, v_inst_47_, v_inst_48_);
lean_inc_ref_n(v___x_51_, 3);
v___x_52_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_52_, 0, lean_box(0));
lean_closure_set(v___x_52_, 1, v___f_50_);
lean_closure_set(v___x_52_, 2, v___x_51_);
v___x_53_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_53_, 0, v___x_51_);
lean_ctor_set(v___x_53_, 1, v___f_50_);
lean_ctor_set(v___x_53_, 2, v___x_52_);
lean_inc_ref(v___x_53_);
v___x_54_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_54_, 0, lean_box(0));
lean_closure_set(v___x_54_, 1, v___x_53_);
lean_closure_set(v___x_54_, 2, v___f_49_);
v___x_55_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, v___x_51_);
lean_closure_set(v___x_55_, 2, v___f_50_);
v___x_56_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_56_, 0, lean_box(0));
lean_closure_set(v___x_56_, 1, v___x_51_);
lean_closure_set(v___x_56_, 2, v___f_50_);
lean_closure_set(v___x_56_, 3, v___f_49_);
lean_closure_set(v___x_56_, 4, v___x_55_);
v___x_57_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_57_, 0, v___x_53_);
lean_ctor_set(v___x_57_, 1, v___f_49_);
lean_ctor_set(v___x_57_, 2, v___x_54_);
lean_ctor_set(v___x_57_, 3, v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___redArg___boxed(lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_LinearEquiv_automorphismGroup___redArg(v_inst_58_, v_inst_59_, v_inst_60_);
lean_dec(v_inst_60_);
lean_dec_ref(v_inst_59_);
lean_dec_ref(v_inst_58_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup(lean_object* v_R_62_, lean_object* v_M_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_LinearEquiv_automorphismGroup___redArg(v_inst_64_, v_inst_65_, v_inst_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup___boxed(lean_object* v_R_68_, lean_object* v_M_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_LinearEquiv_automorphismGroup(v_R_68_, v_M_69_, v_inst_70_, v_inst_71_, v_inst_72_);
lean_dec(v_inst_72_);
lean_dec_ref(v_inst_71_);
lean_dec_ref(v_inst_70_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___lam__0(lean_object* v_e_74_, lean_object* v___y_75_){
_start:
{
lean_object* v_toLinearMap_76_; lean_object* v___x_77_; 
v_toLinearMap_76_ = lean_ctor_get(v_e_74_, 0);
lean_inc(v_toLinearMap_76_);
lean_dec_ref(v_e_74_);
v___x_77_ = lean_apply_1(v_toLinearMap_76_, v___y_75_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom(lean_object* v_R_79_, lean_object* v_M_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v___f_84_; 
v___f_84_ = ((lean_object*)(lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___closed__0));
return v___f_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom___boxed(lean_object* v_R_85_, lean_object* v_M_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_LinearEquiv_automorphismGroup_toLinearMapMonoidHom(v_R_85_, v_M_86_, v_inst_87_, v_inst_88_, v_inst_89_);
lean_dec(v_inst_89_);
lean_dec_ref(v_inst_88_);
lean_dec_ref(v_inst_87_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction___lam__0(lean_object* v_x1_91_, lean_object* v_x2_92_){
_start:
{
lean_object* v_toLinearMap_93_; lean_object* v___x_94_; 
v_toLinearMap_93_ = lean_ctor_get(v_x1_91_, 0);
lean_inc(v_toLinearMap_93_);
lean_dec_ref(v_x1_91_);
v___x_94_ = lean_apply_1(v_toLinearMap_93_, v_x2_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction(lean_object* v_R_96_, lean_object* v_M_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___f_101_; 
v___f_101_ = ((lean_object*)(lp_mathlib_LinearEquiv_applyDistribMulAction___closed__0));
return v___f_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_applyDistribMulAction___boxed(lean_object* v_R_102_, lean_object* v_M_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_LinearEquiv_applyDistribMulAction(v_R_102_, v_M_103_, v_inst_104_, v_inst_105_, v_inst_106_);
lean_dec(v_inst_106_);
lean_dec_ref(v_inst_105_);
lean_dec_ref(v_inst_104_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0(lean_object* v_toZero_108_, lean_object* v_x_109_){
_start:
{
lean_inc(v_toZero_108_);
return v_toZero_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed(lean_object* v_toZero_110_, lean_object* v_x_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0(v_toZero_110_, v_x_111_);
lean_dec(v_x_111_);
lean_dec(v_toZero_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg(lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v_toZero_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v_toZero_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_129_; 
v___x_115_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_114_);
v___x_116_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_115_);
v_toZero_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_toZero_117_);
lean_dec_ref(v___x_116_);
v___x_118_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_113_);
v___x_119_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_118_);
v_toZero_120_ = lean_ctor_get(v___x_119_, 0);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_119_);
if (v_isSharedCheck_129_ == 0)
{
lean_object* v_unused_130_; 
v_unused_130_ = lean_ctor_get(v___x_119_, 1);
lean_dec(v_unused_130_);
v___x_122_ = v___x_119_;
v_isShared_123_ = v_isSharedCheck_129_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_toZero_120_);
lean_dec(v___x_119_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_129_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___f_124_; lean_object* v___f_125_; lean_object* v___x_127_; 
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_124_, 0, v_toZero_117_);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_125_, 0, v_toZero_120_);
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 1, v___f_125_);
lean_ctor_set(v___x_122_, 0, v___f_124_);
v___x_127_ = v___x_122_;
goto v_reusejp_126_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v___f_124_);
lean_ctor_set(v_reuseFailAlloc_128_, 1, v___f_125_);
v___x_127_ = v_reuseFailAlloc_128_;
goto v_reusejp_126_;
}
v_reusejp_126_:
{
return v___x_127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___redArg___boxed(lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_LinearEquiv_ofSubsingleton___redArg(v_inst_131_, v_inst_132_);
lean_dec_ref(v_inst_132_);
lean_dec_ref(v_inst_131_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton(lean_object* v_R_134_, lean_object* v_M_135_, lean_object* v_M_u2082_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_LinearEquiv_ofSubsingleton___redArg(v_inst_138_, v_inst_139_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofSubsingleton___boxed(lean_object* v_R_145_, lean_object* v_M_146_, lean_object* v_M_u2082_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_LinearEquiv_ofSubsingleton(v_R_145_, v_M_146_, v_M_u2082_147_, v_inst_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_inst_152_, v_inst_153_, v_inst_154_);
lean_dec(v_inst_152_);
lean_dec(v_inst_151_);
lean_dec_ref(v_inst_150_);
lean_dec_ref(v_inst_149_);
lean_dec_ref(v_inst_148_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__0(lean_object* v_g_156_, lean_object* v___y_157_){
_start:
{
lean_object* v_toFun_158_; lean_object* v___x_159_; 
v_toFun_158_ = lean_ctor_get(v_g_156_, 0);
lean_inc(v_toFun_158_);
lean_dec_ref(v_g_156_);
v___x_159_ = lean_apply_1(v_toFun_158_, v___y_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__1(lean_object* v___x_160_, lean_object* v___y_161_){
_start:
{
lean_object* v_toFun_162_; lean_object* v___x_163_; 
v_toFun_162_ = lean_ctor_get(v___x_160_, 0);
lean_inc(v_toFun_162_);
lean_dec_ref(v___x_160_);
v___x_163_ = lean_apply_1(v_toFun_162_, v___y_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___redArg(lean_object* v_g_164_){
_start:
{
lean_object* v___f_165_; lean_object* v___x_166_; lean_object* v___f_167_; lean_object* v___x_168_; 
lean_inc_ref(v_g_164_);
v___f_165_ = lean_alloc_closure((void*)(lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_165_, 0, v_g_164_);
v___x_166_ = lp_mathlib_Equiv_symm___redArg(v_g_164_);
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_167_, 0, v___x_166_);
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v___f_165_);
lean_ctor_set(v___x_168_, 1, v___f_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv(lean_object* v_R_169_, lean_object* v_S_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_g_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_Module_compHom_toLinearEquiv___redArg(v_g_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearEquiv___boxed(lean_object* v_R_175_, lean_object* v_S_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_g_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Module_compHom_toLinearEquiv(v_R_175_, v_S_176_, v_inst_177_, v_inst_178_, v_g_179_);
lean_dec_ref(v_inst_178_);
lean_dec_ref(v_inst_177_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv___redArg(lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_s_183_){
_start:
{
lean_object* v___x_184_; lean_object* v_toFun_185_; lean_object* v_invFun_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_193_; 
v___x_184_ = lp_mathlib_DistribMulAction_toAddEquiv___redArg(v_inst_181_, v_inst_182_, v_s_183_);
v_toFun_185_ = lean_ctor_get(v___x_184_, 0);
v_invFun_186_ = lean_ctor_get(v___x_184_, 1);
v_isSharedCheck_193_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_193_ == 0)
{
v___x_188_ = v___x_184_;
v_isShared_189_ = v_isSharedCheck_193_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_invFun_186_);
lean_inc(v_toFun_185_);
lean_dec(v___x_184_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_193_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_191_; 
if (v_isShared_189_ == 0)
{
v___x_191_ = v___x_188_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v_toFun_185_);
lean_ctor_set(v_reuseFailAlloc_192_, 1, v_invFun_186_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
return v___x_191_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv___redArg___boxed(lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_s_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_DistribMulAction_toLinearEquiv___redArg(v_inst_194_, v_inst_195_, v_s_196_);
lean_dec_ref(v_inst_194_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv(lean_object* v_R_198_, lean_object* v_S_199_, lean_object* v_M_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_s_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_DistribMulAction_toLinearEquiv___redArg(v_inst_204_, v_inst_205_, v_s_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toLinearEquiv___boxed(lean_object* v_R_209_, lean_object* v_S_210_, lean_object* v_M_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_s_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_DistribMulAction_toLinearEquiv(v_R_209_, v_S_210_, v_M_211_, v_inst_212_, v_inst_213_, v_inst_214_, v_inst_215_, v_inst_216_, v_inst_217_, v_s_218_);
lean_dec_ref(v_inst_215_);
lean_dec(v_inst_214_);
lean_dec_ref(v_inst_213_);
lean_dec_ref(v_inst_212_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleAut___redArg(lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_toLinearEquiv___boxed), 10, 9);
lean_closure_set(v___x_225_, 0, lean_box(0));
lean_closure_set(v___x_225_, 1, lean_box(0));
lean_closure_set(v___x_225_, 2, lean_box(0));
lean_closure_set(v___x_225_, 3, v_inst_220_);
lean_closure_set(v___x_225_, 4, v_inst_221_);
lean_closure_set(v___x_225_, 5, v_inst_222_);
lean_closure_set(v___x_225_, 6, v_inst_223_);
lean_closure_set(v___x_225_, 7, v_inst_224_);
lean_closure_set(v___x_225_, 8, lean_box(0));
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toModuleAut(lean_object* v_R_226_, lean_object* v_S_227_, lean_object* v_M_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_toLinearEquiv___boxed), 10, 9);
lean_closure_set(v___x_235_, 0, lean_box(0));
lean_closure_set(v___x_235_, 1, lean_box(0));
lean_closure_set(v___x_235_, 2, lean_box(0));
lean_closure_set(v___x_235_, 3, v_inst_229_);
lean_closure_set(v___x_235_, 4, v_inst_230_);
lean_closure_set(v___x_235_, 5, v_inst_231_);
lean_closure_set(v___x_235_, 6, v_inst_232_);
lean_closure_set(v___x_235_, 7, v_inst_233_);
lean_closure_set(v___x_235_, 8, lean_box(0));
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toLinearEquiv___redArg(lean_object* v_e_236_){
_start:
{
lean_object* v_toFun_237_; lean_object* v_invFun_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_245_; 
v_toFun_237_ = lean_ctor_get(v_e_236_, 0);
v_invFun_238_ = lean_ctor_get(v_e_236_, 1);
v_isSharedCheck_245_ = !lean_is_exclusive(v_e_236_);
if (v_isSharedCheck_245_ == 0)
{
v___x_240_ = v_e_236_;
v_isShared_241_ = v_isSharedCheck_245_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_invFun_238_);
lean_inc(v_toFun_237_);
lean_dec(v_e_236_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_245_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v___x_243_; 
if (v_isShared_241_ == 0)
{
v___x_243_ = v___x_240_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v_toFun_237_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v_invFun_238_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toLinearEquiv(lean_object* v_R_246_, lean_object* v_M_247_, lean_object* v_M_u2082_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_e_254_, lean_object* v_h_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v_e_254_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toLinearEquiv___boxed(lean_object* v_R_257_, lean_object* v_M_258_, lean_object* v_M_u2082_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_e_265_, lean_object* v_h_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_AddEquiv_toLinearEquiv(v_R_257_, v_M_258_, v_M_u2082_259_, v_inst_260_, v_inst_261_, v_inst_262_, v_inst_263_, v_inst_264_, v_e_265_, v_h_266_);
lean_dec(v_inst_264_);
lean_dec(v_inst_263_);
lean_dec_ref(v_inst_262_);
lean_dec_ref(v_inst_261_);
lean_dec_ref(v_inst_260_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toNatLinearEquiv___redArg(lean_object* v_e_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v_e_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toNatLinearEquiv(lean_object* v_M_270_, lean_object* v_M_u2082_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_e_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v_e_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toNatLinearEquiv___boxed(lean_object* v_M_276_, lean_object* v_M_u2082_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_e_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_AddEquiv_toNatLinearEquiv(v_M_276_, v_M_u2082_277_, v_inst_278_, v_inst_279_, v_e_280_);
lean_dec_ref(v_inst_279_);
lean_dec_ref(v_inst_278_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toIntLinearEquiv___redArg(lean_object* v_e_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v_e_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toIntLinearEquiv(lean_object* v_M_284_, lean_object* v_M_u2082_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_modM_288_, lean_object* v_modM_u2082_289_, lean_object* v_e_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v_e_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toIntLinearEquiv___boxed(lean_object* v_M_292_, lean_object* v_M_u2082_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_modM_296_, lean_object* v_modM_u2082_297_, lean_object* v_e_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_AddEquiv_toIntLinearEquiv(v_M_292_, v_M_u2082_293_, v_inst_294_, v_inst_295_, v_modM_296_, v_modM_u2082_297_, v_e_298_);
lean_dec(v_modM_u2082_297_);
lean_dec(v_modM_296_);
lean_dec_ref(v_inst_295_);
lean_dec_ref(v_inst_294_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piApply___lam__0(lean_object* v_e_300_, lean_object* v___y_301_, lean_object* v___y_302_){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; 
lean_inc(v___y_302_);
v___x_303_ = lean_apply_1(v___y_301_, v___y_302_);
v___x_304_ = lean_apply_2(v_e_300_, v___y_302_, v___x_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piApply(lean_object* v_R_306_, lean_object* v_M_307_, lean_object* v_V_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_){
_start:
{
lean_object* v___f_312_; 
v___f_312_ = ((lean_object*)(lp_mathlib_LinearMap_piApply___closed__0));
return v___f_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piApply___boxed(lean_object* v_R_313_, lean_object* v_M_314_, lean_object* v_V_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_LinearMap_piApply(v_R_313_, v_M_314_, v_V_315_, v_inst_316_, v_inst_317_, v_inst_318_);
lean_dec(v_inst_318_);
lean_dec_ref(v_inst_317_);
lean_dec_ref(v_inst_316_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___lam__0(lean_object* v_toOne_320_, lean_object* v_f_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lean_apply_1(v_f_321_, v_toOne_320_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___redArg(lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v_toAddCommMonoid_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v_toOne_330_; lean_object* v___x_331_; lean_object* v___f_332_; lean_object* v___f_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v_toAddCommMonoid_327_ = lean_ctor_get(v_inst_324_, 0);
lean_inc_ref(v_toAddCommMonoid_327_);
lean_inc_ref_n(v_inst_324_, 2);
v___x_328_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_324_);
v___x_329_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_328_);
v_toOne_330_ = lean_ctor_get(v___x_329_, 2);
lean_inc(v_toOne_330_);
lean_dec_ref(v___x_329_);
v___x_331_ = lp_mathlib_Semiring_toModule___redArg(v_inst_324_);
v___f_332_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_332_, 0, v_toOne_330_);
v___f_333_ = ((lean_object*)(lp_mathlib_LinearMap_ringLmapEquivSelf___redArg___closed__0));
lean_inc(v___x_331_);
lean_inc(v_inst_326_);
v___x_334_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_smulRight___boxed), 15, 14);
lean_closure_set(v___x_334_, 0, lean_box(0));
lean_closure_set(v___x_334_, 1, lean_box(0));
lean_closure_set(v___x_334_, 2, lean_box(0));
lean_closure_set(v___x_334_, 3, lean_box(0));
lean_closure_set(v___x_334_, 4, v_inst_324_);
lean_closure_set(v___x_334_, 5, v_inst_325_);
lean_closure_set(v___x_334_, 6, v_toAddCommMonoid_327_);
lean_closure_set(v___x_334_, 7, v_inst_326_);
lean_closure_set(v___x_334_, 8, v___x_331_);
lean_closure_set(v___x_334_, 9, v_inst_324_);
lean_closure_set(v___x_334_, 10, v___x_331_);
lean_closure_set(v___x_334_, 11, v_inst_326_);
lean_closure_set(v___x_334_, 12, lean_box(0));
lean_closure_set(v___x_334_, 13, v___f_333_);
v___x_335_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_335_, 0, v___f_332_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf(lean_object* v_R_336_, lean_object* v_S_337_, lean_object* v_M_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_mathlib_LinearMap_ringLmapEquivSelf___redArg(v_inst_339_, v_inst_341_, v_inst_342_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___boxed(lean_object* v_R_346_, lean_object* v_S_347_, lean_object* v_M_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_LinearMap_ringLmapEquivSelf(v_R_346_, v_S_347_, v_M_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_inst_352_, v_inst_353_, v_inst_354_);
lean_dec(v_inst_353_);
lean_dec_ref(v_inst_350_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivNat___redArg(lean_object* v_inst_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___f_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_359_ = lp_mathlib_Nat_instSemiring;
lean_inc_ref_n(v_inst_357_, 2);
v___x_360_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_357_);
lean_inc_ref_n(v_inst_358_, 2);
v___x_361_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_358_);
v___f_362_ = ((lean_object*)(lp_mathlib_addMonoidHomLequivNat___redArg___closed__0));
v___x_363_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_toNatLinearMap___boxed), 5, 4);
lean_closure_set(v___x_363_, 0, lean_box(0));
lean_closure_set(v___x_363_, 1, lean_box(0));
lean_closure_set(v___x_363_, 2, v_inst_357_);
lean_closure_set(v___x_363_, 3, v_inst_358_);
v___x_364_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___boxed), 12, 11);
lean_closure_set(v___x_364_, 0, lean_box(0));
lean_closure_set(v___x_364_, 1, lean_box(0));
lean_closure_set(v___x_364_, 2, lean_box(0));
lean_closure_set(v___x_364_, 3, lean_box(0));
lean_closure_set(v___x_364_, 4, v___x_359_);
lean_closure_set(v___x_364_, 5, v___x_359_);
lean_closure_set(v___x_364_, 6, v_inst_357_);
lean_closure_set(v___x_364_, 7, v_inst_358_);
lean_closure_set(v___x_364_, 8, v___x_360_);
lean_closure_set(v___x_364_, 9, v___x_361_);
lean_closure_set(v___x_364_, 10, v___f_362_);
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_363_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivNat(lean_object* v_A_366_, lean_object* v_B_367_, lean_object* v_R_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lp_mathlib_addMonoidHomLequivNat___redArg(v_inst_370_, v_inst_371_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivNat___boxed(lean_object* v_A_374_, lean_object* v_B_375_, lean_object* v_R_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_addMonoidHomLequivNat(v_A_374_, v_B_375_, v_R_376_, v_inst_377_, v_inst_378_, v_inst_379_, v_inst_380_);
lean_dec(v_inst_380_);
lean_dec_ref(v_inst_377_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivInt___redArg(lean_object* v_inst_382_, lean_object* v_inst_383_){
_start:
{
lean_object* v___x_384_; lean_object* v_toAddMonoid_385_; lean_object* v_toAddMonoid_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___f_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v___x_384_ = lp_mathlib_Int_instCommSemiring;
v_toAddMonoid_385_ = lean_ctor_get(v_inst_383_, 0);
lean_inc_ref(v_toAddMonoid_385_);
v_toAddMonoid_386_ = lean_ctor_get(v_inst_382_, 0);
lean_inc_ref(v_toAddMonoid_386_);
lean_inc_ref(v_inst_382_);
v___x_387_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_382_);
lean_inc_ref(v_inst_383_);
v___x_388_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_383_);
v___f_389_ = ((lean_object*)(lp_mathlib_addMonoidHomLequivNat___redArg___closed__0));
v___x_390_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_toIntLinearMap___boxed), 5, 4);
lean_closure_set(v___x_390_, 0, lean_box(0));
lean_closure_set(v___x_390_, 1, lean_box(0));
lean_closure_set(v___x_390_, 2, v_inst_382_);
lean_closure_set(v___x_390_, 3, v_inst_383_);
v___x_391_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___boxed), 12, 11);
lean_closure_set(v___x_391_, 0, lean_box(0));
lean_closure_set(v___x_391_, 1, lean_box(0));
lean_closure_set(v___x_391_, 2, lean_box(0));
lean_closure_set(v___x_391_, 3, lean_box(0));
lean_closure_set(v___x_391_, 4, v___x_384_);
lean_closure_set(v___x_391_, 5, v___x_384_);
lean_closure_set(v___x_391_, 6, v_toAddMonoid_386_);
lean_closure_set(v___x_391_, 7, v_toAddMonoid_385_);
lean_closure_set(v___x_391_, 8, v___x_387_);
lean_closure_set(v___x_391_, 9, v___x_388_);
lean_closure_set(v___x_391_, 10, v___f_389_);
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_390_);
lean_ctor_set(v___x_392_, 1, v___x_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivInt(lean_object* v_A_393_, lean_object* v_B_394_, lean_object* v_R_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_inst_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lp_mathlib_addMonoidHomLequivInt___redArg(v_inst_397_, v_inst_398_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomLequivInt___boxed(lean_object* v_A_401_, lean_object* v_B_402_, lean_object* v_R_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_addMonoidHomLequivInt(v_A_401_, v_B_402_, v_R_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_inst_407_);
lean_dec(v_inst_407_);
lean_dec_ref(v_inst_404_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndRingEquivInt___redArg(lean_object* v_inst_409_){
_start:
{
lean_object* v___x_410_; lean_object* v_toLinearMap_411_; lean_object* v_invFun_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_419_; 
lean_inc_ref(v_inst_409_);
v___x_410_ = lp_mathlib_addMonoidHomLequivInt___redArg(v_inst_409_, v_inst_409_);
v_toLinearMap_411_ = lean_ctor_get(v___x_410_, 0);
v_invFun_412_ = lean_ctor_get(v___x_410_, 1);
v_isSharedCheck_419_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_419_ == 0)
{
v___x_414_ = v___x_410_;
v_isShared_415_ = v_isSharedCheck_419_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_invFun_412_);
lean_inc(v_toLinearMap_411_);
lean_dec(v___x_410_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_419_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_417_; 
if (v_isShared_415_ == 0)
{
v___x_417_ = v___x_414_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v_toLinearMap_411_);
lean_ctor_set(v_reuseFailAlloc_418_, 1, v_invFun_412_);
v___x_417_ = v_reuseFailAlloc_418_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
return v___x_417_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndRingEquivInt(lean_object* v_A_420_, lean_object* v_inst_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_addMonoidEndRingEquivInt___redArg(v_inst_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero___redArg(lean_object* v_inst_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v_toZero_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v_toZero_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_439_; 
v___x_425_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_424_);
v___x_426_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_425_);
v_toZero_427_ = lean_ctor_get(v___x_426_, 0);
lean_inc(v_toZero_427_);
lean_dec_ref(v___x_426_);
v___x_428_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_423_);
v___x_429_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_428_);
v_toZero_430_ = lean_ctor_get(v___x_429_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_429_);
if (v_isSharedCheck_439_ == 0)
{
lean_object* v_unused_440_; 
v_unused_440_ = lean_ctor_get(v___x_429_, 1);
lean_dec(v_unused_440_);
v___x_432_ = v___x_429_;
v_isShared_433_ = v_isSharedCheck_439_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_toZero_430_);
lean_dec(v___x_429_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_439_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___f_434_; lean_object* v___f_435_; lean_object* v___x_437_; 
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_434_, 0, v_toZero_427_);
v___f_435_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_435_, 0, v_toZero_430_);
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 1, v___f_435_);
lean_ctor_set(v___x_432_, 0, v___f_434_);
v___x_437_ = v___x_432_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v___f_434_);
lean_ctor_set(v_reuseFailAlloc_438_, 1, v___f_435_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero___redArg___boxed(lean_object* v_inst_441_, lean_object* v_inst_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_mathlib_LinearEquiv_instZero___redArg(v_inst_441_, v_inst_442_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_inst_441_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero(lean_object* v_R_444_, lean_object* v_R_u2082_445_, lean_object* v_M_446_, lean_object* v_M_u2082_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_00_u03c3_u2081_u2082_454_, lean_object* v_00_u03c3_u2082_u2081_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_LinearEquiv_instZero___redArg(v_inst_450_, v_inst_451_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instZero___boxed(lean_object* v_R_461_, lean_object* v_R_u2082_462_, lean_object* v_M_463_, lean_object* v_M_u2082_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_00_u03c3_u2081_u2082_471_, lean_object* v_00_u03c3_u2082_u2081_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_LinearEquiv_instZero(v_R_461_, v_R_u2082_462_, v_M_463_, v_M_u2082_464_, v_inst_465_, v_inst_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_, v_00_u03c3_u2081_u2082_471_, v_00_u03c3_u2082_u2081_472_, v_inst_473_, v_inst_474_, v_inst_475_, v_inst_476_);
lean_dec(v_00_u03c3_u2082_u2081_472_);
lean_dec(v_00_u03c3_u2081_u2082_471_);
lean_dec(v_inst_470_);
lean_dec(v_inst_469_);
lean_dec_ref(v_inst_468_);
lean_dec_ref(v_inst_467_);
lean_dec_ref(v_inst_466_);
lean_dec_ref(v_inst_465_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique___redArg(lean_object* v_inst_478_, lean_object* v_inst_479_){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v_toZero_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v_toZero_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_494_; 
v___x_480_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_479_);
v___x_481_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_480_);
v_toZero_482_ = lean_ctor_get(v___x_481_, 0);
lean_inc(v_toZero_482_);
lean_dec_ref(v___x_481_);
v___x_483_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_478_);
v___x_484_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_483_);
v_toZero_485_ = lean_ctor_get(v___x_484_, 0);
v_isSharedCheck_494_ = !lean_is_exclusive(v___x_484_);
if (v_isSharedCheck_494_ == 0)
{
lean_object* v_unused_495_; 
v_unused_495_ = lean_ctor_get(v___x_484_, 1);
lean_dec(v_unused_495_);
v___x_487_ = v___x_484_;
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_toZero_485_);
lean_dec(v___x_484_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___f_489_; lean_object* v___f_490_; lean_object* v___x_492_; 
v___f_489_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_489_, 0, v_toZero_482_);
v___f_490_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_490_, 0, v_toZero_485_);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 1, v___f_490_);
lean_ctor_set(v___x_487_, 0, v___f_489_);
v___x_492_ = v___x_487_;
goto v_reusejp_491_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v___f_489_);
lean_ctor_set(v_reuseFailAlloc_493_, 1, v___f_490_);
v___x_492_ = v_reuseFailAlloc_493_;
goto v_reusejp_491_;
}
v_reusejp_491_:
{
return v___x_492_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique___redArg___boxed(lean_object* v_inst_496_, lean_object* v_inst_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_LinearEquiv_instUnique___redArg(v_inst_496_, v_inst_497_);
lean_dec_ref(v_inst_497_);
lean_dec_ref(v_inst_496_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique(lean_object* v_R_499_, lean_object* v_R_u2082_500_, lean_object* v_M_501_, lean_object* v_M_u2082_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_00_u03c3_u2081_u2082_509_, lean_object* v_00_u03c3_u2082_u2081_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_mathlib_LinearEquiv_instUnique___redArg(v_inst_505_, v_inst_506_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instUnique___boxed(lean_object* v_R_516_, lean_object* v_R_u2082_517_, lean_object* v_M_518_, lean_object* v_M_u2082_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_00_u03c3_u2081_u2082_526_, lean_object* v_00_u03c3_u2082_u2081_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_inst_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_LinearEquiv_instUnique(v_R_516_, v_R_u2082_517_, v_M_518_, v_M_u2082_519_, v_inst_520_, v_inst_521_, v_inst_522_, v_inst_523_, v_inst_524_, v_inst_525_, v_00_u03c3_u2081_u2082_526_, v_00_u03c3_u2082_u2081_527_, v_inst_528_, v_inst_529_, v_inst_530_, v_inst_531_);
lean_dec(v_00_u03c3_u2082_u2081_527_);
lean_dec(v_00_u03c3_u2081_u2082_526_);
lean_dec(v_inst_525_);
lean_dec(v_inst_524_);
lean_dec_ref(v_inst_523_);
lean_dec_ref(v_inst_522_);
lean_dec_ref(v_inst_521_);
lean_dec_ref(v_inst_520_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton___redArg(lean_object* v_inst_533_, lean_object* v_inst_534_){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = lp_mathlib_LinearEquiv_instUnique___redArg(v_inst_533_, v_inst_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton___redArg___boxed(lean_object* v_inst_536_, lean_object* v_inst_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_LinearEquiv_uniqueOfSubsingleton___redArg(v_inst_536_, v_inst_537_);
lean_dec_ref(v_inst_537_);
lean_dec_ref(v_inst_536_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton(lean_object* v_R_539_, lean_object* v_R_u2082_540_, lean_object* v_M_541_, lean_object* v_M_u2082_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_00_u03c3_u2081_u2082_549_, lean_object* v_00_u03c3_u2082_u2081_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_mathlib_LinearEquiv_instUnique___redArg(v_inst_545_, v_inst_546_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueOfSubsingleton___boxed(lean_object* v_R_556_, lean_object* v_R_u2082_557_, lean_object* v_M_558_, lean_object* v_M_u2082_559_, lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_00_u03c3_u2081_u2082_566_, lean_object* v_00_u03c3_u2082_u2081_567_, lean_object* v_inst_568_, lean_object* v_inst_569_, lean_object* v_inst_570_, lean_object* v_inst_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_mathlib_LinearEquiv_uniqueOfSubsingleton(v_R_556_, v_R_u2082_557_, v_M_558_, v_M_u2082_559_, v_inst_560_, v_inst_561_, v_inst_562_, v_inst_563_, v_inst_564_, v_inst_565_, v_00_u03c3_u2081_u2082_566_, v_00_u03c3_u2082_u2081_567_, v_inst_568_, v_inst_569_, v_inst_570_, v_inst_571_);
lean_dec(v_00_u03c3_u2082_u2081_567_);
lean_dec(v_00_u03c3_u2081_u2082_566_);
lean_dec(v_inst_565_);
lean_dec(v_inst_564_);
lean_dec_ref(v_inst_563_);
lean_dec_ref(v_inst_562_);
lean_dec_ref(v_inst_561_);
lean_dec_ref(v_inst_560_);
return v_res_572_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_curry___closed__0(void){
_start:
{
lean_object* v___x_573_; 
v___x_573_ = lp_mathlib_Equiv_curry(lean_box(0), lean_box(0), lean_box(0));
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_curry(lean_object* v_R_574_, lean_object* v_M_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_V_579_, lean_object* v_V_u2082_580_){
_start:
{
lean_object* v___x_581_; lean_object* v_toFun_582_; lean_object* v_invFun_583_; lean_object* v___x_584_; 
v___x_581_ = lean_obj_once(&lp_mathlib_LinearEquiv_curry___closed__0, &lp_mathlib_LinearEquiv_curry___closed__0_once, _init_lp_mathlib_LinearEquiv_curry___closed__0);
v_toFun_582_ = lean_ctor_get(v___x_581_, 0);
v_invFun_583_ = lean_ctor_get(v___x_581_, 1);
lean_inc(v_invFun_583_);
lean_inc(v_toFun_582_);
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v_toFun_582_);
lean_ctor_set(v___x_584_, 1, v_invFun_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_curry___boxed(lean_object* v_R_585_, lean_object* v_M_586_, lean_object* v_inst_587_, lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_V_590_, lean_object* v_V_u2082_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_LinearEquiv_curry(v_R_585_, v_M_586_, v_inst_587_, v_inst_588_, v_inst_589_, v_V_590_, v_V_u2082_591_);
lean_dec(v_inst_589_);
lean_dec_ref(v_inst_588_);
lean_dec_ref(v_inst_587_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg___lam__0(lean_object* v_g_593_, lean_object* v___y_594_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lean_apply_1(v_g_593_, v___y_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg(lean_object* v_f_596_, lean_object* v_g_597_){
_start:
{
lean_object* v___f_598_; lean_object* v___x_599_; 
v___f_598_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_598_, 0, v_g_597_);
v___x_599_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_599_, 0, v_f_596_);
lean_ctor_set(v___x_599_, 1, v___f_598_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap(lean_object* v_R_600_, lean_object* v_R_u2082_601_, lean_object* v_M_602_, lean_object* v_M_u2082_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_inst_607_, lean_object* v_module__M_608_, lean_object* v_module__M_u2082_609_, lean_object* v_00_u03c3_u2081_u2082_610_, lean_object* v_00_u03c3_u2082_u2081_611_, lean_object* v_re_u2081_u2082_612_, lean_object* v_re_u2082_u2081_613_, lean_object* v_f_614_, lean_object* v_g_615_, lean_object* v_h_u2081_616_, lean_object* v_h_u2082_617_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v_f_614_, v_g_615_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinearMap___boxed(lean_object** _args){
lean_object* v_R_619_ = _args[0];
lean_object* v_R_u2082_620_ = _args[1];
lean_object* v_M_621_ = _args[2];
lean_object* v_M_u2082_622_ = _args[3];
lean_object* v_inst_623_ = _args[4];
lean_object* v_inst_624_ = _args[5];
lean_object* v_inst_625_ = _args[6];
lean_object* v_inst_626_ = _args[7];
lean_object* v_module__M_627_ = _args[8];
lean_object* v_module__M_u2082_628_ = _args[9];
lean_object* v_00_u03c3_u2081_u2082_629_ = _args[10];
lean_object* v_00_u03c3_u2082_u2081_630_ = _args[11];
lean_object* v_re_u2081_u2082_631_ = _args[12];
lean_object* v_re_u2082_u2081_632_ = _args[13];
lean_object* v_f_633_ = _args[14];
lean_object* v_g_634_ = _args[15];
lean_object* v_h_u2081_635_ = _args[16];
lean_object* v_h_u2082_636_ = _args[17];
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_LinearEquiv_ofLinearMap(v_R_619_, v_R_u2082_620_, v_M_621_, v_M_u2082_622_, v_inst_623_, v_inst_624_, v_inst_625_, v_inst_626_, v_module__M_627_, v_module__M_u2082_628_, v_00_u03c3_u2081_u2082_629_, v_00_u03c3_u2082_u2081_630_, v_re_u2081_u2082_631_, v_re_u2082_u2081_632_, v_f_633_, v_g_634_, v_h_u2081_635_, v_h_u2082_636_);
lean_dec(v_00_u03c3_u2082_u2081_630_);
lean_dec(v_00_u03c3_u2081_u2082_629_);
lean_dec(v_module__M_u2082_628_);
lean_dec(v_module__M_627_);
lean_dec_ref(v_inst_626_);
lean_dec_ref(v_inst_625_);
lean_dec_ref(v_inst_624_);
lean_dec_ref(v_inst_623_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinear___redArg(lean_object* v_f_638_, lean_object* v_g_639_){
_start:
{
lean_object* v___x_640_; 
v___x_640_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v_f_638_, v_g_639_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinear(lean_object* v_R_641_, lean_object* v_R_u2082_642_, lean_object* v_M_643_, lean_object* v_M_u2082_644_, lean_object* v_inst_645_, lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_module__M_649_, lean_object* v_module__M_u2082_650_, lean_object* v_00_u03c3_u2081_u2082_651_, lean_object* v_00_u03c3_u2082_u2081_652_, lean_object* v_re_u2081_u2082_653_, lean_object* v_re_u2082_u2081_654_, lean_object* v_f_655_, lean_object* v_g_656_, lean_object* v_h_u2081_657_, lean_object* v_h_u2082_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v_f_655_, v_g_656_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofLinear___boxed(lean_object** _args){
lean_object* v_R_660_ = _args[0];
lean_object* v_R_u2082_661_ = _args[1];
lean_object* v_M_662_ = _args[2];
lean_object* v_M_u2082_663_ = _args[3];
lean_object* v_inst_664_ = _args[4];
lean_object* v_inst_665_ = _args[5];
lean_object* v_inst_666_ = _args[6];
lean_object* v_inst_667_ = _args[7];
lean_object* v_module__M_668_ = _args[8];
lean_object* v_module__M_u2082_669_ = _args[9];
lean_object* v_00_u03c3_u2081_u2082_670_ = _args[10];
lean_object* v_00_u03c3_u2082_u2081_671_ = _args[11];
lean_object* v_re_u2081_u2082_672_ = _args[12];
lean_object* v_re_u2082_u2081_673_ = _args[13];
lean_object* v_f_674_ = _args[14];
lean_object* v_g_675_ = _args[15];
lean_object* v_h_u2081_676_ = _args[16];
lean_object* v_h_u2082_677_ = _args[17];
_start:
{
lean_object* v_res_678_; 
v_res_678_ = lp_mathlib_LinearEquiv_ofLinear(v_R_660_, v_R_u2082_661_, v_M_662_, v_M_u2082_663_, v_inst_664_, v_inst_665_, v_inst_666_, v_inst_667_, v_module__M_668_, v_module__M_u2082_669_, v_00_u03c3_u2081_u2082_670_, v_00_u03c3_u2082_u2081_671_, v_re_u2081_u2082_672_, v_re_u2082_u2081_673_, v_f_674_, v_g_675_, v_h_u2081_676_, v_h_u2082_677_);
lean_dec(v_00_u03c3_u2082_u2081_671_);
lean_dec(v_00_u03c3_u2081_u2082_670_);
lean_dec(v_module__M_u2082_669_);
lean_dec(v_module__M_668_);
lean_dec_ref(v_inst_667_);
lean_dec_ref(v_inst_666_);
lean_dec_ref(v_inst_665_);
lean_dec_ref(v_inst_664_);
return v_res_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg___redArg(lean_object* v_inst_679_){
_start:
{
lean_object* v_toNeg_680_; lean_object* v___x_681_; 
v_toNeg_680_ = lean_ctor_get(v_inst_679_, 1);
lean_inc_n(v_toNeg_680_, 2);
v___x_681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_681_, 0, v_toNeg_680_);
lean_ctor_set(v___x_681_, 1, v_toNeg_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg___redArg___boxed(lean_object* v_inst_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_LinearEquiv_neg___redArg(v_inst_682_);
lean_dec_ref(v_inst_682_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg(lean_object* v_R_684_, lean_object* v_M_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_){
_start:
{
lean_object* v___x_689_; 
v___x_689_ = lp_mathlib_LinearEquiv_neg___redArg(v_inst_687_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_neg___boxed(lean_object* v_R_690_, lean_object* v_M_691_, lean_object* v_inst_692_, lean_object* v_inst_693_, lean_object* v_inst_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib_LinearEquiv_neg(v_R_690_, v_M_691_, v_inst_692_, v_inst_693_, v_inst_694_);
lean_dec(v_inst_694_);
lean_dec_ref(v_inst_693_);
lean_dec_ref(v_inst_692_);
return v_res_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg___lam__0(lean_object* v_e_u2082_696_, lean_object* v_e_u2081_697_, lean_object* v_f_698_, lean_object* v___y_699_){
_start:
{
lean_object* v_toLinearMap_700_; lean_object* v___x_701_; lean_object* v_toLinearMap_702_; lean_object* v___f_703_; lean_object* v___x_704_; 
v_toLinearMap_700_ = lean_ctor_get(v_e_u2082_696_, 0);
lean_inc(v_toLinearMap_700_);
lean_dec_ref(v_e_u2082_696_);
v___x_701_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_u2081_697_);
v_toLinearMap_702_ = lean_ctor_get(v___x_701_, 0);
lean_inc(v_toLinearMap_702_);
lean_dec_ref(v___x_701_);
v___f_703_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_703_, 0, v_f_698_);
lean_closure_set(v___f_703_, 1, v_toLinearMap_700_);
v___x_704_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v_toLinearMap_702_, v___f_703_, v___y_699_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg___lam__1(lean_object* v_e_u2082_705_, lean_object* v_e_u2081_706_, lean_object* v_f_707_, lean_object* v___y_708_){
_start:
{
lean_object* v___x_709_; lean_object* v_toLinearMap_710_; lean_object* v_toLinearMap_711_; lean_object* v___f_712_; lean_object* v___x_713_; 
v___x_709_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_u2082_705_);
v_toLinearMap_710_ = lean_ctor_get(v___x_709_, 0);
lean_inc(v_toLinearMap_710_);
lean_dec_ref(v___x_709_);
v_toLinearMap_711_ = lean_ctor_get(v_e_u2081_706_, 0);
lean_inc(v_toLinearMap_711_);
lean_dec_ref(v_e_u2081_706_);
v___f_712_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_712_, 0, v_f_707_);
lean_closure_set(v___f_712_, 1, v_toLinearMap_710_);
v___x_713_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v_toLinearMap_711_, v___f_712_, v___y_708_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(lean_object* v_e_u2081_714_, lean_object* v_e_u2082_715_){
_start:
{
lean_object* v___f_716_; lean_object* v___f_717_; lean_object* v___x_718_; 
lean_inc_ref(v_e_u2081_714_);
lean_inc_ref(v_e_u2082_715_);
v___f_716_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg___lam__0), 4, 2);
lean_closure_set(v___f_716_, 0, v_e_u2082_715_);
lean_closure_set(v___f_716_, 1, v_e_u2081_714_);
v___f_717_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg___lam__1), 4, 2);
lean_closure_set(v___f_717_, 0, v_e_u2082_715_);
lean_closure_set(v___f_717_, 1, v_e_u2081_714_);
v___x_718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_718_, 0, v___f_716_);
lean_ctor_set(v___x_718_, 1, v___f_717_);
return v___x_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv(lean_object* v_R_u2081_719_, lean_object* v_R_u2082_720_, lean_object* v_R_u2081_x27_721_, lean_object* v_R_u2082_x27_722_, lean_object* v_M_u2081_723_, lean_object* v_M_u2082_724_, lean_object* v_M_u2081_x27_725_, lean_object* v_M_u2082_x27_726_, lean_object* v_inst_727_, lean_object* v_inst_728_, lean_object* v_inst_729_, lean_object* v_inst_730_, lean_object* v_inst_731_, lean_object* v_inst_732_, lean_object* v_inst_733_, lean_object* v_inst_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_00_u03c3_u2081_u2082_739_, lean_object* v_00_u03c3_u2082_u2081_740_, lean_object* v_00_u03c3_u2081_x27_u2082_x27_741_, lean_object* v_00_u03c3_u2082_x27_u2081_x27_742_, lean_object* v_00_u03c3_u2081_u2081_x27_743_, lean_object* v_00_u03c3_u2082_u2082_x27_744_, lean_object* v_00_u03c3_u2082_u2081_x27_745_, lean_object* v_00_u03c3_u2081_u2082_x27_746_, lean_object* v_inst_747_, lean_object* v_inst_748_, lean_object* v_inst_749_, lean_object* v_inst_750_, lean_object* v_inst_751_, lean_object* v_inst_752_, lean_object* v_inst_753_, lean_object* v_inst_754_, lean_object* v_e_u2081_755_, lean_object* v_e_u2082_756_){
_start:
{
lean_object* v___x_757_; 
v___x_757_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_u2081_755_, v_e_u2082_756_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___boxed(lean_object** _args){
lean_object* v_R_u2081_758_ = _args[0];
lean_object* v_R_u2082_759_ = _args[1];
lean_object* v_R_u2081_x27_760_ = _args[2];
lean_object* v_R_u2082_x27_761_ = _args[3];
lean_object* v_M_u2081_762_ = _args[4];
lean_object* v_M_u2082_763_ = _args[5];
lean_object* v_M_u2081_x27_764_ = _args[6];
lean_object* v_M_u2082_x27_765_ = _args[7];
lean_object* v_inst_766_ = _args[8];
lean_object* v_inst_767_ = _args[9];
lean_object* v_inst_768_ = _args[10];
lean_object* v_inst_769_ = _args[11];
lean_object* v_inst_770_ = _args[12];
lean_object* v_inst_771_ = _args[13];
lean_object* v_inst_772_ = _args[14];
lean_object* v_inst_773_ = _args[15];
lean_object* v_inst_774_ = _args[16];
lean_object* v_inst_775_ = _args[17];
lean_object* v_inst_776_ = _args[18];
lean_object* v_inst_777_ = _args[19];
lean_object* v_00_u03c3_u2081_u2082_778_ = _args[20];
lean_object* v_00_u03c3_u2082_u2081_779_ = _args[21];
lean_object* v_00_u03c3_u2081_x27_u2082_x27_780_ = _args[22];
lean_object* v_00_u03c3_u2082_x27_u2081_x27_781_ = _args[23];
lean_object* v_00_u03c3_u2081_u2081_x27_782_ = _args[24];
lean_object* v_00_u03c3_u2082_u2082_x27_783_ = _args[25];
lean_object* v_00_u03c3_u2082_u2081_x27_784_ = _args[26];
lean_object* v_00_u03c3_u2081_u2082_x27_785_ = _args[27];
lean_object* v_inst_786_ = _args[28];
lean_object* v_inst_787_ = _args[29];
lean_object* v_inst_788_ = _args[30];
lean_object* v_inst_789_ = _args[31];
lean_object* v_inst_790_ = _args[32];
lean_object* v_inst_791_ = _args[33];
lean_object* v_inst_792_ = _args[34];
lean_object* v_inst_793_ = _args[35];
lean_object* v_e_u2081_794_ = _args[36];
lean_object* v_e_u2082_795_ = _args[37];
_start:
{
lean_object* v_res_796_; 
v_res_796_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv(v_R_u2081_758_, v_R_u2082_759_, v_R_u2081_x27_760_, v_R_u2082_x27_761_, v_M_u2081_762_, v_M_u2082_763_, v_M_u2081_x27_764_, v_M_u2082_x27_765_, v_inst_766_, v_inst_767_, v_inst_768_, v_inst_769_, v_inst_770_, v_inst_771_, v_inst_772_, v_inst_773_, v_inst_774_, v_inst_775_, v_inst_776_, v_inst_777_, v_00_u03c3_u2081_u2082_778_, v_00_u03c3_u2082_u2081_779_, v_00_u03c3_u2081_x27_u2082_x27_780_, v_00_u03c3_u2082_x27_u2081_x27_781_, v_00_u03c3_u2081_u2081_x27_782_, v_00_u03c3_u2082_u2082_x27_783_, v_00_u03c3_u2082_u2081_x27_784_, v_00_u03c3_u2081_u2082_x27_785_, v_inst_786_, v_inst_787_, v_inst_788_, v_inst_789_, v_inst_790_, v_inst_791_, v_inst_792_, v_inst_793_, v_e_u2081_794_, v_e_u2082_795_);
lean_dec(v_00_u03c3_u2081_u2082_x27_785_);
lean_dec(v_00_u03c3_u2082_u2081_x27_784_);
lean_dec(v_00_u03c3_u2082_u2082_x27_783_);
lean_dec(v_00_u03c3_u2081_u2081_x27_782_);
lean_dec(v_00_u03c3_u2082_x27_u2081_x27_781_);
lean_dec(v_00_u03c3_u2081_x27_u2082_x27_780_);
lean_dec(v_00_u03c3_u2082_u2081_779_);
lean_dec(v_00_u03c3_u2081_u2082_778_);
lean_dec(v_inst_777_);
lean_dec(v_inst_776_);
lean_dec(v_inst_775_);
lean_dec(v_inst_774_);
lean_dec_ref(v_inst_773_);
lean_dec_ref(v_inst_772_);
lean_dec_ref(v_inst_771_);
lean_dec_ref(v_inst_770_);
lean_dec_ref(v_inst_769_);
lean_dec_ref(v_inst_768_);
lean_dec_ref(v_inst_767_);
lean_dec_ref(v_inst_766_);
return v_res_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjRingEquiv___redArg(lean_object* v_e_797_){
_start:
{
lean_object* v___x_798_; 
lean_inc_ref(v_e_797_);
v___x_798_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_797_, v_e_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjRingEquiv(lean_object* v_R_u2081_799_, lean_object* v_R_u2082_800_, lean_object* v_M_u2081_801_, lean_object* v_M_u2082_802_, lean_object* v_inst_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_, lean_object* v_00_u03c3_u2081_u2082_809_, lean_object* v_00_u03c3_u2082_u2081_810_, lean_object* v_inst_811_, lean_object* v_inst_812_, lean_object* v_e_813_){
_start:
{
lean_object* v___x_814_; 
lean_inc_ref(v_e_813_);
v___x_814_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_813_, v_e_813_);
return v___x_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjRingEquiv___boxed(lean_object* v_R_u2081_815_, lean_object* v_R_u2082_816_, lean_object* v_M_u2081_817_, lean_object* v_M_u2082_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_, lean_object* v_inst_822_, lean_object* v_inst_823_, lean_object* v_inst_824_, lean_object* v_00_u03c3_u2081_u2082_825_, lean_object* v_00_u03c3_u2082_u2081_826_, lean_object* v_inst_827_, lean_object* v_inst_828_, lean_object* v_e_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_mathlib_LinearEquiv_conjRingEquiv(v_R_u2081_815_, v_R_u2082_816_, v_M_u2081_817_, v_M_u2082_818_, v_inst_819_, v_inst_820_, v_inst_821_, v_inst_822_, v_inst_823_, v_inst_824_, v_00_u03c3_u2081_u2082_825_, v_00_u03c3_u2082_u2081_826_, v_inst_827_, v_inst_828_, v_e_829_);
lean_dec(v_00_u03c3_u2082_u2081_826_);
lean_dec(v_00_u03c3_u2081_u2082_825_);
lean_dec(v_inst_824_);
lean_dec(v_inst_823_);
lean_dec_ref(v_inst_822_);
lean_dec_ref(v_inst_821_);
lean_dec_ref(v_inst_820_);
lean_dec_ref(v_inst_819_);
return v_res_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight___redArg(lean_object* v_inst_831_, lean_object* v_inst_832_, lean_object* v_inst_833_, lean_object* v_e_u2082_834_){
_start:
{
lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v_toFun_837_; lean_object* v_invFun_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_845_; 
v___x_835_ = lp_mathlib_LinearEquiv_refl(lean_box(0), lean_box(0), v_inst_831_, v_inst_832_, v_inst_833_);
v___x_836_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v___x_835_, v_e_u2082_834_);
v_toFun_837_ = lean_ctor_get(v___x_836_, 0);
v_invFun_838_ = lean_ctor_get(v___x_836_, 1);
v_isSharedCheck_845_ = !lean_is_exclusive(v___x_836_);
if (v_isSharedCheck_845_ == 0)
{
v___x_840_ = v___x_836_;
v_isShared_841_ = v_isSharedCheck_845_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_invFun_838_);
lean_inc(v_toFun_837_);
lean_dec(v___x_836_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_845_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
lean_object* v___x_843_; 
if (v_isShared_841_ == 0)
{
v___x_843_ = v___x_840_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v_toFun_837_);
lean_ctor_set(v_reuseFailAlloc_844_, 1, v_invFun_838_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight___redArg___boxed(lean_object* v_inst_846_, lean_object* v_inst_847_, lean_object* v_inst_848_, lean_object* v_e_u2082_849_){
_start:
{
lean_object* v_res_850_; 
v_res_850_ = lp_mathlib_LinearEquiv_domMulActCongrRight___redArg(v_inst_846_, v_inst_847_, v_inst_848_, v_e_u2082_849_);
lean_dec(v_inst_848_);
lean_dec_ref(v_inst_847_);
lean_dec_ref(v_inst_846_);
return v_res_850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight(lean_object* v_S_851_, lean_object* v_R_u2081_852_, lean_object* v_R_u2081_x27_853_, lean_object* v_R_u2082_x27_854_, lean_object* v_M_u2081_855_, lean_object* v_M_u2081_x27_856_, lean_object* v_M_u2082_x27_857_, lean_object* v_inst_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_00_u03c3_u2081_x27_u2082_x27_867_, lean_object* v_00_u03c3_u2082_x27_u2081_x27_868_, lean_object* v_00_u03c3_u2081_u2081_x27_869_, lean_object* v_00_u03c3_u2081_u2082_x27_870_, lean_object* v_inst_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_inst_877_, lean_object* v_e_u2082_878_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_mathlib_LinearEquiv_domMulActCongrRight___redArg(v_inst_858_, v_inst_861_, v_inst_864_, v_e_u2082_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_domMulActCongrRight___boxed(lean_object** _args){
lean_object* v_S_880_ = _args[0];
lean_object* v_R_u2081_881_ = _args[1];
lean_object* v_R_u2081_x27_882_ = _args[2];
lean_object* v_R_u2082_x27_883_ = _args[3];
lean_object* v_M_u2081_884_ = _args[4];
lean_object* v_M_u2081_x27_885_ = _args[5];
lean_object* v_M_u2082_x27_886_ = _args[6];
lean_object* v_inst_887_ = _args[7];
lean_object* v_inst_888_ = _args[8];
lean_object* v_inst_889_ = _args[9];
lean_object* v_inst_890_ = _args[10];
lean_object* v_inst_891_ = _args[11];
lean_object* v_inst_892_ = _args[12];
lean_object* v_inst_893_ = _args[13];
lean_object* v_inst_894_ = _args[14];
lean_object* v_inst_895_ = _args[15];
lean_object* v_00_u03c3_u2081_x27_u2082_x27_896_ = _args[16];
lean_object* v_00_u03c3_u2082_x27_u2081_x27_897_ = _args[17];
lean_object* v_00_u03c3_u2081_u2081_x27_898_ = _args[18];
lean_object* v_00_u03c3_u2081_u2082_x27_899_ = _args[19];
lean_object* v_inst_900_ = _args[20];
lean_object* v_inst_901_ = _args[21];
lean_object* v_inst_902_ = _args[22];
lean_object* v_inst_903_ = _args[23];
lean_object* v_inst_904_ = _args[24];
lean_object* v_inst_905_ = _args[25];
lean_object* v_inst_906_ = _args[26];
lean_object* v_e_u2082_907_ = _args[27];
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_mathlib_LinearEquiv_domMulActCongrRight(v_S_880_, v_R_u2081_881_, v_R_u2081_x27_882_, v_R_u2082_x27_883_, v_M_u2081_884_, v_M_u2081_x27_885_, v_M_u2082_x27_886_, v_inst_887_, v_inst_888_, v_inst_889_, v_inst_890_, v_inst_891_, v_inst_892_, v_inst_893_, v_inst_894_, v_inst_895_, v_00_u03c3_u2081_x27_u2082_x27_896_, v_00_u03c3_u2082_x27_u2081_x27_897_, v_00_u03c3_u2081_u2081_x27_898_, v_00_u03c3_u2081_u2082_x27_899_, v_inst_900_, v_inst_901_, v_inst_902_, v_inst_903_, v_inst_904_, v_inst_905_, v_inst_906_, v_e_u2082_907_);
lean_dec(v_inst_904_);
lean_dec_ref(v_inst_903_);
lean_dec(v_00_u03c3_u2081_u2082_x27_899_);
lean_dec(v_00_u03c3_u2081_u2081_x27_898_);
lean_dec(v_00_u03c3_u2082_x27_u2081_x27_897_);
lean_dec(v_00_u03c3_u2081_x27_u2082_x27_896_);
lean_dec(v_inst_895_);
lean_dec(v_inst_894_);
lean_dec(v_inst_893_);
lean_dec_ref(v_inst_892_);
lean_dec_ref(v_inst_891_);
lean_dec_ref(v_inst_890_);
lean_dec_ref(v_inst_889_);
lean_dec_ref(v_inst_888_);
lean_dec_ref(v_inst_887_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfUnit___redArg(lean_object* v_inst_909_, lean_object* v_inst_910_, lean_object* v_a_911_){
_start:
{
lean_object* v_toMonoid_912_; lean_object* v___x_913_; lean_object* v___f_914_; lean_object* v___x_915_; 
v_toMonoid_912_ = lean_ctor_get(v_inst_909_, 1);
lean_inc_ref(v_toMonoid_912_);
lean_dec_ref(v_inst_909_);
v___x_913_ = lp_mathlib_Units_instDivInvMonoid___redArg(v_toMonoid_912_);
v___f_914_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_914_, 0, v_inst_910_);
v___x_915_ = lp_mathlib_DistribMulAction_toLinearEquiv___redArg(v___x_913_, v___f_914_, v_a_911_);
lean_dec_ref(v___x_913_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfUnit(lean_object* v_R_916_, lean_object* v_M_917_, lean_object* v_inst_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_a_921_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = lp_mathlib_LinearEquiv_smulOfUnit___redArg(v_inst_918_, v_inst_920_, v_a_921_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfUnit___boxed(lean_object* v_R_923_, lean_object* v_M_924_, lean_object* v_inst_925_, lean_object* v_inst_926_, lean_object* v_inst_927_, lean_object* v_a_928_){
_start:
{
lean_object* v_res_929_; 
v_res_929_ = lp_mathlib_LinearEquiv_smulOfUnit(v_R_923_, v_M_924_, v_inst_925_, v_inst_926_, v_inst_927_, v_a_928_);
lean_dec_ref(v_inst_926_);
return v_res_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongr___redArg(lean_object* v_e_u2081_930_, lean_object* v_e_u2082_931_){
_start:
{
lean_object* v___x_932_; lean_object* v_toFun_933_; lean_object* v_invFun_934_; lean_object* v___x_936_; uint8_t v_isShared_937_; uint8_t v_isSharedCheck_941_; 
v___x_932_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_u2081_930_, v_e_u2082_931_);
v_toFun_933_ = lean_ctor_get(v___x_932_, 0);
v_invFun_934_ = lean_ctor_get(v___x_932_, 1);
v_isSharedCheck_941_ = !lean_is_exclusive(v___x_932_);
if (v_isSharedCheck_941_ == 0)
{
v___x_936_ = v___x_932_;
v_isShared_937_ = v_isSharedCheck_941_;
goto v_resetjp_935_;
}
else
{
lean_inc(v_invFun_934_);
lean_inc(v_toFun_933_);
lean_dec(v___x_932_);
v___x_936_ = lean_box(0);
v_isShared_937_ = v_isSharedCheck_941_;
goto v_resetjp_935_;
}
v_resetjp_935_:
{
lean_object* v___x_939_; 
if (v_isShared_937_ == 0)
{
v___x_939_ = v___x_936_;
goto v_reusejp_938_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v_toFun_933_);
lean_ctor_set(v_reuseFailAlloc_940_, 1, v_invFun_934_);
v___x_939_ = v_reuseFailAlloc_940_;
goto v_reusejp_938_;
}
v_reusejp_938_:
{
return v___x_939_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongr(lean_object* v_R_u2081_942_, lean_object* v_R_u2082_943_, lean_object* v_R_u2081_x27_944_, lean_object* v_R_u2082_x27_945_, lean_object* v_M_u2081_946_, lean_object* v_M_u2082_947_, lean_object* v_M_u2081_x27_948_, lean_object* v_M_u2082_x27_949_, lean_object* v_inst_950_, lean_object* v_inst_951_, lean_object* v_inst_952_, lean_object* v_inst_953_, lean_object* v_inst_954_, lean_object* v_inst_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_inst_958_, lean_object* v_inst_959_, lean_object* v_inst_960_, lean_object* v_inst_961_, lean_object* v_00_u03c3_u2081_u2082_962_, lean_object* v_00_u03c3_u2082_u2081_963_, lean_object* v_00_u03c3_u2081_x27_u2082_x27_964_, lean_object* v_00_u03c3_u2082_x27_u2081_x27_965_, lean_object* v_00_u03c3_u2081_u2081_x27_966_, lean_object* v_00_u03c3_u2082_u2082_x27_967_, lean_object* v_00_u03c3_u2082_u2081_x27_968_, lean_object* v_00_u03c3_u2081_u2082_x27_969_, lean_object* v_inst_970_, lean_object* v_inst_971_, lean_object* v_inst_972_, lean_object* v_inst_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_inst_976_, lean_object* v_inst_977_, lean_object* v_e_u2081_978_, lean_object* v_e_u2082_979_){
_start:
{
lean_object* v___x_980_; 
v___x_980_ = lp_mathlib_LinearEquiv_arrowCongr___redArg(v_e_u2081_978_, v_e_u2082_979_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_arrowCongr___boxed(lean_object** _args){
lean_object* v_R_u2081_981_ = _args[0];
lean_object* v_R_u2082_982_ = _args[1];
lean_object* v_R_u2081_x27_983_ = _args[2];
lean_object* v_R_u2082_x27_984_ = _args[3];
lean_object* v_M_u2081_985_ = _args[4];
lean_object* v_M_u2082_986_ = _args[5];
lean_object* v_M_u2081_x27_987_ = _args[6];
lean_object* v_M_u2082_x27_988_ = _args[7];
lean_object* v_inst_989_ = _args[8];
lean_object* v_inst_990_ = _args[9];
lean_object* v_inst_991_ = _args[10];
lean_object* v_inst_992_ = _args[11];
lean_object* v_inst_993_ = _args[12];
lean_object* v_inst_994_ = _args[13];
lean_object* v_inst_995_ = _args[14];
lean_object* v_inst_996_ = _args[15];
lean_object* v_inst_997_ = _args[16];
lean_object* v_inst_998_ = _args[17];
lean_object* v_inst_999_ = _args[18];
lean_object* v_inst_1000_ = _args[19];
lean_object* v_00_u03c3_u2081_u2082_1001_ = _args[20];
lean_object* v_00_u03c3_u2082_u2081_1002_ = _args[21];
lean_object* v_00_u03c3_u2081_x27_u2082_x27_1003_ = _args[22];
lean_object* v_00_u03c3_u2082_x27_u2081_x27_1004_ = _args[23];
lean_object* v_00_u03c3_u2081_u2081_x27_1005_ = _args[24];
lean_object* v_00_u03c3_u2082_u2082_x27_1006_ = _args[25];
lean_object* v_00_u03c3_u2082_u2081_x27_1007_ = _args[26];
lean_object* v_00_u03c3_u2081_u2082_x27_1008_ = _args[27];
lean_object* v_inst_1009_ = _args[28];
lean_object* v_inst_1010_ = _args[29];
lean_object* v_inst_1011_ = _args[30];
lean_object* v_inst_1012_ = _args[31];
lean_object* v_inst_1013_ = _args[32];
lean_object* v_inst_1014_ = _args[33];
lean_object* v_inst_1015_ = _args[34];
lean_object* v_inst_1016_ = _args[35];
lean_object* v_e_u2081_1017_ = _args[36];
lean_object* v_e_u2082_1018_ = _args[37];
_start:
{
lean_object* v_res_1019_; 
v_res_1019_ = lp_mathlib_LinearEquiv_arrowCongr(v_R_u2081_981_, v_R_u2082_982_, v_R_u2081_x27_983_, v_R_u2082_x27_984_, v_M_u2081_985_, v_M_u2082_986_, v_M_u2081_x27_987_, v_M_u2082_x27_988_, v_inst_989_, v_inst_990_, v_inst_991_, v_inst_992_, v_inst_993_, v_inst_994_, v_inst_995_, v_inst_996_, v_inst_997_, v_inst_998_, v_inst_999_, v_inst_1000_, v_00_u03c3_u2081_u2082_1001_, v_00_u03c3_u2082_u2081_1002_, v_00_u03c3_u2081_x27_u2082_x27_1003_, v_00_u03c3_u2082_x27_u2081_x27_1004_, v_00_u03c3_u2081_u2081_x27_1005_, v_00_u03c3_u2082_u2082_x27_1006_, v_00_u03c3_u2082_u2081_x27_1007_, v_00_u03c3_u2081_u2082_x27_1008_, v_inst_1009_, v_inst_1010_, v_inst_1011_, v_inst_1012_, v_inst_1013_, v_inst_1014_, v_inst_1015_, v_inst_1016_, v_e_u2081_1017_, v_e_u2082_1018_);
lean_dec(v_00_u03c3_u2081_u2082_x27_1008_);
lean_dec(v_00_u03c3_u2082_u2081_x27_1007_);
lean_dec(v_00_u03c3_u2082_u2082_x27_1006_);
lean_dec(v_00_u03c3_u2081_u2081_x27_1005_);
lean_dec(v_00_u03c3_u2082_x27_u2081_x27_1004_);
lean_dec(v_00_u03c3_u2081_x27_u2082_x27_1003_);
lean_dec(v_00_u03c3_u2082_u2081_1002_);
lean_dec(v_00_u03c3_u2081_u2082_1001_);
lean_dec(v_inst_1000_);
lean_dec(v_inst_999_);
lean_dec(v_inst_998_);
lean_dec(v_inst_997_);
lean_dec_ref(v_inst_996_);
lean_dec_ref(v_inst_995_);
lean_dec_ref(v_inst_994_);
lean_dec_ref(v_inst_993_);
lean_dec_ref(v_inst_992_);
lean_dec_ref(v_inst_991_);
lean_dec_ref(v_inst_990_);
lean_dec_ref(v_inst_989_);
return v_res_1019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conj___redArg(lean_object* v_e_1020_){
_start:
{
lean_object* v___x_1021_; 
lean_inc_ref(v_e_1020_);
v___x_1021_ = lp_mathlib_LinearEquiv_arrowCongr___redArg(v_e_1020_, v_e_1020_);
return v___x_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conj(lean_object* v_R_u2081_x27_1022_, lean_object* v_R_u2082_x27_1023_, lean_object* v_M_u2081_x27_1024_, lean_object* v_M_u2082_x27_1025_, lean_object* v_inst_1026_, lean_object* v_inst_1027_, lean_object* v_inst_1028_, lean_object* v_inst_1029_, lean_object* v_inst_1030_, lean_object* v_inst_1031_, lean_object* v_00_u03c3_u2081_x27_u2082_x27_1032_, lean_object* v_00_u03c3_u2082_x27_u2081_x27_1033_, lean_object* v_inst_1034_, lean_object* v_inst_1035_, lean_object* v_e_1036_){
_start:
{
lean_object* v___x_1037_; 
lean_inc_ref(v_e_1036_);
v___x_1037_ = lp_mathlib_LinearEquiv_arrowCongr___redArg(v_e_1036_, v_e_1036_);
return v___x_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conj___boxed(lean_object* v_R_u2081_x27_1038_, lean_object* v_R_u2082_x27_1039_, lean_object* v_M_u2081_x27_1040_, lean_object* v_M_u2082_x27_1041_, lean_object* v_inst_1042_, lean_object* v_inst_1043_, lean_object* v_inst_1044_, lean_object* v_inst_1045_, lean_object* v_inst_1046_, lean_object* v_inst_1047_, lean_object* v_00_u03c3_u2081_x27_u2082_x27_1048_, lean_object* v_00_u03c3_u2082_x27_u2081_x27_1049_, lean_object* v_inst_1050_, lean_object* v_inst_1051_, lean_object* v_e_1052_){
_start:
{
lean_object* v_res_1053_; 
v_res_1053_ = lp_mathlib_LinearEquiv_conj(v_R_u2081_x27_1038_, v_R_u2082_x27_1039_, v_M_u2081_x27_1040_, v_M_u2082_x27_1041_, v_inst_1042_, v_inst_1043_, v_inst_1044_, v_inst_1045_, v_inst_1046_, v_inst_1047_, v_00_u03c3_u2081_x27_u2082_x27_1048_, v_00_u03c3_u2082_x27_u2081_x27_1049_, v_inst_1050_, v_inst_1051_, v_e_1052_);
lean_dec(v_00_u03c3_u2082_x27_u2081_x27_1049_);
lean_dec(v_00_u03c3_u2081_x27_u2082_x27_1048_);
lean_dec(v_inst_1047_);
lean_dec(v_inst_1046_);
lean_dec_ref(v_inst_1045_);
lean_dec_ref(v_inst_1044_);
lean_dec_ref(v_inst_1043_);
lean_dec_ref(v_inst_1042_);
return v_res_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight___redArg(lean_object* v_inst_1054_, lean_object* v_inst_1055_, lean_object* v_inst_1056_, lean_object* v_f_1057_){
_start:
{
lean_object* v___x_1058_; lean_object* v___x_1059_; 
v___x_1058_ = lp_mathlib_LinearEquiv_refl(lean_box(0), lean_box(0), v_inst_1054_, v_inst_1055_, v_inst_1056_);
v___x_1059_ = lp_mathlib_LinearEquiv_arrowCongr___redArg(v___x_1058_, v_f_1057_);
return v___x_1059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight___redArg___boxed(lean_object* v_inst_1060_, lean_object* v_inst_1061_, lean_object* v_inst_1062_, lean_object* v_f_1063_){
_start:
{
lean_object* v_res_1064_; 
v_res_1064_ = lp_mathlib_LinearEquiv_congrRight___redArg(v_inst_1060_, v_inst_1061_, v_inst_1062_, v_f_1063_);
lean_dec(v_inst_1062_);
lean_dec_ref(v_inst_1061_);
lean_dec_ref(v_inst_1060_);
return v_res_1064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight(lean_object* v_R_1065_, lean_object* v_M_1066_, lean_object* v_M_u2082_1067_, lean_object* v_M_u2083_1068_, lean_object* v_inst_1069_, lean_object* v_inst_1070_, lean_object* v_inst_1071_, lean_object* v_inst_1072_, lean_object* v_inst_1073_, lean_object* v_inst_1074_, lean_object* v_inst_1075_, lean_object* v_f_1076_){
_start:
{
lean_object* v___x_1077_; 
v___x_1077_ = lp_mathlib_LinearEquiv_congrRight___redArg(v_inst_1069_, v_inst_1070_, v_inst_1073_, v_f_1076_);
return v___x_1077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrRight___boxed(lean_object* v_R_1078_, lean_object* v_M_1079_, lean_object* v_M_u2082_1080_, lean_object* v_M_u2083_1081_, lean_object* v_inst_1082_, lean_object* v_inst_1083_, lean_object* v_inst_1084_, lean_object* v_inst_1085_, lean_object* v_inst_1086_, lean_object* v_inst_1087_, lean_object* v_inst_1088_, lean_object* v_f_1089_){
_start:
{
lean_object* v_res_1090_; 
v_res_1090_ = lp_mathlib_LinearEquiv_congrRight(v_R_1078_, v_M_1079_, v_M_u2082_1080_, v_M_u2083_1081_, v_inst_1082_, v_inst_1083_, v_inst_1084_, v_inst_1085_, v_inst_1086_, v_inst_1087_, v_inst_1088_, v_f_1089_);
lean_dec(v_inst_1088_);
lean_dec(v_inst_1087_);
lean_dec(v_inst_1086_);
lean_dec_ref(v_inst_1085_);
lean_dec_ref(v_inst_1084_);
lean_dec_ref(v_inst_1083_);
lean_dec_ref(v_inst_1082_);
return v_res_1090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft___redArg(lean_object* v_inst_1091_, lean_object* v_inst_1092_, lean_object* v_inst_1093_, lean_object* v_e_1094_){
_start:
{
lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v_toFun_1097_; lean_object* v_invFun_1098_; lean_object* v___x_1100_; uint8_t v_isShared_1101_; uint8_t v_isSharedCheck_1105_; 
v___x_1095_ = lp_mathlib_LinearEquiv_refl(lean_box(0), lean_box(0), v_inst_1092_, v_inst_1091_, v_inst_1093_);
v___x_1096_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_1094_, v___x_1095_);
v_toFun_1097_ = lean_ctor_get(v___x_1096_, 0);
v_invFun_1098_ = lean_ctor_get(v___x_1096_, 1);
v_isSharedCheck_1105_ = !lean_is_exclusive(v___x_1096_);
if (v_isSharedCheck_1105_ == 0)
{
v___x_1100_ = v___x_1096_;
v_isShared_1101_ = v_isSharedCheck_1105_;
goto v_resetjp_1099_;
}
else
{
lean_inc(v_invFun_1098_);
lean_inc(v_toFun_1097_);
lean_dec(v___x_1096_);
v___x_1100_ = lean_box(0);
v_isShared_1101_ = v_isSharedCheck_1105_;
goto v_resetjp_1099_;
}
v_resetjp_1099_:
{
lean_object* v___x_1103_; 
if (v_isShared_1101_ == 0)
{
v___x_1103_ = v___x_1100_;
goto v_reusejp_1102_;
}
else
{
lean_object* v_reuseFailAlloc_1104_; 
v_reuseFailAlloc_1104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1104_, 0, v_toFun_1097_);
lean_ctor_set(v_reuseFailAlloc_1104_, 1, v_invFun_1098_);
v___x_1103_ = v_reuseFailAlloc_1104_;
goto v_reusejp_1102_;
}
v_reusejp_1102_:
{
return v___x_1103_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft___redArg___boxed(lean_object* v_inst_1106_, lean_object* v_inst_1107_, lean_object* v_inst_1108_, lean_object* v_e_1109_){
_start:
{
lean_object* v_res_1110_; 
v_res_1110_ = lp_mathlib_LinearEquiv_congrLeft___redArg(v_inst_1106_, v_inst_1107_, v_inst_1108_, v_e_1109_);
lean_dec(v_inst_1108_);
lean_dec_ref(v_inst_1107_);
lean_dec_ref(v_inst_1106_);
return v_res_1110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft(lean_object* v_M_1111_, lean_object* v_M_u2082_1112_, lean_object* v_M_u2083_1113_, lean_object* v_inst_1114_, lean_object* v_inst_1115_, lean_object* v_inst_1116_, lean_object* v_R_1117_, lean_object* v_S_1118_, lean_object* v_inst_1119_, lean_object* v_inst_1120_, lean_object* v_inst_1121_, lean_object* v_inst_1122_, lean_object* v_inst_1123_, lean_object* v_inst_1124_, lean_object* v_inst_1125_, lean_object* v_e_1126_){
_start:
{
lean_object* v___x_1127_; 
v___x_1127_ = lp_mathlib_LinearEquiv_congrLeft___redArg(v_inst_1114_, v_inst_1119_, v_inst_1123_, v_e_1126_);
return v___x_1127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_congrLeft___boxed(lean_object* v_M_1128_, lean_object* v_M_u2082_1129_, lean_object* v_M_u2083_1130_, lean_object* v_inst_1131_, lean_object* v_inst_1132_, lean_object* v_inst_1133_, lean_object* v_R_1134_, lean_object* v_S_1135_, lean_object* v_inst_1136_, lean_object* v_inst_1137_, lean_object* v_inst_1138_, lean_object* v_inst_1139_, lean_object* v_inst_1140_, lean_object* v_inst_1141_, lean_object* v_inst_1142_, lean_object* v_e_1143_){
_start:
{
lean_object* v_res_1144_; 
v_res_1144_ = lp_mathlib_LinearEquiv_congrLeft(v_M_1128_, v_M_u2082_1129_, v_M_u2083_1130_, v_inst_1131_, v_inst_1132_, v_inst_1133_, v_R_1134_, v_S_1135_, v_inst_1136_, v_inst_1137_, v_inst_1138_, v_inst_1139_, v_inst_1140_, v_inst_1141_, v_inst_1142_, v_e_1143_);
lean_dec(v_inst_1141_);
lean_dec(v_inst_1140_);
lean_dec(v_inst_1139_);
lean_dec(v_inst_1138_);
lean_dec_ref(v_inst_1137_);
lean_dec_ref(v_inst_1136_);
lean_dec_ref(v_inst_1133_);
lean_dec_ref(v_inst_1132_);
lean_dec_ref(v_inst_1131_);
return v_res_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero___redArg(lean_object* v_inst_1145_, lean_object* v_inst_1146_, lean_object* v_a_1147_){
_start:
{
lean_object* v___x_1148_; lean_object* v_toCommSemiring_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
v___x_1148_ = lp_mathlib_Field_toSemifield___redArg(v_inst_1145_);
v_toCommSemiring_1149_ = lean_ctor_get(v___x_1148_, 0);
lean_inc_ref(v_toCommSemiring_1149_);
v___x_1150_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v___x_1148_);
v___x_1151_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_1150_);
lean_dec_ref(v___x_1150_);
v___x_1152_ = lp_mathlib_Units_mk0___redArg(v___x_1151_, v_a_1147_);
v___x_1153_ = lp_mathlib_LinearEquiv_smulOfUnit___redArg(v_toCommSemiring_1149_, v_inst_1146_, v___x_1152_);
return v___x_1153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero___redArg___boxed(lean_object* v_inst_1154_, lean_object* v_inst_1155_, lean_object* v_a_1156_){
_start:
{
lean_object* v_res_1157_; 
v_res_1157_ = lp_mathlib_LinearEquiv_smulOfNeZero___redArg(v_inst_1154_, v_inst_1155_, v_a_1156_);
lean_dec_ref(v_inst_1154_);
return v_res_1157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero(lean_object* v_K_1158_, lean_object* v_M_1159_, lean_object* v_inst_1160_, lean_object* v_inst_1161_, lean_object* v_inst_1162_, lean_object* v_a_1163_, lean_object* v_ha_1164_){
_start:
{
lean_object* v___x_1165_; 
v___x_1165_ = lp_mathlib_LinearEquiv_smulOfNeZero___redArg(v_inst_1160_, v_inst_1162_, v_a_1163_);
return v___x_1165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_smulOfNeZero___boxed(lean_object* v_K_1166_, lean_object* v_M_1167_, lean_object* v_inst_1168_, lean_object* v_inst_1169_, lean_object* v_inst_1170_, lean_object* v_a_1171_, lean_object* v_ha_1172_){
_start:
{
lean_object* v_res_1173_; 
v_res_1173_ = lp_mathlib_LinearEquiv_smulOfNeZero(v_K_1166_, v_M_1167_, v_inst_1168_, v_inst_1169_, v_inst_1170_, v_a_1171_, v_ha_1172_);
lean_dec_ref(v_inst_1169_);
lean_dec_ref(v_inst_1168_);
return v_res_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toLinearEquiv___redArg(lean_object* v_e_1174_){
_start:
{
lean_object* v_toFun_1175_; lean_object* v_invFun_1176_; lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1183_; 
v_toFun_1175_ = lean_ctor_get(v_e_1174_, 0);
v_invFun_1176_ = lean_ctor_get(v_e_1174_, 1);
v_isSharedCheck_1183_ = !lean_is_exclusive(v_e_1174_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1178_ = v_e_1174_;
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
else
{
lean_inc(v_invFun_1176_);
lean_inc(v_toFun_1175_);
lean_dec(v_e_1174_);
v___x_1178_ = lean_box(0);
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
v_resetjp_1177_:
{
lean_object* v___x_1181_; 
if (v_isShared_1179_ == 0)
{
v___x_1181_ = v___x_1178_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_toFun_1175_);
lean_ctor_set(v_reuseFailAlloc_1182_, 1, v_invFun_1176_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
return v___x_1181_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toLinearEquiv(lean_object* v_R_1184_, lean_object* v_M_1185_, lean_object* v_M_u2082_1186_, lean_object* v_inst_1187_, lean_object* v_inst_1188_, lean_object* v_inst_1189_, lean_object* v_inst_1190_, lean_object* v_inst_1191_, lean_object* v_e_1192_, lean_object* v_h_1193_){
_start:
{
lean_object* v___x_1194_; 
v___x_1194_ = lp_mathlib_Equiv_toLinearEquiv___redArg(v_e_1192_);
return v___x_1194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toLinearEquiv___boxed(lean_object* v_R_1195_, lean_object* v_M_1196_, lean_object* v_M_u2082_1197_, lean_object* v_inst_1198_, lean_object* v_inst_1199_, lean_object* v_inst_1200_, lean_object* v_inst_1201_, lean_object* v_inst_1202_, lean_object* v_e_1203_, lean_object* v_h_1204_){
_start:
{
lean_object* v_res_1205_; 
v_res_1205_ = lp_mathlib_Equiv_toLinearEquiv(v_R_1195_, v_M_1196_, v_M_u2082_1197_, v_inst_1198_, v_inst_1199_, v_inst_1200_, v_inst_1201_, v_inst_1202_, v_e_1203_, v_h_1204_);
lean_dec(v_inst_1202_);
lean_dec_ref(v_inst_1201_);
lean_dec(v_inst_1200_);
lean_dec_ref(v_inst_1199_);
lean_dec_ref(v_inst_1198_);
return v_res_1205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft___redArg___lam__0(lean_object* v_f_1206_, lean_object* v_x_1207_, lean_object* v___y_1208_){
_start:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; 
v___x_1209_ = lean_apply_1(v_f_1206_, v___y_1208_);
v___x_1210_ = lean_apply_1(v_x_1207_, v___x_1209_);
return v___x_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft___redArg(lean_object* v_f_1211_){
_start:
{
lean_object* v___f_1212_; 
v___f_1212_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_funLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1212_, 0, v_f_1211_);
return v___f_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft(lean_object* v_R_1213_, lean_object* v_M_1214_, lean_object* v_inst_1215_, lean_object* v_inst_1216_, lean_object* v_inst_1217_, lean_object* v_m_1218_, lean_object* v_n_1219_, lean_object* v_f_1220_){
_start:
{
lean_object* v___f_1221_; 
v___f_1221_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_funLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1221_, 0, v_f_1220_);
return v___f_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_funLeft___boxed(lean_object* v_R_1222_, lean_object* v_M_1223_, lean_object* v_inst_1224_, lean_object* v_inst_1225_, lean_object* v_inst_1226_, lean_object* v_m_1227_, lean_object* v_n_1228_, lean_object* v_f_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_mathlib_LinearMap_funLeft(v_R_1222_, v_M_1223_, v_inst_1224_, v_inst_1225_, v_inst_1226_, v_m_1227_, v_n_1228_, v_f_1229_);
lean_dec(v_inst_1226_);
lean_dec_ref(v_inst_1225_);
lean_dec_ref(v_inst_1224_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft___redArg___lam__0(lean_object* v_e_1231_, lean_object* v___y_1232_){
_start:
{
lean_object* v_toFun_1233_; lean_object* v___x_1234_; 
v_toFun_1233_ = lean_ctor_get(v_e_1231_, 0);
lean_inc(v_toFun_1233_);
lean_dec_ref(v_e_1231_);
v___x_1234_ = lean_apply_1(v_toFun_1233_, v___y_1232_);
return v___x_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft___redArg(lean_object* v_e_1235_){
_start:
{
lean_object* v___f_1236_; lean_object* v___f_1237_; lean_object* v___x_1238_; lean_object* v___f_1239_; lean_object* v___f_1240_; lean_object* v___x_1241_; 
lean_inc_ref(v_e_1235_);
v___f_1236_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_funCongrLeft___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1236_, 0, v_e_1235_);
v___f_1237_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_funLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1237_, 0, v___f_1236_);
v___x_1238_ = lp_mathlib_Equiv_symm___redArg(v_e_1235_);
v___f_1239_ = lean_alloc_closure((void*)(lp_mathlib_Module_compHom_toLinearEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1239_, 0, v___x_1238_);
v___f_1240_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_funLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1240_, 0, v___f_1239_);
v___x_1241_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v___f_1237_, v___f_1240_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft(lean_object* v_R_1242_, lean_object* v_M_1243_, lean_object* v_inst_1244_, lean_object* v_inst_1245_, lean_object* v_inst_1246_, lean_object* v_m_1247_, lean_object* v_n_1248_, lean_object* v_e_1249_){
_start:
{
lean_object* v___x_1250_; 
v___x_1250_ = lp_mathlib_LinearEquiv_funCongrLeft___redArg(v_e_1249_);
return v___x_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funCongrLeft___boxed(lean_object* v_R_1251_, lean_object* v_M_1252_, lean_object* v_inst_1253_, lean_object* v_inst_1254_, lean_object* v_inst_1255_, lean_object* v_m_1256_, lean_object* v_n_1257_, lean_object* v_e_1258_){
_start:
{
lean_object* v_res_1259_; 
v_res_1259_ = lp_mathlib_LinearEquiv_funCongrLeft(v_R_1251_, v_M_1252_, v_inst_1253_, v_inst_1254_, v_inst_1255_, v_m_1256_, v_n_1257_, v_e_1258_);
lean_dec(v_inst_1255_);
lean_dec_ref(v_inst_1254_);
lean_dec_ref(v_inst_1253_);
return v_res_1259_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_sumPiEquivProdPi___closed__0(void){
_start:
{
lean_object* v___x_1260_; 
v___x_1260_ = lp_mathlib_Equiv_sumPiEquivProdPi(lean_box(0), lean_box(0), lean_box(0));
return v___x_1260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumPiEquivProdPi(lean_object* v_R_1261_, lean_object* v_inst_1262_, lean_object* v_S_1263_, lean_object* v_T_1264_, lean_object* v_A_1265_, lean_object* v_inst_1266_, lean_object* v_inst_1267_){
_start:
{
lean_object* v___x_1268_; lean_object* v_toFun_1269_; lean_object* v_invFun_1270_; lean_object* v___x_1271_; 
v___x_1268_ = lean_obj_once(&lp_mathlib_LinearEquiv_sumPiEquivProdPi___closed__0, &lp_mathlib_LinearEquiv_sumPiEquivProdPi___closed__0_once, _init_lp_mathlib_LinearEquiv_sumPiEquivProdPi___closed__0);
v_toFun_1269_ = lean_ctor_get(v___x_1268_, 0);
v_invFun_1270_ = lean_ctor_get(v___x_1268_, 1);
lean_inc(v_invFun_1270_);
lean_inc(v_toFun_1269_);
v___x_1271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1271_, 0, v_toFun_1269_);
lean_ctor_set(v___x_1271_, 1, v_invFun_1270_);
return v___x_1271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumPiEquivProdPi___boxed(lean_object* v_R_1272_, lean_object* v_inst_1273_, lean_object* v_S_1274_, lean_object* v_T_1275_, lean_object* v_A_1276_, lean_object* v_inst_1277_, lean_object* v_inst_1278_){
_start:
{
lean_object* v_res_1279_; 
v_res_1279_ = lp_mathlib_LinearEquiv_sumPiEquivProdPi(v_R_1272_, v_inst_1273_, v_S_1274_, v_T_1275_, v_A_1276_, v_inst_1277_, v_inst_1278_);
lean_dec(v_inst_1278_);
lean_dec_ref(v_inst_1277_);
lean_dec_ref(v_inst_1273_);
return v_res_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piUnique___redArg(lean_object* v_inst_1280_){
_start:
{
lean_object* v___x_1281_; lean_object* v_toFun_1282_; lean_object* v_invFun_1283_; lean_object* v___x_1285_; uint8_t v_isShared_1286_; uint8_t v_isSharedCheck_1290_; 
v___x_1281_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_1280_);
v_toFun_1282_ = lean_ctor_get(v___x_1281_, 0);
v_invFun_1283_ = lean_ctor_get(v___x_1281_, 1);
v_isSharedCheck_1290_ = !lean_is_exclusive(v___x_1281_);
if (v_isSharedCheck_1290_ == 0)
{
v___x_1285_ = v___x_1281_;
v_isShared_1286_ = v_isSharedCheck_1290_;
goto v_resetjp_1284_;
}
else
{
lean_inc(v_invFun_1283_);
lean_inc(v_toFun_1282_);
lean_dec(v___x_1281_);
v___x_1285_ = lean_box(0);
v_isShared_1286_ = v_isSharedCheck_1290_;
goto v_resetjp_1284_;
}
v_resetjp_1284_:
{
lean_object* v___x_1288_; 
if (v_isShared_1286_ == 0)
{
v___x_1288_ = v___x_1285_;
goto v_reusejp_1287_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v_toFun_1282_);
lean_ctor_set(v_reuseFailAlloc_1289_, 1, v_invFun_1283_);
v___x_1288_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1287_;
}
v_reusejp_1287_:
{
return v___x_1288_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piUnique(lean_object* v_00_u03b1_1291_, lean_object* v_inst_1292_, lean_object* v_R_1293_, lean_object* v_inst_1294_, lean_object* v_f_1295_, lean_object* v_inst_1296_, lean_object* v_inst_1297_){
_start:
{
lean_object* v___x_1298_; 
v___x_1298_ = lp_mathlib_LinearEquiv_piUnique___redArg(v_inst_1292_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piUnique___boxed(lean_object* v_00_u03b1_1299_, lean_object* v_inst_1300_, lean_object* v_R_1301_, lean_object* v_inst_1302_, lean_object* v_f_1303_, lean_object* v_inst_1304_, lean_object* v_inst_1305_){
_start:
{
lean_object* v_res_1306_; 
v_res_1306_ = lp_mathlib_LinearEquiv_piUnique(v_00_u03b1_1299_, v_inst_1300_, v_R_1301_, v_inst_1302_, v_f_1303_, v_inst_1304_, v_inst_1305_);
lean_dec(v_inst_1305_);
lean_dec_ref(v_inst_1304_);
lean_dec_ref(v_inst_1302_);
return v_res_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___redArg___lam__0(lean_object* v_toMonoid_1307_, lean_object* v_a_1308_){
_start:
{
lean_object* v___x_1309_; lean_object* v_toFun_1310_; lean_object* v_invFun_1311_; lean_object* v___x_1313_; uint8_t v_isShared_1314_; uint8_t v_isSharedCheck_1318_; 
v___x_1309_ = lp_mathlib_Units_mulLeft___redArg(v_toMonoid_1307_, v_a_1308_);
v_toFun_1310_ = lean_ctor_get(v___x_1309_, 0);
v_invFun_1311_ = lean_ctor_get(v___x_1309_, 1);
v_isSharedCheck_1318_ = !lean_is_exclusive(v___x_1309_);
if (v_isSharedCheck_1318_ == 0)
{
v___x_1313_ = v___x_1309_;
v_isShared_1314_ = v_isSharedCheck_1318_;
goto v_resetjp_1312_;
}
else
{
lean_inc(v_invFun_1311_);
lean_inc(v_toFun_1310_);
lean_dec(v___x_1309_);
v___x_1313_ = lean_box(0);
v_isShared_1314_ = v_isSharedCheck_1318_;
goto v_resetjp_1312_;
}
v_resetjp_1312_:
{
lean_object* v___x_1316_; 
if (v_isShared_1314_ == 0)
{
v___x_1316_ = v___x_1313_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_toFun_1310_);
lean_ctor_set(v_reuseFailAlloc_1317_, 1, v_invFun_1311_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___redArg___lam__0___boxed(lean_object* v_toMonoid_1319_, lean_object* v_a_1320_){
_start:
{
lean_object* v_res_1321_; 
v_res_1321_ = lp_mathlib_Units_mulLeftLinearEquiv___redArg___lam__0(v_toMonoid_1319_, v_a_1320_);
lean_dec_ref(v_toMonoid_1319_);
return v_res_1321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___redArg(lean_object* v_inst_1322_){
_start:
{
lean_object* v_toMonoid_1323_; lean_object* v___f_1324_; 
v_toMonoid_1323_ = lean_ctor_get(v_inst_1322_, 1);
lean_inc_ref(v_toMonoid_1323_);
lean_dec_ref(v_inst_1322_);
v___f_1324_ = lean_alloc_closure((void*)(lp_mathlib_Units_mulLeftLinearEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1324_, 0, v_toMonoid_1323_);
return v___f_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv(lean_object* v_R_1325_, lean_object* v_A_1326_, lean_object* v_inst_1327_, lean_object* v_inst_1328_, lean_object* v_inst_1329_, lean_object* v_inst_1330_){
_start:
{
lean_object* v___x_1331_; 
v___x_1331_ = lp_mathlib_Units_mulLeftLinearEquiv___redArg(v_inst_1328_);
return v___x_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeftLinearEquiv___boxed(lean_object* v_R_1332_, lean_object* v_A_1333_, lean_object* v_inst_1334_, lean_object* v_inst_1335_, lean_object* v_inst_1336_, lean_object* v_inst_1337_){
_start:
{
lean_object* v_res_1338_; 
v_res_1338_ = lp_mathlib_Units_mulLeftLinearEquiv(v_R_1332_, v_A_1333_, v_inst_1334_, v_inst_1335_, v_inst_1336_, v_inst_1337_);
lean_dec(v_inst_1336_);
lean_dec_ref(v_inst_1334_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv___redArg(lean_object* v_inst_1339_, lean_object* v_a_1340_){
_start:
{
lean_object* v_toMonoid_1341_; lean_object* v___x_1342_; lean_object* v_toFun_1343_; lean_object* v_invFun_1344_; lean_object* v___x_1346_; uint8_t v_isShared_1347_; uint8_t v_isSharedCheck_1351_; 
v_toMonoid_1341_ = lean_ctor_get(v_inst_1339_, 1);
v___x_1342_ = lp_mathlib_Units_mulRight___redArg(v_toMonoid_1341_, v_a_1340_);
v_toFun_1343_ = lean_ctor_get(v___x_1342_, 0);
v_invFun_1344_ = lean_ctor_get(v___x_1342_, 1);
v_isSharedCheck_1351_ = !lean_is_exclusive(v___x_1342_);
if (v_isSharedCheck_1351_ == 0)
{
v___x_1346_ = v___x_1342_;
v_isShared_1347_ = v_isSharedCheck_1351_;
goto v_resetjp_1345_;
}
else
{
lean_inc(v_invFun_1344_);
lean_inc(v_toFun_1343_);
lean_dec(v___x_1342_);
v___x_1346_ = lean_box(0);
v_isShared_1347_ = v_isSharedCheck_1351_;
goto v_resetjp_1345_;
}
v_resetjp_1345_:
{
lean_object* v___x_1349_; 
if (v_isShared_1347_ == 0)
{
v___x_1349_ = v___x_1346_;
goto v_reusejp_1348_;
}
else
{
lean_object* v_reuseFailAlloc_1350_; 
v_reuseFailAlloc_1350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1350_, 0, v_toFun_1343_);
lean_ctor_set(v_reuseFailAlloc_1350_, 1, v_invFun_1344_);
v___x_1349_ = v_reuseFailAlloc_1350_;
goto v_reusejp_1348_;
}
v_reusejp_1348_:
{
return v___x_1349_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv___redArg___boxed(lean_object* v_inst_1352_, lean_object* v_a_1353_){
_start:
{
lean_object* v_res_1354_; 
v_res_1354_ = lp_mathlib_Units_mulRightLinearEquiv___redArg(v_inst_1352_, v_a_1353_);
lean_dec_ref(v_inst_1352_);
return v_res_1354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv(lean_object* v_R_1355_, lean_object* v_A_1356_, lean_object* v_inst_1357_, lean_object* v_inst_1358_, lean_object* v_inst_1359_, lean_object* v_inst_1360_, lean_object* v_a_1361_){
_start:
{
lean_object* v___x_1362_; 
v___x_1362_ = lp_mathlib_Units_mulRightLinearEquiv___redArg(v_inst_1358_, v_a_1361_);
return v___x_1362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRightLinearEquiv___boxed(lean_object* v_R_1363_, lean_object* v_A_1364_, lean_object* v_inst_1365_, lean_object* v_inst_1366_, lean_object* v_inst_1367_, lean_object* v_inst_1368_, lean_object* v_a_1369_){
_start:
{
lean_object* v_res_1370_; 
v_res_1370_ = lp_mathlib_Units_mulRightLinearEquiv(v_R_1363_, v_A_1364_, v_inst_1365_, v_inst_1366_, v_inst_1367_, v_inst_1368_, v_a_1369_);
lean_dec(v_inst_1367_);
lean_dec_ref(v_inst_1366_);
lean_dec_ref(v_inst_1365_);
return v_res_1370_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
