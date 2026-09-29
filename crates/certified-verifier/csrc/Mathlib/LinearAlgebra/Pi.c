// Lean compiler output
// Module: Mathlib.LinearAlgebra.Pi
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Fin.Tuple public import Mathlib.Algebra.BigOperators.GroupWithZero.Action public import Mathlib.Algebra.BigOperators.Pi public import Mathlib.Algebra.Module.Prod public import Mathlib.Algebra.Module.Submodule.Ker public import Mathlib.Algebra.Module.Submodule.Range public import Mathlib.Algebra.Module.Equiv.Basic public import Mathlib.Logic.Equiv.Fin.Basic public import Mathlib.LinearAlgebra.Prod public import Mathlib.Data.Fintype.Option
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
lean_object* lp_mathlib_Fin_consEquiv___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_prod___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg(lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_ringLmapEquivSelf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object*);
lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Function_update___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_piFinTwoEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_piCurry(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piOptionEquivProd(lean_object*, lean_object*);
lean_object* lp_mathlib_finTwoArrowEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_const___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_const___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_const___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_const___closed__0_value;
static const lean_closure_object lp_mathlib_LinearMap_const___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_pi___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_const___closed__0_value)} };
static const lean_object* lp_mathlib_LinearMap_const___closed__1 = (const lean_object*)&lp_mathlib_LinearMap_const___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_proj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_proj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_proj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv_linearMapPi___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_linearMapPi___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_linearMapPi___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_linearMapPi___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SMulMemClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_pi___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__1 = (const lean_object*)&lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_diag___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_diag___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_diag___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_piCurry___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_piCurry___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_piOptionEquivProd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_piOptionEquivProd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piOptionEquivProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piOptionEquivProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumArrowLequivProdArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_piFinTwo___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_piFinTwo___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piFinTwo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piFinTwo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_finTwoArrow___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_finTwoArrow___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_finTwoArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_finTwoArrow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_vecEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_vecEmpty___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_vecEmpty___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_vecEmpty___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_vecEmpty_u2082___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_vecEmpty_u2082___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_vecEmpty_u2082___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi___redArg___lam__0(lean_object* v_f_1_, lean_object* v_c_2_, lean_object* v_i_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_f_1_, v_i_3_, v_c_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi___redArg(lean_object* v_f_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_f_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi(lean_object* v_R_7_, lean_object* v_M_u2082_8_, lean_object* v_00_u03b9_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_00_u03c6_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_17_, 0, v_f_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_pi___boxed(lean_object* v_R_18_, lean_object* v_M_u2082_19_, lean_object* v_00_u03b9_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_00_u03c6_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_f_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_LinearMap_pi(v_R_18_, v_M_u2082_19_, v_00_u03b9_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_00_u03c6_24_, v_inst_25_, v_inst_26_, v_f_27_);
lean_dec(v_inst_26_);
lean_dec_ref(v_inst_25_);
lean_dec(v_inst_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const___lam__0(lean_object* v_x_29_, lean_object* v___y_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_LinearMap_id___lam__0(v___y_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const___lam__0___boxed(lean_object* v_x_32_, lean_object* v___y_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_LinearMap_const___lam__0(v_x_32_, v___y_33_);
lean_dec(v___y_33_);
lean_dec(v_x_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const(lean_object* v_R_38_, lean_object* v_M_u2082_39_, lean_object* v_00_u03b9_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; 
v___f_44_ = ((lean_object*)(lp_mathlib_LinearMap_const___closed__1));
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_const___boxed(lean_object* v_R_45_, lean_object* v_M_u2082_46_, lean_object* v_00_u03b9_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_LinearMap_const(v_R_45_, v_M_u2082_46_, v_00_u03b9_47_, v_inst_48_, v_inst_49_, v_inst_50_);
lean_dec(v_inst_50_);
lean_dec_ref(v_inst_49_);
lean_dec_ref(v_inst_48_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_proj___redArg(lean_object* v_i_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_53_, 0, lean_box(0));
lean_closure_set(v___x_53_, 1, lean_box(0));
lean_closure_set(v___x_53_, 2, v_i_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_proj(lean_object* v_R_54_, lean_object* v_00_u03b9_55_, lean_object* v_inst_56_, lean_object* v_00_u03c6_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_i_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_61_, 0, lean_box(0));
lean_closure_set(v___x_61_, 1, lean_box(0));
lean_closure_set(v___x_61_, 2, v_i_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_proj___boxed(lean_object* v_R_62_, lean_object* v_00_u03b9_63_, lean_object* v_inst_64_, lean_object* v_00_u03c6_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_i_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_LinearMap_proj(v_R_62_, v_00_u03b9_63_, v_inst_64_, v_00_u03c6_65_, v_inst_66_, v_inst_67_, v_i_68_);
lean_dec(v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_64_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi___redArg___lam__0(lean_object* v_f_70_, lean_object* v_i_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_73_, 0, lean_box(0));
lean_closure_set(v___x_73_, 1, lean_box(0));
lean_closure_set(v___x_73_, 2, v_i_71_);
v___x_74_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v_f_70_, v___x_73_, v___y_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi___redArg(lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___f_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___f_81_ = ((lean_object*)(lp_mathlib_LinearEquiv_linearMapPi___redArg___closed__0));
v___x_82_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_pi___boxed), 10, 9);
lean_closure_set(v___x_82_, 0, lean_box(0));
lean_closure_set(v___x_82_, 1, lean_box(0));
lean_closure_set(v___x_82_, 2, lean_box(0));
lean_closure_set(v___x_82_, 3, v_inst_76_);
lean_closure_set(v___x_82_, 4, v_inst_77_);
lean_closure_set(v___x_82_, 5, v_inst_78_);
lean_closure_set(v___x_82_, 6, lean_box(0));
lean_closure_set(v___x_82_, 7, v_inst_79_);
lean_closure_set(v___x_82_, 8, v_inst_80_);
v___x_83_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v___f_81_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi(lean_object* v_R_84_, lean_object* v_M_u2082_85_, lean_object* v_00_u03b9_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_00_u03c6_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_S_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_mathlib_LinearEquiv_linearMapPi___redArg(v_inst_87_, v_inst_88_, v_inst_89_, v_inst_91_, v_inst_92_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_linearMapPi___boxed(lean_object* v_R_98_, lean_object* v_M_u2082_99_, lean_object* v_00_u03b9_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_00_u03c6_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_S_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_LinearEquiv_linearMapPi(v_R_98_, v_M_u2082_99_, v_00_u03b9_100_, v_inst_101_, v_inst_102_, v_inst_103_, v_00_u03c6_104_, v_inst_105_, v_inst_106_, v_S_107_, v_inst_108_, v_inst_109_, v_inst_110_);
lean_dec(v_inst_109_);
lean_dec_ref(v_inst_108_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap___redArg___lam__0(lean_object* v_f_112_, lean_object* v_i_113_, lean_object* v___y_114_){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
lean_inc(v_i_113_);
v___x_115_ = lean_apply_1(v_f_112_, v_i_113_);
v___x_116_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_116_, 0, lean_box(0));
lean_closure_set(v___x_116_, 1, lean_box(0));
lean_closure_set(v___x_116_, 2, v_i_113_);
v___x_117_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___x_116_, v___x_115_, v___y_114_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap___redArg(lean_object* v_f_118_){
_start:
{
lean_object* v___f_119_; lean_object* v___f_120_; 
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_piMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_119_, 0, v_f_118_);
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_120_, 0, v___f_119_);
return v___f_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap(lean_object* v_R_121_, lean_object* v_00_u03b9_122_, lean_object* v_inst_123_, lean_object* v_00_u03c6_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_00_u03c8_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_f_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_LinearMap_piMap___redArg(v_f_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_piMap___boxed(lean_object* v_R_132_, lean_object* v_00_u03b9_133_, lean_object* v_inst_134_, lean_object* v_00_u03c6_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_00_u03c8_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_f_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_LinearMap_piMap(v_R_132_, v_00_u03b9_133_, v_inst_134_, v_00_u03c6_135_, v_inst_136_, v_inst_137_, v_00_u03c8_138_, v_inst_139_, v_inst_140_, v_f_141_);
lean_dec(v_inst_140_);
lean_dec_ref(v_inst_139_);
lean_dec(v_inst_137_);
lean_dec_ref(v_inst_136_);
lean_dec_ref(v_inst_134_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft___redArg___lam__0(lean_object* v_f_143_, lean_object* v_h_144_, lean_object* v___y_145_){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = lean_apply_1(v_h_144_, v___y_145_);
v___x_147_ = lean_apply_1(v_f_143_, v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft___redArg(lean_object* v_f_148_){
_start:
{
lean_object* v___f_149_; 
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_149_, 0, v_f_148_);
return v___f_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft(lean_object* v_R_150_, lean_object* v_M_u2082_151_, lean_object* v_M_u2083_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_f_158_, lean_object* v_I_159_){
_start:
{
lean_object* v___f_160_; 
v___f_160_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_160_, 0, v_f_158_);
return v___f_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_compLeft___boxed(lean_object* v_R_161_, lean_object* v_M_u2082_162_, lean_object* v_M_u2083_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_f_169_, lean_object* v_I_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_LinearMap_compLeft(v_R_161_, v_M_u2082_162_, v_M_u2083_163_, v_inst_164_, v_inst_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_f_169_, v_I_170_);
lean_dec(v_inst_168_);
lean_dec_ref(v_inst_167_);
lean_dec(v_inst_166_);
lean_dec_ref(v_inst_165_);
lean_dec_ref(v_inst_164_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single___redArg___lam__0(lean_object* v_inst_172_, lean_object* v_i_173_){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v_toZero_177_; 
v___x_174_ = lean_apply_1(v_inst_172_, v_i_173_);
v___x_175_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_174_);
lean_dec_ref(v___x_174_);
v___x_176_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_175_);
v_toZero_177_ = lean_ctor_get(v___x_176_, 0);
lean_inc(v_toZero_177_);
lean_dec_ref(v___x_176_);
return v_toZero_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single___redArg(lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_i_180_){
_start:
{
lean_object* v___f_181_; lean_object* v___x_182_; 
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_single___redArg___lam__0), 2, 1);
lean_closure_set(v___f_181_, 0, v_inst_178_);
v___x_182_ = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 5);
lean_closure_set(v___x_182_, 0, lean_box(0));
lean_closure_set(v___x_182_, 1, lean_box(0));
lean_closure_set(v___x_182_, 2, v___f_181_);
lean_closure_set(v___x_182_, 3, v_inst_179_);
lean_closure_set(v___x_182_, 4, v_i_180_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single(lean_object* v_R_183_, lean_object* v_00_u03b9_184_, lean_object* v_inst_185_, lean_object* v_00_u03c6_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_i_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_LinearMap_single___redArg(v_inst_187_, v_inst_189_, v_i_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_single___boxed(lean_object* v_R_192_, lean_object* v_00_u03b9_193_, lean_object* v_inst_194_, lean_object* v_00_u03c6_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_i_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_LinearMap_single(v_R_192_, v_00_u03b9_193_, v_inst_194_, v_00_u03c6_195_, v_inst_196_, v_inst_197_, v_inst_198_, v_i_199_);
lean_dec(v_inst_197_);
lean_dec_ref(v_inst_194_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg___lam__0(lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_f_203_, lean_object* v_i_204_, lean_object* v___y_205_){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = lp_mathlib_LinearMap_single___redArg(v_inst_201_, v_inst_202_, v_i_204_);
v___x_207_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___x_206_, v_f_203_, v___y_205_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg___lam__2(lean_object* v___x_208_, lean_object* v_inst_209_, lean_object* v_f_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___f_212_; lean_object* v___x_109__overap_213_; lean_object* v___x_214_; 
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_piMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_212_, 0, v_f_210_);
v___x_109__overap_213_ = lp_mathlib_Finset_sum___redArg(v___x_208_, v_inst_209_, v___f_212_);
v___x_214_ = lean_apply_1(v___x_109__overap_213_, v___y_211_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg___lam__2___boxed(lean_object* v___x_215_, lean_object* v_inst_216_, lean_object* v_f_217_, lean_object* v___y_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_LinearMap_lsum___redArg___lam__2(v___x_215_, v_inst_216_, v_f_217_, v___y_218_);
lean_dec_ref(v___x_215_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___redArg(lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___f_224_; lean_object* v___x_225_; lean_object* v___f_226_; lean_object* v___x_227_; 
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_lsum___redArg___lam__0), 5, 2);
lean_closure_set(v___f_224_, 0, v_inst_220_);
lean_closure_set(v___f_224_, 1, v_inst_221_);
v___x_225_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_222_);
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_lsum___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_226_, 0, v___x_225_);
lean_closure_set(v___f_226_, 1, v_inst_223_);
v___x_227_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_227_, 0, v___f_226_);
lean_ctor_set(v___x_227_, 1, v___f_224_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum(lean_object* v_R_228_, lean_object* v_M_229_, lean_object* v_00_u03b9_230_, lean_object* v_inst_231_, lean_object* v_00_u03c6_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_S_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_LinearMap_lsum___redArg(v_inst_233_, v_inst_235_, v_inst_237_, v_inst_239_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_lsum___boxed(lean_object* v_R_244_, lean_object* v_M_245_, lean_object* v_00_u03b9_246_, lean_object* v_inst_247_, lean_object* v_00_u03c6_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_S_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_LinearMap_lsum(v_R_244_, v_M_245_, v_00_u03b9_246_, v_inst_247_, v_00_u03c6_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_S_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_, v_inst_257_, v_inst_258_);
lean_dec(v_inst_257_);
lean_dec_ref(v_inst_256_);
lean_dec(v_inst_254_);
lean_dec(v_inst_250_);
lean_dec_ref(v_inst_247_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__0(lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_i_262_, lean_object* v___y_263_){
_start:
{
lean_object* v___x_264_; uint8_t v___x_265_; 
lean_inc(v_i_262_);
v___x_264_ = lean_apply_1(v_inst_260_, v_i_262_);
v___x_265_ = lean_unbox(v___x_264_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v_toZero_269_; 
lean_dec(v___y_263_);
v___x_266_ = lean_apply_1(v_inst_261_, v_i_262_);
v___x_267_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_266_);
lean_dec_ref(v___x_266_);
v___x_268_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_267_);
v_toZero_269_ = lean_ctor_get(v___x_268_, 0);
lean_inc(v_toZero_269_);
lean_dec_ref(v___x_268_);
return v_toZero_269_;
}
else
{
lean_object* v___x_270_; 
lean_dec_ref(v_inst_261_);
v___x_270_ = lean_apply_1(v___y_263_, v_i_262_);
return v___x_270_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1(lean_object* v_i_272_, lean_object* v___y_273_){
_start:
{
lean_object* v___x_274_; lean_object* v___f_275_; lean_object* v___x_276_; 
v___x_274_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_274_, 0, lean_box(0));
lean_closure_set(v___x_274_, 1, lean_box(0));
lean_closure_set(v___x_274_, 2, v_i_272_);
v___f_275_ = ((lean_object*)(lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__1___closed__0));
v___x_276_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___f_275_, v___x_274_, v___y_273_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___redArg(lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___f_282_; lean_object* v___f_283_; lean_object* v___f_284_; lean_object* v___f_285_; lean_object* v___x_286_; 
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___lam__0), 4, 2);
lean_closure_set(v___f_282_, 0, v_inst_281_);
lean_closure_set(v___f_282_, 1, v_inst_280_);
v___f_283_ = ((lean_object*)(lp_mathlib_LinearMap_iInfKerProjEquiv___redArg___closed__1));
v___f_284_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_284_, 0, v___f_282_);
v___f_285_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_285_, 0, v___f_284_);
v___x_286_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v___f_283_, v___f_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv(lean_object* v_R_287_, lean_object* v_00_u03b9_288_, lean_object* v_inst_289_, lean_object* v_00_u03c6_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_I_293_, lean_object* v_J_294_, lean_object* v_inst_295_, lean_object* v_hd_296_, lean_object* v_hu_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lp_mathlib_LinearMap_iInfKerProjEquiv___redArg(v_inst_291_, v_inst_295_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_iInfKerProjEquiv___boxed(lean_object* v_R_299_, lean_object* v_00_u03b9_300_, lean_object* v_inst_301_, lean_object* v_00_u03c6_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_I_305_, lean_object* v_J_306_, lean_object* v_inst_307_, lean_object* v_hd_308_, lean_object* v_hu_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_LinearMap_iInfKerProjEquiv(v_R_299_, v_00_u03b9_300_, v_inst_301_, v_00_u03c6_302_, v_inst_303_, v_inst_304_, v_I_305_, v_J_306_, v_inst_307_, v_hd_308_, v_hu_309_);
lean_dec(v_inst_304_);
lean_dec_ref(v_inst_301_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___redArg___lam__0(lean_object* v_inst_311_, lean_object* v_x_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v_toZero_317_; 
v___x_314_ = lean_apply_1(v_inst_311_, v_x_312_);
v___x_315_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_314_);
lean_dec_ref(v___x_314_);
v___x_316_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_315_);
v_toZero_317_ = lean_ctor_get(v___x_316_, 0);
lean_inc(v_toZero_317_);
lean_dec_ref(v___x_316_);
return v_toZero_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___redArg___lam__0___boxed(lean_object* v_inst_318_, lean_object* v_x_319_, lean_object* v___y_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_LinearMap_diag___redArg___lam__0(v_inst_318_, v_x_319_, v___y_320_);
lean_dec(v___y_320_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___redArg(lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_i_325_, lean_object* v_j_326_){
_start:
{
lean_object* v___f_327_; lean_object* v___f_328_; lean_object* v___x_329_; 
v___f_327_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_diag___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_327_, 0, v_inst_323_);
v___f_328_ = ((lean_object*)(lp_mathlib_LinearMap_diag___redArg___closed__0));
v___x_329_ = lp_mathlib_Function_update___redArg(v_inst_324_, v___f_327_, v_i_325_, v___f_328_, v_j_326_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag(lean_object* v_R_330_, lean_object* v_00_u03b9_331_, lean_object* v_inst_332_, lean_object* v_00_u03c6_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_i_337_, lean_object* v_j_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_LinearMap_diag___redArg(v_inst_334_, v_inst_336_, v_i_337_, v_j_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_diag___boxed(lean_object* v_R_340_, lean_object* v_00_u03b9_341_, lean_object* v_inst_342_, lean_object* v_00_u03c6_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_i_347_, lean_object* v_j_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_LinearMap_diag(v_R_340_, v_00_u03b9_341_, v_inst_342_, v_00_u03c6_343_, v_inst_344_, v_inst_345_, v_inst_346_, v_i_347_, v_j_348_);
lean_dec(v_inst_345_);
lean_dec_ref(v_inst_342_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pi(lean_object* v_R_350_, lean_object* v_00_u03b9_351_, lean_object* v_inst_352_, lean_object* v_00_u03c6_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_I_356_, lean_object* v_p_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lean_box(0);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_pi___boxed(lean_object* v_R_359_, lean_object* v_00_u03b9_360_, lean_object* v_inst_361_, lean_object* v_00_u03c6_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_I_365_, lean_object* v_p_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_Submodule_pi(v_R_359_, v_00_u03b9_360_, v_inst_361_, v_00_u03c6_362_, v_inst_363_, v_inst_364_, v_I_365_, v_p_366_);
lean_dec_ref(v_p_366_);
lean_dec(v_inst_364_);
lean_dec_ref(v_inst_363_);
lean_dec_ref(v_inst_361_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg___lam__0(lean_object* v_e_368_, lean_object* v_f_369_, lean_object* v_i_370_){
_start:
{
lean_object* v___x_371_; lean_object* v_toLinearMap_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
lean_inc(v_i_370_);
v___x_371_ = lean_apply_1(v_e_368_, v_i_370_);
v_toLinearMap_372_ = lean_ctor_get(v___x_371_, 0);
lean_inc(v_toLinearMap_372_);
lean_dec_ref(v___x_371_);
v___x_373_ = lean_apply_1(v_f_369_, v_i_370_);
v___x_374_ = lean_apply_1(v_toLinearMap_372_, v___x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg___lam__1(lean_object* v_e_375_, lean_object* v_f_376_, lean_object* v_i_377_){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v_toLinearMap_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
lean_inc(v_i_377_);
v___x_378_ = lean_apply_1(v_e_375_, v_i_377_);
v___x_379_ = lp_mathlib_LinearEquiv_symm___redArg(v___x_378_);
v_toLinearMap_380_ = lean_ctor_get(v___x_379_, 0);
lean_inc(v_toLinearMap_380_);
lean_dec_ref(v___x_379_);
v___x_381_ = lean_apply_1(v_f_376_, v_i_377_);
v___x_382_ = lean_apply_1(v_toLinearMap_380_, v___x_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg(lean_object* v_e_383_){
_start:
{
lean_object* v___f_384_; lean_object* v___f_385_; lean_object* v___x_386_; 
lean_inc_ref(v_e_383_);
v___f_384_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_384_, 0, v_e_383_);
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_piCongrRight___redArg___lam__1), 3, 1);
lean_closure_set(v___f_385_, 0, v_e_383_);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v___f_384_);
lean_ctor_set(v___x_386_, 1, v___f_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight(lean_object* v_R_387_, lean_object* v_00_u03b9_388_, lean_object* v_inst_389_, lean_object* v_00_u03c6_390_, lean_object* v_00_u03c8_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_e_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lp_mathlib_LinearEquiv_piCongrRight___redArg(v_e_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrRight___boxed(lean_object* v_R_398_, lean_object* v_00_u03b9_399_, lean_object* v_inst_400_, lean_object* v_00_u03c6_401_, lean_object* v_00_u03c8_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_e_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_LinearEquiv_piCongrRight(v_R_398_, v_00_u03b9_399_, v_inst_400_, v_00_u03c6_401_, v_00_u03c8_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_e_407_);
lean_dec(v_inst_406_);
lean_dec_ref(v_inst_405_);
lean_dec(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_inst_400_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft_x27___redArg(lean_object* v_e_409_){
_start:
{
lean_object* v___x_410_; lean_object* v_toFun_411_; lean_object* v_invFun_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_419_; 
v___x_410_ = lp_mathlib_Equiv_piCongrLeft_x27___redArg(v_e_409_);
v_toFun_411_ = lean_ctor_get(v___x_410_, 0);
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
lean_inc(v_toFun_411_);
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
lean_ctor_set(v_reuseFailAlloc_418_, 0, v_toFun_411_);
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
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft_x27(lean_object* v_R_420_, lean_object* v_00_u03b9_421_, lean_object* v_00_u03b9_x27_422_, lean_object* v_inst_423_, lean_object* v_00_u03c6_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_e_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lp_mathlib_LinearEquiv_piCongrLeft_x27___redArg(v_e_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft_x27___boxed(lean_object* v_R_429_, lean_object* v_00_u03b9_430_, lean_object* v_00_u03b9_x27_431_, lean_object* v_inst_432_, lean_object* v_00_u03c6_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_e_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_mathlib_LinearEquiv_piCongrLeft_x27(v_R_429_, v_00_u03b9_430_, v_00_u03b9_x27_431_, v_inst_432_, v_00_u03c6_433_, v_inst_434_, v_inst_435_, v_e_436_);
lean_dec(v_inst_435_);
lean_dec_ref(v_inst_434_);
lean_dec_ref(v_inst_432_);
return v_res_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft___redArg(lean_object* v_e_438_){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_439_ = lp_mathlib_Equiv_symm___redArg(v_e_438_);
v___x_440_ = lp_mathlib_LinearEquiv_piCongrLeft_x27___redArg(v___x_439_);
v___x_441_ = lp_mathlib_LinearEquiv_symm___redArg(v___x_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft(lean_object* v_R_442_, lean_object* v_00_u03b9_443_, lean_object* v_00_u03b9_x27_444_, lean_object* v_inst_445_, lean_object* v_00_u03c6_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_e_449_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lp_mathlib_LinearEquiv_piCongrLeft___redArg(v_e_449_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCongrLeft___boxed(lean_object* v_R_451_, lean_object* v_00_u03b9_452_, lean_object* v_00_u03b9_x27_453_, lean_object* v_inst_454_, lean_object* v_00_u03c6_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_e_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_mathlib_LinearEquiv_piCongrLeft(v_R_451_, v_00_u03b9_452_, v_00_u03b9_x27_453_, v_inst_454_, v_00_u03c6_455_, v_inst_456_, v_inst_457_, v_e_458_);
lean_dec(v_inst_457_);
lean_dec_ref(v_inst_456_);
lean_dec_ref(v_inst_454_);
return v_res_459_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_piCurry___closed__0(void){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_Equiv_piCurry(lean_box(0), lean_box(0), lean_box(0));
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCurry(lean_object* v_R_461_, lean_object* v_inst_462_, lean_object* v_00_u03b9_463_, lean_object* v_00_u03ba_464_, lean_object* v_00_u03b1_465_, lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v___x_468_; lean_object* v_toFun_469_; lean_object* v_invFun_470_; lean_object* v___x_471_; 
v___x_468_ = lean_obj_once(&lp_mathlib_LinearEquiv_piCurry___closed__0, &lp_mathlib_LinearEquiv_piCurry___closed__0_once, _init_lp_mathlib_LinearEquiv_piCurry___closed__0);
v_toFun_469_ = lean_ctor_get(v___x_468_, 0);
v_invFun_470_ = lean_ctor_get(v___x_468_, 1);
lean_inc(v_invFun_470_);
lean_inc(v_toFun_469_);
v___x_471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_471_, 0, v_toFun_469_);
lean_ctor_set(v___x_471_, 1, v_invFun_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piCurry___boxed(lean_object* v_R_472_, lean_object* v_inst_473_, lean_object* v_00_u03b9_474_, lean_object* v_00_u03ba_475_, lean_object* v_00_u03b1_476_, lean_object* v_inst_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_LinearEquiv_piCurry(v_R_472_, v_inst_473_, v_00_u03b9_474_, v_00_u03ba_475_, v_00_u03b1_476_, v_inst_477_, v_inst_478_);
lean_dec(v_inst_478_);
lean_dec_ref(v_inst_477_);
lean_dec_ref(v_inst_473_);
return v_res_479_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_piOptionEquivProd___closed__0(void){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_mathlib_Equiv_piOptionEquivProd(lean_box(0), lean_box(0));
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piOptionEquivProd(lean_object* v_R_481_, lean_object* v_inst_482_, lean_object* v_00_u03b9_483_, lean_object* v_M_484_, lean_object* v_inst_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v___x_487_; lean_object* v_toFun_488_; lean_object* v_invFun_489_; lean_object* v___x_490_; 
v___x_487_ = lean_obj_once(&lp_mathlib_LinearEquiv_piOptionEquivProd___closed__0, &lp_mathlib_LinearEquiv_piOptionEquivProd___closed__0_once, _init_lp_mathlib_LinearEquiv_piOptionEquivProd___closed__0);
v_toFun_488_ = lean_ctor_get(v___x_487_, 0);
v_invFun_489_ = lean_ctor_get(v___x_487_, 1);
lean_inc(v_invFun_489_);
lean_inc(v_toFun_488_);
v___x_490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_490_, 0, v_toFun_488_);
lean_ctor_set(v___x_490_, 1, v_invFun_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piOptionEquivProd___boxed(lean_object* v_R_491_, lean_object* v_inst_492_, lean_object* v_00_u03b9_493_, lean_object* v_M_494_, lean_object* v_inst_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_LinearEquiv_piOptionEquivProd(v_R_491_, v_inst_492_, v_00_u03b9_493_, v_M_494_, v_inst_495_, v_inst_496_);
lean_dec(v_inst_496_);
lean_dec_ref(v_inst_495_);
lean_dec_ref(v_inst_492_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__0(lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_x_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_mathlib_LinearMap_ringLmapEquivSelf___redArg(v_inst_498_, v_inst_499_, v_inst_500_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__0___boxed(lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_x_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_LinearEquiv_piRing___redArg___lam__0(v_inst_503_, v_inst_504_, v_inst_505_, v_x_506_);
lean_dec(v_x_506_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__1(lean_object* v_toAddCommMonoid_508_, lean_object* v_i_509_){
_start:
{
lean_inc_ref(v_toAddCommMonoid_508_);
return v_toAddCommMonoid_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg___lam__1___boxed(lean_object* v_toAddCommMonoid_510_, lean_object* v_i_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_LinearEquiv_piRing___redArg___lam__1(v_toAddCommMonoid_510_, v_i_511_);
lean_dec(v_i_511_);
lean_dec_ref(v_toAddCommMonoid_510_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___redArg(lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v_toAddCommMonoid_518_; lean_object* v___f_519_; lean_object* v___f_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; 
v_toAddCommMonoid_518_ = lean_ctor_get(v_inst_513_, 0);
lean_inc_ref(v_toAddCommMonoid_518_);
lean_inc_ref(v_inst_516_);
v___f_519_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_piRing___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_519_, 0, v_inst_513_);
lean_closure_set(v___f_519_, 1, v_inst_516_);
lean_closure_set(v___f_519_, 2, v_inst_517_);
v___f_520_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_piRing___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_520_, 0, v_toAddCommMonoid_518_);
v___x_521_ = lp_mathlib_LinearMap_lsum___redArg(v___f_520_, v_inst_515_, v_inst_516_, v_inst_514_);
v___x_522_ = lp_mathlib_LinearEquiv_symm___redArg(v___x_521_);
v___x_523_ = lp_mathlib_LinearEquiv_piCongrRight___redArg(v___f_519_);
v___x_524_ = lp_mathlib_LinearEquiv_trans___redArg(v___x_522_, v___x_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing(lean_object* v_R_525_, lean_object* v_M_526_, lean_object* v_00_u03b9_527_, lean_object* v_inst_528_, lean_object* v_S_529_, lean_object* v_inst_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib_LinearEquiv_piRing___redArg(v_inst_528_, v_inst_530_, v_inst_531_, v_inst_533_, v_inst_534_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piRing___boxed(lean_object* v_R_538_, lean_object* v_M_539_, lean_object* v_00_u03b9_540_, lean_object* v_inst_541_, lean_object* v_S_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_){
_start:
{
lean_object* v_res_550_; 
v_res_550_ = lp_mathlib_LinearEquiv_piRing(v_R_538_, v_M_539_, v_00_u03b9_540_, v_inst_541_, v_S_542_, v_inst_543_, v_inst_544_, v_inst_545_, v_inst_546_, v_inst_547_, v_inst_548_, v_inst_549_);
lean_dec(v_inst_548_);
lean_dec_ref(v_inst_545_);
return v_res_550_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___closed__0(void){
_start:
{
lean_object* v___x_551_; 
v___x_551_ = lp_mathlib_Equiv_sumArrowEquivProdArrow(lean_box(0), lean_box(0), lean_box(0));
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumArrowLequivProdArrow(lean_object* v_00_u03b1_552_, lean_object* v_00_u03b2_553_, lean_object* v_R_554_, lean_object* v_M_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_inst_558_){
_start:
{
lean_object* v___x_559_; lean_object* v_toFun_560_; lean_object* v_invFun_561_; lean_object* v___x_562_; 
v___x_559_ = lean_obj_once(&lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___closed__0, &lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___closed__0_once, _init_lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___closed__0);
v_toFun_560_ = lean_ctor_get(v___x_559_, 0);
v_invFun_561_ = lean_ctor_get(v___x_559_, 1);
lean_inc(v_invFun_561_);
lean_inc(v_toFun_560_);
v___x_562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_562_, 0, v_toFun_560_);
lean_ctor_set(v___x_562_, 1, v_invFun_561_);
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_sumArrowLequivProdArrow___boxed(lean_object* v_00_u03b1_563_, lean_object* v_00_u03b2_564_, lean_object* v_R_565_, lean_object* v_M_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_inst_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_LinearEquiv_sumArrowLequivProdArrow(v_00_u03b1_563_, v_00_u03b2_564_, v_R_565_, v_M_566_, v_inst_567_, v_inst_568_, v_inst_569_);
lean_dec(v_inst_569_);
lean_dec_ref(v_inst_568_);
lean_dec_ref(v_inst_567_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funUnique___redArg(lean_object* v_inst_571_){
_start:
{
lean_object* v___x_572_; lean_object* v_toFun_573_; lean_object* v_invFun_574_; lean_object* v___x_576_; uint8_t v_isShared_577_; uint8_t v_isSharedCheck_581_; 
v___x_572_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_571_);
v_toFun_573_ = lean_ctor_get(v___x_572_, 0);
v_invFun_574_ = lean_ctor_get(v___x_572_, 1);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_572_);
if (v_isSharedCheck_581_ == 0)
{
v___x_576_ = v___x_572_;
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
else
{
lean_inc(v_invFun_574_);
lean_inc(v_toFun_573_);
lean_dec(v___x_572_);
v___x_576_ = lean_box(0);
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
v_resetjp_575_:
{
lean_object* v___x_579_; 
if (v_isShared_577_ == 0)
{
v___x_579_ = v___x_576_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v_toFun_573_);
lean_ctor_set(v_reuseFailAlloc_580_, 1, v_invFun_574_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funUnique(lean_object* v_00_u03b9_582_, lean_object* v_R_583_, lean_object* v_M_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_inst_587_, lean_object* v_inst_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_mathlib_LinearEquiv_funUnique___redArg(v_inst_585_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_funUnique___boxed(lean_object* v_00_u03b9_590_, lean_object* v_R_591_, lean_object* v_M_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_inst_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_LinearEquiv_funUnique(v_00_u03b9_590_, v_R_591_, v_M_592_, v_inst_593_, v_inst_594_, v_inst_595_, v_inst_596_);
lean_dec(v_inst_596_);
lean_dec_ref(v_inst_595_);
lean_dec_ref(v_inst_594_);
return v_res_597_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_piFinTwo___closed__0(void){
_start:
{
lean_object* v___x_598_; 
v___x_598_ = lp_mathlib_piFinTwoEquiv(lean_box(0));
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piFinTwo(lean_object* v_R_599_, lean_object* v_inst_600_, lean_object* v_M_601_, lean_object* v_inst_602_, lean_object* v_inst_603_){
_start:
{
lean_object* v___x_604_; lean_object* v_toFun_605_; lean_object* v_invFun_606_; lean_object* v___x_607_; 
v___x_604_ = lean_obj_once(&lp_mathlib_LinearEquiv_piFinTwo___closed__0, &lp_mathlib_LinearEquiv_piFinTwo___closed__0_once, _init_lp_mathlib_LinearEquiv_piFinTwo___closed__0);
v_toFun_605_ = lean_ctor_get(v___x_604_, 0);
v_invFun_606_ = lean_ctor_get(v___x_604_, 1);
lean_inc(v_invFun_606_);
lean_inc(v_toFun_605_);
v___x_607_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_607_, 0, v_toFun_605_);
lean_ctor_set(v___x_607_, 1, v_invFun_606_);
return v___x_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_piFinTwo___boxed(lean_object* v_R_608_, lean_object* v_inst_609_, lean_object* v_M_610_, lean_object* v_inst_611_, lean_object* v_inst_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_LinearEquiv_piFinTwo(v_R_608_, v_inst_609_, v_M_610_, v_inst_611_, v_inst_612_);
lean_dec(v_inst_612_);
lean_dec_ref(v_inst_611_);
lean_dec_ref(v_inst_609_);
return v_res_613_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_finTwoArrow___closed__0(void){
_start:
{
lean_object* v___x_614_; 
v___x_614_ = lp_mathlib_finTwoArrowEquiv(lean_box(0));
return v___x_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_finTwoArrow(lean_object* v_R_615_, lean_object* v_M_616_, lean_object* v_inst_617_, lean_object* v_inst_618_, lean_object* v_inst_619_){
_start:
{
lean_object* v___x_620_; lean_object* v_toFun_621_; lean_object* v_invFun_622_; lean_object* v___x_623_; 
v___x_620_ = lean_obj_once(&lp_mathlib_LinearEquiv_finTwoArrow___closed__0, &lp_mathlib_LinearEquiv_finTwoArrow___closed__0_once, _init_lp_mathlib_LinearEquiv_finTwoArrow___closed__0);
v_toFun_621_ = lean_ctor_get(v___x_620_, 0);
v_invFun_622_ = lean_ctor_get(v___x_620_, 1);
lean_inc(v_invFun_622_);
lean_inc(v_toFun_621_);
v___x_623_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_623_, 0, v_toFun_621_);
lean_ctor_set(v___x_623_, 1, v_invFun_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_finTwoArrow___boxed(lean_object* v_R_624_, lean_object* v_M_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_inst_628_){
_start:
{
lean_object* v_res_629_; 
v_res_629_ = lp_mathlib_LinearEquiv_finTwoArrow(v_R_624_, v_M_625_, v_inst_626_, v_inst_627_, v_inst_628_);
lean_dec(v_inst_628_);
lean_dec_ref(v_inst_627_);
lean_dec_ref(v_inst_626_);
return v_res_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consLinearEquiv___redArg(lean_object* v_n_630_){
_start:
{
lean_object* v___x_631_; lean_object* v_toFun_632_; lean_object* v_invFun_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_640_; 
v___x_631_ = lp_mathlib_Fin_consEquiv___redArg(v_n_630_);
v_toFun_632_ = lean_ctor_get(v___x_631_, 0);
v_invFun_633_ = lean_ctor_get(v___x_631_, 1);
v_isSharedCheck_640_ = !lean_is_exclusive(v___x_631_);
if (v_isSharedCheck_640_ == 0)
{
v___x_635_ = v___x_631_;
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_invFun_633_);
lean_inc(v_toFun_632_);
lean_dec(v___x_631_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_636_ == 0)
{
v___x_638_ = v___x_635_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_toFun_632_);
lean_ctor_set(v_reuseFailAlloc_639_, 1, v_invFun_633_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consLinearEquiv(lean_object* v_R_641_, lean_object* v_n_642_, lean_object* v_M_643_, lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_inst_646_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lp_mathlib_Fin_consLinearEquiv___redArg(v_n_642_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consLinearEquiv___boxed(lean_object* v_R_648_, lean_object* v_n_649_, lean_object* v_M_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Fin_consLinearEquiv(v_R_648_, v_n_649_, v_M_650_, v_inst_651_, v_inst_652_, v_inst_653_);
lean_dec(v_inst_653_);
lean_dec_ref(v_inst_652_);
lean_dec_ref(v_inst_651_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty___lam__0(lean_object* v_x_655_, lean_object* v___y_656_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty___lam__0___boxed(lean_object* v_x_657_, lean_object* v___y_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_mathlib_LinearMap_vecEmpty___lam__0(v_x_657_, v___y_658_);
lean_dec(v___y_658_);
lean_dec(v_x_657_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty(lean_object* v_R_661_, lean_object* v_M_662_, lean_object* v_M_u2083_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_inst_668_){
_start:
{
lean_object* v___f_669_; 
v___f_669_ = ((lean_object*)(lp_mathlib_LinearMap_vecEmpty___closed__0));
return v___f_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty___boxed(lean_object* v_R_670_, lean_object* v_M_671_, lean_object* v_M_u2083_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_inst_677_){
_start:
{
lean_object* v_res_678_; 
v_res_678_ = lp_mathlib_LinearMap_vecEmpty(v_R_670_, v_M_671_, v_M_u2083_672_, v_inst_673_, v_inst_674_, v_inst_675_, v_inst_676_, v_inst_677_);
lean_dec(v_inst_677_);
lean_dec(v_inst_676_);
lean_dec_ref(v_inst_675_);
lean_dec_ref(v_inst_674_);
lean_dec_ref(v_inst_673_);
return v_res_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons___redArg(lean_object* v_n_679_, lean_object* v_f_680_, lean_object* v_g_681_){
_start:
{
lean_object* v___x_682_; lean_object* v_toLinearMap_683_; lean_object* v___x_684_; lean_object* v___f_685_; 
v___x_682_ = lp_mathlib_Fin_consLinearEquiv___redArg(v_n_679_);
v_toLinearMap_683_ = lean_ctor_get(v___x_682_, 0);
lean_inc(v_toLinearMap_683_);
lean_dec_ref(v___x_682_);
v___x_684_ = lp_mathlib_LinearMap_prod___redArg(v_f_680_, v_g_681_);
v___f_685_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_685_, 0, v___x_684_);
lean_closure_set(v___f_685_, 1, v_toLinearMap_683_);
return v___f_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons(lean_object* v_R_686_, lean_object* v_M_687_, lean_object* v_M_u2082_688_, lean_object* v_inst_689_, lean_object* v_inst_690_, lean_object* v_inst_691_, lean_object* v_inst_692_, lean_object* v_inst_693_, lean_object* v_n_694_, lean_object* v_f_695_, lean_object* v_g_696_){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = lp_mathlib_LinearMap_vecCons___redArg(v_n_694_, v_f_695_, v_g_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons___boxed(lean_object* v_R_698_, lean_object* v_M_699_, lean_object* v_M_u2082_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_inst_704_, lean_object* v_inst_705_, lean_object* v_n_706_, lean_object* v_f_707_, lean_object* v_g_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_mathlib_LinearMap_vecCons(v_R_698_, v_M_699_, v_M_u2082_700_, v_inst_701_, v_inst_702_, v_inst_703_, v_inst_704_, v_inst_705_, v_n_706_, v_f_707_, v_g_708_);
lean_dec(v_inst_705_);
lean_dec(v_inst_704_);
lean_dec_ref(v_inst_703_);
lean_dec_ref(v_inst_702_);
lean_dec_ref(v_inst_701_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___lam__0(lean_object* v_x_710_, lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___lam__0___boxed(lean_object* v_x_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_LinearMap_vecEmpty_u2082___lam__0(v_x_713_, v___y_714_, v___y_715_);
lean_dec(v___y_715_);
lean_dec(v___y_714_);
lean_dec(v_x_713_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082(lean_object* v_R_718_, lean_object* v_M_719_, lean_object* v_M_u2082_720_, lean_object* v_M_u2083_721_, lean_object* v_inst_722_, lean_object* v_inst_723_, lean_object* v_inst_724_, lean_object* v_inst_725_, lean_object* v_inst_726_, lean_object* v_inst_727_, lean_object* v_inst_728_){
_start:
{
lean_object* v___f_729_; 
v___f_729_ = ((lean_object*)(lp_mathlib_LinearMap_vecEmpty_u2082___closed__0));
return v___f_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecEmpty_u2082___boxed(lean_object* v_R_730_, lean_object* v_M_731_, lean_object* v_M_u2082_732_, lean_object* v_M_u2083_733_, lean_object* v_inst_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_inst_740_){
_start:
{
lean_object* v_res_741_; 
v_res_741_ = lp_mathlib_LinearMap_vecEmpty_u2082(v_R_730_, v_M_731_, v_M_u2082_732_, v_M_u2083_733_, v_inst_734_, v_inst_735_, v_inst_736_, v_inst_737_, v_inst_738_, v_inst_739_, v_inst_740_);
lean_dec(v_inst_740_);
lean_dec(v_inst_739_);
lean_dec(v_inst_738_);
lean_dec_ref(v_inst_737_);
lean_dec_ref(v_inst_736_);
lean_dec_ref(v_inst_735_);
lean_dec_ref(v_inst_734_);
return v_res_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082___redArg___lam__0(lean_object* v_f_742_, lean_object* v_g_743_, lean_object* v_n_744_, lean_object* v_m_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_68__overap_750_; lean_object* v___x_751_; 
lean_inc(v_m_745_);
v___x_748_ = lean_apply_1(v_f_742_, v_m_745_);
v___x_749_ = lean_apply_1(v_g_743_, v_m_745_);
v___x_68__overap_750_ = lp_mathlib_LinearMap_vecCons___redArg(v_n_744_, v___x_748_, v___x_749_);
v___x_751_ = lean_apply_2(v___x_68__overap_750_, v___y_746_, v___y_747_);
return v___x_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082___redArg(lean_object* v_n_752_, lean_object* v_f_753_, lean_object* v_g_754_){
_start:
{
lean_object* v___f_755_; 
v___f_755_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_vecCons_u2082___redArg___lam__0), 6, 3);
lean_closure_set(v___f_755_, 0, v_f_753_);
lean_closure_set(v___f_755_, 1, v_g_754_);
lean_closure_set(v___f_755_, 2, v_n_752_);
return v___f_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082(lean_object* v_R_756_, lean_object* v_M_757_, lean_object* v_M_u2082_758_, lean_object* v_M_u2083_759_, lean_object* v_inst_760_, lean_object* v_inst_761_, lean_object* v_inst_762_, lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_n_767_, lean_object* v_f_768_, lean_object* v_g_769_){
_start:
{
lean_object* v___f_770_; 
v___f_770_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_vecCons_u2082___redArg___lam__0), 6, 3);
lean_closure_set(v___f_770_, 0, v_f_768_);
lean_closure_set(v___f_770_, 1, v_g_769_);
lean_closure_set(v___f_770_, 2, v_n_767_);
return v___f_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_vecCons_u2082___boxed(lean_object* v_R_771_, lean_object* v_M_772_, lean_object* v_M_u2082_773_, lean_object* v_M_u2083_774_, lean_object* v_inst_775_, lean_object* v_inst_776_, lean_object* v_inst_777_, lean_object* v_inst_778_, lean_object* v_inst_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_n_782_, lean_object* v_f_783_, lean_object* v_g_784_){
_start:
{
lean_object* v_res_785_; 
v_res_785_ = lp_mathlib_LinearMap_vecCons_u2082(v_R_771_, v_M_772_, v_M_u2082_773_, v_M_u2083_774_, v_inst_775_, v_inst_776_, v_inst_777_, v_inst_778_, v_inst_779_, v_inst_780_, v_inst_781_, v_n_782_, v_f_783_, v_g_784_);
lean_dec(v_inst_781_);
lean_dec(v_inst_780_);
lean_dec(v_inst_779_);
lean_dec_ref(v_inst_778_);
lean_dec_ref(v_inst_777_);
lean_dec_ref(v_inst_776_);
lean_dec_ref(v_inst_775_);
return v_res_785_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Tuple(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Tuple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Fin_Tuple(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Fin_Tuple(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
