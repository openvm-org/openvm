// Lean compiler output
// Module: Mathlib.Logic.Equiv.Set
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Function public import Mathlib.Logic.Equiv.Defs
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
lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumCompl___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_ofFiberEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumAssoc(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_equivOfEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_equivOfEq___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_equivOfEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_congr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setProdEquivSigma___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setProdEquivSigma___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_setProdEquivSigma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_setProdEquivSigma___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_setProdEquivSigma___closed__0 = (const lean_object*)&lp_mathlib_Equiv_setProdEquivSigma___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_setProdEquivSigma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_setProdEquivSigma___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_setProdEquivSigma___closed__1 = (const lean_object*)&lp_mathlib_Equiv_setProdEquivSigma___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_setProdEquivSigma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_setProdEquivSigma___closed__0_value),((lean_object*)&lp_mathlib_Equiv_setProdEquivSigma___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_setProdEquivSigma___closed__2 = (const lean_object*)&lp_mathlib_Equiv_setProdEquivSigma___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setProdEquivSigma(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setCongr___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_univ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_univ___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_univ___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_univ___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_Set_univ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_univ___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_univ___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Set_univ___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_Set_univ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Set_univ___closed__0_value),((lean_object*)&lp_mathlib_Equiv_Set_univ___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_Set_univ___closed__2 = (const lean_object*)&lp_mathlib_Equiv_Set_univ___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Set_empty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_empty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_empty(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Set_pempty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_pempty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_pempty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_union_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_union_x27___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_union_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_singleton___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_singleton___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_singleton___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_singleton___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Set_insert___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_insert___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_insert___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_insert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumCompl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumCompl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumDiffSubset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumDiffSubset(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Set_unionSumInter___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_unionSumInter___redArg___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_unionSumInter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_unionSumInter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_compl___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_compl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeEquiv___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_compl___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_compl___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_compl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_compl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Set_prod___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_prod___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_prod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univPi___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univPi___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_univPi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_univPi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_univPi___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_univPi___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_Set_univPi___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_univPi___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_univPi___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Set_univPi___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_Set_univPi___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Set_univPi___closed__0_value),((lean_object*)&lp_mathlib_Equiv_Set_univPi___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_Set_univPi___closed__2 = (const lean_object*)&lp_mathlib_Equiv_Set_univPi___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univPi(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Set_sep___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_sep___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_Set_sep___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Set_sep___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sep(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_powerset___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_powerset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_rangeInl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_rangeInl___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_rangeInl___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_rangeInl___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_Set_rangeInl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_rangeInl___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_rangeInl___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Set_rangeInl___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_Set_rangeInl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Set_rangeInl___closed__0_value),((lean_object*)&lp_mathlib_Equiv_Set_rangeInl___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_Set_rangeInl___closed__2 = (const lean_object*)&lp_mathlib_Equiv_Set_rangeInl___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_Set_rangeInr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_rangeInr___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_rangeInr___closed__0 = (const lean_object*)&lp_mathlib_Equiv_Set_rangeInr___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_Set_rangeInr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_Set_rangeInr___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_Set_rangeInr___closed__1 = (const lean_object*)&lp_mathlib_Equiv_Set_rangeInr___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_Set_rangeInr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_Set_rangeInr___closed__0_value),((lean_object*)&lp_mathlib_Equiv_Set_rangeInr___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_Set_rangeInr___closed__2 = (const lean_object*)&lp_mathlib_Equiv_Set_rangeInr___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPreimageEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPreimageEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofPreimageEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofPreimageEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Set_equivOfEq___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_equivOfEq(lean_object* v_00_u03b1_2_, lean_object* v_s_3_, lean_object* v_t_4_, lean_object* v_h_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_obj_once(&lp_mathlib_Set_equivOfEq___closed__0, &lp_mathlib_Set_equivOfEq___closed__0_once, _init_lp_mathlib_Set_equivOfEq___closed__0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_congr(lean_object* v_00_u03b1_7_, lean_object* v_s_8_, lean_object* v_t_9_, lean_object* v_h_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_mathlib_Set_equivOfEq___closed__0, &lp_mathlib_Set_equivOfEq___closed__0_once, _init_lp_mathlib_Set_equivOfEq___closed__0);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setProdEquivSigma___lam__0(lean_object* v_x_12_){
_start:
{
lean_object* v_fst_13_; lean_object* v_snd_14_; lean_object* v___x_16_; uint8_t v_isShared_17_; uint8_t v_isSharedCheck_21_; 
v_fst_13_ = lean_ctor_get(v_x_12_, 0);
v_snd_14_ = lean_ctor_get(v_x_12_, 1);
v_isSharedCheck_21_ = !lean_is_exclusive(v_x_12_);
if (v_isSharedCheck_21_ == 0)
{
v___x_16_ = v_x_12_;
v_isShared_17_ = v_isSharedCheck_21_;
goto v_resetjp_15_;
}
else
{
lean_inc(v_snd_14_);
lean_inc(v_fst_13_);
lean_dec(v_x_12_);
v___x_16_ = lean_box(0);
v_isShared_17_ = v_isSharedCheck_21_;
goto v_resetjp_15_;
}
v_resetjp_15_:
{
lean_object* v___x_19_; 
if (v_isShared_17_ == 0)
{
v___x_19_ = v___x_16_;
goto v_reusejp_18_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v_fst_13_);
lean_ctor_set(v_reuseFailAlloc_20_, 1, v_snd_14_);
v___x_19_ = v_reuseFailAlloc_20_;
goto v_reusejp_18_;
}
v_reusejp_18_:
{
return v___x_19_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setProdEquivSigma___lam__1(lean_object* v_x_22_){
_start:
{
lean_object* v_fst_23_; lean_object* v_snd_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_31_; 
v_fst_23_ = lean_ctor_get(v_x_22_, 0);
v_snd_24_ = lean_ctor_get(v_x_22_, 1);
v_isSharedCheck_31_ = !lean_is_exclusive(v_x_22_);
if (v_isSharedCheck_31_ == 0)
{
v___x_26_ = v_x_22_;
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_snd_24_);
lean_inc(v_fst_23_);
lean_dec(v_x_22_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_29_; 
if (v_isShared_27_ == 0)
{
v___x_29_ = v___x_26_;
goto v_reusejp_28_;
}
else
{
lean_object* v_reuseFailAlloc_30_; 
v_reuseFailAlloc_30_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_30_, 0, v_fst_23_);
lean_ctor_set(v_reuseFailAlloc_30_, 1, v_snd_24_);
v___x_29_ = v_reuseFailAlloc_30_;
goto v_reusejp_28_;
}
v_reusejp_28_:
{
return v___x_29_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setProdEquivSigma(lean_object* v_00_u03b1_37_, lean_object* v_00_u03b2_38_, lean_object* v_s_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = ((lean_object*)(lp_mathlib_Equiv_setProdEquivSigma___closed__2));
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setCongr(lean_object* v_00_u03b1_41_, lean_object* v_00_u03b2_42_, lean_object* v_e_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_44_, 0, lean_box(0));
lean_ctor_set(v___x_44_, 1, lean_box(0));
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setCongr___boxed(lean_object* v_00_u03b1_45_, lean_object* v_00_u03b2_46_, lean_object* v_e_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Equiv_setCongr(v_00_u03b1_45_, v_00_u03b2_46_, v_e_47_);
lean_dec_ref(v_e_47_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image___redArg___lam__0(lean_object* v_e_49_, lean_object* v_x_50_){
_start:
{
lean_object* v_toFun_51_; lean_object* v___x_52_; 
v_toFun_51_ = lean_ctor_get(v_e_49_, 0);
lean_inc(v_toFun_51_);
lean_dec_ref(v_e_49_);
v___x_52_ = lean_apply_1(v_toFun_51_, v_x_50_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image___redArg___lam__1(lean_object* v_e_53_, lean_object* v_y_54_){
_start:
{
lean_object* v___x_55_; lean_object* v_toFun_56_; lean_object* v___x_57_; 
v___x_55_ = lp_mathlib_Equiv_symm___redArg(v_e_53_);
v_toFun_56_ = lean_ctor_get(v___x_55_, 0);
lean_inc(v_toFun_56_);
lean_dec_ref(v___x_55_);
v___x_57_ = lean_apply_1(v_toFun_56_, v_y_54_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image___redArg(lean_object* v_e_58_){
_start:
{
lean_object* v___f_59_; lean_object* v___f_60_; lean_object* v___x_61_; 
lean_inc_ref(v_e_58_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_image___redArg___lam__0), 2, 1);
lean_closure_set(v___f_59_, 0, v_e_58_);
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_image___redArg___lam__1), 2, 1);
lean_closure_set(v___f_60_, 0, v_e_58_);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___f_59_);
lean_ctor_set(v___x_61_, 1, v___f_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_image(lean_object* v_00_u03b1_62_, lean_object* v_00_u03b2_63_, lean_object* v_e_64_, lean_object* v_s_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Equiv_image___redArg(v_e_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__0(lean_object* v_self_67_){
_start:
{
lean_inc(v_self_67_);
return v_self_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__0___boxed(lean_object* v_self_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Equiv_Set_univ___lam__0(v_self_68_);
lean_dec(v_self_68_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__1(lean_object* v_a_70_){
_start:
{
lean_inc(v_a_70_);
return v_a_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ___lam__1___boxed(lean_object* v_a_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_Equiv_Set_univ___lam__1(v_a_71_);
lean_dec(v_a_71_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univ(lean_object* v_00_u03b1_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = ((lean_object*)(lp_mathlib_Equiv_Set_univ___closed__2));
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_empty___closed__0(void){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_empty(lean_object* v_00_u03b1_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lean_obj_once(&lp_mathlib_Equiv_Set_empty___closed__0, &lp_mathlib_Equiv_Set_empty___closed__0_once, _init_lp_mathlib_Equiv_Set_empty___closed__0);
return v___x_82_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_pempty___closed__0(void){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_pempty(lean_object* v_00_u03b1_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_obj_once(&lp_mathlib_Equiv_Set_pempty___closed__0, &lp_mathlib_Equiv_Set_pempty___closed__0_once, _init_lp_mathlib_Equiv_Set_pempty___closed__0);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___lam__0(lean_object* v_inst_86_, lean_object* v_x_87_){
_start:
{
lean_object* v___x_88_; uint8_t v___x_89_; 
lean_inc(v_x_87_);
v___x_88_ = lean_apply_1(v_inst_86_, v_x_87_);
v___x_89_ = lean_unbox(v___x_88_);
if (v___x_89_ == 0)
{
lean_object* v___x_90_; 
v___x_90_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_90_, 0, v_x_87_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; 
v___x_91_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_91_, 0, v_x_87_);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___lam__1(lean_object* v_o_92_){
_start:
{
lean_object* v_val_93_; 
v_val_93_ = lean_ctor_get(v_o_92_, 0);
lean_inc(v_val_93_);
return v_val_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg___lam__1___boxed(lean_object* v_o_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Equiv_Set_union_x27___redArg___lam__1(v_o_94_);
lean_dec_ref(v_o_94_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27___redArg(lean_object* v_inst_97_){
_start:
{
lean_object* v___f_98_; lean_object* v___f_99_; lean_object* v___x_100_; 
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Set_union_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_98_, 0, v_inst_97_);
v___f_99_ = ((lean_object*)(lp_mathlib_Equiv_Set_union_x27___redArg___closed__0));
v___x_100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_100_, 0, v___f_98_);
lean_ctor_set(v___x_100_, 1, v___f_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union_x27(lean_object* v_00_u03b1_101_, lean_object* v_s_102_, lean_object* v_t_103_, lean_object* v_p_104_, lean_object* v_inst_105_, lean_object* v_hs_106_, lean_object* v_ht_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_Equiv_Set_union_x27___redArg(v_inst_105_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union___redArg(lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_Equiv_Set_union_x27___redArg(v_inst_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_union(lean_object* v_00_u03b1_111_, lean_object* v_s_112_, lean_object* v_t_113_, lean_object* v_inst_114_, lean_object* v_H_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_Equiv_Set_union_x27___redArg(v_inst_114_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__0(lean_object* v_x_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_box(0);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__0___boxed(lean_object* v_x_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Equiv_Set_singleton___redArg___lam__0(v_x_119_);
lean_dec(v_x_119_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__1(lean_object* v_a_121_, lean_object* v_x_122_){
_start:
{
lean_inc(v_a_121_);
return v_a_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg___lam__1___boxed(lean_object* v_a_123_, lean_object* v_x_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Equiv_Set_singleton___redArg___lam__1(v_a_123_, v_x_124_);
lean_dec(v_a_123_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton___redArg(lean_object* v_a_127_){
_start:
{
lean_object* v___f_128_; lean_object* v___f_129_; lean_object* v___x_130_; 
v___f_128_ = ((lean_object*)(lp_mathlib_Equiv_Set_singleton___redArg___closed__0));
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Set_singleton___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_129_, 0, v_a_127_);
v___x_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_130_, 0, v___f_128_);
lean_ctor_set(v___x_130_, 1, v___f_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_singleton(lean_object* v_00_u03b1_131_, lean_object* v_a_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_Equiv_Set_singleton___redArg(v_a_132_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_insert___redArg___closed__0(void){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_insert___redArg(lean_object* v_inst_135_, lean_object* v_a_136_){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_137_ = lean_obj_once(&lp_mathlib_Set_equivOfEq___closed__0, &lp_mathlib_Set_equivOfEq___closed__0_once, _init_lp_mathlib_Set_equivOfEq___closed__0);
v___x_138_ = lp_mathlib_Equiv_Set_union_x27___redArg(v_inst_135_);
v___x_139_ = lp_mathlib_Equiv_trans___redArg(v___x_137_, v___x_138_);
v___x_140_ = lean_obj_once(&lp_mathlib_Equiv_Set_insert___redArg___closed__0, &lp_mathlib_Equiv_Set_insert___redArg___closed__0_once, _init_lp_mathlib_Equiv_Set_insert___redArg___closed__0);
v___x_141_ = lp_mathlib_Equiv_Set_singleton___redArg(v_a_136_);
v___x_142_ = lp_mathlib_Equiv_sumCongr___redArg(v___x_140_, v___x_141_);
v___x_143_ = lp_mathlib_Equiv_trans___redArg(v___x_139_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_insert(lean_object* v_00_u03b1_144_, lean_object* v_s_145_, lean_object* v_inst_146_, lean_object* v_a_147_, lean_object* v_H_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_Equiv_Set_insert___redArg(v_inst_146_, v_a_147_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumCompl___redArg(lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumCompl(lean_object* v_00_u03b1_152_, lean_object* v_s_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumDiffSubset___redArg(lean_object* v_inst_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_157_ = lp_mathlib_Equiv_Set_union_x27___redArg(v_inst_156_);
v___x_158_ = lp_mathlib_Equiv_symm___redArg(v___x_157_);
v___x_159_ = lean_obj_once(&lp_mathlib_Set_equivOfEq___closed__0, &lp_mathlib_Set_equivOfEq___closed__0_once, _init_lp_mathlib_Set_equivOfEq___closed__0);
v___x_160_ = lp_mathlib_Equiv_trans___redArg(v___x_158_, v___x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sumDiffSubset(lean_object* v_00_u03b1_161_, lean_object* v_s_162_, lean_object* v_t_163_, lean_object* v_h_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_Equiv_Set_sumDiffSubset___redArg(v_inst_165_);
return v___x_166_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Set_unionSumInter___redArg___lam__0(lean_object* v_inst_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = lean_apply_1(v_inst_167_, v_a_168_);
v___x_170_ = lean_unbox(v___x_169_);
if (v___x_170_ == 0)
{
uint8_t v___x_171_; 
v___x_171_ = 1;
return v___x_171_;
}
else
{
uint8_t v___x_172_; 
v___x_172_ = 0;
return v___x_172_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_unionSumInter___redArg___lam__0___boxed(lean_object* v_inst_173_, lean_object* v_a_174_){
_start:
{
uint8_t v_res_175_; lean_object* v_r_176_; 
v_res_175_ = lp_mathlib_Equiv_Set_unionSumInter___redArg___lam__0(v_inst_173_, v_a_174_);
v_r_176_ = lean_box(v_res_175_);
return v_r_176_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__0(void){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_177_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__1(void){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_Equiv_sumAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_unionSumInter___redArg(lean_object* v_inst_179_){
_start:
{
lean_object* v___f_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
lean_inc_ref(v_inst_179_);
v___f_180_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Set_unionSumInter___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_180_, 0, v_inst_179_);
v___x_181_ = lean_obj_once(&lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__0, &lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__0_once, _init_lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__0);
v___x_182_ = lp_mathlib_Equiv_Set_union_x27___redArg(v_inst_179_);
v___x_183_ = lp_mathlib_Equiv_sumCongr___redArg(v___x_182_, v___x_181_);
v___x_184_ = lp_mathlib_Equiv_trans___redArg(v___x_181_, v___x_183_);
v___x_185_ = lean_obj_once(&lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__1, &lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__1_once, _init_lp_mathlib_Equiv_Set_unionSumInter___redArg___closed__1);
v___x_186_ = lp_mathlib_Equiv_trans___redArg(v___x_184_, v___x_185_);
v___x_187_ = lp_mathlib_Equiv_Set_union_x27___redArg(v___f_180_);
v___x_188_ = lp_mathlib_Equiv_symm___redArg(v___x_187_);
v___x_189_ = lp_mathlib_Equiv_sumCongr___redArg(v___x_181_, v___x_188_);
v___x_190_ = lp_mathlib_Equiv_trans___redArg(v___x_186_, v___x_189_);
v___x_191_ = lp_mathlib_Equiv_trans___redArg(v___x_190_, v___x_181_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_unionSumInter(lean_object* v_00_u03b1_192_, lean_object* v_s_193_, lean_object* v_t_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib_Equiv_Set_unionSumInter___redArg(v_inst_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_compl___redArg___lam__0(lean_object* v_inst_197_, lean_object* v_e_u2080_198_, lean_object* v_inst_199_, lean_object* v_e_u2081_200_){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_201_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_197_);
v___x_202_ = lp_mathlib_Equiv_symm___redArg(v___x_201_);
v___x_203_ = lp_mathlib_Equiv_sumCongr___redArg(v_e_u2080_198_, v_e_u2081_200_);
v___x_204_ = lp_mathlib_Equiv_trans___redArg(v___x_202_, v___x_203_);
v___x_205_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_199_);
v___x_206_ = lp_mathlib_Equiv_trans___redArg(v___x_204_, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_compl___redArg(lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_e_u2080_210_){
_start:
{
lean_object* v___f_211_; lean_object* v___f_212_; lean_object* v___x_213_; 
v___f_211_ = ((lean_object*)(lp_mathlib_Equiv_Set_compl___redArg___closed__0));
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Set_compl___redArg___lam__0), 4, 3);
lean_closure_set(v___f_212_, 0, v_inst_208_);
lean_closure_set(v___f_212_, 1, v_e_u2080_210_);
lean_closure_set(v___f_212_, 2, v_inst_209_);
v___x_213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_213_, 0, v___f_211_);
lean_ctor_set(v___x_213_, 1, v___f_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_compl(lean_object* v_00_u03b1_214_, lean_object* v_00_u03b2_215_, lean_object* v_s_216_, lean_object* v_t_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_e_u2080_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lp_mathlib_Equiv_Set_compl___redArg(v_inst_218_, v_inst_219_, v_e_u2080_220_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_prod___closed__0(void){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_prod(lean_object* v_00_u03b1_223_, lean_object* v_00_u03b2_224_, lean_object* v_s_225_, lean_object* v_t_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lean_obj_once(&lp_mathlib_Equiv_Set_prod___closed__0, &lp_mathlib_Equiv_Set_prod___closed__0_once, _init_lp_mathlib_Equiv_Set_prod___closed__0);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univPi___lam__0(lean_object* v_f_228_, lean_object* v_a_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lean_apply_1(v_f_228_, v_a_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univPi___lam__1(lean_object* v_f_231_, lean_object* v___y_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_apply_1(v_f_231_, v___y_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_univPi(lean_object* v_00_u03b1_239_, lean_object* v_00_u03b2_240_, lean_object* v_s_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = ((lean_object*)(lp_mathlib_Equiv_Set_univPi___closed__2));
return v___x_242_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_sep___closed__0(void){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter(lean_box(0), lean_box(0), lean_box(0));
return v___x_243_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Set_sep___closed__1(void){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_244_ = lean_obj_once(&lp_mathlib_Equiv_Set_sep___closed__0, &lp_mathlib_Equiv_Set_sep___closed__0_once, _init_lp_mathlib_Equiv_Set_sep___closed__0);
v___x_245_ = lp_mathlib_Equiv_symm___redArg(v___x_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_sep(lean_object* v_00_u03b1_246_, lean_object* v_s_247_, lean_object* v_t_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_obj_once(&lp_mathlib_Equiv_Set_sep___closed__1, &lp_mathlib_Equiv_Set_sep___closed__1_once, _init_lp_mathlib_Equiv_Set_sep___closed__1);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_powerset___lam__0(lean_object* v_x_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lean_box(0);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_powerset(lean_object* v_00_u03b1_252_, lean_object* v_S_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, lean_box(0));
lean_ctor_set(v___x_254_, 1, lean_box(0));
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl___lam__0(lean_object* v_x_255_){
_start:
{
lean_object* v_val_256_; 
v_val_256_ = lean_ctor_get(v_x_255_, 0);
lean_inc(v_val_256_);
return v_val_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl___lam__0___boxed(lean_object* v_x_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_Equiv_Set_rangeInl___lam__0(v_x_257_);
lean_dec_ref(v_x_257_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl___lam__1(lean_object* v_x_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_260_, 0, v_x_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInl(lean_object* v_00_u03b1_266_, lean_object* v_00_u03b2_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = ((lean_object*)(lp_mathlib_Equiv_Set_rangeInl___closed__2));
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr___lam__0(lean_object* v_x_269_){
_start:
{
lean_object* v_val_270_; 
v_val_270_ = lean_ctor_get(v_x_269_, 0);
lean_inc(v_val_270_);
return v_val_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr___lam__0___boxed(lean_object* v_x_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_Equiv_Set_rangeInr___lam__0(v_x_271_);
lean_dec_ref(v_x_271_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr___lam__1(lean_object* v_x_273_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_274_, 0, v_x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Set_rangeInr(lean_object* v_00_u03b1_280_, lean_object* v_00_u03b2_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = ((lean_object*)(lp_mathlib_Equiv_Set_rangeInr___closed__2));
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse___redArg___lam__1(lean_object* v_f__inv_283_, lean_object* v_b_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_apply_2(v_f__inv_283_, lean_box(0), v_b_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse___redArg(lean_object* v_f_286_, lean_object* v_f__inv_287_){
_start:
{
lean_object* v___f_288_; lean_object* v___f_289_; lean_object* v___x_290_; 
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Set_univPi___lam__0), 2, 1);
lean_closure_set(v___f_288_, 0, v_f_286_);
v___f_289_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofLeftInverse___redArg___lam__1), 2, 1);
lean_closure_set(v___f_289_, 0, v_f__inv_287_);
v___x_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_290_, 0, v___f_288_);
lean_ctor_set(v___x_290_, 1, v___f_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse(lean_object* v_00_u03b1_291_, lean_object* v_00_u03b2_292_, lean_object* v_f_293_, lean_object* v_f__inv_294_, lean_object* v_hf_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_Equiv_ofLeftInverse___redArg(v_f_293_, v_f__inv_294_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse_x27___redArg___lam__0(lean_object* v_f__inv_297_, lean_object* v_x_298_, lean_object* v___y_299_){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = lean_apply_1(v_f__inv_297_, v___y_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse_x27___redArg(lean_object* v_f_301_, lean_object* v_f__inv_302_){
_start:
{
lean_object* v___f_303_; lean_object* v___x_304_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofLeftInverse_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_303_, 0, v_f__inv_302_);
v___x_304_ = lp_mathlib_Equiv_ofLeftInverse___redArg(v_f_301_, v___f_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverse_x27(lean_object* v_00_u03b1_305_, lean_object* v_00_u03b2_306_, lean_object* v_f_307_, lean_object* v_f__inv_308_, lean_object* v_hf_309_){
_start:
{
lean_object* v___f_310_; lean_object* v___x_311_; 
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofLeftInverse_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_310_, 0, v_f__inv_308_);
v___x_311_ = lp_mathlib_Equiv_ofLeftInverse___redArg(v_f_307_, v___f_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPreimageEquiv___redArg(lean_object* v_f_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPreimageEquiv(lean_object* v_00_u03b1_314_, lean_object* v_00_u03b2_315_, lean_object* v_f_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofPreimageEquiv___redArg(lean_object* v_f_318_, lean_object* v_g_319_, lean_object* v_e_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_mathlib_Equiv_ofFiberEquiv___redArg(v_f_318_, v_g_319_, v_e_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofPreimageEquiv(lean_object* v_00_u03b1_322_, lean_object* v_00_u03b2_323_, lean_object* v_00_u03b3_324_, lean_object* v_f_325_, lean_object* v_g_326_, lean_object* v_e_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_Equiv_ofFiberEquiv___redArg(v_f_325_, v_g_326_, v_e_327_);
return v___x_328_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
}
#ifdef __cplusplus
}
#endif
