// Lean compiler output
// Module: Mathlib.Data.Setoid.Basic
// Imports: public import Init public meta import Init public import Mathlib.Logic.Relation public import Mathlib.Order.CompleteLattice.Basic public import Mathlib.Order.GaloisConnection.Defs
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
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instLE__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_ker(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_ker___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_prodQuotientEquiv_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_prodQuotientEquiv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Setoid_prodQuotientEquiv_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Setoid_prodQuotientEquiv_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_prodQuotientEquiv___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_prodQuotientEquiv___redArg___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__1 = (const lean_object*)&lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__1_value),((lean_object*)&lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__2 = (const lean_object*)&lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instMin__mathlib___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Setoid_instMin__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_instMin__mathlib___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_instMin__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Setoid_instMin__mathlib___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instMin__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Setoid_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Setoid_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instInfSet(lean_object*);
static const lean_ctor_object lp_mathlib_Setoid_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Setoid_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_Setoid_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instPartialOrder(lean_object*);
static lean_once_cell_t lp_mathlib_Setoid_completeLattice___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Setoid_completeLattice___closed__0;
static lean_once_cell_t lp_mathlib_Setoid_completeLattice___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Setoid_completeLattice___closed__1;
static const lean_ctor_object lp_mathlib_Setoid_completeLattice___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Setoid_completeLattice___closed__2 = (const lean_object*)&lp_mathlib_Setoid_completeLattice___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_completeLattice(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientBotEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientBotEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Setoid_quotientBotEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_quotientBotEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_quotientBotEquiv___closed__0 = (const lean_object*)&lp_mathlib_Setoid_quotientBotEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Setoid_quotientBotEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Setoid_quotientBotEquiv___closed__0_value),((lean_object*)&lp_mathlib_Setoid_quotientBotEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Setoid_quotientBotEquiv___closed__1 = (const lean_object*)&lp_mathlib_Setoid_quotientBotEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientBotEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Setoid_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_gi___closed__0 = (const lean_object*)&lp_mathlib_Setoid_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_gi(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_liftEquiv___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Setoid_liftEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_liftEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_liftEquiv___closed__0 = (const lean_object*)&lp_mathlib_Setoid_liftEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Setoid_liftEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Setoid_liftEquiv___closed__0_value),((lean_object*)&lp_mathlib_Setoid_liftEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Setoid_liftEquiv___closed__1 = (const lean_object*)&lp_mathlib_Setoid_liftEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_liftEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_kerLift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_kerLift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_mapOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_mapOfSurjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_comap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__0 = (const lean_object*)&lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__0_value),((lean_object*)&lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__1 = (const lean_object*)&lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_correspondence___elam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_correspondence___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Setoid_correspondence___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_correspondence___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_correspondence___closed__0 = (const lean_object*)&lp_mathlib_Setoid_correspondence___closed__0_value;
static const lean_closure_object lp_mathlib_Setoid_correspondence___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_correspondence___elam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Setoid_correspondence___closed__1 = (const lean_object*)&lp_mathlib_Setoid_correspondence___closed__1_value;
static const lean_ctor_object lp_mathlib_Setoid_correspondence___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Setoid_correspondence___closed__0_value),((lean_object*)&lp_mathlib_Setoid_correspondence___closed__1_value)}};
static const lean_object* lp_mathlib_Setoid_correspondence___closed__2 = (const lean_object*)&lp_mathlib_Setoid_correspondence___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_correspondence(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__0 = (const lean_object*)&lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instLE__mathlib(lean_object* v_00_u03b1_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_ker(lean_object* v_00_u03b1_3_, lean_object* v_00_u03b2_4_, lean_object* v_f_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_ker___boxed(lean_object* v_00_u03b1_7_, lean_object* v_00_u03b2_8_, lean_object* v_f_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Setoid_ker(v_00_u03b1_7_, v_00_u03b2_8_, v_f_9_);
lean_dec(v_f_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prod(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_r_13_, lean_object* v_s_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_prodQuotientEquiv_spec__0___redArg(lean_object* v_q_16_, lean_object* v_f_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_apply_1(v_f_17_, v_q_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_prodQuotientEquiv_spec__0(lean_object* v_00_u03b1_19_, lean_object* v_00_u03b2_20_, lean_object* v_00_u03c6_21_, lean_object* v_q_22_, lean_object* v_f_23_, lean_object* v_h_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_apply_1(v_f_23_, v_q_22_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Setoid_prodQuotientEquiv_spec__1___redArg(lean_object* v_f_26_, lean_object* v_q_u2081_27_, lean_object* v_q_u2082_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_apply_2(v_f_26_, v_q_u2081_27_, v_q_u2082_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00Setoid_prodQuotientEquiv_spec__1(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_r_32_, lean_object* v_s_33_, lean_object* v_f_34_, lean_object* v_h_35_, lean_object* v_q_u2081_36_, lean_object* v_q_u2082_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_apply_2(v_f_34_, v_q_u2081_36_, v_q_u2082_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___lam__0(lean_object* v_q_39_){
_start:
{
lean_object* v_fst_40_; lean_object* v_snd_41_; lean_object* v___x_43_; uint8_t v_isShared_44_; uint8_t v_isSharedCheck_48_; 
v_fst_40_ = lean_ctor_get(v_q_39_, 0);
v_snd_41_ = lean_ctor_get(v_q_39_, 1);
v_isSharedCheck_48_ = !lean_is_exclusive(v_q_39_);
if (v_isSharedCheck_48_ == 0)
{
v___x_43_ = v_q_39_;
v_isShared_44_ = v_isSharedCheck_48_;
goto v_resetjp_42_;
}
else
{
lean_inc(v_snd_41_);
lean_inc(v_fst_40_);
lean_dec(v_q_39_);
v___x_43_ = lean_box(0);
v_isShared_44_ = v_isSharedCheck_48_;
goto v_resetjp_42_;
}
v_resetjp_42_:
{
lean_object* v___x_46_; 
if (v_isShared_44_ == 0)
{
v___x_46_ = v___x_43_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_47_; 
v_reuseFailAlloc_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_47_, 0, v_fst_40_);
lean_ctor_set(v_reuseFailAlloc_47_, 1, v_snd_41_);
v___x_46_ = v_reuseFailAlloc_47_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
return v___x_46_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg___lam__1(lean_object* v_x_49_){
_start:
{
lean_object* v_fst_50_; lean_object* v_snd_51_; lean_object* v___x_53_; uint8_t v_isShared_54_; uint8_t v_isSharedCheck_58_; 
v_fst_50_ = lean_ctor_get(v_x_49_, 0);
v_snd_51_ = lean_ctor_get(v_x_49_, 1);
v_isSharedCheck_58_ = !lean_is_exclusive(v_x_49_);
if (v_isSharedCheck_58_ == 0)
{
v___x_53_ = v_x_49_;
v_isShared_54_ = v_isSharedCheck_58_;
goto v_resetjp_52_;
}
else
{
lean_inc(v_snd_51_);
lean_inc(v_fst_50_);
lean_dec(v_x_49_);
v___x_53_ = lean_box(0);
v_isShared_54_ = v_isSharedCheck_58_;
goto v_resetjp_52_;
}
v_resetjp_52_:
{
lean_object* v___x_56_; 
if (v_isShared_54_ == 0)
{
v___x_56_ = v___x_53_;
goto v_reusejp_55_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v_fst_50_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v_snd_51_);
v___x_56_ = v_reuseFailAlloc_57_;
goto v_reusejp_55_;
}
v_reusejp_55_:
{
return v___x_56_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv___redArg(lean_object* v_r_64_, lean_object* v_s_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = ((lean_object*)(lp_mathlib_Setoid_prodQuotientEquiv___redArg___closed__2));
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_prodQuotientEquiv(lean_object* v_00_u03b1_67_, lean_object* v_00_u03b2_68_, lean_object* v_r_69_, lean_object* v_s_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_Setoid_prodQuotientEquiv___redArg(v_r_69_, v_s_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instMin__mathlib___lam__0(lean_object* v_r_72_, lean_object* v_s_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_box(0);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instMin__mathlib(lean_object* v_00_u03b1_76_){
_start:
{
lean_object* v___f_77_; 
v___f_77_ = ((lean_object*)(lp_mathlib_Setoid_instMin__mathlib___closed__0));
return v___f_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instInfSet___lam__0(lean_object* v_S_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lean_box(0);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instInfSet(lean_object* v_00_u03b1_81_){
_start:
{
lean_object* v___f_82_; 
v___f_82_ = ((lean_object*)(lp_mathlib_Setoid_instInfSet___closed__0));
return v___f_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_instPartialOrder(lean_object* v_00_u03b1_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = ((lean_object*)(lp_mathlib_Setoid_instPartialOrder___closed__0));
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_Setoid_completeLattice___closed__0(void){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_Setoid_instPartialOrder(lean_box(0));
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib_Setoid_completeLattice___closed__1(void){
_start:
{
lean_object* v___f_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___f_89_ = ((lean_object*)(lp_mathlib_Setoid_instInfSet___closed__0));
v___x_90_ = lean_obj_once(&lp_mathlib_Setoid_completeLattice___closed__0, &lp_mathlib_Setoid_completeLattice___closed__0_once, _init_lp_mathlib_Setoid_completeLattice___closed__0);
v___x_91_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_90_, v___f_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_completeLattice(lean_object* v_00_u03b1_94_){
_start:
{
lean_object* v___x_95_; lean_object* v_toLattice_96_; lean_object* v_toSupSet_97_; lean_object* v_toInfSet_98_; lean_object* v_toSemilatticeSup_99_; lean_object* v___f_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_95_ = lean_obj_once(&lp_mathlib_Setoid_completeLattice___closed__1, &lp_mathlib_Setoid_completeLattice___closed__1_once, _init_lp_mathlib_Setoid_completeLattice___closed__1);
v_toLattice_96_ = lean_ctor_get(v___x_95_, 0);
v_toSupSet_97_ = lean_ctor_get(v___x_95_, 1);
v_toInfSet_98_ = lean_ctor_get(v___x_95_, 2);
v_toSemilatticeSup_99_ = lean_ctor_get(v_toLattice_96_, 0);
v___f_100_ = ((lean_object*)(lp_mathlib_Setoid_instMin__mathlib___closed__0));
lean_inc_ref(v_toSemilatticeSup_99_);
v___x_101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_101_, 0, v_toSemilatticeSup_99_);
lean_ctor_set(v___x_101_, 1, v___f_100_);
v___x_102_ = ((lean_object*)(lp_mathlib_Setoid_completeLattice___closed__2));
lean_inc(v_toInfSet_98_);
lean_inc(v_toSupSet_97_);
v___x_103_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_103_, 0, v___x_101_);
lean_ctor_set(v___x_103_, 1, v_toSupSet_97_);
lean_ctor_set(v___x_103_, 2, v_toInfSet_98_);
lean_ctor_set(v___x_103_, 3, v___x_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le___redArg(lean_object* v_a_104_){
_start:
{
lean_inc(v_a_104_);
return v_a_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le___redArg___boxed(lean_object* v_a_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Setoid_map__of__le___redArg(v_a_105_);
lean_dec(v_a_105_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le(lean_object* v_00_u03b1_107_, lean_object* v_s_108_, lean_object* v_t_109_, lean_object* v_h_110_, lean_object* v_a_111_){
_start:
{
lean_inc(v_a_111_);
return v_a_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__of__le___boxed(lean_object* v_00_u03b1_112_, lean_object* v_s_113_, lean_object* v_t_114_, lean_object* v_h_115_, lean_object* v_a_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Setoid_map__of__le(v_00_u03b1_112_, v_s_113_, v_t_114_, v_h_115_, v_a_116_);
lean_dec(v_a_116_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf___redArg(lean_object* v_a_118_){
_start:
{
lean_inc(v_a_118_);
return v_a_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf___redArg___boxed(lean_object* v_a_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Setoid_map__sInf___redArg(v_a_119_);
lean_dec(v_a_119_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf(lean_object* v_00_u03b1_121_, lean_object* v_S_122_, lean_object* v_s_123_, lean_object* v_h_124_, lean_object* v_a_125_){
_start:
{
lean_inc(v_a_125_);
return v_a_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map__sInf___boxed(lean_object* v_00_u03b1_126_, lean_object* v_S_127_, lean_object* v_s_128_, lean_object* v_h_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Setoid_map__sInf(v_00_u03b1_126_, v_S_127_, v_s_128_, v_h_129_, v_a_130_);
lean_dec(v_a_130_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientBotEquiv___lam__0(lean_object* v___y_132_){
_start:
{
lean_inc(v___y_132_);
return v___y_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientBotEquiv___lam__0___boxed(lean_object* v___y_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Setoid_quotientBotEquiv___lam__0(v___y_133_);
lean_dec(v___y_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientBotEquiv(lean_object* v_00_u03b1_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = ((lean_object*)(lp_mathlib_Setoid_quotientBotEquiv___closed__1));
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_gi___lam__0(lean_object* v_r_140_, lean_object* v_x_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lean_box(0);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_gi(lean_object* v_00_u03b1_144_){
_start:
{
lean_object* v___f_145_; 
v___f_145_ = ((lean_object*)(lp_mathlib_Setoid_gi___closed__0));
return v___f_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_liftEquiv___lam__0(lean_object* v_f_146_, lean_object* v___y_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_apply_1(v_f_146_, v___y_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_liftEquiv(lean_object* v_00_u03b1_152_, lean_object* v_00_u03b2_153_, lean_object* v_r_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = ((lean_object*)(lp_mathlib_Setoid_liftEquiv___closed__1));
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_kerLift___redArg(lean_object* v_f_156_, lean_object* v_a_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lean_apply_1(v_f_156_, v_a_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_kerLift(lean_object* v_00_u03b1_159_, lean_object* v_00_u03b2_160_, lean_object* v_f_161_, lean_object* v_a_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lean_apply_1(v_f_161_, v_a_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object* v_g_164_, lean_object* v_b_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_apply_1(v_g_164_, v_b_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientKerEquivOfRightInverse___redArg(lean_object* v_f_167_, lean_object* v_g_168_){
_start:
{
lean_object* v___f_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_Setoid_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_169_, 0, v_g_168_);
v___x_170_ = lean_alloc_closure((void*)(lp_mathlib_Setoid_kerLift), 4, 3);
lean_closure_set(v___x_170_, 0, lean_box(0));
lean_closure_set(v___x_170_, 1, lean_box(0));
lean_closure_set(v___x_170_, 2, v_f_167_);
v___x_171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___f_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientKerEquivOfRightInverse(lean_object* v_00_u03b1_172_, lean_object* v_00_u03b2_173_, lean_object* v_f_174_, lean_object* v_g_175_, lean_object* v_hf_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_mathlib_Setoid_quotientKerEquivOfRightInverse___redArg(v_f_174_, v_g_175_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map(lean_object* v_00_u03b1_178_, lean_object* v_00_u03b2_179_, lean_object* v_r_180_, lean_object* v_f_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lean_box(0);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_map___boxed(lean_object* v_00_u03b1_183_, lean_object* v_00_u03b2_184_, lean_object* v_r_185_, lean_object* v_f_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Setoid_map(v_00_u03b1_183_, v_00_u03b2_184_, v_r_185_, v_f_186_);
lean_dec(v_f_186_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_mapOfSurjective(lean_object* v_00_u03b1_188_, lean_object* v_00_u03b2_189_, lean_object* v_r_190_, lean_object* v_f_191_, lean_object* v_h_192_, lean_object* v_hf_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lean_box(0);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_mapOfSurjective___boxed(lean_object* v_00_u03b1_195_, lean_object* v_00_u03b2_196_, lean_object* v_r_197_, lean_object* v_f_198_, lean_object* v_h_199_, lean_object* v_hf_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Setoid_mapOfSurjective(v_00_u03b1_195_, v_00_u03b2_196_, v_r_197_, v_f_198_, v_h_199_, v_hf_200_);
lean_dec(v_f_198_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_comap(lean_object* v_00_u03b1_202_, lean_object* v_00_u03b2_203_, lean_object* v_f_204_, lean_object* v_r_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lean_box(0);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_comap___boxed(lean_object* v_00_u03b1_207_, lean_object* v_00_u03b2_208_, lean_object* v_f_209_, lean_object* v_r_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_Setoid_comap(v_00_u03b1_207_, v_00_u03b2_208_, v_f_209_, v_r_210_);
lean_dec(v_f_209_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__0___redArg(lean_object* v_q_212_, lean_object* v_f_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_apply_1(v_f_213_, v_q_212_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__0(lean_object* v_00_u03b1_215_, lean_object* v_r_216_, lean_object* v_00_u03c6_217_, lean_object* v_q_218_, lean_object* v_f_219_, lean_object* v_h_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_apply_1(v_f_219_, v_q_218_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__1___redArg(lean_object* v_q_222_, lean_object* v_f_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_apply_1(v_f_223_, v_q_222_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Setoid_quotientQuotientEquivQuotient_spec__1(lean_object* v_00_u03c6_225_, lean_object* v_q_226_, lean_object* v_f_227_, lean_object* v_h_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lean_apply_1(v_f_227_, v_q_226_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___lam__0(lean_object* v_x_230_){
_start:
{
lean_inc(v_x_230_);
return v_x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___lam__0___boxed(lean_object* v_x_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___lam__0(v_x_231_);
lean_dec(v_x_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(lean_object* v_r_236_, lean_object* v_s_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = ((lean_object*)(lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg___closed__1));
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient(lean_object* v_00_u03b1_239_, lean_object* v_r_240_, lean_object* v_s_241_, lean_object* v_h_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(v_r_240_, v_s_241_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_correspondence___elam__0(lean_object* v_00_u03b1_244_, lean_object* v_s_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lean_box(0);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_correspondence___lam__0(lean_object* v_s_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lean_box(0);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_correspondence(lean_object* v_00_u03b1_254_, lean_object* v_r_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = ((lean_object*)(lp_mathlib_Setoid_correspondence___closed__2));
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__0(lean_object* v_r_257_, lean_object* v_x_258_){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; 
v___x_259_ = lean_box(0);
v___x_260_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_box(0), lean_box(0), v_r_257_, v___x_259_, lean_box(0), lean_box(0), lean_box(0));
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__0___boxed(lean_object* v_r_261_, lean_object* v_x_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__0(v_r_261_, v_x_262_);
lean_dec(v_x_262_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__1(lean_object* v_a_264_){
_start:
{
lean_inc(v_a_264_);
return v_a_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__1___boxed(lean_object* v_a_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__1(v_a_265_);
lean_dec(v_a_265_);
return v_res_266_;
}
}
static lean_object* _init_lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__1(void){
_start:
{
lean_object* v___f_268_; lean_object* v___x_269_; 
v___f_268_ = ((lean_object*)(lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__0));
v___x_269_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v___f_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg(lean_object* v_r_270_){
_start:
{
lean_object* v___f_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v___f_271_ = lean_alloc_closure((void*)(lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_271_, 0, v_r_270_);
v___x_272_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v___f_271_);
v___x_273_ = lp_mathlib_Equiv_symm___redArg(v___x_272_);
v___x_274_ = lean_obj_once(&lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__1, &lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__1_once, _init_lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg___closed__1);
v___x_275_ = lp_mathlib_Equiv_trans___redArg(v___x_273_, v___x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Setoid_sigmaQuotientEquivOfLe(lean_object* v_00_u03b1_276_, lean_object* v_r_277_, lean_object* v_s_278_, lean_object* v_hle_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_mathlib_Setoid_sigmaQuotientEquivOfLe___redArg(v_r_277_);
return v___x_280_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
