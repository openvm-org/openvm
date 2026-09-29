// Lean compiler output
// Module: Mathlib.Algebra.MvPolynomial.Equiv
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Finsupp.Fin public import Mathlib.Algebra.MonoidAlgebra.Basic public import Mathlib.Algebra.MvPolynomial.Degrees public import Mathlib.Algebra.MvPolynomial.Rename public import Mathlib.Algebra.Polynomial.AlgebraMap public import Mathlib.Data.Finsupp.Option public import Mathlib.Logic.Equiv.Fin.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MvPolynomial_Equiv_0__Option_elim_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MvPolynomial_Equiv_0__Option_elim_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg___lam__0(lean_object* v_f_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg___lam__1(lean_object* v_g_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_apply_1(v_g_4_, v___y_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg(lean_object* v_f_7_, lean_object* v_g_8_){
_start:
{
lean_object* v___f_9_; lean_object* v___f_10_; lean_object* v___x_11_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg___lam__0), 2, 1);
lean_closure_set(v___f_9_, 0, v_f_7_);
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg___lam__1), 2, 1);
lean_closure_set(v___f_10_, 0, v_g_8_);
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v___f_9_);
lean_ctor_set(v___x_11_, 1, v___f_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial(lean_object* v_R_12_, lean_object* v_S_u2081_13_, lean_object* v_S_u2082_14_, lean_object* v_S_u2083_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_f_18_, lean_object* v_g_19_, lean_object* v_hfgC_20_, lean_object* v_hfgX_21_, lean_object* v_hgfC_22_, lean_object* v_hgfX_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___redArg(v_f_18_, v_g_19_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial___boxed(lean_object* v_R_25_, lean_object* v_S_u2081_26_, lean_object* v_S_u2082_27_, lean_object* v_S_u2083_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_f_31_, lean_object* v_g_32_, lean_object* v_hfgC_33_, lean_object* v_hfgX_34_, lean_object* v_hgfC_35_, lean_object* v_hgfX_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_MvPolynomial_mvPolynomialEquivMvPolynomial(v_R_25_, v_S_u2081_26_, v_S_u2082_27_, v_S_u2083_28_, v_inst_29_, v_inst_30_, v_f_31_, v_g_32_, v_hfgC_33_, v_hfgX_34_, v_hgfC_35_, v_hgfX_36_);
lean_dec_ref(v_inst_30_);
lean_dec_ref(v_inst_29_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MvPolynomial_Equiv_0__Option_elim_match__1_splitter___redArg(lean_object* v_x_38_, lean_object* v_x_39_, lean_object* v_x_40_, lean_object* v_h__1_41_, lean_object* v_h__2_42_){
_start:
{
if (lean_obj_tag(v_x_38_) == 0)
{
lean_object* v___x_43_; 
lean_dec(v_h__1_41_);
v___x_43_ = lean_apply_2(v_h__2_42_, v_x_39_, v_x_40_);
return v___x_43_;
}
else
{
lean_object* v_val_44_; lean_object* v___x_45_; 
lean_dec(v_h__2_42_);
v_val_44_ = lean_ctor_get(v_x_38_, 0);
lean_inc(v_val_44_);
lean_dec_ref_known(v_x_38_, 1);
v___x_45_ = lean_apply_3(v_h__1_41_, v_val_44_, v_x_39_, v_x_40_);
return v___x_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MvPolynomial_Equiv_0__Option_elim_match__1_splitter(lean_object* v_00_u03b1_46_, lean_object* v_00_u03b2_47_, lean_object* v_motive_48_, lean_object* v_x_49_, lean_object* v_x_50_, lean_object* v_x_51_, lean_object* v_h__1_52_, lean_object* v_h__2_53_){
_start:
{
if (lean_obj_tag(v_x_49_) == 0)
{
lean_object* v___x_54_; 
lean_dec(v_h__1_52_);
v___x_54_ = lean_apply_2(v_h__2_53_, v_x_50_, v_x_51_);
return v___x_54_;
}
else
{
lean_object* v_val_55_; lean_object* v___x_56_; 
lean_dec(v_h__2_53_);
v_val_55_ = lean_ctor_get(v_x_49_, 0);
lean_inc(v_val_55_);
lean_dec_ref_known(v_x_49_, 1);
v___x_56_ = lean_apply_3(v_h__1_52_, v_val_55_, v_x_50_, v_x_51_);
return v___x_56_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Fin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Option(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(builtin);
}
#ifdef __cplusplus
}
#endif
