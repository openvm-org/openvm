// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Coeff
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharP.Defs public import Mathlib.Algebra.MonoidAlgebra.Support public import Mathlib.Algebra.Polynomial.Basic public import Mathlib.Algebra.Regular.Basic public import Mathlib.Data.Nat.Choose.Sum
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
lean_object* lp_mathlib_Polynomial_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_constantCoeff___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Polynomial_constantCoeff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Polynomial_constantCoeff___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Polynomial_constantCoeff___closed__0 = (const lean_object*)&lp_mathlib_Polynomial_constantCoeff___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_constantCoeff(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_constantCoeff___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg___lam__0(lean_object* v_f_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_f_1_, v_x1_2_, v_x2_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg___lam__1(lean_object* v_inst_5_, lean_object* v___f_6_, lean_object* v_p_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Polynomial_sum___redArg(v_inst_5_, v_p_7_, v___f_6_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg___lam__1___boxed(lean_object* v_inst_9_, lean_object* v___f_10_, lean_object* v_p_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Polynomial_lsum___redArg___lam__1(v_inst_9_, v___f_10_, v_p_11_);
lean_dec_ref(v_inst_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___redArg(lean_object* v_inst_13_, lean_object* v_f_14_){
_start:
{
lean_object* v___f_15_; lean_object* v___f_16_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_lsum___redArg___lam__0), 3, 1);
lean_closure_set(v___f_15_, 0, v_f_14_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_lsum___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_16_, 0, v_inst_13_);
lean_closure_set(v___f_16_, 1, v___f_15_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum(lean_object* v_R_17_, lean_object* v_A_18_, lean_object* v_M_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_f_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Polynomial_lsum___redArg(v_inst_22_, v_f_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lsum___boxed(lean_object* v_R_27_, lean_object* v_A_28_, lean_object* v_M_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_f_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Polynomial_lsum(v_R_27_, v_A_28_, v_M_29_, v_inst_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_f_35_);
lean_dec(v_inst_34_);
lean_dec(v_inst_33_);
lean_dec_ref(v_inst_31_);
lean_dec_ref(v_inst_30_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff___redArg___lam__0(lean_object* v_n_37_, lean_object* v_p_38_){
_start:
{
lean_object* v_toFun_39_; lean_object* v___x_40_; 
v_toFun_39_ = lean_ctor_get(v_p_38_, 1);
lean_inc(v_toFun_39_);
lean_dec_ref(v_p_38_);
v___x_40_ = lean_apply_1(v_toFun_39_, v_n_37_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff___redArg(lean_object* v_n_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_lcoeff___redArg___lam__0), 2, 1);
lean_closure_set(v___f_42_, 0, v_n_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff(lean_object* v_R_43_, lean_object* v_inst_44_, lean_object* v_n_45_){
_start:
{
lean_object* v___f_46_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_lcoeff___redArg___lam__0), 2, 1);
lean_closure_set(v___f_46_, 0, v_n_45_);
return v___f_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_lcoeff___boxed(lean_object* v_R_47_, lean_object* v_inst_48_, lean_object* v_n_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Polynomial_lcoeff(v_R_47_, v_inst_48_, v_n_49_);
lean_dec_ref(v_inst_48_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_constantCoeff___lam__0(lean_object* v_p_51_){
_start:
{
lean_object* v_toFun_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v_toFun_52_ = lean_ctor_get(v_p_51_, 1);
lean_inc(v_toFun_52_);
lean_dec_ref(v_p_51_);
v___x_53_ = lean_unsigned_to_nat(0u);
v___x_54_ = lean_apply_1(v_toFun_52_, v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_constantCoeff(lean_object* v_R_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___f_58_; 
v___f_58_ = ((lean_object*)(lp_mathlib_Polynomial_constantCoeff___closed__0));
return v___f_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_constantCoeff___boxed(lean_object* v_R_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Polynomial_constantCoeff(v_R_59_, v_inst_60_);
lean_dec_ref(v_inst_60_);
return v_res_61_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharP_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Sum(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Coeff(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharP_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Coeff(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Sum(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Coeff(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Choose_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Coeff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Coeff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Coeff(builtin);
}
#ifdef __cplusplus
}
#endif
