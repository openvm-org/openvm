// Lean compiler output
// Module: Mathlib.Algebra.MvPolynomial.Degrees
// Imports: public import Init public meta import Init public import Mathlib.Algebra.MonoidAlgebra.Degree public import Mathlib.Algebra.MvPolynomial.Rename
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
extern lean_object* lp_mathlib_Nat_instLattice;
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
lean_object* lp_mathlib_Finsupp_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MvPolynomial_totalDegree___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MvPolynomial_totalDegree___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___closed__0 = (const lean_object*)&lp_mathlib_MvPolynomial_totalDegree___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_MvPolynomial_totalDegree___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_degreesLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_degreesLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__0(lean_object* v_x_1_, lean_object* v_e_2_){
_start:
{
lean_inc(v_e_2_);
return v_e_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__0___boxed(lean_object* v_x_3_, lean_object* v_e_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_MvPolynomial_totalDegree___redArg___lam__0(v_x_3_, v_e_4_);
lean_dec(v_e_4_);
lean_dec(v_x_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__1(lean_object* v___x_6_, lean_object* v___f_7_, lean_object* v_s_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Finsupp_sum___redArg(v___x_6_, v_s_8_, v___f_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg___lam__1___boxed(lean_object* v___x_10_, lean_object* v___f_11_, lean_object* v_s_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_MvPolynomial_totalDegree___redArg___lam__1(v___x_10_, v___f_11_, v_s_12_);
lean_dec_ref(v___x_10_);
return v_res_13_;
}
}
static lean_object* _init_lp_mathlib_MvPolynomial_totalDegree___redArg___closed__1(void){
_start:
{
lean_object* v___f_15_; lean_object* v___x_16_; lean_object* v___f_17_; 
v___f_15_ = ((lean_object*)(lp_mathlib_MvPolynomial_totalDegree___redArg___closed__0));
v___x_16_ = lp_mathlib_Nat_instAddCancelCommMonoid;
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_totalDegree___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_17_, 0, v___x_16_);
lean_closure_set(v___f_17_, 1, v___f_15_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___redArg(lean_object* v_p_18_){
_start:
{
lean_object* v___x_19_; lean_object* v_toSemilatticeSup_20_; lean_object* v_support_21_; lean_object* v___x_22_; lean_object* v___f_23_; lean_object* v___x_24_; 
v___x_19_ = lp_mathlib_Nat_instLattice;
v_toSemilatticeSup_20_ = lean_ctor_get(v___x_19_, 0);
v_support_21_ = lean_ctor_get(v_p_18_, 0);
lean_inc(v_support_21_);
lean_dec_ref(v_p_18_);
v___x_22_ = lean_unsigned_to_nat(0u);
v___f_23_ = lean_obj_once(&lp_mathlib_MvPolynomial_totalDegree___redArg___closed__1, &lp_mathlib_MvPolynomial_totalDegree___redArg___closed__1_once, _init_lp_mathlib_MvPolynomial_totalDegree___redArg___closed__1);
lean_inc_ref(v_toSemilatticeSup_20_);
v___x_24_ = lp_mathlib_Finset_sup___redArg(v_toSemilatticeSup_20_, v___x_22_, v_support_21_, v___f_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree(lean_object* v_R_25_, lean_object* v_00_u03c3_26_, lean_object* v_inst_27_, lean_object* v_p_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_MvPolynomial_totalDegree___redArg(v_p_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_totalDegree___boxed(lean_object* v_R_30_, lean_object* v_00_u03c3_31_, lean_object* v_inst_32_, lean_object* v_p_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_MvPolynomial_totalDegree(v_R_30_, v_00_u03c3_31_, v_inst_32_, v_p_33_);
lean_dec_ref(v_inst_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_degreesLE(lean_object* v_R_35_, lean_object* v_00_u03c3_36_, lean_object* v_inst_37_, lean_object* v_s_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_box(0);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_degreesLE___boxed(lean_object* v_R_40_, lean_object* v_00_u03c3_41_, lean_object* v_inst_42_, lean_object* v_s_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_MvPolynomial_degreesLE(v_R_40_, v_00_u03c3_41_, v_inst_42_, v_s_43_);
lean_dec(v_s_43_);
lean_dec_ref(v_inst_42_);
return v_res_44_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MvPolynomial_Degrees(builtin);
}
#ifdef __cplusplus
}
#endif
