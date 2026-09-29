// Lean compiler output
// Module: Mathlib.Logic.Equiv.Fin.Rotate
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Fin.Basic public import Mathlib.Logic.Equiv.Fin.Basic
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_finAddFlip(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* l_Fin_sub(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_add(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_finRotate___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_finRotate___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_finRotate(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finRotate___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCycle(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_finRotate___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finRotate(lean_object* v_x_2_){
_start:
{
lean_object* v_zero_3_; uint8_t v_isZero_4_; 
v_zero_3_ = lean_unsigned_to_nat(0u);
v_isZero_4_ = lean_nat_dec_eq(v_x_2_, v_zero_3_);
if (v_isZero_4_ == 1)
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_finRotate___closed__0, &lp_mathlib_finRotate___closed__0_once, _init_lp_mathlib_finRotate___closed__0);
return v___x_5_;
}
else
{
lean_object* v_one_6_; lean_object* v_n_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v_one_6_ = lean_unsigned_to_nat(1u);
v_n_7_ = lean_nat_sub(v_x_2_, v_one_6_);
lean_inc(v_n_7_);
v___x_8_ = lp_mathlib_finAddFlip(v_n_7_, v_one_6_);
v___x_9_ = lean_nat_add(v_one_6_, v_n_7_);
v___x_10_ = lean_nat_add(v_n_7_, v_one_6_);
lean_dec(v_n_7_);
v___x_11_ = lp_mathlib_finCongr(v___x_9_, v___x_10_, lean_box(0));
lean_dec(v___x_10_);
lean_dec(v___x_9_);
v___x_12_ = lp_mathlib_Equiv_trans___redArg(v___x_8_, v___x_11_);
return v___x_12_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_finRotate___boxed(lean_object* v_x_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_finRotate(v_x_13_);
lean_dec(v_x_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__0(lean_object* v_n_15_, lean_object* v_k_16_, lean_object* v_i_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = l_Fin_add(v_n_15_, v_i_17_, v_k_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__0___boxed(lean_object* v_n_19_, lean_object* v_k_20_, lean_object* v_i_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_finCycle___lam__0(v_n_19_, v_k_20_, v_i_21_);
lean_dec(v_i_21_);
lean_dec(v_k_20_);
lean_dec(v_n_19_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__1(lean_object* v_n_23_, lean_object* v_k_24_, lean_object* v_i_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = l_Fin_sub(v_n_23_, v_i_25_, v_k_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCycle___lam__1___boxed(lean_object* v_n_27_, lean_object* v_k_28_, lean_object* v_i_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_finCycle___lam__1(v_n_27_, v_k_28_, v_i_29_);
lean_dec(v_i_29_);
lean_dec(v_k_28_);
lean_dec(v_n_27_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCycle(lean_object* v_n_31_, lean_object* v_k_32_){
_start:
{
lean_object* v___f_33_; lean_object* v___f_34_; lean_object* v___x_35_; 
lean_inc(v_k_32_);
lean_inc(v_n_31_);
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_finCycle___lam__0___boxed), 3, 2);
lean_closure_set(v___f_33_, 0, v_n_31_);
lean_closure_set(v___f_33_, 1, v_k_32_);
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_finCycle___lam__1___boxed), 3, 2);
lean_closure_set(v___f_34_, 0, v_n_31_);
lean_closure_set(v___f_34_, 1, v_k_32_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v___f_33_);
lean_ctor_set(v___x_35_, 1, v___f_34_);
return v___x_35_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(builtin);
}
#ifdef __cplusplus
}
#endif
