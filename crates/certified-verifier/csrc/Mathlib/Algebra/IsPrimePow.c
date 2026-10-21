// Lean compiler output
// Module: Mathlib.Algebra.IsPrimePow
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Ring.Nat public import Mathlib.Order.Nat public import Mathlib.Data.Nat.Prime.Basic public import Mathlib.Data.Nat.Log public import Mathlib.Data.Nat.Prime.Pow public import Mathlib.Tactic.CrossRefAttribute
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_minFac(lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_log(lean_object*, lean_object*);
uint8_t l_Nat_decidableExistsLE___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableIsPrimePowNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableIsPrimePowNat___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableIsPrimePowNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableIsPrimePowNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableIsPrimePowNat___lam__0(lean_object* v_n_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_3_ = lean_unsigned_to_nat(0u);
v___x_4_ = lean_nat_dec_lt(v___x_3_, v_a_2_);
if (v___x_4_ == 0)
{
return v___x_4_;
}
else
{
lean_object* v___x_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_5_ = lp_mathlib_Nat_minFac(v_n_1_);
v___x_6_ = lean_nat_pow(v___x_5_, v_a_2_);
lean_dec(v___x_5_);
v___x_7_ = lean_nat_dec_eq(v_n_1_, v___x_6_);
lean_dec(v___x_6_);
return v___x_7_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableIsPrimePowNat___lam__0___boxed(lean_object* v_n_8_, lean_object* v_a_9_){
_start:
{
uint8_t v_res_10_; lean_object* v_r_11_; 
v_res_10_ = lp_mathlib_instDecidableIsPrimePowNat___lam__0(v_n_8_, v_a_9_);
lean_dec(v_a_9_);
lean_dec(v_n_8_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableIsPrimePowNat(lean_object* v_n_12_){
_start:
{
lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___x_15_; uint8_t v___x_16_; 
lean_inc(v_n_12_);
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_instDecidableIsPrimePowNat___lam__0___boxed), 2, 1);
lean_closure_set(v___f_13_, 0, v_n_12_);
v___x_14_ = lean_unsigned_to_nat(2u);
v___x_15_ = lp_mathlib_Nat_log(v___x_14_, v_n_12_);
v___x_16_ = l_Nat_decidableExistsLE___redArg(v___f_13_, v___x_15_);
lean_dec(v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableIsPrimePowNat___boxed(lean_object* v_n_17_){
_start:
{
uint8_t v_res_18_; lean_object* v_r_19_; 
v_res_18_ = lp_mathlib_instDecidableIsPrimePowNat(v_n_17_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Log(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Pow(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Log(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Log(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_IsPrimePow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Log(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Prime_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_IsPrimePow(builtin);
}
#ifdef __cplusplus
}
#endif
