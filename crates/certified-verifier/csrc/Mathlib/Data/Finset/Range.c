// Lean compiler output
// Module: Mathlib.Data.Finset.Range
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Insert public import Mathlib.Data.Multiset.Range public import Mathlib.Order.Interval.Set.Defs
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_range(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = l_List_range(v_n_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__0(lean_object* v_k_3_, lean_object* v_i_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_nat_sub(v_i_4_, v_k_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__0___boxed(lean_object* v_k_6_, lean_object* v_i_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_notMemRangeEquiv___lam__0(v_k_6_, v_i_7_);
lean_dec(v_i_7_);
lean_dec(v_k_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__1(lean_object* v_k_9_, lean_object* v_j_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_nat_add(v_j_10_, v_k_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv___lam__1___boxed(lean_object* v_k_12_, lean_object* v_j_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_notMemRangeEquiv___lam__1(v_k_12_, v_j_13_);
lean_dec(v_j_13_);
lean_dec(v_k_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_notMemRangeEquiv(lean_object* v_k_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___x_18_; 
lean_inc(v_k_15_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_notMemRangeEquiv___lam__0___boxed), 2, 1);
lean_closure_set(v___f_16_, 0, v_k_15_);
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_notMemRangeEquiv___lam__1___boxed), 2, 1);
lean_closure_set(v___f_17_, 0, v_k_15_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v___f_16_);
lean_ctor_set(v___x_18_, 1, v___f_17_);
return v___x_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Range(builtin);
}
#ifdef __cplusplus
}
#endif
