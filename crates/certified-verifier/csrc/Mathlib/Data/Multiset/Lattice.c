// Lean compiler output
// Module: Mathlib.Data.Multiset.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.FinsetOps public import Mathlib.Data.Multiset.Fold
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
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sup___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v_sup_4_; lean_object* v___x_5_; 
v_sup_4_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_sup_4_);
lean_dec_ref(v_inst_1_);
v___x_5_ = lean_apply_2(v_sup_4_, v_x1_2_, v_x2_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sup___redArg(lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_s_8_){
_start:
{
lean_object* v___f_9_; lean_object* v___x_10_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_inst_6_);
v___x_10_ = l_List_foldrTR___redArg(v___f_9_, v_inst_7_, v_s_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sup(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_s_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Multiset_sup___redArg(v_inst_12_, v_inst_13_, v_s_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inf___redArg___lam__0(lean_object* v_inst_16_, lean_object* v_x1_17_, lean_object* v_x2_18_){
_start:
{
lean_object* v_inf_19_; lean_object* v___x_20_; 
v_inf_19_ = lean_ctor_get(v_inst_16_, 1);
lean_inc(v_inf_19_);
lean_dec_ref(v_inst_16_);
v___x_20_ = lean_apply_2(v_inf_19_, v_x1_17_, v_x2_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inf___redArg(lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_s_23_){
_start:
{
lean_object* v___f_24_; lean_object* v___x_25_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_inf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_21_);
v___x_25_ = l_List_foldrTR___redArg(v___f_24_, v_inst_22_, v_s_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inf(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_s_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_Multiset_inf___redArg(v_inst_27_, v_inst_28_, v_s_29_);
return v___x_30_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
