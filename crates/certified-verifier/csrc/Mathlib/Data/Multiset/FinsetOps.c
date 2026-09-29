// Lean compiler output
// Module: Mathlib.Data.Multiset.FinsetOps
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.Dedup public import Mathlib.Data.List.Infix
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
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_insert(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinsert(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndunion(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_ndinter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_s_3_){
_start:
{
lean_object* v___f_4_; uint8_t v___x_5_; 
v___f_4_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_4_, 0, v_inst_1_);
lean_inc(v_s_3_);
lean_inc(v_a_2_);
v___x_5_ = l_List_elem___redArg(v___f_4_, v_a_2_, v_s_3_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; 
v___x_6_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_6_, 0, v_a_2_);
lean_ctor_set(v___x_6_, 1, v_s_3_);
return v___x_6_;
}
else
{
lean_dec(v_a_2_);
return v_s_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinsert(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_a_9_, lean_object* v_s_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_Multiset_ndinsert___redArg(v_inst_8_, v_a_9_, v_s_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object* v_inst_12_, lean_object* v_s_13_, lean_object* v_t_14_){
_start:
{
lean_object* v___f_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___f_15_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_15_, 0, v_inst_12_);
v___x_16_ = lean_alloc_closure((void*)(l_List_insert), 4, 2);
lean_closure_set(v___x_16_, 0, lean_box(0));
lean_closure_set(v___x_16_, 1, v___f_15_);
v___x_17_ = l_List_foldrTR___redArg(v___x_16_, v_t_14_, v_s_13_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndunion(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_, lean_object* v_s_20_, lean_object* v_t_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_19_, v_s_20_, v_t_21_);
return v___x_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_ndinter___redArg___lam__0(lean_object* v_inst_23_, lean_object* v_t_24_, lean_object* v_a_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_23_, v_a_25_, v_t_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinter___redArg___lam__0___boxed(lean_object* v_inst_27_, lean_object* v_t_28_, lean_object* v_a_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_Multiset_ndinter___redArg___lam__0(v_inst_27_, v_t_28_, v_a_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinter___redArg(lean_object* v_inst_32_, lean_object* v_s_33_, lean_object* v_t_34_){
_start:
{
lean_object* v___f_35_; lean_object* v___x_36_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_ndinter___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_35_, 0, v_inst_32_);
lean_closure_set(v___f_35_, 1, v_t_34_);
v___x_36_ = lp_mathlib_Multiset_filter___redArg(v___f_35_, v_s_33_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ndinter(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_s_39_, lean_object* v_t_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Multiset_ndinter___redArg(v_inst_38_, v_s_39_, v_t_40_);
return v___x_41_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Dedup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Infix(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Infix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Dedup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Infix(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Dedup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Infix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
}
#ifdef __cplusplus
}
#endif
