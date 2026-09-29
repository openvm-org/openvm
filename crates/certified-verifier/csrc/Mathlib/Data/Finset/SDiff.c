// Lean compiler output
// Module: Mathlib.Data.Finset.SDiff
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Insert public import Mathlib.Data.Finset.Lattice.Basic
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
lean_object* lp_mathlib_Finset_instLattice___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_s_u2081_2_, lean_object* v_s_u2082_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_mathlib_Multiset_sub___redArg(v_inst_1_, v_s_u2081_2_, v_s_u2082_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instSDiff___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSDiff(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___f_9_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instSDiff___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_inst_8_);
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___f_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
lean_inc_ref(v_inst_10_);
v___x_11_ = lp_mathlib_Finset_instLattice___redArg(v_inst_10_);
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instSDiff___redArg___lam__0), 3, 1);
lean_closure_set(v___f_12_, 0, v_inst_10_);
v___x_13_ = lean_box(0);
v___x_14_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_14_, 0, v___x_11_);
lean_ctor_set(v___x_14_, 1, v___f_12_);
lean_ctor_set(v___x_14_, 2, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instGeneralizedBooleanAlgebra(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_Finset_instGeneralizedBooleanAlgebra___redArg(v_inst_16_);
return v___x_17_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
}
#ifdef __cplusplus
}
#endif
