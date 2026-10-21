// Lean compiler output
// Module: Mathlib.Data.Finset.NAry
// Imports: public import Init public meta import Init public import Mathlib.Basic.Finite.Prod public import Mathlib.Data.Finset.Lattice.Prod public import Mathlib.Data.Set.Lattice.Image
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
lean_object* l_Function_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image_u2082___redArg(lean_object* v_inst_1_, lean_object* v_f_2_, lean_object* v_s_3_, lean_object* v_t_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_5_ = lean_alloc_closure((void*)(l_Function_uncurry), 5, 4);
lean_closure_set(v___x_5_, 0, lean_box(0));
lean_closure_set(v___x_5_, 1, lean_box(0));
lean_closure_set(v___x_5_, 2, lean_box(0));
lean_closure_set(v___x_5_, 3, v_f_2_);
v___x_6_ = lp_mathlib_Multiset_product___redArg(v_s_3_, v_t_4_);
v___x_7_ = lp_mathlib_Finset_image___redArg(v_inst_1_, v___x_5_, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_image_u2082(lean_object* v_00_u03b1_8_, lean_object* v_00_u03b2_9_, lean_object* v_00_u03b3_10_, lean_object* v_inst_11_, lean_object* v_f_12_, lean_object* v_s_13_, lean_object* v_t_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Finset_image_u2082___redArg(v_inst_11_, v_f_12_, v_s_13_, v_t_14_);
return v___x_15_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_NAry(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_NAry(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_NAry(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_NAry(builtin);
}
#ifdef __cplusplus
}
#endif
