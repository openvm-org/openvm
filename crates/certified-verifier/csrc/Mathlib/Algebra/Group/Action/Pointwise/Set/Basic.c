// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Pointwise.Set.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Basic public import Mathlib.Algebra.Group.Action.Opposite public import Mathlib.Algebra.Group.Pointwise.Set.Scalar public import Mathlib.Algebra.Group.Units.Equiv public import Mathlib.Data.Set.Lattice.Image public import Mathlib.Data.Set.Pairwise.Basic public import Mathlib.Algebra.Group.Pointwise.Set.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Set_mulActionSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mulActionSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addActionSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addActionSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mulActionSet(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b2_2_, lean_object* v_inst_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mulActionSet___boxed(lean_object* v_00_u03b1_6_, lean_object* v_00_u03b2_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Set_mulActionSet(v_00_u03b1_6_, v_00_u03b2_7_, v_inst_8_, v_inst_9_);
lean_dec(v_inst_9_);
lean_dec_ref(v_inst_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addActionSet(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addActionSet___boxed(lean_object* v_00_u03b1_16_, lean_object* v_00_u03b2_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Set_addActionSet(v_00_u03b1_16_, v_00_u03b2_17_, v_inst_18_, v_inst_19_);
lean_dec(v_inst_19_);
lean_dec_ref(v_inst_18_);
return v_res_20_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
