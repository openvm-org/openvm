// Lean compiler output
// Module: Mathlib.Order.Cofinal
// Imports: public import Init public meta import Init public import Mathlib.Order.Bounds.Defs public import Mathlib.Order.GaloisConnection.Defs public import Mathlib.Order.Hom.Basic public import Mathlib.Order.Interval.Set.Defs public import Mathlib.Order.SetNotation public import Mathlib.Tactic.ToAdditive import Mathlib.Data.Set.Lattice.Indexed
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
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedSubtypeSetIsCofinal___boxed(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_instInhabitedSubtypeSetIsCofinal(v_00_u03b1_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Cofinal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Cofinal(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Cofinal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Cofinal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Cofinal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Cofinal(builtin);
}
#ifdef __cplusplus
}
#endif
