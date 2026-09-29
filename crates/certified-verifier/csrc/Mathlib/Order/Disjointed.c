// Lean compiler output
// Module: Mathlib.Order.Disjointed
// Imports: public import Init public meta import Init public import Mathlib.Order.PartialSups public import Mathlib.Order.Interval.Finset.Fin public import Mathlib.Order.SuccPred.LinearLocallyFinite public import Mathlib.Order.Interval.Finset.SuccPred public import Mathlib.Data.Finset.Lattice.Union
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
lean_object* lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Finset_Iio___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_disjointed___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_disjointed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_disjointed___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_disjointed___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_f_3_, lean_object* v_i_4_){
_start:
{
lean_object* v_toSDiff_5_; lean_object* v_toBot_6_; lean_object* v___x_7_; lean_object* v_toLattice_8_; lean_object* v_toSemilatticeSup_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v_toSDiff_5_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_toSDiff_5_);
v_toBot_6_ = lean_ctor_get(v_inst_1_, 2);
lean_inc(v_toBot_6_);
v___x_7_ = lp_mathlib_GeneralizedBooleanAlgebra_toGeneralizedCoheytingAlgebra___redArg(v_inst_1_);
v_toLattice_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_toLattice_8_);
lean_dec_ref(v___x_7_);
v_toSemilatticeSup_9_ = lean_ctor_get(v_toLattice_8_, 0);
lean_inc_ref(v_toSemilatticeSup_9_);
lean_dec_ref(v_toLattice_8_);
lean_inc(v_f_3_);
lean_inc(v_i_4_);
v___x_10_ = lean_apply_1(v_f_3_, v_i_4_);
v___x_11_ = lp_mathlib_Finset_Iio___redArg(v_inst_2_, v_i_4_);
v___x_12_ = lp_mathlib_Finset_sup___redArg(v_toSemilatticeSup_9_, v_toBot_6_, v___x_11_, v_f_3_);
v___x_13_ = lean_apply_2(v_toSDiff_5_, v___x_10_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_disjointed(lean_object* v_00_u03b1_14_, lean_object* v_00_u03b9_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_f_19_, lean_object* v_i_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_disjointed___redArg(v_inst_16_, v_inst_18_, v_f_19_, v_i_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_disjointed___boxed(lean_object* v_00_u03b1_22_, lean_object* v_00_u03b9_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_f_27_, lean_object* v_i_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_disjointed(v_00_u03b1_22_, v_00_u03b9_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_f_27_, v_i_28_);
lean_dec_ref(v_inst_25_);
return v_res_29_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_PartialSups(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Disjointed(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_PartialSups(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Disjointed(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_PartialSups(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Disjointed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_PartialSups(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Disjointed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Disjointed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Disjointed(builtin);
}
#ifdef __cplusplus
}
#endif
