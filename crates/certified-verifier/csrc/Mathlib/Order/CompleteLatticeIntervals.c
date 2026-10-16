// Lean compiler output
// Module: Mathlib.Order.CompleteLatticeIntervals
// Imports: public import Init public meta import Init public import Mathlib.Order.ConditionallyCompleteLattice.Basic public import Mathlib.Order.LatticeIntervals public import Mathlib.Order.Interval.Set.OrdConnected
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
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_Set_Iic_instLatticeElem___redArg(lean_object*);
lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object*);
lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0(lean_object* v_toSupSet_1_, lean_object* v_S_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_toSupSet_1_, lean_box(0));
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1(lean_object* v_toLattice_4_, lean_object* v_toInfSet_5_, lean_object* v_a_6_, lean_object* v_S_7_){
_start:
{
lean_object* v_inf_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v_inf_8_ = lean_ctor_get(v_toLattice_4_, 1);
lean_inc(v_inf_8_);
lean_dec_ref(v_toLattice_4_);
v___x_9_ = lean_apply_1(v_toInfSet_5_, lean_box(0));
v___x_10_ = lean_apply_2(v_inf_8_, v_a_6_, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice___redArg(lean_object* v_inst_11_, lean_object* v_a_12_){
_start:
{
lean_object* v___x_13_; lean_object* v_toLattice_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v_toConditionallyCompletePartialOrderSup_17_; lean_object* v_toSupSet_18_; lean_object* v___x_19_; lean_object* v_toBoundedOrder_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_39_; 
lean_inc_ref(v_inst_11_);
v___x_13_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v_inst_11_);
v_toLattice_14_ = lean_ctor_get(v___x_13_, 0);
lean_inc_ref_n(v_toLattice_14_, 2);
v___x_15_ = lp_mathlib_Set_Iic_instLatticeElem___redArg(v_toLattice_14_);
v___x_16_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_13_);
v_toConditionallyCompletePartialOrderSup_17_ = lean_ctor_get(v___x_16_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_17_);
v_toSupSet_18_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_17_, 1);
lean_inc(v_toSupSet_18_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_17_);
v___x_19_ = lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(v___x_16_);
v_toBoundedOrder_20_ = lean_ctor_get(v_inst_11_, 3);
v_isSharedCheck_39_ = !lean_is_exclusive(v_inst_11_);
if (v_isSharedCheck_39_ == 0)
{
lean_object* v_unused_40_; lean_object* v_unused_41_; lean_object* v_unused_42_; 
v_unused_40_ = lean_ctor_get(v_inst_11_, 2);
lean_dec(v_unused_40_);
v_unused_41_ = lean_ctor_get(v_inst_11_, 1);
lean_dec(v_unused_41_);
v_unused_42_ = lean_ctor_get(v_inst_11_, 0);
lean_dec(v_unused_42_);
v___x_22_ = v_inst_11_;
v_isShared_23_ = v_isSharedCheck_39_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_toBoundedOrder_20_);
lean_dec(v_inst_11_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_39_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v_toInfSet_24_; lean_object* v_toOrderBot_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_37_; 
v_toInfSet_24_ = lean_ctor_get(v___x_19_, 1);
lean_inc(v_toInfSet_24_);
lean_dec_ref(v___x_19_);
v_toOrderBot_25_ = lean_ctor_get(v_toBoundedOrder_20_, 1);
v_isSharedCheck_37_ = !lean_is_exclusive(v_toBoundedOrder_20_);
if (v_isSharedCheck_37_ == 0)
{
lean_object* v_unused_38_; 
v_unused_38_ = lean_ctor_get(v_toBoundedOrder_20_, 0);
lean_dec(v_unused_38_);
v___x_27_ = v_toBoundedOrder_20_;
v_isShared_28_ = v_isSharedCheck_37_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_toOrderBot_25_);
lean_dec(v_toBoundedOrder_20_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_37_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___f_29_; lean_object* v___f_30_; lean_object* v___x_32_; 
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_29_, 0, v_toSupSet_18_);
lean_inc(v_a_12_);
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Set_Iic_instCompleteLattice___redArg___lam__1), 4, 3);
lean_closure_set(v___f_30_, 0, v_toLattice_14_);
lean_closure_set(v___f_30_, 1, v_toInfSet_24_);
lean_closure_set(v___f_30_, 2, v_a_12_);
if (v_isShared_28_ == 0)
{
lean_ctor_set(v___x_27_, 0, v_a_12_);
v___x_32_ = v___x_27_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_a_12_);
lean_ctor_set(v_reuseFailAlloc_36_, 1, v_toOrderBot_25_);
v___x_32_ = v_reuseFailAlloc_36_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
lean_object* v___x_34_; 
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 3, v___x_32_);
lean_ctor_set(v___x_22_, 2, v___f_30_);
lean_ctor_set(v___x_22_, 1, v___f_29_);
lean_ctor_set(v___x_22_, 0, v___x_15_);
v___x_34_ = v___x_22_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v___x_15_);
lean_ctor_set(v_reuseFailAlloc_35_, 1, v___f_29_);
lean_ctor_set(v_reuseFailAlloc_35_, 2, v___f_30_);
lean_ctor_set(v_reuseFailAlloc_35_, 3, v___x_32_);
v___x_34_ = v_reuseFailAlloc_35_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
return v___x_34_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Iic_instCompleteLattice(lean_object* v_00_u03b1_43_, lean_object* v_inst_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_Set_Iic_instCompleteLattice___redArg(v_inst_44_, v_a_45_);
return v___x_46_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(builtin);
}
#ifdef __cplusplus
}
#endif
