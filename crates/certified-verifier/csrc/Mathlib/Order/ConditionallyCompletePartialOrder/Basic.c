// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompletePartialOrder.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.CompleteLattice.Defs public import Mathlib.Order.ConditionallyCompletePartialOrder.Defs
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
lean_object* lp_mathlib_OrderDual_instPreorder(lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_supSet___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderInfOfConditionallyCompletePartialOrderSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderInfOfConditionallyCompletePartialOrderSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderSupOfConditionallyCompletePartialOrderInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderSupOfConditionallyCompletePartialOrderInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderInfOfConditionallyCompletePartialOrderSup___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toPartialOrder_2_; lean_object* v_toSupSet_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_12_; 
v_toPartialOrder_2_ = lean_ctor_get(v_inst_1_, 0);
v_toSupSet_3_ = lean_ctor_get(v_inst_1_, 1);
v_isSharedCheck_12_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_12_ == 0)
{
v___x_5_ = v_inst_1_;
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toSupSet_3_);
lean_inc(v_toPartialOrder_2_);
lean_dec(v_inst_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_7_; lean_object* v___f_8_; lean_object* v___x_10_; 
v___x_7_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_toPartialOrder_2_);
lean_dec_ref(v_toPartialOrder_2_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_toSupSet_3_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 1, v___f_8_);
lean_ctor_set(v___x_5_, 0, v___x_7_);
v___x_10_ = v___x_5_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_11_; 
v_reuseFailAlloc_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_11_, 0, v___x_7_);
lean_ctor_set(v_reuseFailAlloc_11_, 1, v___f_8_);
v___x_10_ = v_reuseFailAlloc_11_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderInfOfConditionallyCompletePartialOrderSup(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_OrderDual_instConditionallyCompletePartialOrderInfOfConditionallyCompletePartialOrderSup___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderSupOfConditionallyCompletePartialOrderInf___redArg(lean_object* v_inst_16_){
_start:
{
lean_object* v_toPartialOrder_17_; lean_object* v_toInfSet_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_27_; 
v_toPartialOrder_17_ = lean_ctor_get(v_inst_16_, 0);
v_toInfSet_18_ = lean_ctor_get(v_inst_16_, 1);
v_isSharedCheck_27_ = !lean_is_exclusive(v_inst_16_);
if (v_isSharedCheck_27_ == 0)
{
v___x_20_ = v_inst_16_;
v_isShared_21_ = v_isSharedCheck_27_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_toInfSet_18_);
lean_inc(v_toPartialOrder_17_);
lean_dec(v_inst_16_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_27_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_22_; lean_object* v___f_23_; lean_object* v___x_25_; 
v___x_22_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_toPartialOrder_17_);
lean_dec_ref(v_toPartialOrder_17_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_supSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_23_, 0, v_toInfSet_18_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 1, v___f_23_);
lean_ctor_set(v___x_20_, 0, v___x_22_);
v___x_25_ = v___x_20_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_22_);
lean_ctor_set(v_reuseFailAlloc_26_, 1, v___f_23_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrderSupOfConditionallyCompletePartialOrderInf(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_OrderDual_instConditionallyCompletePartialOrderSupOfConditionallyCompletePartialOrderInf___redArg(v_inst_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrder___redArg(lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v_toConditionallyCompletePartialOrderSup_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_43_; 
lean_inc_ref(v_inst_31_);
v___x_32_ = lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(v_inst_31_);
v___x_33_ = lp_mathlib_OrderDual_instConditionallyCompletePartialOrderSupOfConditionallyCompletePartialOrderInf___redArg(v___x_32_);
v_toConditionallyCompletePartialOrderSup_34_ = lean_ctor_get(v_inst_31_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v_inst_31_);
if (v_isSharedCheck_43_ == 0)
{
lean_object* v_unused_44_; 
v_unused_44_ = lean_ctor_get(v_inst_31_, 1);
lean_dec(v_unused_44_);
v___x_36_ = v_inst_31_;
v_isShared_37_ = v_isSharedCheck_43_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_toConditionallyCompletePartialOrderSup_34_);
lean_dec(v_inst_31_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_43_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_38_; lean_object* v_toInfSet_39_; lean_object* v___x_41_; 
v___x_38_ = lp_mathlib_OrderDual_instConditionallyCompletePartialOrderInfOfConditionallyCompletePartialOrderSup___redArg(v_toConditionallyCompletePartialOrderSup_34_);
v_toInfSet_39_ = lean_ctor_get(v___x_38_, 1);
lean_inc(v_toInfSet_39_);
lean_dec_ref(v___x_38_);
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 1, v_toInfSet_39_);
lean_ctor_set(v___x_36_, 0, v___x_33_);
v___x_41_ = v___x_36_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_33_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_toInfSet_39_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompletePartialOrder(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_OrderDual_instConditionallyCompletePartialOrder___redArg(v_inst_46_);
return v___x_47_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
