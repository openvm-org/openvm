// Lean compiler output
// Module: Mathlib.Order.BourbakiWitt
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Lattice.Bounded public import Mathlib.Order.OmegaCompletePartialOrder
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
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSetLikeNonemptyChain(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instPartialOrderNonemptyChain___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instPartialOrderNonemptyChain___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderNonemptyChain(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSetLikeNonemptyChain(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_instPartialOrderNonemptyChain___closed__0(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_box(0);
v___x_5_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderNonemptyChain(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_obj_once(&lp_mathlib_instPartialOrderNonemptyChain___closed__0, &lp_mathlib_instPartialOrderNonemptyChain___closed__0_once, _init_lp_mathlib_instPartialOrderNonemptyChain___closed__0);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg___lam__0(lean_object* v_cSup_9_, lean_object* v_c_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = lean_box(0);
v___x_12_ = lean_apply_1(v_cSup_9_, v___x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg___lam__0___boxed(lean_object* v_cSup_13_, lean_object* v_c_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg___lam__0(v_cSup_13_, v_c_14_);
lean_dec(v_c_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg(lean_object* v_inst_16_){
_start:
{
lean_object* v_toPartialOrder_17_; lean_object* v_cSup_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_26_; 
v_toPartialOrder_17_ = lean_ctor_get(v_inst_16_, 0);
v_cSup_18_ = lean_ctor_get(v_inst_16_, 1);
v_isSharedCheck_26_ = !lean_is_exclusive(v_inst_16_);
if (v_isSharedCheck_26_ == 0)
{
v___x_20_ = v_inst_16_;
v_isShared_21_ = v_isSharedCheck_26_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_cSup_18_);
lean_inc(v_toPartialOrder_17_);
lean_dec(v_inst_16_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_26_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___f_22_; lean_object* v___x_24_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_22_, 0, v_cSup_18_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 1, v___f_22_);
v___x_24_ = v___x_20_;
goto v_reusejp_23_;
}
else
{
lean_object* v_reuseFailAlloc_25_; 
v_reuseFailAlloc_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_25_, 0, v_toPartialOrder_17_);
lean_ctor_set(v_reuseFailAlloc_25_, 1, v___f_22_);
v___x_24_ = v_reuseFailAlloc_25_;
goto v_reusejp_23_;
}
v_reusejp_23_:
{
return v___x_24_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_ChainCompletePartialOrder_instOmegaCompletePartialOrder___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg___lam__0(lean_object* v_toSupSet_30_, lean_object* v_c_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_apply_1(v_toSupSet_30_, lean_box(0));
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg(lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; lean_object* v_toPartialOrder_35_; lean_object* v___x_36_; lean_object* v_toSupSet_37_; lean_object* v___x_39_; uint8_t v_isShared_40_; uint8_t v_isSharedCheck_45_; 
lean_inc_ref(v_inst_33_);
v___x_34_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_33_);
v_toPartialOrder_35_ = lean_ctor_get(v___x_34_, 0);
lean_inc_ref(v_toPartialOrder_35_);
lean_dec_ref(v___x_34_);
v___x_36_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_33_);
v_toSupSet_37_ = lean_ctor_get(v___x_36_, 1);
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_36_);
if (v_isSharedCheck_45_ == 0)
{
lean_object* v_unused_46_; 
v_unused_46_ = lean_ctor_get(v___x_36_, 0);
lean_dec(v_unused_46_);
v___x_39_ = v___x_36_;
v_isShared_40_ = v_isSharedCheck_45_;
goto v_resetjp_38_;
}
else
{
lean_inc(v_toSupSet_37_);
lean_dec(v___x_36_);
v___x_39_ = lean_box(0);
v_isShared_40_ = v_isSharedCheck_45_;
goto v_resetjp_38_;
}
v_resetjp_38_:
{
lean_object* v___f_41_; lean_object* v___x_43_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_41_, 0, v_toSupSet_37_);
if (v_isShared_40_ == 0)
{
lean_ctor_set(v___x_39_, 1, v___f_41_);
lean_ctor_set(v___x_39_, 0, v_toPartialOrder_35_);
v___x_43_ = v___x_39_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v_toPartialOrder_35_);
lean_ctor_set(v_reuseFailAlloc_44_, 1, v___f_41_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_ChainCompletePartialOrder_instOfCompleteLattice___redArg(v_inst_48_);
return v___x_49_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
}
#ifdef __cplusplus
}
#endif
