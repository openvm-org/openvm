// Lean compiler output
// Module: Mathlib.Order.SupClosed
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Lattice.Prod public import Mathlib.Data.Finset.Powerset public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Order.Closure public import Mathlib.Order.ConditionallyCompleteLattice.Finset
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
LEAN_EXPORT lean_object* lp_mathlib_supClosure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supClosure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infClosure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infClosure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supClosure(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_supClosure___boxed(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_supClosure(v_00_u03b1_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infClosure(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infClosure___boxed(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_infClosure(v_00_u03b1_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_latticeClosure(v_00_u03b1_16_, v_inst_17_);
lean_dec_ref(v_inst_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0(lean_object* v_sSup_19_, lean_object* v_s_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_apply_1(v_sSup_19_, lean_box(0));
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg(lean_object* v_inst_22_, lean_object* v_sSup_23_){
_start:
{
lean_object* v_toPartialOrder_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_32_; 
v_toPartialOrder_24_ = lean_ctor_get(v_inst_22_, 0);
v_isSharedCheck_32_ = !lean_is_exclusive(v_inst_22_);
if (v_isSharedCheck_32_ == 0)
{
lean_object* v_unused_33_; 
v_unused_33_ = lean_ctor_get(v_inst_22_, 1);
lean_dec(v_unused_33_);
v___x_26_ = v_inst_22_;
v_isShared_27_ = v_isSharedCheck_32_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_toPartialOrder_24_);
lean_dec(v_inst_22_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_32_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___f_28_; lean_object* v___x_30_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_28_, 0, v_sSup_23_);
if (v_isShared_27_ == 0)
{
lean_ctor_set(v___x_26_, 1, v___f_28_);
v___x_30_ = v___x_26_;
goto v_reusejp_29_;
}
else
{
lean_object* v_reuseFailAlloc_31_; 
v_reuseFailAlloc_31_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_31_, 0, v_toPartialOrder_24_);
lean_ctor_set(v_reuseFailAlloc_31_, 1, v___f_28_);
v___x_30_ = v_reuseFailAlloc_31_;
goto v_reusejp_29_;
}
v_reusejp_29_:
{
return v___x_30_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_sSup_36_, lean_object* v_h_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg(v_inst_35_, v_sSup_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg(lean_object* v_inst_39_, lean_object* v_sSup_40_){
_start:
{
lean_object* v_toPartialOrder_41_; lean_object* v___x_43_; uint8_t v_isShared_44_; uint8_t v_isSharedCheck_49_; 
v_toPartialOrder_41_ = lean_ctor_get(v_inst_39_, 0);
v_isSharedCheck_49_ = !lean_is_exclusive(v_inst_39_);
if (v_isSharedCheck_49_ == 0)
{
lean_object* v_unused_50_; 
v_unused_50_ = lean_ctor_get(v_inst_39_, 1);
lean_dec(v_unused_50_);
v___x_43_ = v_inst_39_;
v_isShared_44_ = v_isSharedCheck_49_;
goto v_resetjp_42_;
}
else
{
lean_inc(v_toPartialOrder_41_);
lean_dec(v_inst_39_);
v___x_43_ = lean_box(0);
v_isShared_44_ = v_isSharedCheck_49_;
goto v_resetjp_42_;
}
v_resetjp_42_:
{
lean_object* v___f_45_; lean_object* v___x_47_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_45_, 0, v_sSup_40_);
if (v_isShared_44_ == 0)
{
lean_ctor_set(v___x_43_, 1, v___f_45_);
v___x_47_ = v___x_43_;
goto v_reusejp_46_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v_toPartialOrder_41_);
lean_ctor_set(v_reuseFailAlloc_48_, 1, v___f_45_);
v___x_47_ = v_reuseFailAlloc_48_;
goto v_reusejp_46_;
}
v_reusejp_46_:
{
return v___x_47_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_sSup_53_, lean_object* v_h_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg(v_inst_52_, v_sSup_53_);
return v___x_55_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Closure(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Closure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Closure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Closure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SupClosed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SupClosed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SupClosed(builtin);
}
#ifdef __cplusplus
}
#endif
