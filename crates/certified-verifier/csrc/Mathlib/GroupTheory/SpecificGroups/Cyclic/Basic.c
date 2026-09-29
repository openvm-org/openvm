// Lean compiler output
// Module: Mathlib.GroupTheory.SpecificGroups.Cyclic.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.ZMod.QuotientGroup
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
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc_ref(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_IsCyclic_commGroup___redArg(v_inst_2_);
lean_dec_ref(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_inc_ref(v_inst_5_);
return v_inst_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsCyclic_commGroup___boxed(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_IsCyclic_commGroup(v_00_u03b1_7_, v_inst_8_, v_inst_9_);
lean_dec_ref(v_inst_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup___redArg(lean_object* v_inst_11_){
_start:
{
lean_inc_ref(v_inst_11_);
return v_inst_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup___redArg___boxed(lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_IsAddCyclic_addCommGroup___redArg(v_inst_12_);
lean_dec_ref(v_inst_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_inc_ref(v_inst_15_);
return v_inst_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAddCyclic_addCommGroup___boxed(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_IsAddCyclic_addCommGroup(v_00_u03b1_17_, v_inst_18_, v_inst_19_);
lean_dec_ref(v_inst_18_);
return v_res_20_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_QuotientGroup(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_QuotientGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_ZMod_QuotientGroup(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_QuotientGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
