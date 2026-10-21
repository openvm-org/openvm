// Lean compiler output
// Module: Mathlib.Order.PartialSups
// Imports: public import Init public meta import Init public import Mathlib.Order.ConditionallyCompleteLattice.Indexed public import Mathlib.Order.Interval.Finset.Nat public import Mathlib.Order.SuccPred.Basic import Mathlib.Data.Finset.Max import Mathlib.Data.Fintype.Order
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
lean_object* lp_mathlib_Finset_Iic___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups_gi___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_partialSups_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_partialSups_gi___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_partialSups_gi___closed__0 = (const lean_object*)&lp_mathlib_partialSups_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_partialSups_gi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups_gi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialSups___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_f_3_, lean_object* v_i_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lp_mathlib_Finset_Iic___redArg(v_inst_1_, v_i_4_);
v___x_6_ = lp_mathlib_Finset_sup_x27___redArg(v_inst_2_, v___x_5_, v_f_3_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialSups___redArg(lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_f_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_partialSups___redArg___lam__0), 4, 3);
lean_closure_set(v___f_10_, 0, v_inst_8_);
lean_closure_set(v___f_10_, 1, v_inst_7_);
lean_closure_set(v___f_10_, 2, v_f_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialSups(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b9_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_partialSups___redArg___lam__0), 4, 3);
lean_closure_set(v___f_17_, 0, v_inst_15_);
lean_closure_set(v___f_17_, 1, v_inst_13_);
lean_closure_set(v___f_17_, 2, v_f_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialSups___boxed(lean_object* v_00_u03b1_18_, lean_object* v_00_u03b9_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_f_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_partialSups(v_00_u03b1_18_, v_00_u03b9_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_f_23_);
lean_dec_ref(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialSups_gi___lam__0(lean_object* v_f_25_, lean_object* v_h_26_, lean_object* v___y_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_apply_1(v_f_25_, v___y_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialSups_gi(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b9_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = ((lean_object*)(lp_mathlib_partialSups_gi___closed__0));
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_partialSups_gi___boxed(lean_object* v_00_u03b1_36_, lean_object* v_00_u03b9_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_partialSups_gi(v_00_u03b1_36_, v_00_u03b9_37_, v_inst_38_, v_inst_39_, v_inst_40_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_38_);
return v_res_41_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Order(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_PartialSups(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_PartialSups(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Order(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_PartialSups(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_PartialSups(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_PartialSups(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_PartialSups(builtin);
}
#ifdef __cplusplus
}
#endif
