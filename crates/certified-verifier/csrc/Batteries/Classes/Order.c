// Lean compiler output
// Module: Batteries.Classes.Order
// Imports: public import Init public meta import Init public import Batteries.Tactic.Basic public import Batteries.Tactic.SeqFocus
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
LEAN_EXPORT uint8_t lp_batteries_Ordering_byKey___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Ordering_byKey___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Ordering_byKey(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Ordering_byKey___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Ordering_byKey___redArg(lean_object* v_f_1_, lean_object* v_cmp_2_, lean_object* v_a_3_, lean_object* v_b_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; uint8_t v___x_8_; 
lean_inc(v_f_1_);
v___x_5_ = lean_apply_1(v_f_1_, v_a_3_);
v___x_6_ = lean_apply_1(v_f_1_, v_b_4_);
v___x_7_ = lean_apply_2(v_cmp_2_, v___x_5_, v___x_6_);
v___x_8_ = lean_unbox(v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Ordering_byKey___redArg___boxed(lean_object* v_f_9_, lean_object* v_cmp_10_, lean_object* v_a_11_, lean_object* v_b_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_batteries_Ordering_byKey___redArg(v_f_9_, v_cmp_10_, v_a_11_, v_b_12_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Ordering_byKey(lean_object* v_00_u03b1_15_, lean_object* v_00_u03b2_16_, lean_object* v_f_17_, lean_object* v_cmp_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; uint8_t v___x_24_; 
lean_inc(v_f_17_);
v___x_21_ = lean_apply_1(v_f_17_, v_a_19_);
v___x_22_ = lean_apply_1(v_f_17_, v_b_20_);
v___x_23_ = lean_apply_2(v_cmp_18_, v___x_21_, v___x_22_);
v___x_24_ = lean_unbox(v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Ordering_byKey___boxed(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_f_27_, lean_object* v_cmp_28_, lean_object* v_a_29_, lean_object* v_b_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_batteries_Ordering_byKey(v_00_u03b1_25_, v_00_u03b2_26_, v_f_27_, v_cmp_28_, v_a_29_, v_b_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Classes_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Classes_Order(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Tactic_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Classes_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Classes_Order(builtin);
}
#ifdef __cplusplus
}
#endif
