// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.Cover public import Mathlib.Order.Interval.Finset.Defs public import Mathlib.Order.Preorder.Finite
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
lean_object* lp_mathlib_Set_instFintypeIcc___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_fintypeInterOfLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfMemBounds___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfMemBounds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfMemBounds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_IicFinsetSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_IicFinsetSet___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_IicFinsetSet___closed__0 = (const lean_object*)&lp_mathlib_Equiv_IicFinsetSet___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_IicFinsetSet___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_IicFinsetSet___closed__0_value),((lean_object*)&lp_mathlib_Equiv_IicFinsetSet___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_IicFinsetSet___closed__1 = (const lean_object*)&lp_mathlib_Equiv_IicFinsetSet___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfMemBounds___redArg(lean_object* v_a_1_, lean_object* v_b_2_, lean_object* v_inst_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lp_mathlib_Set_instFintypeIcc___redArg(v_inst_3_, v_a_1_, v_b_2_);
v___x_6_ = lp_mathlib_Set_fintypeInterOfLeft___redArg(v___x_5_, v_inst_4_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfMemBounds(lean_object* v_00_u03b1_7_, lean_object* v_a_8_, lean_object* v_b_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_s_12_, lean_object* v_inst_13_, lean_object* v_ha_14_, lean_object* v_hb_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Set_fintypeOfMemBounds___redArg(v_a_8_, v_b_9_, v_inst_11_, v_inst_13_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeOfMemBounds___boxed(lean_object* v_00_u03b1_17_, lean_object* v_a_18_, lean_object* v_b_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_s_22_, lean_object* v_inst_23_, lean_object* v_ha_24_, lean_object* v_hb_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Set_fintypeOfMemBounds(v_00_u03b1_17_, v_a_18_, v_b_19_, v_inst_20_, v_inst_21_, v_s_22_, v_inst_23_, v_ha_24_, v_hb_25_);
lean_dec_ref(v_inst_20_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet___lam__0(lean_object* v_b_27_){
_start:
{
lean_inc(v_b_27_);
return v_b_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet___lam__0___boxed(lean_object* v_b_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Equiv_IicFinsetSet___lam__0(v_b_28_);
lean_dec(v_b_28_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet(lean_object* v_00_u03b1_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_a_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = ((lean_object*)(lp_mathlib_Equiv_IicFinsetSet___closed__1));
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_IicFinsetSet___boxed(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_a_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Equiv_IicFinsetSet(v_00_u03b1_38_, v_inst_39_, v_inst_40_, v_a_41_);
lean_dec(v_a_41_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot___redArg(lean_object* v_inst_43_){
_start:
{
lean_inc(v_inst_43_);
return v_inst_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot___redArg___boxed(lean_object* v_inst_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Finset_instUniqueSubtypeMemIicBot___redArg(v_inst_44_);
lean_dec(v_inst_44_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_){
_start:
{
lean_inc(v_inst_49_);
return v_inst_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instUniqueSubtypeMemIicBot___boxed(lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Finset_instUniqueSubtypeMemIicBot(v_00_u03b1_50_, v_inst_51_, v_inst_52_, v_inst_53_);
lean_dec(v_inst_53_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
return v_res_54_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Preorder_Finite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Preorder_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Preorder_Finite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Preorder_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Interval_Finset_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
