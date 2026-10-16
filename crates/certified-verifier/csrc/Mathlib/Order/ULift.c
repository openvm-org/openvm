// Lean compiler output
// Module: Mathlib.Order.ULift
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.ULift public import Mathlib.Order.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLE__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLT__mathlib(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBEq__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBEq__mathlib(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrd__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrd__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMax__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMax__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMax__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMin__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMin__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSDiff__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSDiff__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompl___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompl(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_ULift_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_ULift_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_ULift_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLE__mathlib(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLT__mathlib(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0(lean_object* v_inst_7_, lean_object* v_x_8_, lean_object* v_y_9_){
_start:
{
lean_object* v___x_10_; uint8_t v___x_11_; 
v___x_10_ = lean_apply_2(v_inst_7_, v_x_8_, v_y_9_);
v___x_11_ = lean_unbox(v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0___boxed(lean_object* v_inst_12_, lean_object* v_x_13_, lean_object* v_y_14_){
_start:
{
uint8_t v_res_15_; lean_object* v_r_16_; 
v_res_15_ = lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0(v_inst_12_, v_x_13_, v_y_14_);
v_r_16_ = lean_box(v_res_15_);
return v_r_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBEq__mathlib___redArg(lean_object* v_inst_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_18_, 0, v_inst_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instBEq__mathlib(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___f_21_; 
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instBEq__mathlib___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_21_, 0, v_inst_20_);
return v___f_21_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0(lean_object* v_inst_22_, lean_object* v_x_23_, lean_object* v_y_24_){
_start:
{
lean_object* v___x_25_; uint8_t v___x_26_; 
v___x_25_ = lean_apply_2(v_inst_22_, v_x_23_, v_y_24_);
v___x_26_ = lean_unbox(v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0___boxed(lean_object* v_inst_27_, lean_object* v_x_28_, lean_object* v_y_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0(v_inst_27_, v_x_28_, v_y_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrd__mathlib___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_33_, 0, v_inst_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instOrd__mathlib(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___f_36_; 
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_36_, 0, v_inst_35_);
return v___f_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMax__mathlib___redArg___lam__0(lean_object* v_inst_37_, lean_object* v_x_38_, lean_object* v_y_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_apply_2(v_inst_37_, v_x_38_, v_y_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMax__mathlib___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMax__mathlib(lean_object* v_00_u03b1_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_45_, 0, v_inst_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMin__mathlib___redArg(lean_object* v_inst_46_){
_start:
{
lean_object* v___f_47_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_47_, 0, v_inst_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instMin__mathlib(lean_object* v_00_u03b1_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___f_50_; 
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_50_, 0, v_inst_49_);
return v___f_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSDiff__mathlib___redArg(lean_object* v_inst_51_){
_start:
{
lean_object* v___f_52_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_52_, 0, v_inst_51_);
return v___f_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSDiff__mathlib(lean_object* v_00_u03b1_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v___f_55_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_55_, 0, v_inst_54_);
return v___f_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompl___redArg___lam__0(lean_object* v_inst_56_, lean_object* v_x_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_apply_1(v_inst_56_, v_x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompl___redArg(lean_object* v_inst_59_){
_start:
{
lean_object* v___f_60_; 
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompl___redArg___lam__0), 2, 1);
lean_closure_set(v___f_60_, 0, v_inst_59_);
return v___f_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompl(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___f_63_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompl___redArg___lam__0), 2, 1);
lean_closure_set(v___f_63_, 0, v_inst_62_);
return v___f_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPreorder(lean_object* v_00_u03b1_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = ((lean_object*)(lp_mathlib_ULift_instPreorder___closed__0));
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPreorder___boxed(lean_object* v_00_u03b1_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_ULift_instPreorder(v_00_u03b1_70_, v_inst_71_);
lean_dec_ref(v_inst_71_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPartialOrder(lean_object* v_00_u03b1_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = ((lean_object*)(lp_mathlib_ULift_instPreorder___closed__0));
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instPartialOrder___boxed(lean_object* v_00_u03b1_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_ULift_instPartialOrder(v_00_u03b1_76_, v_inst_77_);
lean_dec_ref(v_inst_77_);
return v_res_78_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ULift(builtin);
}
#ifdef __cplusplus
}
#endif
