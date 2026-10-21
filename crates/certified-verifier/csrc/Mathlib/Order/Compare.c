// Lean compiler output
// Module: Mathlib.Order.Compare
// Imports: public import Init public meta import Init public import Mathlib.Data.Ordering.Basic public import Mathlib.Order.OrderDual
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
uint8_t l_instDecidableEqOrdering(uint8_t, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_cmpLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cmpLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_cmpLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cmpLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_cmpLE___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_y_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
lean_inc_ref(v_inst_1_);
lean_inc(v_y_3_);
lean_inc(v_x_2_);
v___x_4_ = lean_apply_2(v_inst_1_, v_x_2_, v_y_3_);
v___x_5_ = lean_unbox(v___x_4_);
if (v___x_5_ == 0)
{
uint8_t v___x_6_; 
lean_dec(v_y_3_);
lean_dec(v_x_2_);
lean_dec_ref(v_inst_1_);
v___x_6_ = 2;
return v___x_6_;
}
else
{
lean_object* v___x_7_; uint8_t v___x_8_; 
v___x_7_ = lean_apply_2(v_inst_1_, v_y_3_, v_x_2_);
v___x_8_ = lean_unbox(v___x_7_);
if (v___x_8_ == 0)
{
uint8_t v___x_9_; 
v___x_9_ = 0;
return v___x_9_;
}
else
{
uint8_t v___x_10_; 
v___x_10_ = 1;
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_cmpLE___redArg___boxed(lean_object* v_inst_11_, lean_object* v_x_12_, lean_object* v_y_13_){
_start:
{
uint8_t v_res_14_; lean_object* v_r_15_; 
v_res_14_ = lp_mathlib_cmpLE___redArg(v_inst_11_, v_x_12_, v_y_13_);
v_r_15_ = lean_box(v_res_14_);
return v_r_15_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_cmpLE(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_x_19_, lean_object* v_y_20_){
_start:
{
uint8_t v___x_21_; 
v___x_21_ = lp_mathlib_cmpLE___redArg(v_inst_18_, v_x_19_, v_y_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cmpLE___boxed(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_x_25_, lean_object* v_y_26_){
_start:
{
uint8_t v_res_27_; lean_object* v_r_28_; 
v_res_27_ = lp_mathlib_cmpLE(v_00_u03b1_22_, v_inst_23_, v_inst_24_, v_x_25_, v_y_26_);
v_r_28_ = lean_box(v_res_27_);
return v_r_28_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__0(lean_object* v_cmp_29_, lean_object* v_a_30_, lean_object* v_b_31_){
_start:
{
lean_object* v___x_32_; uint8_t v___x_33_; uint8_t v___x_34_; uint8_t v___x_35_; 
v___x_32_ = lean_apply_2(v_cmp_29_, v_a_30_, v_b_31_);
v___x_33_ = 2;
v___x_34_ = lean_unbox(v___x_32_);
v___x_35_ = l_instDecidableEqOrdering(v___x_34_, v___x_33_);
if (v___x_35_ == 0)
{
uint8_t v___x_36_; 
v___x_36_ = 1;
return v___x_36_;
}
else
{
uint8_t v___x_37_; 
v___x_37_ = 0;
return v___x_37_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__0___boxed(lean_object* v_cmp_38_, lean_object* v_a_39_, lean_object* v_b_40_){
_start:
{
uint8_t v_res_41_; lean_object* v_r_42_; 
v_res_41_ = lp_mathlib_linearOrderOfCompares___redArg___lam__0(v_cmp_38_, v_a_39_, v_b_40_);
v_r_42_ = lean_box(v_res_41_);
return v_r_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__1(lean_object* v_H_43_, lean_object* v_x_44_, lean_object* v_y_45_){
_start:
{
lean_object* v___x_46_; uint8_t v___x_47_; 
lean_inc(v_y_45_);
lean_inc(v_x_44_);
v___x_46_ = lean_apply_2(v_H_43_, v_x_44_, v_y_45_);
v___x_47_ = lean_unbox(v___x_46_);
if (v___x_47_ == 0)
{
lean_dec(v_x_44_);
return v_y_45_;
}
else
{
lean_dec(v_y_45_);
return v_x_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__2(lean_object* v_H_48_, lean_object* v_x_49_, lean_object* v_y_50_){
_start:
{
lean_object* v___x_51_; uint8_t v___x_52_; 
lean_inc(v_y_50_);
lean_inc(v_x_49_);
v___x_51_ = lean_apply_2(v_H_48_, v_x_49_, v_y_50_);
v___x_52_ = lean_unbox(v___x_51_);
if (v___x_52_ == 0)
{
lean_dec(v_y_50_);
return v_x_49_;
}
else
{
lean_dec(v_x_49_);
return v_y_50_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__3(lean_object* v_cmp_53_, lean_object* v_a_54_, lean_object* v_b_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; uint8_t v___x_58_; uint8_t v___x_59_; 
v___x_56_ = lean_apply_2(v_cmp_53_, v_a_54_, v_b_55_);
v___x_57_ = 0;
v___x_58_ = lean_unbox(v___x_56_);
v___x_59_ = l_instDecidableEqOrdering(v___x_58_, v___x_57_);
if (v___x_59_ == 0)
{
uint8_t v___x_60_; uint8_t v___x_61_; uint8_t v___x_62_; 
v___x_60_ = 1;
v___x_61_ = lean_unbox(v___x_56_);
v___x_62_ = l_instDecidableEqOrdering(v___x_61_, v___x_60_);
if (v___x_62_ == 0)
{
uint8_t v___x_63_; 
v___x_63_ = 2;
return v___x_63_;
}
else
{
return v___x_60_;
}
}
else
{
return v___x_57_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__3___boxed(lean_object* v_cmp_64_, lean_object* v_a_65_, lean_object* v_b_66_){
_start:
{
uint8_t v_res_67_; lean_object* v_r_68_; 
v_res_67_ = lp_mathlib_linearOrderOfCompares___redArg___lam__3(v_cmp_64_, v_a_65_, v_b_66_);
v_r_68_ = lean_box(v_res_67_);
return v_r_68_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__4(lean_object* v_cmp_69_, lean_object* v_a_70_, lean_object* v_b_71_){
_start:
{
lean_object* v___x_72_; uint8_t v___x_73_; uint8_t v___x_74_; uint8_t v___x_75_; 
v___x_72_ = lean_apply_2(v_cmp_69_, v_a_70_, v_b_71_);
v___x_73_ = 1;
v___x_74_ = lean_unbox(v___x_72_);
v___x_75_ = l_instDecidableEqOrdering(v___x_74_, v___x_73_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__4___boxed(lean_object* v_cmp_76_, lean_object* v_a_77_, lean_object* v_b_78_){
_start:
{
uint8_t v_res_79_; lean_object* v_r_80_; 
v_res_79_ = lp_mathlib_linearOrderOfCompares___redArg___lam__4(v_cmp_76_, v_a_77_, v_b_78_);
v_r_80_ = lean_box(v_res_79_);
return v_r_80_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfCompares___redArg___lam__5(lean_object* v_cmp_81_, lean_object* v_a_82_, lean_object* v_b_83_){
_start:
{
lean_object* v___x_84_; uint8_t v___x_85_; uint8_t v___x_86_; uint8_t v___x_87_; 
v___x_84_ = lean_apply_2(v_cmp_81_, v_a_82_, v_b_83_);
v___x_85_ = 0;
v___x_86_ = lean_unbox(v___x_84_);
v___x_87_ = l_instDecidableEqOrdering(v___x_86_, v___x_85_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg___lam__5___boxed(lean_object* v_cmp_88_, lean_object* v_a_89_, lean_object* v_b_90_){
_start:
{
uint8_t v_res_91_; lean_object* v_r_92_; 
v_res_91_ = lp_mathlib_linearOrderOfCompares___redArg___lam__5(v_cmp_88_, v_a_89_, v_b_90_);
v_r_92_ = lean_box(v_res_91_);
return v_r_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares___redArg(lean_object* v_inst_93_, lean_object* v_cmp_94_){
_start:
{
lean_object* v_H_95_; lean_object* v___f_96_; lean_object* v___f_97_; lean_object* v___f_98_; lean_object* v___f_99_; lean_object* v___f_100_; lean_object* v___x_101_; 
lean_inc_ref_n(v_cmp_94_, 3);
v_H_95_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfCompares___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v_H_95_, 0, v_cmp_94_);
lean_inc_ref_n(v_H_95_, 2);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfCompares___redArg___lam__1), 3, 1);
lean_closure_set(v___f_96_, 0, v_H_95_);
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfCompares___redArg___lam__2), 3, 1);
lean_closure_set(v___f_97_, 0, v_H_95_);
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfCompares___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_98_, 0, v_cmp_94_);
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfCompares___redArg___lam__4___boxed), 3, 1);
lean_closure_set(v___f_99_, 0, v_cmp_94_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfCompares___redArg___lam__5___boxed), 3, 1);
lean_closure_set(v___f_100_, 0, v_cmp_94_);
v___x_101_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_101_, 0, v_inst_93_);
lean_ctor_set(v___x_101_, 1, v___f_96_);
lean_ctor_set(v___x_101_, 2, v___f_97_);
lean_ctor_set(v___x_101_, 3, v___f_98_);
lean_ctor_set(v___x_101_, 4, v_H_95_);
lean_ctor_set(v___x_101_, 5, v___f_99_);
lean_ctor_set(v___x_101_, 6, v___f_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfCompares(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_, lean_object* v_cmp_104_, lean_object* v_h_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_mathlib_linearOrderOfCompares___redArg(v_inst_103_, v_cmp_104_);
return v___x_106_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Compare(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Compare(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Compare(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Compare(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Compare(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Compare(builtin);
}
#ifdef __cplusplus
}
#endif
