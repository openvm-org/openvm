// Lean compiler output
// Module: Mathlib.Data.ULift
// Imports: public import Init public meta import Init public import Mathlib.Control.ULift public import Mathlib.Logic.Equiv.Basic
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
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_plift(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_PLift_instUnique___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PLift_instUnique___redArg___closed__0;
static lean_once_cell_t lp_mathlib_PLift_instUnique___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PLift_instUnique___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_PLift_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PLift_instDecidableEq__mathlib___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___redArg___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_PLift_instDecidableEq__mathlib___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PLift_instDecidableEq__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ULift_instUnique___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_instUnique___redArg___closed__0;
static lean_once_cell_t lp_mathlib_ULift_instUnique___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_instUnique___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_ULift_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instUnique(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ULift_instDecidableEq__mathlib___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_instDecidableEq__mathlib___redArg___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_ULift_instDecidableEq__mathlib___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDecidableEq__mathlib___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ULift_instDecidableEq__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDecidableEq__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_PLift_instUnique___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_plift(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_PLift_instUnique___redArg___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_mathlib_PLift_instUnique___redArg___closed__0, &lp_mathlib_PLift_instUnique___redArg___closed__0_once, _init_lp_mathlib_PLift_instUnique___redArg___closed__0);
v___x_3_ = lp_mathlib_Equiv_symm___redArg(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instUnique___redArg(lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toFun_6_; lean_object* v___x_7_; 
v___x_5_ = lean_obj_once(&lp_mathlib_PLift_instUnique___redArg___closed__1, &lp_mathlib_PLift_instUnique___redArg___closed__1_once, _init_lp_mathlib_PLift_instUnique___redArg___closed__1);
v_toFun_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toFun_6_);
v___x_7_ = lean_apply_1(v_toFun_6_, v_inst_4_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instUnique(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_PLift_instUnique___redArg(v_inst_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___redArg___lam__0(lean_object* v___x_11_, lean_object* v___y_12_){
_start:
{
lean_object* v_toFun_13_; lean_object* v___x_14_; 
v_toFun_13_ = lean_ctor_get(v___x_11_, 0);
lean_inc(v_toFun_13_);
lean_dec_ref(v___x_11_);
v___x_14_ = lean_apply_1(v_toFun_13_, v___y_12_);
return v___x_14_;
}
}
static lean_object* _init_lp_mathlib_PLift_instDecidableEq__mathlib___redArg___closed__0(void){
_start:
{
lean_object* v___x_15_; lean_object* v___f_16_; 
v___x_15_ = lean_obj_once(&lp_mathlib_PLift_instUnique___redArg___closed__0, &lp_mathlib_PLift_instUnique___redArg___closed__0_once, _init_lp_mathlib_PLift_instUnique___redArg___closed__0);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_PLift_instDecidableEq__mathlib___redArg___lam__0), 2, 1);
lean_closure_set(v___f_16_, 0, v___x_15_);
return v___f_16_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PLift_instDecidableEq__mathlib___redArg(lean_object* v_inst_17_, lean_object* v_a_18_, lean_object* v_b_19_){
_start:
{
lean_object* v___f_20_; uint8_t v___x_21_; 
v___f_20_ = lean_obj_once(&lp_mathlib_PLift_instDecidableEq__mathlib___redArg___closed__0, &lp_mathlib_PLift_instDecidableEq__mathlib___redArg___closed__0_once, _init_lp_mathlib_PLift_instDecidableEq__mathlib___redArg___closed__0);
v___x_21_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_20_, v_inst_17_, v_a_18_, v_b_19_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___redArg___boxed(lean_object* v_inst_22_, lean_object* v_a_23_, lean_object* v_b_24_){
_start:
{
uint8_t v_res_25_; lean_object* v_r_26_; 
v_res_25_ = lp_mathlib_PLift_instDecidableEq__mathlib___redArg(v_inst_22_, v_a_23_, v_b_24_);
v_r_26_ = lean_box(v_res_25_);
return v_r_26_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PLift_instDecidableEq__mathlib(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_, lean_object* v_a_29_, lean_object* v_b_30_){
_start:
{
uint8_t v___x_31_; 
v___x_31_ = lp_mathlib_PLift_instDecidableEq__mathlib___redArg(v_inst_28_, v_a_29_, v_b_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_instDecidableEq__mathlib___boxed(lean_object* v_00_u03b1_32_, lean_object* v_inst_33_, lean_object* v_a_34_, lean_object* v_b_35_){
_start:
{
uint8_t v_res_36_; lean_object* v_r_37_; 
v_res_36_ = lp_mathlib_PLift_instDecidableEq__mathlib(v_00_u03b1_32_, v_inst_33_, v_a_34_, v_b_35_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
static lean_object* _init_lp_mathlib_ULift_instUnique___redArg___closed__0(void){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_ULift_instUnique___redArg___closed__1(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_39_ = lean_obj_once(&lp_mathlib_ULift_instUnique___redArg___closed__0, &lp_mathlib_ULift_instUnique___redArg___closed__0_once, _init_lp_mathlib_ULift_instUnique___redArg___closed__0);
v___x_40_ = lp_mathlib_Equiv_symm___redArg(v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instUnique___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; lean_object* v_toFun_43_; lean_object* v___x_44_; 
v___x_42_ = lean_obj_once(&lp_mathlib_ULift_instUnique___redArg___closed__1, &lp_mathlib_ULift_instUnique___redArg___closed__1_once, _init_lp_mathlib_ULift_instUnique___redArg___closed__1);
v_toFun_43_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_toFun_43_);
v___x_44_ = lean_apply_1(v_toFun_43_, v_inst_41_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instUnique(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_ULift_instUnique___redArg(v_inst_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_ULift_instDecidableEq__mathlib___redArg___closed__0(void){
_start:
{
lean_object* v___x_48_; lean_object* v___f_49_; 
v___x_48_ = lean_obj_once(&lp_mathlib_ULift_instUnique___redArg___closed__0, &lp_mathlib_ULift_instUnique___redArg___closed__0_once, _init_lp_mathlib_ULift_instUnique___redArg___closed__0);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_PLift_instDecidableEq__mathlib___redArg___lam__0), 2, 1);
lean_closure_set(v___f_49_, 0, v___x_48_);
return v___f_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instDecidableEq__mathlib___redArg(lean_object* v_inst_50_, lean_object* v_a_51_, lean_object* v_b_52_){
_start:
{
lean_object* v___f_53_; uint8_t v___x_54_; 
v___f_53_ = lean_obj_once(&lp_mathlib_ULift_instDecidableEq__mathlib___redArg___closed__0, &lp_mathlib_ULift_instDecidableEq__mathlib___redArg___closed__0_once, _init_lp_mathlib_ULift_instDecidableEq__mathlib___redArg___closed__0);
v___x_54_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_53_, v_inst_50_, v_a_51_, v_b_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDecidableEq__mathlib___redArg___boxed(lean_object* v_inst_55_, lean_object* v_a_56_, lean_object* v_b_57_){
_start:
{
uint8_t v_res_58_; lean_object* v_r_59_; 
v_res_58_ = lp_mathlib_ULift_instDecidableEq__mathlib___redArg(v_inst_55_, v_a_56_, v_b_57_);
v_r_59_ = lean_box(v_res_58_);
return v_r_59_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instDecidableEq__mathlib(lean_object* v_00_u03b1_60_, lean_object* v_inst_61_, lean_object* v_a_62_, lean_object* v_b_63_){
_start:
{
uint8_t v___x_64_; 
v___x_64_ = lp_mathlib_ULift_instDecidableEq__mathlib___redArg(v_inst_61_, v_a_62_, v_b_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDecidableEq__mathlib___boxed(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_a_67_, lean_object* v_b_68_){
_start:
{
uint8_t v_res_69_; lean_object* v_r_70_; 
v_res_69_ = lp_mathlib_ULift_instDecidableEq__mathlib(v_00_u03b1_65_, v_inst_66_, v_a_67_, v_b_68_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Control_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ULift(builtin);
}
#ifdef __cplusplus
}
#endif
