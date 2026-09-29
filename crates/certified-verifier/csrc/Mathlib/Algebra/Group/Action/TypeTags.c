// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.TypeTags
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Defs public import Mathlib.Algebra.Group.TypeTags.Basic
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
lean_object* lp_mathlib_Additive_toMul(lean_object*);
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
static lean_once_cell_t lp_mathlib_Additive_vadd___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Additive_vadd___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_vadd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_vadd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_vadd(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Multiplicative_smul___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiplicative_smul___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_smul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Additive_vadd___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_vadd___redArg___lam__0(lean_object* v_inst_2_, lean_object* v_a_3_, lean_object* v_x_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toFun_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Additive_vadd___redArg___lam__0___closed__0, &lp_mathlib_Additive_vadd___redArg___lam__0___closed__0_once, _init_lp_mathlib_Additive_vadd___redArg___lam__0___closed__0);
v_toFun_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toFun_6_);
v___x_7_ = lean_apply_1(v_toFun_6_, v_a_3_);
v___x_8_ = lean_apply_2(v_inst_2_, v___x_7_, v_x_4_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_vadd___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Additive_vadd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_vadd(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_Additive_vadd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_inst_13_);
return v___f_14_;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_smul___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_smul___redArg___lam__0(lean_object* v_inst_16_, lean_object* v_a_17_, lean_object* v_x_18_){
_start:
{
lean_object* v___x_19_; lean_object* v_toFun_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_19_ = lean_obj_once(&lp_mathlib_Multiplicative_smul___redArg___lam__0___closed__0, &lp_mathlib_Multiplicative_smul___redArg___lam__0___closed__0_once, _init_lp_mathlib_Multiplicative_smul___redArg___lam__0___closed__0);
v_toFun_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc(v_toFun_20_);
v___x_21_ = lean_apply_1(v_toFun_20_, v_a_17_);
v___x_22_ = lean_apply_2(v_inst_16_, v___x_21_, v_x_18_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_smul___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_smul(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_28_, 0, v_inst_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addAction___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Additive_vadd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_30_, 0, v_inst_29_);
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addAction(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Additive_vadd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_35_, 0, v_inst_34_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addAction___boxed(lean_object* v_00_u03b1_36_, lean_object* v_00_u03b2_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Additive_addAction(v_00_u03b1_36_, v_00_u03b2_37_, v_inst_38_, v_inst_39_);
lean_dec_ref(v_inst_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulAction___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulAction(lean_object* v_00_u03b1_43_, lean_object* v_00_u03b2_44_, lean_object* v_inst_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___f_47_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_47_, 0, v_inst_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulAction___boxed(lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Multiplicative_mulAction(v_00_u03b1_48_, v_00_u03b2_49_, v_inst_50_, v_inst_51_);
lean_dec_ref(v_inst_50_);
return v_res_52_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_TypeTags(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_TypeTags(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_TypeTags(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_TypeTags(builtin);
}
#ifdef __cplusplus
}
#endif
