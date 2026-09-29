// Lean compiler output
// Module: Mathlib.Data.Set.Insert
// Imports: public import Init public meta import Init public import Aesop public import Mathlib.Data.Set.Disjoint public import Mathlib.Tactic.Simproc.ExistsAndEq
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
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___aux__1___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___aux__1(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_y_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
lean_inc(v_y_3_);
v___x_4_ = lean_apply_2(v_inst_1_, v_y_3_, v_x_2_);
v___x_5_ = lean_unbox(v___x_4_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; 
v___x_6_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6_, 0, v_y_3_);
return v___x_6_;
}
else
{
lean_object* v___x_7_; 
lean_dec(v_y_3_);
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1(lean_object* v_x_8_, lean_object* v_y_9_){
_start:
{
if (lean_obj_tag(v_y_9_) == 0)
{
lean_inc(v_x_8_);
return v_x_8_;
}
else
{
lean_object* v_val_10_; 
v_val_10_ = lean_ctor_get(v_y_9_, 0);
lean_inc(v_val_10_);
return v_val_10_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1___boxed(lean_object* v_x_11_, lean_object* v_y_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1(v_x_11_, v_y_12_);
lean_dec(v_y_12_);
lean_dec(v_x_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption___redArg(lean_object* v_inst_14_, lean_object* v_x_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___x_18_; 
lean_inc(v_x_15_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__0), 3, 2);
lean_closure_set(v___f_16_, 0, v_inst_14_);
lean_closure_set(v___f_16_, 1, v_x_15_);
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_Set_subtypeInsertEquivOption___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_17_, 0, v_x_15_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v___f_16_);
lean_ctor_set(v___x_18_, 1, v___f_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtypeInsertEquivOption(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_t_21_, lean_object* v_x_22_, lean_object* v_h_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Set_subtypeInsertEquivOption___redArg(v_inst_20_, v_x_22_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg(lean_object* v_a_25_){
_start:
{
lean_inc(v_a_25_);
return v_a_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___redArg___boxed(lean_object* v_a_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Set_uniqueSingleton___redArg(v_a_26_);
lean_dec(v_a_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton(lean_object* v_00_u03b1_28_, lean_object* v_a_29_){
_start:
{
lean_inc(v_a_29_);
return v_a_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_uniqueSingleton___boxed(lean_object* v_00_u03b1_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Set_uniqueSingleton(v_00_u03b1_30_, v_a_31_);
lean_dec(v_a_31_);
return v_res_32_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___aux__1___redArg(uint8_t v_inst_33_){
_start:
{
return v_inst_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___aux__1___redArg___boxed(lean_object* v_inst_34_){
_start:
{
uint8_t v_inst_5__boxed_35_; uint8_t v_res_36_; lean_object* v_r_37_; 
v_inst_5__boxed_35_ = lean_unbox(v_inst_34_);
v_res_36_ = lp_mathlib_Set_decidableSingleton___aux__1___redArg(v_inst_5__boxed_35_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___aux__1(lean_object* v_00_u03b1_38_, lean_object* v_a_39_, lean_object* v_b_40_, uint8_t v_inst_41_){
_start:
{
return v_inst_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___aux__1___boxed(lean_object* v_00_u03b1_42_, lean_object* v_a_43_, lean_object* v_b_44_, lean_object* v_inst_45_){
_start:
{
uint8_t v_inst_8__boxed_46_; uint8_t v_res_47_; lean_object* v_r_48_; 
v_inst_8__boxed_46_ = lean_unbox(v_inst_45_);
v_res_47_ = lp_mathlib_Set_decidableSingleton___aux__1(v_00_u03b1_42_, v_a_43_, v_b_44_, v_inst_8__boxed_46_);
lean_dec(v_b_44_);
lean_dec(v_a_43_);
v_r_48_ = lean_box(v_res_47_);
return v_r_48_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton___redArg(uint8_t v_inst_49_){
_start:
{
return v_inst_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___redArg___boxed(lean_object* v_inst_50_){
_start:
{
uint8_t v_inst_6__boxed_51_; uint8_t v_res_52_; lean_object* v_r_53_; 
v_inst_6__boxed_51_ = lean_unbox(v_inst_50_);
v_res_52_ = lp_mathlib_Set_decidableSingleton___redArg(v_inst_6__boxed_51_);
v_r_53_ = lean_box(v_res_52_);
return v_r_53_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableSingleton(lean_object* v_00_u03b1_54_, lean_object* v_a_55_, lean_object* v_b_56_, uint8_t v_inst_57_){
_start:
{
return v_inst_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableSingleton___boxed(lean_object* v_00_u03b1_58_, lean_object* v_a_59_, lean_object* v_b_60_, lean_object* v_inst_61_){
_start:
{
uint8_t v_inst_9__boxed_62_; uint8_t v_res_63_; lean_object* v_r_64_; 
v_inst_9__boxed_62_ = lean_unbox(v_inst_61_);
v_res_63_ = lp_mathlib_Set_decidableSingleton(v_00_u03b1_58_, v_a_59_, v_b_60_, v_inst_9__boxed_62_);
lean_dec(v_b_60_);
lean_dec(v_a_59_);
v_r_64_ = lean_box(v_res_63_);
return v_r_64_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Disjoint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Disjoint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
}
#ifdef __cplusplus
}
#endif
