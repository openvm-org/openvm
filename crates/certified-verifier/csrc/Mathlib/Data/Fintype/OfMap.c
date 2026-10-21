// Lean compiler output
// Module: Mathlib.Data.Fintype.OfMap
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.Defs public import Mathlib.Data.Finset.Image
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
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofMultiset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofList(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofBijective___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofBijective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSurjective___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instEmpty;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instPEmpty;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofMultiset___redArg(lean_object* v_inst_1_, lean_object* v_s_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_List_dedup___redArg(v_inst_1_, v_s_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofMultiset(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_s_6_, lean_object* v_H_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_List_dedup___redArg(v_inst_5_, v_s_6_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofList___redArg(lean_object* v_inst_9_, lean_object* v_l_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_List_dedup___redArg(v_inst_9_, v_l_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofList(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_l_14_, lean_object* v_H_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_List_dedup___redArg(v_inst_13_, v_l_14_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofBijective___redArg(lean_object* v_inst_17_, lean_object* v_f_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Finset_map___redArg(v_f_18_, v_inst_17_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofBijective(lean_object* v_00_u03b1_20_, lean_object* v_00_u03b2_21_, lean_object* v_inst_22_, lean_object* v_f_23_, lean_object* v_H_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_Finset_map___redArg(v_f_23_, v_inst_22_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSurjective___redArg(lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_f_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Finset_image___redArg(v_inst_26_, v_f_28_, v_inst_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSurjective(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_f_34_, lean_object* v_H_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Finset_image___redArg(v_inst_32_, v_f_34_, v_inst_33_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofEquiv___redArg___lam__0(lean_object* v_f_37_, lean_object* v___y_38_){
_start:
{
lean_object* v_toFun_39_; lean_object* v___x_40_; 
v_toFun_39_ = lean_ctor_get(v_f_37_, 0);
lean_inc(v_toFun_39_);
lean_dec_ref(v_f_37_);
v___x_40_ = lean_apply_1(v_toFun_39_, v___y_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object* v_inst_41_, lean_object* v_f_42_){
_start:
{
lean_object* v___f_43_; lean_object* v___x_44_; 
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_ofEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_43_, 0, v_f_42_);
v___x_44_ = lp_mathlib_Finset_map___redArg(v___f_43_, v_inst_41_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofEquiv(lean_object* v_00_u03b2_45_, lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_f_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_47_, v_f_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSubsingleton___redArg(lean_object* v_a_50_){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = lean_box(0);
v___x_52_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_52_, 0, v_a_50_);
lean_ctor_set(v___x_52_, 1, v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofSubsingleton(lean_object* v_00_u03b1_53_, lean_object* v_a_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Fintype_ofSubsingleton___redArg(v_a_54_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofIsEmpty(lean_object* v_00_u03b1_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_box(0);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_Fintype_instEmpty(void){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_box(0);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_Fintype_instPEmpty(void){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_box(0);
return v___x_61_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Fintype_instEmpty = _init_lp_mathlib_Fintype_instEmpty();
lean_mark_persistent(lp_mathlib_Fintype_instEmpty);
lp_mathlib_Fintype_instPEmpty = _init_lp_mathlib_Fintype_instPEmpty();
lean_mark_persistent(lp_mathlib_Fintype_instPEmpty);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
}
#ifdef __cplusplus
}
#endif
