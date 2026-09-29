// Lean compiler output
// Module: Mathlib.Data.Set.Restrict
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Image
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
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_codRestrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict___redArg(lean_object* v_f_1_, lean_object* v_x_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_1_, v_x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict(lean_object* v_00_u03b1_4_, lean_object* v_00_u03c0_5_, lean_object* v_s_6_, lean_object* v_f_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_apply_1(v_f_7_, v_x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict_u2082___redArg(lean_object* v_f_10_, lean_object* v_x_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_1(v_f_10_, v_x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_domRestrict_u2082(lean_object* v_00_u03b1_13_, lean_object* v_00_u03c0_14_, lean_object* v_s_15_, lean_object* v_t_16_, lean_object* v_hst_17_, lean_object* v_f_18_, lean_object* v_x_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_apply_1(v_f_18_, v_x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_codRestrict___redArg(lean_object* v_f_21_, lean_object* v_x_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_apply_1(v_f_21_, v_x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_codRestrict(lean_object* v_00_u03b1_24_, lean_object* v_00_u03b9_25_, lean_object* v_f_26_, lean_object* v_s_27_, lean_object* v_h_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_apply_1(v_f_26_, v_x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict___redArg(lean_object* v_f_31_, lean_object* v_a_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_apply_1(v_f_31_, v_a_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict(lean_object* v_00_u03b1_34_, lean_object* v_00_u03c0_35_, lean_object* v_s_36_, lean_object* v_f_37_, lean_object* v_a_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_apply_1(v_f_37_, v_a_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict_u2082___redArg(lean_object* v_f_40_, lean_object* v_a_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_apply_1(v_f_40_, v_a_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_restrict_u2082(lean_object* v_00_u03b1_43_, lean_object* v_00_u03c0_44_, lean_object* v_s_45_, lean_object* v_t_46_, lean_object* v_hst_47_, lean_object* v_f_48_, lean_object* v_a_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_apply_1(v_f_48_, v_a_49_);
return v___x_50_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Restrict(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Restrict(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Restrict(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Restrict(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Restrict(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Restrict(builtin);
}
#ifdef __cplusplus
}
#endif
