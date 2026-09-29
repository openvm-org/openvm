// Lean compiler output
// Module: Mathlib.GroupTheory.Coset.Card
// Imports: public import Init public meta import Init public import Mathlib.GroupTheory.Coset.Basic public import Mathlib.SetTheory.Cardinal.Finite
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
lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_fintype___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintype___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_box(0);
v___x_4_ = lp_mathlib_Quotient_fintype___redArg(v_inst_1_, v___x_3_, v_inst_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintype(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_s_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_QuotientGroup_fintype___redArg(v_inst_7_, v_inst_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintype___boxed(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_s_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_QuotientGroup_fintype(v_00_u03b1_11_, v_inst_12_, v_inst_13_, v_s_14_, v_inst_15_);
lean_dec_ref(v_inst_12_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintype___redArg(lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_box(0);
v___x_20_ = lp_mathlib_Quotient_fintype___redArg(v_inst_17_, v___x_19_, v_inst_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintype(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_s_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_QuotientAddGroup_fintype___redArg(v_inst_23_, v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintype___boxed(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_s_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_QuotientAddGroup_fintype(v_00_u03b1_27_, v_inst_28_, v_inst_29_, v_s_30_, v_inst_31_);
lean_dec_ref(v_inst_28_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel___redArg(lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_35_ = lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg(v_inst_33_);
v___x_36_ = lp_mathlib_Equiv_symm___redArg(v___x_35_);
v___x_37_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_34_, v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel___redArg___boxed(lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_QuotientGroup_fintypeQuotientRightRel___redArg(v_inst_38_, v_inst_39_);
lean_dec_ref(v_inst_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_s_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_QuotientGroup_fintypeQuotientRightRel___redArg(v_inst_42_, v_inst_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_fintypeQuotientRightRel___boxed(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_s_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_QuotientGroup_fintypeQuotientRightRel(v_00_u03b1_46_, v_inst_47_, v_s_48_, v_inst_49_);
lean_dec_ref(v_inst_47_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___redArg(lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg(v_inst_51_);
v___x_54_ = lp_mathlib_Equiv_symm___redArg(v___x_53_);
v___x_55_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_52_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___redArg___boxed(lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___redArg(v_inst_56_, v_inst_57_);
lean_dec_ref(v_inst_56_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel(lean_object* v_00_u03b1_59_, lean_object* v_inst_60_, lean_object* v_s_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___redArg(v_inst_60_, v_inst_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel___boxed(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_s_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_QuotientAddGroup_fintypeQuotientRightRel(v_00_u03b1_64_, v_inst_65_, v_s_66_, v_inst_67_);
lean_dec_ref(v_inst_65_);
return v_res_68_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Card(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Coset_Card(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Card(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Coset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Coset_Card(builtin);
}
#ifdef __cplusplus
}
#endif
