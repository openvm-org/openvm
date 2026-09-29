// Lean compiler output
// Module: Batteries.Data.List.Perm
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias public import Batteries.Data.List.Count import Batteries.Util.ProofWanted
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
uint8_t lp_batteries_List_isSubperm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_List_countBefore___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_idxOfNth___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_decidableSubperm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_decidableSubperm___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_decidableSubperm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_decidableSubperm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Subperm_idxInj___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Subperm_idxInj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Perm_idxBij___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_Perm_idxBij(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_decidableSubperm___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = lp_batteries_List_isSubperm___redArg(v_inst_1_, v_x_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_decidableSubperm___redArg___boxed(lean_object* v_inst_5_, lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_batteries_List_decidableSubperm___redArg(v_inst_5_, v_x_6_, v_x_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_decidableSubperm(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_x_13_, lean_object* v_x_14_){
_start:
{
uint8_t v___x_15_; 
v___x_15_ = lp_batteries_List_isSubperm___redArg(v_inst_11_, v_x_13_, v_x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_decidableSubperm___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_x_19_, lean_object* v_x_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_batteries_List_decidableSubperm(v_00_u03b1_16_, v_inst_17_, v_inst_18_, v_x_19_, v_x_20_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___redArg(uint8_t v_c_23_, lean_object* v_h__1_24_, lean_object* v_h__2_25_){
_start:
{
if (v_c_23_ == 0)
{
lean_object* v___x_26_; lean_object* v___x_27_; 
lean_dec(v_h__1_24_);
v___x_26_ = lean_box(0);
v___x_27_ = lean_apply_1(v_h__2_25_, v___x_26_);
return v___x_27_;
}
else
{
lean_object* v___x_28_; lean_object* v___x_29_; 
lean_dec(v_h__2_25_);
v___x_28_ = lean_box(0);
v___x_29_ = lean_apply_1(v_h__1_24_, v___x_28_);
return v___x_29_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___redArg___boxed(lean_object* v_c_30_, lean_object* v_h__1_31_, lean_object* v_h__2_32_){
_start:
{
uint8_t v_c_24__boxed_33_; lean_object* v_res_34_; 
v_c_24__boxed_33_ = lean_unbox(v_c_30_);
v_res_34_ = lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___redArg(v_c_24__boxed_33_, v_h__1_31_, v_h__2_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter(lean_object* v_motive_35_, uint8_t v_c_36_, lean_object* v_h__1_37_, lean_object* v_h__2_38_){
_start:
{
if (v_c_36_ == 0)
{
lean_object* v___x_39_; lean_object* v___x_40_; 
lean_dec(v_h__1_37_);
v___x_39_ = lean_box(0);
v___x_40_ = lean_apply_1(v_h__2_38_, v___x_39_);
return v___x_40_;
}
else
{
lean_object* v___x_41_; lean_object* v___x_42_; 
lean_dec(v_h__2_38_);
v___x_41_ = lean_box(0);
v___x_42_ = lean_apply_1(v_h__1_37_, v___x_41_);
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter___boxed(lean_object* v_motive_43_, lean_object* v_c_44_, lean_object* v_h__1_45_, lean_object* v_h__2_46_){
_start:
{
uint8_t v_c_35__boxed_47_; lean_object* v_res_48_; 
v_c_35__boxed_47_ = lean_unbox(v_c_44_);
v_res_48_ = lp_batteries___private_Batteries_Data_List_Perm_0__cond_match__1_splitter(v_motive_43_, v_c_35__boxed_47_, v_h__1_45_, v_h__2_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Subperm_idxInj___redArg(lean_object* v_inst_49_, lean_object* v_xs_50_, lean_object* v_ys_51_, lean_object* v_i_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
lean_inc(v_i_52_);
v___x_53_ = l_List_get___redArg(v_xs_50_, v_i_52_);
lean_inc(v___x_53_);
lean_inc_ref(v_inst_49_);
v___x_54_ = lp_batteries_List_countBefore___redArg(v_inst_49_, v___x_53_, v_xs_50_, v_i_52_);
v___x_55_ = lp_batteries_List_idxOfNth___redArg(v_inst_49_, v___x_53_, v_ys_51_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Subperm_idxInj(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_xs_59_, lean_object* v_ys_60_, lean_object* v_h_61_, lean_object* v_i_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_batteries_List_Subperm_idxInj___redArg(v_inst_57_, v_xs_59_, v_ys_60_, v_i_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Perm_idxBij___redArg(lean_object* v_inst_64_, lean_object* v_xs_65_, lean_object* v_ys_66_, lean_object* v_i_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_batteries_List_Subperm_idxInj___redArg(v_inst_64_, v_xs_65_, v_ys_66_, v_i_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_Perm_idxBij(lean_object* v_00_u03b1_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_xs_72_, lean_object* v_ys_73_, lean_object* v_h_74_, lean_object* v_i_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_batteries_List_Subperm_idxInj___redArg(v_inst_70_, v_xs_72_, v_ys_73_, v_i_75_);
return v___x_76_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Count(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_ProofWanted(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_ProofWanted(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Count(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_ProofWanted(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_List_Perm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_ProofWanted(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_List_Perm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_List_Perm(builtin);
}
#ifdef __cplusplus
}
#endif
