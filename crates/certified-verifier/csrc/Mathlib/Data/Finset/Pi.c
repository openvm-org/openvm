// Lean compiler output
// Module: Mathlib.Data.Finset.Pi
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Union public import Mathlib.Data.Multiset.Pi public import Mathlib.Logic.Function.DependsOn
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
lean_object* lp_mathlib_Multiset_Pi_cons___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_pi___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_empty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_empty___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_pi___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_pi___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_piDiag___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_const___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Finset_piDiag___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_piDiag___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_piDiag___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_piDiag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_empty(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b2_2_, lean_object* v_a_3_, lean_object* v_h_4_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_empty___boxed(lean_object* v_00_u03b1_5_, lean_object* v_00_u03b2_6_, lean_object* v_a_7_, lean_object* v_h_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Finset_Pi_empty(v_00_u03b1_5_, v_00_u03b2_6_, v_a_7_, v_h_8_);
lean_dec(v_a_7_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_pi___redArg___lam__0(lean_object* v_t_10_, lean_object* v_a_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_1(v_t_10_, v_a_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_pi___redArg(lean_object* v_inst_13_, lean_object* v_s_14_, lean_object* v_t_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___x_17_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Finset_pi___redArg___lam__0), 2, 1);
lean_closure_set(v___f_16_, 0, v_t_15_);
v___x_17_ = lp_mathlib_Multiset_pi___redArg(v_inst_13_, v_s_14_, v___f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_pi(lean_object* v_00_u03b1_18_, lean_object* v_00_u03b2_19_, lean_object* v_inst_20_, lean_object* v_s_21_, lean_object* v_t_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_Finset_pi___redArg(v_inst_20_, v_s_21_, v_t_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons___redArg(lean_object* v_inst_24_, lean_object* v_a_25_, lean_object* v_b_26_, lean_object* v_f_27_, lean_object* v_a_x27_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Multiset_Pi_cons___redArg(v_inst_24_, v_a_25_, v_b_26_, v_f_27_, v_a_x27_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons___redArg___boxed(lean_object* v_inst_30_, lean_object* v_a_31_, lean_object* v_b_32_, lean_object* v_f_33_, lean_object* v_a_x27_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Finset_Pi_cons___redArg(v_inst_30_, v_a_31_, v_b_32_, v_f_33_, v_a_x27_34_);
lean_dec(v_b_32_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons(lean_object* v_00_u03b1_36_, lean_object* v_00_u03b4_37_, lean_object* v_inst_38_, lean_object* v_s_39_, lean_object* v_a_40_, lean_object* v_b_41_, lean_object* v_f_42_, lean_object* v_a_x27_43_, lean_object* v_h_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_Multiset_Pi_cons___redArg(v_inst_38_, v_a_40_, v_b_41_, v_f_42_, v_a_x27_43_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Pi_cons___boxed(lean_object* v_00_u03b1_46_, lean_object* v_00_u03b4_47_, lean_object* v_inst_48_, lean_object* v_s_49_, lean_object* v_a_50_, lean_object* v_b_51_, lean_object* v_f_52_, lean_object* v_a_x27_53_, lean_object* v_h_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Finset_Pi_cons(v_00_u03b1_46_, v_00_u03b4_47_, v_inst_48_, v_s_49_, v_a_50_, v_b_51_, v_f_52_, v_a_x27_53_, v_h_54_);
lean_dec(v_b_51_);
lean_dec(v_s_49_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_piDiag___redArg(lean_object* v_s_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib_Finset_piDiag___redArg___closed__0));
v___x_60_ = lp_mathlib_Finset_image___redArg(v_inst_58_, v___x_59_, v_s_57_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_piDiag(lean_object* v_00_u03b1_61_, lean_object* v_s_62_, lean_object* v_00_u03b9_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Finset_piDiag___redArg(v_s_62_, v_inst_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict___redArg(lean_object* v_f_66_, lean_object* v_x_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lean_apply_1(v_f_66_, v_x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict(lean_object* v_00_u03b9_69_, lean_object* v_00_u03c0_70_, lean_object* v_s_71_, lean_object* v_f_72_, lean_object* v_x_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_apply_1(v_f_72_, v_x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict___boxed(lean_object* v_00_u03b9_75_, lean_object* v_00_u03c0_76_, lean_object* v_s_77_, lean_object* v_f_78_, lean_object* v_x_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_Finset_restrict(v_00_u03b9_75_, v_00_u03c0_76_, v_s_77_, v_f_78_, v_x_79_);
lean_dec(v_s_77_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict_u2082___redArg(lean_object* v_f_81_, lean_object* v_i_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_apply_1(v_f_81_, v_i_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict_u2082(lean_object* v_00_u03b9_84_, lean_object* v_00_u03c0_85_, lean_object* v_s_86_, lean_object* v_t_87_, lean_object* v_hst_88_, lean_object* v_f_89_, lean_object* v_i_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_apply_1(v_f_89_, v_i_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_restrict_u2082___boxed(lean_object* v_00_u03b9_92_, lean_object* v_00_u03c0_93_, lean_object* v_s_94_, lean_object* v_t_95_, lean_object* v_hst_96_, lean_object* v_f_97_, lean_object* v_i_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Finset_restrict_u2082(v_00_u03b9_92_, v_00_u03c0_93_, v_s_94_, v_t_95_, v_hst_96_, v_f_97_, v_i_98_);
lean_dec(v_t_95_);
lean_dec(v_s_94_);
return v_res_99_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_DependsOn(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_DependsOn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_DependsOn(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_DependsOn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
