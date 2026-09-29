// Lean compiler output
// Module: Mathlib.Algebra.MvPolynomial.Eval
// Imports: public import Init public meta import Init public import Mathlib.Algebra.MvPolynomial.Basic
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
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Finsupp_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finsupp_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082Hom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082Hom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MvPolynomial_eval___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MvPolynomial_eval___redArg___closed__0 = (const lean_object*)&lp_mathlib_MvPolynomial_eval___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aeval___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aeval(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082AlgHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082AlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__0(lean_object* v_g_1_, lean_object* v_toNPow_2_, lean_object* v_n_3_, lean_object* v_e_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_apply_1(v_g_1_, v_n_3_);
v___x_6_ = lean_apply_2(v_toNPow_2_, v_e_4_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__1(lean_object* v_f_7_, lean_object* v_toMonoid_8_, lean_object* v___f_9_, lean_object* v_toMul_10_, lean_object* v_s_11_, lean_object* v_a_12_){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_13_ = lean_apply_1(v_f_7_, v_a_12_);
v___x_14_ = lp_mathlib_Finsupp_prod___redArg(v_toMonoid_8_, v_s_11_, v___f_9_);
v___x_15_ = lean_apply_2(v_toMul_10_, v___x_13_, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__1___boxed(lean_object* v_f_16_, lean_object* v_toMonoid_17_, lean_object* v___f_18_, lean_object* v_toMul_19_, lean_object* v_s_20_, lean_object* v_a_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__1(v_f_16_, v_toMonoid_17_, v___f_18_, v_toMul_19_, v_s_20_, v_a_21_);
lean_dec_ref(v_toMonoid_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___redArg(lean_object* v_inst_23_, lean_object* v_f_24_, lean_object* v_g_25_, lean_object* v_p_26_){
_start:
{
lean_object* v_toAddCommMonoid_27_; lean_object* v_toMonoid_28_; lean_object* v___x_29_; lean_object* v_toMul_30_; lean_object* v_toNPow_31_; lean_object* v___f_32_; lean_object* v___f_33_; lean_object* v___x_34_; 
v_toAddCommMonoid_27_ = lean_ctor_get(v_inst_23_, 0);
lean_inc_ref(v_toAddCommMonoid_27_);
v_toMonoid_28_ = lean_ctor_get(v_inst_23_, 1);
lean_inc_ref(v_toMonoid_28_);
v___x_29_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_23_);
v_toMul_30_ = lean_ctor_get(v___x_29_, 0);
lean_inc(v_toMul_30_);
lean_dec_ref(v___x_29_);
v_toNPow_31_ = lean_ctor_get(v_toMonoid_28_, 2);
lean_inc(v_toNPow_31_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__0), 4, 2);
lean_closure_set(v___f_32_, 0, v_g_25_);
lean_closure_set(v___f_32_, 1, v_toNPow_31_);
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___redArg___lam__1___boxed), 6, 4);
lean_closure_set(v___f_33_, 0, v_f_24_);
lean_closure_set(v___f_33_, 1, v_toMonoid_28_);
lean_closure_set(v___f_33_, 2, v___f_32_);
lean_closure_set(v___f_33_, 3, v_toMul_30_);
v___x_34_ = lp_mathlib_Finsupp_sum___redArg(v_toAddCommMonoid_27_, v_p_26_, v___f_33_);
lean_dec_ref(v_toAddCommMonoid_27_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082(lean_object* v_R_35_, lean_object* v_S_u2081_36_, lean_object* v_00_u03c3_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_f_40_, lean_object* v_g_41_, lean_object* v_p_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_MvPolynomial_eval_u2082___redArg(v_inst_39_, v_f_40_, v_g_41_, v_p_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082___boxed(lean_object* v_R_44_, lean_object* v_S_u2081_45_, lean_object* v_00_u03c3_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_f_49_, lean_object* v_g_50_, lean_object* v_p_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_MvPolynomial_eval_u2082(v_R_44_, v_S_u2081_45_, v_00_u03c3_46_, v_inst_47_, v_inst_48_, v_f_49_, v_g_50_, v_p_51_);
lean_dec_ref(v_inst_47_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082Hom___redArg(lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_f_55_, lean_object* v_g_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___boxed), 8, 7);
lean_closure_set(v___x_57_, 0, lean_box(0));
lean_closure_set(v___x_57_, 1, lean_box(0));
lean_closure_set(v___x_57_, 2, lean_box(0));
lean_closure_set(v___x_57_, 3, v_inst_53_);
lean_closure_set(v___x_57_, 4, v_inst_54_);
lean_closure_set(v___x_57_, 5, v_f_55_);
lean_closure_set(v___x_57_, 6, v_g_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082Hom(lean_object* v_R_58_, lean_object* v_S_u2081_59_, lean_object* v_00_u03c3_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_f_63_, lean_object* v_g_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___boxed), 8, 7);
lean_closure_set(v___x_65_, 0, lean_box(0));
lean_closure_set(v___x_65_, 1, lean_box(0));
lean_closure_set(v___x_65_, 2, lean_box(0));
lean_closure_set(v___x_65_, 3, v_inst_61_);
lean_closure_set(v___x_65_, 4, v_inst_62_);
lean_closure_set(v___x_65_, 5, v_f_63_);
lean_closure_set(v___x_65_, 6, v_g_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval___redArg(lean_object* v_inst_67_, lean_object* v_f_68_){
_start:
{
lean_object* v___f_69_; lean_object* v___x_70_; 
v___f_69_ = ((lean_object*)(lp_mathlib_MvPolynomial_eval___redArg___closed__0));
lean_inc_ref(v_inst_67_);
v___x_70_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___boxed), 8, 7);
lean_closure_set(v___x_70_, 0, lean_box(0));
lean_closure_set(v___x_70_, 1, lean_box(0));
lean_closure_set(v___x_70_, 2, lean_box(0));
lean_closure_set(v___x_70_, 3, v_inst_67_);
lean_closure_set(v___x_70_, 4, v_inst_67_);
lean_closure_set(v___x_70_, 5, v___f_69_);
lean_closure_set(v___x_70_, 6, v_f_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval(lean_object* v_R_71_, lean_object* v_00_u03c3_72_, lean_object* v_inst_73_, lean_object* v_f_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_MvPolynomial_eval___redArg(v_inst_73_, v_f_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aeval___redArg(lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_f_79_){
_start:
{
lean_object* v_algebraMap_80_; lean_object* v___x_81_; 
v_algebraMap_80_ = lean_ctor_get(v_inst_78_, 1);
lean_inc(v_algebraMap_80_);
lean_dec_ref(v_inst_78_);
v___x_81_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___boxed), 8, 7);
lean_closure_set(v___x_81_, 0, lean_box(0));
lean_closure_set(v___x_81_, 1, lean_box(0));
lean_closure_set(v___x_81_, 2, lean_box(0));
lean_closure_set(v___x_81_, 3, v_inst_76_);
lean_closure_set(v___x_81_, 4, v_inst_77_);
lean_closure_set(v___x_81_, 5, v_algebraMap_80_);
lean_closure_set(v___x_81_, 6, v_f_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aeval(lean_object* v_R_82_, lean_object* v_S_u2081_83_, lean_object* v_00_u03c3_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_f_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_MvPolynomial_aeval___redArg(v_inst_85_, v_inst_86_, v_inst_87_, v_f_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082AlgHom___redArg(lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_g_93_){
_start:
{
lean_object* v_algebraMap_94_; lean_object* v___x_95_; 
v_algebraMap_94_ = lean_ctor_get(v_inst_92_, 1);
lean_inc(v_algebraMap_94_);
lean_dec_ref(v_inst_92_);
v___x_95_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___boxed), 8, 7);
lean_closure_set(v___x_95_, 0, lean_box(0));
lean_closure_set(v___x_95_, 1, lean_box(0));
lean_closure_set(v___x_95_, 2, lean_box(0));
lean_closure_set(v___x_95_, 3, v_inst_90_);
lean_closure_set(v___x_95_, 4, v_inst_91_);
lean_closure_set(v___x_95_, 5, v_algebraMap_94_);
lean_closure_set(v___x_95_, 6, v_g_93_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_eval_u2082AlgHom(lean_object* v_R_96_, lean_object* v_S_u2081_97_, lean_object* v_00_u03c3_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_g_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_MvPolynomial_eval_u2082AlgHom___redArg(v_inst_99_, v_inst_100_, v_inst_101_, v_g_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower___redArg___lam__0(lean_object* v_f_104_, lean_object* v___y_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lean_apply_1(v_f_104_, v___y_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower___redArg(lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_f_109_, lean_object* v_X_110_){
_start:
{
lean_object* v___f_111_; lean_object* v___x_112_; 
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_aevalTower___redArg___lam__0), 2, 1);
lean_closure_set(v___f_111_, 0, v_f_109_);
v___x_112_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_eval_u2082___boxed), 8, 7);
lean_closure_set(v___x_112_, 0, lean_box(0));
lean_closure_set(v___x_112_, 1, lean_box(0));
lean_closure_set(v___x_112_, 2, lean_box(0));
lean_closure_set(v___x_112_, 3, v_inst_107_);
lean_closure_set(v___x_112_, 4, v_inst_108_);
lean_closure_set(v___x_112_, 5, v___f_111_);
lean_closure_set(v___x_112_, 6, v_X_110_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower(lean_object* v_R_113_, lean_object* v_00_u03c3_114_, lean_object* v_inst_115_, lean_object* v_S_116_, lean_object* v_A_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_f_122_, lean_object* v_X_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_MvPolynomial_aevalTower___redArg(v_inst_115_, v_inst_119_, v_f_122_, v_X_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_aevalTower___boxed(lean_object* v_R_125_, lean_object* v_00_u03c3_126_, lean_object* v_inst_127_, lean_object* v_S_128_, lean_object* v_A_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_f_134_, lean_object* v_X_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_MvPolynomial_aevalTower(v_R_125_, v_00_u03c3_126_, v_inst_127_, v_S_128_, v_A_129_, v_inst_130_, v_inst_131_, v_inst_132_, v_inst_133_, v_f_134_, v_X_135_);
lean_dec_ref(v_inst_133_);
lean_dec_ref(v_inst_132_);
lean_dec_ref(v_inst_130_);
return v_res_136_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MvPolynomial_Eval(builtin);
}
#ifdef __cplusplus
}
#endif
