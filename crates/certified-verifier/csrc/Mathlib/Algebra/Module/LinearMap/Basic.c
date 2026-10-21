// Lean compiler output
// Module: Mathlib.Algebra.Module.LinearMap.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.LinearMap.Defs public import Mathlib.Algebra.Module.Pi public import Mathlib.Algebra.Module.Torsion.Pi public import Mathlib.GroupTheory.GroupAction.DomAct.Basic
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
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_elim___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ltoFun___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_ltoFun___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_ltoFun___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_ltoFun___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_ltoFun___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ltoFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ltoFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ltoFun___lam__0(lean_object* v_f_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ltoFun(lean_object* v_R_5_, lean_object* v_M_6_, lean_object* v_N_7_, lean_object* v_A_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = ((lean_object*)(lp_mathlib_LinearMap_ltoFun___closed__0));
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ltoFun___boxed(lean_object* v_R_18_, lean_object* v_M_19_, lean_object* v_N_20_, lean_object* v_A_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_LinearMap_ltoFun(v_R_18_, v_M_19_, v_N_20_, v_A_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_inst_27_, v_inst_28_, v_inst_29_);
lean_dec(v_inst_28_);
lean_dec(v_inst_27_);
lean_dec_ref(v_inst_26_);
lean_dec(v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec_ref(v_inst_22_);
return v_res_30_;
}
}
static lean_object* _init_lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_obj_once(&lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__0, &lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__0_once, _init_lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__0);
v___x_33_ = lp_mathlib_Equiv_symm___redArg(v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0(lean_object* v_inst_34_, lean_object* v_a_35_, lean_object* v_f_36_, lean_object* v___y_37_){
_start:
{
lean_object* v___x_38_; lean_object* v_toFun_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_38_ = lean_obj_once(&lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__1, &lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__1_once, _init_lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0___closed__1);
v_toFun_39_ = lean_ctor_get(v___x_38_, 0);
lean_inc(v_toFun_39_);
v___x_40_ = lean_apply_1(v_toFun_39_, v_a_35_);
v___x_41_ = lean_apply_2(v_inst_34_, v___x_40_, v___y_37_);
v___x_42_ = lean_apply_1(v_f_36_, v___x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___redArg(lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; 
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0), 4, 1);
lean_closure_set(v___f_44_, 0, v_inst_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct(lean_object* v_R_45_, lean_object* v_R_x27_46_, lean_object* v_M_47_, lean_object* v_M_x27_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_00_u03c3_u2081_u2082_55_, lean_object* v_S_x27_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v___f_60_; 
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0), 4, 1);
lean_closure_set(v___f_60_, 0, v_inst_58_);
return v___f_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMulDomMulAct___boxed(lean_object* v_R_61_, lean_object* v_R_x27_62_, lean_object* v_M_63_, lean_object* v_M_x27_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_00_u03c3_u2081_u2082_71_, lean_object* v_S_x27_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_LinearMap_instSMulDomMulAct(v_R_61_, v_R_x27_62_, v_M_63_, v_M_x27_64_, v_inst_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_inst_69_, v_inst_70_, v_00_u03c3_u2081_u2082_71_, v_S_x27_72_, v_inst_73_, v_inst_74_, v_inst_75_);
lean_dec_ref(v_inst_73_);
lean_dec(v_00_u03c3_u2081_u2082_71_);
lean_dec(v_inst_70_);
lean_dec(v_inst_69_);
lean_dec_ref(v_inst_68_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_65_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass___redArg(lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0), 4, 1);
lean_closure_set(v___f_78_, 0, v_inst_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass(lean_object* v_R_79_, lean_object* v_R_x27_80_, lean_object* v_M_81_, lean_object* v_M_x27_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_00_u03c3_u2081_u2082_89_, lean_object* v_S_x27_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_){
_start:
{
lean_object* v___f_94_; 
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0), 4, 1);
lean_closure_set(v___f_94_, 0, v_inst_92_);
return v___f_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass___boxed(lean_object* v_R_95_, lean_object* v_R_x27_96_, lean_object* v_M_97_, lean_object* v_M_x27_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_00_u03c3_u2081_u2082_105_, lean_object* v_S_x27_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_LinearMap_instDistribMulActionDomMulActOfSMulCommClass(v_R_95_, v_R_x27_96_, v_M_97_, v_M_x27_98_, v_inst_99_, v_inst_100_, v_inst_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_00_u03c3_u2081_u2082_105_, v_S_x27_106_, v_inst_107_, v_inst_108_, v_inst_109_);
lean_dec_ref(v_inst_107_);
lean_dec(v_00_u03c3_u2081_u2082_105_);
lean_dec(v_inst_104_);
lean_dec(v_inst_103_);
lean_dec_ref(v_inst_102_);
lean_dec_ref(v_inst_101_);
lean_dec_ref(v_inst_100_);
lean_dec_ref(v_inst_99_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass___redArg(lean_object* v_inst_111_){
_start:
{
lean_object* v___f_112_; 
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0), 4, 1);
lean_closure_set(v___f_112_, 0, v_inst_111_);
return v___f_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass(lean_object* v_R_113_, lean_object* v_R_x27_114_, lean_object* v_S_115_, lean_object* v_M_116_, lean_object* v_M_x27_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_00_u03c3_u2081_u2082_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v___f_128_; 
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMulDomMulAct___redArg___lam__0), 4, 1);
lean_closure_set(v___f_128_, 0, v_inst_126_);
return v___f_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass___boxed(lean_object* v_R_129_, lean_object* v_R_x27_130_, lean_object* v_S_131_, lean_object* v_M_132_, lean_object* v_M_x27_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_00_u03c3_u2081_u2082_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_LinearMap_instModuleDomMulActOfSMulCommClass(v_R_129_, v_R_x27_130_, v_S_131_, v_M_132_, v_M_x27_133_, v_inst_134_, v_inst_135_, v_inst_136_, v_inst_137_, v_inst_138_, v_inst_139_, v_00_u03c3_u2081_u2082_140_, v_inst_141_, v_inst_142_, v_inst_143_);
lean_dec_ref(v_inst_141_);
lean_dec(v_00_u03c3_u2081_u2082_140_);
lean_dec(v_inst_139_);
lean_dec(v_inst_138_);
lean_dec_ref(v_inst_137_);
lean_dec_ref(v_inst_136_);
lean_dec_ref(v_inst_135_);
lean_dec_ref(v_inst_134_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft___redArg___lam__0(lean_object* v_toZero_145_, lean_object* v_x_146_){
_start:
{
lean_inc(v_toZero_145_);
return v_toZero_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft___redArg___lam__0___boxed(lean_object* v_toZero_147_, lean_object* v_x_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Sum_elimZeroLeft___redArg___lam__0(v_toZero_147_, v_x_148_);
lean_dec(v_x_148_);
lean_dec(v_toZero_147_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft___redArg(lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; lean_object* v_toZero_152_; lean_object* v___f_153_; lean_object* v___x_154_; 
v___x_151_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_150_);
v_toZero_152_ = lean_ctor_get(v___x_151_, 1);
lean_inc(v_toZero_152_);
lean_dec_ref(v___x_151_);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_Sum_elimZeroLeft___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_153_, 0, v_toZero_152_);
v___x_154_ = lean_alloc_closure((void*)(l_Sum_elim), 6, 4);
lean_closure_set(v___x_154_, 0, lean_box(0));
lean_closure_set(v___x_154_, 1, lean_box(0));
lean_closure_set(v___x_154_, 2, lean_box(0));
lean_closure_set(v___x_154_, 3, v___f_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroLeft(lean_object* v_00_u03b9_155_, lean_object* v_00_u03ba_156_, lean_object* v_R_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_Sum_elimZeroLeft___redArg(v_inst_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroRight___redArg___lam__1(lean_object* v___f_160_, lean_object* v_f_161_, lean_object* v___y_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = l_Sum_elim___redArg(v_f_161_, v___f_160_, v___y_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroRight___redArg(lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; lean_object* v_toZero_166_; lean_object* v___f_167_; lean_object* v___f_168_; 
v___x_165_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_164_);
v_toZero_166_ = lean_ctor_get(v___x_165_, 1);
lean_inc(v_toZero_166_);
lean_dec_ref(v___x_165_);
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_Sum_elimZeroLeft___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_167_, 0, v_toZero_166_);
v___f_168_ = lean_alloc_closure((void*)(lp_mathlib_Sum_elimZeroRight___redArg___lam__1), 3, 1);
lean_closure_set(v___f_168_, 0, v___f_167_);
return v___f_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_elimZeroRight(lean_object* v_00_u03b9_169_, lean_object* v_00_u03ba_170_, lean_object* v_R_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib_Sum_elimZeroRight___redArg(v_inst_172_);
return v___x_173_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
