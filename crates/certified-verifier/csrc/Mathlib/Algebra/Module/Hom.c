// Lean compiler output
// Module: Mathlib.Algebra.Module.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Instances public import Mathlib.Algebra.GroupWithZero.Action.End public import Mathlib.Algebra.GroupWithZero.Action.Hom public import Mathlib.Algebra.Module.End public import Mathlib.Algebra.Ring.Opposite public import Mathlib.GroupTheory.GroupAction.DomAct.Basic
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
lean_object* lp_mathlib_ZeroHom_instSMulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_End_applyDistribMulAction___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instDomMulActModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instDomMulActModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instDomMulActModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instModule___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddMonoid_End_applyModule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoid_End_applyDistribMulAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoid_End_applyModule___closed__0 = (const lean_object*)&lp_mathlib_AddMonoid_End_applyModule___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_applyModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_applyModule___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instModule___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___f_2_; 
v___f_2_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instSMulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2_, 0, v_inst_1_);
return v___f_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instModule(lean_object* v_R_3_, lean_object* v_A_4_, lean_object* v_B_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instSMulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instModule___boxed(lean_object* v_R_11_, lean_object* v_A_12_, lean_object* v_B_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_ZeroHom_instModule(v_R_11_, v_A_12_, v_B_13_, v_inst_14_, v_inst_15_, v_inst_16_, v_inst_17_);
lean_dec_ref(v_inst_16_);
lean_dec_ref(v_inst_15_);
lean_dec_ref(v_inst_14_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instModule___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instSMulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instModule(lean_object* v_R_21_, lean_object* v_A_22_, lean_object* v_B_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instSMulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_28_, 0, v_inst_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instModule___boxed(lean_object* v_R_29_, lean_object* v_A_30_, lean_object* v_B_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_AddMonoidHom_instModule(v_R_29_, v_A_30_, v_B_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_inst_35_);
lean_dec_ref(v_inst_34_);
lean_dec_ref(v_inst_33_);
lean_dec_ref(v_inst_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instDomMulActModule___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_38_, 0, v_inst_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instDomMulActModule(lean_object* v_S_39_, lean_object* v_M_40_, lean_object* v_M_u2082_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___f_46_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_DomMulAct_instSMulAddMonoidHom___redArg___lam__0), 4, 1);
lean_closure_set(v___f_46_, 0, v_inst_45_);
return v___f_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instDomMulActModule___boxed(lean_object* v_S_47_, lean_object* v_M_48_, lean_object* v_M_u2082_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_AddMonoidHom_instDomMulActModule(v_S_47_, v_M_48_, v_M_u2082_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_inst_53_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
lean_dec_ref(v_inst_50_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___redArg___lam__0(lean_object* v_f_55_, lean_object* v_inst_56_, lean_object* v_r_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_apply_1(v_f_55_, v_a_58_);
v___x_60_ = lean_apply_2(v_inst_56_, v_r_57_, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___redArg(lean_object* v_inst_61_, lean_object* v_r_62_, lean_object* v_f_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___redArg___lam__0), 4, 3);
lean_closure_set(v___f_64_, 0, v_f_63_);
lean_closure_set(v___f_64_, 1, v_inst_61_);
lean_closure_set(v___f_64_, 2, v_r_62_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1(lean_object* v_M_65_, lean_object* v_A_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_r_69_, lean_object* v_f_70_){
_start:
{
lean_object* v___f_71_; 
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___redArg___lam__0), 4, 3);
lean_closure_set(v___f_71_, 0, v_f_70_);
lean_closure_set(v___f_71_, 1, v_inst_68_);
lean_closure_set(v___f_71_, 2, v_r_69_);
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed(lean_object* v_M_72_, lean_object* v_A_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_r_76_, lean_object* v_f_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_AddMonoid_End_instDistribSMul___aux__1(v_M_72_, v_A_73_, v_inst_74_, v_inst_75_, v_r_76_, v_f_77_);
lean_dec_ref(v_inst_74_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed), 6, 4);
lean_closure_set(v___x_81_, 0, lean_box(0));
lean_closure_set(v___x_81_, 1, lean_box(0));
lean_closure_set(v___x_81_, 2, v_inst_79_);
lean_closure_set(v___x_81_, 3, v_inst_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribSMul(lean_object* v_M_82_, lean_object* v_A_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed), 6, 4);
lean_closure_set(v___x_86_, 0, lean_box(0));
lean_closure_set(v___x_86_, 1, lean_box(0));
lean_closure_set(v___x_86_, 2, v_inst_84_);
lean_closure_set(v___x_86_, 3, v_inst_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribMulAction___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed), 6, 4);
lean_closure_set(v___x_89_, 0, lean_box(0));
lean_closure_set(v___x_89_, 1, lean_box(0));
lean_closure_set(v___x_89_, 2, v_inst_87_);
lean_closure_set(v___x_89_, 3, v_inst_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribMulAction(lean_object* v_R_90_, lean_object* v_A_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed), 6, 4);
lean_closure_set(v___x_95_, 0, lean_box(0));
lean_closure_set(v___x_95_, 1, lean_box(0));
lean_closure_set(v___x_95_, 2, v_inst_93_);
lean_closure_set(v___x_95_, 3, v_inst_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instDistribMulAction___boxed(lean_object* v_R_96_, lean_object* v_A_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_AddMonoid_End_instDistribMulAction(v_R_96_, v_A_97_, v_inst_98_, v_inst_99_, v_inst_100_);
lean_dec_ref(v_inst_98_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instModule___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed), 6, 4);
lean_closure_set(v___x_104_, 0, lean_box(0));
lean_closure_set(v___x_104_, 1, lean_box(0));
lean_closure_set(v___x_104_, 2, v_inst_102_);
lean_closure_set(v___x_104_, 3, v_inst_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instModule(lean_object* v_R_105_, lean_object* v_A_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instDistribSMul___aux__1___boxed), 6, 4);
lean_closure_set(v___x_110_, 0, lean_box(0));
lean_closure_set(v___x_110_, 1, lean_box(0));
lean_closure_set(v___x_110_, 2, v_inst_108_);
lean_closure_set(v___x_110_, 3, v_inst_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instModule___boxed(lean_object* v_R_111_, lean_object* v_A_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_AddMonoid_End_instModule(v_R_111_, v_A_112_, v_inst_113_, v_inst_114_, v_inst_115_);
lean_dec_ref(v_inst_113_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_applyModule(lean_object* v_A_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___f_120_; 
v___f_120_ = ((lean_object*)(lp_mathlib_AddMonoid_End_applyModule___closed__0));
return v___f_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_applyModule___boxed(lean_object* v_A_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_AddMonoid_End_applyModule(v_A_121_, v_inst_122_);
lean_dec_ref(v_inst_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul___redArg(lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(v_inst_124_, v_inst_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul___redArg___boxed(lean_object* v_inst_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_AddMonoidHom_smul___redArg(v_inst_127_, v_inst_128_);
lean_dec_ref(v_inst_127_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul(lean_object* v_R_130_, lean_object* v_M_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(v_inst_133_, v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_smul___boxed(lean_object* v_R_136_, lean_object* v_M_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_AddMonoidHom_smul(v_R_136_, v_M_137_, v_inst_138_, v_inst_139_, v_inst_140_);
lean_dec_ref(v_inst_139_);
lean_dec_ref(v_inst_138_);
return v_res_141_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_DomAct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
