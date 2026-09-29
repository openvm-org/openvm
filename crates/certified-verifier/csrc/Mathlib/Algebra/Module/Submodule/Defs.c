// Lean compiler output
// Module: Mathlib.Algebra.Module.Submodule.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Defs public import Mathlib.GroupTheory.GroupAction.SubMulAction public import Mathlib.Algebra.Group.Submonoid.Basic
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_SetLike_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_setLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_setLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Submodule_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofLinearComb(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofLinearComb___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toAddSubgroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubMulAction(lean_object* v_R_1_, lean_object* v_M_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_self_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toSubMulAction___boxed(lean_object* v_R_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_self_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Submodule_toSubMulAction(v_R_8_, v_M_9_, v_inst_10_, v_inst_11_, v_inst_12_, v_self_13_);
lean_dec(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_setLike(lean_object* v_R_15_, lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_box(0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_setLike___boxed(lean_object* v_R_21_, lean_object* v_M_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Submodule_setLike(v_R_21_, v_M_22_, v_inst_23_, v_inst_24_, v_inst_25_);
lean_dec(v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
return v_res_26_;
}
}
static lean_object* _init_lp_mathlib_Submodule_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = lean_box(0);
v___x_28_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPartialOrder(lean_object* v_R_29_, lean_object* v_M_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lean_obj_once(&lp_mathlib_Submodule_instPartialOrder___closed__0, &lp_mathlib_Submodule_instPartialOrder___closed__0_once, _init_lp_mathlib_Submodule_instPartialOrder___closed__0);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instPartialOrder___boxed(lean_object* v_R_35_, lean_object* v_M_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Submodule_instPartialOrder(v_R_35_, v_M_36_, v_inst_37_, v_inst_38_, v_inst_39_);
lean_dec(v_inst_39_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofClass(lean_object* v_S_41_, lean_object* v_R_42_, lean_object* v_M_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_s_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lean_box(0);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofClass___boxed(lean_object* v_S_52_, lean_object* v_R_53_, lean_object* v_M_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_s_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_Submodule_ofClass(v_S_52_, v_R_53_, v_M_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_inst_59_, v_inst_60_, v_s_61_);
lean_dec(v_s_61_);
lean_dec(v_inst_57_);
lean_dec_ref(v_inst_56_);
lean_dec_ref(v_inst_55_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofLinearComb(lean_object* v_R_63_, lean_object* v_M_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_C_68_, lean_object* v_nonempty_69_, lean_object* v_linearComb_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_box(0);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_ofLinearComb___boxed(lean_object* v_R_72_, lean_object* v_M_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_C_77_, lean_object* v_nonempty_78_, lean_object* v_linearComb_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_Submodule_ofLinearComb(v_R_72_, v_M_73_, v_inst_74_, v_inst_75_, v_inst_76_, v_C_77_, v_nonempty_78_, v_linearComb_79_);
lean_dec(v_inst_76_);
lean_dec_ref(v_inst_75_);
lean_dec_ref(v_inst_74_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_copy(lean_object* v_R_81_, lean_object* v_M_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_p_86_, lean_object* v_s_87_, lean_object* v_hs_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_box(0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_copy___boxed(lean_object* v_R_90_, lean_object* v_M_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_p_95_, lean_object* v_s_96_, lean_object* v_hs_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Submodule_copy(v_R_90_, v_M_91_, v_inst_92_, v_inst_93_, v_inst_94_, v_p_95_, v_s_96_, v_hs_97_);
lean_dec(v_inst_94_);
lean_dec_ref(v_inst_93_);
lean_dec_ref(v_inst_92_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule___redArg(lean_object* v_inst_99_){
_start:
{
lean_object* v___f_100_; 
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_100_, 0, v_inst_99_);
return v___f_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule(lean_object* v_R_101_, lean_object* v_M_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_A_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_S_x27_110_){
_start:
{
lean_object* v___f_111_; 
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_111_, 0, v_inst_105_);
return v___f_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule___boxed(lean_object* v_R_112_, lean_object* v_M_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_A_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_S_x27_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_SMulMemClass_toModule(v_R_112_, v_M_113_, v_inst_114_, v_inst_115_, v_inst_116_, v_A_117_, v_inst_118_, v_inst_119_, v_inst_120_, v_S_x27_121_);
lean_dec(v_S_x27_121_);
lean_dec_ref(v_inst_115_);
lean_dec_ref(v_inst_114_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule_x27___redArg(lean_object* v_inst_123_){
_start:
{
lean_object* v___f_124_; 
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_124_, 0, v_inst_123_);
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule_x27(lean_object* v_S_125_, lean_object* v_R_x27_126_, lean_object* v_R_127_, lean_object* v_A_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_s_139_){
_start:
{
lean_object* v___f_140_; 
v___f_140_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_140_, 0, v_inst_134_);
return v___f_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulMemClass_toModule_x27___boxed(lean_object* v_S_141_, lean_object* v_R_x27_142_, lean_object* v_R_143_, lean_object* v_A_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_s_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_SMulMemClass_toModule_x27(v_S_141_, v_R_x27_142_, v_R_143_, v_A_144_, v_inst_145_, v_inst_146_, v_inst_147_, v_inst_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_inst_152_, v_inst_153_, v_inst_154_, v_s_155_);
lean_dec(v_s_155_);
lean_dec(v_inst_149_);
lean_dec_ref(v_inst_148_);
lean_dec(v_inst_147_);
lean_dec_ref(v_inst_146_);
lean_dec_ref(v_inst_145_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add___redArg___lam__0(lean_object* v_toAdd_157_, lean_object* v_x_158_, lean_object* v_y_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_apply_2(v_toAdd_157_, v_x_158_, v_y_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add___redArg(lean_object* v_inst_161_){
_start:
{
lean_object* v_toAdd_162_; lean_object* v___f_163_; 
v_toAdd_162_ = lean_ctor_get(v_inst_161_, 1);
lean_inc(v_toAdd_162_);
lean_dec_ref(v_inst_161_);
v___f_163_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_add___redArg___lam__0), 3, 1);
lean_closure_set(v___f_163_, 0, v_toAdd_162_);
return v___f_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add(lean_object* v_R_164_, lean_object* v_M_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_module__M_168_, lean_object* v_p_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_Submodule_add___redArg(v_inst_167_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_add___boxed(lean_object* v_R_171_, lean_object* v_M_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_module__M_175_, lean_object* v_p_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib_Submodule_add(v_R_171_, v_M_172_, v_inst_173_, v_inst_174_, v_module__M_175_, v_p_176_);
lean_dec(v_module__M_175_);
lean_dec_ref(v_inst_173_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero___redArg(lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v_toZero_181_; 
v___x_179_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_178_);
v___x_180_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_179_);
v_toZero_181_ = lean_ctor_get(v___x_180_, 0);
lean_inc(v_toZero_181_);
lean_dec_ref(v___x_180_);
return v_toZero_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero___redArg___boxed(lean_object* v_inst_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_Submodule_zero___redArg(v_inst_182_);
lean_dec_ref(v_inst_182_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero(lean_object* v_R_184_, lean_object* v_M_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_module__M_188_, lean_object* v_p_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_Submodule_zero___redArg(v_inst_187_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_zero___boxed(lean_object* v_R_191_, lean_object* v_M_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_module__M_195_, lean_object* v_p_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Submodule_zero(v_R_191_, v_M_192_, v_inst_193_, v_inst_194_, v_module__M_195_, v_p_196_);
lean_dec(v_module__M_195_);
lean_dec_ref(v_inst_194_);
lean_dec_ref(v_inst_193_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited___redArg(lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v_toZero_201_; 
v___x_199_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_198_);
v___x_200_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_199_);
v_toZero_201_ = lean_ctor_get(v___x_200_, 0);
lean_inc(v_toZero_201_);
lean_dec_ref(v___x_200_);
return v_toZero_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited___redArg___boxed(lean_object* v_inst_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_Submodule_inhabited___redArg(v_inst_202_);
lean_dec_ref(v_inst_202_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited(lean_object* v_R_204_, lean_object* v_M_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_module__M_208_, lean_object* v_p_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_Submodule_inhabited___redArg(v_inst_207_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited___boxed(lean_object* v_R_211_, lean_object* v_M_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_module__M_215_, lean_object* v_p_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Submodule_inhabited(v_R_211_, v_M_212_, v_inst_213_, v_inst_214_, v_module__M_215_, v_p_216_);
lean_dec(v_module__M_215_);
lean_dec_ref(v_inst_214_);
lean_dec_ref(v_inst_213_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul___redArg___lam__0(lean_object* v_inst_218_, lean_object* v_c_219_, lean_object* v_x_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_apply_2(v_inst_218_, v_c_219_, v_x_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul___redArg(lean_object* v_inst_222_){
_start:
{
lean_object* v___f_223_; 
v___f_223_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_223_, 0, v_inst_222_);
return v___f_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul(lean_object* v_S_224_, lean_object* v_R_225_, lean_object* v_M_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_module__M_229_, lean_object* v_p_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_){
_start:
{
lean_object* v___f_234_; 
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_234_, 0, v_inst_232_);
return v___f_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_smul___boxed(lean_object* v_S_235_, lean_object* v_R_236_, lean_object* v_M_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_module__M_240_, lean_object* v_p_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_Submodule_smul(v_S_235_, v_R_236_, v_M_237_, v_inst_238_, v_inst_239_, v_module__M_240_, v_p_241_, v_inst_242_, v_inst_243_, v_inst_244_);
lean_dec(v_inst_242_);
lean_dec(v_module__M_240_);
lean_dec_ref(v_inst_239_);
lean_dec_ref(v_inst_238_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommMonoid___redArg(lean_object* v_inst_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommMonoid(lean_object* v_R_248_, lean_object* v_M_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_module__M_252_, lean_object* v_p_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_251_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommMonoid___boxed(lean_object* v_R_255_, lean_object* v_M_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_module__M_259_, lean_object* v_p_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Submodule_addCommMonoid(v_R_255_, v_M_256_, v_inst_257_, v_inst_258_, v_module__M_259_, v_p_260_);
lean_dec(v_module__M_259_);
lean_dec_ref(v_inst_257_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module_x27___redArg(lean_object* v_inst_262_){
_start:
{
lean_object* v___f_263_; 
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_263_, 0, v_inst_262_);
return v___f_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module_x27(lean_object* v_S_264_, lean_object* v_R_265_, lean_object* v_M_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_module__M_269_, lean_object* v_p_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v___f_275_; 
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_275_, 0, v_inst_273_);
return v___f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module_x27___boxed(lean_object* v_S_276_, lean_object* v_R_277_, lean_object* v_M_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_module__M_281_, lean_object* v_p_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Submodule_module_x27(v_S_276_, v_R_277_, v_M_278_, v_inst_279_, v_inst_280_, v_module__M_281_, v_p_282_, v_inst_283_, v_inst_284_, v_inst_285_, v_inst_286_);
lean_dec(v_inst_284_);
lean_dec_ref(v_inst_283_);
lean_dec(v_module__M_281_);
lean_dec_ref(v_inst_280_);
lean_dec_ref(v_inst_279_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module___redArg(lean_object* v_module__M_288_){
_start:
{
lean_object* v___f_289_; 
v___f_289_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_289_, 0, v_module__M_288_);
return v___f_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module(lean_object* v_R_290_, lean_object* v_M_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_module__M_294_, lean_object* v_p_295_){
_start:
{
lean_object* v___f_296_; 
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_296_, 0, v_module__M_294_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_module___boxed(lean_object* v_R_297_, lean_object* v_M_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_module__M_301_, lean_object* v_p_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_Submodule_module(v_R_297_, v_M_298_, v_inst_299_, v_inst_300_, v_module__M_301_, v_p_302_);
lean_dec_ref(v_inst_300_);
lean_dec_ref(v_inst_299_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toAddSubgroup___redArg(lean_object* v_p_304_){
_start:
{
return v_p_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toAddSubgroup(lean_object* v_R_305_, lean_object* v_M_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_module__M_309_, lean_object* v_p_310_){
_start:
{
return v_p_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_toAddSubgroup___boxed(lean_object* v_R_311_, lean_object* v_M_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_module__M_315_, lean_object* v_p_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_Submodule_toAddSubgroup(v_R_311_, v_M_312_, v_inst_313_, v_inst_314_, v_module__M_315_, v_p_316_);
lean_dec(v_module__M_315_);
lean_dec_ref(v_inst_314_);
lean_dec_ref(v_inst_313_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommGroup___redArg(lean_object* v_inst_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommGroup(lean_object* v_R_320_, lean_object* v_M_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_module__M_324_, lean_object* v_p_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_323_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_addCommGroup___boxed(lean_object* v_R_327_, lean_object* v_M_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_module__M_331_, lean_object* v_p_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_Submodule_addCommGroup(v_R_327_, v_M_328_, v_inst_329_, v_inst_330_, v_module__M_331_, v_p_332_);
lean_dec(v_module__M_331_);
lean_dec_ref(v_inst_329_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module_x27___redArg(lean_object* v_inst_334_){
_start:
{
lean_object* v___f_335_; 
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_335_, 0, v_inst_334_);
return v___f_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module_x27(lean_object* v_S_336_, lean_object* v_R_337_, lean_object* v_M_338_, lean_object* v_T_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_t_350_){
_start:
{
lean_object* v___f_351_; 
v___f_351_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_351_, 0, v_inst_345_);
return v___f_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module_x27___boxed(lean_object* v_S_352_, lean_object* v_R_353_, lean_object* v_M_354_, lean_object* v_T_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_t_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_SubmoduleClass_module_x27(v_S_352_, v_R_353_, v_M_354_, v_T_355_, v_inst_356_, v_inst_357_, v_inst_358_, v_inst_359_, v_inst_360_, v_inst_361_, v_inst_362_, v_inst_363_, v_inst_364_, v_inst_365_, v_t_366_);
lean_dec(v_t_366_);
lean_dec(v_inst_360_);
lean_dec(v_inst_359_);
lean_dec_ref(v_inst_358_);
lean_dec_ref(v_inst_357_);
lean_dec_ref(v_inst_356_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module___redArg(lean_object* v_inst_368_){
_start:
{
lean_object* v___f_369_; 
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_369_, 0, v_inst_368_);
return v___f_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module(lean_object* v_S_370_, lean_object* v_R_371_, lean_object* v_M_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_s_379_){
_start:
{
lean_object* v___f_380_; 
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_380_, 0, v_inst_375_);
return v___f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmoduleClass_module___boxed(lean_object* v_S_381_, lean_object* v_R_382_, lean_object* v_M_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_s_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_SubmoduleClass_module(v_S_381_, v_R_382_, v_M_383_, v_inst_384_, v_inst_385_, v_inst_386_, v_inst_387_, v_inst_388_, v_inst_389_, v_s_390_);
lean_dec(v_s_390_);
lean_dec_ref(v_inst_385_);
lean_dec_ref(v_inst_384_);
return v_res_391_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
