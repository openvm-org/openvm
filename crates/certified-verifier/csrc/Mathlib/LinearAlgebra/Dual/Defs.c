// Lean compiler output
// Module: Mathlib.LinearAlgebra.Dual.Defs
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.BilinearMap public import Mathlib.LinearAlgebra.Span.Defs public import Mathlib.Tactic.CrossRefAttribute
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
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_llcomp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_flip___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_domRestrict___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
static const lean_closure_object lp_mathlib_Module_dualPairing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Module_dualPairing___closed__0 = (const lean_object*)&lp_mathlib_Module_dualPairing___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Module_dualPairing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_dualPairing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Module_Dual_eval___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Dual_eval___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_eval(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Module_Dual_transpose___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Module_Dual_transpose___redArg___closed__0 = (const lean_object*)&lp_mathlib_Module_Dual_transpose___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_transpose___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_transpose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_dualMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_dualMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_dualMap___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_dualMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_dualMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_dualRestrict___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_domRestrict___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_dualRestrict___closed__0 = (const lean_object*)&lp_mathlib_Submodule_dualRestrict___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualAnnihilator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualAnnihilator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualCoannihilator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualCoannihilator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_dualPairing(lean_object* v_R_2_, lean_object* v_M_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = ((lean_object*)(lp_mathlib_Module_dualPairing___closed__0));
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_dualPairing___boxed(lean_object* v_R_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Module_dualPairing(v_R_8_, v_M_9_, v_inst_10_, v_inst_11_, v_inst_12_);
lean_dec(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg___lam__0(lean_object* v_toZero_14_, lean_object* v_x_15_){
_start:
{
lean_inc(v_toZero_14_);
return v_toZero_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg___lam__0___boxed(lean_object* v_toZero_16_, lean_object* v_x_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Module_Dual_instInhabited___redArg___lam__0(v_toZero_16_, v_x_17_);
lean_dec(v_x_17_);
lean_dec(v_toZero_16_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v_toAddCommMonoid_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v_toZero_23_; lean_object* v___f_24_; 
v_toAddCommMonoid_20_ = lean_ctor_get(v_inst_19_, 0);
v___x_21_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_20_);
v___x_22_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_21_);
v_toZero_23_ = lean_ctor_get(v___x_22_, 0);
lean_inc(v_toZero_23_);
lean_dec_ref(v___x_22_);
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Module_Dual_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_24_, 0, v_toZero_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___redArg___boxed(lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Module_Dual_instInhabited___redArg(v_inst_25_);
lean_dec_ref(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited(lean_object* v_M_27_, lean_object* v_inst_28_, lean_object* v_R_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Module_Dual_instInhabited___redArg(v_inst_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_instInhabited___boxed(lean_object* v_M_33_, lean_object* v_inst_34_, lean_object* v_R_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Module_Dual_instInhabited(v_M_33_, v_inst_34_, v_R_35_, v_inst_36_, v_inst_37_);
lean_dec(v_inst_37_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_34_);
return v_res_38_;
}
}
static lean_object* _init_lp_mathlib_Module_Dual_eval___closed__0(void){
_start:
{
lean_object* v___f_39_; lean_object* v___x_40_; 
v___f_39_ = ((lean_object*)(lp_mathlib_Module_dualPairing___closed__0));
v___x_40_ = lp_mathlib_LinearMap_flip___redArg(v___f_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_eval(lean_object* v_R_41_, lean_object* v_M_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_obj_once(&lp_mathlib_Module_Dual_eval___closed__0, &lp_mathlib_Module_Dual_eval___closed__0_once, _init_lp_mathlib_Module_Dual_eval___closed__0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_eval___boxed(lean_object* v_R_47_, lean_object* v_M_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Module_Dual_eval(v_R_47_, v_M_48_, v_inst_49_, v_inst_50_, v_inst_51_);
lean_dec(v_inst_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_transpose___redArg(lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_toAddCommMonoid_59_; lean_object* v___x_60_; lean_object* v___f_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v_toAddCommMonoid_59_ = lean_ctor_get(v_inst_54_, 0);
lean_inc_ref(v_toAddCommMonoid_59_);
v___x_60_ = lp_mathlib_Semiring_toModule___redArg(v_inst_54_);
v___f_61_ = ((lean_object*)(lp_mathlib_Module_Dual_transpose___redArg___closed__0));
lean_inc_ref_n(v_inst_54_, 2);
v___x_62_ = lp_mathlib_LinearMap_llcomp___redArg(v_inst_54_, v_inst_54_, v_inst_54_, v_inst_55_, v_inst_57_, v_toAddCommMonoid_59_, v_inst_56_, v_inst_58_, v___x_60_, v___f_61_, v___f_61_, v___f_61_);
v___x_63_ = lp_mathlib_LinearMap_flip___redArg(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_Dual_transpose(lean_object* v_R_64_, lean_object* v_M_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_M_x27_69_, lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_Module_Dual_transpose___redArg(v_inst_66_, v_inst_67_, v_inst_68_, v_inst_70_, v_inst_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_dualMap___redArg(lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_f_78_){
_start:
{
lean_object* v___x_18__overap_79_; lean_object* v___x_80_; 
v___x_18__overap_79_ = lp_mathlib_Module_Dual_transpose___redArg(v_inst_73_, v_inst_74_, v_inst_75_, v_inst_76_, v_inst_77_);
v___x_80_ = lean_apply_1(v___x_18__overap_79_, v_f_78_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_dualMap(lean_object* v_R_81_, lean_object* v_M_u2081_82_, lean_object* v_M_u2082_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_f_89_){
_start:
{
lean_object* v___x_25__overap_90_; lean_object* v___x_91_; 
v___x_25__overap_90_ = lp_mathlib_Module_Dual_transpose___redArg(v_inst_84_, v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_);
v___x_91_ = lean_apply_1(v___x_25__overap_90_, v_f_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_dualMap___redArg___lam__0(lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_toLinearMap_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v___x_61__overap_100_; lean_object* v___x_101_; 
v___x_61__overap_100_ = lp_mathlib_Module_Dual_transpose___redArg(v_inst_92_, v_inst_93_, v_inst_94_, v_inst_95_, v_inst_96_);
v___x_101_ = lean_apply_3(v___x_61__overap_100_, v_toLinearMap_97_, v___y_98_, v___y_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_dualMap___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_f_107_){
_start:
{
lean_object* v_toLinearMap_108_; lean_object* v___x_109_; lean_object* v_toLinearMap_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_120_; 
v_toLinearMap_108_ = lean_ctor_get(v_f_107_, 0);
lean_inc(v_toLinearMap_108_);
v___x_109_ = lp_mathlib_LinearEquiv_symm___redArg(v_f_107_);
v_toLinearMap_110_ = lean_ctor_get(v___x_109_, 0);
v_isSharedCheck_120_ = !lean_is_exclusive(v___x_109_);
if (v_isSharedCheck_120_ == 0)
{
lean_object* v_unused_121_; 
v_unused_121_ = lean_ctor_get(v___x_109_, 1);
lean_dec(v_unused_121_);
v___x_112_ = v___x_109_;
v_isShared_113_ = v_isSharedCheck_120_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_toLinearMap_110_);
lean_dec(v___x_109_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_120_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_39__overap_114_; lean_object* v___x_115_; lean_object* v___f_116_; lean_object* v___x_118_; 
lean_inc(v_inst_106_);
lean_inc_ref(v_inst_105_);
lean_inc(v_inst_104_);
lean_inc_ref(v_inst_103_);
lean_inc_ref(v_inst_102_);
v___x_39__overap_114_ = lp_mathlib_Module_Dual_transpose___redArg(v_inst_102_, v_inst_103_, v_inst_104_, v_inst_105_, v_inst_106_);
v___x_115_ = lean_apply_1(v___x_39__overap_114_, v_toLinearMap_108_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_dualMap___redArg___lam__0), 8, 6);
lean_closure_set(v___f_116_, 0, v_inst_102_);
lean_closure_set(v___f_116_, 1, v_inst_105_);
lean_closure_set(v___f_116_, 2, v_inst_106_);
lean_closure_set(v___f_116_, 3, v_inst_103_);
lean_closure_set(v___f_116_, 4, v_inst_104_);
lean_closure_set(v___f_116_, 5, v_toLinearMap_110_);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 1, v___f_116_);
lean_ctor_set(v___x_112_, 0, v___x_115_);
v___x_118_ = v___x_112_;
goto v_reusejp_117_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v___x_115_);
lean_ctor_set(v_reuseFailAlloc_119_, 1, v___f_116_);
v___x_118_ = v_reuseFailAlloc_119_;
goto v_reusejp_117_;
}
v_reusejp_117_:
{
return v___x_118_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_dualMap(lean_object* v_R_122_, lean_object* v_M_u2081_123_, lean_object* v_M_u2082_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_f_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_LinearEquiv_dualMap___redArg(v_inst_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_, v_f_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualRestrict(lean_object* v_R_133_, lean_object* v_M_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_W_138_){
_start:
{
lean_object* v___f_139_; 
v___f_139_ = ((lean_object*)(lp_mathlib_Submodule_dualRestrict___closed__0));
return v___f_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualRestrict___boxed(lean_object* v_R_140_, lean_object* v_M_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_W_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Submodule_dualRestrict(v_R_140_, v_M_141_, v_inst_142_, v_inst_143_, v_inst_144_, v_W_145_);
lean_dec(v_inst_144_);
lean_dec_ref(v_inst_143_);
lean_dec_ref(v_inst_142_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualAnnihilator(lean_object* v_R_147_, lean_object* v_M_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_W_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_box(0);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualAnnihilator___boxed(lean_object* v_R_154_, lean_object* v_M_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_W_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Submodule_dualAnnihilator(v_R_154_, v_M_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_W_159_);
lean_dec(v_inst_158_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_156_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualCoannihilator(lean_object* v_R_161_, lean_object* v_M_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_00_u03a6_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_box(0);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_dualCoannihilator___boxed(lean_object* v_R_168_, lean_object* v_M_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_00_u03a6_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Submodule_dualCoannihilator(v_R_168_, v_M_169_, v_inst_170_, v_inst_171_, v_inst_172_, v_00_u03a6_173_);
lean_dec(v_inst_172_);
lean_dec_ref(v_inst_171_);
lean_dec_ref(v_inst_170_);
return v_res_174_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
