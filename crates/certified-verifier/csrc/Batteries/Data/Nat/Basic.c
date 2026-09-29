// Lean compiler output
// Module: Batteries.Data.Nat.Basic
// Imports: public import Init public meta import Init
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Bool_toNat(uint8_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Fin_foldr_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRec___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRec(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_ofBits___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_ofBits___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_ofBits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn___redArg(lean_object* v_t_1_, lean_object* v_zero_2_, lean_object* v_succ_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = l_Nat_recCompiled___redArg(v_zero_2_, v_succ_3_, v_t_1_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn___redArg___boxed(lean_object* v_t_5_, lean_object* v_zero_6_, lean_object* v_succ_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_batteries_Nat_recAuxOn___redArg(v_t_5_, v_zero_6_, v_succ_7_);
lean_dec(v_zero_6_);
lean_dec(v_t_5_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn(lean_object* v_motive_9_, lean_object* v_t_10_, lean_object* v_zero_11_, lean_object* v_succ_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = l_Nat_recCompiled___redArg(v_zero_11_, v_succ_12_, v_t_10_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recAuxOn___boxed(lean_object* v_motive_14_, lean_object* v_t_15_, lean_object* v_zero_16_, lean_object* v_succ_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_batteries_Nat_recAuxOn(v_motive_14_, v_t_15_, v_zero_16_, v_succ_17_);
lean_dec(v_zero_16_);
lean_dec(v_t_15_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRec___redArg(lean_object* v_ind_19_, lean_object* v_t_20_){
_start:
{
lean_object* v___f_21_; lean_object* v___x_22_; 
lean_inc(v_ind_19_);
v___f_21_ = lean_alloc_closure((void*)(lp_batteries_Nat_strongRec___redArg___lam__0), 3, 1);
lean_closure_set(v___f_21_, 0, v_ind_19_);
v___x_22_ = lean_apply_2(v_ind_19_, v_t_20_, v___f_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRec___redArg___lam__0(lean_object* v_ind_23_, lean_object* v_m_24_, lean_object* v_x_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_batteries_Nat_strongRec___redArg(v_ind_23_, v_m_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRec(lean_object* v_motive_27_, lean_object* v_ind_28_, lean_object* v_t_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_batteries_Nat_strongRec___redArg(v_ind_28_, v_t_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure___redArg(lean_object* v_ind_31_, lean_object* v_x_32_){
_start:
{
lean_object* v___f_33_; lean_object* v___x_34_; 
lean_inc(v_ind_31_);
v___f_33_ = lean_alloc_closure((void*)(lp_batteries_Nat_strongRecMeasure___redArg___lam__0), 3, 1);
lean_closure_set(v___f_33_, 0, v_ind_31_);
v___x_34_ = lean_apply_2(v_ind_31_, v_x_32_, v___f_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure___redArg___lam__0(lean_object* v_ind_35_, lean_object* v_y_36_, lean_object* v_x_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_batteries_Nat_strongRecMeasure___redArg(v_ind_35_, v_y_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure(lean_object* v_00_u03b1_39_, lean_object* v_f_40_, lean_object* v_motive_41_, lean_object* v_ind_42_, lean_object* v_x_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_batteries_Nat_strongRecMeasure___redArg(v_ind_42_, v_x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_strongRecMeasure___boxed(lean_object* v_00_u03b1_45_, lean_object* v_f_46_, lean_object* v_motive_47_, lean_object* v_ind_48_, lean_object* v_x_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_batteries_Nat_strongRecMeasure(v_00_u03b1_45_, v_f_46_, v_motive_47_, v_ind_48_, v_x_49_);
lean_dec_ref(v_f_46_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagAux___redArg(lean_object* v_zero__left_51_, lean_object* v_zero__right_52_, lean_object* v_succ__succ_53_, lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
lean_object* v_zero_56_; uint8_t v_isZero_57_; 
v_zero_56_ = lean_unsigned_to_nat(0u);
v_isZero_57_ = lean_nat_dec_eq(v_x_54_, v_zero_56_);
if (v_isZero_57_ == 1)
{
lean_object* v___x_58_; 
lean_dec(v_x_54_);
lean_dec(v_succ__succ_53_);
lean_dec(v_zero__right_52_);
v___x_58_ = lean_apply_1(v_zero__left_51_, v_x_55_);
return v___x_58_;
}
else
{
uint8_t v_isZero_59_; 
v_isZero_59_ = lean_nat_dec_eq(v_x_55_, v_zero_56_);
if (v_isZero_59_ == 1)
{
lean_object* v___x_60_; 
lean_dec(v_x_55_);
lean_dec(v_succ__succ_53_);
lean_dec(v_zero__left_51_);
v___x_60_ = lean_apply_1(v_zero__right_52_, v_x_54_);
return v___x_60_;
}
else
{
lean_object* v_one_61_; lean_object* v_n_62_; lean_object* v_n_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v_one_61_ = lean_unsigned_to_nat(1u);
v_n_62_ = lean_nat_sub(v_x_54_, v_one_61_);
lean_dec(v_x_54_);
v_n_63_ = lean_nat_sub(v_x_55_, v_one_61_);
lean_dec(v_x_55_);
lean_inc(v_n_63_);
lean_inc(v_n_62_);
lean_inc(v_succ__succ_53_);
v___x_64_ = lp_batteries_Nat_recDiagAux___redArg(v_zero__left_51_, v_zero__right_52_, v_succ__succ_53_, v_n_62_, v_n_63_);
v___x_65_ = lean_apply_3(v_succ__succ_53_, v_n_62_, v_n_63_, v___x_64_);
return v___x_65_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagAux(lean_object* v_motive_66_, lean_object* v_zero__left_67_, lean_object* v_zero__right_68_, lean_object* v_succ__succ_69_, lean_object* v_x_70_, lean_object* v_x_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_batteries_Nat_recDiagAux___redArg(v_zero__left_67_, v_zero__right_68_, v_succ__succ_69_, v_x_70_, v_x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left___redArg(lean_object* v_zero__zero_73_, lean_object* v_zero__succ_74_, lean_object* v_n_75_){
_start:
{
lean_object* v_zero_76_; uint8_t v_isZero_77_; 
v_zero_76_ = lean_unsigned_to_nat(0u);
v_isZero_77_ = lean_nat_dec_eq(v_n_75_, v_zero_76_);
if (v_isZero_77_ == 1)
{
lean_dec(v_zero__succ_74_);
lean_inc(v_zero__zero_73_);
return v_zero__zero_73_;
}
else
{
lean_object* v_one_78_; lean_object* v_n_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v_one_78_ = lean_unsigned_to_nat(1u);
v_n_79_ = lean_nat_sub(v_n_75_, v_one_78_);
lean_inc(v_zero__succ_74_);
v___x_80_ = lp_batteries_Nat_recDiag_left___redArg(v_zero__zero_73_, v_zero__succ_74_, v_n_79_);
v___x_81_ = lean_apply_2(v_zero__succ_74_, v_n_79_, v___x_80_);
return v___x_81_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left___redArg___boxed(lean_object* v_zero__zero_82_, lean_object* v_zero__succ_83_, lean_object* v_n_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_batteries_Nat_recDiag_left___redArg(v_zero__zero_82_, v_zero__succ_83_, v_n_84_);
lean_dec(v_n_84_);
lean_dec(v_zero__zero_82_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left(lean_object* v_motive_86_, lean_object* v_zero__zero_87_, lean_object* v_zero__succ_88_, lean_object* v_n_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_batteries_Nat_recDiag_left___redArg(v_zero__zero_87_, v_zero__succ_88_, v_n_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_left___boxed(lean_object* v_motive_91_, lean_object* v_zero__zero_92_, lean_object* v_zero__succ_93_, lean_object* v_n_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_batteries_Nat_recDiag_left(v_motive_91_, v_zero__zero_92_, v_zero__succ_93_, v_n_94_);
lean_dec(v_n_94_);
lean_dec(v_zero__zero_92_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right___redArg(lean_object* v_zero__zero_96_, lean_object* v_succ__zero_97_, lean_object* v_m_98_){
_start:
{
lean_object* v_zero_99_; uint8_t v_isZero_100_; 
v_zero_99_ = lean_unsigned_to_nat(0u);
v_isZero_100_ = lean_nat_dec_eq(v_m_98_, v_zero_99_);
if (v_isZero_100_ == 1)
{
lean_dec(v_succ__zero_97_);
lean_inc(v_zero__zero_96_);
return v_zero__zero_96_;
}
else
{
lean_object* v_one_101_; lean_object* v_n_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v_one_101_ = lean_unsigned_to_nat(1u);
v_n_102_ = lean_nat_sub(v_m_98_, v_one_101_);
lean_inc(v_succ__zero_97_);
v___x_103_ = lp_batteries_Nat_recDiag_right___redArg(v_zero__zero_96_, v_succ__zero_97_, v_n_102_);
v___x_104_ = lean_apply_2(v_succ__zero_97_, v_n_102_, v___x_103_);
return v___x_104_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right___redArg___boxed(lean_object* v_zero__zero_105_, lean_object* v_succ__zero_106_, lean_object* v_m_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_batteries_Nat_recDiag_right___redArg(v_zero__zero_105_, v_succ__zero_106_, v_m_107_);
lean_dec(v_m_107_);
lean_dec(v_zero__zero_105_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right(lean_object* v_motive_109_, lean_object* v_zero__zero_110_, lean_object* v_succ__zero_111_, lean_object* v_m_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_batteries_Nat_recDiag_right___redArg(v_zero__zero_110_, v_succ__zero_111_, v_m_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag_right___boxed(lean_object* v_motive_114_, lean_object* v_zero__zero_115_, lean_object* v_succ__zero_116_, lean_object* v_m_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_batteries_Nat_recDiag_right(v_motive_114_, v_zero__zero_115_, v_succ__zero_116_, v_m_117_);
lean_dec(v_m_117_);
lean_dec(v_zero__zero_115_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag___redArg(lean_object* v_zero__zero_119_, lean_object* v_zero__succ_120_, lean_object* v_succ__zero_121_, lean_object* v_succ__succ_122_, lean_object* v_m_123_, lean_object* v_n_124_){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
lean_inc(v_zero__zero_119_);
v___x_125_ = lean_alloc_closure((void*)(lp_batteries_Nat_recDiag_left___boxed), 4, 3);
lean_closure_set(v___x_125_, 0, lean_box(0));
lean_closure_set(v___x_125_, 1, v_zero__zero_119_);
lean_closure_set(v___x_125_, 2, v_zero__succ_120_);
v___x_126_ = lean_alloc_closure((void*)(lp_batteries_Nat_recDiag_right___boxed), 4, 3);
lean_closure_set(v___x_126_, 0, lean_box(0));
lean_closure_set(v___x_126_, 1, v_zero__zero_119_);
lean_closure_set(v___x_126_, 2, v_succ__zero_121_);
v___x_127_ = lp_batteries_Nat_recDiagAux___redArg(v___x_125_, v___x_126_, v_succ__succ_122_, v_m_123_, v_n_124_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiag(lean_object* v_motive_128_, lean_object* v_zero__zero_129_, lean_object* v_zero__succ_130_, lean_object* v_succ__zero_131_, lean_object* v_succ__succ_132_, lean_object* v_m_133_, lean_object* v_n_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_batteries_Nat_recDiag___redArg(v_zero__zero_129_, v_zero__succ_130_, v_succ__zero_131_, v_succ__succ_132_, v_m_133_, v_n_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagOn___redArg(lean_object* v_m_136_, lean_object* v_n_137_, lean_object* v_zero__zero_138_, lean_object* v_zero__succ_139_, lean_object* v_succ__zero_140_, lean_object* v_succ__succ_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_batteries_Nat_recDiag___redArg(v_zero__zero_138_, v_zero__succ_139_, v_succ__zero_140_, v_succ__succ_141_, v_m_136_, v_n_137_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagOn(lean_object* v_motive_143_, lean_object* v_m_144_, lean_object* v_n_145_, lean_object* v_zero__zero_146_, lean_object* v_zero__succ_147_, lean_object* v_succ__zero_148_, lean_object* v_succ__succ_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_batteries_Nat_recDiag___redArg(v_zero__zero_146_, v_zero__succ_147_, v_succ__zero_148_, v_succ__succ_149_, v_m_144_, v_n_145_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__0(lean_object* v_zero__succ_151_, lean_object* v_x_152_, lean_object* v_x_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_apply_1(v_zero__succ_151_, v_x_152_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__0___boxed(lean_object* v_zero__succ_155_, lean_object* v_x_156_, lean_object* v_x_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_batteries_Nat_casesDiagOn___redArg___lam__0(v_zero__succ_155_, v_x_156_, v_x_157_);
lean_dec(v_x_157_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__1(lean_object* v_succ__zero_159_, lean_object* v_x_160_, lean_object* v_x_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_apply_1(v_succ__zero_159_, v_x_160_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__1___boxed(lean_object* v_succ__zero_163_, lean_object* v_x_164_, lean_object* v_x_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_batteries_Nat_casesDiagOn___redArg___lam__1(v_succ__zero_163_, v_x_164_, v_x_165_);
lean_dec(v_x_165_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__2(lean_object* v_succ__succ_167_, lean_object* v_x_168_, lean_object* v_x_169_, lean_object* v_x_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_apply_2(v_succ__succ_167_, v_x_168_, v_x_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg___lam__2___boxed(lean_object* v_succ__succ_172_, lean_object* v_x_173_, lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_batteries_Nat_casesDiagOn___redArg___lam__2(v_succ__succ_172_, v_x_173_, v_x_174_, v_x_175_);
lean_dec(v_x_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn___redArg(lean_object* v_m_177_, lean_object* v_n_178_, lean_object* v_zero__zero_179_, lean_object* v_zero__succ_180_, lean_object* v_succ__zero_181_, lean_object* v_succ__succ_182_){
_start:
{
lean_object* v___f_183_; lean_object* v___f_184_; lean_object* v___f_185_; lean_object* v___x_186_; 
v___f_183_ = lean_alloc_closure((void*)(lp_batteries_Nat_casesDiagOn___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_183_, 0, v_zero__succ_180_);
v___f_184_ = lean_alloc_closure((void*)(lp_batteries_Nat_casesDiagOn___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_184_, 0, v_succ__zero_181_);
v___f_185_ = lean_alloc_closure((void*)(lp_batteries_Nat_casesDiagOn___redArg___lam__2___boxed), 4, 1);
lean_closure_set(v___f_185_, 0, v_succ__succ_182_);
v___x_186_ = lp_batteries_Nat_recDiag___redArg(v_zero__zero_179_, v___f_183_, v___f_184_, v___f_185_, v_m_177_, v_n_178_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_casesDiagOn(lean_object* v_motive_187_, lean_object* v_m_188_, lean_object* v_n_189_, lean_object* v_zero__zero_190_, lean_object* v_zero__succ_191_, lean_object* v_succ__zero_192_, lean_object* v_succ__succ_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_batteries_Nat_casesDiagOn___redArg(v_m_188_, v_n_189_, v_zero__zero_190_, v_zero__succ_191_, v_succ__zero_192_, v_succ__succ_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_ofBits___lam__0(lean_object* v_f_195_, lean_object* v_i_196_, lean_object* v_v_197_){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; uint8_t v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_198_ = lean_unsigned_to_nat(2u);
v___x_199_ = lean_nat_mul(v___x_198_, v_v_197_);
v___x_200_ = lean_apply_1(v_f_195_, v_i_196_);
v___x_201_ = lean_unbox(v___x_200_);
v___x_202_ = l_Bool_toNat(v___x_201_);
v___x_203_ = lean_nat_add(v___x_199_, v___x_202_);
lean_dec(v___x_202_);
lean_dec(v___x_199_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_ofBits___lam__0___boxed(lean_object* v_f_204_, lean_object* v_i_205_, lean_object* v_v_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_batteries_Nat_ofBits___lam__0(v_f_204_, v_i_205_, v_v_206_);
lean_dec(v_v_206_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_ofBits(lean_object* v_n_208_, lean_object* v_f_209_){
_start:
{
lean_object* v___f_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___f_210_ = lean_alloc_closure((void*)(lp_batteries_Nat_ofBits___lam__0___boxed), 3, 1);
lean_closure_set(v___f_210_, 0, v_f_209_);
v___x_211_ = lean_unsigned_to_nat(0u);
v___x_212_ = l_Fin_foldr_loop___redArg(v___f_210_, v_n_208_, v___x_211_);
return v___x_212_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_Nat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_Nat_Basic(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_Nat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_Nat_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
