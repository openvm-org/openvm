// Lean compiler output
// Module: Mathlib.Algebra.Field.Subfield.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Basic public import Mathlib.Algebra.Ring.Subring.Defs public import Mathlib.Algebra.Order.Ring.Unbundled.Rat
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
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_InvMemClass_inv___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_SubgroupClass_div___redArg(lean_object*);
lean_object* lp_mathlib_SubgroupClass_instZPow___redArg(lean_object*);
lean_object* lp_mathlib_ZPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAddSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAddSubgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subfield_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subfield_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toSubfield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toSubfield(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toSubfield___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instRingSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instRingSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toDivisionRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toField(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subfield_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subfield_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subfield_subtype___closed__0 = (const lean_object*)&lp_mathlib_Subfield_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast___redArg___lam__0(lean_object* v_toNNRatCast_1_, lean_object* v_q_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_toNNRatCast_1_, v_q_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast___redArg(lean_object* v_inst_4_){
_start:
{
lean_object* v_toNNRatCast_5_; lean_object* v___f_6_; 
v_toNNRatCast_5_ = lean_ctor_get(v_inst_4_, 4);
lean_inc(v_toNNRatCast_5_);
lean_dec_ref(v_inst_4_);
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_toNNRatCast_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast(lean_object* v_K_7_, lean_object* v_inst_8_, lean_object* v_S_9_, lean_object* v_inst_10_, lean_object* v_h_11_, lean_object* v_s_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_SubfieldClass_instNNRatCast___redArg(v_inst_8_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instNNRatCast___boxed(lean_object* v_K_14_, lean_object* v_inst_15_, lean_object* v_S_16_, lean_object* v_inst_17_, lean_object* v_h_18_, lean_object* v_s_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_SubfieldClass_instNNRatCast(v_K_14_, v_inst_15_, v_S_16_, v_inst_17_, v_h_18_, v_s_19_);
lean_dec(v_s_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast___redArg___lam__0(lean_object* v_toRatCast_21_, lean_object* v_q_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_apply_1(v_toRatCast_21_, v_q_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast___redArg(lean_object* v_inst_24_){
_start:
{
lean_object* v_toRatCast_25_; lean_object* v___f_26_; 
v_toRatCast_25_ = lean_ctor_get(v_inst_24_, 5);
lean_inc(v_toRatCast_25_);
lean_dec_ref(v_inst_24_);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_instRatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_26_, 0, v_toRatCast_25_);
return v___f_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast(lean_object* v_K_27_, lean_object* v_inst_28_, lean_object* v_S_29_, lean_object* v_inst_30_, lean_object* v_h_31_, lean_object* v_s_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_SubfieldClass_instRatCast___redArg(v_inst_28_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instRatCast___boxed(lean_object* v_K_34_, lean_object* v_inst_35_, lean_object* v_S_36_, lean_object* v_inst_37_, lean_object* v_h_38_, lean_object* v_s_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_SubfieldClass_instRatCast(v_K_34_, v_inst_35_, v_S_36_, v_inst_37_, v_h_38_, v_s_39_);
lean_dec(v_s_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat___redArg___lam__0(lean_object* v_inst_41_, lean_object* v_q_42_, lean_object* v_x_43_){
_start:
{
lean_object* v_nnqsmul_44_; lean_object* v___x_45_; 
v_nnqsmul_44_ = lean_ctor_get(v_inst_41_, 6);
lean_inc(v_nnqsmul_44_);
lean_dec_ref(v_inst_41_);
v___x_45_ = lean_apply_2(v_nnqsmul_44_, v_q_42_, v_x_43_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat___redArg(lean_object* v_inst_46_){
_start:
{
lean_object* v___f_47_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_instSMulNNRat___redArg___lam__0), 3, 1);
lean_closure_set(v___f_47_, 0, v_inst_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat(lean_object* v_K_48_, lean_object* v_inst_49_, lean_object* v_S_50_, lean_object* v_inst_51_, lean_object* v_h_52_, lean_object* v_s_53_){
_start:
{
lean_object* v___f_54_; 
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_instSMulNNRat___redArg___lam__0), 3, 1);
lean_closure_set(v___f_54_, 0, v_inst_49_);
return v___f_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulNNRat___boxed(lean_object* v_K_55_, lean_object* v_inst_56_, lean_object* v_S_57_, lean_object* v_inst_58_, lean_object* v_h_59_, lean_object* v_s_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_SubfieldClass_instSMulNNRat(v_K_55_, v_inst_56_, v_S_57_, v_inst_58_, v_h_59_, v_s_60_);
lean_dec(v_s_60_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat___redArg___lam__0(lean_object* v_inst_62_, lean_object* v_q_63_, lean_object* v_x_64_){
_start:
{
lean_object* v_qsmul_65_; lean_object* v___x_66_; 
v_qsmul_65_ = lean_ctor_get(v_inst_62_, 7);
lean_inc(v_qsmul_65_);
lean_dec_ref(v_inst_62_);
v___x_66_ = lean_apply_2(v_qsmul_65_, v_q_63_, v_x_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat___redArg(lean_object* v_inst_67_){
_start:
{
lean_object* v___f_68_; 
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_instSMulRat___redArg___lam__0), 3, 1);
lean_closure_set(v___f_68_, 0, v_inst_67_);
return v___f_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat(lean_object* v_K_69_, lean_object* v_inst_70_, lean_object* v_S_71_, lean_object* v_inst_72_, lean_object* v_h_73_, lean_object* v_s_74_){
_start:
{
lean_object* v___f_75_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_instSMulRat___redArg___lam__0), 3, 1);
lean_closure_set(v___f_75_, 0, v_inst_70_);
return v___f_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_instSMulRat___boxed(lean_object* v_K_76_, lean_object* v_inst_77_, lean_object* v_S_78_, lean_object* v_inst_79_, lean_object* v_h_80_, lean_object* v_s_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_SubfieldClass_instSMulRat(v_K_76_, v_inst_77_, v_S_78_, v_inst_79_, v_h_80_, v_s_81_);
lean_dec(v_s_81_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__0(lean_object* v_nnqsmul_83_, lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_apply_2(v_nnqsmul_83_, v_a_84_, v_a_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__1(lean_object* v_qsmul_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lean_apply_2(v_qsmul_87_, v_a_88_, v_a_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___redArg(lean_object* v_inst_91_){
_start:
{
lean_object* v_toRing_92_; lean_object* v_nnqsmul_93_; lean_object* v_qsmul_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v_toInv_100_; lean_object* v___f_101_; lean_object* v___f_102_; lean_object* v___f_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___f_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v_toRing_92_ = lean_ctor_get(v_inst_91_, 0);
v_nnqsmul_93_ = lean_ctor_get(v_inst_91_, 6);
v_qsmul_94_ = lean_ctor_get(v_inst_91_, 7);
lean_inc_ref(v_toRing_92_);
v___x_95_ = lp_mathlib_SubringClass_toRing___redArg(v_toRing_92_);
v___x_96_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_91_);
v___x_97_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_96_);
lean_dec_ref(v___x_96_);
v___x_98_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_97_);
v___x_99_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_98_);
lean_dec_ref(v___x_98_);
v_toInv_100_ = lean_ctor_get(v___x_99_, 1);
lean_inc(v_toInv_100_);
lean_dec_ref(v___x_99_);
lean_inc(v_nnqsmul_93_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__0), 3, 1);
lean_closure_set(v___f_101_, 0, v_nnqsmul_93_);
lean_inc(v_qsmul_94_);
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__1), 3, 1);
lean_closure_set(v___f_102_, 0, v_qsmul_94_);
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_103_, 0, v_toInv_100_);
v___x_104_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_91_);
lean_inc_ref(v___x_104_);
v___x_105_ = lp_mathlib_SubgroupClass_div___redArg(v___x_104_);
v___x_106_ = lp_mathlib_SubgroupClass_instZPow___redArg(v___x_104_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_107_, 0, v___x_106_);
lean_inc_ref(v_inst_91_);
v___x_108_ = lp_mathlib_SubfieldClass_instNNRatCast___redArg(v_inst_91_);
v___x_109_ = lp_mathlib_SubfieldClass_instRatCast___redArg(v_inst_91_);
v___x_110_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_110_, 0, v___x_95_);
lean_ctor_set(v___x_110_, 1, v___f_103_);
lean_ctor_set(v___x_110_, 2, v___x_105_);
lean_ctor_set(v___x_110_, 3, v___f_107_);
lean_ctor_set(v___x_110_, 4, v___x_108_);
lean_ctor_set(v___x_110_, 5, v___x_109_);
lean_ctor_set(v___x_110_, 6, v___f_101_);
lean_ctor_set(v___x_110_, 7, v___f_102_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing(lean_object* v_K_111_, lean_object* v_inst_112_, lean_object* v_S_113_, lean_object* v_inst_114_, lean_object* v_h_115_, lean_object* v_s_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_SubfieldClass_toDivisionRing___redArg(v_inst_112_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toDivisionRing___boxed(lean_object* v_K_118_, lean_object* v_inst_119_, lean_object* v_S_120_, lean_object* v_inst_121_, lean_object* v_h_122_, lean_object* v_s_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_SubfieldClass_toDivisionRing(v_K_118_, v_inst_119_, v_S_120_, v_inst_121_, v_h_122_, v_s_123_);
lean_dec(v_s_123_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toField___redArg(lean_object* v_inst_125_){
_start:
{
lean_object* v_toCommRing_126_; lean_object* v_nnqsmul_127_; lean_object* v_qsmul_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v_toInv_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v_toZPow_139_; lean_object* v___f_140_; lean_object* v___f_141_; lean_object* v___f_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v_toCommRing_126_ = lean_ctor_get(v_inst_125_, 0);
v_nnqsmul_127_ = lean_ctor_get(v_inst_125_, 6);
lean_inc(v_nnqsmul_127_);
v_qsmul_128_ = lean_ctor_get(v_inst_125_, 7);
lean_inc(v_qsmul_128_);
lean_inc_ref(v_toCommRing_126_);
v___x_129_ = lp_mathlib_SubringClass_toRing___redArg(v_toCommRing_126_);
v___x_130_ = lp_mathlib_Field_toSemifield___redArg(v_inst_125_);
v___x_131_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v___x_130_);
lean_dec_ref(v___x_130_);
v___x_132_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v___x_131_);
v___x_133_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_132_);
lean_dec_ref(v___x_132_);
v_toInv_134_ = lean_ctor_get(v___x_133_, 1);
lean_inc(v_toInv_134_);
lean_dec_ref(v___x_133_);
v___x_135_ = lp_mathlib_Field_toDivisionRing___redArg(v_inst_125_);
v___x_136_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v___x_135_);
lean_inc_ref_n(v___x_135_, 2);
v___x_137_ = lp_mathlib_SubfieldClass_toDivisionRing___redArg(v___x_135_);
v___x_138_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v___x_137_);
lean_dec_ref(v___x_137_);
v_toZPow_139_ = lean_ctor_get(v___x_138_, 3);
lean_inc(v_toZPow_139_);
lean_dec_ref(v___x_138_);
v___f_140_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__0), 3, 1);
lean_closure_set(v___f_140_, 0, v_nnqsmul_127_);
v___f_141_ = lean_alloc_closure((void*)(lp_mathlib_SubfieldClass_toDivisionRing___redArg___lam__1), 3, 1);
lean_closure_set(v___f_141_, 0, v_qsmul_128_);
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_142_, 0, v_toInv_134_);
v___x_143_ = lp_mathlib_SubgroupClass_div___redArg(v___x_136_);
v___x_144_ = lp_mathlib_SubfieldClass_instNNRatCast___redArg(v___x_135_);
v___x_145_ = lp_mathlib_SubfieldClass_instRatCast___redArg(v___x_135_);
v___x_146_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_146_, 0, v___x_129_);
lean_ctor_set(v___x_146_, 1, v___f_142_);
lean_ctor_set(v___x_146_, 2, v___x_143_);
lean_ctor_set(v___x_146_, 3, v_toZPow_139_);
lean_ctor_set(v___x_146_, 4, v___x_144_);
lean_ctor_set(v___x_146_, 5, v___x_145_);
lean_ctor_set(v___x_146_, 6, v___f_140_);
lean_ctor_set(v___x_146_, 7, v___f_141_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toField(lean_object* v_S_147_, lean_object* v_K_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_s_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_SubfieldClass_toField___redArg(v_inst_149_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubfieldClass_toField___boxed(lean_object* v_S_154_, lean_object* v_K_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_s_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_SubfieldClass_toField(v_S_154_, v_K_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_s_159_);
lean_dec(v_s_159_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAddSubgroup(lean_object* v_K_161_, lean_object* v_inst_162_, lean_object* v_s_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lean_box(0);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAddSubgroup___boxed(lean_object* v_K_165_, lean_object* v_inst_166_, lean_object* v_s_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Subfield_toAddSubgroup(v_K_165_, v_inst_166_, v_s_167_);
lean_dec_ref(v_inst_166_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSetLike(lean_object* v_K_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSetLike___boxed(lean_object* v_K_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Subfield_instSetLike(v_K_172_, v_inst_173_);
lean_dec_ref(v_inst_173_);
return v_res_174_;
}
}
static lean_object* _init_lp_mathlib_Subfield_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_175_ = lean_box(0);
v___x_176_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPartialOrder(lean_object* v_K_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_obj_once(&lp_mathlib_Subfield_instPartialOrder___closed__0, &lp_mathlib_Subfield_instPartialOrder___closed__0_once, _init_lp_mathlib_Subfield_instPartialOrder___closed__0);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPartialOrder___boxed(lean_object* v_K_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Subfield_instPartialOrder(v_K_180_, v_inst_181_);
lean_dec_ref(v_inst_181_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_copy(lean_object* v_K_183_, lean_object* v_inst_184_, lean_object* v_S_185_, lean_object* v_s_186_, lean_object* v_hs_187_){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lean_box(0);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_copy___boxed(lean_object* v_K_189_, lean_object* v_inst_190_, lean_object* v_S_191_, lean_object* v_s_192_, lean_object* v_hs_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_Subfield_copy(v_K_189_, v_inst_190_, v_S_191_, v_s_192_, v_hs_193_);
lean_dec_ref(v_inst_190_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toSubfield___redArg(lean_object* v_s_195_){
_start:
{
return v_s_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toSubfield(lean_object* v_K_196_, lean_object* v_inst_197_, lean_object* v_s_198_, lean_object* v_hinv_199_){
_start:
{
return v_s_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toSubfield___boxed(lean_object* v_K_200_, lean_object* v_inst_201_, lean_object* v_s_202_, lean_object* v_hinv_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Subring_toSubfield(v_K_200_, v_inst_201_, v_s_202_, v_hinv_203_);
lean_dec_ref(v_inst_201_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instRingSubtypeMem___redArg(lean_object* v_inst_205_){
_start:
{
lean_object* v_toRing_206_; lean_object* v___x_207_; 
v_toRing_206_ = lean_ctor_get(v_inst_205_, 0);
lean_inc_ref(v_toRing_206_);
lean_dec_ref(v_inst_205_);
v___x_207_ = lp_mathlib_SubringClass_toRing___redArg(v_toRing_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instRingSubtypeMem(lean_object* v_K_208_, lean_object* v_inst_209_, lean_object* v_s_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_Subfield_instRingSubtypeMem___redArg(v_inst_209_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___redArg___lam__0(lean_object* v_toDiv_212_, lean_object* v_x_213_, lean_object* v_y_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lean_apply_2(v_toDiv_212_, v_x_213_, v_y_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___redArg(lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; lean_object* v_toDiv_218_; lean_object* v___f_219_; 
v___x_217_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_216_);
v_toDiv_218_ = lean_ctor_get(v___x_217_, 2);
lean_inc(v_toDiv_218_);
lean_dec_ref(v___x_217_);
v___f_219_ = lean_alloc_closure((void*)(lp_mathlib_Subfield_instDivSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_219_, 0, v_toDiv_218_);
return v___f_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___redArg___boxed(lean_object* v_inst_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Subfield_instDivSubtypeMem___redArg(v_inst_220_);
lean_dec_ref(v_inst_220_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem(lean_object* v_K_222_, lean_object* v_inst_223_, lean_object* v_s_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lp_mathlib_Subfield_instDivSubtypeMem___redArg(v_inst_223_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDivSubtypeMem___boxed(lean_object* v_K_226_, lean_object* v_inst_227_, lean_object* v_s_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Subfield_instDivSubtypeMem(v_K_226_, v_inst_227_, v_s_228_);
lean_dec_ref(v_inst_227_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___redArg___lam__0(lean_object* v_toInv_230_, lean_object* v_x_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lean_apply_1(v_toInv_230_, v_x_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___redArg(lean_object* v_inst_233_){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v_toInv_238_; lean_object* v___f_239_; 
v___x_234_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_233_);
v___x_235_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_234_);
lean_dec_ref(v___x_234_);
v___x_236_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_235_);
v___x_237_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_236_);
lean_dec_ref(v___x_236_);
v_toInv_238_ = lean_ctor_get(v___x_237_, 1);
lean_inc(v_toInv_238_);
lean_dec_ref(v___x_237_);
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_Subfield_instInvSubtypeMem___redArg___lam__0), 2, 1);
lean_closure_set(v___f_239_, 0, v_toInv_238_);
return v___f_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___redArg___boxed(lean_object* v_inst_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Subfield_instInvSubtypeMem___redArg(v_inst_240_);
lean_dec_ref(v_inst_240_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem(lean_object* v_K_242_, lean_object* v_inst_243_, lean_object* v_s_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_Subfield_instInvSubtypeMem___redArg(v_inst_243_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInvSubtypeMem___boxed(lean_object* v_K_246_, lean_object* v_inst_247_, lean_object* v_s_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_Subfield_instInvSubtypeMem(v_K_246_, v_inst_247_, v_s_248_);
lean_dec_ref(v_inst_247_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___redArg___lam__0(lean_object* v_toZPow_250_, lean_object* v_x_251_, lean_object* v_z_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lean_apply_2(v_toZPow_250_, v_z_252_, v_x_251_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___redArg(lean_object* v_inst_254_){
_start:
{
lean_object* v___x_255_; lean_object* v_toZPow_256_; lean_object* v___f_257_; 
v___x_255_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_254_);
v_toZPow_256_ = lean_ctor_get(v___x_255_, 3);
lean_inc(v_toZPow_256_);
lean_dec_ref(v___x_255_);
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_Subfield_instPowSubtypeMemInt___redArg___lam__0), 3, 1);
lean_closure_set(v___f_257_, 0, v_toZPow_256_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___redArg___boxed(lean_object* v_inst_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_Subfield_instPowSubtypeMemInt___redArg(v_inst_258_);
lean_dec_ref(v_inst_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt(lean_object* v_K_260_, lean_object* v_inst_261_, lean_object* v_s_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_mathlib_Subfield_instPowSubtypeMemInt___redArg(v_inst_261_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instPowSubtypeMemInt___boxed(lean_object* v_K_264_, lean_object* v_inst_265_, lean_object* v_s_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_Subfield_instPowSubtypeMemInt(v_K_264_, v_inst_265_, v_s_266_);
lean_dec_ref(v_inst_265_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toDivisionRing___redArg(lean_object* v_inst_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_mathlib_SubfieldClass_toDivisionRing___redArg(v_inst_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toDivisionRing(lean_object* v_K_270_, lean_object* v_inst_271_, lean_object* v_s_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_mathlib_SubfieldClass_toDivisionRing___redArg(v_inst_271_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toField___redArg(lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_SubfieldClass_toField___redArg(v_inst_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toField(lean_object* v_K_276_, lean_object* v_inst_277_, lean_object* v_s_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_SubfieldClass_toField___redArg(v_inst_277_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype___lam__0(lean_object* v_self_280_){
_start:
{
lean_inc(v_self_280_);
return v_self_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype___lam__0___boxed(lean_object* v_self_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_Subfield_subtype___lam__0(v_self_281_);
lean_dec(v_self_281_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype(lean_object* v_K_284_, lean_object* v_inst_285_, lean_object* v_s_286_){
_start:
{
lean_object* v___f_287_; 
v___f_287_ = ((lean_object*)(lp_mathlib_Subfield_subtype___closed__0));
return v___f_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_subtype___boxed(lean_object* v_K_288_, lean_object* v_inst_289_, lean_object* v_s_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Subfield_subtype(v_K_288_, v_inst_289_, v_s_290_);
lean_dec_ref(v_inst_289_);
return v_res_291_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
