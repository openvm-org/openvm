// Lean compiler output
// Module: Aesop.Tree.Traversal
// Imports: public import Init public meta import Init public import Aesop.Tree.Data
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
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_goal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_goal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_rapp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_rapp_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_mvarCluster_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_mvarCluster_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseUp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseUp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseUp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseUp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_aesop_Aesop_TreeRef_ctorIdx(v_x_5_);
lean_dec_ref(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorElim___redArg(lean_object* v_t_7_, lean_object* v_k_8_){
_start:
{
lean_object* v_gref_9_; lean_object* v___x_10_; 
v_gref_9_ = lean_ctor_get(v_t_7_, 0);
lean_inc(v_gref_9_);
lean_dec_ref(v_t_7_);
v___x_10_ = lean_apply_1(v_k_8_, v_gref_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, lean_object* v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_13_, v_k_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_ctorElim___boxed(lean_object* v_motive_17_, lean_object* v_ctorIdx_18_, lean_object* v_t_19_, lean_object* v_h_20_, lean_object* v_k_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_aesop_Aesop_TreeRef_ctorElim(v_motive_17_, v_ctorIdx_18_, v_t_19_, v_h_20_, v_k_21_);
lean_dec(v_ctorIdx_18_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_goal_elim___redArg(lean_object* v_t_23_, lean_object* v_goal_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_23_, v_goal_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_goal_elim(lean_object* v_motive_26_, lean_object* v_t_27_, lean_object* v_h_28_, lean_object* v_goal_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_27_, v_goal_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_rapp_elim___redArg(lean_object* v_t_31_, lean_object* v_rapp_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_31_, v_rapp_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_rapp_elim(lean_object* v_motive_34_, lean_object* v_t_35_, lean_object* v_h_36_, lean_object* v_rapp_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_35_, v_rapp_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_mvarCluster_elim___redArg(lean_object* v_t_39_, lean_object* v_mvarCluster_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_39_, v_mvarCluster_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_mvarCluster_elim(lean_object* v_motive_42_, lean_object* v_t_43_, lean_object* v_h_44_, lean_object* v_mvarCluster_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_aesop_Aesop_TreeRef_ctorElim___redArg(v_t_43_, v_mvarCluster_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__1(lean_object* v_visitGoalPost_47_, lean_object* v_gref_48_, lean_object* v_____r_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_apply_1(v_visitGoalPost_47_, v_gref_48_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__5(lean_object* v_visitRappPost_51_, lean_object* v_rref_52_, lean_object* v_____r_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lean_apply_1(v_visitRappPost_51_, v_rref_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__2(lean_object* v_toApplicative_55_, lean_object* v_toBind_56_, lean_object* v___f_57_, lean_object* v_inst_58_, lean_object* v___f_59_, lean_object* v_____do__lift_60_){
_start:
{
lean_object* v___x_61_; lean_object* v_elimGoal_62_; lean_object* v___x_63_; lean_object* v_children_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_61_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_62_ = lean_ctor_get(v___x_61_, 1);
lean_inc_ref(v_elimGoal_62_);
v___x_63_ = lean_apply_1(v_elimGoal_62_, v_____do__lift_60_);
v_children_64_ = lean_ctor_get(v___x_63_, 2);
lean_inc_ref(v_children_64_);
lean_dec_ref(v___x_63_);
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = lean_array_get_size(v_children_64_);
v___x_67_ = lean_box(0);
v___x_68_ = lean_nat_dec_lt(v___x_65_, v___x_66_);
if (v___x_68_ == 0)
{
lean_object* v_toPure_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
lean_dec_ref(v_children_64_);
lean_dec(v___f_59_);
lean_dec_ref(v_inst_58_);
v_toPure_69_ = lean_ctor_get(v_toApplicative_55_, 1);
lean_inc(v_toPure_69_);
lean_dec_ref(v_toApplicative_55_);
v___x_70_ = lean_apply_2(v_toPure_69_, lean_box(0), v___x_67_);
v___x_71_ = lean_apply_4(v_toBind_56_, lean_box(0), lean_box(0), v___x_70_, v___f_57_);
return v___x_71_;
}
else
{
uint8_t v___x_72_; 
v___x_72_ = lean_nat_dec_le(v___x_66_, v___x_66_);
if (v___x_72_ == 0)
{
if (v___x_68_ == 0)
{
lean_object* v_toPure_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
lean_dec_ref(v_children_64_);
lean_dec(v___f_59_);
lean_dec_ref(v_inst_58_);
v_toPure_73_ = lean_ctor_get(v_toApplicative_55_, 1);
lean_inc(v_toPure_73_);
lean_dec_ref(v_toApplicative_55_);
v___x_74_ = lean_apply_2(v_toPure_73_, lean_box(0), v___x_67_);
v___x_75_ = lean_apply_4(v_toBind_56_, lean_box(0), lean_box(0), v___x_74_, v___f_57_);
return v___x_75_;
}
else
{
size_t v___x_76_; size_t v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
lean_dec_ref(v_toApplicative_55_);
v___x_76_ = ((size_t)0ULL);
v___x_77_ = lean_usize_of_nat(v___x_66_);
v___x_78_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_58_, v___f_59_, v_children_64_, v___x_76_, v___x_77_, v___x_67_);
v___x_79_ = lean_apply_4(v_toBind_56_, lean_box(0), lean_box(0), v___x_78_, v___f_57_);
return v___x_79_;
}
}
else
{
size_t v___x_80_; size_t v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
lean_dec_ref(v_toApplicative_55_);
v___x_80_ = ((size_t)0ULL);
v___x_81_ = lean_usize_of_nat(v___x_66_);
v___x_82_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_58_, v___f_59_, v_children_64_, v___x_80_, v___x_81_, v___x_67_);
v___x_83_ = lean_apply_4(v_toBind_56_, lean_box(0), lean_box(0), v___x_82_, v___f_57_);
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__6(lean_object* v_toApplicative_84_, lean_object* v_toBind_85_, lean_object* v___f_86_, lean_object* v_inst_87_, lean_object* v___f_88_, lean_object* v_____do__lift_89_){
_start:
{
lean_object* v___x_90_; lean_object* v_elimRapp_91_; lean_object* v___x_92_; lean_object* v_children_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_90_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_91_ = lean_ctor_get(v___x_90_, 3);
lean_inc_ref(v_elimRapp_91_);
v___x_92_ = lean_apply_1(v_elimRapp_91_, v_____do__lift_89_);
v_children_93_ = lean_ctor_get(v___x_92_, 2);
lean_inc_ref(v_children_93_);
lean_dec_ref(v___x_92_);
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = lean_array_get_size(v_children_93_);
v___x_96_ = lean_box(0);
v___x_97_ = lean_nat_dec_lt(v___x_94_, v___x_95_);
if (v___x_97_ == 0)
{
lean_object* v_toPure_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
lean_dec_ref(v_children_93_);
lean_dec(v___f_88_);
lean_dec_ref(v_inst_87_);
v_toPure_98_ = lean_ctor_get(v_toApplicative_84_, 1);
lean_inc(v_toPure_98_);
lean_dec_ref(v_toApplicative_84_);
v___x_99_ = lean_apply_2(v_toPure_98_, lean_box(0), v___x_96_);
v___x_100_ = lean_apply_4(v_toBind_85_, lean_box(0), lean_box(0), v___x_99_, v___f_86_);
return v___x_100_;
}
else
{
uint8_t v___x_101_; 
v___x_101_ = lean_nat_dec_le(v___x_95_, v___x_95_);
if (v___x_101_ == 0)
{
if (v___x_97_ == 0)
{
lean_object* v_toPure_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec_ref(v_children_93_);
lean_dec(v___f_88_);
lean_dec_ref(v_inst_87_);
v_toPure_102_ = lean_ctor_get(v_toApplicative_84_, 1);
lean_inc(v_toPure_102_);
lean_dec_ref(v_toApplicative_84_);
v___x_103_ = lean_apply_2(v_toPure_102_, lean_box(0), v___x_96_);
v___x_104_ = lean_apply_4(v_toBind_85_, lean_box(0), lean_box(0), v___x_103_, v___f_86_);
return v___x_104_;
}
else
{
size_t v___x_105_; size_t v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
lean_dec_ref(v_toApplicative_84_);
v___x_105_ = ((size_t)0ULL);
v___x_106_ = lean_usize_of_nat(v___x_95_);
v___x_107_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_87_, v___f_88_, v_children_93_, v___x_105_, v___x_106_, v___x_96_);
v___x_108_ = lean_apply_4(v_toBind_85_, lean_box(0), lean_box(0), v___x_107_, v___f_86_);
return v___x_108_;
}
}
else
{
size_t v___x_109_; size_t v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
lean_dec_ref(v_toApplicative_84_);
v___x_109_ = ((size_t)0ULL);
v___x_110_ = lean_usize_of_nat(v___x_95_);
v___x_111_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_87_, v___f_88_, v_children_93_, v___x_109_, v___x_110_, v___x_96_);
v___x_112_ = lean_apply_4(v_toBind_85_, lean_box(0), lean_box(0), v___x_111_, v___f_86_);
return v___x_112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__11(lean_object* v_toApplicative_113_, lean_object* v_cref_114_, lean_object* v_inst_115_, lean_object* v_toBind_116_, lean_object* v___f_117_, uint8_t v_____do__lift_118_){
_start:
{
if (v_____do__lift_118_ == 0)
{
lean_object* v_toPure_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
lean_dec(v___f_117_);
lean_dec(v_toBind_116_);
lean_dec(v_inst_115_);
lean_dec(v_cref_114_);
v_toPure_119_ = lean_ctor_get(v_toApplicative_113_, 1);
lean_inc(v_toPure_119_);
lean_dec_ref(v_toApplicative_113_);
v___x_120_ = lean_box(0);
v___x_121_ = lean_apply_2(v_toPure_119_, lean_box(0), v___x_120_);
return v___x_121_;
}
else
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
lean_dec_ref(v_toApplicative_113_);
v___x_122_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_122_, 0, lean_box(0));
lean_closure_set(v___x_122_, 1, lean_box(0));
lean_closure_set(v___x_122_, 2, v_cref_114_);
v___x_123_ = lean_apply_2(v_inst_115_, lean_box(0), v___x_122_);
v___x_124_ = lean_apply_4(v_toBind_116_, lean_box(0), lean_box(0), v___x_123_, v___f_117_);
return v___x_124_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__11___boxed(lean_object* v_toApplicative_125_, lean_object* v_cref_126_, lean_object* v_inst_127_, lean_object* v_toBind_128_, lean_object* v___f_129_, lean_object* v_____do__lift_130_){
_start:
{
uint8_t v_____do__lift_1041__boxed_131_; lean_object* v_res_132_; 
v_____do__lift_1041__boxed_131_ = lean_unbox(v_____do__lift_130_);
v_res_132_ = lp_aesop_Aesop_traverseDown___redArg___lam__11(v_toApplicative_125_, v_cref_126_, v_inst_127_, v_toBind_128_, v___f_129_, v_____do__lift_1041__boxed_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__3(lean_object* v_toApplicative_133_, lean_object* v_gref_134_, lean_object* v_inst_135_, lean_object* v_toBind_136_, lean_object* v___f_137_, uint8_t v_____do__lift_138_){
_start:
{
if (v_____do__lift_138_ == 0)
{
lean_object* v_toPure_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
lean_dec(v___f_137_);
lean_dec(v_toBind_136_);
lean_dec(v_inst_135_);
lean_dec(v_gref_134_);
v_toPure_139_ = lean_ctor_get(v_toApplicative_133_, 1);
lean_inc(v_toPure_139_);
lean_dec_ref(v_toApplicative_133_);
v___x_140_ = lean_box(0);
v___x_141_ = lean_apply_2(v_toPure_139_, lean_box(0), v___x_140_);
return v___x_141_;
}
else
{
lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
lean_dec_ref(v_toApplicative_133_);
v___x_142_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_142_, 0, lean_box(0));
lean_closure_set(v___x_142_, 1, lean_box(0));
lean_closure_set(v___x_142_, 2, v_gref_134_);
v___x_143_ = lean_apply_2(v_inst_135_, lean_box(0), v___x_142_);
v___x_144_ = lean_apply_4(v_toBind_136_, lean_box(0), lean_box(0), v___x_143_, v___f_137_);
return v___x_144_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__3___boxed(lean_object* v_toApplicative_145_, lean_object* v_gref_146_, lean_object* v_inst_147_, lean_object* v_toBind_148_, lean_object* v___f_149_, lean_object* v_____do__lift_150_){
_start:
{
uint8_t v_____do__lift_1063__boxed_151_; lean_object* v_res_152_; 
v_____do__lift_1063__boxed_151_ = lean_unbox(v_____do__lift_150_);
v_res_152_ = lp_aesop_Aesop_traverseDown___redArg___lam__3(v_toApplicative_145_, v_gref_146_, v_inst_147_, v_toBind_148_, v___f_149_, v_____do__lift_1063__boxed_151_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__10(lean_object* v_toApplicative_153_, lean_object* v_toBind_154_, lean_object* v___f_155_, lean_object* v_inst_156_, lean_object* v___f_157_, lean_object* v_____do__lift_158_){
_start:
{
lean_object* v___x_159_; lean_object* v_elimMVarCluster_160_; lean_object* v___x_161_; lean_object* v_goals_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_159_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_160_ = lean_ctor_get(v___x_159_, 5);
lean_inc_ref(v_elimMVarCluster_160_);
v___x_161_ = lean_apply_1(v_elimMVarCluster_160_, v_____do__lift_158_);
v_goals_162_ = lean_ctor_get(v___x_161_, 1);
lean_inc_ref(v_goals_162_);
lean_dec_ref(v___x_161_);
v___x_163_ = lean_unsigned_to_nat(0u);
v___x_164_ = lean_array_get_size(v_goals_162_);
v___x_165_ = lean_box(0);
v___x_166_ = lean_nat_dec_lt(v___x_163_, v___x_164_);
if (v___x_166_ == 0)
{
lean_object* v_toPure_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
lean_dec_ref(v_goals_162_);
lean_dec(v___f_157_);
lean_dec_ref(v_inst_156_);
v_toPure_167_ = lean_ctor_get(v_toApplicative_153_, 1);
lean_inc(v_toPure_167_);
lean_dec_ref(v_toApplicative_153_);
v___x_168_ = lean_apply_2(v_toPure_167_, lean_box(0), v___x_165_);
v___x_169_ = lean_apply_4(v_toBind_154_, lean_box(0), lean_box(0), v___x_168_, v___f_155_);
return v___x_169_;
}
else
{
uint8_t v___x_170_; 
v___x_170_ = lean_nat_dec_le(v___x_164_, v___x_164_);
if (v___x_170_ == 0)
{
if (v___x_166_ == 0)
{
lean_object* v_toPure_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
lean_dec_ref(v_goals_162_);
lean_dec(v___f_157_);
lean_dec_ref(v_inst_156_);
v_toPure_171_ = lean_ctor_get(v_toApplicative_153_, 1);
lean_inc(v_toPure_171_);
lean_dec_ref(v_toApplicative_153_);
v___x_172_ = lean_apply_2(v_toPure_171_, lean_box(0), v___x_165_);
v___x_173_ = lean_apply_4(v_toBind_154_, lean_box(0), lean_box(0), v___x_172_, v___f_155_);
return v___x_173_;
}
else
{
size_t v___x_174_; size_t v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
lean_dec_ref(v_toApplicative_153_);
v___x_174_ = ((size_t)0ULL);
v___x_175_ = lean_usize_of_nat(v___x_164_);
v___x_176_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_156_, v___f_157_, v_goals_162_, v___x_174_, v___x_175_, v___x_165_);
v___x_177_ = lean_apply_4(v_toBind_154_, lean_box(0), lean_box(0), v___x_176_, v___f_155_);
return v___x_177_;
}
}
else
{
size_t v___x_178_; size_t v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
lean_dec_ref(v_toApplicative_153_);
v___x_178_ = ((size_t)0ULL);
v___x_179_ = lean_usize_of_nat(v___x_164_);
v___x_180_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_156_, v___f_157_, v_goals_162_, v___x_178_, v___x_179_, v___x_165_);
v___x_181_ = lean_apply_4(v_toBind_154_, lean_box(0), lean_box(0), v___x_180_, v___f_155_);
return v___x_181_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__9(lean_object* v_visitMVarClusterPost_182_, lean_object* v_cref_183_, lean_object* v_____r_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lean_apply_1(v_visitMVarClusterPost_182_, v_cref_183_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__7(lean_object* v_toApplicative_186_, lean_object* v_rref_187_, lean_object* v_inst_188_, lean_object* v_toBind_189_, lean_object* v___f_190_, uint8_t v_____do__lift_191_){
_start:
{
if (v_____do__lift_191_ == 0)
{
lean_object* v_toPure_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
lean_dec(v___f_190_);
lean_dec(v_toBind_189_);
lean_dec(v_inst_188_);
lean_dec(v_rref_187_);
v_toPure_192_ = lean_ctor_get(v_toApplicative_186_, 1);
lean_inc(v_toPure_192_);
lean_dec_ref(v_toApplicative_186_);
v___x_193_ = lean_box(0);
v___x_194_ = lean_apply_2(v_toPure_192_, lean_box(0), v___x_193_);
return v___x_194_;
}
else
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
lean_dec_ref(v_toApplicative_186_);
v___x_195_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_195_, 0, lean_box(0));
lean_closure_set(v___x_195_, 1, lean_box(0));
lean_closure_set(v___x_195_, 2, v_rref_187_);
v___x_196_ = lean_apply_2(v_inst_188_, lean_box(0), v___x_195_);
v___x_197_ = lean_apply_4(v_toBind_189_, lean_box(0), lean_box(0), v___x_196_, v___f_190_);
return v___x_197_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__7___boxed(lean_object* v_toApplicative_198_, lean_object* v_rref_199_, lean_object* v_inst_200_, lean_object* v_toBind_201_, lean_object* v___f_202_, lean_object* v_____do__lift_203_){
_start:
{
uint8_t v_____do__lift_1137__boxed_204_; lean_object* v_res_205_; 
v_____do__lift_1137__boxed_204_ = lean_unbox(v_____do__lift_203_);
v_res_205_ = lp_aesop_Aesop_traverseDown___redArg___lam__7(v_toApplicative_198_, v_rref_199_, v_inst_200_, v_toBind_201_, v___f_202_, v_____do__lift_1137__boxed_204_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__4(lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_visitGoalPre_208_, lean_object* v_visitGoalPost_209_, lean_object* v_visitRappPre_210_, lean_object* v_visitRappPost_211_, lean_object* v_visitMVarClusterPre_212_, lean_object* v_visitMVarClusterPost_213_, lean_object* v_x_214_, lean_object* v___y_215_){
_start:
{
lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_216_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_216_, 0, v___y_215_);
v___x_217_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_206_, v_inst_207_, v_visitGoalPre_208_, v_visitGoalPost_209_, v_visitRappPre_210_, v_visitRappPost_211_, v_visitMVarClusterPre_212_, v_visitMVarClusterPost_213_, v___x_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__8(lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_visitGoalPre_220_, lean_object* v_visitGoalPost_221_, lean_object* v_visitRappPre_222_, lean_object* v_visitRappPost_223_, lean_object* v_visitMVarClusterPre_224_, lean_object* v_visitMVarClusterPost_225_, lean_object* v_x_226_, lean_object* v___y_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_228_, 0, v___y_227_);
v___x_229_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_218_, v_inst_219_, v_visitGoalPre_220_, v_visitGoalPost_221_, v_visitRappPre_222_, v_visitRappPost_223_, v_visitMVarClusterPre_224_, v_visitMVarClusterPost_225_, v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg(lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_visitGoalPre_232_, lean_object* v_visitGoalPost_233_, lean_object* v_visitRappPre_234_, lean_object* v_visitRappPost_235_, lean_object* v_visitMVarClusterPre_236_, lean_object* v_visitMVarClusterPost_237_, lean_object* v_x_238_){
_start:
{
switch(lean_obj_tag(v_x_238_))
{
case 0:
{
lean_object* v_toApplicative_239_; lean_object* v_toBind_240_; lean_object* v_gref_241_; lean_object* v___f_242_; lean_object* v___f_243_; lean_object* v___f_244_; lean_object* v___f_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v_toApplicative_239_ = lean_ctor_get(v_inst_230_, 0);
lean_inc_ref_n(v_toApplicative_239_, 2);
v_toBind_240_ = lean_ctor_get(v_inst_230_, 1);
lean_inc_n(v_toBind_240_, 3);
v_gref_241_ = lean_ctor_get(v_x_238_, 0);
lean_inc_n(v_gref_241_, 3);
lean_dec_ref_known(v_x_238_, 1);
lean_inc(v_visitGoalPost_233_);
lean_inc(v_visitGoalPre_232_);
lean_inc(v_inst_231_);
lean_inc_ref(v_inst_230_);
v___f_242_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__0), 10, 8);
lean_closure_set(v___f_242_, 0, v_inst_230_);
lean_closure_set(v___f_242_, 1, v_inst_231_);
lean_closure_set(v___f_242_, 2, v_visitGoalPre_232_);
lean_closure_set(v___f_242_, 3, v_visitGoalPost_233_);
lean_closure_set(v___f_242_, 4, v_visitRappPre_234_);
lean_closure_set(v___f_242_, 5, v_visitRappPost_235_);
lean_closure_set(v___f_242_, 6, v_visitMVarClusterPre_236_);
lean_closure_set(v___f_242_, 7, v_visitMVarClusterPost_237_);
v___f_243_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__1), 3, 2);
lean_closure_set(v___f_243_, 0, v_visitGoalPost_233_);
lean_closure_set(v___f_243_, 1, v_gref_241_);
v___f_244_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__2), 6, 5);
lean_closure_set(v___f_244_, 0, v_toApplicative_239_);
lean_closure_set(v___f_244_, 1, v_toBind_240_);
lean_closure_set(v___f_244_, 2, v___f_243_);
lean_closure_set(v___f_244_, 3, v_inst_230_);
lean_closure_set(v___f_244_, 4, v___f_242_);
v___f_245_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__3___boxed), 6, 5);
lean_closure_set(v___f_245_, 0, v_toApplicative_239_);
lean_closure_set(v___f_245_, 1, v_gref_241_);
lean_closure_set(v___f_245_, 2, v_inst_231_);
lean_closure_set(v___f_245_, 3, v_toBind_240_);
lean_closure_set(v___f_245_, 4, v___f_244_);
v___x_246_ = lean_apply_1(v_visitGoalPre_232_, v_gref_241_);
v___x_247_ = lean_apply_4(v_toBind_240_, lean_box(0), lean_box(0), v___x_246_, v___f_245_);
return v___x_247_;
}
case 1:
{
lean_object* v_toApplicative_248_; lean_object* v_toBind_249_; lean_object* v_rref_250_; lean_object* v___f_251_; lean_object* v___f_252_; lean_object* v___f_253_; lean_object* v___f_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v_toApplicative_248_ = lean_ctor_get(v_inst_230_, 0);
lean_inc_ref_n(v_toApplicative_248_, 2);
v_toBind_249_ = lean_ctor_get(v_inst_230_, 1);
lean_inc_n(v_toBind_249_, 3);
v_rref_250_ = lean_ctor_get(v_x_238_, 0);
lean_inc_n(v_rref_250_, 3);
lean_dec_ref_known(v_x_238_, 1);
lean_inc(v_visitRappPost_235_);
lean_inc(v_visitRappPre_234_);
lean_inc(v_inst_231_);
lean_inc_ref(v_inst_230_);
v___f_251_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__4), 10, 8);
lean_closure_set(v___f_251_, 0, v_inst_230_);
lean_closure_set(v___f_251_, 1, v_inst_231_);
lean_closure_set(v___f_251_, 2, v_visitGoalPre_232_);
lean_closure_set(v___f_251_, 3, v_visitGoalPost_233_);
lean_closure_set(v___f_251_, 4, v_visitRappPre_234_);
lean_closure_set(v___f_251_, 5, v_visitRappPost_235_);
lean_closure_set(v___f_251_, 6, v_visitMVarClusterPre_236_);
lean_closure_set(v___f_251_, 7, v_visitMVarClusterPost_237_);
v___f_252_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__5), 3, 2);
lean_closure_set(v___f_252_, 0, v_visitRappPost_235_);
lean_closure_set(v___f_252_, 1, v_rref_250_);
v___f_253_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__6), 6, 5);
lean_closure_set(v___f_253_, 0, v_toApplicative_248_);
lean_closure_set(v___f_253_, 1, v_toBind_249_);
lean_closure_set(v___f_253_, 2, v___f_252_);
lean_closure_set(v___f_253_, 3, v_inst_230_);
lean_closure_set(v___f_253_, 4, v___f_251_);
v___f_254_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_254_, 0, v_toApplicative_248_);
lean_closure_set(v___f_254_, 1, v_rref_250_);
lean_closure_set(v___f_254_, 2, v_inst_231_);
lean_closure_set(v___f_254_, 3, v_toBind_249_);
lean_closure_set(v___f_254_, 4, v___f_253_);
v___x_255_ = lean_apply_1(v_visitRappPre_234_, v_rref_250_);
v___x_256_ = lean_apply_4(v_toBind_249_, lean_box(0), lean_box(0), v___x_255_, v___f_254_);
return v___x_256_;
}
default: 
{
lean_object* v_toApplicative_257_; lean_object* v_toBind_258_; lean_object* v_cref_259_; lean_object* v___f_260_; lean_object* v___f_261_; lean_object* v___f_262_; lean_object* v___f_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v_toApplicative_257_ = lean_ctor_get(v_inst_230_, 0);
lean_inc_ref_n(v_toApplicative_257_, 2);
v_toBind_258_ = lean_ctor_get(v_inst_230_, 1);
lean_inc_n(v_toBind_258_, 3);
v_cref_259_ = lean_ctor_get(v_x_238_, 0);
lean_inc_n(v_cref_259_, 3);
lean_dec_ref_known(v_x_238_, 1);
lean_inc(v_visitMVarClusterPost_237_);
lean_inc(v_visitMVarClusterPre_236_);
lean_inc(v_inst_231_);
lean_inc_ref(v_inst_230_);
v___f_260_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__8), 10, 8);
lean_closure_set(v___f_260_, 0, v_inst_230_);
lean_closure_set(v___f_260_, 1, v_inst_231_);
lean_closure_set(v___f_260_, 2, v_visitGoalPre_232_);
lean_closure_set(v___f_260_, 3, v_visitGoalPost_233_);
lean_closure_set(v___f_260_, 4, v_visitRappPre_234_);
lean_closure_set(v___f_260_, 5, v_visitRappPost_235_);
lean_closure_set(v___f_260_, 6, v_visitMVarClusterPre_236_);
lean_closure_set(v___f_260_, 7, v_visitMVarClusterPost_237_);
v___f_261_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__9), 3, 2);
lean_closure_set(v___f_261_, 0, v_visitMVarClusterPost_237_);
lean_closure_set(v___f_261_, 1, v_cref_259_);
v___f_262_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__10), 6, 5);
lean_closure_set(v___f_262_, 0, v_toApplicative_257_);
lean_closure_set(v___f_262_, 1, v_toBind_258_);
lean_closure_set(v___f_262_, 2, v___f_261_);
lean_closure_set(v___f_262_, 3, v_inst_230_);
lean_closure_set(v___f_262_, 4, v___f_260_);
v___f_263_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__11___boxed), 6, 5);
lean_closure_set(v___f_263_, 0, v_toApplicative_257_);
lean_closure_set(v___f_263_, 1, v_cref_259_);
lean_closure_set(v___f_263_, 2, v_inst_231_);
lean_closure_set(v___f_263_, 3, v_toBind_258_);
lean_closure_set(v___f_263_, 4, v___f_262_);
v___x_264_ = lean_apply_1(v_visitMVarClusterPre_236_, v_cref_259_);
v___x_265_ = lean_apply_4(v_toBind_258_, lean_box(0), lean_box(0), v___x_264_, v___f_263_);
return v___x_265_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___redArg___lam__0(lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_visitGoalPre_268_, lean_object* v_visitGoalPost_269_, lean_object* v_visitRappPre_270_, lean_object* v_visitRappPost_271_, lean_object* v_visitMVarClusterPre_272_, lean_object* v_visitMVarClusterPost_273_, lean_object* v_x_274_, lean_object* v___y_275_){
_start:
{
lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_276_, 0, v___y_275_);
v___x_277_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_266_, v_inst_267_, v_visitGoalPre_268_, v_visitGoalPost_269_, v_visitRappPre_270_, v_visitRappPost_271_, v_visitMVarClusterPre_272_, v_visitMVarClusterPost_273_, v___x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown(lean_object* v_m_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_visitGoalPre_281_, lean_object* v_visitGoalPost_282_, lean_object* v_visitRappPre_283_, lean_object* v_visitRappPost_284_, lean_object* v_visitMVarClusterPre_285_, lean_object* v_visitMVarClusterPost_286_, lean_object* v_x_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_279_, v_inst_280_, v_visitGoalPre_281_, v_visitGoalPost_282_, v_visitRappPre_283_, v_visitRappPost_284_, v_visitMVarClusterPre_285_, v_visitMVarClusterPost_286_, v_x_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg___lam__3(lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_visitGoalPre_291_, lean_object* v_visitGoalPost_292_, lean_object* v_visitRappPre_293_, lean_object* v_visitRappPost_294_, lean_object* v_visitMVarClusterPre_295_, lean_object* v_visitMVarClusterPost_296_, lean_object* v_toBind_297_, lean_object* v___f_298_, lean_object* v_____do__lift_299_){
_start:
{
lean_object* v___x_300_; lean_object* v_elimRapp_301_; lean_object* v___x_302_; lean_object* v_parent_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_300_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_301_ = lean_ctor_get(v___x_300_, 3);
lean_inc_ref(v_elimRapp_301_);
v___x_302_ = lean_apply_1(v_elimRapp_301_, v_____do__lift_299_);
v_parent_303_ = lean_ctor_get(v___x_302_, 1);
lean_inc(v_parent_303_);
lean_dec_ref(v___x_302_);
v___x_304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_304_, 0, v_parent_303_);
v___x_305_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_289_, v_inst_290_, v_visitGoalPre_291_, v_visitGoalPost_292_, v_visitRappPre_293_, v_visitRappPost_294_, v_visitMVarClusterPre_295_, v_visitMVarClusterPost_296_, v___x_304_);
v___x_306_ = lean_apply_4(v_toBind_297_, lean_box(0), lean_box(0), v___x_305_, v___f_298_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg___lam__4(lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_visitGoalPre_309_, lean_object* v_visitGoalPost_310_, lean_object* v_visitRappPre_311_, lean_object* v_visitRappPost_312_, lean_object* v_visitMVarClusterPre_313_, lean_object* v_visitMVarClusterPost_314_, lean_object* v_toBind_315_, lean_object* v___f_316_, lean_object* v_cref_317_, lean_object* v_____do__lift_318_){
_start:
{
lean_object* v___x_319_; lean_object* v_elimMVarCluster_320_; lean_object* v___x_321_; lean_object* v_parent_x3f_322_; 
v___x_319_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_320_ = lean_ctor_get(v___x_319_, 5);
lean_inc_ref(v_elimMVarCluster_320_);
v___x_321_ = lean_apply_1(v_elimMVarCluster_320_, v_____do__lift_318_);
v_parent_x3f_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc(v_parent_x3f_322_);
lean_dec_ref(v___x_321_);
if (lean_obj_tag(v_parent_x3f_322_) == 1)
{
lean_object* v_val_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_332_; 
lean_dec(v_cref_317_);
v_val_323_ = lean_ctor_get(v_parent_x3f_322_, 0);
v_isSharedCheck_332_ = !lean_is_exclusive(v_parent_x3f_322_);
if (v_isSharedCheck_332_ == 0)
{
v___x_325_ = v_parent_x3f_322_;
v_isShared_326_ = v_isSharedCheck_332_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_val_323_);
lean_dec(v_parent_x3f_322_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_332_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v_val_323_);
v___x_328_ = v_reuseFailAlloc_331_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_329_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_307_, v_inst_308_, v_visitGoalPre_309_, v_visitGoalPost_310_, v_visitRappPre_311_, v_visitRappPost_312_, v_visitMVarClusterPre_313_, v_visitMVarClusterPost_314_, v___x_328_);
v___x_330_ = lean_apply_4(v_toBind_315_, lean_box(0), lean_box(0), v___x_329_, v___f_316_);
return v___x_330_;
}
}
}
else
{
lean_object* v___x_333_; 
lean_dec(v_parent_x3f_322_);
lean_dec(v___f_316_);
lean_dec(v_toBind_315_);
lean_dec(v_visitMVarClusterPre_313_);
lean_dec(v_visitRappPost_312_);
lean_dec(v_visitRappPre_311_);
lean_dec(v_visitGoalPost_310_);
lean_dec(v_visitGoalPre_309_);
lean_dec(v_inst_308_);
lean_dec_ref(v_inst_307_);
v___x_333_ = lean_apply_1(v_visitMVarClusterPost_314_, v_cref_317_);
return v___x_333_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg(lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_visitGoalPre_336_, lean_object* v_visitGoalPost_337_, lean_object* v_visitRappPre_338_, lean_object* v_visitRappPost_339_, lean_object* v_visitMVarClusterPre_340_, lean_object* v_visitMVarClusterPost_341_, lean_object* v_x_342_){
_start:
{
switch(lean_obj_tag(v_x_342_))
{
case 0:
{
lean_object* v_toApplicative_343_; lean_object* v_toBind_344_; lean_object* v_gref_345_; lean_object* v___f_346_; lean_object* v___f_347_; lean_object* v___f_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v_toApplicative_343_ = lean_ctor_get(v_inst_334_, 0);
lean_inc_ref(v_toApplicative_343_);
v_toBind_344_ = lean_ctor_get(v_inst_334_, 1);
lean_inc_n(v_toBind_344_, 3);
v_gref_345_ = lean_ctor_get(v_x_342_, 0);
lean_inc_n(v_gref_345_, 3);
lean_dec_ref_known(v_x_342_, 1);
lean_inc(v_visitGoalPost_337_);
v___f_346_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__1), 3, 2);
lean_closure_set(v___f_346_, 0, v_visitGoalPost_337_);
lean_closure_set(v___f_346_, 1, v_gref_345_);
lean_inc(v_visitGoalPre_336_);
lean_inc(v_inst_335_);
v___f_347_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseUp___redArg___lam__1), 11, 10);
lean_closure_set(v___f_347_, 0, v_inst_334_);
lean_closure_set(v___f_347_, 1, v_inst_335_);
lean_closure_set(v___f_347_, 2, v_visitGoalPre_336_);
lean_closure_set(v___f_347_, 3, v_visitGoalPost_337_);
lean_closure_set(v___f_347_, 4, v_visitRappPre_338_);
lean_closure_set(v___f_347_, 5, v_visitRappPost_339_);
lean_closure_set(v___f_347_, 6, v_visitMVarClusterPre_340_);
lean_closure_set(v___f_347_, 7, v_visitMVarClusterPost_341_);
lean_closure_set(v___f_347_, 8, v_toBind_344_);
lean_closure_set(v___f_347_, 9, v___f_346_);
v___f_348_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__3___boxed), 6, 5);
lean_closure_set(v___f_348_, 0, v_toApplicative_343_);
lean_closure_set(v___f_348_, 1, v_gref_345_);
lean_closure_set(v___f_348_, 2, v_inst_335_);
lean_closure_set(v___f_348_, 3, v_toBind_344_);
lean_closure_set(v___f_348_, 4, v___f_347_);
v___x_349_ = lean_apply_1(v_visitGoalPre_336_, v_gref_345_);
v___x_350_ = lean_apply_4(v_toBind_344_, lean_box(0), lean_box(0), v___x_349_, v___f_348_);
return v___x_350_;
}
case 1:
{
lean_object* v_toApplicative_351_; lean_object* v_toBind_352_; lean_object* v_rref_353_; lean_object* v___f_354_; lean_object* v___f_355_; lean_object* v___f_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v_toApplicative_351_ = lean_ctor_get(v_inst_334_, 0);
lean_inc_ref(v_toApplicative_351_);
v_toBind_352_ = lean_ctor_get(v_inst_334_, 1);
lean_inc_n(v_toBind_352_, 3);
v_rref_353_ = lean_ctor_get(v_x_342_, 0);
lean_inc_n(v_rref_353_, 3);
lean_dec_ref_known(v_x_342_, 1);
lean_inc(v_visitRappPost_339_);
v___f_354_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__5), 3, 2);
lean_closure_set(v___f_354_, 0, v_visitRappPost_339_);
lean_closure_set(v___f_354_, 1, v_rref_353_);
lean_inc(v_visitRappPre_338_);
lean_inc(v_inst_335_);
v___f_355_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseUp___redArg___lam__3), 11, 10);
lean_closure_set(v___f_355_, 0, v_inst_334_);
lean_closure_set(v___f_355_, 1, v_inst_335_);
lean_closure_set(v___f_355_, 2, v_visitGoalPre_336_);
lean_closure_set(v___f_355_, 3, v_visitGoalPost_337_);
lean_closure_set(v___f_355_, 4, v_visitRappPre_338_);
lean_closure_set(v___f_355_, 5, v_visitRappPost_339_);
lean_closure_set(v___f_355_, 6, v_visitMVarClusterPre_340_);
lean_closure_set(v___f_355_, 7, v_visitMVarClusterPost_341_);
lean_closure_set(v___f_355_, 8, v_toBind_352_);
lean_closure_set(v___f_355_, 9, v___f_354_);
v___f_356_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_356_, 0, v_toApplicative_351_);
lean_closure_set(v___f_356_, 1, v_rref_353_);
lean_closure_set(v___f_356_, 2, v_inst_335_);
lean_closure_set(v___f_356_, 3, v_toBind_352_);
lean_closure_set(v___f_356_, 4, v___f_355_);
v___x_357_ = lean_apply_1(v_visitRappPre_338_, v_rref_353_);
v___x_358_ = lean_apply_4(v_toBind_352_, lean_box(0), lean_box(0), v___x_357_, v___f_356_);
return v___x_358_;
}
default: 
{
lean_object* v_toApplicative_359_; lean_object* v_toBind_360_; lean_object* v_cref_361_; lean_object* v___f_362_; lean_object* v___f_363_; lean_object* v___f_364_; lean_object* v___x_365_; lean_object* v___x_366_; 
v_toApplicative_359_ = lean_ctor_get(v_inst_334_, 0);
lean_inc_ref(v_toApplicative_359_);
v_toBind_360_ = lean_ctor_get(v_inst_334_, 1);
lean_inc_n(v_toBind_360_, 3);
v_cref_361_ = lean_ctor_get(v_x_342_, 0);
lean_inc_n(v_cref_361_, 4);
lean_dec_ref_known(v_x_342_, 1);
lean_inc(v_visitMVarClusterPost_341_);
v___f_362_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__9), 3, 2);
lean_closure_set(v___f_362_, 0, v_visitMVarClusterPost_341_);
lean_closure_set(v___f_362_, 1, v_cref_361_);
lean_inc(v_visitMVarClusterPre_340_);
lean_inc(v_inst_335_);
v___f_363_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseUp___redArg___lam__4), 12, 11);
lean_closure_set(v___f_363_, 0, v_inst_334_);
lean_closure_set(v___f_363_, 1, v_inst_335_);
lean_closure_set(v___f_363_, 2, v_visitGoalPre_336_);
lean_closure_set(v___f_363_, 3, v_visitGoalPost_337_);
lean_closure_set(v___f_363_, 4, v_visitRappPre_338_);
lean_closure_set(v___f_363_, 5, v_visitRappPost_339_);
lean_closure_set(v___f_363_, 6, v_visitMVarClusterPre_340_);
lean_closure_set(v___f_363_, 7, v_visitMVarClusterPost_341_);
lean_closure_set(v___f_363_, 8, v_toBind_360_);
lean_closure_set(v___f_363_, 9, v___f_362_);
lean_closure_set(v___f_363_, 10, v_cref_361_);
v___f_364_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traverseDown___redArg___lam__11___boxed), 6, 5);
lean_closure_set(v___f_364_, 0, v_toApplicative_359_);
lean_closure_set(v___f_364_, 1, v_cref_361_);
lean_closure_set(v___f_364_, 2, v_inst_335_);
lean_closure_set(v___f_364_, 3, v_toBind_360_);
lean_closure_set(v___f_364_, 4, v___f_363_);
v___x_365_ = lean_apply_1(v_visitMVarClusterPre_340_, v_cref_361_);
v___x_366_ = lean_apply_4(v_toBind_360_, lean_box(0), lean_box(0), v___x_365_, v___f_364_);
return v___x_366_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___redArg___lam__1(lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_visitGoalPre_369_, lean_object* v_visitGoalPost_370_, lean_object* v_visitRappPre_371_, lean_object* v_visitRappPost_372_, lean_object* v_visitMVarClusterPre_373_, lean_object* v_visitMVarClusterPost_374_, lean_object* v_toBind_375_, lean_object* v___f_376_, lean_object* v_____do__lift_377_){
_start:
{
lean_object* v___x_378_; lean_object* v_elimGoal_379_; lean_object* v___x_380_; lean_object* v_parent_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_378_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_379_ = lean_ctor_get(v___x_378_, 1);
lean_inc_ref(v_elimGoal_379_);
v___x_380_ = lean_apply_1(v_elimGoal_379_, v_____do__lift_377_);
v_parent_381_ = lean_ctor_get(v___x_380_, 1);
lean_inc(v_parent_381_);
lean_dec_ref(v___x_380_);
v___x_382_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_382_, 0, v_parent_381_);
v___x_383_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_367_, v_inst_368_, v_visitGoalPre_369_, v_visitGoalPost_370_, v_visitRappPre_371_, v_visitRappPost_372_, v_visitMVarClusterPre_373_, v_visitMVarClusterPost_374_, v___x_382_);
v___x_384_ = lean_apply_4(v_toBind_375_, lean_box(0), lean_box(0), v___x_383_, v___f_376_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp(lean_object* v_m_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_visitGoalPre_388_, lean_object* v_visitGoalPost_389_, lean_object* v_visitRappPre_390_, lean_object* v_visitRappPost_391_, lean_object* v_visitMVarClusterPre_392_, lean_object* v_visitMVarClusterPost_393_, lean_object* v_x_394_){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_386_, v_inst_387_, v_visitGoalPre_388_, v_visitGoalPost_389_, v_visitRappPre_390_, v_visitRappPost_391_, v_visitMVarClusterPre_392_, v_visitMVarClusterPost_393_, v_x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown___redArg___lam__0(lean_object* v_toPure_396_, lean_object* v_x_397_){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = lean_box(0);
v___x_399_ = lean_apply_2(v_toPure_396_, lean_box(0), v___x_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown___redArg___lam__0___boxed(lean_object* v_toPure_400_, lean_object* v_x_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_aesop_Aesop_preTraverseDown___redArg___lam__0(v_toPure_400_, v_x_401_);
lean_dec(v_x_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown___redArg(lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_visitGoal_405_, lean_object* v_visitRapp_406_, lean_object* v_visitMVarCluster_407_, lean_object* v_a_408_){
_start:
{
lean_object* v_toApplicative_409_; lean_object* v_toPure_410_; lean_object* v___f_411_; lean_object* v___x_412_; 
v_toApplicative_409_ = lean_ctor_get(v_inst_403_, 0);
v_toPure_410_ = lean_ctor_get(v_toApplicative_409_, 1);
lean_inc(v_toPure_410_);
v___f_411_ = lean_alloc_closure((void*)(lp_aesop_Aesop_preTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_411_, 0, v_toPure_410_);
lean_inc_ref_n(v___f_411_, 2);
v___x_412_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_403_, v_inst_404_, v_visitGoal_405_, v___f_411_, v_visitRapp_406_, v___f_411_, v_visitMVarCluster_407_, v___f_411_, v_a_408_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseDown(lean_object* v_m_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_visitGoal_416_, lean_object* v_visitRapp_417_, lean_object* v_visitMVarCluster_418_, lean_object* v_a_419_){
_start:
{
lean_object* v_toApplicative_420_; lean_object* v_toPure_421_; lean_object* v___f_422_; lean_object* v___x_423_; 
v_toApplicative_420_ = lean_ctor_get(v_inst_414_, 0);
v_toPure_421_ = lean_ctor_get(v_toApplicative_420_, 1);
lean_inc(v_toPure_421_);
v___f_422_ = lean_alloc_closure((void*)(lp_aesop_Aesop_preTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_422_, 0, v_toPure_421_);
lean_inc_ref_n(v___f_422_, 2);
v___x_423_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_414_, v_inst_415_, v_visitGoal_416_, v___f_422_, v_visitRapp_417_, v___f_422_, v_visitMVarCluster_418_, v___f_422_, v_a_419_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseUp___redArg(lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_visitGoal_426_, lean_object* v_visitRapp_427_, lean_object* v_visitMVarCluster_428_, lean_object* v_a_429_){
_start:
{
lean_object* v_toApplicative_430_; lean_object* v_toPure_431_; lean_object* v___f_432_; lean_object* v___x_433_; 
v_toApplicative_430_ = lean_ctor_get(v_inst_424_, 0);
v_toPure_431_ = lean_ctor_get(v_toApplicative_430_, 1);
lean_inc(v_toPure_431_);
v___f_432_ = lean_alloc_closure((void*)(lp_aesop_Aesop_preTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_432_, 0, v_toPure_431_);
lean_inc_ref_n(v___f_432_, 2);
v___x_433_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_424_, v_inst_425_, v_visitGoal_426_, v___f_432_, v_visitRapp_427_, v___f_432_, v_visitMVarCluster_428_, v___f_432_, v_a_429_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_preTraverseUp(lean_object* v_m_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_visitGoal_437_, lean_object* v_visitRapp_438_, lean_object* v_visitMVarCluster_439_, lean_object* v_a_440_){
_start:
{
lean_object* v_toApplicative_441_; lean_object* v_toPure_442_; lean_object* v___f_443_; lean_object* v___x_444_; 
v_toApplicative_441_ = lean_ctor_get(v_inst_435_, 0);
v_toPure_442_ = lean_ctor_get(v_toApplicative_441_, 1);
lean_inc(v_toPure_442_);
v___f_443_ = lean_alloc_closure((void*)(lp_aesop_Aesop_preTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_443_, 0, v_toPure_442_);
lean_inc_ref_n(v___f_443_, 2);
v___x_444_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_435_, v_inst_436_, v_visitGoal_437_, v___f_443_, v_visitRapp_438_, v___f_443_, v_visitMVarCluster_439_, v___f_443_, v_a_440_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown___redArg___lam__0(lean_object* v_toPure_445_, lean_object* v_x_446_){
_start:
{
uint8_t v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_447_ = 1;
v___x_448_ = lean_box(v___x_447_);
v___x_449_ = lean_apply_2(v_toPure_445_, lean_box(0), v___x_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown___redArg___lam__0___boxed(lean_object* v_toPure_450_, lean_object* v_x_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_aesop_Aesop_postTraverseDown___redArg___lam__0(v_toPure_450_, v_x_451_);
lean_dec(v_x_451_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown___redArg(lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_visitGoal_455_, lean_object* v_visitRapp_456_, lean_object* v_visitMVarCluster_457_, lean_object* v_a_458_){
_start:
{
lean_object* v_toApplicative_459_; lean_object* v_toPure_460_; lean_object* v___f_461_; lean_object* v___x_462_; 
v_toApplicative_459_ = lean_ctor_get(v_inst_453_, 0);
v_toPure_460_ = lean_ctor_get(v_toApplicative_459_, 1);
lean_inc(v_toPure_460_);
v___f_461_ = lean_alloc_closure((void*)(lp_aesop_Aesop_postTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_461_, 0, v_toPure_460_);
lean_inc_ref_n(v___f_461_, 2);
v___x_462_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_453_, v_inst_454_, v___f_461_, v_visitGoal_455_, v___f_461_, v_visitRapp_456_, v___f_461_, v_visitMVarCluster_457_, v_a_458_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseDown(lean_object* v_m_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_visitGoal_466_, lean_object* v_visitRapp_467_, lean_object* v_visitMVarCluster_468_, lean_object* v_a_469_){
_start:
{
lean_object* v_toApplicative_470_; lean_object* v_toPure_471_; lean_object* v___f_472_; lean_object* v___x_473_; 
v_toApplicative_470_ = lean_ctor_get(v_inst_464_, 0);
v_toPure_471_ = lean_ctor_get(v_toApplicative_470_, 1);
lean_inc(v_toPure_471_);
v___f_472_ = lean_alloc_closure((void*)(lp_aesop_Aesop_postTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_472_, 0, v_toPure_471_);
lean_inc_ref_n(v___f_472_, 2);
v___x_473_ = lp_aesop_Aesop_traverseDown___redArg(v_inst_464_, v_inst_465_, v___f_472_, v_visitGoal_466_, v___f_472_, v_visitRapp_467_, v___f_472_, v_visitMVarCluster_468_, v_a_469_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseUp___redArg(lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_visitGoal_476_, lean_object* v_visitRapp_477_, lean_object* v_visitMVarCluster_478_, lean_object* v_a_479_){
_start:
{
lean_object* v_toApplicative_480_; lean_object* v_toPure_481_; lean_object* v___f_482_; lean_object* v___x_483_; 
v_toApplicative_480_ = lean_ctor_get(v_inst_474_, 0);
v_toPure_481_ = lean_ctor_get(v_toApplicative_480_, 1);
lean_inc(v_toPure_481_);
v___f_482_ = lean_alloc_closure((void*)(lp_aesop_Aesop_postTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_482_, 0, v_toPure_481_);
lean_inc_ref_n(v___f_482_, 2);
v___x_483_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_474_, v_inst_475_, v___f_482_, v_visitGoal_476_, v___f_482_, v_visitRapp_477_, v___f_482_, v_visitMVarCluster_478_, v_a_479_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_postTraverseUp(lean_object* v_m_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_visitGoal_487_, lean_object* v_visitRapp_488_, lean_object* v_visitMVarCluster_489_, lean_object* v_a_490_){
_start:
{
lean_object* v_toApplicative_491_; lean_object* v_toPure_492_; lean_object* v___f_493_; lean_object* v___x_494_; 
v_toApplicative_491_ = lean_ctor_get(v_inst_485_, 0);
v_toPure_492_ = lean_ctor_get(v_toApplicative_491_, 1);
lean_inc(v_toPure_492_);
v___f_493_ = lean_alloc_closure((void*)(lp_aesop_Aesop_postTraverseDown___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_493_, 0, v_toPure_492_);
lean_inc_ref_n(v___f_493_, 2);
v___x_494_ = lp_aesop_Aesop_traverseUp___redArg(v_inst_485_, v_inst_486_, v___f_493_, v_visitGoal_487_, v___f_493_, v_visitRapp_488_, v___f_493_, v_visitMVarCluster_489_, v_a_490_);
return v___x_494_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_Traversal(builtin);
}
#ifdef __cplusplus
}
#endif
