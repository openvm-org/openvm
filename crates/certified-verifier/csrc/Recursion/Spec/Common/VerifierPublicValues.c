// Lean compiler output
// Module: Recursion.Spec.Common.VerifierPublicValues
// Imports: public import Init public meta import Init public import Recursion.Spec.Common.RawCarrier public import Recursion.Spec.Common.VmSegmentPublicValues public import Recursion.Spec.VerifierAirId
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
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Array_instDecidableEqImpl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_maxRecursionDepth;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_openVMVerifierPvsAirId;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_vkCommitWidth;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_verifierBasePvsWidth;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_openVMVmPvsAirId;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_openvm_x2dfv_Recursion_Spec_maxRecursionDepth(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(256u);
return v___x_1_;
}
}
static lean_object* _init_lp_openvm_x2dfv_Recursion_Spec_openVMVerifierPvsAirId(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
}
static lean_object* _init_lp_openvm_x2dfv_Recursion_Spec_vkCommitWidth(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(16u);
return v___x_3_;
}
}
static lean_object* _init_lp_openvm_x2dfv_Recursion_Spec_verifierBasePvsWidth(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(66u);
return v___x_4_;
}
}
static lean_object* _init_lp_openvm_x2dfv_Recursion_Spec_openVMVmPvsAirId(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(1u);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg___lam__0(lean_object* v_values_6_, lean_object* v_zero_7_, lean_object* v_i_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = l_List_getD___redArg(v_values_6_, v_i_8_, v_zero_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg___lam__0___boxed(lean_object* v_values_10_, lean_object* v_zero_11_, lean_object* v_i_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg___lam__0(v_values_10_, v_zero_11_, v_i_12_);
lean_dec(v_zero_11_);
lean_dec(v_values_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(lean_object* v_zero_14_, lean_object* v_values_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___f_16_ = lean_alloc_closure((void*)(lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_16_, 0, v_values_15_);
lean_closure_set(v___f_16_, 1, v_zero_14_);
v___x_17_ = lean_unsigned_to_nat(8u);
v___x_18_ = l_Array_ofFn___redArg(v___x_17_, v___f_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf(lean_object* v_F_19_, lean_object* v_zero_20_, lean_object* v_values_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(v_zero_20_, v_values_21_);
return v___x_22_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(lean_object* v_inst_23_, lean_object* v_x_24_, lean_object* v_x_25_){
_start:
{
lean_object* v_cachedCommit_26_; lean_object* v_vkPreHash_27_; lean_object* v_cachedCommit_28_; lean_object* v_vkPreHash_29_; uint8_t v___x_30_; 
v_cachedCommit_26_ = lean_ctor_get(v_x_24_, 0);
v_vkPreHash_27_ = lean_ctor_get(v_x_24_, 1);
v_cachedCommit_28_ = lean_ctor_get(v_x_25_, 0);
v_vkPreHash_29_ = lean_ctor_get(v_x_25_, 1);
lean_inc_ref(v_inst_23_);
v___x_30_ = l_Array_instDecidableEqImpl___redArg(v_inst_23_, v_cachedCommit_26_, v_cachedCommit_28_);
if (v___x_30_ == 0)
{
lean_dec_ref(v_inst_23_);
return v___x_30_;
}
else
{
uint8_t v___x_31_; 
v___x_31_ = l_Array_instDecidableEqImpl___redArg(v_inst_23_, v_vkPreHash_27_, v_vkPreHash_29_);
return v___x_31_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg___boxed(lean_object* v_inst_32_, lean_object* v_x_33_, lean_object* v_x_34_){
_start:
{
uint8_t v_res_35_; lean_object* v_r_36_; 
v_res_35_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v_inst_32_, v_x_33_, v_x_34_);
lean_dec_ref(v_x_34_);
lean_dec_ref(v_x_33_);
v_r_36_ = lean_box(v_res_35_);
return v_r_36_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq(lean_object* v_F_37_, lean_object* v_inst_38_, lean_object* v_x_39_, lean_object* v_x_40_){
_start:
{
uint8_t v___x_41_; 
v___x_41_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v_inst_38_, v_x_39_, v_x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___boxed(lean_object* v_F_42_, lean_object* v_inst_43_, lean_object* v_x_44_, lean_object* v_x_45_){
_start:
{
uint8_t v_res_46_; lean_object* v_r_47_; 
v_res_46_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq(v_F_42_, v_inst_43_, v_x_44_, v_x_45_);
lean_dec_ref(v_x_45_);
lean_dec_ref(v_x_44_);
v_r_47_ = lean_box(v_res_46_);
return v_r_47_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___redArg(lean_object* v_inst_48_, lean_object* v_x_49_, lean_object* v_x_50_){
_start:
{
uint8_t v___x_51_; 
v___x_51_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v_inst_48_, v_x_49_, v_x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___redArg___boxed(lean_object* v_inst_52_, lean_object* v_x_53_, lean_object* v_x_54_){
_start:
{
uint8_t v_res_55_; lean_object* v_r_56_; 
v_res_55_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___redArg(v_inst_52_, v_x_53_, v_x_54_);
lean_dec_ref(v_x_54_);
lean_dec_ref(v_x_53_);
v_r_56_ = lean_box(v_res_55_);
return v_r_56_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData(lean_object* v_F_57_, lean_object* v_inst_58_, lean_object* v_x_59_, lean_object* v_x_60_){
_start:
{
uint8_t v___x_61_; 
v___x_61_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v_inst_58_, v_x_59_, v_x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData___boxed(lean_object* v_F_62_, lean_object* v_inst_63_, lean_object* v_x_64_, lean_object* v_x_65_){
_start:
{
uint8_t v_res_66_; lean_object* v_r_67_; 
v_res_66_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData(v_F_62_, v_inst_63_, v_x_64_, v_x_65_);
lean_dec_ref(v_x_65_);
lean_dec_ref(v_x_64_);
v_r_67_ = lean_box(v_res_66_);
return v_r_67_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(lean_object* v_zero_68_, lean_object* v_values_69_, lean_object* v_offset_70_){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
lean_inc(v_offset_70_);
v___x_71_ = l_List_drop___redArg(v_offset_70_, v_values_69_);
lean_inc(v_zero_68_);
v___x_72_ = lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(v_zero_68_, v___x_71_);
v___x_73_ = lean_unsigned_to_nat(8u);
v___x_74_ = lean_nat_add(v_offset_70_, v___x_73_);
lean_dec(v_offset_70_);
v___x_75_ = l_List_drop___redArg(v___x_74_, v_values_69_);
v___x_76_ = lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(v_zero_68_, v___x_75_);
v___x_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_72_);
lean_ctor_set(v___x_77_, 1, v___x_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg___boxed(lean_object* v_zero_78_, lean_object* v_values_79_, lean_object* v_offset_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(v_zero_78_, v_values_79_, v_offset_80_);
lean_dec(v_values_79_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt(lean_object* v_F_82_, lean_object* v_zero_83_, lean_object* v_values_84_, lean_object* v_offset_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(v_zero_83_, v_values_84_, v_offset_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___boxed(lean_object* v_F_87_, lean_object* v_zero_88_, lean_object* v_values_89_, lean_object* v_offset_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt(v_F_87_, v_zero_88_, v_values_89_, v_offset_90_);
lean_dec(v_values_89_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(lean_object* v_zero_92_, lean_object* v_values_93_){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_94_ = l_List_lengthTR___redArg(v_values_93_);
v___x_95_ = lean_unsigned_to_nat(66u);
v___x_96_ = lean_nat_dec_eq(v___x_94_, v___x_95_);
lean_dec(v___x_94_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; 
lean_dec(v_zero_92_);
v___x_97_ = lean_box(0);
return v___x_97_;
}
else
{
if (lean_obj_tag(v_values_93_) == 1)
{
lean_object* v_head_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
v_head_98_ = lean_ctor_get(v_values_93_, 0);
v___x_99_ = lean_unsigned_to_nat(49u);
v___x_100_ = l_List_drop___redArg(v___x_99_, v_values_93_);
if (lean_obj_tag(v___x_100_) == 1)
{
lean_object* v_head_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_head_101_ = lean_ctor_get(v___x_100_, 0);
lean_inc(v_head_101_);
lean_dec_ref_known(v___x_100_, 2);
v___x_102_ = lean_unsigned_to_nat(1u);
lean_inc_n(v_zero_92_, 3);
v___x_103_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(v_zero_92_, v_values_93_, v___x_102_);
v___x_104_ = lean_unsigned_to_nat(17u);
v___x_105_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(v_zero_92_, v_values_93_, v___x_104_);
v___x_106_ = lean_unsigned_to_nat(33u);
v___x_107_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(v_zero_92_, v_values_93_, v___x_106_);
v___x_108_ = lean_unsigned_to_nat(50u);
v___x_109_ = lp_openvm_x2dfv_Recursion_Spec_VkCommitData_fromListAt___redArg(v_zero_92_, v_values_93_, v___x_108_);
lean_inc(v_head_98_);
v___x_110_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_110_, 0, v_head_98_);
lean_ctor_set(v___x_110_, 1, v___x_103_);
lean_ctor_set(v___x_110_, 2, v___x_105_);
lean_ctor_set(v___x_110_, 3, v___x_107_);
lean_ctor_set(v___x_110_, 4, v_head_101_);
lean_ctor_set(v___x_110_, 5, v___x_109_);
v___x_111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
return v___x_111_;
}
else
{
lean_object* v___x_112_; 
lean_dec(v___x_100_);
lean_dec(v_zero_92_);
v___x_112_ = lean_box(0);
return v___x_112_;
}
}
else
{
lean_object* v___x_113_; 
lean_dec(v_zero_92_);
v___x_113_ = lean_box(0);
return v___x_113_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg___boxed(lean_object* v_zero_114_, lean_object* v_values_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(v_zero_114_, v_values_115_);
lean_dec(v_values_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f(lean_object* v_F_117_, lean_object* v_zero_118_, lean_object* v_values_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(v_zero_118_, v_values_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___boxed(lean_object* v_F_121_, lean_object* v_zero_122_, lean_object* v_values_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f(v_F_121_, v_zero_122_, v_values_123_);
lean_dec(v_values_123_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f___redArg(lean_object* v_zero_125_, lean_object* v_values_126_){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = lean_array_to_list(v_values_126_);
v___x_128_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromList_x3f___redArg(v_zero_125_, v___x_127_);
lean_dec(v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f(lean_object* v_F_129_, lean_object* v_n_130_, lean_object* v_zero_131_, lean_object* v_values_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f___redArg(v_zero_131_, v_values_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f___boxed(lean_object* v_F_134_, lean_object* v_n_135_, lean_object* v_zero_136_, lean_object* v_values_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_openvm_x2dfv_Recursion_Spec_VerifierBasePvsData_fromVector_x3f(v_F_134_, v_n_135_, v_zero_136_, v_values_137_);
lean_dec(v_n_135_);
return v_res_138_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_Recursion_Spec_Common_RawCarrier(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_Recursion_Spec_Common_VmSegmentPublicValues(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_Recursion_Spec_VerifierAirId(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_Recursion_Spec_Common_VerifierPublicValues(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_Recursion_Spec_Common_RawCarrier(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_Recursion_Spec_Common_VmSegmentPublicValues(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_Recursion_Spec_VerifierAirId(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_Recursion_Spec_maxRecursionDepth = _init_lp_openvm_x2dfv_Recursion_Spec_maxRecursionDepth();
lean_mark_persistent(lp_openvm_x2dfv_Recursion_Spec_maxRecursionDepth);
lp_openvm_x2dfv_Recursion_Spec_openVMVerifierPvsAirId = _init_lp_openvm_x2dfv_Recursion_Spec_openVMVerifierPvsAirId();
lean_mark_persistent(lp_openvm_x2dfv_Recursion_Spec_openVMVerifierPvsAirId);
lp_openvm_x2dfv_Recursion_Spec_vkCommitWidth = _init_lp_openvm_x2dfv_Recursion_Spec_vkCommitWidth();
lean_mark_persistent(lp_openvm_x2dfv_Recursion_Spec_vkCommitWidth);
lp_openvm_x2dfv_Recursion_Spec_verifierBasePvsWidth = _init_lp_openvm_x2dfv_Recursion_Spec_verifierBasePvsWidth();
lean_mark_persistent(lp_openvm_x2dfv_Recursion_Spec_verifierBasePvsWidth);
lp_openvm_x2dfv_Recursion_Spec_openVMVmPvsAirId = _init_lp_openvm_x2dfv_Recursion_Spec_openVMVmPvsAirId();
lean_mark_persistent(lp_openvm_x2dfv_Recursion_Spec_openVMVmPvsAirId);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
