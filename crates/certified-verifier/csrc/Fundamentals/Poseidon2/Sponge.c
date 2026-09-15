// Lean compiler output
// Module: Fundamentals.Poseidon2.Sponge
// Imports: public import Init public meta import Init public import Fundamentals.Poseidon2.Generic
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
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_stateAt___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_permute___redArg(lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint8_t l_List_beq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressWithCapacity___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressWithCapacity(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSliceState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSliceState(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg___lam__0(lean_object* v_to_1_, lean_object* v_x_2_){
_start:
{
lean_object* v_fieldOps_3_; lean_object* v_toRingOps_4_; lean_object* v_toSemiringOps_5_; lean_object* v_zero_6_; 
v_fieldOps_3_ = lean_ctor_get(v_to_1_, 0);
v_toRingOps_4_ = lean_ctor_get(v_fieldOps_3_, 0);
v_toSemiringOps_5_ = lean_ctor_get(v_toRingOps_4_, 0);
v_zero_6_ = lean_ctor_get(v_toSemiringOps_5_, 0);
lean_inc(v_zero_6_);
return v_zero_6_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg___lam__0___boxed(lean_object* v_to_7_, lean_object* v_x_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg___lam__0(v_to_7_, v_x_8_);
lean_dec(v_x_8_);
lean_dec_ref(v_to_7_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg(lean_object* v_to_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___f_11_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_11_, 0, v_to_10_);
v___x_12_ = lean_unsigned_to_nat(16u);
v___x_13_ = l_Array_ofFn___redArg(v___x_12_, v___f_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState(lean_object* v_F_14_, lean_object* v_EF_15_, lean_object* v_to_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg(v_to_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressWithCapacity___redArg(lean_object* v_to_18_, lean_object* v_left_19_, lean_object* v_right_20_){
_start:
{
lean_object* v_fieldOps_21_; lean_object* v___x_22_; 
v_fieldOps_21_ = lean_ctor_get(v_to_18_, 0);
lean_inc_ref(v_fieldOps_21_);
lean_dec_ref(v_to_18_);
v___x_22_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(v_fieldOps_21_, v_left_19_, v_right_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressWithCapacity(lean_object* v_F_23_, lean_object* v_EF_24_, lean_object* v_to_25_, lean_object* v_left_26_, lean_object* v_right_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressWithCapacity___redArg(v_to_25_, v_left_26_, v_right_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg___lam__0(lean_object* v_values_29_, lean_object* v_state_30_, lean_object* v_idx_31_){
_start:
{
lean_object* v___x_32_; 
lean_inc(v_idx_31_);
v___x_32_ = l_List_get_x3fInternal___redArg(v_values_29_, v_idx_31_);
if (lean_obj_tag(v___x_32_) == 0)
{
lean_object* v___x_33_; 
v___x_33_ = lean_array_fget_borrowed(v_state_30_, v_idx_31_);
lean_dec(v_idx_31_);
lean_inc(v___x_33_);
return v___x_33_;
}
else
{
lean_object* v_val_34_; 
lean_dec(v_idx_31_);
v_val_34_ = lean_ctor_get(v___x_32_, 0);
lean_inc(v_val_34_);
lean_dec_ref_known(v___x_32_, 1);
return v_val_34_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg___lam__0___boxed(lean_object* v_values_35_, lean_object* v_state_36_, lean_object* v_idx_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg___lam__0(v_values_35_, v_state_36_, v_idx_37_);
lean_dec_ref(v_state_36_);
lean_dec(v_values_35_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg(lean_object* v_state_39_, lean_object* v_values_40_){
_start:
{
lean_object* v___f_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___f_41_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_41_, 0, v_values_40_);
lean_closure_set(v___f_41_, 1, v_state_39_);
v___x_42_ = lean_unsigned_to_nat(16u);
v___x_43_ = l_Array_ofFn___redArg(v___x_42_, v___f_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix(lean_object* v_F_44_, lean_object* v_state_45_, lean_object* v_values_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg(v_state_45_, v_values_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg(lean_object* v_to_50_, lean_object* v_x_51_, lean_object* v_x_52_, lean_object* v_x_53_){
_start:
{
lean_object* v_zero_54_; uint8_t v_isZero_55_; 
v_zero_54_ = lean_unsigned_to_nat(0u);
v_isZero_55_ = lean_nat_dec_eq(v_x_51_, v_zero_54_);
if (v_isZero_55_ == 1)
{
lean_dec(v_x_53_);
lean_dec(v_x_51_);
lean_dec_ref(v_to_50_);
return v_x_52_;
}
else
{
if (lean_obj_tag(v_x_53_) == 0)
{
lean_dec(v_x_51_);
lean_dec_ref(v_to_50_);
return v_x_52_;
}
else
{
lean_object* v_fieldOps_56_; lean_object* v_one_57_; lean_object* v_n_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v_overwritten_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v_fieldOps_56_ = lean_ctor_get(v_to_50_, 0);
v_one_57_ = lean_unsigned_to_nat(1u);
v_n_58_ = lean_nat_sub(v_x_51_, v_one_57_);
lean_dec(v_x_51_);
v___x_59_ = lean_unsigned_to_nat(8u);
v___x_60_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg___closed__0));
lean_inc(v_x_53_);
v___x_61_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_x_53_, v_x_53_, v___x_59_, v___x_60_);
v_overwritten_62_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_overwritePrefix___redArg(v_x_52_, v___x_61_);
lean_inc_ref(v_fieldOps_56_);
v___x_63_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_permute___redArg(v_fieldOps_56_, v_overwritten_62_);
v___x_64_ = l_List_drop___redArg(v___x_59_, v_x_53_);
lean_dec(v_x_53_);
v_x_51_ = v_n_58_;
v_x_52_ = v___x_63_;
v_x_53_ = v___x_64_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks(lean_object* v_F_66_, lean_object* v_EF_67_, lean_object* v_to_68_, lean_object* v_x_69_, lean_object* v_x_70_, lean_object* v_x_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg(v_to_68_, v_x_69_, v_x_70_, v_x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSliceState___redArg(lean_object* v_to_73_, lean_object* v_values_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = l_List_lengthTR___redArg(v_values_74_);
lean_inc_ref(v_to_73_);
v___x_76_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_zeroState___redArg(v_to_73_);
v___x_77_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_absorbBlocks___redArg(v_to_73_, v___x_75_, v___x_76_, v_values_74_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSliceState(lean_object* v_F_78_, lean_object* v_EF_79_, lean_object* v_to_80_, lean_object* v_values_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSliceState___redArg(v_to_80_, v_values_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg___lam__0(lean_object* v_to_83_, lean_object* v_state_84_, lean_object* v_idx_85_){
_start:
{
lean_object* v_fieldOps_86_; lean_object* v___x_87_; 
v_fieldOps_86_ = lean_ctor_get(v_to_83_, 0);
v___x_87_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fieldOps_86_, v_state_84_, v_idx_85_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg___lam__0___boxed(lean_object* v_to_88_, lean_object* v_state_89_, lean_object* v_idx_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg___lam__0(v_to_88_, v_state_89_, v_idx_90_);
lean_dec_ref(v_to_88_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg(lean_object* v_to_92_, lean_object* v_state_93_){
_start:
{
lean_object* v___f_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___f_94_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_94_, 0, v_to_92_);
lean_closure_set(v___f_94_, 1, v_state_93_);
v___x_95_ = lean_unsigned_to_nat(8u);
v___x_96_ = l_Array_ofFn___redArg(v___x_95_, v___f_94_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest(lean_object* v_F_97_, lean_object* v_EF_98_, lean_object* v_to_99_, lean_object* v_state_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg(v_to_99_, v_state_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice___redArg(lean_object* v_to_102_, lean_object* v_values_103_){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_inc_ref(v_to_102_);
v___x_104_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSliceState___redArg(v_to_102_, v_values_103_);
v___x_105_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_squeezeDigest___redArg(v_to_102_, v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice(lean_object* v_F_106_, lean_object* v_EF_107_, lean_object* v_to_108_, lean_object* v_values_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_hashSlice___redArg(v_to_108_, v_values_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(lean_object* v_to_111_, lean_object* v_left_112_, lean_object* v_right_113_){
_start:
{
lean_object* v___x_114_; lean_object* v_fst_115_; 
v___x_114_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressWithCapacity___redArg(v_to_111_, v_left_112_, v_right_113_);
v_fst_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_fst_115_);
lean_dec_ref(v___x_114_);
return v_fst_115_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest(lean_object* v_F_116_, lean_object* v_EF_117_, lean_object* v_to_118_, lean_object* v_left_119_, lean_object* v_right_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(v_to_118_, v_left_119_, v_right_120_);
return v___x_121_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg(lean_object* v_inst_122_, lean_object* v_a_123_, lean_object* v_b_124_){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_125_ = lean_array_to_list(v_a_123_);
v___x_126_ = lean_array_to_list(v_b_124_);
v___x_127_ = l_List_beq___redArg(v_inst_122_, v___x_125_, v___x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg___boxed(lean_object* v_inst_128_, lean_object* v_a_129_, lean_object* v_b_130_){
_start:
{
uint8_t v_res_131_; lean_object* v_r_132_; 
v_res_131_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg(v_inst_128_, v_a_129_, v_b_130_);
v_r_132_ = lean_box(v_res_131_);
return v_r_132_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq(lean_object* v_F_133_, lean_object* v_inst_134_, lean_object* v_a_135_, lean_object* v_b_136_){
_start:
{
uint8_t v___x_137_; 
v___x_137_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg(v_inst_134_, v_a_135_, v_b_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___boxed(lean_object* v_F_138_, lean_object* v_inst_139_, lean_object* v_a_140_, lean_object* v_b_141_){
_start:
{
uint8_t v_res_142_; lean_object* v_r_143_; 
v_res_142_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq(v_F_138_, v_inst_139_, v_a_140_, v_b_141_);
v_r_143_ = lean_box(v_res_142_);
return v_r_143_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg___lam__0(lean_object* v_to_144_, lean_object* v_acc_145_, lean_object* v_sibling_146_){
_start:
{
lean_object* v_fst_147_; lean_object* v_snd_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_165_; 
v_fst_147_ = lean_ctor_get(v_acc_145_, 0);
v_snd_148_ = lean_ctor_get(v_acc_145_, 1);
v_isSharedCheck_165_ = !lean_is_exclusive(v_acc_145_);
if (v_isSharedCheck_165_ == 0)
{
v___x_150_ = v_acc_145_;
v_isShared_151_ = v_isSharedCheck_165_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_snd_148_);
lean_inc(v_fst_147_);
lean_dec(v_acc_145_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_165_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___y_153_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_159_ = lean_unsigned_to_nat(2u);
v___x_160_ = lean_nat_mod(v_snd_148_, v___x_159_);
v___x_161_ = lean_unsigned_to_nat(0u);
v___x_162_ = lean_nat_dec_eq(v___x_160_, v___x_161_);
lean_dec(v___x_160_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; 
v___x_163_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(v_to_144_, v_sibling_146_, v_fst_147_);
v___y_153_ = v___x_163_;
goto v___jp_152_;
}
else
{
lean_object* v___x_164_; 
v___x_164_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_compressDigest___redArg(v_to_144_, v_fst_147_, v_sibling_146_);
v___y_153_ = v___x_164_;
goto v___jp_152_;
}
v___jp_152_:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_157_; 
v___x_154_ = lean_unsigned_to_nat(1u);
v___x_155_ = lean_nat_shiftr(v_snd_148_, v___x_154_);
lean_dec(v_snd_148_);
if (v_isShared_151_ == 0)
{
lean_ctor_set(v___x_150_, 1, v___x_155_);
lean_ctor_set(v___x_150_, 0, v___y_153_);
v___x_157_ = v___x_150_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v___y_153_);
lean_ctor_set(v_reuseFailAlloc_158_, 1, v___x_155_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg(lean_object* v_inst_166_, lean_object* v_to_167_, lean_object* v_root_168_, lean_object* v_idx_169_, lean_object* v_leaf_170_, lean_object* v_merkleProof_171_){
_start:
{
lean_object* v___f_172_; lean_object* v___x_173_; lean_object* v_final_174_; lean_object* v_fst_175_; uint8_t v___x_176_; 
v___f_172_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg___lam__0), 3, 1);
lean_closure_set(v___f_172_, 0, v_to_167_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v_leaf_170_);
lean_ctor_set(v___x_173_, 1, v_idx_169_);
v_final_174_ = l_List_foldl___redArg(v___f_172_, v___x_173_, v_merkleProof_171_);
v_fst_175_ = lean_ctor_get(v_final_174_, 0);
lean_inc(v_fst_175_);
lean_dec(v_final_174_);
v___x_176_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_digestEq___redArg(v_inst_166_, v_fst_175_, v_root_168_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg___boxed(lean_object* v_inst_177_, lean_object* v_to_178_, lean_object* v_root_179_, lean_object* v_idx_180_, lean_object* v_leaf_181_, lean_object* v_merkleProof_182_){
_start:
{
uint8_t v_res_183_; lean_object* v_r_184_; 
v_res_183_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg(v_inst_177_, v_to_178_, v_root_179_, v_idx_180_, v_leaf_181_, v_merkleProof_182_);
v_r_184_ = lean_box(v_res_183_);
return v_r_184_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify(lean_object* v_F_185_, lean_object* v_EF_186_, lean_object* v_inst_187_, lean_object* v_to_188_, lean_object* v_root_189_, lean_object* v_idx_190_, lean_object* v_leaf_191_, lean_object* v_merkleProof_192_){
_start:
{
uint8_t v___x_193_; 
v___x_193_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___redArg(v_inst_187_, v_to_188_, v_root_189_, v_idx_190_, v_leaf_191_, v_merkleProof_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify___boxed(lean_object* v_F_194_, lean_object* v_EF_195_, lean_object* v_inst_196_, lean_object* v_to_197_, lean_object* v_root_198_, lean_object* v_idx_199_, lean_object* v_leaf_200_, lean_object* v_merkleProof_201_){
_start:
{
uint8_t v_res_202_; lean_object* v_r_203_; 
v_res_202_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge_merkleVerify(v_F_194_, v_EF_195_, v_inst_196_, v_to_197_, v_root_198_, v_idx_199_, v_leaf_200_, v_merkleProof_201_);
v_r_203_ = lean_box(v_res_202_);
return v_r_203_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Sponge(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Generic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
