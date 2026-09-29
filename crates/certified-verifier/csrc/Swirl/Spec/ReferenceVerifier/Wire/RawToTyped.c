// Lean compiler output
// Module: Swirl.Spec.ReferenceVerifier.Wire.RawToTyped
// Imports: public import Init public meta import Init public import Fundamentals.Spec.BabyBear.Raw public import Fundamentals.Spec.BabyBearExt4.Raw public import Fundamentals.Spec.Poseidon2.Raw public import Fundamentals.Spec.Runtime.Config public import Swirl.Spec.ReferenceVerifier.Proof public import Fundamentals.Spec.Runtime.VerifyingKey public import Swirl.Spec.ReferenceVerifier.Wire.Raw
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
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
lean_object* lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(lean_object*);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_fOfWord(uint32_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_fOfWord___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_efOfWords(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_efOfWords___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirRoundConfigOfRaw(uint32_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirRoundConfigOfRaw___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProximityOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProximityOfRaw___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirConfigOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirConfigOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_logUpOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_logUpOfRaw___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_systemParamsOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingParamsOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_linearConstraintOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_entryOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_entryOfRaw___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionNodeOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionDagOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionDagOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicInteractionOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicConstraintsDagOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicConstraintsDagOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeyOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeyOfRaw(lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_innerOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_innerOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrLayerClaimsOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_pvListOfRaw(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_fOfWord(uint32_t v_v_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_uint32_to_nat(v_v_1_);
v___x_3_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_2_);
lean_dec(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_fOfWord___boxed(lean_object* v_v_4_){
_start:
{
uint32_t v_v_boxed_5_; lean_object* v_res_6_; 
v_v_boxed_5_ = lean_unbox_uint32(v_v_4_);
lean_dec(v_v_4_);
v_res_6_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_fOfWord(v_v_boxed_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_efOfWords(lean_object* v_v_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_v_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_efOfWords___boxed(lean_object* v_v_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_efOfWords(v_v_9_);
lean_dec_ref(v_v_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords___lam__0(lean_object* v_v_11_, lean_object* v_idx_12_){
_start:
{
lean_object* v___x_13_; uint32_t v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_13_ = lean_array_fget_borrowed(v_v_11_, v_idx_12_);
v___x_14_ = lean_unbox_uint32(v___x_13_);
v___x_15_ = lean_uint32_to_nat(v___x_14_);
v___x_16_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_15_);
lean_dec(v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords___lam__0___boxed(lean_object* v_v_17_, lean_object* v_idx_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords___lam__0(v_v_17_, v_idx_18_);
lean_dec(v_idx_18_);
lean_dec_ref(v_v_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords(lean_object* v_v_20_){
_start:
{
lean_object* v___f_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___f_21_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_digestOfWords___lam__0___boxed), 2, 1);
lean_closure_set(v___f_21_, 0, v_v_20_);
v___x_22_ = lean_unsigned_to_nat(8u);
v___x_23_ = l_Array_ofFn___redArg(v___x_22_, v___f_21_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirRoundConfigOfRaw(uint32_t v_r_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_uint32_to_nat(v_r_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirRoundConfigOfRaw___boxed(lean_object* v_r_26_){
_start:
{
uint32_t v_r_boxed_27_; lean_object* v_res_28_; 
v_r_boxed_27_ = lean_unbox_uint32(v_r_26_);
lean_dec(v_r_26_);
v_res_28_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirRoundConfigOfRaw(v_r_boxed_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProximityOfRaw(lean_object* v_x_29_){
_start:
{
switch(lean_obj_tag(v_x_29_))
{
case 0:
{
lean_object* v___x_30_; 
v___x_30_ = lean_box(0);
return v___x_30_;
}
case 1:
{
uint64_t v_m_31_; uint64_t v_listStartRound_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v_m_31_ = lean_ctor_get_uint64(v_x_29_, 0);
v_listStartRound_32_ = lean_ctor_get_uint64(v_x_29_, 8);
v___x_33_ = lean_uint64_to_nat(v_m_31_);
v___x_34_ = lean_uint64_to_nat(v_listStartRound_32_);
v___x_35_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_35_, 0, v___x_33_);
lean_ctor_set(v___x_35_, 1, v___x_34_);
return v___x_35_;
}
default: 
{
uint64_t v_m_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v_m_36_ = lean_ctor_get_uint64(v_x_29_, 0);
v___x_37_ = lean_uint64_to_nat(v_m_36_);
v___x_38_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
return v___x_38_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProximityOfRaw___boxed(lean_object* v_x_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProximityOfRaw(v_x_39_);
lean_dec(v_x_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirConfigOfRaw_spec__0(lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
if (lean_obj_tag(v_a_41_) == 0)
{
lean_object* v___x_43_; 
v___x_43_ = l_List_reverse___redArg(v_a_42_);
return v___x_43_;
}
else
{
lean_object* v_head_44_; lean_object* v_tail_45_; lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_55_; 
v_head_44_ = lean_ctor_get(v_a_41_, 0);
v_tail_45_ = lean_ctor_get(v_a_41_, 1);
v_isSharedCheck_55_ = !lean_is_exclusive(v_a_41_);
if (v_isSharedCheck_55_ == 0)
{
v___x_47_ = v_a_41_;
v_isShared_48_ = v_isSharedCheck_55_;
goto v_resetjp_46_;
}
else
{
lean_inc(v_tail_45_);
lean_inc(v_head_44_);
lean_dec(v_a_41_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_55_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
uint32_t v___x_49_; lean_object* v___x_50_; lean_object* v___x_52_; 
v___x_49_ = lean_unbox_uint32(v_head_44_);
lean_dec(v_head_44_);
v___x_50_ = lean_uint32_to_nat(v___x_49_);
if (v_isShared_48_ == 0)
{
lean_ctor_set(v___x_47_, 1, v_a_42_);
lean_ctor_set(v___x_47_, 0, v___x_50_);
v___x_52_ = v___x_47_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_54_; 
v_reuseFailAlloc_54_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_54_, 0, v___x_50_);
lean_ctor_set(v_reuseFailAlloc_54_, 1, v_a_42_);
v___x_52_ = v_reuseFailAlloc_54_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
v_a_41_ = v_tail_45_;
v_a_42_ = v___x_52_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirConfigOfRaw(lean_object* v_w_56_){
_start:
{
uint32_t v_k_57_; lean_object* v_rounds_58_; uint32_t v_muPowBits_59_; uint32_t v_queryPhasePowBits_60_; uint32_t v_foldingPowBits_61_; lean_object* v_proximity_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v_k_57_ = lean_ctor_get_uint32(v_w_56_, sizeof(void*)*2);
v_rounds_58_ = lean_ctor_get(v_w_56_, 0);
lean_inc_ref(v_rounds_58_);
v_muPowBits_59_ = lean_ctor_get_uint32(v_w_56_, sizeof(void*)*2 + 4);
v_queryPhasePowBits_60_ = lean_ctor_get_uint32(v_w_56_, sizeof(void*)*2 + 8);
v_foldingPowBits_61_ = lean_ctor_get_uint32(v_w_56_, sizeof(void*)*2 + 12);
v_proximity_62_ = lean_ctor_get(v_w_56_, 1);
lean_inc(v_proximity_62_);
lean_dec_ref(v_w_56_);
v___x_63_ = lean_uint32_to_nat(v_k_57_);
v___x_64_ = lean_array_to_list(v_rounds_58_);
v___x_65_ = lean_box(0);
v___x_66_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirConfigOfRaw_spec__0(v___x_64_, v___x_65_);
v___x_67_ = lean_uint32_to_nat(v_muPowBits_59_);
v___x_68_ = lean_uint32_to_nat(v_queryPhasePowBits_60_);
v___x_69_ = lean_uint32_to_nat(v_foldingPowBits_61_);
v___x_70_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProximityOfRaw(v_proximity_62_);
lean_dec(v_proximity_62_);
v___x_71_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_71_, 0, v___x_63_);
lean_ctor_set(v___x_71_, 1, v___x_66_);
lean_ctor_set(v___x_71_, 2, v___x_67_);
lean_ctor_set(v___x_71_, 3, v___x_68_);
lean_ctor_set(v___x_71_, 4, v___x_69_);
lean_ctor_set(v___x_71_, 5, v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_logUpOfRaw(lean_object* v_lp_72_){
_start:
{
uint32_t v_maxInteractionCount_73_; uint32_t v_logMaxMessageLength_74_; uint32_t v_powBits_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_maxInteractionCount_73_ = lean_ctor_get_uint32(v_lp_72_, 0);
v_logMaxMessageLength_74_ = lean_ctor_get_uint32(v_lp_72_, 4);
v_powBits_75_ = lean_ctor_get_uint32(v_lp_72_, 8);
v___x_76_ = lean_uint32_to_nat(v_maxInteractionCount_73_);
v___x_77_ = lean_uint32_to_nat(v_logMaxMessageLength_74_);
v___x_78_ = lean_uint32_to_nat(v_powBits_75_);
v___x_79_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_79_, 0, v___x_76_);
lean_ctor_set(v___x_79_, 1, v___x_77_);
lean_ctor_set(v___x_79_, 2, v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_logUpOfRaw___boxed(lean_object* v_lp_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_logUpOfRaw(v_lp_80_);
lean_dec_ref(v_lp_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_systemParamsOfRaw(lean_object* v_p_82_){
_start:
{
uint32_t v_lSkip_83_; uint32_t v_nStack_84_; uint32_t v_wStack_85_; uint32_t v_logBlowup_86_; lean_object* v_whir_87_; lean_object* v_logup_88_; uint32_t v_maxConstraintDegree_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v_lSkip_83_ = lean_ctor_get_uint32(v_p_82_, sizeof(void*)*2);
v_nStack_84_ = lean_ctor_get_uint32(v_p_82_, sizeof(void*)*2 + 4);
v_wStack_85_ = lean_ctor_get_uint32(v_p_82_, sizeof(void*)*2 + 8);
v_logBlowup_86_ = lean_ctor_get_uint32(v_p_82_, sizeof(void*)*2 + 12);
v_whir_87_ = lean_ctor_get(v_p_82_, 0);
lean_inc_ref(v_whir_87_);
v_logup_88_ = lean_ctor_get(v_p_82_, 1);
lean_inc_ref(v_logup_88_);
v_maxConstraintDegree_89_ = lean_ctor_get_uint32(v_p_82_, sizeof(void*)*2 + 16);
lean_dec_ref(v_p_82_);
v___x_90_ = lean_uint32_to_nat(v_lSkip_83_);
v___x_91_ = lean_uint32_to_nat(v_nStack_84_);
v___x_92_ = lean_uint32_to_nat(v_wStack_85_);
v___x_93_ = lean_uint32_to_nat(v_logBlowup_86_);
v___x_94_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirConfigOfRaw(v_whir_87_);
v___x_95_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_logUpOfRaw(v_logup_88_);
lean_dec_ref(v_logup_88_);
v___x_96_ = lean_uint32_to_nat(v_maxConstraintDegree_89_);
v___x_97_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_97_, 0, v___x_90_);
lean_ctor_set(v___x_97_, 1, v___x_91_);
lean_ctor_set(v___x_97_, 2, v___x_92_);
lean_ctor_set(v___x_97_, 3, v___x_93_);
lean_ctor_set(v___x_97_, 4, v___x_94_);
lean_ctor_set(v___x_97_, 5, v___x_95_);
lean_ctor_set(v___x_97_, 6, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw_spec__0(lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
if (lean_obj_tag(v_a_98_) == 0)
{
lean_object* v___x_100_; 
v___x_100_ = l_List_reverse___redArg(v_a_99_);
return v___x_100_;
}
else
{
lean_object* v_head_101_; lean_object* v_tail_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_112_; 
v_head_101_ = lean_ctor_get(v_a_98_, 0);
v_tail_102_ = lean_ctor_get(v_a_98_, 1);
v_isSharedCheck_112_ = !lean_is_exclusive(v_a_98_);
if (v_isSharedCheck_112_ == 0)
{
v___x_104_ = v_a_98_;
v_isShared_105_ = v_isSharedCheck_112_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_tail_102_);
lean_inc(v_head_101_);
lean_dec(v_a_98_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_112_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
uint32_t v___x_106_; lean_object* v___x_107_; lean_object* v___x_109_; 
v___x_106_ = lean_unbox_uint32(v_head_101_);
lean_dec(v_head_101_);
v___x_107_ = lean_uint32_to_nat(v___x_106_);
if (v_isShared_105_ == 0)
{
lean_ctor_set(v___x_104_, 1, v_a_99_);
lean_ctor_set(v___x_104_, 0, v___x_107_);
v___x_109_ = v___x_104_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_111_; 
v_reuseFailAlloc_111_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_111_, 0, v___x_107_);
lean_ctor_set(v_reuseFailAlloc_111_, 1, v_a_99_);
v___x_109_ = v_reuseFailAlloc_111_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
v_a_98_ = v_tail_102_;
v_a_99_ = v___x_109_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw(lean_object* v_w_113_){
_start:
{
lean_object* v_preprocessed_114_; lean_object* v_cachedMains_115_; uint32_t v_commonMain_116_; lean_object* v___y_118_; 
v_preprocessed_114_ = lean_ctor_get(v_w_113_, 0);
lean_inc(v_preprocessed_114_);
v_cachedMains_115_ = lean_ctor_get(v_w_113_, 1);
lean_inc_ref(v_cachedMains_115_);
v_commonMain_116_ = lean_ctor_get_uint32(v_w_113_, sizeof(void*)*2);
lean_dec_ref(v_w_113_);
if (lean_obj_tag(v_preprocessed_114_) == 0)
{
lean_object* v___x_124_; 
v___x_124_ = lean_box(0);
v___y_118_ = v___x_124_;
goto v___jp_117_;
}
else
{
lean_object* v_val_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_134_; 
v_val_125_ = lean_ctor_get(v_preprocessed_114_, 0);
v_isSharedCheck_134_ = !lean_is_exclusive(v_preprocessed_114_);
if (v_isSharedCheck_134_ == 0)
{
v___x_127_ = v_preprocessed_114_;
v_isShared_128_ = v_isSharedCheck_134_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_val_125_);
lean_dec(v_preprocessed_114_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_134_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
uint32_t v___x_129_; lean_object* v___x_130_; lean_object* v___x_132_; 
v___x_129_ = lean_unbox_uint32(v_val_125_);
lean_dec(v_val_125_);
v___x_130_ = lean_uint32_to_nat(v___x_129_);
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 0, v___x_130_);
v___x_132_ = v___x_127_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_130_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
v___y_118_ = v___x_132_;
goto v___jp_117_;
}
}
}
v___jp_117_:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_119_ = lean_array_to_list(v_cachedMains_115_);
v___x_120_ = lean_box(0);
v___x_121_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw_spec__0(v___x_119_, v___x_120_);
v___x_122_ = lean_uint32_to_nat(v_commonMain_116_);
v___x_123_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_123_, 0, v___y_118_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_122_);
return v___x_123_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingParamsOfRaw(lean_object* v_p_135_){
_start:
{
lean_object* v_width_136_; uint32_t v_numPublicValues_137_; uint8_t v_needRot_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v_width_136_ = lean_ctor_get(v_p_135_, 0);
lean_inc_ref(v_width_136_);
v_numPublicValues_137_ = lean_ctor_get_uint32(v_p_135_, sizeof(void*)*1);
v_needRot_138_ = lean_ctor_get_uint8(v_p_135_, sizeof(void*)*1 + 4);
lean_dec_ref(v_p_135_);
v___x_139_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw(v_width_136_);
v___x_140_ = lean_uint32_to_nat(v_numPublicValues_137_);
v___x_141_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_141_, 0, v___x_139_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
lean_ctor_set_uint8(v___x_141_, sizeof(void*)*2, v_needRot_138_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw___lam__0(lean_object* v_commit_142_, lean_object* v_idx_143_){
_start:
{
lean_object* v___x_144_; uint32_t v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_144_ = lean_array_fget_borrowed(v_commit_142_, v_idx_143_);
v___x_145_ = lean_unbox_uint32(v___x_144_);
v___x_146_ = lean_uint32_to_nat(v___x_145_);
v___x_147_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_146_);
lean_dec(v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw___lam__0___boxed(lean_object* v_commit_148_, lean_object* v_idx_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw___lam__0(v_commit_148_, v_idx_149_);
lean_dec(v_idx_149_);
lean_dec_ref(v_commit_148_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw(lean_object* v_d_151_){
_start:
{
lean_object* v_commit_152_; lean_object* v_hypercubeDim_153_; uint32_t v_stackingWidth_154_; lean_object* v___f_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v_commit_152_ = lean_ctor_get(v_d_151_, 0);
lean_inc_ref(v_commit_152_);
v_hypercubeDim_153_ = lean_ctor_get(v_d_151_, 1);
lean_inc(v_hypercubeDim_153_);
v_stackingWidth_154_ = lean_ctor_get_uint32(v_d_151_, sizeof(void*)*2);
lean_dec_ref(v_d_151_);
v___f_155_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw___lam__0___boxed), 2, 1);
lean_closure_set(v___f_155_, 0, v_commit_152_);
v___x_156_ = lean_unsigned_to_nat(8u);
v___x_157_ = l_Array_ofFn___redArg(v___x_156_, v___f_155_);
v___x_158_ = lean_uint32_to_nat(v_stackingWidth_154_);
v___x_159_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_159_, 0, v___x_157_);
lean_ctor_set(v___x_159_, 1, v_hypercubeDim_153_);
lean_ctor_set(v___x_159_, 2, v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_linearConstraintOfRaw(lean_object* v_lc_160_){
_start:
{
lean_object* v_coefficients_161_; uint32_t v_threshold_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v_coefficients_161_ = lean_ctor_get(v_lc_160_, 0);
lean_inc_ref(v_coefficients_161_);
v_threshold_162_ = lean_ctor_get_uint32(v_lc_160_, sizeof(void*)*1);
lean_dec_ref(v_lc_160_);
v___x_163_ = lean_array_to_list(v_coefficients_161_);
v___x_164_ = lean_box(0);
v___x_165_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw_spec__0(v___x_163_, v___x_164_);
v___x_166_ = lean_uint32_to_nat(v_threshold_162_);
v___x_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_165_);
lean_ctor_set(v___x_167_, 1, v___x_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_entryOfRaw(lean_object* v_x_168_){
_start:
{
switch(lean_obj_tag(v_x_168_))
{
case 0:
{
uint32_t v_offset_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v_offset_169_ = lean_ctor_get_uint32(v_x_168_, 0);
v___x_170_ = lean_uint32_to_nat(v_offset_169_);
v___x_171_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
return v___x_171_;
}
case 1:
{
uint32_t v_partIndex_172_; uint32_t v_offset_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v_partIndex_172_ = lean_ctor_get_uint32(v_x_168_, 0);
v_offset_173_ = lean_ctor_get_uint32(v_x_168_, 4);
v___x_174_ = lean_uint32_to_nat(v_partIndex_172_);
v___x_175_ = lean_uint32_to_nat(v_offset_173_);
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_174_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
return v___x_176_;
}
case 2:
{
lean_object* v___x_177_; 
v___x_177_ = lean_box(2);
return v___x_177_;
}
default: 
{
lean_object* v___x_178_; 
v___x_178_ = lean_box(3);
return v___x_178_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_entryOfRaw___boxed(lean_object* v_x_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_entryOfRaw(v_x_179_);
lean_dec(v_x_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw(lean_object* v_v_181_){
_start:
{
lean_object* v_entry_182_; uint32_t v_index_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v_entry_182_ = lean_ctor_get(v_v_181_, 0);
v_index_183_ = lean_ctor_get_uint32(v_v_181_, sizeof(void*)*1);
v___x_184_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_entryOfRaw(v_entry_182_);
v___x_185_ = lean_uint32_to_nat(v_index_183_);
v___x_186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_184_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw___boxed(lean_object* v_v_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw(v_v_187_);
lean_dec_ref(v_v_187_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionNodeOfRaw(lean_object* v_x_189_){
_start:
{
switch(lean_obj_tag(v_x_189_))
{
case 0:
{
lean_object* v_v_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_198_; 
v_v_190_ = lean_ctor_get(v_x_189_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v_x_189_);
if (v_isSharedCheck_198_ == 0)
{
v___x_192_ = v_x_189_;
v_isShared_193_ = v_isSharedCheck_198_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_v_190_);
lean_dec(v_x_189_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_198_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_194_; lean_object* v___x_196_; 
v___x_194_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw(v_v_190_);
lean_dec_ref(v_v_190_);
if (v_isShared_193_ == 0)
{
lean_ctor_set(v___x_192_, 0, v___x_194_);
v___x_196_ = v___x_192_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v___x_194_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
case 1:
{
lean_object* v___x_199_; 
v___x_199_ = lean_box(1);
return v___x_199_;
}
case 2:
{
lean_object* v___x_200_; 
v___x_200_ = lean_box(2);
return v___x_200_;
}
case 3:
{
lean_object* v___x_201_; 
v___x_201_ = lean_box(3);
return v___x_201_;
}
case 4:
{
uint32_t v_c_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v_c_202_ = lean_ctor_get_uint32(v_x_189_, 0);
lean_dec_ref_known(v_x_189_, 0);
v___x_203_ = lean_uint32_to_nat(v_c_202_);
v___x_204_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_203_);
lean_dec(v___x_203_);
v___x_205_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
return v___x_205_;
}
case 5:
{
uint32_t v_leftIdx_206_; uint32_t v_rightIdx_207_; uint64_t v_degreeMultiple_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v_leftIdx_206_ = lean_ctor_get_uint32(v_x_189_, 8);
v_rightIdx_207_ = lean_ctor_get_uint32(v_x_189_, 12);
v_degreeMultiple_208_ = lean_ctor_get_uint64(v_x_189_, 0);
lean_dec_ref_known(v_x_189_, 0);
v___x_209_ = lean_uint32_to_nat(v_leftIdx_206_);
v___x_210_ = lean_uint32_to_nat(v_rightIdx_207_);
v___x_211_ = lean_uint64_to_nat(v_degreeMultiple_208_);
v___x_212_ = lean_alloc_ctor(5, 3, 0);
lean_ctor_set(v___x_212_, 0, v___x_209_);
lean_ctor_set(v___x_212_, 1, v___x_210_);
lean_ctor_set(v___x_212_, 2, v___x_211_);
return v___x_212_;
}
case 6:
{
uint32_t v_leftIdx_213_; uint32_t v_rightIdx_214_; uint64_t v_degreeMultiple_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_leftIdx_213_ = lean_ctor_get_uint32(v_x_189_, 8);
v_rightIdx_214_ = lean_ctor_get_uint32(v_x_189_, 12);
v_degreeMultiple_215_ = lean_ctor_get_uint64(v_x_189_, 0);
lean_dec_ref_known(v_x_189_, 0);
v___x_216_ = lean_uint32_to_nat(v_leftIdx_213_);
v___x_217_ = lean_uint32_to_nat(v_rightIdx_214_);
v___x_218_ = lean_uint64_to_nat(v_degreeMultiple_215_);
v___x_219_ = lean_alloc_ctor(6, 3, 0);
lean_ctor_set(v___x_219_, 0, v___x_216_);
lean_ctor_set(v___x_219_, 1, v___x_217_);
lean_ctor_set(v___x_219_, 2, v___x_218_);
return v___x_219_;
}
case 7:
{
uint32_t v_idx_220_; uint64_t v_degreeMultiple_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v_idx_220_ = lean_ctor_get_uint32(v_x_189_, 8);
v_degreeMultiple_221_ = lean_ctor_get_uint64(v_x_189_, 0);
lean_dec_ref_known(v_x_189_, 0);
v___x_222_ = lean_uint32_to_nat(v_idx_220_);
v___x_223_ = lean_uint64_to_nat(v_degreeMultiple_221_);
v___x_224_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_222_);
lean_ctor_set(v___x_224_, 1, v___x_223_);
return v___x_224_;
}
default: 
{
uint32_t v_leftIdx_225_; uint32_t v_rightIdx_226_; uint64_t v_degreeMultiple_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v_leftIdx_225_ = lean_ctor_get_uint32(v_x_189_, 8);
v_rightIdx_226_ = lean_ctor_get_uint32(v_x_189_, 12);
v_degreeMultiple_227_ = lean_ctor_get_uint64(v_x_189_, 0);
lean_dec_ref_known(v_x_189_, 0);
v___x_228_ = lean_uint32_to_nat(v_leftIdx_225_);
v___x_229_ = lean_uint32_to_nat(v_rightIdx_226_);
v___x_230_ = lean_uint64_to_nat(v_degreeMultiple_227_);
v___x_231_ = lean_alloc_ctor(8, 3, 0);
lean_ctor_set(v___x_231_, 0, v___x_228_);
lean_ctor_set(v___x_231_, 1, v___x_229_);
lean_ctor_set(v___x_231_, 2, v___x_230_);
return v___x_231_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionDagOfRaw_spec__0(lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
if (lean_obj_tag(v_a_232_) == 0)
{
lean_object* v___x_234_; 
v___x_234_ = l_List_reverse___redArg(v_a_233_);
return v___x_234_;
}
else
{
lean_object* v_head_235_; lean_object* v_tail_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_245_; 
v_head_235_ = lean_ctor_get(v_a_232_, 0);
v_tail_236_ = lean_ctor_get(v_a_232_, 1);
v_isSharedCheck_245_ = !lean_is_exclusive(v_a_232_);
if (v_isSharedCheck_245_ == 0)
{
v___x_238_ = v_a_232_;
v_isShared_239_ = v_isSharedCheck_245_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_tail_236_);
lean_inc(v_head_235_);
lean_dec(v_a_232_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_245_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v___x_242_; 
v___x_240_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionNodeOfRaw(v_head_235_);
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 1, v_a_233_);
lean_ctor_set(v___x_238_, 0, v___x_240_);
v___x_242_ = v___x_238_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v___x_240_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v_a_233_);
v___x_242_ = v_reuseFailAlloc_244_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
v_a_232_ = v_tail_236_;
v_a_233_ = v___x_242_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionDagOfRaw(lean_object* v_d_246_){
_start:
{
lean_object* v_nodes_247_; lean_object* v_constraintIdx_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_260_; 
v_nodes_247_ = lean_ctor_get(v_d_246_, 0);
v_constraintIdx_248_ = lean_ctor_get(v_d_246_, 1);
v_isSharedCheck_260_ = !lean_is_exclusive(v_d_246_);
if (v_isSharedCheck_260_ == 0)
{
v___x_250_ = v_d_246_;
v_isShared_251_ = v_isSharedCheck_260_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_constraintIdx_248_);
lean_inc(v_nodes_247_);
lean_dec(v_d_246_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_260_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_258_; 
v___x_252_ = lean_array_to_list(v_nodes_247_);
v___x_253_ = lean_box(0);
v___x_254_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionDagOfRaw_spec__0(v___x_252_, v___x_253_);
v___x_255_ = lean_array_to_list(v_constraintIdx_248_);
v___x_256_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw_spec__0(v___x_255_, v___x_253_);
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 1, v___x_256_);
lean_ctor_set(v___x_250_, 0, v___x_254_);
v___x_258_ = v___x_250_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v___x_254_);
lean_ctor_set(v_reuseFailAlloc_259_, 1, v___x_256_);
v___x_258_ = v_reuseFailAlloc_259_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
return v___x_258_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicInteractionOfRaw(lean_object* v_i_261_){
_start:
{
lean_object* v_message_262_; uint32_t v_count_263_; uint32_t v_busIndex_264_; uint32_t v_countWeight_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v_message_262_ = lean_ctor_get(v_i_261_, 0);
lean_inc_ref(v_message_262_);
v_count_263_ = lean_ctor_get_uint32(v_i_261_, sizeof(void*)*1);
v_busIndex_264_ = lean_ctor_get_uint32(v_i_261_, sizeof(void*)*1 + 4);
v_countWeight_265_ = lean_ctor_get_uint32(v_i_261_, sizeof(void*)*1 + 8);
lean_dec_ref(v_i_261_);
v___x_266_ = lean_array_to_list(v_message_262_);
v___x_267_ = lean_box(0);
v___x_268_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceWidthOfRaw_spec__0(v___x_266_, v___x_267_);
v___x_269_ = lean_uint32_to_nat(v_count_263_);
v___x_270_ = lean_uint32_to_nat(v_busIndex_264_);
v___x_271_ = lean_uint32_to_nat(v_countWeight_265_);
v___x_272_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_272_, 0, v___x_268_);
lean_ctor_set(v___x_272_, 1, v___x_269_);
lean_ctor_set(v___x_272_, 2, v___x_270_);
lean_ctor_set(v___x_272_, 3, v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicConstraintsDagOfRaw_spec__0(lean_object* v_a_273_, lean_object* v_a_274_){
_start:
{
if (lean_obj_tag(v_a_273_) == 0)
{
lean_object* v___x_275_; 
v___x_275_ = l_List_reverse___redArg(v_a_274_);
return v___x_275_;
}
else
{
lean_object* v_head_276_; lean_object* v_tail_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_286_; 
v_head_276_ = lean_ctor_get(v_a_273_, 0);
v_tail_277_ = lean_ctor_get(v_a_273_, 1);
v_isSharedCheck_286_ = !lean_is_exclusive(v_a_273_);
if (v_isSharedCheck_286_ == 0)
{
v___x_279_ = v_a_273_;
v_isShared_280_ = v_isSharedCheck_286_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_tail_277_);
lean_inc(v_head_276_);
lean_dec(v_a_273_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_286_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v___x_281_; lean_object* v___x_283_; 
v___x_281_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicInteractionOfRaw(v_head_276_);
if (v_isShared_280_ == 0)
{
lean_ctor_set(v___x_279_, 1, v_a_274_);
lean_ctor_set(v___x_279_, 0, v___x_281_);
v___x_283_ = v___x_279_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v___x_281_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v_a_274_);
v___x_283_ = v_reuseFailAlloc_285_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
v_a_273_ = v_tail_277_;
v_a_274_ = v___x_283_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicConstraintsDagOfRaw(lean_object* v_d_287_){
_start:
{
lean_object* v_constraints_288_; lean_object* v_interactions_289_; lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_300_; 
v_constraints_288_ = lean_ctor_get(v_d_287_, 0);
v_interactions_289_ = lean_ctor_get(v_d_287_, 1);
v_isSharedCheck_300_ = !lean_is_exclusive(v_d_287_);
if (v_isSharedCheck_300_ == 0)
{
v___x_291_ = v_d_287_;
v_isShared_292_ = v_isSharedCheck_300_;
goto v_resetjp_290_;
}
else
{
lean_inc(v_interactions_289_);
lean_inc(v_constraints_288_);
lean_dec(v_d_287_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_300_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_298_; 
v___x_293_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicExpressionDagOfRaw(v_constraints_288_);
v___x_294_ = lean_array_to_list(v_interactions_289_);
v___x_295_ = lean_box(0);
v___x_296_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicConstraintsDagOfRaw_spec__0(v___x_294_, v___x_295_);
if (v_isShared_292_ == 0)
{
lean_ctor_set(v___x_291_, 1, v___x_296_);
lean_ctor_set(v___x_291_, 0, v___x_293_);
v___x_298_ = v___x_291_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v___x_293_);
lean_ctor_set(v_reuseFailAlloc_299_, 1, v___x_296_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeyOfRaw_spec__0(lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
if (lean_obj_tag(v_a_301_) == 0)
{
lean_object* v___x_303_; 
v___x_303_ = l_List_reverse___redArg(v_a_302_);
return v___x_303_;
}
else
{
lean_object* v_head_304_; lean_object* v_tail_305_; lean_object* v___x_307_; uint8_t v_isShared_308_; uint8_t v_isSharedCheck_314_; 
v_head_304_ = lean_ctor_get(v_a_301_, 0);
v_tail_305_ = lean_ctor_get(v_a_301_, 1);
v_isSharedCheck_314_ = !lean_is_exclusive(v_a_301_);
if (v_isSharedCheck_314_ == 0)
{
v___x_307_ = v_a_301_;
v_isShared_308_ = v_isSharedCheck_314_;
goto v_resetjp_306_;
}
else
{
lean_inc(v_tail_305_);
lean_inc(v_head_304_);
lean_dec(v_a_301_);
v___x_307_ = lean_box(0);
v_isShared_308_ = v_isSharedCheck_314_;
goto v_resetjp_306_;
}
v_resetjp_306_:
{
lean_object* v___x_309_; lean_object* v___x_311_; 
v___x_309_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicVariableOfRaw(v_head_304_);
lean_dec(v_head_304_);
if (v_isShared_308_ == 0)
{
lean_ctor_set(v___x_307_, 1, v_a_302_);
lean_ctor_set(v___x_307_, 0, v___x_309_);
v___x_311_ = v___x_307_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_309_);
lean_ctor_set(v_reuseFailAlloc_313_, 1, v_a_302_);
v___x_311_ = v_reuseFailAlloc_313_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
v_a_301_ = v_tail_305_;
v_a_302_ = v___x_311_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeyOfRaw(lean_object* v_sk_315_){
_start:
{
lean_object* v_preprocessedData_316_; lean_object* v_params_317_; lean_object* v_symbolicConstraints_318_; uint32_t v_maxConstraintDegree_319_; uint8_t v_isRequired_320_; lean_object* v_unusedVariables_321_; lean_object* v___y_323_; 
v_preprocessedData_316_ = lean_ctor_get(v_sk_315_, 0);
lean_inc(v_preprocessedData_316_);
v_params_317_ = lean_ctor_get(v_sk_315_, 1);
lean_inc_ref(v_params_317_);
v_symbolicConstraints_318_ = lean_ctor_get(v_sk_315_, 2);
lean_inc_ref(v_symbolicConstraints_318_);
v_maxConstraintDegree_319_ = lean_ctor_get_uint32(v_sk_315_, sizeof(void*)*4);
v_isRequired_320_ = lean_ctor_get_uint8(v_sk_315_, sizeof(void*)*4 + 4);
v_unusedVariables_321_ = lean_ctor_get(v_sk_315_, 3);
lean_inc_ref(v_unusedVariables_321_);
lean_dec_ref(v_sk_315_);
if (lean_obj_tag(v_preprocessedData_316_) == 0)
{
lean_object* v___x_332_; 
v___x_332_ = lean_box(0);
v___y_323_ = v___x_332_;
goto v___jp_322_;
}
else
{
lean_object* v_val_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_341_; 
v_val_333_ = lean_ctor_get(v_preprocessedData_316_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v_preprocessedData_316_);
if (v_isSharedCheck_341_ == 0)
{
v___x_335_ = v_preprocessedData_316_;
v_isShared_336_ = v_isSharedCheck_341_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_val_333_);
lean_dec(v_preprocessedData_316_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_341_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_337_; lean_object* v___x_339_; 
v___x_337_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_preprocessedDataOfRaw(v_val_333_);
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 0, v___x_337_);
v___x_339_ = v___x_335_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_337_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
v___y_323_ = v___x_339_;
goto v___jp_322_;
}
}
}
v___jp_322_:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_324_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingParamsOfRaw(v_params_317_);
v___x_325_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_symbolicConstraintsDagOfRaw(v_symbolicConstraints_318_);
v___x_326_ = lean_uint32_to_nat(v_maxConstraintDegree_319_);
v___x_327_ = lean_array_to_list(v_unusedVariables_321_);
v___x_328_ = lean_box(0);
v___x_329_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeyOfRaw_spec__0(v___x_327_, v___x_328_);
v___x_330_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_330_, 0, v___y_323_);
lean_ctor_set(v___x_330_, 1, v___x_324_);
lean_ctor_set(v___x_330_, 2, v___x_325_);
lean_ctor_set(v___x_330_, 3, v___x_326_);
lean_ctor_set(v___x_330_, 4, v___x_329_);
lean_ctor_set_uint8(v___x_330_, sizeof(void*)*5, v_isRequired_320_);
v___x_331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
return v___x_331_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw(lean_object* v_x_344_){
_start:
{
if (lean_obj_tag(v_x_344_) == 0)
{
lean_object* v___x_345_; 
v___x_345_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw___closed__0));
return v___x_345_;
}
else
{
lean_object* v_head_346_; lean_object* v_tail_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_365_; 
v_head_346_ = lean_ctor_get(v_x_344_, 0);
v_tail_347_ = lean_ctor_get(v_x_344_, 1);
v_isSharedCheck_365_ = !lean_is_exclusive(v_x_344_);
if (v_isSharedCheck_365_ == 0)
{
v___x_349_ = v_x_344_;
v_isShared_350_ = v_isSharedCheck_365_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_tail_347_);
lean_inc(v_head_346_);
lean_dec(v_x_344_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_365_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_351_; lean_object* v_a_352_; lean_object* v___x_353_; 
v___x_351_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeyOfRaw(v_head_346_);
v_a_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_a_352_);
lean_dec_ref(v___x_351_);
v___x_353_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw(v_tail_347_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_dec(v_a_352_);
lean_del_object(v___x_349_);
return v___x_353_;
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_364_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_364_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_364_ == 0)
{
v___x_356_ = v___x_353_;
v_isShared_357_ = v_isSharedCheck_364_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_353_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_364_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 1, v_a_354_);
lean_ctor_set(v___x_349_, 0, v_a_352_);
v___x_359_ = v___x_349_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v_a_352_);
lean_ctor_set(v_reuseFailAlloc_363_, 1, v_a_354_);
v___x_359_ = v_reuseFailAlloc_363_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
lean_object* v___x_361_; 
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v___x_359_);
v___x_361_ = v___x_356_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v___x_359_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_innerOfRaw_spec__0(lean_object* v_a_366_, lean_object* v_a_367_){
_start:
{
if (lean_obj_tag(v_a_366_) == 0)
{
lean_object* v___x_368_; 
v___x_368_ = l_List_reverse___redArg(v_a_367_);
return v___x_368_;
}
else
{
lean_object* v_head_369_; lean_object* v_tail_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_379_; 
v_head_369_ = lean_ctor_get(v_a_366_, 0);
v_tail_370_ = lean_ctor_get(v_a_366_, 1);
v_isSharedCheck_379_ = !lean_is_exclusive(v_a_366_);
if (v_isSharedCheck_379_ == 0)
{
v___x_372_ = v_a_366_;
v_isShared_373_ = v_isSharedCheck_379_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_tail_370_);
lean_inc(v_head_369_);
lean_dec(v_a_366_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_379_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_374_; lean_object* v___x_376_; 
v___x_374_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_linearConstraintOfRaw(v_head_369_);
if (v_isShared_373_ == 0)
{
lean_ctor_set(v___x_372_, 1, v_a_367_);
lean_ctor_set(v___x_372_, 0, v___x_374_);
v___x_376_ = v___x_372_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v___x_374_);
lean_ctor_set(v_reuseFailAlloc_378_, 1, v_a_367_);
v___x_376_ = v_reuseFailAlloc_378_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
v_a_366_ = v_tail_370_;
v_a_367_ = v___x_376_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_innerOfRaw(lean_object* v_m_380_){
_start:
{
lean_object* v_params_381_; lean_object* v_perAir_382_; lean_object* v_traceHeightConstraints_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_412_; 
v_params_381_ = lean_ctor_get(v_m_380_, 0);
v_perAir_382_ = lean_ctor_get(v_m_380_, 1);
v_traceHeightConstraints_383_ = lean_ctor_get(v_m_380_, 2);
v_isSharedCheck_412_ = !lean_is_exclusive(v_m_380_);
if (v_isSharedCheck_412_ == 0)
{
v___x_385_ = v_m_380_;
v_isShared_386_ = v_isSharedCheck_412_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_traceHeightConstraints_383_);
lean_inc(v_perAir_382_);
lean_inc(v_params_381_);
lean_dec(v_m_380_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_412_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_387_ = lean_array_to_list(v_perAir_382_);
v___x_388_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_starkVerifyingKeysOfRaw(v___x_387_);
if (lean_obj_tag(v___x_388_) == 0)
{
lean_object* v_a_389_; lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_396_; 
lean_del_object(v___x_385_);
lean_dec_ref(v_traceHeightConstraints_383_);
lean_dec_ref(v_params_381_);
v_a_389_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_396_ == 0)
{
v___x_391_ = v___x_388_;
v_isShared_392_ = v_isSharedCheck_396_;
goto v_resetjp_390_;
}
else
{
lean_inc(v_a_389_);
lean_dec(v___x_388_);
v___x_391_ = lean_box(0);
v_isShared_392_ = v_isSharedCheck_396_;
goto v_resetjp_390_;
}
v_resetjp_390_:
{
lean_object* v___x_394_; 
if (v_isShared_392_ == 0)
{
v___x_394_ = v___x_391_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v_a_389_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_411_; 
v_a_397_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_411_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_411_ == 0)
{
v___x_399_ = v___x_388_;
v_isShared_400_ = v_isSharedCheck_411_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_388_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_411_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_406_; 
v___x_401_ = lean_array_to_list(v_traceHeightConstraints_383_);
v___x_402_ = lean_box(0);
v___x_403_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_innerOfRaw_spec__0(v___x_401_, v___x_402_);
v___x_404_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_systemParamsOfRaw(v_params_381_);
if (v_isShared_386_ == 0)
{
lean_ctor_set(v___x_385_, 2, v___x_403_);
lean_ctor_set(v___x_385_, 1, v_a_397_);
lean_ctor_set(v___x_385_, 0, v___x_404_);
v___x_406_ = v___x_385_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v___x_404_);
lean_ctor_set(v_reuseFailAlloc_410_, 1, v_a_397_);
lean_ctor_set(v_reuseFailAlloc_410_, 2, v___x_403_);
v___x_406_ = v_reuseFailAlloc_410_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
lean_object* v___x_408_; 
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 0, v___x_406_);
v___x_408_ = v___x_399_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v___x_406_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
return v___x_408_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw___lam__0(lean_object* v_preHash_413_, lean_object* v_idx_414_){
_start:
{
lean_object* v___x_415_; uint32_t v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_415_ = lean_array_fget_borrowed(v_preHash_413_, v_idx_414_);
v___x_416_ = lean_unbox_uint32(v___x_415_);
v___x_417_ = lean_uint32_to_nat(v___x_416_);
v___x_418_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_417_);
lean_dec(v___x_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw___lam__0___boxed(lean_object* v_preHash_419_, lean_object* v_idx_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw___lam__0(v_preHash_419_, v_idx_420_);
lean_dec(v_idx_420_);
lean_dec_ref(v_preHash_419_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw(lean_object* v_raw_422_){
_start:
{
lean_object* v_inner_423_; lean_object* v_preHash_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_451_; 
v_inner_423_ = lean_ctor_get(v_raw_422_, 0);
v_preHash_424_ = lean_ctor_get(v_raw_422_, 1);
v_isSharedCheck_451_ = !lean_is_exclusive(v_raw_422_);
if (v_isSharedCheck_451_ == 0)
{
v___x_426_ = v_raw_422_;
v_isShared_427_ = v_isSharedCheck_451_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_preHash_424_);
lean_inc(v_inner_423_);
lean_dec(v_raw_422_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_451_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_428_; 
v___x_428_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_innerOfRaw(v_inner_423_);
if (lean_obj_tag(v___x_428_) == 0)
{
lean_object* v_a_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_436_; 
lean_del_object(v___x_426_);
lean_dec_ref(v_preHash_424_);
v_a_429_ = lean_ctor_get(v___x_428_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v___x_428_);
if (v_isSharedCheck_436_ == 0)
{
v___x_431_ = v___x_428_;
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_a_429_);
lean_dec(v___x_428_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___x_434_; 
if (v_isShared_432_ == 0)
{
v___x_434_ = v___x_431_;
goto v_reusejp_433_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v_a_429_);
v___x_434_ = v_reuseFailAlloc_435_;
goto v_reusejp_433_;
}
v_reusejp_433_:
{
return v___x_434_;
}
}
}
else
{
lean_object* v_a_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_450_; 
v_a_437_ = lean_ctor_get(v___x_428_, 0);
v_isSharedCheck_450_ = !lean_is_exclusive(v___x_428_);
if (v_isSharedCheck_450_ == 0)
{
v___x_439_ = v___x_428_;
v_isShared_440_ = v_isSharedCheck_450_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_a_437_);
lean_dec(v___x_428_);
v___x_439_ = lean_box(0);
v_isShared_440_ = v_isSharedCheck_450_;
goto v_resetjp_438_;
}
v_resetjp_438_:
{
lean_object* v___f_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_445_; 
v___f_441_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_vkOfRaw___lam__0___boxed), 2, 1);
lean_closure_set(v___f_441_, 0, v_preHash_424_);
v___x_442_ = lean_unsigned_to_nat(8u);
v___x_443_ = l_Array_ofFn___redArg(v___x_442_, v___f_441_);
if (v_isShared_427_ == 0)
{
lean_ctor_set(v___x_426_, 1, v___x_443_);
lean_ctor_set(v___x_426_, 0, v_a_437_);
v___x_445_ = v___x_426_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_449_; 
v_reuseFailAlloc_449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_449_, 0, v_a_437_);
lean_ctor_set(v_reuseFailAlloc_449_, 1, v___x_443_);
v___x_445_ = v_reuseFailAlloc_449_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
lean_object* v___x_447_; 
if (v_isShared_440_ == 0)
{
lean_ctor_set(v___x_439_, 0, v___x_445_);
v___x_447_ = v___x_439_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v___x_445_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0___lam__0(lean_object* v_head_452_, lean_object* v_idx_453_){
_start:
{
lean_object* v___x_454_; uint32_t v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_454_ = lean_array_fget_borrowed(v_head_452_, v_idx_453_);
v___x_455_ = lean_unbox_uint32(v___x_454_);
v___x_456_ = lean_uint32_to_nat(v___x_455_);
v___x_457_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_456_);
lean_dec(v___x_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0___lam__0___boxed(lean_object* v_head_458_, lean_object* v_idx_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0___lam__0(v_head_458_, v_idx_459_);
lean_dec(v_idx_459_);
lean_dec_ref(v_head_458_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0(lean_object* v_a_461_, lean_object* v_a_462_){
_start:
{
if (lean_obj_tag(v_a_461_) == 0)
{
lean_object* v___x_463_; 
v___x_463_ = l_List_reverse___redArg(v_a_462_);
return v___x_463_;
}
else
{
lean_object* v_head_464_; lean_object* v_tail_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_476_; 
v_head_464_ = lean_ctor_get(v_a_461_, 0);
v_tail_465_ = lean_ctor_get(v_a_461_, 1);
v_isSharedCheck_476_ = !lean_is_exclusive(v_a_461_);
if (v_isSharedCheck_476_ == 0)
{
v___x_467_ = v_a_461_;
v_isShared_468_ = v_isSharedCheck_476_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_tail_465_);
lean_inc(v_head_464_);
lean_dec(v_a_461_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_476_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___f_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_473_; 
v___f_469_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0___lam__0___boxed), 2, 1);
lean_closure_set(v___f_469_, 0, v_head_464_);
v___x_470_ = lean_unsigned_to_nat(8u);
v___x_471_ = l_Array_ofFn___redArg(v___x_470_, v___f_469_);
if (v_isShared_468_ == 0)
{
lean_ctor_set(v___x_467_, 1, v_a_462_);
lean_ctor_set(v___x_467_, 0, v___x_471_);
v___x_473_ = v___x_467_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_475_; 
v_reuseFailAlloc_475_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_475_, 0, v___x_471_);
lean_ctor_set(v_reuseFailAlloc_475_, 1, v_a_462_);
v___x_473_ = v_reuseFailAlloc_475_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
v_a_461_ = v_tail_465_;
v_a_462_ = v___x_473_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw(lean_object* v_t_477_){
_start:
{
uint32_t v_logHeight_478_; lean_object* v_cachedCommitments_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; 
v_logHeight_478_ = lean_ctor_get_uint32(v_t_477_, sizeof(void*)*1);
v_cachedCommitments_479_ = lean_ctor_get(v_t_477_, 0);
lean_inc_ref(v_cachedCommitments_479_);
lean_dec_ref(v_t_477_);
v___x_480_ = lean_uint32_to_nat(v_logHeight_478_);
v___x_481_ = lean_array_to_list(v_cachedCommitments_479_);
v___x_482_ = lean_box(0);
v___x_483_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0(v___x_481_, v___x_482_);
v___x_484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_484_, 0, v___x_480_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrLayerClaimsOfRaw(lean_object* v_c_485_){
_start:
{
lean_object* v_pXi0_486_; lean_object* v_pXi1_487_; lean_object* v_qXi0_488_; lean_object* v_qXi1_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_500_; 
v_pXi0_486_ = lean_ctor_get(v_c_485_, 0);
v_pXi1_487_ = lean_ctor_get(v_c_485_, 1);
v_qXi0_488_ = lean_ctor_get(v_c_485_, 2);
v_qXi1_489_ = lean_ctor_get(v_c_485_, 3);
v_isSharedCheck_500_ = !lean_is_exclusive(v_c_485_);
if (v_isSharedCheck_500_ == 0)
{
v___x_491_ = v_c_485_;
v_isShared_492_ = v_isSharedCheck_500_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_qXi1_489_);
lean_inc(v_qXi0_488_);
lean_inc(v_pXi1_487_);
lean_inc(v_pXi0_486_);
lean_dec(v_c_485_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_500_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_498_; 
v___x_493_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_pXi0_486_);
lean_dec_ref(v_pXi0_486_);
v___x_494_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_pXi1_487_);
lean_dec_ref(v_pXi1_487_);
v___x_495_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_qXi0_488_);
lean_dec_ref(v_qXi0_488_);
v___x_496_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_qXi1_489_);
lean_dec_ref(v_qXi1_489_);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 3, v___x_496_);
lean_ctor_set(v___x_491_, 2, v___x_495_);
lean_ctor_set(v___x_491_, 1, v___x_494_);
lean_ctor_set(v___x_491_, 0, v___x_493_);
v___x_498_ = v___x_491_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_499_; 
v_reuseFailAlloc_499_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_499_, 0, v___x_493_);
lean_ctor_set(v_reuseFailAlloc_499_, 1, v___x_494_);
lean_ctor_set(v_reuseFailAlloc_499_, 2, v___x_495_);
lean_ctor_set(v_reuseFailAlloc_499_, 3, v___x_496_);
v___x_498_ = v_reuseFailAlloc_499_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
return v___x_498_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__2(lean_object* v_a_501_, lean_object* v_a_502_){
_start:
{
if (lean_obj_tag(v_a_501_) == 0)
{
lean_object* v___x_503_; 
v___x_503_ = l_List_reverse___redArg(v_a_502_);
return v___x_503_;
}
else
{
lean_object* v_head_504_; lean_object* v_tail_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_514_; 
v_head_504_ = lean_ctor_get(v_a_501_, 0);
v_tail_505_ = lean_ctor_get(v_a_501_, 1);
v_isSharedCheck_514_ = !lean_is_exclusive(v_a_501_);
if (v_isSharedCheck_514_ == 0)
{
v___x_507_ = v_a_501_;
v_isShared_508_ = v_isSharedCheck_514_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_tail_505_);
lean_inc(v_head_504_);
lean_dec(v_a_501_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_514_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_509_; lean_object* v___x_511_; 
v___x_509_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrLayerClaimsOfRaw(v_head_504_);
if (v_isShared_508_ == 0)
{
lean_ctor_set(v___x_507_, 1, v_a_502_);
lean_ctor_set(v___x_507_, 0, v___x_509_);
v___x_511_ = v___x_507_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v___x_509_);
lean_ctor_set(v_reuseFailAlloc_513_, 1, v_a_502_);
v___x_511_ = v_reuseFailAlloc_513_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
v_a_501_ = v_tail_505_;
v_a_502_ = v___x_511_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0(size_t v_sz_515_, size_t v_i_516_, lean_object* v_bs_517_){
_start:
{
uint8_t v___x_518_; 
v___x_518_ = lean_usize_dec_lt(v_i_516_, v_sz_515_);
if (v___x_518_ == 0)
{
return v_bs_517_;
}
else
{
lean_object* v_v_519_; lean_object* v___x_520_; lean_object* v_bs_x27_521_; lean_object* v___x_522_; size_t v___x_523_; size_t v___x_524_; lean_object* v___x_525_; 
v_v_519_ = lean_array_uget(v_bs_517_, v_i_516_);
v___x_520_ = lean_unsigned_to_nat(0u);
v_bs_x27_521_ = lean_array_uset(v_bs_517_, v_i_516_, v___x_520_);
v___x_522_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_v_519_);
lean_dec(v_v_519_);
v___x_523_ = ((size_t)1ULL);
v___x_524_ = lean_usize_add(v_i_516_, v___x_523_);
v___x_525_ = lean_array_uset(v_bs_x27_521_, v_i_516_, v___x_522_);
v_i_516_ = v___x_524_;
v_bs_517_ = v___x_525_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0___boxed(lean_object* v_sz_527_, lean_object* v_i_528_, lean_object* v_bs_529_){
_start:
{
size_t v_sz_boxed_530_; size_t v_i_boxed_531_; lean_object* v_res_532_; 
v_sz_boxed_530_ = lean_unbox_usize(v_sz_527_);
lean_dec(v_sz_527_);
v_i_boxed_531_ = lean_unbox_usize(v_i_528_);
lean_dec(v_i_528_);
v_res_532_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0(v_sz_boxed_530_, v_i_boxed_531_, v_bs_529_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__1(lean_object* v_a_533_, lean_object* v_a_534_){
_start:
{
if (lean_obj_tag(v_a_533_) == 0)
{
lean_object* v___x_535_; 
v___x_535_ = l_List_reverse___redArg(v_a_534_);
return v___x_535_;
}
else
{
lean_object* v_head_536_; lean_object* v_tail_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_548_; 
v_head_536_ = lean_ctor_get(v_a_533_, 0);
v_tail_537_ = lean_ctor_get(v_a_533_, 1);
v_isSharedCheck_548_ = !lean_is_exclusive(v_a_533_);
if (v_isSharedCheck_548_ == 0)
{
v___x_539_ = v_a_533_;
v_isShared_540_ = v_isSharedCheck_548_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_tail_537_);
lean_inc(v_head_536_);
lean_dec(v_a_533_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_548_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
size_t v_sz_541_; size_t v___x_542_; lean_object* v___x_543_; lean_object* v___x_545_; 
v_sz_541_ = lean_array_size(v_head_536_);
v___x_542_ = ((size_t)0ULL);
v___x_543_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0(v_sz_541_, v___x_542_, v_head_536_);
if (v_isShared_540_ == 0)
{
lean_ctor_set(v___x_539_, 1, v_a_534_);
lean_ctor_set(v___x_539_, 0, v___x_543_);
v___x_545_ = v___x_539_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_543_);
lean_ctor_set(v_reuseFailAlloc_547_, 1, v_a_534_);
v___x_545_ = v_reuseFailAlloc_547_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
v_a_533_ = v_tail_537_;
v_a_534_ = v___x_545_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__3(lean_object* v_a_549_, lean_object* v_a_550_){
_start:
{
if (lean_obj_tag(v_a_549_) == 0)
{
lean_object* v___x_551_; 
v___x_551_ = l_List_reverse___redArg(v_a_550_);
return v___x_551_;
}
else
{
lean_object* v_head_552_; lean_object* v_tail_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_564_; 
v_head_552_ = lean_ctor_get(v_a_549_, 0);
v_tail_553_ = lean_ctor_get(v_a_549_, 1);
v_isSharedCheck_564_ = !lean_is_exclusive(v_a_549_);
if (v_isSharedCheck_564_ == 0)
{
v___x_555_ = v_a_549_;
v_isShared_556_ = v_isSharedCheck_564_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_tail_553_);
lean_inc(v_head_552_);
lean_dec(v_a_549_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_564_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_561_; 
v___x_557_ = lean_array_to_list(v_head_552_);
v___x_558_ = lean_box(0);
v___x_559_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__1(v___x_557_, v___x_558_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 1, v_a_550_);
lean_ctor_set(v___x_555_, 0, v___x_559_);
v___x_561_ = v___x_555_;
goto v_reusejp_560_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_559_);
lean_ctor_set(v_reuseFailAlloc_563_, 1, v_a_550_);
v___x_561_ = v_reuseFailAlloc_563_;
goto v_reusejp_560_;
}
v_reusejp_560_:
{
v_a_549_ = v_tail_553_;
v_a_550_ = v___x_561_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw(lean_object* v_g_565_){
_start:
{
uint32_t v_logupPowWitness_566_; lean_object* v_q0Claim_567_; lean_object* v_claimsPerLayer_568_; lean_object* v_sumcheckPolys_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; 
v_logupPowWitness_566_ = lean_ctor_get_uint32(v_g_565_, sizeof(void*)*3);
v_q0Claim_567_ = lean_ctor_get(v_g_565_, 0);
lean_inc_ref(v_q0Claim_567_);
v_claimsPerLayer_568_ = lean_ctor_get(v_g_565_, 1);
lean_inc_ref(v_claimsPerLayer_568_);
v_sumcheckPolys_569_ = lean_ctor_get(v_g_565_, 2);
lean_inc_ref(v_sumcheckPolys_569_);
lean_dec_ref(v_g_565_);
v___x_570_ = lean_uint32_to_nat(v_logupPowWitness_566_);
v___x_571_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_570_);
lean_dec(v___x_570_);
v___x_572_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_q0Claim_567_);
lean_dec_ref(v_q0Claim_567_);
v___x_573_ = lean_array_to_list(v_claimsPerLayer_568_);
v___x_574_ = lean_box(0);
v___x_575_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__2(v___x_573_, v___x_574_);
v___x_576_ = lean_array_to_list(v_sumcheckPolys_569_);
v___x_577_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__3(v___x_576_, v___x_574_);
v___x_578_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_578_, 0, v___x_571_);
lean_ctor_set(v___x_578_, 1, v___x_572_);
lean_ctor_set(v___x_578_, 2, v___x_575_);
lean_ctor_set(v___x_578_, 3, v___x_577_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(lean_object* v_a_579_, lean_object* v_a_580_){
_start:
{
if (lean_obj_tag(v_a_579_) == 0)
{
lean_object* v___x_581_; 
v___x_581_ = l_List_reverse___redArg(v_a_580_);
return v___x_581_;
}
else
{
lean_object* v_head_582_; lean_object* v_tail_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_592_; 
v_head_582_ = lean_ctor_get(v_a_579_, 0);
v_tail_583_ = lean_ctor_get(v_a_579_, 1);
v_isSharedCheck_592_ = !lean_is_exclusive(v_a_579_);
if (v_isSharedCheck_592_ == 0)
{
v___x_585_ = v_a_579_;
v_isShared_586_ = v_isSharedCheck_592_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_tail_583_);
lean_inc(v_head_582_);
lean_dec(v_a_579_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_592_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v___x_587_; lean_object* v___x_589_; 
v___x_587_ = lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofUInt32Words(v_head_582_);
lean_dec(v_head_582_);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 1, v_a_580_);
lean_ctor_set(v___x_585_, 0, v___x_587_);
v___x_589_ = v___x_585_;
goto v_reusejp_588_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v___x_587_);
lean_ctor_set(v_reuseFailAlloc_591_, 1, v_a_580_);
v___x_589_ = v_reuseFailAlloc_591_;
goto v_reusejp_588_;
}
v_reusejp_588_:
{
v_a_579_ = v_tail_583_;
v_a_580_ = v___x_589_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__1(lean_object* v_a_593_, lean_object* v_a_594_){
_start:
{
if (lean_obj_tag(v_a_593_) == 0)
{
lean_object* v___x_595_; 
v___x_595_ = l_List_reverse___redArg(v_a_594_);
return v___x_595_;
}
else
{
lean_object* v_head_596_; lean_object* v_tail_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_608_; 
v_head_596_ = lean_ctor_get(v_a_593_, 0);
v_tail_597_ = lean_ctor_get(v_a_593_, 1);
v_isSharedCheck_608_ = !lean_is_exclusive(v_a_593_);
if (v_isSharedCheck_608_ == 0)
{
v___x_599_ = v_a_593_;
v_isShared_600_ = v_isSharedCheck_608_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_tail_597_);
lean_inc(v_head_596_);
lean_dec(v_a_593_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_608_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_605_; 
v___x_601_ = lean_array_to_list(v_head_596_);
v___x_602_ = lean_box(0);
v___x_603_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_601_, v___x_602_);
if (v_isShared_600_ == 0)
{
lean_ctor_set(v___x_599_, 1, v_a_594_);
lean_ctor_set(v___x_599_, 0, v___x_603_);
v___x_605_ = v___x_599_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_603_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v_a_594_);
v___x_605_ = v_reuseFailAlloc_607_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
v_a_593_ = v_tail_597_;
v_a_594_ = v___x_605_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__2(lean_object* v_a_609_, lean_object* v_a_610_){
_start:
{
if (lean_obj_tag(v_a_609_) == 0)
{
lean_object* v___x_611_; 
v___x_611_ = l_List_reverse___redArg(v_a_610_);
return v___x_611_;
}
else
{
lean_object* v_head_612_; lean_object* v_tail_613_; lean_object* v___x_615_; uint8_t v_isShared_616_; uint8_t v_isSharedCheck_624_; 
v_head_612_ = lean_ctor_get(v_a_609_, 0);
v_tail_613_ = lean_ctor_get(v_a_609_, 1);
v_isSharedCheck_624_ = !lean_is_exclusive(v_a_609_);
if (v_isSharedCheck_624_ == 0)
{
v___x_615_ = v_a_609_;
v_isShared_616_ = v_isSharedCheck_624_;
goto v_resetjp_614_;
}
else
{
lean_inc(v_tail_613_);
lean_inc(v_head_612_);
lean_dec(v_a_609_);
v___x_615_ = lean_box(0);
v_isShared_616_ = v_isSharedCheck_624_;
goto v_resetjp_614_;
}
v_resetjp_614_:
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_621_; 
v___x_617_ = lean_array_to_list(v_head_612_);
v___x_618_ = lean_box(0);
v___x_619_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__1(v___x_617_, v___x_618_);
if (v_isShared_616_ == 0)
{
lean_ctor_set(v___x_615_, 1, v_a_610_);
lean_ctor_set(v___x_615_, 0, v___x_619_);
v___x_621_ = v___x_615_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v___x_619_);
lean_ctor_set(v_reuseFailAlloc_623_, 1, v_a_610_);
v___x_621_ = v_reuseFailAlloc_623_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
v_a_609_ = v_tail_613_;
v_a_610_ = v___x_621_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw(lean_object* v_b_625_){
_start:
{
lean_object* v_numeratorTermPerAir_626_; lean_object* v_denominatorTermPerAir_627_; lean_object* v_univariateRoundCoeffs_628_; lean_object* v_sumcheckRoundPolys_629_; lean_object* v_columnOpenings_630_; lean_object* v___x_632_; uint8_t v_isShared_633_; uint8_t v_isSharedCheck_648_; 
v_numeratorTermPerAir_626_ = lean_ctor_get(v_b_625_, 0);
v_denominatorTermPerAir_627_ = lean_ctor_get(v_b_625_, 1);
v_univariateRoundCoeffs_628_ = lean_ctor_get(v_b_625_, 2);
v_sumcheckRoundPolys_629_ = lean_ctor_get(v_b_625_, 3);
v_columnOpenings_630_ = lean_ctor_get(v_b_625_, 4);
v_isSharedCheck_648_ = !lean_is_exclusive(v_b_625_);
if (v_isSharedCheck_648_ == 0)
{
v___x_632_ = v_b_625_;
v_isShared_633_ = v_isSharedCheck_648_;
goto v_resetjp_631_;
}
else
{
lean_inc(v_columnOpenings_630_);
lean_inc(v_sumcheckRoundPolys_629_);
lean_inc(v_univariateRoundCoeffs_628_);
lean_inc(v_denominatorTermPerAir_627_);
lean_inc(v_numeratorTermPerAir_626_);
lean_dec(v_b_625_);
v___x_632_ = lean_box(0);
v_isShared_633_ = v_isSharedCheck_648_;
goto v_resetjp_631_;
}
v_resetjp_631_:
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_646_; 
v___x_634_ = lean_array_to_list(v_numeratorTermPerAir_626_);
v___x_635_ = lean_box(0);
v___x_636_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_634_, v___x_635_);
v___x_637_ = lean_array_to_list(v_denominatorTermPerAir_627_);
v___x_638_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_637_, v___x_635_);
v___x_639_ = lean_array_to_list(v_univariateRoundCoeffs_628_);
v___x_640_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_639_, v___x_635_);
v___x_641_ = lean_array_to_list(v_sumcheckRoundPolys_629_);
v___x_642_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__1(v___x_641_, v___x_635_);
v___x_643_ = lean_array_to_list(v_columnOpenings_630_);
v___x_644_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__2(v___x_643_, v___x_635_);
if (v_isShared_633_ == 0)
{
lean_ctor_set(v___x_632_, 4, v___x_644_);
lean_ctor_set(v___x_632_, 3, v___x_642_);
lean_ctor_set(v___x_632_, 2, v___x_640_);
lean_ctor_set(v___x_632_, 1, v___x_638_);
lean_ctor_set(v___x_632_, 0, v___x_636_);
v___x_646_ = v___x_632_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v___x_636_);
lean_ctor_set(v_reuseFailAlloc_647_, 1, v___x_638_);
lean_ctor_set(v_reuseFailAlloc_647_, 2, v___x_640_);
lean_ctor_set(v_reuseFailAlloc_647_, 3, v___x_642_);
lean_ctor_set(v_reuseFailAlloc_647_, 4, v___x_644_);
v___x_646_ = v_reuseFailAlloc_647_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
return v___x_646_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw_spec__0(lean_object* v_a_649_, lean_object* v_a_650_){
_start:
{
if (lean_obj_tag(v_a_649_) == 0)
{
lean_object* v___x_651_; 
v___x_651_ = l_List_reverse___redArg(v_a_650_);
return v___x_651_;
}
else
{
lean_object* v_head_652_; lean_object* v_tail_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_664_; 
v_head_652_ = lean_ctor_get(v_a_649_, 0);
v_tail_653_ = lean_ctor_get(v_a_649_, 1);
v_isSharedCheck_664_ = !lean_is_exclusive(v_a_649_);
if (v_isSharedCheck_664_ == 0)
{
v___x_655_ = v_a_649_;
v_isShared_656_ = v_isSharedCheck_664_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_tail_653_);
lean_inc(v_head_652_);
lean_dec(v_a_649_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_664_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
size_t v_sz_657_; size_t v___x_658_; lean_object* v___x_659_; lean_object* v___x_661_; 
v_sz_657_ = lean_array_size(v_head_652_);
v___x_658_ = ((size_t)0ULL);
v___x_659_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw_spec__0(v_sz_657_, v___x_658_, v_head_652_);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 1, v_a_650_);
lean_ctor_set(v___x_655_, 0, v___x_659_);
v___x_661_ = v___x_655_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v___x_659_);
lean_ctor_set(v_reuseFailAlloc_663_, 1, v_a_650_);
v___x_661_ = v_reuseFailAlloc_663_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
v_a_649_ = v_tail_653_;
v_a_650_ = v___x_661_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw(lean_object* v_s_665_){
_start:
{
lean_object* v_univariateRoundCoeffs_666_; lean_object* v_sumcheckRoundPolys_667_; lean_object* v_stackingOpenings_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v_univariateRoundCoeffs_666_ = lean_ctor_get(v_s_665_, 0);
lean_inc_ref(v_univariateRoundCoeffs_666_);
v_sumcheckRoundPolys_667_ = lean_ctor_get(v_s_665_, 1);
lean_inc_ref(v_sumcheckRoundPolys_667_);
v_stackingOpenings_668_ = lean_ctor_get(v_s_665_, 2);
lean_inc_ref(v_stackingOpenings_668_);
lean_dec_ref(v_s_665_);
v___x_669_ = lean_array_to_list(v_univariateRoundCoeffs_666_);
v___x_670_ = lean_box(0);
v___x_671_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_669_, v___x_670_);
v___x_672_ = lean_array_to_list(v_sumcheckRoundPolys_667_);
v___x_673_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw_spec__0(v___x_672_, v___x_670_);
v___x_674_ = lean_array_to_list(v_stackingOpenings_668_);
v___x_675_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__1(v___x_674_, v___x_670_);
v___x_676_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_676_, 0, v___x_671_);
lean_ctor_set(v___x_676_, 1, v___x_673_);
lean_ctor_set(v___x_676_, 2, v___x_675_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__1(lean_object* v_a_677_, lean_object* v_a_678_){
_start:
{
if (lean_obj_tag(v_a_677_) == 0)
{
lean_object* v___x_679_; 
v___x_679_ = l_List_reverse___redArg(v_a_678_);
return v___x_679_;
}
else
{
lean_object* v_head_680_; lean_object* v_tail_681_; lean_object* v___x_683_; uint8_t v_isShared_684_; uint8_t v_isSharedCheck_692_; 
v_head_680_ = lean_ctor_get(v_a_677_, 0);
v_tail_681_ = lean_ctor_get(v_a_677_, 1);
v_isSharedCheck_692_ = !lean_is_exclusive(v_a_677_);
if (v_isSharedCheck_692_ == 0)
{
v___x_683_ = v_a_677_;
v_isShared_684_ = v_isSharedCheck_692_;
goto v_resetjp_682_;
}
else
{
lean_inc(v_tail_681_);
lean_inc(v_head_680_);
lean_dec(v_a_677_);
v___x_683_ = lean_box(0);
v_isShared_684_ = v_isSharedCheck_692_;
goto v_resetjp_682_;
}
v_resetjp_682_:
{
uint32_t v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_689_; 
v___x_685_ = lean_unbox_uint32(v_head_680_);
lean_dec(v_head_680_);
v___x_686_ = lean_uint32_to_nat(v___x_685_);
v___x_687_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_686_);
lean_dec(v___x_686_);
if (v_isShared_684_ == 0)
{
lean_ctor_set(v___x_683_, 1, v_a_678_);
lean_ctor_set(v___x_683_, 0, v___x_687_);
v___x_689_ = v___x_683_;
goto v_reusejp_688_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v___x_687_);
lean_ctor_set(v_reuseFailAlloc_691_, 1, v_a_678_);
v___x_689_ = v_reuseFailAlloc_691_;
goto v_reusejp_688_;
}
v_reusejp_688_:
{
v_a_677_ = v_tail_681_;
v_a_678_ = v___x_689_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__2(lean_object* v_a_693_, lean_object* v_a_694_){
_start:
{
if (lean_obj_tag(v_a_693_) == 0)
{
lean_object* v___x_695_; 
v___x_695_ = l_List_reverse___redArg(v_a_694_);
return v___x_695_;
}
else
{
lean_object* v_head_696_; lean_object* v_tail_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_708_; 
v_head_696_ = lean_ctor_get(v_a_693_, 0);
v_tail_697_ = lean_ctor_get(v_a_693_, 1);
v_isSharedCheck_708_ = !lean_is_exclusive(v_a_693_);
if (v_isSharedCheck_708_ == 0)
{
v___x_699_ = v_a_693_;
v_isShared_700_ = v_isSharedCheck_708_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_tail_697_);
lean_inc(v_head_696_);
lean_dec(v_a_693_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_708_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_705_; 
v___x_701_ = lean_array_to_list(v_head_696_);
v___x_702_ = lean_box(0);
v___x_703_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__1(v___x_701_, v___x_702_);
if (v_isShared_700_ == 0)
{
lean_ctor_set(v___x_699_, 1, v_a_694_);
lean_ctor_set(v___x_699_, 0, v___x_703_);
v___x_705_ = v___x_699_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v___x_703_);
lean_ctor_set(v_reuseFailAlloc_707_, 1, v_a_694_);
v___x_705_ = v_reuseFailAlloc_707_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
v_a_693_ = v_tail_697_;
v_a_694_ = v___x_705_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__3(lean_object* v_a_709_, lean_object* v_a_710_){
_start:
{
if (lean_obj_tag(v_a_709_) == 0)
{
lean_object* v___x_711_; 
v___x_711_ = l_List_reverse___redArg(v_a_710_);
return v___x_711_;
}
else
{
lean_object* v_head_712_; lean_object* v_tail_713_; lean_object* v___x_715_; uint8_t v_isShared_716_; uint8_t v_isSharedCheck_724_; 
v_head_712_ = lean_ctor_get(v_a_709_, 0);
v_tail_713_ = lean_ctor_get(v_a_709_, 1);
v_isSharedCheck_724_ = !lean_is_exclusive(v_a_709_);
if (v_isSharedCheck_724_ == 0)
{
v___x_715_ = v_a_709_;
v_isShared_716_ = v_isSharedCheck_724_;
goto v_resetjp_714_;
}
else
{
lean_inc(v_tail_713_);
lean_inc(v_head_712_);
lean_dec(v_a_709_);
v___x_715_ = lean_box(0);
v_isShared_716_ = v_isSharedCheck_724_;
goto v_resetjp_714_;
}
v_resetjp_714_:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_721_; 
v___x_717_ = lean_array_to_list(v_head_712_);
v___x_718_ = lean_box(0);
v___x_719_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__2(v___x_717_, v___x_718_);
if (v_isShared_716_ == 0)
{
lean_ctor_set(v___x_715_, 1, v_a_710_);
lean_ctor_set(v___x_715_, 0, v___x_719_);
v___x_721_ = v___x_715_;
goto v_reusejp_720_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v___x_719_);
lean_ctor_set(v_reuseFailAlloc_723_, 1, v_a_710_);
v___x_721_ = v_reuseFailAlloc_723_;
goto v_reusejp_720_;
}
v_reusejp_720_:
{
v_a_709_ = v_tail_713_;
v_a_710_ = v___x_721_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__4(lean_object* v_a_725_, lean_object* v_a_726_){
_start:
{
if (lean_obj_tag(v_a_725_) == 0)
{
lean_object* v___x_727_; 
v___x_727_ = l_List_reverse___redArg(v_a_726_);
return v___x_727_;
}
else
{
lean_object* v_head_728_; lean_object* v_tail_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_740_; 
v_head_728_ = lean_ctor_get(v_a_725_, 0);
v_tail_729_ = lean_ctor_get(v_a_725_, 1);
v_isSharedCheck_740_ = !lean_is_exclusive(v_a_725_);
if (v_isSharedCheck_740_ == 0)
{
v___x_731_ = v_a_725_;
v_isShared_732_ = v_isSharedCheck_740_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_tail_729_);
lean_inc(v_head_728_);
lean_dec(v_a_725_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_740_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_737_; 
v___x_733_ = lean_array_to_list(v_head_728_);
v___x_734_ = lean_box(0);
v___x_735_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__3(v___x_733_, v___x_734_);
if (v_isShared_732_ == 0)
{
lean_ctor_set(v___x_731_, 1, v_a_726_);
lean_ctor_set(v___x_731_, 0, v___x_735_);
v___x_737_ = v___x_731_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v___x_735_);
lean_ctor_set(v_reuseFailAlloc_739_, 1, v_a_726_);
v___x_737_ = v_reuseFailAlloc_739_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
v_a_725_ = v_tail_729_;
v_a_726_ = v___x_737_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__0(lean_object* v_a_741_, lean_object* v_a_742_){
_start:
{
if (lean_obj_tag(v_a_741_) == 0)
{
lean_object* v___x_743_; 
v___x_743_ = l_List_reverse___redArg(v_a_742_);
return v___x_743_;
}
else
{
lean_object* v_head_744_; lean_object* v_tail_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_756_; 
v_head_744_ = lean_ctor_get(v_a_741_, 0);
v_tail_745_ = lean_ctor_get(v_a_741_, 1);
v_isSharedCheck_756_ = !lean_is_exclusive(v_a_741_);
if (v_isSharedCheck_756_ == 0)
{
v___x_747_ = v_a_741_;
v_isShared_748_ = v_isSharedCheck_756_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_tail_745_);
lean_inc(v_head_744_);
lean_dec(v_a_741_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_756_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_753_; 
v___x_749_ = lean_array_to_list(v_head_744_);
v___x_750_ = lean_box(0);
v___x_751_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0(v___x_749_, v___x_750_);
if (v_isShared_748_ == 0)
{
lean_ctor_set(v___x_747_, 1, v_a_742_);
lean_ctor_set(v___x_747_, 0, v___x_751_);
v___x_753_ = v___x_747_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v___x_751_);
lean_ctor_set(v_reuseFailAlloc_755_, 1, v_a_742_);
v___x_753_ = v_reuseFailAlloc_755_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
v_a_741_ = v_tail_745_;
v_a_742_ = v___x_753_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__5(lean_object* v_a_757_, lean_object* v_a_758_){
_start:
{
if (lean_obj_tag(v_a_757_) == 0)
{
lean_object* v___x_759_; 
v___x_759_ = l_List_reverse___redArg(v_a_758_);
return v___x_759_;
}
else
{
lean_object* v_head_760_; lean_object* v_tail_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_772_; 
v_head_760_ = lean_ctor_get(v_a_757_, 0);
v_tail_761_ = lean_ctor_get(v_a_757_, 1);
v_isSharedCheck_772_ = !lean_is_exclusive(v_a_757_);
if (v_isSharedCheck_772_ == 0)
{
v___x_763_ = v_a_757_;
v_isShared_764_ = v_isSharedCheck_772_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_tail_761_);
lean_inc(v_head_760_);
lean_dec(v_a_757_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_772_;
goto v_resetjp_762_;
}
v_resetjp_762_:
{
lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_769_; 
v___x_765_ = lean_array_to_list(v_head_760_);
v___x_766_ = lean_box(0);
v___x_767_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__0(v___x_765_, v___x_766_);
if (v_isShared_764_ == 0)
{
lean_ctor_set(v___x_763_, 1, v_a_758_);
lean_ctor_set(v___x_763_, 0, v___x_767_);
v___x_769_ = v___x_763_;
goto v_reusejp_768_;
}
else
{
lean_object* v_reuseFailAlloc_771_; 
v_reuseFailAlloc_771_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_771_, 0, v___x_767_);
lean_ctor_set(v_reuseFailAlloc_771_, 1, v_a_758_);
v___x_769_ = v_reuseFailAlloc_771_;
goto v_reusejp_768_;
}
v_reusejp_768_:
{
v_a_757_ = v_tail_761_;
v_a_758_ = v___x_769_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw(lean_object* v_w_773_, lean_object* v_muPowWitness_774_){
_start:
{
lean_object* v_whirSumcheckPolys_775_; lean_object* v_codewordCommits_776_; lean_object* v_oodValues_777_; lean_object* v_foldingPowWitnesses_778_; lean_object* v_queryPhasePowWitnesses_779_; lean_object* v_initialRoundOpenedRows_780_; lean_object* v_initialRoundMerkleProofs_781_; lean_object* v_codewordOpenedValues_782_; lean_object* v_codewordMerkleProofs_783_; lean_object* v_finalPoly_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; 
v_whirSumcheckPolys_775_ = lean_ctor_get(v_w_773_, 0);
lean_inc_ref(v_whirSumcheckPolys_775_);
v_codewordCommits_776_ = lean_ctor_get(v_w_773_, 1);
lean_inc_ref(v_codewordCommits_776_);
v_oodValues_777_ = lean_ctor_get(v_w_773_, 2);
lean_inc_ref(v_oodValues_777_);
v_foldingPowWitnesses_778_ = lean_ctor_get(v_w_773_, 3);
lean_inc_ref(v_foldingPowWitnesses_778_);
v_queryPhasePowWitnesses_779_ = lean_ctor_get(v_w_773_, 4);
lean_inc_ref(v_queryPhasePowWitnesses_779_);
v_initialRoundOpenedRows_780_ = lean_ctor_get(v_w_773_, 5);
lean_inc_ref(v_initialRoundOpenedRows_780_);
v_initialRoundMerkleProofs_781_ = lean_ctor_get(v_w_773_, 6);
lean_inc_ref(v_initialRoundMerkleProofs_781_);
v_codewordOpenedValues_782_ = lean_ctor_get(v_w_773_, 7);
lean_inc_ref(v_codewordOpenedValues_782_);
v_codewordMerkleProofs_783_ = lean_ctor_get(v_w_773_, 8);
lean_inc_ref(v_codewordMerkleProofs_783_);
v_finalPoly_784_ = lean_ctor_get(v_w_773_, 9);
lean_inc_ref(v_finalPoly_784_);
lean_dec_ref(v_w_773_);
v___x_785_ = lean_array_to_list(v_whirSumcheckPolys_775_);
v___x_786_ = lean_box(0);
v___x_787_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw_spec__0(v___x_785_, v___x_786_);
v___x_788_ = lean_array_to_list(v_codewordCommits_776_);
v___x_789_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw_spec__0(v___x_788_, v___x_786_);
v___x_790_ = lean_array_to_list(v_oodValues_777_);
v___x_791_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_790_, v___x_786_);
v___x_792_ = lean_array_to_list(v_foldingPowWitnesses_778_);
v___x_793_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__1(v___x_792_, v___x_786_);
v___x_794_ = lean_array_to_list(v_queryPhasePowWitnesses_779_);
v___x_795_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__1(v___x_794_, v___x_786_);
v___x_796_ = lean_array_to_list(v_initialRoundOpenedRows_780_);
v___x_797_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__4(v___x_796_, v___x_786_);
v___x_798_ = lean_array_to_list(v_initialRoundMerkleProofs_781_);
v___x_799_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__5(v___x_798_, v___x_786_);
v___x_800_ = lean_array_to_list(v_codewordOpenedValues_782_);
v___x_801_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__2(v___x_800_, v___x_786_);
v___x_802_ = lean_array_to_list(v_codewordMerkleProofs_783_);
v___x_803_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__5(v___x_802_, v___x_786_);
v___x_804_ = lean_array_to_list(v_finalPoly_784_);
v___x_805_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw_spec__0(v___x_804_, v___x_786_);
v___x_806_ = lean_alloc_ctor(0, 11, 0);
lean_ctor_set(v___x_806_, 0, v_muPowWitness_774_);
lean_ctor_set(v___x_806_, 1, v___x_787_);
lean_ctor_set(v___x_806_, 2, v___x_789_);
lean_ctor_set(v___x_806_, 3, v___x_791_);
lean_ctor_set(v___x_806_, 4, v___x_793_);
lean_ctor_set(v___x_806_, 5, v___x_795_);
lean_ctor_set(v___x_806_, 6, v___x_797_);
lean_ctor_set(v___x_806_, 7, v___x_799_);
lean_ctor_set(v___x_806_, 8, v___x_801_);
lean_ctor_set(v___x_806_, 9, v___x_803_);
lean_ctor_set(v___x_806_, 10, v___x_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw___lam__0(lean_object* v_commonMainCommit_807_, lean_object* v_idx_808_){
_start:
{
lean_object* v___x_809_; uint32_t v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; 
v___x_809_ = lean_array_fget_borrowed(v_commonMainCommit_807_, v_idx_808_);
v___x_810_ = lean_unbox_uint32(v___x_809_);
v___x_811_ = lean_uint32_to_nat(v___x_810_);
v___x_812_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_811_);
lean_dec(v___x_811_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw___lam__0___boxed(lean_object* v_commonMainCommit_813_, lean_object* v_idx_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw___lam__0(v_commonMainCommit_813_, v_idx_814_);
lean_dec(v_idx_814_);
lean_dec_ref(v_commonMainCommit_813_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw_spec__0(lean_object* v_a_816_, lean_object* v_a_817_){
_start:
{
if (lean_obj_tag(v_a_816_) == 0)
{
lean_object* v___x_818_; 
v___x_818_ = l_List_reverse___redArg(v_a_817_);
return v___x_818_;
}
else
{
lean_object* v_head_819_; lean_object* v_tail_820_; lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_840_; 
v_head_819_ = lean_ctor_get(v_a_816_, 0);
v_tail_820_ = lean_ctor_get(v_a_816_, 1);
v_isSharedCheck_840_ = !lean_is_exclusive(v_a_816_);
if (v_isSharedCheck_840_ == 0)
{
v___x_822_ = v_a_816_;
v_isShared_823_ = v_isSharedCheck_840_;
goto v_resetjp_821_;
}
else
{
lean_inc(v_tail_820_);
lean_inc(v_head_819_);
lean_dec(v_a_816_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_840_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v___y_825_; 
if (lean_obj_tag(v_head_819_) == 0)
{
lean_object* v___x_830_; 
v___x_830_ = lean_box(0);
v___y_825_ = v___x_830_;
goto v___jp_824_;
}
else
{
lean_object* v_val_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_839_; 
v_val_831_ = lean_ctor_get(v_head_819_, 0);
v_isSharedCheck_839_ = !lean_is_exclusive(v_head_819_);
if (v_isSharedCheck_839_ == 0)
{
v___x_833_ = v_head_819_;
v_isShared_834_ = v_isSharedCheck_839_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_val_831_);
lean_dec(v_head_819_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_839_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v___x_835_; lean_object* v___x_837_; 
v___x_835_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_traceVDataOfRaw(v_val_831_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 0, v___x_835_);
v___x_837_ = v___x_833_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v___x_835_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
v___y_825_ = v___x_837_;
goto v___jp_824_;
}
}
}
v___jp_824_:
{
lean_object* v___x_827_; 
if (v_isShared_823_ == 0)
{
lean_ctor_set(v___x_822_, 1, v_a_817_);
lean_ctor_set(v___x_822_, 0, v___y_825_);
v___x_827_ = v___x_822_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_829_; 
v_reuseFailAlloc_829_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_829_, 0, v___y_825_);
lean_ctor_set(v_reuseFailAlloc_829_, 1, v_a_817_);
v___x_827_ = v_reuseFailAlloc_829_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
v_a_816_ = v_tail_820_;
v_a_817_ = v___x_827_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw(lean_object* v_raw_841_){
_start:
{
lean_object* v_stackingProof_842_; lean_object* v_commonMainCommit_843_; lean_object* v_traceVdata_844_; lean_object* v_gkrProof_845_; lean_object* v_batchConstraintProof_846_; lean_object* v_whirProof_847_; uint32_t v_muPowWitness_848_; lean_object* v___f_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v_stackingProof_842_ = lean_ctor_get(v_raw_841_, 4);
lean_inc_ref(v_stackingProof_842_);
v_commonMainCommit_843_ = lean_ctor_get(v_raw_841_, 0);
lean_inc_ref(v_commonMainCommit_843_);
v_traceVdata_844_ = lean_ctor_get(v_raw_841_, 1);
lean_inc_ref(v_traceVdata_844_);
v_gkrProof_845_ = lean_ctor_get(v_raw_841_, 2);
lean_inc_ref(v_gkrProof_845_);
v_batchConstraintProof_846_ = lean_ctor_get(v_raw_841_, 3);
lean_inc_ref(v_batchConstraintProof_846_);
v_whirProof_847_ = lean_ctor_get(v_raw_841_, 5);
lean_inc_ref(v_whirProof_847_);
lean_dec_ref(v_raw_841_);
v_muPowWitness_848_ = lean_ctor_get_uint32(v_stackingProof_842_, sizeof(void*)*3);
v___f_849_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw___lam__0___boxed), 2, 1);
lean_closure_set(v___f_849_, 0, v_commonMainCommit_843_);
v___x_850_ = lean_unsigned_to_nat(8u);
v___x_851_ = l_Array_ofFn___redArg(v___x_850_, v___f_849_);
v___x_852_ = lean_array_to_list(v_traceVdata_844_);
v___x_853_ = lean_box(0);
v___x_854_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_proofOfRaw_spec__0(v___x_852_, v___x_853_);
v___x_855_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_gkrProofOfRaw(v_gkrProof_845_);
v___x_856_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_batchConstraintProofOfRaw(v_batchConstraintProof_846_);
v___x_857_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_stackingProofOfRaw(v_stackingProof_842_);
v___x_858_ = lean_uint32_to_nat(v_muPowWitness_848_);
v___x_859_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_858_);
lean_dec(v___x_858_);
v___x_860_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw(v_whirProof_847_, v___x_859_);
v___x_861_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_861_, 0, v___x_851_);
lean_ctor_set(v___x_861_, 1, v___x_854_);
lean_ctor_set(v___x_861_, 2, v___x_853_);
lean_ctor_set(v___x_861_, 3, v___x_855_);
lean_ctor_set(v___x_861_, 4, v___x_856_);
lean_ctor_set(v___x_861_, 5, v___x_857_);
lean_ctor_set(v___x_861_, 6, v___x_860_);
v___x_862_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_862_, 0, v___x_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_RawToTyped_pvListOfRaw(lean_object* v_raw_863_){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v___x_864_ = lean_array_to_list(v_raw_863_);
v___x_865_ = lean_box(0);
v___x_866_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Wire_RawToTyped_whirProofOfRaw_spec__2(v___x_864_, v___x_865_);
return v___x_866_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_BabyBearExt4_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Config(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Proof(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_VerifyingKey(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Wire_Raw(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Wire_RawToTyped(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_BabyBearExt4_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Proof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_VerifyingKey(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Wire_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
