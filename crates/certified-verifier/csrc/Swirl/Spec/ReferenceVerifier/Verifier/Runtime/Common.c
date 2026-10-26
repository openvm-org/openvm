// Lean compiler output
// Module: Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Common
// Imports: public import Init public meta import Init public import Fundamentals.Spec.FieldOps public import Swirl.Spec.ReferenceVerifier.Proof public import Swirl.Spec.ReferenceVerifier.Runtime.PolyCommon public import Fundamentals.Spec.Runtime.VerifyingKey
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
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_MergeSort_Internal_mergeSortTR_u2082___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(lean_object* v_traceVdata_1_, lean_object* v_airId_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = l_List_get_x3fInternal___redArg(v_traceVdata_1_, v_airId_2_);
if (lean_obj_tag(v___x_3_) == 0)
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
else
{
lean_object* v_val_5_; 
v_val_5_ = lean_ctor_get(v___x_3_, 0);
lean_inc(v_val_5_);
lean_dec_ref_known(v___x_3_, 1);
return v_val_5_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg___boxed(lean_object* v_traceVdata_6_, lean_object* v_airId_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_6_, v_airId_7_);
lean_dec(v_traceVdata_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt(lean_object* v_Digest_9_, lean_object* v_traceVdata_10_, lean_object* v_airId_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_10_, v_airId_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___boxed(lean_object* v_Digest_13_, lean_object* v_traceVdata_14_, lean_object* v_airId_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt(v_Digest_13_, v_traceVdata_14_, v_airId_15_);
lean_dec(v_traceVdata_14_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(lean_object* v_vk_17_, lean_object* v_airId_18_){
_start:
{
lean_object* v_inner_19_; lean_object* v_perAir_20_; lean_object* v___x_21_; 
v_inner_19_ = lean_ctor_get(v_vk_17_, 0);
v_perAir_20_ = lean_ctor_get(v_inner_19_, 1);
v___x_21_ = l_List_get_x3fInternal___redArg(v_perAir_20_, v_airId_18_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg___boxed(lean_object* v_vk_22_, lean_object* v_airId_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_22_, v_airId_23_);
lean_dec_ref(v_vk_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt(lean_object* v_F_25_, lean_object* v_Digest_26_, lean_object* v_vk_27_, lean_object* v_airId_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_27_, v_airId_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___boxed(lean_object* v_F_30_, lean_object* v_Digest_31_, lean_object* v_vk_32_, lean_object* v_airId_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt(v_F_30_, v_Digest_31_, v_vk_32_, v_airId_33_);
lean_dec_ref(v_vk_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg(lean_object* v_x_35_, lean_object* v_x_36_){
_start:
{
if (lean_obj_tag(v_x_36_) == 0)
{
return v_x_35_;
}
else
{
lean_object* v_head_37_; 
v_head_37_ = lean_ctor_get(v_x_36_, 0);
if (lean_obj_tag(v_head_37_) == 0)
{
lean_object* v_tail_38_; 
v_tail_38_ = lean_ctor_get(v_x_36_, 1);
v_x_36_ = v_tail_38_;
goto _start;
}
else
{
lean_object* v_tail_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v_tail_40_ = lean_ctor_get(v_x_36_, 1);
v___x_41_ = lean_unsigned_to_nat(1u);
v___x_42_ = lean_nat_add(v_x_35_, v___x_41_);
lean_dec(v_x_35_);
v_x_35_ = v___x_42_;
v_x_36_ = v_tail_40_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg___boxed(lean_object* v_x_44_, lean_object* v_x_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg(v_x_44_, v_x_45_);
lean_dec(v_x_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(lean_object* v_traceVdata_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = lean_unsigned_to_nat(0u);
v___x_49_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg(v___x_48_, v_traceVdata_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg___boxed(lean_object* v_traceVdata_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(v_traceVdata_50_);
lean_dec(v_traceVdata_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces(lean_object* v_Digest_52_, lean_object* v_traceVdata_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(v_traceVdata_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___boxed(lean_object* v_Digest_55_, lean_object* v_traceVdata_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces(v_Digest_55_, v_traceVdata_56_);
lean_dec(v_traceVdata_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0(lean_object* v_Digest_58_, lean_object* v_x_59_, lean_object* v_x_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___redArg(v_x_59_, v_x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0___boxed(lean_object* v_Digest_62_, lean_object* v_x_63_, lean_object* v_x_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces_spec__0(v_Digest_62_, v_x_63_, v_x_64_);
lean_dec(v_x_64_);
return v_res_65_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg(lean_object* v_traceVdata_66_, lean_object* v_a_67_, lean_object* v_b_68_){
_start:
{
lean_object* v___x_69_; uint8_t v___x_70_; 
lean_inc(v_a_67_);
v___x_69_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_66_, v_a_67_);
v___x_70_ = lean_nat_dec_le(v_a_67_, v_b_68_);
lean_dec(v_a_67_);
if (lean_obj_tag(v___x_69_) == 0)
{
lean_object* v___x_71_; 
v___x_71_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_66_, v_b_68_);
if (lean_obj_tag(v___x_71_) == 0)
{
return v___x_70_;
}
else
{
uint8_t v___x_72_; 
lean_dec_ref_known(v___x_71_, 1);
v___x_72_ = 0;
return v___x_72_;
}
}
else
{
lean_object* v_val_73_; lean_object* v___x_74_; 
v_val_73_ = lean_ctor_get(v___x_69_, 0);
lean_inc(v_val_73_);
lean_dec_ref_known(v___x_69_, 1);
v___x_74_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_66_, v_b_68_);
if (lean_obj_tag(v___x_74_) == 0)
{
uint8_t v___x_75_; 
lean_dec(v_val_73_);
v___x_75_ = 1;
return v___x_75_;
}
else
{
lean_object* v_val_76_; lean_object* v_logHeight_77_; lean_object* v_logHeight_78_; uint8_t v___x_79_; 
v_val_76_ = lean_ctor_get(v___x_74_, 0);
lean_inc(v_val_76_);
lean_dec_ref_known(v___x_74_, 1);
v_logHeight_77_ = lean_ctor_get(v_val_76_, 0);
lean_inc(v_logHeight_77_);
lean_dec(v_val_76_);
v_logHeight_78_ = lean_ctor_get(v_val_73_, 0);
lean_inc(v_logHeight_78_);
lean_dec(v_val_73_);
v___x_79_ = lean_nat_dec_lt(v_logHeight_77_, v_logHeight_78_);
if (v___x_79_ == 0)
{
uint8_t v___x_80_; 
v___x_80_ = lean_nat_dec_eq(v_logHeight_78_, v_logHeight_77_);
lean_dec(v_logHeight_77_);
lean_dec(v_logHeight_78_);
if (v___x_80_ == 0)
{
return v___x_80_;
}
else
{
return v___x_70_;
}
}
else
{
lean_dec(v_logHeight_78_);
lean_dec(v_logHeight_77_);
return v___x_79_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg___boxed(lean_object* v_traceVdata_81_, lean_object* v_a_82_, lean_object* v_b_83_){
_start:
{
uint8_t v_res_84_; lean_object* v_r_85_; 
v_res_84_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg(v_traceVdata_81_, v_a_82_, v_b_83_);
lean_dec(v_traceVdata_81_);
v_r_85_ = lean_box(v_res_84_);
return v_r_85_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel(lean_object* v_Digest_86_, lean_object* v_traceVdata_87_, lean_object* v_a_88_, lean_object* v_b_89_){
_start:
{
uint8_t v___x_90_; 
v___x_90_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg(v_traceVdata_87_, v_a_88_, v_b_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___boxed(lean_object* v_Digest_91_, lean_object* v_traceVdata_92_, lean_object* v_a_93_, lean_object* v_b_94_){
_start:
{
uint8_t v_res_95_; lean_object* v_r_96_; 
v_res_95_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel(v_Digest_91_, v_traceVdata_92_, v_a_93_, v_b_94_);
lean_dec(v_traceVdata_92_);
v_r_96_ = lean_box(v_res_95_);
return v_r_96_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___lam__0(lean_object* v_traceVdata_97_, lean_object* v_a_98_, lean_object* v_b_99_){
_start:
{
uint8_t v___x_100_; 
v___x_100_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableRelAirSortRel___redArg(v_traceVdata_97_, v_a_98_, v_b_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___lam__0___boxed(lean_object* v_traceVdata_101_, lean_object* v_a_102_, lean_object* v_b_103_){
_start:
{
uint8_t v_res_104_; lean_object* v_r_105_; 
v_res_104_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___lam__0(v_traceVdata_101_, v_a_102_, v_b_103_);
lean_dec(v_traceVdata_101_);
v_r_105_ = lean_box(v_res_104_);
return v_r_105_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg(lean_object* v_vk_108_, lean_object* v_traceVdata_109_){
_start:
{
lean_object* v_inner_110_; lean_object* v_perAir_111_; lean_object* v___f_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v_inner_110_ = lean_ctor_get(v_vk_108_, 0);
v_perAir_111_ = lean_ctor_get(v_inner_110_, 1);
lean_inc(v_traceVdata_109_);
v___f_112_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_112_, 0, v_traceVdata_109_);
v___x_113_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(v_traceVdata_109_);
lean_dec(v_traceVdata_109_);
v___x_114_ = l_List_lengthTR___redArg(v_perAir_111_);
v___x_115_ = l_List_range(v___x_114_);
v___x_116_ = l_List_MergeSort_Internal_mergeSortTR_u2082___redArg(v___x_115_, v___f_112_);
v___x_117_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___closed__0));
lean_inc(v___x_116_);
v___x_118_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v___x_116_, v___x_116_, v___x_113_, v___x_117_);
lean_dec(v___x_116_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg___boxed(lean_object* v_vk_119_, lean_object* v_traceVdata_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg(v_vk_119_, v_traceVdata_120_);
lean_dec_ref(v_vk_119_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId(lean_object* v_F_122_, lean_object* v_Digest_123_, lean_object* v_vk_124_, lean_object* v_traceVdata_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg(v_vk_124_, v_traceVdata_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___boxed(lean_object* v_F_127_, lean_object* v_Digest_128_, lean_object* v_vk_129_, lean_object* v_traceVdata_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId(v_F_127_, v_Digest_128_, v_vk_129_, v_traceVdata_130_);
lean_dec_ref(v_vk_129_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers_spec__0___redArg(lean_object* v_fo_132_, lean_object* v_x_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
if (lean_obj_tag(v_a_134_) == 0)
{
lean_object* v___x_136_; 
lean_dec(v_x_133_);
lean_dec_ref(v_fo_132_);
v___x_136_ = l_List_reverse___redArg(v_a_135_);
return v___x_136_;
}
else
{
lean_object* v_toRingOps_137_; lean_object* v_toSemiringOps_138_; lean_object* v_head_139_; lean_object* v_tail_140_; lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_150_; 
v_toRingOps_137_ = lean_ctor_get(v_fo_132_, 0);
v_toSemiringOps_138_ = lean_ctor_get(v_toRingOps_137_, 0);
v_head_139_ = lean_ctor_get(v_a_134_, 0);
v_tail_140_ = lean_ctor_get(v_a_134_, 1);
v_isSharedCheck_150_ = !lean_is_exclusive(v_a_134_);
if (v_isSharedCheck_150_ == 0)
{
v___x_142_ = v_a_134_;
v_isShared_143_ = v_isSharedCheck_150_;
goto v_resetjp_141_;
}
else
{
lean_inc(v_tail_140_);
lean_inc(v_head_139_);
lean_dec(v_a_134_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_150_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
lean_object* v_pow_144_; lean_object* v___x_145_; lean_object* v___x_147_; 
v_pow_144_ = lean_ctor_get(v_toSemiringOps_138_, 5);
lean_inc(v_pow_144_);
lean_inc(v_x_133_);
v___x_145_ = lean_apply_2(v_pow_144_, v_x_133_, v_head_139_);
if (v_isShared_143_ == 0)
{
lean_ctor_set(v___x_142_, 1, v_a_135_);
lean_ctor_set(v___x_142_, 0, v___x_145_);
v___x_147_ = v___x_142_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_149_, 1, v_a_135_);
v___x_147_ = v_reuseFailAlloc_149_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
v_a_134_ = v_tail_140_;
v_a_135_ = v___x_147_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(lean_object* v_fo_151_, lean_object* v_x_152_, lean_object* v_count_153_){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = l_List_range(v_count_153_);
v___x_155_ = lean_box(0);
v___x_156_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers_spec__0___redArg(v_fo_151_, v_x_152_, v___x_154_, v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers(lean_object* v_EF_157_, lean_object* v_fo_158_, lean_object* v_x_159_, lean_object* v_count_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_158_, v_x_159_, v_count_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers_spec__0(lean_object* v_EF_162_, lean_object* v_fo_163_, lean_object* v_x_164_, lean_object* v_a_165_, lean_object* v_a_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers_spec__0___redArg(v_fo_163_, v_x_164_, v_a_165_, v_a_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg(lean_object* v_fo_168_, lean_object* v_step_169_, lean_object* v_idx_170_, lean_object* v_a_171_){
_start:
{
if (lean_obj_tag(v_a_171_) == 0)
{
lean_object* v_toRingOps_172_; lean_object* v_toSemiringOps_173_; lean_object* v_zero_174_; 
v_toRingOps_172_ = lean_ctor_get(v_fo_168_, 0);
lean_inc_ref(v_toRingOps_172_);
lean_dec_ref(v_fo_168_);
v_toSemiringOps_173_ = lean_ctor_get(v_toRingOps_172_, 0);
lean_inc_ref(v_toSemiringOps_173_);
lean_dec_ref(v_toRingOps_172_);
v_zero_174_ = lean_ctor_get(v_toSemiringOps_173_, 0);
lean_inc(v_zero_174_);
lean_dec_ref(v_toSemiringOps_173_);
return v_zero_174_;
}
else
{
lean_object* v_head_175_; lean_object* v_tail_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v_tailSum_179_; lean_object* v___x_180_; lean_object* v___x_181_; uint8_t v___x_182_; 
v_head_175_ = lean_ctor_get(v_a_171_, 0);
lean_inc(v_head_175_);
v_tail_176_ = lean_ctor_get(v_a_171_, 1);
lean_inc(v_tail_176_);
lean_dec_ref_known(v_a_171_, 2);
v___x_177_ = lean_unsigned_to_nat(1u);
v___x_178_ = lean_nat_add(v_idx_170_, v___x_177_);
lean_inc_ref(v_fo_168_);
v_tailSum_179_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg(v_fo_168_, v_step_169_, v___x_178_, v_tail_176_);
lean_dec(v___x_178_);
v___x_180_ = lean_nat_mod(v_idx_170_, v_step_169_);
v___x_181_ = lean_unsigned_to_nat(0u);
v___x_182_ = lean_nat_dec_eq(v___x_180_, v___x_181_);
lean_dec(v___x_180_);
if (v___x_182_ == 0)
{
lean_dec(v_head_175_);
lean_dec_ref(v_fo_168_);
return v_tailSum_179_;
}
else
{
lean_object* v_toRingOps_183_; lean_object* v_toSemiringOps_184_; lean_object* v_add_185_; lean_object* v___x_186_; 
v_toRingOps_183_ = lean_ctor_get(v_fo_168_, 0);
lean_inc_ref(v_toRingOps_183_);
lean_dec_ref(v_fo_168_);
v_toSemiringOps_184_ = lean_ctor_get(v_toRingOps_183_, 0);
lean_inc_ref(v_toSemiringOps_184_);
lean_dec_ref(v_toRingOps_183_);
v_add_185_ = lean_ctor_get(v_toSemiringOps_184_, 3);
lean_inc(v_add_185_);
lean_dec_ref(v_toSemiringOps_184_);
v___x_186_ = lean_apply_2(v_add_185_, v_head_175_, v_tailSum_179_);
return v___x_186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg___boxed(lean_object* v_fo_187_, lean_object* v_step_188_, lean_object* v_idx_189_, lean_object* v_a_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg(v_fo_187_, v_step_188_, v_idx_189_, v_a_190_);
lean_dec(v_idx_189_);
lean_dec(v_step_188_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go(lean_object* v_EF_192_, lean_object* v_fo_193_, lean_object* v_step_194_, lean_object* v_idx_195_, lean_object* v_a_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg(v_fo_193_, v_step_194_, v_idx_195_, v_a_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___boxed(lean_object* v_EF_198_, lean_object* v_fo_199_, lean_object* v_step_200_, lean_object* v_idx_201_, lean_object* v_a_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go(v_EF_198_, v_fo_199_, v_step_200_, v_idx_201_, v_a_202_);
lean_dec(v_idx_201_);
lean_dec(v_step_200_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(lean_object* v_fo_204_, lean_object* v_coeffs_205_, lean_object* v_step_206_){
_start:
{
lean_object* v___x_207_; uint8_t v___x_208_; 
v___x_207_ = lean_unsigned_to_nat(0u);
v___x_208_ = lean_nat_dec_eq(v_step_206_, v___x_207_);
if (v___x_208_ == 0)
{
lean_object* v___x_209_; 
v___x_209_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth_go___redArg(v_fo_204_, v_step_206_, v___x_207_, v_coeffs_205_);
return v___x_209_;
}
else
{
lean_object* v_toRingOps_210_; lean_object* v_toSemiringOps_211_; lean_object* v_zero_212_; 
lean_dec(v_coeffs_205_);
v_toRingOps_210_ = lean_ctor_get(v_fo_204_, 0);
lean_inc_ref(v_toRingOps_210_);
lean_dec_ref(v_fo_204_);
v_toSemiringOps_211_ = lean_ctor_get(v_toRingOps_210_, 0);
lean_inc_ref(v_toSemiringOps_211_);
lean_dec_ref(v_toRingOps_210_);
v_zero_212_ = lean_ctor_get(v_toSemiringOps_211_, 0);
lean_inc(v_zero_212_);
lean_dec_ref(v_toSemiringOps_211_);
return v_zero_212_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg___boxed(lean_object* v_fo_213_, lean_object* v_coeffs_214_, lean_object* v_step_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(v_fo_213_, v_coeffs_214_, v_step_215_);
lean_dec(v_step_215_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth(lean_object* v_EF_217_, lean_object* v_fo_218_, lean_object* v_coeffs_219_, lean_object* v_step_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(v_fo_218_, v_coeffs_219_, v_step_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___boxed(lean_object* v_EF_222_, lean_object* v_fo_223_, lean_object* v_coeffs_224_, lean_object* v_step_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth(v_EF_222_, v_fo_223_, v_coeffs_224_, v_step_225_);
lean_dec(v_step_225_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg(lean_object* v_fo_227_, lean_object* v_x_228_, lean_object* v_x_229_){
_start:
{
if (lean_obj_tag(v_x_229_) == 0)
{
lean_dec_ref(v_fo_227_);
return v_x_228_;
}
else
{
lean_object* v_toRingOps_230_; lean_object* v_toSemiringOps_231_; lean_object* v_tail_232_; lean_object* v_fst_233_; lean_object* v_snd_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_248_; 
v_toRingOps_230_ = lean_ctor_get(v_fo_227_, 0);
v_toSemiringOps_231_ = lean_ctor_get(v_toRingOps_230_, 0);
v_tail_232_ = lean_ctor_get(v_x_229_, 1);
v_fst_233_ = lean_ctor_get(v_x_228_, 0);
v_snd_234_ = lean_ctor_get(v_x_228_, 1);
v_isSharedCheck_248_ = !lean_is_exclusive(v_x_228_);
if (v_isSharedCheck_248_ == 0)
{
v___x_236_ = v_x_228_;
v_isShared_237_ = v_isSharedCheck_248_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_snd_234_);
lean_inc(v_fst_233_);
lean_dec(v_x_228_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_248_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v_one_238_; lean_object* v_add_239_; lean_object* v_mul_240_; lean_object* v_nextPow_241_; lean_object* v_onePlusPow_242_; lean_object* v_nextSum_243_; lean_object* v___x_245_; 
v_one_238_ = lean_ctor_get(v_toSemiringOps_231_, 1);
v_add_239_ = lean_ctor_get(v_toSemiringOps_231_, 3);
v_mul_240_ = lean_ctor_get(v_toSemiringOps_231_, 4);
lean_inc_n(v_mul_240_, 2);
lean_inc_n(v_fst_233_, 2);
v_nextPow_241_ = lean_apply_2(v_mul_240_, v_fst_233_, v_fst_233_);
lean_inc(v_add_239_);
lean_inc(v_one_238_);
v_onePlusPow_242_ = lean_apply_2(v_add_239_, v_one_238_, v_fst_233_);
v_nextSum_243_ = lean_apply_2(v_mul_240_, v_snd_234_, v_onePlusPow_242_);
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 1, v_nextSum_243_);
lean_ctor_set(v___x_236_, 0, v_nextPow_241_);
v___x_245_ = v___x_236_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_nextPow_241_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v_nextSum_243_);
v___x_245_ = v_reuseFailAlloc_247_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
v_x_228_ = v___x_245_;
v_x_229_ = v_tail_232_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg___boxed(lean_object* v_fo_249_, lean_object* v_x_250_, lean_object* v_x_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg(v_fo_249_, v_x_250_, v_x_251_);
lean_dec(v_x_251_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(lean_object* v_fo_253_, lean_object* v_m_254_, lean_object* v_l_255_){
_start:
{
lean_object* v_toRingOps_256_; lean_object* v_toSemiringOps_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_268_; 
v_toRingOps_256_ = lean_ctor_get(v_fo_253_, 0);
lean_inc_ref(v_toRingOps_256_);
v_toSemiringOps_257_ = lean_ctor_get(v_toRingOps_256_, 0);
v_isSharedCheck_268_ = !lean_is_exclusive(v_toRingOps_256_);
if (v_isSharedCheck_268_ == 0)
{
lean_object* v_unused_269_; 
v_unused_269_ = lean_ctor_get(v_toRingOps_256_, 1);
lean_dec(v_unused_269_);
v___x_259_ = v_toRingOps_256_;
v_isShared_260_ = v_isSharedCheck_268_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_toSemiringOps_257_);
lean_dec(v_toRingOps_256_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_268_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v_one_261_; lean_object* v___x_263_; 
v_one_261_ = lean_ctor_get(v_toSemiringOps_257_, 1);
lean_inc(v_one_261_);
lean_dec_ref(v_toSemiringOps_257_);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 1, v_one_261_);
lean_ctor_set(v___x_259_, 0, v_m_254_);
v___x_263_ = v___x_259_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v_m_254_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_one_261_);
v___x_263_ = v_reuseFailAlloc_267_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
lean_object* v___x_264_; lean_object* v_result_265_; lean_object* v_snd_266_; 
v___x_264_ = l_List_range(v_l_255_);
v_result_265_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg(v_fo_253_, v___x_263_, v___x_264_);
lean_dec(v___x_264_);
v_snd_266_ = lean_ctor_get(v_result_265_, 1);
lean_inc(v_snd_266_);
lean_dec_ref(v_result_265_);
return v_snd_266_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2(lean_object* v_EF_270_, lean_object* v_fo_271_, lean_object* v_m_272_, lean_object* v_l_273_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(v_fo_271_, v_m_272_, v_l_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0(lean_object* v_EF_275_, lean_object* v_fo_276_, lean_object* v_x_277_, lean_object* v_x_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___redArg(v_fo_276_, v_x_277_, v_x_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0___boxed(lean_object* v_EF_280_, lean_object* v_fo_281_, lean_object* v_x_282_, lean_object* v_x_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2_spec__0(v_EF_280_, v_fo_281_, v_x_282_, v_x_283_);
lean_dec(v_x_283_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(lean_object* v_fo_285_, lean_object* v_p_286_, lean_object* v_lSkip_287_){
_start:
{
lean_object* v_toRingOps_288_; lean_object* v_toSemiringOps_289_; lean_object* v_natCast_290_; lean_object* v_mul_291_; lean_object* v___x_292_; lean_object* v_n_293_; lean_object* v_coeffSum_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v_toRingOps_288_ = lean_ctor_get(v_fo_285_, 0);
v_toSemiringOps_289_ = lean_ctor_get(v_toRingOps_288_, 0);
v_natCast_290_ = lean_ctor_get(v_toSemiringOps_289_, 2);
lean_inc(v_natCast_290_);
v_mul_291_ = lean_ctor_get(v_toSemiringOps_289_, 4);
lean_inc(v_mul_291_);
v___x_292_ = lean_unsigned_to_nat(2u);
v_n_293_ = lean_nat_pow(v___x_292_, v_lSkip_287_);
v_coeffSum_294_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(v_fo_285_, v_p_286_, v_n_293_);
v___x_295_ = lean_apply_1(v_natCast_290_, v_n_293_);
v___x_296_ = lean_apply_2(v_mul_291_, v_coeffSum_294_, v___x_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg___boxed(lean_object* v_fo_297_, lean_object* v_p_298_, lean_object* v_lSkip_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(v_fo_297_, v_p_298_, v_lSkip_299_);
lean_dec(v_lSkip_299_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain(lean_object* v_EF_301_, lean_object* v_fo_302_, lean_object* v_p_303_, lean_object* v_lSkip_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(v_fo_302_, v_p_303_, v_lSkip_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___boxed(lean_object* v_EF_306_, lean_object* v_fo_307_, lean_object* v_p_308_, lean_object* v_lSkip_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain(v_EF_306_, v_fo_307_, v_p_308_, v_lSkip_309_);
lean_dec(v_lSkip_309_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse_spec__0___redArg(lean_object* v___x_311_, lean_object* v_a_312_, lean_object* v_a_313_){
_start:
{
if (lean_obj_tag(v_a_312_) == 0)
{
lean_object* v___x_314_; 
lean_dec(v___x_311_);
v___x_314_ = l_List_reverse___redArg(v_a_313_);
return v___x_314_;
}
else
{
lean_object* v_head_315_; lean_object* v_tail_316_; lean_object* v___x_318_; uint8_t v_isShared_319_; uint8_t v_isSharedCheck_325_; 
v_head_315_ = lean_ctor_get(v_a_312_, 0);
v_tail_316_ = lean_ctor_get(v_a_312_, 1);
v_isSharedCheck_325_ = !lean_is_exclusive(v_a_312_);
if (v_isSharedCheck_325_ == 0)
{
v___x_318_ = v_a_312_;
v_isShared_319_ = v_isSharedCheck_325_;
goto v_resetjp_317_;
}
else
{
lean_inc(v_tail_316_);
lean_inc(v_head_315_);
lean_dec(v_a_312_);
v___x_318_ = lean_box(0);
v_isShared_319_ = v_isSharedCheck_325_;
goto v_resetjp_317_;
}
v_resetjp_317_:
{
lean_object* v___x_320_; lean_object* v___x_322_; 
lean_inc(v___x_311_);
v___x_320_ = lean_apply_1(v___x_311_, v_head_315_);
if (v_isShared_319_ == 0)
{
lean_ctor_set(v___x_318_, 1, v_a_313_);
lean_ctor_set(v___x_318_, 0, v___x_320_);
v___x_322_ = v___x_318_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v___x_320_);
lean_ctor_set(v_reuseFailAlloc_324_, 1, v_a_313_);
v___x_322_ = v_reuseFailAlloc_324_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
v_a_312_ = v_tail_316_;
v_a_313_ = v___x_322_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(lean_object* v_fo_326_, lean_object* v_xs_327_){
_start:
{
lean_object* v_inv_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v_inv_328_ = lean_ctor_get(v_fo_326_, 1);
lean_inc(v_inv_328_);
lean_dec_ref(v_fo_326_);
v___x_329_ = lean_box(0);
v___x_330_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse_spec__0___redArg(v_inv_328_, v_xs_327_, v___x_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse(lean_object* v_EF_331_, lean_object* v_fo_332_, lean_object* v_xs_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(v_fo_332_, v_xs_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse_spec__0(lean_object* v_EF_335_, lean_object* v___x_336_, lean_object* v_a_337_, lean_object* v_a_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse_spec__0___redArg(v___x_336_, v_a_337_, v_a_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__0(lean_object* v_mul_340_, lean_object* v_fst_341_, lean_object* v_snd_342_, lean_object* v_sub_343_, lean_object* v_fst_344_, lean_object* v_half_345_, lean_object* v_add_346_, lean_object* v_hi_347_, lean_object* v_zero_348_, lean_object* v_lo_349_, lean_object* v_stride_350_, lean_object* v_invTw_351_, lean_object* v_tw_352_, lean_object* v_i_353_){
_start:
{
lean_object* v___y_355_; lean_object* v___y_356_; lean_object* v___y_357_; lean_object* v___y_358_; lean_object* v___y_368_; lean_object* v___y_369_; lean_object* v___y_370_; lean_object* v___y_374_; lean_object* v___y_375_; lean_object* v_twIdx_378_; lean_object* v___y_380_; lean_object* v___x_383_; 
v_twIdx_378_ = lean_nat_mul(v_i_353_, v_stride_350_);
lean_inc(v_twIdx_378_);
v___x_383_ = l_List_get_x3fInternal___redArg(v_tw_352_, v_twIdx_378_);
if (lean_obj_tag(v___x_383_) == 0)
{
lean_inc(v_zero_348_);
v___y_380_ = v_zero_348_;
goto v___jp_379_;
}
else
{
lean_object* v_val_384_; 
v_val_384_ = lean_ctor_get(v___x_383_, 0);
lean_inc(v_val_384_);
lean_dec_ref_known(v___x_383_, 1);
v___y_380_ = v_val_384_;
goto v___jp_379_;
}
v___jp_354_:
{
lean_object* v_t_359_; lean_object* v_tInv_360_; lean_object* v_alphaMinusT_361_; lean_object* v_loMinusHi_362_; lean_object* v_prod0_363_; lean_object* v_prod1_364_; lean_object* v_scaled_365_; lean_object* v___x_366_; 
lean_inc_n(v_mul_340_, 4);
v_t_359_ = lean_apply_2(v_mul_340_, v___y_357_, v_fst_341_);
v_tInv_360_ = lean_apply_2(v_mul_340_, v___y_356_, v_snd_342_);
lean_inc(v_sub_343_);
v_alphaMinusT_361_ = lean_apply_2(v_sub_343_, v_fst_344_, v_t_359_);
lean_inc(v___y_355_);
v_loMinusHi_362_ = lean_apply_2(v_sub_343_, v___y_355_, v___y_358_);
v_prod0_363_ = lean_apply_2(v_mul_340_, v_alphaMinusT_361_, v_loMinusHi_362_);
v_prod1_364_ = lean_apply_2(v_mul_340_, v_prod0_363_, v_tInv_360_);
v_scaled_365_ = lean_apply_2(v_mul_340_, v_prod1_364_, v_half_345_);
v___x_366_ = lean_apply_2(v_add_346_, v___y_355_, v_scaled_365_);
return v___x_366_;
}
v___jp_367_:
{
lean_object* v___x_371_; 
v___x_371_ = l_List_get_x3fInternal___redArg(v_hi_347_, v_i_353_);
if (lean_obj_tag(v___x_371_) == 0)
{
v___y_355_ = v___y_370_;
v___y_356_ = v___y_368_;
v___y_357_ = v___y_369_;
v___y_358_ = v_zero_348_;
goto v___jp_354_;
}
else
{
lean_object* v_val_372_; 
lean_dec(v_zero_348_);
v_val_372_ = lean_ctor_get(v___x_371_, 0);
lean_inc(v_val_372_);
lean_dec_ref_known(v___x_371_, 1);
v___y_355_ = v___y_370_;
v___y_356_ = v___y_368_;
v___y_357_ = v___y_369_;
v___y_358_ = v_val_372_;
goto v___jp_354_;
}
}
v___jp_373_:
{
lean_object* v___x_376_; 
lean_inc(v_i_353_);
v___x_376_ = l_List_get_x3fInternal___redArg(v_lo_349_, v_i_353_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_inc(v_zero_348_);
v___y_368_ = v___y_375_;
v___y_369_ = v___y_374_;
v___y_370_ = v_zero_348_;
goto v___jp_367_;
}
else
{
lean_object* v_val_377_; 
v_val_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_val_377_);
lean_dec_ref_known(v___x_376_, 1);
v___y_368_ = v___y_375_;
v___y_369_ = v___y_374_;
v___y_370_ = v_val_377_;
goto v___jp_367_;
}
}
v___jp_379_:
{
lean_object* v___x_381_; 
v___x_381_ = l_List_get_x3fInternal___redArg(v_invTw_351_, v_twIdx_378_);
if (lean_obj_tag(v___x_381_) == 0)
{
lean_inc(v_zero_348_);
v___y_374_ = v___y_380_;
v___y_375_ = v_zero_348_;
goto v___jp_373_;
}
else
{
lean_object* v_val_382_; 
v_val_382_ = lean_ctor_get(v___x_381_, 0);
lean_inc(v_val_382_);
lean_dec_ref_known(v___x_381_, 1);
v___y_374_ = v___y_380_;
v___y_375_ = v_val_382_;
goto v___jp_373_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__0___boxed(lean_object* v_mul_385_, lean_object* v_fst_386_, lean_object* v_snd_387_, lean_object* v_sub_388_, lean_object* v_fst_389_, lean_object* v_half_390_, lean_object* v_add_391_, lean_object* v_hi_392_, lean_object* v_zero_393_, lean_object* v_lo_394_, lean_object* v_stride_395_, lean_object* v_invTw_396_, lean_object* v_tw_397_, lean_object* v_i_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__0(v_mul_385_, v_fst_386_, v_snd_387_, v_sub_388_, v_fst_389_, v_half_390_, v_add_391_, v_hi_392_, v_zero_393_, v_lo_394_, v_stride_395_, v_invTw_396_, v_tw_397_, v_i_398_);
lean_dec(v_tw_397_);
lean_dec(v_invTw_396_);
lean_dec(v_stride_395_);
lean_dec(v_lo_394_);
lean_dec(v_hi_392_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1(lean_object* v___x_402_, lean_object* v___x_403_, lean_object* v_mul_404_, lean_object* v_sub_405_, lean_object* v_half_406_, lean_object* v_add_407_, lean_object* v_zero_408_, lean_object* v_invTw_409_, lean_object* v_tw_410_, lean_object* v_cur_411_, lean_object* v_entry_412_){
_start:
{
lean_object* v_snd_413_; lean_object* v_snd_414_; lean_object* v_fst_415_; lean_object* v_fst_416_; lean_object* v_fst_417_; lean_object* v_snd_418_; lean_object* v___x_419_; lean_object* v_m_420_; lean_object* v___x_421_; lean_object* v_lo_422_; lean_object* v_hi_423_; lean_object* v_stride_424_; lean_object* v___f_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v_snd_413_ = lean_ctor_get(v_entry_412_, 1);
lean_inc(v_snd_413_);
v_snd_414_ = lean_ctor_get(v_snd_413_, 1);
lean_inc(v_snd_414_);
v_fst_415_ = lean_ctor_get(v_entry_412_, 0);
lean_inc(v_fst_415_);
lean_dec_ref(v_entry_412_);
v_fst_416_ = lean_ctor_get(v_snd_413_, 0);
lean_inc(v_fst_416_);
lean_dec(v_snd_413_);
v_fst_417_ = lean_ctor_get(v_snd_414_, 0);
lean_inc(v_fst_417_);
v_snd_418_ = lean_ctor_get(v_snd_414_, 1);
lean_inc(v_snd_418_);
lean_dec(v_snd_414_);
v___x_419_ = l_List_lengthTR___redArg(v_cur_411_);
v_m_420_ = lean_nat_shiftr(v___x_419_, v___x_402_);
lean_dec(v___x_419_);
v___x_421_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___closed__0));
lean_inc_n(v_m_420_, 2);
lean_inc(v_cur_411_);
v_lo_422_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_cur_411_, v_cur_411_, v_m_420_, v___x_421_);
v_hi_423_ = l_List_drop___redArg(v_m_420_, v_cur_411_);
lean_dec(v_cur_411_);
v_stride_424_ = lean_nat_pow(v___x_403_, v_fst_415_);
lean_dec(v_fst_415_);
v___f_425_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__0___boxed), 14, 13);
lean_closure_set(v___f_425_, 0, v_mul_404_);
lean_closure_set(v___f_425_, 1, v_fst_417_);
lean_closure_set(v___f_425_, 2, v_snd_418_);
lean_closure_set(v___f_425_, 3, v_sub_405_);
lean_closure_set(v___f_425_, 4, v_fst_416_);
lean_closure_set(v___f_425_, 5, v_half_406_);
lean_closure_set(v___f_425_, 6, v_add_407_);
lean_closure_set(v___f_425_, 7, v_hi_423_);
lean_closure_set(v___f_425_, 8, v_zero_408_);
lean_closure_set(v___f_425_, 9, v_lo_422_);
lean_closure_set(v___f_425_, 10, v_stride_424_);
lean_closure_set(v___f_425_, 11, v_invTw_409_);
lean_closure_set(v___f_425_, 12, v_tw_410_);
v___x_426_ = l_List_range(v_m_420_);
v___x_427_ = lean_box(0);
v___x_428_ = l_List_mapTR_loop___redArg(v___f_425_, v___x_426_, v___x_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___boxed(lean_object* v___x_429_, lean_object* v___x_430_, lean_object* v_mul_431_, lean_object* v_sub_432_, lean_object* v_half_433_, lean_object* v_add_434_, lean_object* v_zero_435_, lean_object* v_invTw_436_, lean_object* v_tw_437_, lean_object* v_cur_438_, lean_object* v_entry_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1(v___x_429_, v___x_430_, v_mul_431_, v_sub_432_, v_half_433_, v_add_434_, v_zero_435_, v_invTw_436_, v_tw_437_, v_cur_438_, v_entry_439_);
lean_dec(v___x_430_);
lean_dec(v___x_429_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg(lean_object* v_fo_441_, lean_object* v_inst_442_, lean_object* v_values_443_, lean_object* v_alphas_444_, lean_object* v_x_445_){
_start:
{
lean_object* v_toRingOps_446_; lean_object* v_inv_447_; lean_object* v_k_448_; lean_object* v_omegaK_449_; lean_object* v_invOmegaK_450_; lean_object* v_toSemiringOps_451_; lean_object* v_sub_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v_zero_455_; lean_object* v_natCast_456_; lean_object* v_add_457_; lean_object* v_mul_458_; lean_object* v___x_459_; lean_object* v_twLen_460_; lean_object* v_tw_461_; lean_object* v_xPows_462_; lean_object* v_invTw_463_; lean_object* v_xInv_464_; lean_object* v_xInvPows_465_; lean_object* v___x_466_; lean_object* v_half_467_; lean_object* v___f_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v_folded_473_; 
v_toRingOps_446_ = lean_ctor_get(v_fo_441_, 0);
v_inv_447_ = lean_ctor_get(v_fo_441_, 1);
lean_inc_n(v_inv_447_, 3);
v_k_448_ = l_List_lengthTR___redArg(v_alphas_444_);
lean_inc_n(v_k_448_, 3);
v_omegaK_449_ = lean_apply_1(v_inst_442_, v_k_448_);
lean_inc(v_omegaK_449_);
v_invOmegaK_450_ = lean_apply_1(v_inv_447_, v_omegaK_449_);
v_toSemiringOps_451_ = lean_ctor_get(v_toRingOps_446_, 0);
lean_inc_ref_n(v_toSemiringOps_451_, 2);
v_sub_452_ = lean_ctor_get(v_toRingOps_446_, 1);
lean_inc(v_sub_452_);
v___x_453_ = lean_unsigned_to_nat(1u);
v___x_454_ = lean_nat_sub(v_k_448_, v___x_453_);
v_zero_455_ = lean_ctor_get(v_toSemiringOps_451_, 0);
lean_inc_n(v_zero_455_, 2);
v_natCast_456_ = lean_ctor_get(v_toSemiringOps_451_, 2);
lean_inc(v_natCast_456_);
v_add_457_ = lean_ctor_get(v_toSemiringOps_451_, 3);
lean_inc(v_add_457_);
v_mul_458_ = lean_ctor_get(v_toSemiringOps_451_, 4);
lean_inc(v_mul_458_);
v___x_459_ = lean_unsigned_to_nat(2u);
v_twLen_460_ = lean_nat_pow(v___x_459_, v___x_454_);
lean_dec(v___x_454_);
lean_inc(v_twLen_460_);
lean_inc_ref(v_fo_441_);
v_tw_461_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_441_, v_omegaK_449_, v_twLen_460_);
v_xPows_462_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_451_, v_x_445_, v_k_448_);
v_invTw_463_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_441_, v_invOmegaK_450_, v_twLen_460_);
v_xInv_464_ = lean_apply_1(v_inv_447_, v_x_445_);
v_xInvPows_465_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_451_, v_xInv_464_, v_k_448_);
lean_dec(v_xInv_464_);
v___x_466_ = lean_apply_1(v_natCast_456_, v___x_459_);
v_half_467_ = lean_apply_1(v_inv_447_, v___x_466_);
v___f_468_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg___lam__1___boxed), 11, 9);
lean_closure_set(v___f_468_, 0, v___x_453_);
lean_closure_set(v___f_468_, 1, v___x_459_);
lean_closure_set(v___f_468_, 2, v_mul_458_);
lean_closure_set(v___f_468_, 3, v_sub_452_);
lean_closure_set(v___f_468_, 4, v_half_467_);
lean_closure_set(v___f_468_, 5, v_add_457_);
lean_closure_set(v___f_468_, 6, v_zero_455_);
lean_closure_set(v___f_468_, 7, v_invTw_463_);
lean_closure_set(v___f_468_, 8, v_tw_461_);
v___x_469_ = l_List_range(v_k_448_);
v___x_470_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_xPows_462_, v_xInvPows_465_);
v___x_471_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_alphas_444_, v___x_470_);
v___x_472_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_469_, v___x_471_);
v_folded_473_ = l_List_foldl___redArg(v___f_468_, v_values_443_, v___x_472_);
if (lean_obj_tag(v_folded_473_) == 0)
{
return v_zero_455_;
}
else
{
lean_object* v_head_474_; 
lean_dec(v_zero_455_);
v_head_474_ = lean_ctor_get(v_folded_473_, 0);
lean_inc(v_head_474_);
lean_dec_ref_known(v_folded_473_, 2);
return v_head_474_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold(lean_object* v_EF_475_, lean_object* v_fo_476_, lean_object* v_inst_477_, lean_object* v_values_478_, lean_object* v_alphas_479_, lean_object* v_x_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg(v_fo_476_, v_inst_477_, v_values_478_, v_alphas_479_, v_x_480_);
return v___x_481_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_FieldOps(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Proof(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Runtime_PolyCommon(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_VerifyingKey(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Common(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_FieldOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Proof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Runtime_PolyCommon(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_VerifyingKey(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
