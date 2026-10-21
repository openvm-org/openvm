// Lean compiler output
// Module: VmVerifier.DumpProof
// Imports: public import Init public meta import Init public import VmVerifier.Spec.Wire
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
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* lean_get_stdin();
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
lean_object* lean_get_stderr();
lean_object* lp_openvm_x2dfv_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
lean_object* lean_byte_array_size(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_Spec_Wire_readBaseline(lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(lean_object*, lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
lean_object* lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_rawFieldsToCsv(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_vkDigest(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_proofDigest(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_vkCommitDigest(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_baselineDigest(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest(lean_object*);
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode___boxed(lean_object*);
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "vm_dump_proof: stdin framing error (received "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__0_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " bytes)"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__1_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "vm_dump_proof: vk parse error: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__2 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__2_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "vm_dump_proof: baseline parse error: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__3 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__3_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "vm_dump_proof: proof parse error: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__4 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__4_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "vm_dump_proof: public-values parse error: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__5 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__5_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "vm_dump_proof: user-PV proof parse error: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__6 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__6_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "vk: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__7 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__7_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "baseline: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__8 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__8_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "proof: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__9 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__9_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "pv: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__10 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__10_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "user-pvs: "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__11 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__11_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__1;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__2;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main();
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed(lean_object*);
LEAN_EXPORT lean_object* _lean_main();
LEAN_EXPORT lean_object* lp_openvm_x2dfv_main___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(lean_object* v_a_1_, lean_object* v_a_2_){
_start:
{
if (lean_obj_tag(v_a_1_) == 0)
{
lean_object* v___x_3_; 
v___x_3_ = l_List_reverse___redArg(v_a_2_);
return v___x_3_;
}
else
{
lean_object* v_head_4_; lean_object* v_tail_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_16_; 
v_head_4_ = lean_ctor_get(v_a_1_, 0);
v_tail_5_ = lean_ctor_get(v_a_1_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v_a_1_);
if (v_isSharedCheck_16_ == 0)
{
v___x_7_ = v_a_1_;
v_isShared_8_ = v_isSharedCheck_16_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_tail_5_);
lean_inc(v_head_4_);
lean_dec(v_a_1_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_16_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
uint32_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_13_; 
v___x_9_ = lean_unbox_uint32(v_head_4_);
lean_dec(v_head_4_);
v___x_10_ = lean_uint32_to_nat(v___x_9_);
v___x_11_ = l_Nat_reprFast(v___x_10_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 1, v_a_2_);
lean_ctor_set(v___x_7_, 0, v___x_11_);
v___x_13_ = v___x_7_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v___x_11_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v_a_2_);
v___x_13_ = v_reuseFailAlloc_15_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
v_a_1_ = v_tail_5_;
v_a_2_ = v___x_13_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv(lean_object* v_digest_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_19_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_20_ = lean_array_to_list(v_digest_18_);
v___x_21_ = lean_box(0);
v___x_22_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(v___x_20_, v___x_21_);
v___x_23_ = l_String_intercalate(v___x_19_, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_rawFieldsToCsv(lean_object* v_values_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_25_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_26_ = lean_array_to_list(v_values_24_);
v___x_27_ = lean_box(0);
v___x_28_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(v___x_26_, v___x_27_);
v___x_29_ = l_String_intercalate(v___x_25_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_vkDigest(lean_object* v_vk_30_){
_start:
{
lean_object* v_preHash_31_; lean_object* v___x_32_; 
v_preHash_31_ = lean_ctor_get(v_vk_30_, 1);
lean_inc_ref(v_preHash_31_);
lean_dec_ref(v_vk_30_);
v___x_32_ = lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv(v_preHash_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_proofDigest(lean_object* v_proof_33_){
_start:
{
lean_object* v_commonMainCommit_34_; lean_object* v___x_35_; 
v_commonMainCommit_34_ = lean_ctor_get(v_proof_33_, 0);
lean_inc_ref(v_commonMainCommit_34_);
lean_dec_ref(v_proof_33_);
v___x_35_ = lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv(v_commonMainCommit_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
if (lean_obj_tag(v_a_36_) == 0)
{
lean_object* v___x_38_; 
v___x_38_ = l_List_reverse___redArg(v_a_37_);
return v___x_38_;
}
else
{
lean_object* v_head_39_; lean_object* v_tail_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_49_; 
v_head_39_ = lean_ctor_get(v_a_36_, 0);
v_tail_40_ = lean_ctor_get(v_a_36_, 1);
v_isSharedCheck_49_ = !lean_is_exclusive(v_a_36_);
if (v_isSharedCheck_49_ == 0)
{
v___x_42_ = v_a_36_;
v_isShared_43_ = v_isSharedCheck_49_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_tail_40_);
lean_inc(v_head_39_);
lean_dec(v_a_36_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_49_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v___x_44_; lean_object* v___x_46_; 
v___x_44_ = lp_openvm_x2dfv_VmVerifier_DumpProof_rawFieldsToCsv(v_head_39_);
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 1, v_a_37_);
lean_ctor_set(v___x_42_, 0, v___x_44_);
v___x_46_ = v___x_42_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v___x_44_);
lean_ctor_set(v_reuseFailAlloc_48_, 1, v_a_37_);
v___x_46_ = v_reuseFailAlloc_48_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
v_a_36_ = v_tail_40_;
v_a_37_ = v___x_46_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest(lean_object* v_publicValues_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_52_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest___closed__0));
v___x_53_ = lean_array_to_list(v_publicValues_51_);
v___x_54_ = lean_box(0);
v___x_55_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(v___x_53_, v___x_54_);
v___x_56_ = l_String_intercalate(v___x_52_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
if (lean_obj_tag(v_a_57_) == 0)
{
lean_object* v___x_59_; 
v___x_59_ = l_List_reverse___redArg(v_a_58_);
return v___x_59_;
}
else
{
lean_object* v_head_60_; lean_object* v_tail_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_70_; 
v_head_60_ = lean_ctor_get(v_a_57_, 0);
v_tail_61_ = lean_ctor_get(v_a_57_, 1);
v_isSharedCheck_70_ = !lean_is_exclusive(v_a_57_);
if (v_isSharedCheck_70_ == 0)
{
v___x_63_ = v_a_57_;
v_isShared_64_ = v_isSharedCheck_70_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_tail_61_);
lean_inc(v_head_60_);
lean_dec(v_a_57_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_70_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v___x_65_; lean_object* v___x_67_; 
v___x_65_ = l_Nat_reprFast(v_head_60_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 1, v_a_58_);
lean_ctor_set(v___x_63_, 0, v___x_65_);
v___x_67_ = v___x_63_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_69_; 
v_reuseFailAlloc_69_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_69_, 0, v___x_65_);
lean_ctor_set(v_reuseFailAlloc_69_, 1, v_a_58_);
v___x_67_ = v_reuseFailAlloc_69_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
v_a_57_ = v_tail_61_;
v_a_58_ = v___x_67_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(lean_object* v_digest_71_){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_72_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_73_ = lean_array_to_list(v_digest_71_);
v___x_74_ = lean_box(0);
v___x_75_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(v___x_73_, v___x_74_);
v___x_76_ = l_String_intercalate(v___x_72_, v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_vkCommitDigest(lean_object* v_commit_77_){
_start:
{
lean_object* v_cachedCommit_78_; lean_object* v_vkPreHash_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_90_; 
v_cachedCommit_78_ = lean_ctor_get(v_commit_77_, 0);
v_vkPreHash_79_ = lean_ctor_get(v_commit_77_, 1);
v_isSharedCheck_90_ = !lean_is_exclusive(v_commit_77_);
if (v_isSharedCheck_90_ == 0)
{
v___x_81_ = v_commit_77_;
v_isShared_82_ = v_isSharedCheck_90_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_vkPreHash_79_);
lean_inc(v_cachedCommit_78_);
lean_dec(v_commit_77_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_90_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_87_; 
v___x_83_ = lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(v_cachedCommit_78_);
v___x_84_ = lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(v_vkPreHash_79_);
v___x_85_ = lean_box(0);
if (v_isShared_82_ == 0)
{
lean_ctor_set_tag(v___x_81_, 1);
lean_ctor_set(v___x_81_, 1, v___x_85_);
lean_ctor_set(v___x_81_, 0, v___x_84_);
v___x_87_ = v___x_81_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_89_; 
v_reuseFailAlloc_89_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_89_, 0, v___x_84_);
lean_ctor_set(v_reuseFailAlloc_89_, 1, v___x_85_);
v___x_87_ = v_reuseFailAlloc_89_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
lean_object* v___x_88_; 
v___x_88_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_83_);
lean_ctor_set(v___x_88_, 1, v___x_87_);
return v___x_88_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_baselineDigest(lean_object* v_baseline_91_){
_start:
{
lean_object* v_memoryDimensions_92_; lean_object* v_programCommit_93_; lean_object* v_initialState_94_; lean_object* v_initialPc_95_; lean_object* v_numUserPvs_96_; lean_object* v_appVkCommit_97_; lean_object* v_leafVkCommit_98_; lean_object* v_internalForLeafVkCommit_99_; lean_object* v_internalRecursiveVkCommit_100_; lean_object* v_addrSpaceHeight_101_; lean_object* v_addressHeight_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_131_; 
v_memoryDimensions_92_ = lean_ctor_get(v_baseline_91_, 3);
lean_inc_ref(v_memoryDimensions_92_);
v_programCommit_93_ = lean_ctor_get(v_baseline_91_, 0);
lean_inc_ref(v_programCommit_93_);
v_initialState_94_ = lean_ctor_get(v_baseline_91_, 1);
lean_inc_ref(v_initialState_94_);
v_initialPc_95_ = lean_ctor_get(v_baseline_91_, 2);
lean_inc(v_initialPc_95_);
v_numUserPvs_96_ = lean_ctor_get(v_baseline_91_, 4);
lean_inc(v_numUserPvs_96_);
v_appVkCommit_97_ = lean_ctor_get(v_baseline_91_, 5);
lean_inc_ref(v_appVkCommit_97_);
v_leafVkCommit_98_ = lean_ctor_get(v_baseline_91_, 6);
lean_inc_ref(v_leafVkCommit_98_);
v_internalForLeafVkCommit_99_ = lean_ctor_get(v_baseline_91_, 7);
lean_inc_ref(v_internalForLeafVkCommit_99_);
v_internalRecursiveVkCommit_100_ = lean_ctor_get(v_baseline_91_, 8);
lean_inc_ref(v_internalRecursiveVkCommit_100_);
lean_dec_ref(v_baseline_91_);
v_addrSpaceHeight_101_ = lean_ctor_get(v_memoryDimensions_92_, 0);
v_addressHeight_102_ = lean_ctor_get(v_memoryDimensions_92_, 1);
v_isSharedCheck_131_ = !lean_is_exclusive(v_memoryDimensions_92_);
if (v_isSharedCheck_131_ == 0)
{
v___x_104_ = v_memoryDimensions_92_;
v_isShared_105_ = v_isSharedCheck_131_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_addressHeight_102_);
lean_inc(v_addrSpaceHeight_101_);
lean_dec(v_memoryDimensions_92_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_131_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_115_; 
v___x_106_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest___closed__0));
v___x_107_ = lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(v_programCommit_93_);
v___x_108_ = lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(v_initialState_94_);
v___x_109_ = l_Nat_reprFast(v_initialPc_95_);
v___x_110_ = l_Nat_reprFast(v_addrSpaceHeight_101_);
v___x_111_ = l_Nat_reprFast(v_addressHeight_102_);
v___x_112_ = l_Nat_reprFast(v_numUserPvs_96_);
v___x_113_ = lean_box(0);
if (v_isShared_105_ == 0)
{
lean_ctor_set_tag(v___x_104_, 1);
lean_ctor_set(v___x_104_, 1, v___x_113_);
lean_ctor_set(v___x_104_, 0, v___x_112_);
v___x_115_ = v___x_104_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v___x_112_);
lean_ctor_set(v_reuseFailAlloc_130_, 1, v___x_113_);
v___x_115_ = v_reuseFailAlloc_130_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_111_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
v___x_117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_110_);
lean_ctor_set(v___x_117_, 1, v___x_116_);
v___x_118_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_109_);
lean_ctor_set(v___x_118_, 1, v___x_117_);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_108_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
v___x_120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_107_);
lean_ctor_set(v___x_120_, 1, v___x_119_);
v___x_121_ = lp_openvm_x2dfv_VmVerifier_DumpProof_vkCommitDigest(v_appVkCommit_97_);
v___x_122_ = l_List_appendTR___redArg(v___x_120_, v___x_121_);
v___x_123_ = lp_openvm_x2dfv_VmVerifier_DumpProof_vkCommitDigest(v_leafVkCommit_98_);
v___x_124_ = l_List_appendTR___redArg(v___x_122_, v___x_123_);
v___x_125_ = lp_openvm_x2dfv_VmVerifier_DumpProof_vkCommitDigest(v_internalForLeafVkCommit_99_);
v___x_126_ = l_List_appendTR___redArg(v___x_124_, v___x_125_);
v___x_127_ = lp_openvm_x2dfv_VmVerifier_DumpProof_vkCommitDigest(v_internalRecursiveVkCommit_100_);
v___x_128_ = l_List_appendTR___redArg(v___x_126_, v___x_127_);
v___x_129_ = l_String_intercalate(v___x_106_, v___x_128_);
return v___x_129_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
if (lean_obj_tag(v_a_132_) == 0)
{
lean_object* v___x_134_; 
v___x_134_ = l_List_reverse___redArg(v_a_133_);
return v___x_134_;
}
else
{
lean_object* v_head_135_; lean_object* v_tail_136_; lean_object* v___x_138_; uint8_t v_isShared_139_; uint8_t v_isSharedCheck_145_; 
v_head_135_ = lean_ctor_get(v_a_132_, 0);
v_tail_136_ = lean_ctor_get(v_a_132_, 1);
v_isSharedCheck_145_ = !lean_is_exclusive(v_a_132_);
if (v_isSharedCheck_145_ == 0)
{
v___x_138_ = v_a_132_;
v_isShared_139_ = v_isSharedCheck_145_;
goto v_resetjp_137_;
}
else
{
lean_inc(v_tail_136_);
lean_inc(v_head_135_);
lean_dec(v_a_132_);
v___x_138_ = lean_box(0);
v_isShared_139_ = v_isSharedCheck_145_;
goto v_resetjp_137_;
}
v_resetjp_137_:
{
lean_object* v___x_140_; lean_object* v___x_142_; 
v___x_140_ = lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(v_head_135_);
if (v_isShared_139_ == 0)
{
lean_ctor_set(v___x_138_, 1, v_a_133_);
lean_ctor_set(v___x_138_, 0, v___x_140_);
v___x_142_ = v___x_138_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v___x_140_);
lean_ctor_set(v_reuseFailAlloc_144_, 1, v_a_133_);
v___x_142_ = v_reuseFailAlloc_144_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
v_a_132_ = v_tail_136_;
v_a_133_ = v___x_142_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest(lean_object* v_proof_147_){
_start:
{
lean_object* v_authenticationPath_148_; lean_object* v_publicValues_149_; lean_object* v_publicValuesCommit_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v_authenticationPath_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v_publicValues_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v_authenticationPath_148_ = lean_ctor_get(v_proof_147_, 0);
lean_inc(v_authenticationPath_148_);
v_publicValues_149_ = lean_ctor_get(v_proof_147_, 1);
lean_inc(v_publicValues_149_);
v_publicValuesCommit_150_ = lean_ctor_get(v_proof_147_, 2);
lean_inc_ref(v_publicValuesCommit_150_);
lean_dec_ref(v_proof_147_);
v___x_151_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest___closed__0));
v___x_152_ = lean_box(0);
v___x_153_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(v_authenticationPath_148_, v___x_152_);
v_authenticationPath_154_ = l_String_intercalate(v___x_151_, v___x_153_);
v___x_155_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_156_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(v_publicValues_149_, v___x_152_);
v_publicValues_157_ = l_String_intercalate(v___x_155_, v___x_156_);
v___x_158_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest___closed__0));
v___x_159_ = lp_openvm_x2dfv_VmVerifier_DumpProof_digestToCsv(v_publicValuesCommit_150_);
v___x_160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v___x_152_);
v___x_161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_161_, 0, v_publicValues_157_);
lean_ctor_set(v___x_161_, 1, v___x_160_);
v___x_162_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_162_, 0, v_authenticationPath_154_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
v___x_163_ = l_String_intercalate(v___x_158_, v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(lean_object* v_x_164_){
_start:
{
switch(lean_obj_tag(v_x_164_))
{
case 0:
{
uint32_t v___x_165_; 
v___x_165_ = 10;
return v___x_165_;
}
case 1:
{
uint32_t v___x_166_; 
v___x_166_ = 11;
return v___x_166_;
}
case 2:
{
uint32_t v___x_167_; 
v___x_167_ = 12;
return v___x_167_;
}
default: 
{
uint32_t v___x_168_; 
v___x_168_ = 13;
return v___x_168_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode___boxed(lean_object* v_x_169_){
_start:
{
uint32_t v_res_170_; lean_object* v_r_171_; 
v_res_170_ = lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(v_x_169_);
lean_dec_ref(v_x_169_);
v_r_171_ = lean_box_uint32(v_res_170_);
return v_r_171_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__1(void){
_start:
{
uint32_t v___x_184_; lean_object* v___x_185_; 
v___x_184_ = 20;
v___x_185_ = lean_box_uint32(v___x_184_);
return v___x_185_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__2(void){
_start:
{
uint32_t v___x_186_; lean_object* v___x_187_; 
v___x_186_ = 0;
v___x_187_ = lean_box_uint32(v___x_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main(){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_189_ = lean_get_stdin();
v___x_190_ = l_IO_FS_Stream_readBinToEnd(v___x_189_);
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v_a_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v_a_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_a_191_);
lean_dec_ref_known(v___x_190_, 1);
v___x_192_ = lean_get_stderr();
v___x_193_ = lp_openvm_x2dfv_VmVerifier_Spec_Wire_parseFiveBlobs(v_a_191_);
if (lean_obj_tag(v___x_193_) == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_194_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__0));
v___x_195_ = lean_byte_array_size(v_a_191_);
lean_dec(v_a_191_);
v___x_196_ = l_Nat_reprFast(v___x_195_);
v___x_197_ = lean_string_append(v___x_194_, v___x_196_);
lean_dec_ref(v___x_196_);
v___x_198_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__1));
v___x_199_ = lean_string_append(v___x_197_, v___x_198_);
v___x_200_ = l_IO_FS_Stream_putStrLn(v___x_192_, v___x_199_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_208_; 
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_208_ == 0)
{
lean_object* v_unused_209_; 
v_unused_209_ = lean_ctor_get(v___x_200_, 0);
lean_dec(v_unused_209_);
v___x_202_ = v___x_200_;
v_isShared_203_ = v_isSharedCheck_208_;
goto v_resetjp_201_;
}
else
{
lean_dec(v___x_200_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_208_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_204_; lean_object* v___x_206_; 
v___x_204_ = lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__1;
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 0, v___x_204_);
v___x_206_ = v___x_202_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v___x_204_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
else
{
lean_object* v_a_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_217_; 
v_a_210_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_217_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_217_ == 0)
{
v___x_212_ = v___x_200_;
v_isShared_213_ = v_isSharedCheck_217_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_a_210_);
lean_dec(v___x_200_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_217_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_215_; 
if (v_isShared_213_ == 0)
{
v___x_215_ = v___x_212_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v_a_210_);
v___x_215_ = v_reuseFailAlloc_216_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
return v___x_215_;
}
}
}
}
else
{
lean_object* v_val_218_; lean_object* v_snd_219_; lean_object* v_snd_220_; lean_object* v_snd_221_; lean_object* v_fst_222_; lean_object* v_fst_223_; lean_object* v_fst_224_; lean_object* v_fst_225_; lean_object* v_snd_226_; lean_object* v___x_227_; 
lean_dec(v_a_191_);
v_val_218_ = lean_ctor_get(v___x_193_, 0);
lean_inc(v_val_218_);
lean_dec_ref_known(v___x_193_, 1);
v_snd_219_ = lean_ctor_get(v_val_218_, 1);
lean_inc(v_snd_219_);
v_snd_220_ = lean_ctor_get(v_snd_219_, 1);
lean_inc(v_snd_220_);
v_snd_221_ = lean_ctor_get(v_snd_220_, 1);
lean_inc(v_snd_221_);
v_fst_222_ = lean_ctor_get(v_val_218_, 0);
lean_inc(v_fst_222_);
lean_dec(v_val_218_);
v_fst_223_ = lean_ctor_get(v_snd_219_, 0);
lean_inc(v_fst_223_);
lean_dec(v_snd_219_);
v_fst_224_ = lean_ctor_get(v_snd_220_, 0);
lean_inc(v_fst_224_);
lean_dec(v_snd_220_);
v_fst_225_ = lean_ctor_get(v_snd_221_, 0);
lean_inc(v_fst_225_);
v_snd_226_ = lean_ctor_get(v_snd_221_, 1);
lean_inc(v_snd_226_);
lean_dec(v_snd_221_);
v___x_227_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(v_fst_222_);
if (lean_obj_tag(v___x_227_) == 0)
{
lean_object* v_a_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
lean_dec(v_snd_226_);
lean_dec(v_fst_225_);
lean_dec(v_fst_224_);
lean_dec(v_fst_223_);
v_a_228_ = lean_ctor_get(v___x_227_, 0);
lean_inc_n(v_a_228_, 2);
lean_dec_ref_known(v___x_227_, 1);
v___x_229_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__2));
v___x_230_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_228_);
v___x_231_ = lean_string_append(v___x_229_, v___x_230_);
lean_dec_ref(v___x_230_);
v___x_232_ = l_IO_FS_Stream_putStrLn(v___x_192_, v___x_231_);
if (lean_obj_tag(v___x_232_) == 0)
{
lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_241_; 
v_isSharedCheck_241_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_241_ == 0)
{
lean_object* v_unused_242_; 
v_unused_242_ = lean_ctor_get(v___x_232_, 0);
lean_dec(v_unused_242_);
v___x_234_ = v___x_232_;
v_isShared_235_ = v_isSharedCheck_241_;
goto v_resetjp_233_;
}
else
{
lean_dec(v___x_232_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_241_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
uint32_t v___x_236_; lean_object* v___x_237_; lean_object* v___x_239_; 
v___x_236_ = lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(v_a_228_);
lean_dec(v_a_228_);
v___x_237_ = lean_box_uint32(v___x_236_);
if (v_isShared_235_ == 0)
{
lean_ctor_set(v___x_234_, 0, v___x_237_);
v___x_239_ = v___x_234_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v___x_237_);
v___x_239_ = v_reuseFailAlloc_240_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
return v___x_239_;
}
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
lean_dec(v_a_228_);
v_a_243_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_232_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_232_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_248_; 
if (v_isShared_246_ == 0)
{
v___x_248_ = v___x_245_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_a_243_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
else
{
lean_object* v_a_251_; lean_object* v___x_252_; 
v_a_251_ = lean_ctor_get(v___x_227_, 0);
lean_inc(v_a_251_);
lean_dec_ref_known(v___x_227_, 1);
v___x_252_ = lp_openvm_x2dfv_VmVerifier_Spec_Wire_readBaseline(v_fst_223_);
if (lean_obj_tag(v___x_252_) == 0)
{
lean_object* v_a_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
lean_dec(v_a_251_);
lean_dec(v_snd_226_);
lean_dec(v_fst_225_);
lean_dec(v_fst_224_);
v_a_253_ = lean_ctor_get(v___x_252_, 0);
lean_inc_n(v_a_253_, 2);
lean_dec_ref_known(v___x_252_, 1);
v___x_254_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__3));
v___x_255_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_253_);
v___x_256_ = lean_string_append(v___x_254_, v___x_255_);
lean_dec_ref(v___x_255_);
v___x_257_ = l_IO_FS_Stream_putStrLn(v___x_192_, v___x_256_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_266_; 
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_266_ == 0)
{
lean_object* v_unused_267_; 
v_unused_267_ = lean_ctor_get(v___x_257_, 0);
lean_dec(v_unused_267_);
v___x_259_ = v___x_257_;
v_isShared_260_ = v_isSharedCheck_266_;
goto v_resetjp_258_;
}
else
{
lean_dec(v___x_257_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_266_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
uint32_t v___x_261_; lean_object* v___x_262_; lean_object* v___x_264_; 
v___x_261_ = lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(v_a_253_);
lean_dec(v_a_253_);
v___x_262_ = lean_box_uint32(v___x_261_);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 0, v___x_262_);
v___x_264_ = v___x_259_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v___x_262_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
else
{
lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
lean_dec(v_a_253_);
v_a_268_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_275_ == 0)
{
v___x_270_ = v___x_257_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_dec(v___x_257_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_273_; 
if (v_isShared_271_ == 0)
{
v___x_273_ = v___x_270_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_268_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
}
else
{
lean_object* v_a_276_; lean_object* v___x_277_; 
v_a_276_ = lean_ctor_get(v___x_252_, 0);
lean_inc(v_a_276_);
lean_dec_ref_known(v___x_252_, 1);
v___x_277_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(v_fst_224_);
if (lean_obj_tag(v___x_277_) == 0)
{
lean_object* v_a_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; 
lean_dec(v_a_276_);
lean_dec(v_a_251_);
lean_dec(v_snd_226_);
lean_dec(v_fst_225_);
v_a_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc_n(v_a_278_, 2);
lean_dec_ref_known(v___x_277_, 1);
v___x_279_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__4));
v___x_280_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_278_);
v___x_281_ = lean_string_append(v___x_279_, v___x_280_);
lean_dec_ref(v___x_280_);
v___x_282_ = l_IO_FS_Stream_putStrLn(v___x_192_, v___x_281_);
if (lean_obj_tag(v___x_282_) == 0)
{
lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_291_; 
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_291_ == 0)
{
lean_object* v_unused_292_; 
v_unused_292_ = lean_ctor_get(v___x_282_, 0);
lean_dec(v_unused_292_);
v___x_284_ = v___x_282_;
v_isShared_285_ = v_isSharedCheck_291_;
goto v_resetjp_283_;
}
else
{
lean_dec(v___x_282_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_291_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
uint32_t v___x_286_; lean_object* v___x_287_; lean_object* v___x_289_; 
v___x_286_ = lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(v_a_278_);
lean_dec(v_a_278_);
v___x_287_ = lean_box_uint32(v___x_286_);
if (v_isShared_285_ == 0)
{
lean_ctor_set(v___x_284_, 0, v___x_287_);
v___x_289_ = v___x_284_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v___x_287_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
else
{
lean_object* v_a_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_300_; 
lean_dec(v_a_278_);
v_a_293_ = lean_ctor_get(v___x_282_, 0);
v_isSharedCheck_300_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_300_ == 0)
{
v___x_295_ = v___x_282_;
v_isShared_296_ = v_isSharedCheck_300_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_a_293_);
lean_dec(v___x_282_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_300_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v___x_298_; 
if (v_isShared_296_ == 0)
{
v___x_298_ = v___x_295_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_a_293_);
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
else
{
lean_object* v_a_301_; lean_object* v___x_302_; 
v_a_301_ = lean_ctor_get(v___x_277_, 0);
lean_inc(v_a_301_);
lean_dec_ref_known(v___x_277_, 1);
lean_inc(v_a_251_);
v___x_302_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(v_a_251_, v_fst_225_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v_a_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
lean_dec(v_a_301_);
lean_dec(v_a_276_);
lean_dec(v_a_251_);
lean_dec(v_snd_226_);
v_a_303_ = lean_ctor_get(v___x_302_, 0);
lean_inc_n(v_a_303_, 2);
lean_dec_ref_known(v___x_302_, 1);
v___x_304_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__5));
v___x_305_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_303_);
v___x_306_ = lean_string_append(v___x_304_, v___x_305_);
lean_dec_ref(v___x_305_);
v___x_307_ = l_IO_FS_Stream_putStrLn(v___x_192_, v___x_306_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v___x_309_; uint8_t v_isShared_310_; uint8_t v_isSharedCheck_316_; 
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_316_ == 0)
{
lean_object* v_unused_317_; 
v_unused_317_ = lean_ctor_get(v___x_307_, 0);
lean_dec(v_unused_317_);
v___x_309_ = v___x_307_;
v_isShared_310_ = v_isSharedCheck_316_;
goto v_resetjp_308_;
}
else
{
lean_dec(v___x_307_);
v___x_309_ = lean_box(0);
v_isShared_310_ = v_isSharedCheck_316_;
goto v_resetjp_308_;
}
v_resetjp_308_:
{
uint32_t v___x_311_; lean_object* v___x_312_; lean_object* v___x_314_; 
v___x_311_ = lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(v_a_303_);
lean_dec(v_a_303_);
v___x_312_ = lean_box_uint32(v___x_311_);
if (v_isShared_310_ == 0)
{
lean_ctor_set(v___x_309_, 0, v___x_312_);
v___x_314_ = v___x_309_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v___x_312_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
else
{
lean_object* v_a_318_; lean_object* v___x_320_; uint8_t v_isShared_321_; uint8_t v_isSharedCheck_325_; 
lean_dec(v_a_303_);
v_a_318_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_325_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_325_ == 0)
{
v___x_320_ = v___x_307_;
v_isShared_321_ = v_isSharedCheck_325_;
goto v_resetjp_319_;
}
else
{
lean_inc(v_a_318_);
lean_dec(v___x_307_);
v___x_320_ = lean_box(0);
v_isShared_321_ = v_isSharedCheck_325_;
goto v_resetjp_319_;
}
v_resetjp_319_:
{
lean_object* v___x_323_; 
if (v_isShared_321_ == 0)
{
v___x_323_ = v___x_320_;
goto v_reusejp_322_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v_a_318_);
v___x_323_ = v_reuseFailAlloc_324_;
goto v_reusejp_322_;
}
v_reusejp_322_:
{
return v___x_323_;
}
}
}
}
else
{
lean_object* v_a_326_; lean_object* v___x_327_; 
v_a_326_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_a_326_);
lean_dec_ref_known(v___x_302_, 1);
v___x_327_ = lp_openvm_x2dfv_VmVerifier_Spec_Wire_readUserPvsProof(v_snd_226_);
if (lean_obj_tag(v___x_327_) == 0)
{
lean_object* v_a_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
lean_dec(v_a_326_);
lean_dec(v_a_301_);
lean_dec(v_a_276_);
lean_dec(v_a_251_);
v_a_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc_n(v_a_328_, 2);
lean_dec_ref_known(v___x_327_, 1);
v___x_329_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__6));
v___x_330_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_328_);
v___x_331_ = lean_string_append(v___x_329_, v___x_330_);
lean_dec_ref(v___x_330_);
v___x_332_ = l_IO_FS_Stream_putStrLn(v___x_192_, v___x_331_);
if (lean_obj_tag(v___x_332_) == 0)
{
lean_object* v___x_334_; uint8_t v_isShared_335_; uint8_t v_isSharedCheck_341_; 
v_isSharedCheck_341_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_341_ == 0)
{
lean_object* v_unused_342_; 
v_unused_342_ = lean_ctor_get(v___x_332_, 0);
lean_dec(v_unused_342_);
v___x_334_ = v___x_332_;
v_isShared_335_ = v_isSharedCheck_341_;
goto v_resetjp_333_;
}
else
{
lean_dec(v___x_332_);
v___x_334_ = lean_box(0);
v_isShared_335_ = v_isSharedCheck_341_;
goto v_resetjp_333_;
}
v_resetjp_333_:
{
uint32_t v___x_336_; lean_object* v___x_337_; lean_object* v___x_339_; 
v___x_336_ = lp_openvm_x2dfv_VmVerifier_DumpProof_parseErrorExitCode(v_a_328_);
lean_dec(v_a_328_);
v___x_337_ = lean_box_uint32(v___x_336_);
if (v_isShared_335_ == 0)
{
lean_ctor_set(v___x_334_, 0, v___x_337_);
v___x_339_ = v___x_334_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_337_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
else
{
lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_350_; 
lean_dec(v_a_328_);
v_a_343_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_350_ == 0)
{
v___x_345_ = v___x_332_;
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_332_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_348_; 
if (v_isShared_346_ == 0)
{
v___x_348_ = v___x_345_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_a_343_);
v___x_348_ = v_reuseFailAlloc_349_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
return v___x_348_;
}
}
}
}
else
{
lean_object* v_a_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
lean_dec_ref(v___x_192_);
v_a_351_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_a_351_);
lean_dec_ref_known(v___x_327_, 1);
v___x_352_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__7));
v___x_353_ = lp_openvm_x2dfv_VmVerifier_DumpProof_vkDigest(v_a_251_);
v___x_354_ = lean_string_append(v___x_352_, v___x_353_);
lean_dec_ref(v___x_353_);
v___x_355_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_354_);
if (lean_obj_tag(v___x_355_) == 0)
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; 
lean_dec_ref_known(v___x_355_, 1);
v___x_356_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__8));
v___x_357_ = lp_openvm_x2dfv_VmVerifier_DumpProof_baselineDigest(v_a_276_);
v___x_358_ = lean_string_append(v___x_356_, v___x_357_);
lean_dec_ref(v___x_357_);
v___x_359_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_358_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
lean_dec_ref_known(v___x_359_, 1);
v___x_360_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__9));
v___x_361_ = lp_openvm_x2dfv_VmVerifier_DumpProof_proofDigest(v_a_301_);
v___x_362_ = lean_string_append(v___x_360_, v___x_361_);
lean_dec_ref(v___x_361_);
v___x_363_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_362_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
lean_dec_ref_known(v___x_363_, 1);
v___x_364_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__10));
v___x_365_ = lp_openvm_x2dfv_VmVerifier_DumpProof_publicValuesDigest(v_a_326_);
v___x_366_ = lean_string_append(v___x_364_, v___x_365_);
lean_dec_ref(v___x_365_);
v___x_367_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_366_);
if (lean_obj_tag(v___x_367_) == 0)
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
lean_dec_ref_known(v___x_367_, 1);
v___x_368_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_DumpProof_main___closed__11));
v___x_369_ = lp_openvm_x2dfv_VmVerifier_DumpProof_userPvsProofDigest(v_a_351_);
v___x_370_ = lean_string_append(v___x_368_, v___x_369_);
lean_dec_ref(v___x_369_);
v___x_371_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_370_);
if (lean_obj_tag(v___x_371_) == 0)
{
lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_379_; 
v_isSharedCheck_379_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_379_ == 0)
{
lean_object* v_unused_380_; 
v_unused_380_ = lean_ctor_get(v___x_371_, 0);
lean_dec(v_unused_380_);
v___x_373_ = v___x_371_;
v_isShared_374_ = v_isSharedCheck_379_;
goto v_resetjp_372_;
}
else
{
lean_dec(v___x_371_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_379_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; lean_object* v___x_377_; 
v___x_375_ = lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__2;
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_375_);
v___x_377_ = v___x_373_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v___x_375_);
v___x_377_ = v_reuseFailAlloc_378_;
goto v_reusejp_376_;
}
v_reusejp_376_:
{
return v___x_377_;
}
}
}
else
{
lean_object* v_a_381_; lean_object* v___x_383_; uint8_t v_isShared_384_; uint8_t v_isSharedCheck_388_; 
v_a_381_ = lean_ctor_get(v___x_371_, 0);
v_isSharedCheck_388_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_388_ == 0)
{
v___x_383_ = v___x_371_;
v_isShared_384_ = v_isSharedCheck_388_;
goto v_resetjp_382_;
}
else
{
lean_inc(v_a_381_);
lean_dec(v___x_371_);
v___x_383_ = lean_box(0);
v_isShared_384_ = v_isSharedCheck_388_;
goto v_resetjp_382_;
}
v_resetjp_382_:
{
lean_object* v___x_386_; 
if (v_isShared_384_ == 0)
{
v___x_386_ = v___x_383_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v_a_381_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
}
}
else
{
lean_object* v_a_389_; lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_396_; 
lean_dec(v_a_351_);
v_a_389_ = lean_ctor_get(v___x_367_, 0);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_367_);
if (v_isSharedCheck_396_ == 0)
{
v___x_391_ = v___x_367_;
v_isShared_392_ = v_isSharedCheck_396_;
goto v_resetjp_390_;
}
else
{
lean_inc(v_a_389_);
lean_dec(v___x_367_);
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
v_reuseFailAlloc_395_ = lean_alloc_ctor(1, 1, 0);
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
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
lean_dec(v_a_351_);
lean_dec(v_a_326_);
v_a_397_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_404_ == 0)
{
v___x_399_ = v___x_363_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_363_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
if (v_isShared_400_ == 0)
{
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_a_397_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
else
{
lean_object* v_a_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
lean_dec(v_a_351_);
lean_dec(v_a_326_);
lean_dec(v_a_301_);
v_a_405_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_359_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_359_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_405_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
else
{
lean_object* v_a_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_420_; 
lean_dec(v_a_351_);
lean_dec(v_a_326_);
lean_dec(v_a_301_);
lean_dec(v_a_276_);
v_a_413_ = lean_ctor_get(v___x_355_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_420_ == 0)
{
v___x_415_ = v___x_355_;
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_355_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_418_; 
if (v_isShared_416_ == 0)
{
v___x_418_ = v___x_415_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_a_413_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_428_; 
v_a_421_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_428_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_428_ == 0)
{
v___x_423_ = v___x_190_;
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_a_421_);
lean_dec(v___x_190_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v___x_426_; 
if (v_isShared_424_ == 0)
{
v___x_426_ = v___x_423_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v_a_421_);
v___x_426_ = v_reuseFailAlloc_427_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
return v___x_426_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed(lean_object* v_a_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_openvm_x2dfv_VmVerifier_DumpProof_main();
return v_res_430_;
}
}
LEAN_EXPORT lean_object* _lean_main(){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lp_openvm_x2dfv_VmVerifier_DumpProof_main();
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_main___boxed(lean_object* v_a_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = _lean_main();
return v_res_434_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VmVerifier_Spec_Wire(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VmVerifier_DumpProof(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VmVerifier_Spec_Wire(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__1 = _init_lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__1();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__1);
lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__2 = _init_lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__2();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_DumpProof_main___boxed__const__2);
return lean_io_result_mk_ok(lean_box(0));
}
char ** lean_setup_args(int argc, char ** argv);
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#endif
lean_object* run_main(int argc, char ** argv) {
    return _lean_main();
}
int main(int argc, char ** argv) {
#if defined(WIN32) || defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS);
  SetConsoleOutputCP(CP_UTF8);
#endif
  lean_object* res;
  argv = lean_setup_args(argc, argv);
  res = initialize_openvm_x2dfv_VmVerifier_DumpProof(1 /* builtin */);
  lean_io_mark_end_initialization();
  if (lean_io_result_is_ok(res)) {
    lean_dec_ref(res);
    lean_init_task_manager();
    res = lean_run_main(&run_main, argc, argv);
  }
  lean_finalize_task_manager();
  if (lean_io_result_is_ok(res)) {
    int ret = lean_unbox_uint32(lean_io_result_get_value(res));
    lean_dec_ref(res);
    return ret;
  } else {
    lean_io_result_show_error(res);
    lean_dec_ref(res);
    return 1;
  }
}
#ifdef __cplusplus
}
#endif
