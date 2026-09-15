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
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lean_get_stdout();
lean_object* lean_get_stdin();
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
lean_object* lean_get_stderr();
lean_object* lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
lean_object* lean_byte_array_size(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(lean_object*);
lean_object* lp_workspace_VmVerifier_Spec_Wire_readBaseline(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(lean_object*, lean_object*);
lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_rawDigestToCsv(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_rawFieldsToCsv(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_vkDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_proofDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_workspace_VmVerifier_DumpProof_publicValuesDigest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_workspace_VmVerifier_DumpProof_publicValuesDigest___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_publicValuesDigest___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_publicValuesDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_digestToCsv(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_vkCommitDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_baselineDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_workspace_VmVerifier_DumpProof_userPvsProofDigest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_workspace_VmVerifier_DumpProof_userPvsProofDigest___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_userPvsProofDigest___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_userPvsProofDigest(lean_object*);
LEAN_EXPORT uint32_t lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_parseErrorExitCode___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "vm_dump_proof: stdin framing error (received "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__0_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " bytes)"};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__1 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__1_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "vm_dump_proof: vk parse error: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__2 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__2_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "vm_dump_proof: baseline parse error: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__3 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__3_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "vm_dump_proof: proof parse error: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__4 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__4_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "vm_dump_proof: public-values parse error: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__5 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__5_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "vm_dump_proof: user-PV proof parse error: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__6 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__6_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "vk: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__7 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__7_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "baseline: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__8 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__8_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "proof: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__9 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__9_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "pv: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__10 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__10_value;
static const lean_string_object lp_workspace_VmVerifier_DumpProof_main___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "user-pvs: "};
static const lean_object* lp_workspace_VmVerifier_DumpProof_main___closed__11 = (const lean_object*)&lp_workspace_VmVerifier_DumpProof_main___closed__11_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_main___boxed__const__1;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_main___boxed__const__2;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_main();
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_main___boxed(lean_object*);
LEAN_EXPORT lean_object* _lean_main();
LEAN_EXPORT lean_object* lp_workspace_main___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(lean_object* v_a_1_, lean_object* v_a_2_){
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_rawDigestToCsv(lean_object* v_digest_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_19_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_20_ = lean_array_to_list(v_digest_18_);
v___x_21_ = lean_box(0);
v___x_22_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(v___x_20_, v___x_21_);
v___x_23_ = l_String_intercalate(v___x_19_, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_rawFieldsToCsv(lean_object* v_values_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_25_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_26_ = lean_array_to_list(v_values_24_);
v___x_27_ = lean_box(0);
v___x_28_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(v___x_26_, v___x_27_);
v___x_29_ = l_String_intercalate(v___x_25_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_vkDigest(lean_object* v_vk_30_){
_start:
{
lean_object* v_preHash_31_; lean_object* v___x_32_; 
v_preHash_31_ = lean_ctor_get(v_vk_30_, 1);
lean_inc_ref(v_preHash_31_);
lean_dec_ref(v_vk_30_);
v___x_32_ = lp_workspace_VmVerifier_DumpProof_rawDigestToCsv(v_preHash_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_proofDigest(lean_object* v_proof_33_){
_start:
{
lean_object* v_commonMainCommit_34_; lean_object* v___x_35_; 
v_commonMainCommit_34_ = lean_ctor_get(v_proof_33_, 0);
lean_inc_ref(v_commonMainCommit_34_);
lean_dec_ref(v_proof_33_);
v___x_35_ = lp_workspace_VmVerifier_DumpProof_rawDigestToCsv(v_commonMainCommit_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(lean_object* v_a_36_, lean_object* v_a_37_){
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
v___x_44_ = lp_workspace_VmVerifier_DumpProof_rawFieldsToCsv(v_head_39_);
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_publicValuesDigest(lean_object* v_publicValues_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_52_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_publicValuesDigest___closed__0));
v___x_53_ = lean_array_to_list(v_publicValues_51_);
v___x_54_ = lean_box(0);
v___x_55_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(v___x_53_, v___x_54_);
v___x_56_ = l_String_intercalate(v___x_52_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(lean_object* v_a_57_, lean_object* v_a_58_){
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_digestToCsv(lean_object* v_digest_71_){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_72_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_73_ = lean_array_to_list(v_digest_71_);
v___x_74_ = lean_box(0);
v___x_75_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(v___x_73_, v___x_74_);
v___x_76_ = l_String_intercalate(v___x_72_, v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_vkCommitDigest(lean_object* v_commit_77_){
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
v___x_83_ = lp_workspace_VmVerifier_DumpProof_digestToCsv(v_cachedCommit_78_);
v___x_84_ = lp_workspace_VmVerifier_DumpProof_digestToCsv(v_vkPreHash_79_);
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_baselineDigest(lean_object* v_baseline_91_){
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
v___x_106_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_publicValuesDigest___closed__0));
v___x_107_ = lp_workspace_VmVerifier_DumpProof_digestToCsv(v_programCommit_93_);
v___x_108_ = lp_workspace_VmVerifier_DumpProof_digestToCsv(v_initialState_94_);
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
v___x_121_ = lp_workspace_VmVerifier_DumpProof_vkCommitDigest(v_appVkCommit_97_);
v___x_122_ = l_List_appendTR___redArg(v___x_120_, v___x_121_);
v___x_123_ = lp_workspace_VmVerifier_DumpProof_vkCommitDigest(v_leafVkCommit_98_);
v___x_124_ = l_List_appendTR___redArg(v___x_122_, v___x_123_);
v___x_125_ = lp_workspace_VmVerifier_DumpProof_vkCommitDigest(v_internalForLeafVkCommit_99_);
v___x_126_ = l_List_appendTR___redArg(v___x_124_, v___x_125_);
v___x_127_ = lp_workspace_VmVerifier_DumpProof_vkCommitDigest(v_internalRecursiveVkCommit_100_);
v___x_128_ = l_List_appendTR___redArg(v___x_126_, v___x_127_);
v___x_129_ = l_String_intercalate(v___x_106_, v___x_128_);
return v___x_129_;
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(lean_object* v_a_132_, lean_object* v_a_133_){
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
v___x_140_ = lp_workspace_VmVerifier_DumpProof_digestToCsv(v_head_135_);
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_userPvsProofDigest(lean_object* v_proof_147_){
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
v___x_151_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_userPvsProofDigest___closed__0));
v___x_152_ = lean_box(0);
v___x_153_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(v_authenticationPath_148_, v___x_152_);
v_authenticationPath_154_ = l_String_intercalate(v___x_151_, v___x_153_);
v___x_155_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_rawDigestToCsv___closed__0));
v___x_156_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(v_publicValues_149_, v___x_152_);
v_publicValues_157_ = l_String_intercalate(v___x_155_, v___x_156_);
v___x_158_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_publicValuesDigest___closed__0));
v___x_159_ = lp_workspace_VmVerifier_DumpProof_digestToCsv(v_publicValuesCommit_150_);
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
LEAN_EXPORT uint32_t lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(lean_object* v_x_164_){
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_parseErrorExitCode___boxed(lean_object* v_x_169_){
_start:
{
uint32_t v_res_170_; lean_object* v_r_171_; 
v_res_170_ = lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(v_x_169_);
lean_dec_ref(v_x_169_);
v_r_171_ = lean_box_uint32(v_res_170_);
return v_r_171_;
}
}
LEAN_EXPORT lean_object* lp_workspace_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(lean_object* v_s_172_){
_start:
{
lean_object* v___x_174_; lean_object* v_putStr_175_; lean_object* v___x_176_; 
v___x_174_ = lean_get_stdout();
v_putStr_175_ = lean_ctor_get(v___x_174_, 4);
lean_inc_ref(v_putStr_175_);
lean_dec_ref(v___x_174_);
v___x_176_ = lean_apply_2(v_putStr_175_, v_s_172_, lean_box(0));
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_workspace_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0___boxed(lean_object* v_s_177_, lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_workspace_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(v_s_177_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(lean_object* v_s_180_){
_start:
{
uint32_t v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = 10;
v___x_183_ = lean_string_push(v_s_180_, v___x_182_);
v___x_184_ = lp_workspace_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(v___x_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0___boxed(lean_object* v_s_185_, lean_object* v_a_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(v_s_185_);
return v_res_187_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_DumpProof_main___boxed__const__1(void){
_start:
{
uint32_t v___x_200_; lean_object* v___x_201_; 
v___x_200_ = 20;
v___x_201_ = lean_box_uint32(v___x_200_);
return v___x_201_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_DumpProof_main___boxed__const__2(void){
_start:
{
uint32_t v___x_202_; lean_object* v___x_203_; 
v___x_202_ = 0;
v___x_203_ = lean_box_uint32(v___x_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_main(){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = lean_get_stdin();
v___x_206_ = l_IO_FS_Stream_readBinToEnd(v___x_205_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
lean_inc(v_a_207_);
lean_dec_ref_known(v___x_206_, 1);
v___x_208_ = lean_get_stderr();
v___x_209_ = lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs(v_a_207_);
if (lean_obj_tag(v___x_209_) == 0)
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_210_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__0));
v___x_211_ = lean_byte_array_size(v_a_207_);
lean_dec(v_a_207_);
v___x_212_ = l_Nat_reprFast(v___x_211_);
v___x_213_ = lean_string_append(v___x_210_, v___x_212_);
lean_dec_ref(v___x_212_);
v___x_214_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__1));
v___x_215_ = lean_string_append(v___x_213_, v___x_214_);
v___x_216_ = l_IO_FS_Stream_putStrLn(v___x_208_, v___x_215_);
if (lean_obj_tag(v___x_216_) == 0)
{
lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_224_; 
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_216_);
if (v_isSharedCheck_224_ == 0)
{
lean_object* v_unused_225_; 
v_unused_225_ = lean_ctor_get(v___x_216_, 0);
lean_dec(v_unused_225_);
v___x_218_ = v___x_216_;
v_isShared_219_ = v_isSharedCheck_224_;
goto v_resetjp_217_;
}
else
{
lean_dec(v___x_216_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_224_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_220_; lean_object* v___x_222_; 
v___x_220_ = lp_workspace_VmVerifier_DumpProof_main___boxed__const__1;
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_220_);
v___x_222_ = v___x_218_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_220_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
else
{
lean_object* v_a_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_233_; 
v_a_226_ = lean_ctor_get(v___x_216_, 0);
v_isSharedCheck_233_ = !lean_is_exclusive(v___x_216_);
if (v_isSharedCheck_233_ == 0)
{
v___x_228_ = v___x_216_;
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_a_226_);
lean_dec(v___x_216_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_231_; 
if (v_isShared_229_ == 0)
{
v___x_231_ = v___x_228_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_a_226_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
}
else
{
lean_object* v_val_234_; lean_object* v_snd_235_; lean_object* v_snd_236_; lean_object* v_snd_237_; lean_object* v_fst_238_; lean_object* v_fst_239_; lean_object* v_fst_240_; lean_object* v_fst_241_; lean_object* v_snd_242_; lean_object* v___x_243_; 
lean_dec(v_a_207_);
v_val_234_ = lean_ctor_get(v___x_209_, 0);
lean_inc(v_val_234_);
lean_dec_ref_known(v___x_209_, 1);
v_snd_235_ = lean_ctor_get(v_val_234_, 1);
lean_inc(v_snd_235_);
v_snd_236_ = lean_ctor_get(v_snd_235_, 1);
lean_inc(v_snd_236_);
v_snd_237_ = lean_ctor_get(v_snd_236_, 1);
lean_inc(v_snd_237_);
v_fst_238_ = lean_ctor_get(v_val_234_, 0);
lean_inc(v_fst_238_);
lean_dec(v_val_234_);
v_fst_239_ = lean_ctor_get(v_snd_235_, 0);
lean_inc(v_fst_239_);
lean_dec(v_snd_235_);
v_fst_240_ = lean_ctor_get(v_snd_236_, 0);
lean_inc(v_fst_240_);
lean_dec(v_snd_236_);
v_fst_241_ = lean_ctor_get(v_snd_237_, 0);
lean_inc(v_fst_241_);
v_snd_242_ = lean_ctor_get(v_snd_237_, 1);
lean_inc(v_snd_242_);
lean_dec(v_snd_237_);
v___x_243_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(v_fst_238_);
if (lean_obj_tag(v___x_243_) == 0)
{
lean_object* v_a_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
lean_dec(v_snd_242_);
lean_dec(v_fst_241_);
lean_dec(v_fst_240_);
lean_dec(v_fst_239_);
v_a_244_ = lean_ctor_get(v___x_243_, 0);
lean_inc_n(v_a_244_, 2);
lean_dec_ref_known(v___x_243_, 1);
v___x_245_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__2));
v___x_246_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_244_);
v___x_247_ = lean_string_append(v___x_245_, v___x_246_);
lean_dec_ref(v___x_246_);
v___x_248_ = l_IO_FS_Stream_putStrLn(v___x_208_, v___x_247_);
if (lean_obj_tag(v___x_248_) == 0)
{
lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_257_; 
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_248_);
if (v_isSharedCheck_257_ == 0)
{
lean_object* v_unused_258_; 
v_unused_258_ = lean_ctor_get(v___x_248_, 0);
lean_dec(v_unused_258_);
v___x_250_ = v___x_248_;
v_isShared_251_ = v_isSharedCheck_257_;
goto v_resetjp_249_;
}
else
{
lean_dec(v___x_248_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_257_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
uint32_t v___x_252_; lean_object* v___x_253_; lean_object* v___x_255_; 
v___x_252_ = lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(v_a_244_);
lean_dec(v_a_244_);
v___x_253_ = lean_box_uint32(v___x_252_);
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 0, v___x_253_);
v___x_255_ = v___x_250_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v___x_253_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
else
{
lean_object* v_a_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_266_; 
lean_dec(v_a_244_);
v_a_259_ = lean_ctor_get(v___x_248_, 0);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_248_);
if (v_isSharedCheck_266_ == 0)
{
v___x_261_ = v___x_248_;
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_a_259_);
lean_dec(v___x_248_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_264_; 
if (v_isShared_262_ == 0)
{
v___x_264_ = v___x_261_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v_a_259_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
}
else
{
lean_object* v_a_267_; lean_object* v___x_268_; 
v_a_267_ = lean_ctor_get(v___x_243_, 0);
lean_inc(v_a_267_);
lean_dec_ref_known(v___x_243_, 1);
v___x_268_ = lp_workspace_VmVerifier_Spec_Wire_readBaseline(v_fst_239_);
if (lean_obj_tag(v___x_268_) == 0)
{
lean_object* v_a_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
lean_dec(v_a_267_);
lean_dec(v_snd_242_);
lean_dec(v_fst_241_);
lean_dec(v_fst_240_);
v_a_269_ = lean_ctor_get(v___x_268_, 0);
lean_inc_n(v_a_269_, 2);
lean_dec_ref_known(v___x_268_, 1);
v___x_270_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__3));
v___x_271_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_269_);
v___x_272_ = lean_string_append(v___x_270_, v___x_271_);
lean_dec_ref(v___x_271_);
v___x_273_ = l_IO_FS_Stream_putStrLn(v___x_208_, v___x_272_);
if (lean_obj_tag(v___x_273_) == 0)
{
lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_282_; 
v_isSharedCheck_282_ = !lean_is_exclusive(v___x_273_);
if (v_isSharedCheck_282_ == 0)
{
lean_object* v_unused_283_; 
v_unused_283_ = lean_ctor_get(v___x_273_, 0);
lean_dec(v_unused_283_);
v___x_275_ = v___x_273_;
v_isShared_276_ = v_isSharedCheck_282_;
goto v_resetjp_274_;
}
else
{
lean_dec(v___x_273_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_282_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
uint32_t v___x_277_; lean_object* v___x_278_; lean_object* v___x_280_; 
v___x_277_ = lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(v_a_269_);
lean_dec(v_a_269_);
v___x_278_ = lean_box_uint32(v___x_277_);
if (v_isShared_276_ == 0)
{
lean_ctor_set(v___x_275_, 0, v___x_278_);
v___x_280_ = v___x_275_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v___x_278_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
return v___x_280_;
}
}
}
else
{
lean_object* v_a_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
lean_dec(v_a_269_);
v_a_284_ = lean_ctor_get(v___x_273_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_273_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_273_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_a_284_);
lean_dec(v___x_273_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
else
{
lean_object* v_a_292_; lean_object* v___x_293_; 
v_a_292_ = lean_ctor_get(v___x_268_, 0);
lean_inc(v_a_292_);
lean_dec_ref_known(v___x_268_, 1);
v___x_293_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(v_fst_240_);
if (lean_obj_tag(v___x_293_) == 0)
{
lean_object* v_a_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
lean_dec(v_a_292_);
lean_dec(v_a_267_);
lean_dec(v_snd_242_);
lean_dec(v_fst_241_);
v_a_294_ = lean_ctor_get(v___x_293_, 0);
lean_inc_n(v_a_294_, 2);
lean_dec_ref_known(v___x_293_, 1);
v___x_295_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__4));
v___x_296_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_294_);
v___x_297_ = lean_string_append(v___x_295_, v___x_296_);
lean_dec_ref(v___x_296_);
v___x_298_ = l_IO_FS_Stream_putStrLn(v___x_208_, v___x_297_);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_307_; 
v_isSharedCheck_307_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_307_ == 0)
{
lean_object* v_unused_308_; 
v_unused_308_ = lean_ctor_get(v___x_298_, 0);
lean_dec(v_unused_308_);
v___x_300_ = v___x_298_;
v_isShared_301_ = v_isSharedCheck_307_;
goto v_resetjp_299_;
}
else
{
lean_dec(v___x_298_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_307_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
uint32_t v___x_302_; lean_object* v___x_303_; lean_object* v___x_305_; 
v___x_302_ = lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(v_a_294_);
lean_dec(v_a_294_);
v___x_303_ = lean_box_uint32(v___x_302_);
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v___x_303_);
v___x_305_ = v___x_300_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v___x_303_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
}
else
{
lean_object* v_a_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_316_; 
lean_dec(v_a_294_);
v_a_309_ = lean_ctor_get(v___x_298_, 0);
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_316_ == 0)
{
v___x_311_ = v___x_298_;
v_isShared_312_ = v_isSharedCheck_316_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_a_309_);
lean_dec(v___x_298_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_316_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_314_; 
if (v_isShared_312_ == 0)
{
v___x_314_ = v___x_311_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v_a_309_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
}
else
{
lean_object* v_a_317_; lean_object* v___x_318_; 
v_a_317_ = lean_ctor_get(v___x_293_, 0);
lean_inc(v_a_317_);
lean_dec_ref_known(v___x_293_, 1);
lean_inc(v_a_267_);
v___x_318_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(v_a_267_, v_fst_241_);
if (lean_obj_tag(v___x_318_) == 0)
{
lean_object* v_a_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
lean_dec(v_a_317_);
lean_dec(v_a_292_);
lean_dec(v_a_267_);
lean_dec(v_snd_242_);
v_a_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc_n(v_a_319_, 2);
lean_dec_ref_known(v___x_318_, 1);
v___x_320_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__5));
v___x_321_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_319_);
v___x_322_ = lean_string_append(v___x_320_, v___x_321_);
lean_dec_ref(v___x_321_);
v___x_323_ = l_IO_FS_Stream_putStrLn(v___x_208_, v___x_322_);
if (lean_obj_tag(v___x_323_) == 0)
{
lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_332_; 
v_isSharedCheck_332_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_332_ == 0)
{
lean_object* v_unused_333_; 
v_unused_333_ = lean_ctor_get(v___x_323_, 0);
lean_dec(v_unused_333_);
v___x_325_ = v___x_323_;
v_isShared_326_ = v_isSharedCheck_332_;
goto v_resetjp_324_;
}
else
{
lean_dec(v___x_323_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_332_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
uint32_t v___x_327_; lean_object* v___x_328_; lean_object* v___x_330_; 
v___x_327_ = lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(v_a_319_);
lean_dec(v_a_319_);
v___x_328_ = lean_box_uint32(v___x_327_);
if (v_isShared_326_ == 0)
{
lean_ctor_set(v___x_325_, 0, v___x_328_);
v___x_330_ = v___x_325_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___x_328_);
v___x_330_ = v_reuseFailAlloc_331_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
return v___x_330_;
}
}
}
else
{
lean_object* v_a_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_341_; 
lean_dec(v_a_319_);
v_a_334_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_341_ == 0)
{
v___x_336_ = v___x_323_;
v_isShared_337_ = v_isSharedCheck_341_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_a_334_);
lean_dec(v___x_323_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_341_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v___x_339_; 
if (v_isShared_337_ == 0)
{
v___x_339_ = v___x_336_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v_a_334_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
}
else
{
lean_object* v_a_342_; lean_object* v___x_343_; 
v_a_342_ = lean_ctor_get(v___x_318_, 0);
lean_inc(v_a_342_);
lean_dec_ref_known(v___x_318_, 1);
v___x_343_ = lp_workspace_VmVerifier_Spec_Wire_readUserPvsProof(v_snd_242_);
if (lean_obj_tag(v___x_343_) == 0)
{
lean_object* v_a_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec(v_a_342_);
lean_dec(v_a_317_);
lean_dec(v_a_292_);
lean_dec(v_a_267_);
v_a_344_ = lean_ctor_get(v___x_343_, 0);
lean_inc_n(v_a_344_, 2);
lean_dec_ref_known(v___x_343_, 1);
v___x_345_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__6));
v___x_346_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(v_a_344_);
v___x_347_ = lean_string_append(v___x_345_, v___x_346_);
lean_dec_ref(v___x_346_);
v___x_348_ = l_IO_FS_Stream_putStrLn(v___x_208_, v___x_347_);
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v___x_350_; uint8_t v_isShared_351_; uint8_t v_isSharedCheck_357_; 
v_isSharedCheck_357_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_357_ == 0)
{
lean_object* v_unused_358_; 
v_unused_358_ = lean_ctor_get(v___x_348_, 0);
lean_dec(v_unused_358_);
v___x_350_ = v___x_348_;
v_isShared_351_ = v_isSharedCheck_357_;
goto v_resetjp_349_;
}
else
{
lean_dec(v___x_348_);
v___x_350_ = lean_box(0);
v_isShared_351_ = v_isSharedCheck_357_;
goto v_resetjp_349_;
}
v_resetjp_349_:
{
uint32_t v___x_352_; lean_object* v___x_353_; lean_object* v___x_355_; 
v___x_352_ = lp_workspace_VmVerifier_DumpProof_parseErrorExitCode(v_a_344_);
lean_dec(v_a_344_);
v___x_353_ = lean_box_uint32(v___x_352_);
if (v_isShared_351_ == 0)
{
lean_ctor_set(v___x_350_, 0, v___x_353_);
v___x_355_ = v___x_350_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_353_);
v___x_355_ = v_reuseFailAlloc_356_;
goto v_reusejp_354_;
}
v_reusejp_354_:
{
return v___x_355_;
}
}
}
else
{
lean_object* v_a_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_366_; 
lean_dec(v_a_344_);
v_a_359_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_366_ == 0)
{
v___x_361_ = v___x_348_;
v_isShared_362_ = v_isSharedCheck_366_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_a_359_);
lean_dec(v___x_348_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_366_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v___x_364_; 
if (v_isShared_362_ == 0)
{
v___x_364_ = v___x_361_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v_a_359_);
v___x_364_ = v_reuseFailAlloc_365_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
return v___x_364_;
}
}
}
}
else
{
lean_object* v_a_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
lean_dec_ref(v___x_208_);
v_a_367_ = lean_ctor_get(v___x_343_, 0);
lean_inc(v_a_367_);
lean_dec_ref_known(v___x_343_, 1);
v___x_368_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__7));
v___x_369_ = lp_workspace_VmVerifier_DumpProof_vkDigest(v_a_267_);
v___x_370_ = lean_string_append(v___x_368_, v___x_369_);
lean_dec_ref(v___x_369_);
v___x_371_ = lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(v___x_370_);
if (lean_obj_tag(v___x_371_) == 0)
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
lean_dec_ref_known(v___x_371_, 1);
v___x_372_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__8));
v___x_373_ = lp_workspace_VmVerifier_DumpProof_baselineDigest(v_a_292_);
v___x_374_ = lean_string_append(v___x_372_, v___x_373_);
lean_dec_ref(v___x_373_);
v___x_375_ = lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(v___x_374_);
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
lean_dec_ref_known(v___x_375_, 1);
v___x_376_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__9));
v___x_377_ = lp_workspace_VmVerifier_DumpProof_proofDigest(v_a_317_);
v___x_378_ = lean_string_append(v___x_376_, v___x_377_);
lean_dec_ref(v___x_377_);
v___x_379_ = lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(v___x_378_);
if (lean_obj_tag(v___x_379_) == 0)
{
lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; 
lean_dec_ref_known(v___x_379_, 1);
v___x_380_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__10));
v___x_381_ = lp_workspace_VmVerifier_DumpProof_publicValuesDigest(v_a_342_);
v___x_382_ = lean_string_append(v___x_380_, v___x_381_);
lean_dec_ref(v___x_381_);
v___x_383_ = lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(v___x_382_);
if (lean_obj_tag(v___x_383_) == 0)
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
lean_dec_ref_known(v___x_383_, 1);
v___x_384_ = ((lean_object*)(lp_workspace_VmVerifier_DumpProof_main___closed__11));
v___x_385_ = lp_workspace_VmVerifier_DumpProof_userPvsProofDigest(v_a_367_);
v___x_386_ = lean_string_append(v___x_384_, v___x_385_);
lean_dec_ref(v___x_385_);
v___x_387_ = lp_workspace_IO_println___at___00VmVerifier_DumpProof_main_spec__0(v___x_386_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_395_; 
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_395_ == 0)
{
lean_object* v_unused_396_; 
v_unused_396_ = lean_ctor_get(v___x_387_, 0);
lean_dec(v_unused_396_);
v___x_389_ = v___x_387_;
v_isShared_390_ = v_isSharedCheck_395_;
goto v_resetjp_388_;
}
else
{
lean_dec(v___x_387_);
v___x_389_ = lean_box(0);
v_isShared_390_ = v_isSharedCheck_395_;
goto v_resetjp_388_;
}
v_resetjp_388_:
{
lean_object* v___x_391_; lean_object* v___x_393_; 
v___x_391_ = lp_workspace_VmVerifier_DumpProof_main___boxed__const__2;
if (v_isShared_390_ == 0)
{
lean_ctor_set(v___x_389_, 0, v___x_391_);
v___x_393_ = v___x_389_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v___x_391_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
v_a_397_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_404_ == 0)
{
v___x_399_ = v___x_387_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_387_);
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
lean_dec(v_a_367_);
v_a_405_ = lean_ctor_get(v___x_383_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_383_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_383_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_383_);
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
lean_dec(v_a_367_);
lean_dec(v_a_342_);
v_a_413_ = lean_ctor_get(v___x_379_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_420_ == 0)
{
v___x_415_ = v___x_379_;
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_379_);
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
else
{
lean_object* v_a_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_428_; 
lean_dec(v_a_367_);
lean_dec(v_a_342_);
lean_dec(v_a_317_);
v_a_421_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_428_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_428_ == 0)
{
v___x_423_ = v___x_375_;
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_a_421_);
lean_dec(v___x_375_);
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
else
{
lean_object* v_a_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_436_; 
lean_dec(v_a_367_);
lean_dec(v_a_342_);
lean_dec(v_a_317_);
lean_dec(v_a_292_);
v_a_429_ = lean_ctor_get(v___x_371_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_436_ == 0)
{
v___x_431_ = v___x_371_;
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_a_429_);
lean_dec(v___x_371_);
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
v_reuseFailAlloc_435_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
}
}
}
}
else
{
lean_object* v_a_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_444_; 
v_a_437_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_444_ == 0)
{
v___x_439_ = v___x_206_;
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_a_437_);
lean_dec(v___x_206_);
v___x_439_ = lean_box(0);
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
v_resetjp_438_:
{
lean_object* v___x_442_; 
if (v_isShared_440_ == 0)
{
v___x_442_ = v___x_439_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_a_437_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_DumpProof_main___boxed(lean_object* v_a_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_workspace_VmVerifier_DumpProof_main();
return v_res_446_;
}
}
LEAN_EXPORT lean_object* _lean_main(){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lp_workspace_VmVerifier_DumpProof_main();
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_workspace_main___boxed(lean_object* v_a_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = _lean_main();
return v_res_450_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_workspace_VmVerifier_Spec_Wire(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_workspace_VmVerifier_DumpProof(uint8_t builtin) {
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
res = initialize_workspace_VmVerifier_Spec_Wire(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_workspace_VmVerifier_DumpProof_main___boxed__const__1 = _init_lp_workspace_VmVerifier_DumpProof_main___boxed__const__1();
lean_mark_persistent(lp_workspace_VmVerifier_DumpProof_main___boxed__const__1);
lp_workspace_VmVerifier_DumpProof_main___boxed__const__2 = _init_lp_workspace_VmVerifier_DumpProof_main___boxed__const__2();
lean_mark_persistent(lp_workspace_VmVerifier_DumpProof_main___boxed__const__2);
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
  res = initialize_workspace_VmVerifier_DumpProof(1 /* builtin */);
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
