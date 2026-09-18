// Lean compiler output
// Module: VmVerifier.Spec.Wire
// Imports: public import Init public meta import Init public import Swirl.Spec.ReferenceVerifier.Wire.RawToTyped public import VmVerifier.Spec.Types
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
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
uint32_t lean_uint8_to_uint32(uint8_t);
uint32_t lean_uint32_shift_left(uint32_t, uint32_t);
uint32_t lean_uint32_lor(uint32_t, uint32_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_byte_array_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
uint8_t lean_byte_array_fget(lean_object*, lean_object*);
lean_object* l_ByteArray_extract(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(uint8_t, uint8_t, uint8_t, uint8_t);
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
static lean_once_cell_t lp_workspace_VmVerifier_Spec_Wire_baselineMagic___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_Spec_Wire_baselineMagic___closed__0;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_baselineMagic;
static lean_once_cell_t lp_workspace_VmVerifier_Spec_Wire_userPvsMagic___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_Spec_Wire_userPvsMagic___closed__0;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_userPvsMagic;
static const lean_string_object lp_workspace_VmVerifier_Spec_Wire_ensureEnd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "trailing bytes: "};
static const lean_object* lp_workspace_VmVerifier_Spec_Wire_ensureEnd___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_Spec_Wire_ensureEnd___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_ensureEnd(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readDigest___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readDigest___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readVkCommit(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBaselineM(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBaseline(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___lam__0(lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readU32LE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readU32LE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_sliceBytes(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_sliceBytes___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBlobAt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBlobAt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs___boxed(lean_object*);
static lean_object* _init_lp_workspace_VmVerifier_Spec_Wire_baselineMagic___closed__0(void){
_start:
{
uint8_t v___x_1_; uint8_t v___x_2_; uint8_t v___x_3_; uint8_t v___x_4_; lean_object* v___x_5_; 
v___x_1_ = 76;
v___x_2_ = 66;
v___x_3_ = 77;
v___x_4_ = 86;
v___x_5_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(v___x_4_, v___x_3_, v___x_2_, v___x_1_);
return v___x_5_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_Spec_Wire_baselineMagic(void){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_obj_once(&lp_workspace_VmVerifier_Spec_Wire_baselineMagic___closed__0, &lp_workspace_VmVerifier_Spec_Wire_baselineMagic___closed__0_once, _init_lp_workspace_VmVerifier_Spec_Wire_baselineMagic___closed__0);
return v___x_6_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_Spec_Wire_userPvsMagic___closed__0(void){
_start:
{
uint8_t v___x_7_; uint8_t v___x_8_; uint8_t v___x_9_; uint8_t v___x_10_; lean_object* v___x_11_; 
v___x_7_ = 83;
v___x_8_ = 86;
v___x_9_ = 80;
v___x_10_ = 85;
v___x_11_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(v___x_10_, v___x_9_, v___x_8_, v___x_7_);
return v___x_11_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_Spec_Wire_userPvsMagic(void){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_obj_once(&lp_workspace_VmVerifier_Spec_Wire_userPvsMagic___closed__0, &lp_workspace_VmVerifier_Spec_Wire_userPvsMagic___closed__0_once, _init_lp_workspace_VmVerifier_Spec_Wire_userPvsMagic___closed__0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_ensureEnd(lean_object* v_a_14_){
_start:
{
lean_object* v_data_15_; lean_object* v_offset_16_; lean_object* v___x_17_; uint8_t v___x_18_; 
v_data_15_ = lean_ctor_get(v_a_14_, 0);
v_offset_16_ = lean_ctor_get(v_a_14_, 1);
v___x_17_ = lean_byte_array_size(v_data_15_);
v___x_18_ = lean_nat_dec_eq(v_offset_16_, v___x_17_);
if (v___x_18_ == 0)
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_19_ = ((lean_object*)(lp_workspace_VmVerifier_Spec_Wire_ensureEnd___closed__0));
v___x_20_ = lean_nat_sub(v___x_17_, v_offset_16_);
v___x_21_ = l_Nat_reprFast(v___x_20_);
v___x_22_ = lean_string_append(v___x_19_, v___x_21_);
lean_dec_ref(v___x_21_);
lean_inc(v_offset_16_);
v___x_23_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_23_, 0, v_offset_16_);
lean_ctor_set(v___x_23_, 1, v___x_22_);
v___x_24_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
lean_ctor_set(v___x_24_, 1, v_a_14_);
return v___x_24_;
}
else
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = lean_box(0);
v___x_26_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_26_, 0, v___x_25_);
lean_ctor_set(v___x_26_, 1, v_a_14_);
return v___x_26_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readDigest___lam__0(lean_object* v_a_27_, lean_object* v_idx_28_){
_start:
{
lean_object* v___x_29_; uint32_t v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_29_ = lean_array_fget_borrowed(v_a_27_, v_idx_28_);
v___x_30_ = lean_unbox_uint32(v___x_29_);
v___x_31_ = lean_uint32_to_nat(v___x_30_);
v___x_32_ = lp_swirl_x2drbr_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_31_);
lean_dec(v___x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readDigest___lam__0___boxed(lean_object* v_a_33_, lean_object* v_idx_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_workspace_VmVerifier_Spec_Wire_readDigest___lam__0(v_a_33_, v_idx_34_);
lean_dec(v_idx_34_);
lean_dec_ref(v_a_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readDigest(lean_object* v_a_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(v_a_36_);
if (lean_obj_tag(v___x_37_) == 0)
{
lean_object* v_a_38_; lean_object* v_a_39_; lean_object* v___x_41_; uint8_t v_isShared_42_; uint8_t v_isSharedCheck_49_; 
v_a_38_ = lean_ctor_get(v___x_37_, 0);
v_a_39_ = lean_ctor_get(v___x_37_, 1);
v_isSharedCheck_49_ = !lean_is_exclusive(v___x_37_);
if (v_isSharedCheck_49_ == 0)
{
v___x_41_ = v___x_37_;
v_isShared_42_ = v_isSharedCheck_49_;
goto v_resetjp_40_;
}
else
{
lean_inc(v_a_39_);
lean_inc(v_a_38_);
lean_dec(v___x_37_);
v___x_41_ = lean_box(0);
v_isShared_42_ = v_isSharedCheck_49_;
goto v_resetjp_40_;
}
v_resetjp_40_:
{
lean_object* v___f_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_47_; 
v___f_43_ = lean_alloc_closure((void*)(lp_workspace_VmVerifier_Spec_Wire_readDigest___lam__0___boxed), 2, 1);
lean_closure_set(v___f_43_, 0, v_a_38_);
v___x_44_ = lean_unsigned_to_nat(8u);
v___x_45_ = l_Array_ofFn___redArg(v___x_44_, v___f_43_);
if (v_isShared_42_ == 0)
{
lean_ctor_set(v___x_41_, 0, v___x_45_);
v___x_47_ = v___x_41_;
goto v_reusejp_46_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v___x_45_);
lean_ctor_set(v_reuseFailAlloc_48_, 1, v_a_39_);
v___x_47_ = v_reuseFailAlloc_48_;
goto v_reusejp_46_;
}
v_reusejp_46_:
{
return v___x_47_;
}
}
}
else
{
lean_object* v_a_50_; lean_object* v_a_51_; lean_object* v___x_53_; uint8_t v_isShared_54_; uint8_t v_isSharedCheck_58_; 
v_a_50_ = lean_ctor_get(v___x_37_, 0);
v_a_51_ = lean_ctor_get(v___x_37_, 1);
v_isSharedCheck_58_ = !lean_is_exclusive(v___x_37_);
if (v_isSharedCheck_58_ == 0)
{
v___x_53_ = v___x_37_;
v_isShared_54_ = v_isSharedCheck_58_;
goto v_resetjp_52_;
}
else
{
lean_inc(v_a_51_);
lean_inc(v_a_50_);
lean_dec(v___x_37_);
v___x_53_ = lean_box(0);
v_isShared_54_ = v_isSharedCheck_58_;
goto v_resetjp_52_;
}
v_resetjp_52_:
{
lean_object* v___x_56_; 
if (v_isShared_54_ == 0)
{
v___x_56_ = v___x_53_;
goto v_reusejp_55_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v_a_50_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v_a_51_);
v___x_56_ = v_reuseFailAlloc_57_;
goto v_reusejp_55_;
}
v_reusejp_55_:
{
return v___x_56_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readVkCommit(lean_object* v_a_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_workspace_VmVerifier_Spec_Wire_readDigest(v_a_59_);
if (lean_obj_tag(v___x_60_) == 0)
{
lean_object* v_a_61_; lean_object* v_a_62_; lean_object* v___x_63_; 
v_a_61_ = lean_ctor_get(v___x_60_, 0);
lean_inc(v_a_61_);
v_a_62_ = lean_ctor_get(v___x_60_, 1);
lean_inc(v_a_62_);
lean_dec_ref_known(v___x_60_, 2);
v___x_63_ = lp_workspace_VmVerifier_Spec_Wire_readDigest(v_a_62_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v_a_64_; lean_object* v_a_65_; lean_object* v___x_67_; uint8_t v_isShared_68_; uint8_t v_isSharedCheck_73_; 
v_a_64_ = lean_ctor_get(v___x_63_, 0);
v_a_65_ = lean_ctor_get(v___x_63_, 1);
v_isSharedCheck_73_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_73_ == 0)
{
v___x_67_ = v___x_63_;
v_isShared_68_ = v_isSharedCheck_73_;
goto v_resetjp_66_;
}
else
{
lean_inc(v_a_65_);
lean_inc(v_a_64_);
lean_dec(v___x_63_);
v___x_67_ = lean_box(0);
v_isShared_68_ = v_isSharedCheck_73_;
goto v_resetjp_66_;
}
v_resetjp_66_:
{
lean_object* v___x_69_; lean_object* v___x_71_; 
v___x_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_69_, 0, v_a_61_);
lean_ctor_set(v___x_69_, 1, v_a_64_);
if (v_isShared_68_ == 0)
{
lean_ctor_set(v___x_67_, 0, v___x_69_);
v___x_71_ = v___x_67_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v___x_69_);
lean_ctor_set(v_reuseFailAlloc_72_, 1, v_a_65_);
v___x_71_ = v_reuseFailAlloc_72_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
return v___x_71_;
}
}
}
else
{
lean_object* v_a_74_; lean_object* v_a_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_82_; 
lean_dec(v_a_61_);
v_a_74_ = lean_ctor_get(v___x_63_, 0);
v_a_75_ = lean_ctor_get(v___x_63_, 1);
v_isSharedCheck_82_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_82_ == 0)
{
v___x_77_ = v___x_63_;
v_isShared_78_ = v_isSharedCheck_82_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_a_75_);
lean_inc(v_a_74_);
lean_dec(v___x_63_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_82_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v___x_80_; 
if (v_isShared_78_ == 0)
{
v___x_80_ = v___x_77_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v_a_74_);
lean_ctor_set(v_reuseFailAlloc_81_, 1, v_a_75_);
v___x_80_ = v_reuseFailAlloc_81_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
return v___x_80_;
}
}
}
}
else
{
lean_object* v_a_83_; lean_object* v_a_84_; lean_object* v___x_86_; uint8_t v_isShared_87_; uint8_t v_isSharedCheck_91_; 
v_a_83_ = lean_ctor_get(v___x_60_, 0);
v_a_84_ = lean_ctor_get(v___x_60_, 1);
v_isSharedCheck_91_ = !lean_is_exclusive(v___x_60_);
if (v_isSharedCheck_91_ == 0)
{
v___x_86_ = v___x_60_;
v_isShared_87_ = v_isSharedCheck_91_;
goto v_resetjp_85_;
}
else
{
lean_inc(v_a_84_);
lean_inc(v_a_83_);
lean_dec(v___x_60_);
v___x_86_ = lean_box(0);
v_isShared_87_ = v_isSharedCheck_91_;
goto v_resetjp_85_;
}
v_resetjp_85_:
{
lean_object* v___x_89_; 
if (v_isShared_87_ == 0)
{
v___x_89_ = v___x_86_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_90_; 
v_reuseFailAlloc_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_90_, 0, v_a_83_);
lean_ctor_set(v_reuseFailAlloc_90_, 1, v_a_84_);
v___x_89_ = v_reuseFailAlloc_90_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
return v___x_89_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBaselineM(lean_object* v_a_92_){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = lp_workspace_VmVerifier_Spec_Wire_baselineMagic;
v___x_94_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(v___x_93_, v_a_92_);
if (lean_obj_tag(v___x_94_) == 0)
{
lean_object* v_a_95_; lean_object* v___x_96_; 
v_a_95_ = lean_ctor_get(v___x_94_, 1);
lean_inc(v_a_95_);
lean_dec_ref_known(v___x_94_, 2);
v___x_96_ = lp_workspace_VmVerifier_Spec_Wire_readDigest(v_a_95_);
if (lean_obj_tag(v___x_96_) == 0)
{
lean_object* v_a_97_; lean_object* v_a_98_; lean_object* v___x_99_; 
v_a_97_ = lean_ctor_get(v___x_96_, 0);
lean_inc(v_a_97_);
v_a_98_ = lean_ctor_get(v___x_96_, 1);
lean_inc(v_a_98_);
lean_dec_ref_known(v___x_96_, 2);
v___x_99_ = lp_workspace_VmVerifier_Spec_Wire_readDigest(v_a_98_);
if (lean_obj_tag(v___x_99_) == 0)
{
lean_object* v_a_100_; lean_object* v_a_101_; lean_object* v___x_102_; 
v_a_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_a_100_);
v_a_101_ = lean_ctor_get(v___x_99_, 1);
lean_inc(v_a_101_);
lean_dec_ref_known(v___x_99_, 2);
v___x_102_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(v_a_101_);
if (lean_obj_tag(v___x_102_) == 0)
{
lean_object* v_a_103_; lean_object* v_a_104_; lean_object* v___x_105_; 
v_a_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_a_103_);
v_a_104_ = lean_ctor_get(v___x_102_, 1);
lean_inc(v_a_104_);
lean_dec_ref_known(v___x_102_, 2);
v___x_105_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(v_a_104_);
if (lean_obj_tag(v___x_105_) == 0)
{
lean_object* v_a_106_; lean_object* v_a_107_; lean_object* v___x_108_; 
v_a_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc(v_a_106_);
v_a_107_ = lean_ctor_get(v___x_105_, 1);
lean_inc(v_a_107_);
lean_dec_ref_known(v___x_105_, 2);
v___x_108_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(v_a_107_);
if (lean_obj_tag(v___x_108_) == 0)
{
lean_object* v_a_109_; lean_object* v_a_110_; lean_object* v___x_111_; 
v_a_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_a_109_);
v_a_110_ = lean_ctor_get(v___x_108_, 1);
lean_inc(v_a_110_);
lean_dec_ref_known(v___x_108_, 2);
v___x_111_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(v_a_110_);
if (lean_obj_tag(v___x_111_) == 0)
{
lean_object* v_a_112_; lean_object* v_a_113_; lean_object* v___x_114_; 
v_a_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_a_112_);
v_a_113_ = lean_ctor_get(v___x_111_, 1);
lean_inc(v_a_113_);
lean_dec_ref_known(v___x_111_, 2);
v___x_114_ = lp_workspace_VmVerifier_Spec_Wire_readVkCommit(v_a_113_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v_a_116_; lean_object* v___x_117_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_a_115_);
v_a_116_ = lean_ctor_get(v___x_114_, 1);
lean_inc(v_a_116_);
lean_dec_ref_known(v___x_114_, 2);
v___x_117_ = lp_workspace_VmVerifier_Spec_Wire_readVkCommit(v_a_116_);
if (lean_obj_tag(v___x_117_) == 0)
{
lean_object* v_a_118_; lean_object* v_a_119_; lean_object* v___x_120_; 
v_a_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc(v_a_118_);
v_a_119_ = lean_ctor_get(v___x_117_, 1);
lean_inc(v_a_119_);
lean_dec_ref_known(v___x_117_, 2);
v___x_120_ = lp_workspace_VmVerifier_Spec_Wire_readVkCommit(v_a_119_);
if (lean_obj_tag(v___x_120_) == 0)
{
lean_object* v_a_121_; lean_object* v_a_122_; lean_object* v___x_123_; 
v_a_121_ = lean_ctor_get(v___x_120_, 0);
lean_inc(v_a_121_);
v_a_122_ = lean_ctor_get(v___x_120_, 1);
lean_inc(v_a_122_);
lean_dec_ref_known(v___x_120_, 2);
v___x_123_ = lp_workspace_VmVerifier_Spec_Wire_readVkCommit(v_a_122_);
if (lean_obj_tag(v___x_123_) == 0)
{
lean_object* v_a_124_; lean_object* v_a_125_; lean_object* v___x_126_; 
v_a_124_ = lean_ctor_get(v___x_123_, 0);
lean_inc(v_a_124_);
v_a_125_ = lean_ctor_get(v___x_123_, 1);
lean_inc(v_a_125_);
lean_dec_ref_known(v___x_123_, 2);
v___x_126_ = lp_workspace_VmVerifier_Spec_Wire_ensureEnd(v_a_125_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_139_; 
v_a_127_ = lean_ctor_get(v___x_126_, 1);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_139_ == 0)
{
lean_object* v_unused_140_; 
v_unused_140_ = lean_ctor_get(v___x_126_, 0);
lean_dec(v_unused_140_);
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_139_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_139_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
uint32_t v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_137_; 
v___x_131_ = lean_unbox_uint32(v_a_103_);
lean_dec(v_a_103_);
v___x_132_ = lean_uint32_to_nat(v___x_131_);
v___x_133_ = lp_swirl_x2drbr_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_132_);
lean_dec(v___x_132_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v_a_106_);
lean_ctor_set(v___x_134_, 1, v_a_109_);
v___x_135_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v___x_135_, 0, v_a_97_);
lean_ctor_set(v___x_135_, 1, v_a_100_);
lean_ctor_set(v___x_135_, 2, v___x_133_);
lean_ctor_set(v___x_135_, 3, v___x_134_);
lean_ctor_set(v___x_135_, 4, v_a_112_);
lean_ctor_set(v___x_135_, 5, v_a_115_);
lean_ctor_set(v___x_135_, 6, v_a_118_);
lean_ctor_set(v___x_135_, 7, v_a_121_);
lean_ctor_set(v___x_135_, 8, v_a_124_);
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 0, v___x_135_);
v___x_137_ = v___x_129_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_135_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v_a_127_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
else
{
lean_object* v_a_141_; lean_object* v_a_142_; lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_149_; 
lean_dec(v_a_124_);
lean_dec(v_a_121_);
lean_dec(v_a_118_);
lean_dec(v_a_115_);
lean_dec(v_a_112_);
lean_dec(v_a_109_);
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_141_ = lean_ctor_get(v___x_126_, 0);
v_a_142_ = lean_ctor_get(v___x_126_, 1);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_149_ == 0)
{
v___x_144_ = v___x_126_;
v_isShared_145_ = v_isSharedCheck_149_;
goto v_resetjp_143_;
}
else
{
lean_inc(v_a_142_);
lean_inc(v_a_141_);
lean_dec(v___x_126_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_149_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
lean_object* v___x_147_; 
if (v_isShared_145_ == 0)
{
v___x_147_ = v___x_144_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v_a_141_);
lean_ctor_set(v_reuseFailAlloc_148_, 1, v_a_142_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
}
else
{
lean_object* v_a_150_; lean_object* v_a_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_158_; 
lean_dec(v_a_121_);
lean_dec(v_a_118_);
lean_dec(v_a_115_);
lean_dec(v_a_112_);
lean_dec(v_a_109_);
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_150_ = lean_ctor_get(v___x_123_, 0);
v_a_151_ = lean_ctor_get(v___x_123_, 1);
v_isSharedCheck_158_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_158_ == 0)
{
v___x_153_ = v___x_123_;
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_a_151_);
lean_inc(v_a_150_);
lean_dec(v___x_123_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_156_; 
if (v_isShared_154_ == 0)
{
v___x_156_ = v___x_153_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_a_150_);
lean_ctor_set(v_reuseFailAlloc_157_, 1, v_a_151_);
v___x_156_ = v_reuseFailAlloc_157_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
return v___x_156_;
}
}
}
}
else
{
lean_object* v_a_159_; lean_object* v_a_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_167_; 
lean_dec(v_a_118_);
lean_dec(v_a_115_);
lean_dec(v_a_112_);
lean_dec(v_a_109_);
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_159_ = lean_ctor_get(v___x_120_, 0);
v_a_160_ = lean_ctor_get(v___x_120_, 1);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_167_ == 0)
{
v___x_162_ = v___x_120_;
v_isShared_163_ = v_isSharedCheck_167_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_a_160_);
lean_inc(v_a_159_);
lean_dec(v___x_120_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_167_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_165_; 
if (v_isShared_163_ == 0)
{
v___x_165_ = v___x_162_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v_a_159_);
lean_ctor_set(v_reuseFailAlloc_166_, 1, v_a_160_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
else
{
lean_object* v_a_168_; lean_object* v_a_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_176_; 
lean_dec(v_a_115_);
lean_dec(v_a_112_);
lean_dec(v_a_109_);
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_168_ = lean_ctor_get(v___x_117_, 0);
v_a_169_ = lean_ctor_get(v___x_117_, 1);
v_isSharedCheck_176_ = !lean_is_exclusive(v___x_117_);
if (v_isSharedCheck_176_ == 0)
{
v___x_171_ = v___x_117_;
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_a_169_);
lean_inc(v_a_168_);
lean_dec(v___x_117_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_174_; 
if (v_isShared_172_ == 0)
{
v___x_174_ = v___x_171_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_a_168_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v_a_169_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
else
{
lean_object* v_a_177_; lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_185_; 
lean_dec(v_a_112_);
lean_dec(v_a_109_);
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_177_ = lean_ctor_get(v___x_114_, 0);
v_a_178_ = lean_ctor_get(v___x_114_, 1);
v_isSharedCheck_185_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_185_ == 0)
{
v___x_180_ = v___x_114_;
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_inc(v_a_177_);
lean_dec(v___x_114_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_183_; 
if (v_isShared_181_ == 0)
{
v___x_183_ = v___x_180_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v_a_177_);
lean_ctor_set(v_reuseFailAlloc_184_, 1, v_a_178_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
else
{
lean_object* v_a_186_; lean_object* v_a_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_194_; 
lean_dec(v_a_109_);
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_186_ = lean_ctor_get(v___x_111_, 0);
v_a_187_ = lean_ctor_get(v___x_111_, 1);
v_isSharedCheck_194_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_194_ == 0)
{
v___x_189_ = v___x_111_;
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
else
{
lean_inc(v_a_187_);
lean_inc(v_a_186_);
lean_dec(v___x_111_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_192_; 
if (v_isShared_190_ == 0)
{
v___x_192_ = v___x_189_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v_a_186_);
lean_ctor_set(v_reuseFailAlloc_193_, 1, v_a_187_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
}
else
{
lean_object* v_a_195_; lean_object* v_a_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_203_; 
lean_dec(v_a_106_);
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_195_ = lean_ctor_get(v___x_108_, 0);
v_a_196_ = lean_ctor_get(v___x_108_, 1);
v_isSharedCheck_203_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_203_ == 0)
{
v___x_198_ = v___x_108_;
v_isShared_199_ = v_isSharedCheck_203_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_a_196_);
lean_inc(v_a_195_);
lean_dec(v___x_108_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_203_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v___x_201_; 
if (v_isShared_199_ == 0)
{
v___x_201_ = v___x_198_;
goto v_reusejp_200_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v_a_195_);
lean_ctor_set(v_reuseFailAlloc_202_, 1, v_a_196_);
v___x_201_ = v_reuseFailAlloc_202_;
goto v_reusejp_200_;
}
v_reusejp_200_:
{
return v___x_201_;
}
}
}
}
else
{
lean_object* v_a_204_; lean_object* v_a_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_212_; 
lean_dec(v_a_103_);
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_204_ = lean_ctor_get(v___x_105_, 0);
v_a_205_ = lean_ctor_get(v___x_105_, 1);
v_isSharedCheck_212_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_212_ == 0)
{
v___x_207_ = v___x_105_;
v_isShared_208_ = v_isSharedCheck_212_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_a_205_);
lean_inc(v_a_204_);
lean_dec(v___x_105_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_212_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_210_; 
if (v_isShared_208_ == 0)
{
v___x_210_ = v___x_207_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_a_204_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v_a_205_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
}
}
else
{
lean_object* v_a_213_; lean_object* v_a_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_221_; 
lean_dec(v_a_100_);
lean_dec(v_a_97_);
v_a_213_ = lean_ctor_get(v___x_102_, 0);
v_a_214_ = lean_ctor_get(v___x_102_, 1);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_221_ == 0)
{
v___x_216_ = v___x_102_;
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_a_214_);
lean_inc(v_a_213_);
lean_dec(v___x_102_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_219_; 
if (v_isShared_217_ == 0)
{
v___x_219_ = v___x_216_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_a_213_);
lean_ctor_set(v_reuseFailAlloc_220_, 1, v_a_214_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
}
else
{
lean_object* v_a_222_; lean_object* v_a_223_; lean_object* v___x_225_; uint8_t v_isShared_226_; uint8_t v_isSharedCheck_230_; 
lean_dec(v_a_97_);
v_a_222_ = lean_ctor_get(v___x_99_, 0);
v_a_223_ = lean_ctor_get(v___x_99_, 1);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_99_);
if (v_isSharedCheck_230_ == 0)
{
v___x_225_ = v___x_99_;
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
else
{
lean_inc(v_a_223_);
lean_inc(v_a_222_);
lean_dec(v___x_99_);
v___x_225_ = lean_box(0);
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
v_resetjp_224_:
{
lean_object* v___x_228_; 
if (v_isShared_226_ == 0)
{
v___x_228_ = v___x_225_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_a_222_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v_a_223_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
}
}
else
{
lean_object* v_a_231_; lean_object* v_a_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_239_; 
v_a_231_ = lean_ctor_get(v___x_96_, 0);
v_a_232_ = lean_ctor_get(v___x_96_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_96_);
if (v_isSharedCheck_239_ == 0)
{
v___x_234_ = v___x_96_;
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_a_232_);
lean_inc(v_a_231_);
lean_dec(v___x_96_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_237_; 
if (v_isShared_235_ == 0)
{
v___x_237_ = v___x_234_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v_a_231_);
lean_ctor_set(v_reuseFailAlloc_238_, 1, v_a_232_);
v___x_237_ = v_reuseFailAlloc_238_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
return v___x_237_;
}
}
}
}
else
{
lean_object* v_a_240_; lean_object* v_a_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
v_a_240_ = lean_ctor_get(v___x_94_, 0);
v_a_241_ = lean_ctor_get(v___x_94_, 1);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_248_ == 0)
{
v___x_243_ = v___x_94_;
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_a_241_);
lean_inc(v_a_240_);
lean_dec(v___x_94_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
if (v_isShared_244_ == 0)
{
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_a_240_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v_a_241_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBaseline(lean_object* v_data_249_){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_250_ = lean_alloc_closure((void*)(lp_workspace_VmVerifier_Spec_Wire_readBaselineM), 1, 0);
v___x_251_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(v___x_250_, v_data_249_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___lam__0(lean_object* v___y_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(v___y_252_);
if (lean_obj_tag(v___x_253_) == 0)
{
lean_object* v_a_254_; lean_object* v_a_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_265_; 
v_a_254_ = lean_ctor_get(v___x_253_, 0);
v_a_255_ = lean_ctor_get(v___x_253_, 1);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_265_ == 0)
{
v___x_257_ = v___x_253_;
v_isShared_258_ = v_isSharedCheck_265_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_a_255_);
lean_inc(v_a_254_);
lean_dec(v___x_253_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_265_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
uint32_t v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_263_; 
v___x_259_ = lean_unbox_uint32(v_a_254_);
lean_dec(v_a_254_);
v___x_260_ = lean_uint32_to_nat(v___x_259_);
v___x_261_ = lp_swirl_x2drbr_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_260_);
lean_dec(v___x_260_);
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 0, v___x_261_);
v___x_263_ = v___x_257_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v___x_261_);
lean_ctor_set(v_reuseFailAlloc_264_, 1, v_a_255_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
}
else
{
lean_object* v_a_266_; lean_object* v_a_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_274_; 
v_a_266_ = lean_ctor_get(v___x_253_, 0);
v_a_267_ = lean_ctor_get(v___x_253_, 1);
v_isSharedCheck_274_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_274_ == 0)
{
v___x_269_ = v___x_253_;
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_a_267_);
lean_inc(v_a_266_);
lean_dec(v___x_253_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_272_; 
if (v_isShared_270_ == 0)
{
v___x_272_ = v___x_269_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v_a_266_);
lean_ctor_set(v_reuseFailAlloc_273_, 1, v_a_267_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM(lean_object* v_a_276_){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = lp_workspace_VmVerifier_Spec_Wire_userPvsMagic;
v___x_278_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(v___x_277_, v_a_276_);
if (lean_obj_tag(v___x_278_) == 0)
{
lean_object* v_a_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v_a_279_ = lean_ctor_get(v___x_278_, 1);
lean_inc(v_a_279_);
lean_dec_ref_known(v___x_278_, 2);
v___x_280_ = lean_alloc_closure((void*)(lp_workspace_VmVerifier_Spec_Wire_readDigest), 1, 0);
v___x_281_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_280_, v_a_279_);
if (lean_obj_tag(v___x_281_) == 0)
{
lean_object* v_a_282_; lean_object* v_a_283_; lean_object* v___f_284_; lean_object* v___x_285_; 
v_a_282_ = lean_ctor_get(v___x_281_, 0);
lean_inc(v_a_282_);
v_a_283_ = lean_ctor_get(v___x_281_, 1);
lean_inc(v_a_283_);
lean_dec_ref_known(v___x_281_, 2);
v___f_284_ = ((lean_object*)(lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM___closed__0));
v___x_285_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___f_284_, v_a_283_);
if (lean_obj_tag(v___x_285_) == 0)
{
lean_object* v_a_286_; lean_object* v_a_287_; lean_object* v___x_288_; 
v_a_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_a_286_);
v_a_287_ = lean_ctor_get(v___x_285_, 1);
lean_inc(v_a_287_);
lean_dec_ref_known(v___x_285_, 2);
v___x_288_ = lp_workspace_VmVerifier_Spec_Wire_readDigest(v_a_287_);
if (lean_obj_tag(v___x_288_) == 0)
{
lean_object* v_a_289_; lean_object* v_a_290_; lean_object* v___x_291_; 
v_a_289_ = lean_ctor_get(v___x_288_, 0);
lean_inc(v_a_289_);
v_a_290_ = lean_ctor_get(v___x_288_, 1);
lean_inc(v_a_290_);
lean_dec_ref_known(v___x_288_, 2);
v___x_291_ = lp_workspace_VmVerifier_Spec_Wire_ensureEnd(v_a_290_);
if (lean_obj_tag(v___x_291_) == 0)
{
lean_object* v_a_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_302_; 
v_a_292_ = lean_ctor_get(v___x_291_, 1);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_291_);
if (v_isSharedCheck_302_ == 0)
{
lean_object* v_unused_303_; 
v_unused_303_ = lean_ctor_get(v___x_291_, 0);
lean_dec(v_unused_303_);
v___x_294_ = v___x_291_;
v_isShared_295_ = v_isSharedCheck_302_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_a_292_);
lean_dec(v___x_291_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_302_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_300_; 
v___x_296_ = lean_array_to_list(v_a_282_);
v___x_297_ = lean_array_to_list(v_a_286_);
v___x_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_298_, 0, v___x_296_);
lean_ctor_set(v___x_298_, 1, v___x_297_);
lean_ctor_set(v___x_298_, 2, v_a_289_);
if (v_isShared_295_ == 0)
{
lean_ctor_set(v___x_294_, 0, v___x_298_);
v___x_300_ = v___x_294_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v___x_298_);
lean_ctor_set(v_reuseFailAlloc_301_, 1, v_a_292_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
else
{
lean_object* v_a_304_; lean_object* v_a_305_; lean_object* v___x_307_; uint8_t v_isShared_308_; uint8_t v_isSharedCheck_312_; 
lean_dec(v_a_289_);
lean_dec(v_a_286_);
lean_dec(v_a_282_);
v_a_304_ = lean_ctor_get(v___x_291_, 0);
v_a_305_ = lean_ctor_get(v___x_291_, 1);
v_isSharedCheck_312_ = !lean_is_exclusive(v___x_291_);
if (v_isSharedCheck_312_ == 0)
{
v___x_307_ = v___x_291_;
v_isShared_308_ = v_isSharedCheck_312_;
goto v_resetjp_306_;
}
else
{
lean_inc(v_a_305_);
lean_inc(v_a_304_);
lean_dec(v___x_291_);
v___x_307_ = lean_box(0);
v_isShared_308_ = v_isSharedCheck_312_;
goto v_resetjp_306_;
}
v_resetjp_306_:
{
lean_object* v___x_310_; 
if (v_isShared_308_ == 0)
{
v___x_310_ = v___x_307_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v_a_304_);
lean_ctor_set(v_reuseFailAlloc_311_, 1, v_a_305_);
v___x_310_ = v_reuseFailAlloc_311_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
return v___x_310_;
}
}
}
}
else
{
lean_object* v_a_313_; lean_object* v_a_314_; lean_object* v___x_316_; uint8_t v_isShared_317_; uint8_t v_isSharedCheck_321_; 
lean_dec(v_a_286_);
lean_dec(v_a_282_);
v_a_313_ = lean_ctor_get(v___x_288_, 0);
v_a_314_ = lean_ctor_get(v___x_288_, 1);
v_isSharedCheck_321_ = !lean_is_exclusive(v___x_288_);
if (v_isSharedCheck_321_ == 0)
{
v___x_316_ = v___x_288_;
v_isShared_317_ = v_isSharedCheck_321_;
goto v_resetjp_315_;
}
else
{
lean_inc(v_a_314_);
lean_inc(v_a_313_);
lean_dec(v___x_288_);
v___x_316_ = lean_box(0);
v_isShared_317_ = v_isSharedCheck_321_;
goto v_resetjp_315_;
}
v_resetjp_315_:
{
lean_object* v___x_319_; 
if (v_isShared_317_ == 0)
{
v___x_319_ = v___x_316_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v_a_313_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v_a_314_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
}
}
else
{
lean_object* v_a_322_; lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
lean_dec(v_a_282_);
v_a_322_ = lean_ctor_get(v___x_285_, 0);
v_a_323_ = lean_ctor_get(v___x_285_, 1);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_285_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_285_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_inc(v_a_322_);
lean_dec(v___x_285_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
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
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_322_);
lean_ctor_set(v_reuseFailAlloc_329_, 1, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
else
{
lean_object* v_a_331_; lean_object* v_a_332_; lean_object* v___x_334_; uint8_t v_isShared_335_; uint8_t v_isSharedCheck_339_; 
v_a_331_ = lean_ctor_get(v___x_281_, 0);
v_a_332_ = lean_ctor_get(v___x_281_, 1);
v_isSharedCheck_339_ = !lean_is_exclusive(v___x_281_);
if (v_isSharedCheck_339_ == 0)
{
v___x_334_ = v___x_281_;
v_isShared_335_ = v_isSharedCheck_339_;
goto v_resetjp_333_;
}
else
{
lean_inc(v_a_332_);
lean_inc(v_a_331_);
lean_dec(v___x_281_);
v___x_334_ = lean_box(0);
v_isShared_335_ = v_isSharedCheck_339_;
goto v_resetjp_333_;
}
v_resetjp_333_:
{
lean_object* v___x_337_; 
if (v_isShared_335_ == 0)
{
v___x_337_ = v___x_334_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v_a_331_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v_a_332_);
v___x_337_ = v_reuseFailAlloc_338_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
return v___x_337_;
}
}
}
}
else
{
lean_object* v_a_340_; lean_object* v_a_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_348_; 
v_a_340_ = lean_ctor_get(v___x_278_, 0);
v_a_341_ = lean_ctor_get(v___x_278_, 1);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_348_ == 0)
{
v___x_343_ = v___x_278_;
v_isShared_344_ = v_isSharedCheck_348_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_a_341_);
lean_inc(v_a_340_);
lean_dec(v___x_278_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_348_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___x_346_; 
if (v_isShared_344_ == 0)
{
v___x_346_ = v___x_343_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v_a_340_);
lean_ctor_set(v_reuseFailAlloc_347_, 1, v_a_341_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProof(lean_object* v_data_349_){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_350_ = lean_alloc_closure((void*)(lp_workspace_VmVerifier_Spec_Wire_readUserPvsProofM), 1, 0);
v___x_351_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(v___x_350_, v_data_349_);
return v___x_351_;
}
}
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(lean_object* v_xs_352_, lean_object* v_i_353_){
_start:
{
lean_object* v___x_354_; uint8_t v___x_355_; 
v___x_354_ = lean_byte_array_size(v_xs_352_);
v___x_355_ = lean_nat_dec_lt(v_i_353_, v___x_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0___boxed(lean_object* v_xs_356_, lean_object* v_i_357_){
_start:
{
uint8_t v_res_358_; lean_object* v_r_359_; 
v_res_358_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(v_xs_356_, v_i_357_);
lean_dec(v_i_357_);
lean_dec_ref(v_xs_356_);
v_r_359_ = lean_box(v_res_358_);
return v_r_359_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readU32LE(lean_object* v_bytes_360_, lean_object* v_offset_361_){
_start:
{
uint32_t v___y_363_; uint32_t v___y_364_; uint32_t v___y_365_; uint8_t v___y_366_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; uint8_t v___x_382_; 
v___x_379_ = lean_unsigned_to_nat(4u);
v___x_380_ = lean_nat_add(v_offset_361_, v___x_379_);
v___x_381_ = lean_byte_array_size(v_bytes_360_);
v___x_382_ = lean_nat_dec_le(v___x_380_, v___x_381_);
lean_dec(v___x_380_);
if (v___x_382_ == 0)
{
lean_object* v___x_383_; 
v___x_383_ = lean_box(0);
return v___x_383_;
}
else
{
uint8_t v___x_384_; uint32_t v___y_386_; uint32_t v___y_387_; uint8_t v___y_388_; uint32_t v___y_398_; uint8_t v___y_399_; uint8_t v___y_409_; uint8_t v___x_418_; 
v___x_384_ = 0;
v___x_418_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(v_bytes_360_, v_offset_361_);
if (v___x_418_ == 0)
{
lean_object* v___x_419_; lean_object* v___x_420_; uint8_t v___x_421_; 
v___x_419_ = lean_box(v___x_384_);
v___x_420_ = l_outOfBounds___redArg(v___x_419_);
lean_dec(v___x_419_);
v___x_421_ = lean_unbox(v___x_420_);
lean_dec(v___x_420_);
v___y_409_ = v___x_421_;
goto v___jp_408_;
}
else
{
uint8_t v___x_422_; 
v___x_422_ = lean_byte_array_fget(v_bytes_360_, v_offset_361_);
v___y_409_ = v___x_422_;
goto v___jp_408_;
}
v___jp_385_:
{
uint32_t v_b2_389_; lean_object* v___x_390_; lean_object* v___x_391_; uint8_t v___x_392_; 
v_b2_389_ = lean_uint8_to_uint32(v___y_388_);
v___x_390_ = lean_unsigned_to_nat(3u);
v___x_391_ = lean_nat_add(v_offset_361_, v___x_390_);
v___x_392_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(v_bytes_360_, v___x_391_);
if (v___x_392_ == 0)
{
lean_object* v___x_393_; lean_object* v___x_394_; uint8_t v___x_395_; 
lean_dec(v___x_391_);
v___x_393_ = lean_box(v___x_384_);
v___x_394_ = l_outOfBounds___redArg(v___x_393_);
lean_dec(v___x_393_);
v___x_395_ = lean_unbox(v___x_394_);
lean_dec(v___x_394_);
v___y_363_ = v___y_386_;
v___y_364_ = v___y_387_;
v___y_365_ = v_b2_389_;
v___y_366_ = v___x_395_;
goto v___jp_362_;
}
else
{
uint8_t v___x_396_; 
v___x_396_ = lean_byte_array_fget(v_bytes_360_, v___x_391_);
lean_dec(v___x_391_);
v___y_363_ = v___y_386_;
v___y_364_ = v___y_387_;
v___y_365_ = v_b2_389_;
v___y_366_ = v___x_396_;
goto v___jp_362_;
}
}
v___jp_397_:
{
uint32_t v_b1_400_; lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v_b1_400_ = lean_uint8_to_uint32(v___y_399_);
v___x_401_ = lean_unsigned_to_nat(2u);
v___x_402_ = lean_nat_add(v_offset_361_, v___x_401_);
v___x_403_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(v_bytes_360_, v___x_402_);
if (v___x_403_ == 0)
{
lean_object* v___x_404_; lean_object* v___x_405_; uint8_t v___x_406_; 
lean_dec(v___x_402_);
v___x_404_ = lean_box(v___x_384_);
v___x_405_ = l_outOfBounds___redArg(v___x_404_);
lean_dec(v___x_404_);
v___x_406_ = lean_unbox(v___x_405_);
lean_dec(v___x_405_);
v___y_386_ = v___y_398_;
v___y_387_ = v_b1_400_;
v___y_388_ = v___x_406_;
goto v___jp_385_;
}
else
{
uint8_t v___x_407_; 
v___x_407_ = lean_byte_array_fget(v_bytes_360_, v___x_402_);
lean_dec(v___x_402_);
v___y_386_ = v___y_398_;
v___y_387_ = v_b1_400_;
v___y_388_ = v___x_407_;
goto v___jp_385_;
}
}
v___jp_408_:
{
uint32_t v_b0_410_; lean_object* v___x_411_; lean_object* v___x_412_; uint8_t v___x_413_; 
v_b0_410_ = lean_uint8_to_uint32(v___y_409_);
v___x_411_ = lean_unsigned_to_nat(1u);
v___x_412_ = lean_nat_add(v_offset_361_, v___x_411_);
v___x_413_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE___lam__0(v_bytes_360_, v___x_412_);
if (v___x_413_ == 0)
{
lean_object* v___x_414_; lean_object* v___x_415_; uint8_t v___x_416_; 
lean_dec(v___x_412_);
v___x_414_ = lean_box(v___x_384_);
v___x_415_ = l_outOfBounds___redArg(v___x_414_);
lean_dec(v___x_414_);
v___x_416_ = lean_unbox(v___x_415_);
lean_dec(v___x_415_);
v___y_398_ = v_b0_410_;
v___y_399_ = v___x_416_;
goto v___jp_397_;
}
else
{
uint8_t v___x_417_; 
v___x_417_ = lean_byte_array_fget(v_bytes_360_, v___x_412_);
lean_dec(v___x_412_);
v___y_398_ = v_b0_410_;
v___y_399_ = v___x_417_;
goto v___jp_397_;
}
}
}
v___jp_362_:
{
uint32_t v_b3_367_; uint32_t v___x_368_; uint32_t v___x_369_; uint32_t v___x_370_; uint32_t v___x_371_; uint32_t v___x_372_; uint32_t v___x_373_; uint32_t v___x_374_; uint32_t v___x_375_; uint32_t v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v_b3_367_ = lean_uint8_to_uint32(v___y_366_);
v___x_368_ = 8;
v___x_369_ = lean_uint32_shift_left(v___y_364_, v___x_368_);
v___x_370_ = lean_uint32_lor(v___y_363_, v___x_369_);
v___x_371_ = 16;
v___x_372_ = lean_uint32_shift_left(v___y_365_, v___x_371_);
v___x_373_ = lean_uint32_lor(v___x_370_, v___x_372_);
v___x_374_ = 24;
v___x_375_ = lean_uint32_shift_left(v_b3_367_, v___x_374_);
v___x_376_ = lean_uint32_lor(v___x_373_, v___x_375_);
v___x_377_ = lean_box_uint32(v___x_376_);
v___x_378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
return v___x_378_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readU32LE___boxed(lean_object* v_bytes_423_, lean_object* v_offset_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE(v_bytes_423_, v_offset_424_);
lean_dec(v_offset_424_);
lean_dec_ref(v_bytes_423_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_sliceBytes(lean_object* v_bytes_426_, lean_object* v_offset_427_, lean_object* v_len_428_){
_start:
{
lean_object* v___x_429_; lean_object* v___x_430_; uint8_t v___x_431_; 
v___x_429_ = lean_nat_add(v_offset_427_, v_len_428_);
v___x_430_ = lean_byte_array_size(v_bytes_426_);
v___x_431_ = lean_nat_dec_le(v___x_429_, v___x_430_);
if (v___x_431_ == 0)
{
lean_object* v___x_432_; 
lean_dec(v___x_429_);
lean_dec(v_offset_427_);
v___x_432_ = lean_box(0);
return v___x_432_;
}
else
{
lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_433_ = l_ByteArray_extract(v_bytes_426_, v_offset_427_, v___x_429_);
lean_dec(v___x_429_);
v___x_434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_434_, 0, v___x_433_);
return v___x_434_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_sliceBytes___boxed(lean_object* v_bytes_435_, lean_object* v_offset_436_, lean_object* v_len_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_workspace_VmVerifier_Spec_Wire_sliceBytes(v_bytes_435_, v_offset_436_, v_len_437_);
lean_dec(v_len_437_);
lean_dec_ref(v_bytes_435_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBlobAt(lean_object* v_bytes_439_, lean_object* v_offset_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_workspace_VmVerifier_Spec_Wire_readU32LE(v_bytes_439_, v_offset_440_);
if (lean_obj_tag(v___x_441_) == 0)
{
lean_object* v___x_442_; 
v___x_442_ = lean_box(0);
return v___x_442_;
}
else
{
lean_object* v_val_443_; lean_object* v___x_444_; lean_object* v___x_445_; uint32_t v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v_val_443_ = lean_ctor_get(v___x_441_, 0);
lean_inc(v_val_443_);
lean_dec_ref_known(v___x_441_, 1);
v___x_444_ = lean_unsigned_to_nat(4u);
v___x_445_ = lean_nat_add(v_offset_440_, v___x_444_);
v___x_446_ = lean_unbox_uint32(v_val_443_);
lean_dec(v_val_443_);
v___x_447_ = lean_uint32_to_nat(v___x_446_);
lean_inc(v___x_445_);
v___x_448_ = lp_workspace_VmVerifier_Spec_Wire_sliceBytes(v_bytes_439_, v___x_445_, v___x_447_);
if (lean_obj_tag(v___x_448_) == 0)
{
lean_object* v___x_449_; 
lean_dec(v___x_447_);
lean_dec(v___x_445_);
v___x_449_ = lean_box(0);
return v___x_449_;
}
else
{
lean_object* v_val_450_; lean_object* v___x_452_; uint8_t v_isShared_453_; uint8_t v_isSharedCheck_459_; 
v_val_450_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_459_ == 0)
{
v___x_452_ = v___x_448_;
v_isShared_453_ = v_isSharedCheck_459_;
goto v_resetjp_451_;
}
else
{
lean_inc(v_val_450_);
lean_dec(v___x_448_);
v___x_452_ = lean_box(0);
v_isShared_453_ = v_isSharedCheck_459_;
goto v_resetjp_451_;
}
v_resetjp_451_:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
v___x_454_ = lean_nat_add(v___x_445_, v___x_447_);
lean_dec(v___x_447_);
lean_dec(v___x_445_);
v___x_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_455_, 0, v_val_450_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
if (v_isShared_453_ == 0)
{
lean_ctor_set(v___x_452_, 0, v___x_455_);
v___x_457_ = v___x_452_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_455_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_readBlobAt___boxed(lean_object* v_bytes_460_, lean_object* v_offset_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_workspace_VmVerifier_Spec_Wire_readBlobAt(v_bytes_460_, v_offset_461_);
lean_dec(v_offset_461_);
lean_dec_ref(v_bytes_460_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object* v_bytes_463_){
_start:
{
lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_464_ = lean_unsigned_to_nat(0u);
v___x_465_ = lp_workspace_VmVerifier_Spec_Wire_readBlobAt(v_bytes_463_, v___x_464_);
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v___x_466_; 
v___x_466_ = lean_box(0);
return v___x_466_;
}
else
{
lean_object* v_val_467_; lean_object* v_fst_468_; lean_object* v_snd_469_; lean_object* v___x_470_; 
v_val_467_ = lean_ctor_get(v___x_465_, 0);
lean_inc(v_val_467_);
lean_dec_ref_known(v___x_465_, 1);
v_fst_468_ = lean_ctor_get(v_val_467_, 0);
lean_inc(v_fst_468_);
v_snd_469_ = lean_ctor_get(v_val_467_, 1);
lean_inc(v_snd_469_);
lean_dec(v_val_467_);
v___x_470_ = lp_workspace_VmVerifier_Spec_Wire_readBlobAt(v_bytes_463_, v_snd_469_);
lean_dec(v_snd_469_);
if (lean_obj_tag(v___x_470_) == 0)
{
lean_object* v___x_471_; 
lean_dec(v_fst_468_);
v___x_471_ = lean_box(0);
return v___x_471_;
}
else
{
lean_object* v_val_472_; lean_object* v_fst_473_; lean_object* v_snd_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_527_; 
v_val_472_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_val_472_);
lean_dec_ref_known(v___x_470_, 1);
v_fst_473_ = lean_ctor_get(v_val_472_, 0);
v_snd_474_ = lean_ctor_get(v_val_472_, 1);
v_isSharedCheck_527_ = !lean_is_exclusive(v_val_472_);
if (v_isSharedCheck_527_ == 0)
{
v___x_476_ = v_val_472_;
v_isShared_477_ = v_isSharedCheck_527_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_snd_474_);
lean_inc(v_fst_473_);
lean_dec(v_val_472_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_527_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v___x_478_; 
v___x_478_ = lp_workspace_VmVerifier_Spec_Wire_readBlobAt(v_bytes_463_, v_snd_474_);
lean_dec(v_snd_474_);
if (lean_obj_tag(v___x_478_) == 0)
{
lean_object* v___x_479_; 
lean_del_object(v___x_476_);
lean_dec(v_fst_473_);
lean_dec(v_fst_468_);
v___x_479_ = lean_box(0);
return v___x_479_;
}
else
{
lean_object* v_val_480_; lean_object* v_fst_481_; lean_object* v_snd_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_526_; 
v_val_480_ = lean_ctor_get(v___x_478_, 0);
lean_inc(v_val_480_);
lean_dec_ref_known(v___x_478_, 1);
v_fst_481_ = lean_ctor_get(v_val_480_, 0);
v_snd_482_ = lean_ctor_get(v_val_480_, 1);
v_isSharedCheck_526_ = !lean_is_exclusive(v_val_480_);
if (v_isSharedCheck_526_ == 0)
{
v___x_484_ = v_val_480_;
v_isShared_485_ = v_isSharedCheck_526_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_snd_482_);
lean_inc(v_fst_481_);
lean_dec(v_val_480_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_526_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; 
v___x_486_ = lp_workspace_VmVerifier_Spec_Wire_readBlobAt(v_bytes_463_, v_snd_482_);
lean_dec(v_snd_482_);
if (lean_obj_tag(v___x_486_) == 0)
{
lean_object* v___x_487_; 
lean_del_object(v___x_484_);
lean_dec(v_fst_481_);
lean_del_object(v___x_476_);
lean_dec(v_fst_473_);
lean_dec(v_fst_468_);
v___x_487_ = lean_box(0);
return v___x_487_;
}
else
{
lean_object* v_val_488_; lean_object* v_fst_489_; lean_object* v_snd_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_525_; 
v_val_488_ = lean_ctor_get(v___x_486_, 0);
lean_inc(v_val_488_);
lean_dec_ref_known(v___x_486_, 1);
v_fst_489_ = lean_ctor_get(v_val_488_, 0);
v_snd_490_ = lean_ctor_get(v_val_488_, 1);
v_isSharedCheck_525_ = !lean_is_exclusive(v_val_488_);
if (v_isSharedCheck_525_ == 0)
{
v___x_492_ = v_val_488_;
v_isShared_493_ = v_isSharedCheck_525_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_snd_490_);
lean_inc(v_fst_489_);
lean_dec(v_val_488_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_525_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_494_; 
v___x_494_ = lp_workspace_VmVerifier_Spec_Wire_readBlobAt(v_bytes_463_, v_snd_490_);
lean_dec(v_snd_490_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v___x_495_; 
lean_del_object(v___x_492_);
lean_dec(v_fst_489_);
lean_del_object(v___x_484_);
lean_dec(v_fst_481_);
lean_del_object(v___x_476_);
lean_dec(v_fst_473_);
lean_dec(v_fst_468_);
v___x_495_ = lean_box(0);
return v___x_495_;
}
else
{
lean_object* v_val_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_524_; 
v_val_496_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_524_ == 0)
{
v___x_498_ = v___x_494_;
v_isShared_499_ = v_isSharedCheck_524_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_val_496_);
lean_dec(v___x_494_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_524_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v_fst_500_; lean_object* v_snd_501_; lean_object* v___x_503_; uint8_t v_isShared_504_; uint8_t v_isSharedCheck_523_; 
v_fst_500_ = lean_ctor_get(v_val_496_, 0);
v_snd_501_ = lean_ctor_get(v_val_496_, 1);
v_isSharedCheck_523_ = !lean_is_exclusive(v_val_496_);
if (v_isSharedCheck_523_ == 0)
{
v___x_503_ = v_val_496_;
v_isShared_504_ = v_isSharedCheck_523_;
goto v_resetjp_502_;
}
else
{
lean_inc(v_snd_501_);
lean_inc(v_fst_500_);
lean_dec(v_val_496_);
v___x_503_ = lean_box(0);
v_isShared_504_ = v_isSharedCheck_523_;
goto v_resetjp_502_;
}
v_resetjp_502_:
{
lean_object* v___x_505_; uint8_t v___x_506_; 
v___x_505_ = lean_byte_array_size(v_bytes_463_);
v___x_506_ = lean_nat_dec_eq(v_snd_501_, v___x_505_);
lean_dec(v_snd_501_);
if (v___x_506_ == 0)
{
lean_object* v___x_507_; 
lean_del_object(v___x_503_);
lean_dec(v_fst_500_);
lean_del_object(v___x_498_);
lean_del_object(v___x_492_);
lean_dec(v_fst_489_);
lean_del_object(v___x_484_);
lean_dec(v_fst_481_);
lean_del_object(v___x_476_);
lean_dec(v_fst_473_);
lean_dec(v_fst_468_);
v___x_507_ = lean_box(0);
return v___x_507_;
}
else
{
lean_object* v___x_509_; 
if (v_isShared_504_ == 0)
{
lean_ctor_set(v___x_503_, 1, v_fst_500_);
lean_ctor_set(v___x_503_, 0, v_fst_489_);
v___x_509_ = v___x_503_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_fst_489_);
lean_ctor_set(v_reuseFailAlloc_522_, 1, v_fst_500_);
v___x_509_ = v_reuseFailAlloc_522_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
lean_object* v___x_511_; 
if (v_isShared_493_ == 0)
{
lean_ctor_set(v___x_492_, 1, v___x_509_);
lean_ctor_set(v___x_492_, 0, v_fst_481_);
v___x_511_ = v___x_492_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v_fst_481_);
lean_ctor_set(v_reuseFailAlloc_521_, 1, v___x_509_);
v___x_511_ = v_reuseFailAlloc_521_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
lean_object* v___x_513_; 
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 1, v___x_511_);
lean_ctor_set(v___x_484_, 0, v_fst_473_);
v___x_513_ = v___x_484_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v_fst_473_);
lean_ctor_set(v_reuseFailAlloc_520_, 1, v___x_511_);
v___x_513_ = v_reuseFailAlloc_520_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
lean_object* v___x_515_; 
if (v_isShared_477_ == 0)
{
lean_ctor_set(v___x_476_, 1, v___x_513_);
lean_ctor_set(v___x_476_, 0, v_fst_468_);
v___x_515_ = v___x_476_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_fst_468_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v___x_513_);
v___x_515_ = v_reuseFailAlloc_519_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
lean_object* v___x_517_; 
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 0, v___x_515_);
v___x_517_ = v___x_498_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v___x_515_);
v___x_517_ = v_reuseFailAlloc_518_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
return v___x_517_;
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
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs___boxed(lean_object* v_bytes_528_){
_start:
{
lean_object* v_res_529_; 
v_res_529_ = lp_workspace_VmVerifier_Spec_Wire_parseFiveBlobs(v_bytes_528_);
lean_dec_ref(v_bytes_528_);
return v_res_529_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dfv_Swirl_Spec_ReferenceVerifier_Wire_RawToTyped(uint8_t builtin);
lean_object* initialize_workspace_VmVerifier_Spec_Types(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_workspace_VmVerifier_Spec_Wire(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dfv_Swirl_Spec_ReferenceVerifier_Wire_RawToTyped(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_workspace_VmVerifier_Spec_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_workspace_VmVerifier_Spec_Wire_baselineMagic = _init_lp_workspace_VmVerifier_Spec_Wire_baselineMagic();
lean_mark_persistent(lp_workspace_VmVerifier_Spec_Wire_baselineMagic);
lp_workspace_VmVerifier_Spec_Wire_userPvsMagic = _init_lp_workspace_VmVerifier_Spec_Wire_userPvsMagic();
lean_mark_persistent(lp_workspace_VmVerifier_Spec_Wire_userPvsMagic);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
