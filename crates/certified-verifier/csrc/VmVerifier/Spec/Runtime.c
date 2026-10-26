// Lean compiler output
// Module: VmVerifier.Spec.Runtime
// Imports: public import Init public meta import Init public import Fundamentals.Spec.Poseidon2.Raw public import Swirl.Spec.ReferenceVerifier.Runtime.RawInstances public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Main public import Swirl.Spec.ReferenceVerifier.VerifierBabyBearPoseidon2 public import VmVerifier.Spec.Wire
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
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_compressDigest(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_merkleVerify(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Array_instDecidableEqImpl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofBase(lean_object*);
uint32_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_exitCode(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_parseVmProofData_x3f(lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(lean_object*);
uint8_t lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_decodeAll(lean_object*, lean_object*, lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_Spec_Wire_readBaseline(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
extern lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_RawInstances_rawFieldOpsEF;
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_hashSlice(lean_object*);
extern lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_init___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__0;
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawZero;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawOne;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest___lam__0___boxed(lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VmVerifier_rawZeroDigest___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__0_value;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__1;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawLeafDigest(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawMerkleNode(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawMerkleNode___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_rawPublicValuesMerkleRoot_x3f_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instDecidableEqFin___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1))} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___closed__0_value;
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___closed__0;
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_verifyVmProofData(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmProofData___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofPvs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofPvs___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError_default;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError;
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_hashSlice, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__0_value;
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_compressDigest, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__1_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig(lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_BabyBearExt4_Raw_ofBase, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__0_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__1_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner(lean_object*, lean_object*);
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__0_value;
static const lean_ctor_object lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__1_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof(lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1;
static lean_once_cell_t lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_rawZero(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(0u);
return v___x_1_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_rawOne(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(1u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest___lam__0(lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(0u);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawZeroDigest___lam__0___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_openvm_x2dfv_VmVerifier_rawZeroDigest___lam__0(v_x_5_);
lean_dec(v_x_5_);
return v_res_6_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__1(void){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_8_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__0));
v___x_9_ = lean_unsigned_to_nat(8u);
v___x_10_ = l_Array_ofFn___redArg(v___x_9_, v___f_8_);
return v___x_10_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_rawZeroDigest(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__1, &lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VmVerifier_rawZeroDigest___closed__1);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawLeafDigest(lean_object* v_values_12_){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lp_openvm_x2dfv_VmVerifier_rawZeroDigest;
v___x_14_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_compressDigest(v_values_12_, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawMerkleNode(lean_object* v_x_15_, lean_object* v_x_16_, lean_object* v_x_17_){
_start:
{
lean_object* v_zero_18_; uint8_t v_isZero_19_; 
v_zero_18_ = lean_unsigned_to_nat(0u);
v_isZero_19_ = lean_nat_dec_eq(v_x_15_, v_zero_18_);
if (v_isZero_19_ == 1)
{
lean_object* v___x_20_; 
v___x_20_ = lean_apply_1(v_x_16_, v_x_17_);
return v___x_20_;
}
else
{
lean_object* v_one_21_; lean_object* v_n_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v_one_21_ = lean_unsigned_to_nat(1u);
v_n_22_ = lean_nat_sub(v_x_15_, v_one_21_);
v___x_23_ = lean_unsigned_to_nat(2u);
v___x_24_ = lean_nat_mul(v___x_23_, v_x_17_);
lean_dec(v_x_17_);
lean_inc(v___x_24_);
lean_inc_ref(v_x_16_);
v___x_25_ = lp_openvm_x2dfv_VmVerifier_rawMerkleNode(v_n_22_, v_x_16_, v___x_24_);
v___x_26_ = lean_nat_add(v___x_24_, v_one_21_);
lean_dec(v___x_24_);
v___x_27_ = lp_openvm_x2dfv_VmVerifier_rawMerkleNode(v_n_22_, v_x_16_, v___x_26_);
lean_dec(v_n_22_);
v___x_28_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_compressDigest(v___x_25_, v___x_27_);
return v___x_28_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawMerkleNode___boxed(lean_object* v_x_29_, lean_object* v_x_30_, lean_object* v_x_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_openvm_x2dfv_VmVerifier_rawMerkleNode(v_x_29_, v_x_30_, v_x_31_);
lean_dec(v_x_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg(lean_object* v_chunkSize_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_){
_start:
{
if (lean_obj_tag(v_a_34_) == 0)
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = lean_array_to_list(v_a_35_);
v___x_38_ = lean_array_push(v_a_36_, v___x_37_);
v___x_39_ = lean_array_to_list(v___x_38_);
return v___x_39_;
}
else
{
lean_object* v_head_40_; lean_object* v_tail_41_; lean_object* v___x_42_; uint8_t v___x_43_; 
v_head_40_ = lean_ctor_get(v_a_34_, 0);
lean_inc(v_head_40_);
v_tail_41_ = lean_ctor_get(v_a_34_, 1);
lean_inc(v_tail_41_);
lean_dec_ref_known(v_a_34_, 2);
v___x_42_ = lean_array_get_size(v_a_35_);
v___x_43_ = lean_nat_dec_eq(v___x_42_, v_chunkSize_33_);
if (v___x_43_ == 0)
{
lean_object* v___x_44_; 
v___x_44_ = lean_array_push(v_a_35_, v_head_40_);
v_a_34_ = v_tail_41_;
v_a_35_ = v___x_44_;
goto _start;
}
else
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_46_ = lean_mk_empty_array_with_capacity(v_chunkSize_33_);
v___x_47_ = lean_array_push(v___x_46_, v_head_40_);
v___x_48_ = lean_array_to_list(v_a_35_);
v___x_49_ = lean_array_push(v_a_36_, v___x_48_);
v_a_34_ = v_tail_41_;
v_a_35_ = v___x_47_;
v_a_36_ = v___x_49_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg___boxed(lean_object* v_chunkSize_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg(v_chunkSize_51_, v_a_52_, v_a_53_, v_a_54_);
lean_dec(v_chunkSize_51_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go(lean_object* v_00_u03b1_56_, lean_object* v_chunkSize_57_, lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg(v_chunkSize_57_, v_a_58_, v_a_59_, v_a_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks_go___boxed(lean_object* v_00_u03b1_62_, lean_object* v_chunkSize_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_openvm_x2dfv_VmVerifier_rawToChunks_go(v_00_u03b1_62_, v_chunkSize_63_, v_a_64_, v_a_65_, v_a_66_);
lean_dec(v_chunkSize_63_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg(lean_object* v_x_70_, lean_object* v_x_71_){
_start:
{
if (lean_obj_tag(v_x_71_) == 0)
{
lean_object* v___x_72_; 
v___x_72_ = lean_box(0);
return v___x_72_;
}
else
{
lean_object* v_head_73_; lean_object* v_tail_74_; lean_object* v___x_75_; uint8_t v___x_76_; 
v_head_73_ = lean_ctor_get(v_x_71_, 0);
v_tail_74_ = lean_ctor_get(v_x_71_, 1);
v___x_75_ = lean_unsigned_to_nat(0u);
v___x_76_ = lean_nat_dec_eq(v_x_70_, v___x_75_);
if (v___x_76_ == 0)
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
lean_inc(v_tail_74_);
lean_inc(v_head_73_);
lean_dec_ref_known(v_x_71_, 2);
v___x_77_ = lean_unsigned_to_nat(1u);
v___x_78_ = lean_mk_empty_array_with_capacity(v___x_77_);
v___x_79_ = lean_array_push(v___x_78_, v_head_73_);
v___x_80_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg___closed__0));
v___x_81_ = lp_openvm_x2dfv_VmVerifier_rawToChunks_go___redArg(v_x_70_, v_tail_74_, v___x_79_, v___x_80_);
return v___x_81_;
}
else
{
lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_82_ = lean_box(0);
v___x_83_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_83_, 0, v_x_71_);
lean_ctor_set(v___x_83_, 1, v___x_82_);
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg___boxed(lean_object* v_x_84_, lean_object* v_x_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg(v_x_84_, v_x_85_);
lean_dec(v_x_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks(lean_object* v_00_u03b1_87_, lean_object* v_x_88_, lean_object* v_x_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg(v_x_88_, v_x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawToChunks___boxed(lean_object* v_00_u03b1_91_, lean_object* v_x_92_, lean_object* v_x_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_openvm_x2dfv_VmVerifier_rawToChunks(v_00_u03b1_91_, v_x_92_, v_x_93_);
lean_dec(v_x_92_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0(lean_object* v___x_95_, lean_object* v_index_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = l_List_get_x3fInternal___redArg(v___x_95_, v_index_96_);
if (lean_obj_tag(v___x_97_) == 0)
{
lean_object* v___x_98_; 
v___x_98_ = lp_openvm_x2dfv_VmVerifier_rawZeroDigest;
return v___x_98_;
}
else
{
lean_object* v_val_99_; 
v_val_99_ = lean_ctor_get(v___x_97_, 0);
lean_inc(v_val_99_);
lean_dec_ref_known(v___x_97_, 1);
return v_val_99_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0___boxed(lean_object* v___x_100_, lean_object* v_index_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0(v___x_100_, v_index_101_);
lean_dec(v___x_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_rawPublicValuesMerkleRoot_x3f_spec__0(lean_object* v_a_103_, lean_object* v_a_104_){
_start:
{
if (lean_obj_tag(v_a_103_) == 0)
{
lean_object* v___x_105_; 
v___x_105_ = l_List_reverse___redArg(v_a_104_);
return v___x_105_;
}
else
{
lean_object* v_head_106_; lean_object* v_tail_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_118_; 
v_head_106_ = lean_ctor_get(v_a_103_, 0);
v_tail_107_ = lean_ctor_get(v_a_103_, 1);
v_isSharedCheck_118_ = !lean_is_exclusive(v_a_103_);
if (v_isSharedCheck_118_ == 0)
{
v___x_109_ = v_a_103_;
v_isShared_110_ = v_isSharedCheck_118_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_tail_107_);
lean_inc(v_head_106_);
lean_dec(v_a_103_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_118_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_115_; 
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = lp_openvm_x2dfv_Recursion_Spec_digestPrefixOf___redArg(v___x_111_, v_head_106_);
v___x_113_ = lp_openvm_x2dfv_VmVerifier_rawLeafDigest(v___x_112_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 1, v_a_104_);
lean_ctor_set(v___x_109_, 0, v___x_113_);
v___x_115_ = v___x_109_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_113_);
lean_ctor_set(v_reuseFailAlloc_117_, 1, v_a_104_);
v___x_115_ = v_reuseFailAlloc_117_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
v_a_103_ = v_tail_107_;
v_a_104_ = v___x_115_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f(lean_object* v_values_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(v_values_119_);
if (lean_obj_tag(v___x_120_) == 0)
{
lean_object* v___x_121_; 
lean_dec(v_values_119_);
v___x_121_ = lean_box(0);
return v___x_121_;
}
else
{
lean_object* v_val_122_; lean_object* v___x_124_; uint8_t v_isShared_125_; uint8_t v_isSharedCheck_136_; 
v_val_122_ = lean_ctor_get(v___x_120_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_136_ == 0)
{
v___x_124_ = v___x_120_;
v_isShared_125_ = v_isSharedCheck_136_;
goto v_resetjp_123_;
}
else
{
lean_inc(v_val_122_);
lean_dec(v___x_120_);
v___x_124_ = lean_box(0);
v_isShared_125_ = v_isSharedCheck_136_;
goto v_resetjp_123_;
}
v_resetjp_123_:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___f_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_134_; 
v___x_126_ = lean_unsigned_to_nat(8u);
v___x_127_ = lp_openvm_x2dfv_VmVerifier_rawToChunks___redArg(v___x_126_, v_values_119_);
v___x_128_ = lean_box(0);
v___x_129_ = lp_openvm_x2dfv_List_mapTR_loop___at___00VmVerifier_rawPublicValuesMerkleRoot_x3f_spec__0(v___x_127_, v___x_128_);
v___f_130_ = lean_alloc_closure((void*)(lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0___boxed), 2, 1);
lean_closure_set(v___f_130_, 0, v___x_129_);
v___x_131_ = lean_unsigned_to_nat(0u);
v___x_132_ = lp_openvm_x2dfv_VmVerifier_rawMerkleNode(v_val_122_, v___f_130_, v___x_131_);
lean_dec(v_val_122_);
if (v_isShared_125_ == 0)
{
lean_ctor_set(v___x_124_, 0, v___x_132_);
v___x_134_ = v___x_124_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___x_132_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter___redArg(lean_object* v_x_137_, lean_object* v_h__1_138_, lean_object* v_h__2_139_){
_start:
{
if (lean_obj_tag(v_x_137_) == 0)
{
lean_object* v___x_140_; lean_object* v___x_141_; 
lean_dec(v_h__2_139_);
v___x_140_ = lean_box(0);
v___x_141_ = lean_apply_1(v_h__1_138_, v___x_140_);
return v___x_141_;
}
else
{
lean_object* v_val_142_; lean_object* v___x_143_; 
lean_dec(v_h__1_138_);
v_val_142_ = lean_ctor_get(v_x_137_, 0);
lean_inc(v_val_142_);
lean_dec_ref_known(v_x_137_, 1);
v___x_143_ = lean_apply_1(v_h__2_139_, v_val_142_);
return v___x_143_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter(lean_object* v_motive_144_, lean_object* v_x_145_, lean_object* v_h__1_146_, lean_object* v_h__2_147_){
_start:
{
if (lean_obj_tag(v_x_145_) == 0)
{
lean_object* v___x_148_; lean_object* v___x_149_; 
lean_dec(v_h__2_147_);
v___x_148_ = lean_box(0);
v___x_149_ = lean_apply_1(v_h__1_146_, v___x_148_);
return v___x_149_;
}
else
{
lean_object* v_val_150_; lean_object* v___x_151_; 
lean_dec(v_h__1_146_);
v_val_150_ = lean_ctor_get(v_x_145_, 0);
lean_inc(v_val_150_);
lean_dec_ref_known(v_x_145_, 1);
v___x_151_ = lean_apply_1(v_h__2_147_, v_val_150_);
return v___x_151_;
}
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0(lean_object* v_a_154_, lean_object* v_b_155_){
_start:
{
lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_156_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0));
v___x_157_ = l_Array_instDecidableEqImpl___redArg(v___x_156_, v_a_154_, v_b_155_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___boxed(lean_object* v_a_158_, lean_object* v_b_159_){
_start:
{
uint8_t v_res_160_; lean_object* v_r_161_; 
v_res_160_ = lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0(v_a_158_, v_b_159_);
lean_dec_ref(v_b_159_);
lean_dec_ref(v_a_158_);
v_r_161_ = lean_box(v_res_160_);
return v_r_161_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid(lean_object* v_userProof_163_, lean_object* v_dimensions_164_, lean_object* v_finalRoot_165_){
_start:
{
lean_object* v_authenticationPath_166_; lean_object* v_publicValues_167_; lean_object* v_publicValuesCommit_168_; lean_object* v___x_169_; 
v_authenticationPath_166_ = lean_ctor_get(v_userProof_163_, 0);
lean_inc(v_authenticationPath_166_);
v_publicValues_167_ = lean_ctor_get(v_userProof_163_, 1);
lean_inc(v_publicValues_167_);
v_publicValuesCommit_168_ = lean_ctor_get(v_userProof_163_, 2);
lean_inc_ref(v_publicValuesCommit_168_);
lean_dec_ref(v_userProof_163_);
v___x_169_ = lp_openvm_x2dfv_VmVerifier_publicValuesHeight_x3f(v_publicValues_167_);
if (lean_obj_tag(v___x_169_) == 0)
{
uint8_t v___x_170_; 
lean_dec_ref(v_publicValuesCommit_168_);
lean_dec(v_publicValues_167_);
lean_dec(v_authenticationPath_166_);
lean_dec_ref(v_finalRoot_165_);
v___x_170_ = 0;
return v___x_170_;
}
else
{
lean_object* v_val_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_193_; 
v_val_171_ = lean_ctor_get(v___x_169_, 0);
v_isSharedCheck_193_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_193_ == 0)
{
v___x_173_ = v___x_169_;
v_isShared_174_ = v_isSharedCheck_193_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_val_171_);
lean_dec(v___x_169_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_193_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_175_; uint8_t v___x_176_; 
v___x_175_ = lp_openvm_x2dfv_VmVerifier_MemoryDimensions_overallHeight(v_dimensions_164_);
v___x_176_ = lean_nat_dec_le(v_val_171_, v___x_175_);
if (v___x_176_ == 0)
{
lean_dec(v___x_175_);
lean_del_object(v___x_173_);
lean_dec(v_val_171_);
lean_dec_ref(v_publicValuesCommit_168_);
lean_dec(v_publicValues_167_);
lean_dec(v_authenticationPath_166_);
lean_dec_ref(v_finalRoot_165_);
return v___x_176_;
}
else
{
lean_object* v___x_177_; lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_177_ = l_List_lengthTR___redArg(v_authenticationPath_166_);
v___x_178_ = lean_nat_sub(v___x_175_, v_val_171_);
lean_dec(v___x_175_);
v___x_179_ = lean_nat_dec_eq(v___x_177_, v___x_178_);
lean_dec(v___x_178_);
lean_dec(v___x_177_);
if (v___x_179_ == 0)
{
lean_del_object(v___x_173_);
lean_dec(v_val_171_);
lean_dec_ref(v_publicValuesCommit_168_);
lean_dec(v_publicValues_167_);
lean_dec(v_authenticationPath_166_);
lean_dec_ref(v_finalRoot_165_);
return v___x_179_;
}
else
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; uint8_t v___x_186_; 
v___x_180_ = lean_unsigned_to_nat(3u);
v___x_181_ = lean_unsigned_to_nat(0u);
v___x_182_ = lp_openvm_x2dfv_VmVerifier_MemoryDimensions_labelToIndex(v_dimensions_164_, v___x_180_, v___x_181_);
v___x_183_ = lean_unsigned_to_nat(2u);
v___x_184_ = lean_nat_pow(v___x_183_, v_val_171_);
lean_dec(v_val_171_);
v___x_185_ = lean_nat_div(v___x_182_, v___x_184_);
lean_dec(v___x_184_);
lean_dec(v___x_182_);
lean_inc_ref(v_publicValuesCommit_168_);
v___x_186_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Raw_merkleVerify(v_finalRoot_165_, v___x_185_, v_publicValuesCommit_168_, v_authenticationPath_166_);
if (v___x_186_ == 0)
{
lean_del_object(v___x_173_);
lean_dec_ref(v_publicValuesCommit_168_);
lean_dec(v_publicValues_167_);
return v___x_186_;
}
else
{
lean_object* v___f_187_; lean_object* v___x_188_; lean_object* v___x_190_; 
v___f_187_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___closed__0));
v___x_188_ = lp_openvm_x2dfv_VmVerifier_rawPublicValuesMerkleRoot_x3f(v_publicValues_167_);
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 0, v_publicValuesCommit_168_);
v___x_190_ = v___x_173_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v_publicValuesCommit_168_);
v___x_190_ = v_reuseFailAlloc_192_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
uint8_t v___x_191_; 
v___x_191_ = l_Option_instDecidableEq___redArg(v___f_187_, v___x_188_, v___x_190_);
return v___x_191_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___boxed(lean_object* v_userProof_194_, lean_object* v_dimensions_195_, lean_object* v_finalRoot_196_){
_start:
{
uint8_t v_res_197_; lean_object* v_r_198_; 
v_res_197_ = lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid(v_userProof_194_, v_dimensions_195_, v_finalRoot_196_);
lean_dec_ref(v_dimensions_195_);
v_r_198_ = lean_box(v_res_197_);
return v_r_198_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___closed__0(void){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = lean_unsigned_to_nat(2u);
v___x_200_ = lp_swirl_x2dfv_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_199_);
return v___x_200_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid(lean_object* v_vmPublicValues_201_, lean_object* v_userPublicValuesProof_202_, lean_object* v_baseline_203_){
_start:
{
lean_object* v_toAggregationPublicValues_204_; lean_object* v_vm_205_; lean_object* v_memory_206_; lean_object* v_programCommit_207_; lean_object* v_initialState_208_; lean_object* v_initialPc_209_; lean_object* v_memoryDimensions_210_; lean_object* v_numUserPvs_211_; lean_object* v_appVkCommit_212_; lean_object* v_leafVkCommit_213_; lean_object* v_internalForLeafVkCommit_214_; lean_object* v_internalRecursiveVkCommit_215_; lean_object* v_constraintEvalCachedCommit_216_; lean_object* v_toAggregationBasePublicValues_217_; lean_object* v_programCommit_218_; lean_object* v_connector_219_; lean_object* v_initialRoot_220_; lean_object* v_finalRoot_221_; uint8_t v___x_222_; 
v_toAggregationPublicValues_204_ = lean_ctor_get(v_vmPublicValues_201_, 0);
lean_inc_ref(v_toAggregationPublicValues_204_);
v_vm_205_ = lean_ctor_get(v_toAggregationPublicValues_204_, 1);
lean_inc_ref(v_vm_205_);
v_memory_206_ = lean_ctor_get(v_vm_205_, 2);
lean_inc_ref(v_memory_206_);
v_programCommit_207_ = lean_ctor_get(v_baseline_203_, 0);
v_initialState_208_ = lean_ctor_get(v_baseline_203_, 1);
v_initialPc_209_ = lean_ctor_get(v_baseline_203_, 2);
v_memoryDimensions_210_ = lean_ctor_get(v_baseline_203_, 3);
v_numUserPvs_211_ = lean_ctor_get(v_baseline_203_, 4);
v_appVkCommit_212_ = lean_ctor_get(v_baseline_203_, 5);
v_leafVkCommit_213_ = lean_ctor_get(v_baseline_203_, 6);
v_internalForLeafVkCommit_214_ = lean_ctor_get(v_baseline_203_, 7);
v_internalRecursiveVkCommit_215_ = lean_ctor_get(v_baseline_203_, 8);
v_constraintEvalCachedCommit_216_ = lean_ctor_get(v_vmPublicValues_201_, 1);
lean_inc_ref(v_constraintEvalCachedCommit_216_);
lean_dec_ref(v_vmPublicValues_201_);
v_toAggregationBasePublicValues_217_ = lean_ctor_get(v_toAggregationPublicValues_204_, 0);
lean_inc_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_toAggregationPublicValues_204_);
v_programCommit_218_ = lean_ctor_get(v_vm_205_, 0);
lean_inc_ref(v_programCommit_218_);
v_connector_219_ = lean_ctor_get(v_vm_205_, 1);
lean_inc_ref(v_connector_219_);
lean_dec_ref(v_vm_205_);
v_initialRoot_220_ = lean_ctor_get(v_memory_206_, 0);
lean_inc_ref(v_initialRoot_220_);
v_finalRoot_221_ = lean_ctor_get(v_memory_206_, 1);
lean_inc_ref(v_finalRoot_221_);
lean_dec_ref(v_memory_206_);
lean_inc_ref(v_userPublicValuesProof_202_);
v___x_222_ = lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid(v_userPublicValuesProof_202_, v_memoryDimensions_210_, v_finalRoot_221_);
if (v___x_222_ == 0)
{
lean_dec_ref(v_initialRoot_220_);
lean_dec_ref(v_connector_219_);
lean_dec_ref(v_programCommit_218_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
lean_dec_ref(v_userPublicValuesProof_202_);
return v___x_222_;
}
else
{
lean_object* v_publicValues_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v_publicValues_223_ = lean_ctor_get(v_userPublicValuesProof_202_, 1);
lean_inc(v_publicValues_223_);
lean_dec_ref(v_userPublicValuesProof_202_);
v___x_224_ = l_List_lengthTR___redArg(v_publicValues_223_);
lean_dec(v_publicValues_223_);
v___x_225_ = lean_nat_dec_eq(v___x_224_, v_numUserPvs_211_);
lean_dec(v___x_224_);
if (v___x_225_ == 0)
{
lean_dec_ref(v_initialRoot_220_);
lean_dec_ref(v_connector_219_);
lean_dec_ref(v_programCommit_218_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_225_;
}
else
{
lean_object* v___x_226_; uint8_t v___x_227_; 
v___x_226_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0));
v___x_227_ = l_Array_instDecidableEqImpl___redArg(v___x_226_, v_programCommit_218_, v_programCommit_207_);
lean_dec_ref(v_programCommit_218_);
if (v___x_227_ == 0)
{
lean_dec_ref(v_initialRoot_220_);
lean_dec_ref(v_connector_219_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_227_;
}
else
{
uint8_t v___x_228_; 
v___x_228_ = l_Array_instDecidableEqImpl___redArg(v___x_226_, v_initialRoot_220_, v_initialState_208_);
lean_dec_ref(v_initialRoot_220_);
if (v___x_228_ == 0)
{
lean_dec_ref(v_connector_219_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_228_;
}
else
{
lean_object* v_initialPc_229_; lean_object* v_exitCode_230_; lean_object* v_isTerminate_231_; uint8_t v___x_232_; 
v_initialPc_229_ = lean_ctor_get(v_connector_219_, 0);
lean_inc(v_initialPc_229_);
v_exitCode_230_ = lean_ctor_get(v_connector_219_, 2);
lean_inc(v_exitCode_230_);
v_isTerminate_231_ = lean_ctor_get(v_connector_219_, 3);
lean_inc(v_isTerminate_231_);
lean_dec_ref(v_connector_219_);
v___x_232_ = lean_nat_dec_eq(v_initialPc_229_, v_initialPc_209_);
lean_dec(v_initialPc_229_);
if (v___x_232_ == 0)
{
lean_dec(v_isTerminate_231_);
lean_dec(v_exitCode_230_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_232_;
}
else
{
lean_object* v___x_233_; uint8_t v___x_234_; 
v___x_233_ = lean_unsigned_to_nat(0u);
v___x_234_ = lean_nat_dec_eq(v_exitCode_230_, v___x_233_);
lean_dec(v_exitCode_230_);
if (v___x_234_ == 0)
{
lean_dec(v_isTerminate_231_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_234_;
}
else
{
lean_object* v___x_235_; uint8_t v___x_236_; 
v___x_235_ = lean_unsigned_to_nat(1u);
v___x_236_ = lean_nat_dec_eq(v_isTerminate_231_, v___x_235_);
lean_dec(v_isTerminate_231_);
if (v___x_236_ == 0)
{
lean_dec_ref(v_toAggregationBasePublicValues_217_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_236_;
}
else
{
lean_object* v_internalFlag_237_; lean_object* v_appVkCommit_238_; lean_object* v_leafVkCommit_239_; lean_object* v_internalForLeafVkCommit_240_; lean_object* v_recursionDepth_241_; lean_object* v_internalRecursiveVkCommit_242_; lean_object* v___x_243_; uint8_t v___x_244_; 
v_internalFlag_237_ = lean_ctor_get(v_toAggregationBasePublicValues_217_, 0);
lean_inc(v_internalFlag_237_);
v_appVkCommit_238_ = lean_ctor_get(v_toAggregationBasePublicValues_217_, 1);
lean_inc_ref(v_appVkCommit_238_);
v_leafVkCommit_239_ = lean_ctor_get(v_toAggregationBasePublicValues_217_, 2);
lean_inc_ref(v_leafVkCommit_239_);
v_internalForLeafVkCommit_240_ = lean_ctor_get(v_toAggregationBasePublicValues_217_, 3);
lean_inc_ref(v_internalForLeafVkCommit_240_);
v_recursionDepth_241_ = lean_ctor_get(v_toAggregationBasePublicValues_217_, 4);
lean_inc(v_recursionDepth_241_);
v_internalRecursiveVkCommit_242_ = lean_ctor_get(v_toAggregationBasePublicValues_217_, 5);
lean_inc_ref(v_internalRecursiveVkCommit_242_);
lean_dec_ref(v_toAggregationBasePublicValues_217_);
v___x_243_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___closed__0, &lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___closed__0_once, _init_lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___closed__0);
v___x_244_ = lean_nat_dec_eq(v_internalFlag_237_, v___x_243_);
lean_dec(v_internalFlag_237_);
if (v___x_244_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_242_);
lean_dec(v_recursionDepth_241_);
lean_dec_ref(v_internalForLeafVkCommit_240_);
lean_dec_ref(v_leafVkCommit_239_);
lean_dec_ref(v_appVkCommit_238_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_244_;
}
else
{
uint8_t v___x_245_; 
v___x_245_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_226_, v_appVkCommit_238_, v_appVkCommit_212_);
lean_dec_ref(v_appVkCommit_238_);
if (v___x_245_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_242_);
lean_dec(v_recursionDepth_241_);
lean_dec_ref(v_internalForLeafVkCommit_240_);
lean_dec_ref(v_leafVkCommit_239_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_245_;
}
else
{
uint8_t v___x_246_; 
v___x_246_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_226_, v_leafVkCommit_239_, v_leafVkCommit_213_);
lean_dec_ref(v_leafVkCommit_239_);
if (v___x_246_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_242_);
lean_dec(v_recursionDepth_241_);
lean_dec_ref(v_internalForLeafVkCommit_240_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_246_;
}
else
{
uint8_t v___x_247_; 
v___x_247_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_226_, v_internalForLeafVkCommit_240_, v_internalForLeafVkCommit_214_);
lean_dec_ref(v_internalForLeafVkCommit_240_);
if (v___x_247_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_242_);
lean_dec(v_recursionDepth_241_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_247_;
}
else
{
uint8_t v___x_248_; 
v___x_248_ = lean_nat_dec_lt(v___x_233_, v_recursionDepth_241_);
if (v___x_248_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_242_);
lean_dec(v_recursionDepth_241_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_248_;
}
else
{
lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_249_ = lean_unsigned_to_nat(256u);
v___x_250_ = lean_nat_dec_le(v_recursionDepth_241_, v___x_249_);
if (v___x_250_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_242_);
lean_dec(v_recursionDepth_241_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_250_;
}
else
{
uint8_t v___x_251_; 
v___x_251_ = lean_nat_dec_eq(v_recursionDepth_241_, v___x_235_);
lean_dec(v_recursionDepth_241_);
if (v___x_251_ == 0)
{
uint8_t v___x_252_; 
v___x_252_ = lp_openvm_x2dfv_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_226_, v_internalRecursiveVkCommit_242_, v_internalRecursiveVkCommit_215_);
lean_dec_ref(v_internalRecursiveVkCommit_242_);
if (v___x_252_ == 0)
{
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_252_;
}
else
{
lean_object* v_cachedCommit_253_; uint8_t v___x_254_; 
v_cachedCommit_253_ = lean_ctor_get(v_internalRecursiveVkCommit_215_, 0);
v___x_254_ = l_Array_instDecidableEqImpl___redArg(v___x_226_, v_constraintEvalCachedCommit_216_, v_cachedCommit_253_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_254_;
}
}
else
{
lean_object* v_cachedCommit_255_; lean_object* v_vkPreHash_256_; lean_object* v___x_257_; uint8_t v___x_258_; 
v_cachedCommit_255_ = lean_ctor_get(v_internalRecursiveVkCommit_242_, 0);
lean_inc_ref(v_cachedCommit_255_);
v_vkPreHash_256_ = lean_ctor_get(v_internalRecursiveVkCommit_242_, 1);
lean_inc_ref(v_vkPreHash_256_);
lean_dec_ref(v_internalRecursiveVkCommit_242_);
v___x_257_ = lp_openvm_x2dfv_VmVerifier_rawZeroDigest;
v___x_258_ = l_Array_instDecidableEqImpl___redArg(v___x_226_, v_cachedCommit_255_, v___x_257_);
lean_dec_ref(v_cachedCommit_255_);
if (v___x_258_ == 0)
{
lean_dec_ref(v_vkPreHash_256_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_258_;
}
else
{
uint8_t v___x_259_; 
v___x_259_ = l_Array_instDecidableEqImpl___redArg(v___x_226_, v_vkPreHash_256_, v___x_257_);
lean_dec_ref(v_vkPreHash_256_);
if (v___x_259_ == 0)
{
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_259_;
}
else
{
lean_object* v_cachedCommit_260_; uint8_t v___x_261_; 
v_cachedCommit_260_ = lean_ctor_get(v_internalForLeafVkCommit_214_, 0);
v___x_261_ = l_Array_instDecidableEqImpl___redArg(v___x_226_, v_constraintEvalCachedCommit_216_, v_cachedCommit_260_);
lean_dec_ref(v_constraintEvalCachedCommit_216_);
return v___x_261_;
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
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid___boxed(lean_object* v_vmPublicValues_262_, lean_object* v_userPublicValuesProof_263_, lean_object* v_baseline_264_){
_start:
{
uint8_t v_res_265_; lean_object* v_r_266_; 
v_res_265_ = lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid(v_vmPublicValues_262_, v_userPublicValuesProof_263_, v_baseline_264_);
lean_dec_ref(v_baseline_264_);
v_r_266_ = lean_box(v_res_265_);
return v_r_266_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_verifyVmProofData(lean_object* v_vmPublicValues_267_, lean_object* v_userPublicValuesProof_268_, lean_object* v_baseline_269_){
_start:
{
uint8_t v___x_270_; 
v___x_270_ = lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid(v_vmPublicValues_267_, v_userPublicValuesProof_268_, v_baseline_269_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmProofData___boxed(lean_object* v_vmPublicValues_271_, lean_object* v_userPublicValuesProof_272_, lean_object* v_baseline_273_){
_start:
{
uint8_t v_res_274_; lean_object* v_r_275_; 
v_res_274_ = lp_openvm_x2dfv_VmVerifier_verifyVmProofData(v_vmPublicValues_271_, v_userPublicValuesProof_272_, v_baseline_273_);
lean_dec_ref(v_baseline_273_);
v_r_275_ = lean_box(v_res_274_);
return v_r_275_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofPvs(lean_object* v_vk_276_, lean_object* v_proof_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_openvm_x2dfv_VmVerifier_parseVmProofData_x3f(v_proof_277_);
if (lean_obj_tag(v___x_278_) == 0)
{
uint8_t v___x_279_; 
v___x_279_ = 0;
return v___x_279_;
}
else
{
lean_object* v_val_280_; lean_object* v_fst_281_; lean_object* v_snd_282_; lean_object* v_baseline_283_; uint8_t v___x_284_; 
v_val_280_ = lean_ctor_get(v___x_278_, 0);
lean_inc(v_val_280_);
lean_dec_ref_known(v___x_278_, 1);
v_fst_281_ = lean_ctor_get(v_val_280_, 0);
lean_inc(v_fst_281_);
v_snd_282_ = lean_ctor_get(v_val_280_, 1);
lean_inc(v_snd_282_);
lean_dec(v_val_280_);
v_baseline_283_ = lean_ctor_get(v_vk_276_, 1);
v___x_284_ = lp_openvm_x2dfv_VmVerifier_decidableVmProofDataRawValid(v_fst_281_, v_snd_282_, v_baseline_283_);
return v___x_284_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofPvs___boxed(lean_object* v_vk_285_, lean_object* v_proof_286_){
_start:
{
uint8_t v_res_287_; lean_object* v_r_288_; 
v_res_287_ = lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofPvs(v_vk_285_, v_proof_286_);
lean_dec_ref(v_vk_285_);
v_r_288_ = lean_box(v_res_287_);
return v_r_288_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorIdx(lean_object* v_x_289_){
_start:
{
if (lean_obj_tag(v_x_289_) == 0)
{
lean_object* v___x_290_; 
v___x_290_ = lean_unsigned_to_nat(0u);
return v___x_290_;
}
else
{
lean_object* v___x_291_; 
v___x_291_ = lean_unsigned_to_nat(1u);
return v___x_291_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorIdx___boxed(lean_object* v_x_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorIdx(v_x_292_);
lean_dec(v_x_292_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(lean_object* v_t_294_, lean_object* v_k_295_){
_start:
{
if (lean_obj_tag(v_t_294_) == 0)
{
uint8_t v_error_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v_error_296_ = lean_ctor_get_uint8(v_t_294_, 0);
v___x_297_ = lean_box(v_error_296_);
v___x_298_ = lean_apply_1(v_k_295_, v___x_297_);
return v___x_298_;
}
else
{
return v_k_295_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg___boxed(lean_object* v_t_299_, lean_object* v_k_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_299_, v_k_300_);
lean_dec(v_t_299_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim(lean_object* v_motive_302_, lean_object* v_ctorIdx_303_, lean_object* v_t_304_, lean_object* v_h_305_, lean_object* v_k_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_304_, v_k_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___boxed(lean_object* v_motive_308_, lean_object* v_ctorIdx_309_, lean_object* v_t_310_, lean_object* v_h_311_, lean_object* v_k_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim(v_motive_308_, v_ctorIdx_309_, v_t_310_, v_h_311_, v_k_312_);
lean_dec(v_t_310_);
lean_dec(v_ctorIdx_309_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___redArg(lean_object* v_t_314_, lean_object* v_stark_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_314_, v_stark_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___redArg___boxed(lean_object* v_t_317_, lean_object* v_stark_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___redArg(v_t_317_, v_stark_318_);
lean_dec(v_t_317_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim(lean_object* v_motive_320_, lean_object* v_t_321_, lean_object* v_h_322_, lean_object* v_stark_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_321_, v_stark_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim___boxed(lean_object* v_motive_325_, lean_object* v_t_326_, lean_object* v_h_327_, lean_object* v_stark_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_stark_elim(v_motive_325_, v_t_326_, v_h_327_, v_stark_328_);
lean_dec(v_t_326_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___redArg(lean_object* v_t_330_, lean_object* v_publicValues_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_330_, v_publicValues_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___redArg___boxed(lean_object* v_t_333_, lean_object* v_publicValues_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___redArg(v_t_333_, v_publicValues_334_);
lean_dec(v_t_333_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim(lean_object* v_motive_336_, lean_object* v_t_337_, lean_object* v_h_338_, lean_object* v_publicValues_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_337_, v_publicValues_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim___boxed(lean_object* v_motive_341_, lean_object* v_t_342_, lean_object* v_h_343_, lean_object* v_publicValues_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_openvm_x2dfv_VmVerifier_VmStarkProofError_publicValues_elim(v_motive_341_, v_t_342_, v_h_343_, v_publicValues_344_);
lean_dec(v_t_342_);
return v_res_345_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError_default(void){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lean_box(1);
return v___x_346_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError(void){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lean_box(1);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig(lean_object* v_vk_350_){
_start:
{
lean_object* v_internalRecursiveVk_351_; lean_object* v_inner_352_; lean_object* v_params_353_; lean_object* v___x_355_; uint8_t v_isShared_356_; uint8_t v_isSharedCheck_362_; 
v_internalRecursiveVk_351_ = lean_ctor_get(v_vk_350_, 0);
lean_inc_ref(v_internalRecursiveVk_351_);
lean_dec_ref(v_vk_350_);
v_inner_352_ = lean_ctor_get(v_internalRecursiveVk_351_, 0);
lean_inc_ref(v_inner_352_);
lean_dec_ref(v_internalRecursiveVk_351_);
v_params_353_ = lean_ctor_get(v_inner_352_, 0);
v_isSharedCheck_362_ = !lean_is_exclusive(v_inner_352_);
if (v_isSharedCheck_362_ == 0)
{
lean_object* v_unused_363_; lean_object* v_unused_364_; 
v_unused_363_ = lean_ctor_get(v_inner_352_, 2);
lean_dec(v_unused_363_);
v_unused_364_ = lean_ctor_get(v_inner_352_, 1);
lean_dec(v_unused_364_);
v___x_355_ = v_inner_352_;
v_isShared_356_ = v_isSharedCheck_362_;
goto v_resetjp_354_;
}
else
{
lean_inc(v_params_353_);
lean_dec(v_inner_352_);
v___x_355_ = lean_box(0);
v_isShared_356_ = v_isSharedCheck_362_;
goto v_resetjp_354_;
}
v_resetjp_354_:
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_360_; 
v___x_357_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__0));
v___x_358_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig___closed__1));
if (v_isShared_356_ == 0)
{
lean_ctor_set(v___x_355_, 2, v___x_358_);
lean_ctor_set(v___x_355_, 1, v___x_357_);
v___x_360_ = v___x_355_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v_params_353_);
lean_ctor_set(v_reuseFailAlloc_361_, 1, v___x_357_);
lean_ctor_set(v_reuseFailAlloc_361_, 2, v___x_358_);
v___x_360_ = v_reuseFailAlloc_361_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
return v___x_360_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner(lean_object* v_vk_368_, lean_object* v_proof_369_){
_start:
{
lean_object* v_internalRecursiveVk_370_; lean_object* v_inner_371_; lean_object* v___f_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v_internalRecursiveVk_370_ = lean_ctor_get(v_vk_368_, 0);
lean_inc_ref(v_internalRecursiveVk_370_);
v_inner_371_ = lean_ctor_get(v_proof_369_, 0);
lean_inc_ref(v_inner_371_);
lean_dec_ref(v_proof_369_);
v___f_372_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__0));
v___x_373_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_RawInstances_rawFieldOpsEF;
v___x_374_ = lp_openvm_x2dfv_VmVerifier_vmStarkProtocolConfig(v_vk_368_);
v___x_375_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_init___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__0;
v___x_376_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__1(v___x_373_, v___f_372_, v___x_374_, v_internalRecursiveVk_370_, v_inner_371_, v___x_375_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_384_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_384_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_384_ == 0)
{
v___x_379_ = v___x_376_;
v_isShared_380_ = v_isSharedCheck_384_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_a_377_);
lean_dec(v___x_376_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_384_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_382_; 
if (v_isShared_380_ == 0)
{
v___x_382_ = v___x_379_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_383_; 
v_reuseFailAlloc_383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_383_, 0, v_a_377_);
v___x_382_ = v_reuseFailAlloc_383_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
return v___x_382_;
}
}
}
else
{
lean_object* v___x_385_; 
lean_dec_ref_known(v___x_376_, 1);
v___x_385_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner___closed__1));
return v___x_385_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof(lean_object* v_vk_390_, lean_object* v_proof_391_){
_start:
{
lean_object* v___x_392_; 
lean_inc_ref(v_proof_391_);
lean_inc_ref(v_vk_390_);
v___x_392_ = lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofInner(v_vk_390_, v_proof_391_);
if (lean_obj_tag(v___x_392_) == 0)
{
lean_object* v_a_393_; lean_object* v___x_395_; uint8_t v_isShared_396_; uint8_t v_isSharedCheck_402_; 
lean_dec_ref(v_proof_391_);
lean_dec_ref(v_vk_390_);
v_a_393_ = lean_ctor_get(v___x_392_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_392_);
if (v_isSharedCheck_402_ == 0)
{
v___x_395_ = v___x_392_;
v_isShared_396_ = v_isSharedCheck_402_;
goto v_resetjp_394_;
}
else
{
lean_inc(v_a_393_);
lean_dec(v___x_392_);
v___x_395_ = lean_box(0);
v_isShared_396_ = v_isSharedCheck_402_;
goto v_resetjp_394_;
}
v_resetjp_394_:
{
lean_object* v___x_397_; uint8_t v___x_398_; lean_object* v___x_400_; 
v___x_397_ = lean_alloc_ctor(0, 0, 1);
v___x_398_ = lean_unbox(v_a_393_);
lean_dec(v_a_393_);
lean_ctor_set_uint8(v___x_397_, 0, v___x_398_);
if (v_isShared_396_ == 0)
{
lean_ctor_set(v___x_395_, 0, v___x_397_);
v___x_400_ = v___x_395_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v___x_397_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
else
{
uint8_t v___x_403_; 
lean_dec_ref_known(v___x_392_, 1);
v___x_403_ = lp_openvm_x2dfv_VmVerifier_verifyVmStarkProofPvs(v_vk_390_, v_proof_391_);
lean_dec_ref(v_vk_390_);
if (v___x_403_ == 0)
{
lean_object* v___x_404_; 
v___x_404_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__0));
return v___x_404_;
}
else
{
lean_object* v___x_405_; 
v___x_405_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof___closed__1));
return v___x_405_;
}
}
}
}
LEAN_EXPORT uint32_t lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0(lean_object* v_error_406_){
_start:
{
lean_object* v___x_407_; uint32_t v___x_408_; 
v___x_407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_407_, 0, v_error_406_);
v___x_408_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_exitCode(v___x_407_);
lean_dec_ref_known(v___x_407_, 1);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0___boxed(lean_object* v_error_409_){
_start:
{
uint32_t v_res_410_; lean_object* v_r_411_; 
v_res_410_ = lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0(v_error_409_);
v_r_411_ = lean_box_uint32(v_res_410_);
return v_r_411_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof(lean_object* v_vkBytes_412_, lean_object* v_baselineBytes_413_, lean_object* v_proofBytes_414_, lean_object* v_pvBytes_415_, lean_object* v_userPvsBytes_416_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_decodeAll(v_vkBytes_412_, v_proofBytes_414_, v_pvBytes_415_);
if (lean_obj_tag(v___x_417_) == 0)
{
lean_object* v_a_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_427_; 
lean_dec_ref(v_userPvsBytes_416_);
lean_dec_ref(v_baselineBytes_413_);
v_a_418_ = lean_ctor_get(v___x_417_, 0);
v_isSharedCheck_427_ = !lean_is_exclusive(v___x_417_);
if (v_isSharedCheck_427_ == 0)
{
v___x_420_ = v___x_417_;
v_isShared_421_ = v_isSharedCheck_427_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_a_418_);
lean_dec(v___x_417_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_427_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
uint32_t v___x_422_; lean_object* v___x_423_; lean_object* v___x_425_; 
v___x_422_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_exitCode(v_a_418_);
lean_dec(v_a_418_);
v___x_423_ = lean_box_uint32(v___x_422_);
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 0, v___x_423_);
v___x_425_ = v___x_420_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v___x_423_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
else
{
lean_object* v_a_428_; lean_object* v_snd_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_477_; 
v_a_428_ = lean_ctor_get(v___x_417_, 0);
lean_inc(v_a_428_);
lean_dec_ref_known(v___x_417_, 1);
v_snd_429_ = lean_ctor_get(v_a_428_, 1);
v_isSharedCheck_477_ = !lean_is_exclusive(v_a_428_);
if (v_isSharedCheck_477_ == 0)
{
lean_object* v_unused_478_; 
v_unused_478_ = lean_ctor_get(v_a_428_, 0);
lean_dec(v_unused_478_);
v___x_431_ = v_a_428_;
v_isShared_432_ = v_isSharedCheck_477_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_snd_429_);
lean_dec(v_a_428_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_477_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v_fst_433_; lean_object* v_snd_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_476_; 
v_fst_433_ = lean_ctor_get(v_snd_429_, 0);
v_snd_434_ = lean_ctor_get(v_snd_429_, 1);
v_isSharedCheck_476_ = !lean_is_exclusive(v_snd_429_);
if (v_isSharedCheck_476_ == 0)
{
v___x_436_ = v_snd_429_;
v_isShared_437_ = v_isSharedCheck_476_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_snd_434_);
lean_inc(v_fst_433_);
lean_dec(v_snd_429_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_476_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_438_; 
v___x_438_ = lp_openvm_x2dfv_VmVerifier_Spec_Wire_readBaseline(v_baselineBytes_413_);
if (lean_obj_tag(v___x_438_) == 0)
{
lean_object* v_a_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_448_; 
lean_del_object(v___x_436_);
lean_dec(v_snd_434_);
lean_dec(v_fst_433_);
lean_del_object(v___x_431_);
lean_dec_ref(v_userPvsBytes_416_);
v_a_439_ = lean_ctor_get(v___x_438_, 0);
v_isSharedCheck_448_ = !lean_is_exclusive(v___x_438_);
if (v_isSharedCheck_448_ == 0)
{
v___x_441_ = v___x_438_;
v_isShared_442_ = v_isSharedCheck_448_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_a_439_);
lean_dec(v___x_438_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_448_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
uint32_t v___x_443_; lean_object* v___x_444_; lean_object* v___x_446_; 
v___x_443_ = lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0(v_a_439_);
v___x_444_ = lean_box_uint32(v___x_443_);
if (v_isShared_442_ == 0)
{
lean_ctor_set(v___x_441_, 0, v___x_444_);
v___x_446_ = v___x_441_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_447_; 
v_reuseFailAlloc_447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_447_, 0, v___x_444_);
v___x_446_ = v_reuseFailAlloc_447_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
return v___x_446_;
}
}
}
else
{
lean_object* v_a_449_; lean_object* v___x_450_; 
v_a_449_ = lean_ctor_get(v___x_438_, 0);
lean_inc(v_a_449_);
lean_dec_ref_known(v___x_438_, 1);
v___x_450_ = lp_openvm_x2dfv_VmVerifier_Spec_Wire_readUserPvsProof(v_userPvsBytes_416_);
if (lean_obj_tag(v___x_450_) == 0)
{
lean_object* v_a_451_; lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_460_; 
lean_dec(v_a_449_);
lean_del_object(v___x_436_);
lean_dec(v_snd_434_);
lean_dec(v_fst_433_);
lean_del_object(v___x_431_);
v_a_451_ = lean_ctor_get(v___x_450_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_460_ == 0)
{
v___x_453_ = v___x_450_;
v_isShared_454_ = v_isSharedCheck_460_;
goto v_resetjp_452_;
}
else
{
lean_inc(v_a_451_);
lean_dec(v___x_450_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_460_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
uint32_t v___x_455_; lean_object* v___x_456_; lean_object* v___x_458_; 
v___x_455_ = lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof___lam__0(v_a_451_);
v___x_456_ = lean_box_uint32(v___x_455_);
if (v_isShared_454_ == 0)
{
lean_ctor_set(v___x_453_, 0, v___x_456_);
v___x_458_ = v___x_453_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v___x_456_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
else
{
lean_object* v_a_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_475_; 
v_a_461_ = lean_ctor_get(v___x_450_, 0);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_475_ == 0)
{
v___x_463_ = v___x_450_;
v_isShared_464_ = v_isSharedCheck_475_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_a_461_);
lean_dec(v___x_450_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_475_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_466_; 
if (v_isShared_432_ == 0)
{
lean_ctor_set(v___x_431_, 1, v_a_449_);
lean_ctor_set(v___x_431_, 0, v_fst_433_);
v___x_466_ = v___x_431_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v_fst_433_);
lean_ctor_set(v_reuseFailAlloc_474_, 1, v_a_449_);
v___x_466_ = v_reuseFailAlloc_474_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
lean_object* v___x_467_; lean_object* v___x_469_; 
v___x_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_467_, 0, v_snd_434_);
lean_ctor_set(v___x_467_, 1, v_a_461_);
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 1, v___x_467_);
lean_ctor_set(v___x_436_, 0, v___x_466_);
v___x_469_ = v___x_436_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v___x_466_);
lean_ctor_set(v_reuseFailAlloc_473_, 1, v___x_467_);
v___x_469_ = v_reuseFailAlloc_473_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
lean_object* v___x_471_; 
if (v_isShared_464_ == 0)
{
lean_ctor_set(v___x_463_, 0, v___x_469_);
v___x_471_ = v___x_463_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_472_; 
v_reuseFailAlloc_472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_472_, 0, v___x_469_);
v___x_471_ = v_reuseFailAlloc_472_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
return v___x_471_;
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
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1(void){
_start:
{
uint32_t v___x_479_; lean_object* v___x_480_; 
v___x_479_ = 12;
v___x_480_ = lean_box_uint32(v___x_479_);
return v___x_480_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0(void){
_start:
{
lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_481_ = lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1;
v___x_482_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_482_, 0, v___x_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof(lean_object* v_vkBytes_483_, lean_object* v_baselineBytes_484_, lean_object* v_proofBytes_485_, lean_object* v_pvBytes_486_, lean_object* v_userPvsBytes_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lp_openvm_x2dfv_VmVerifier_decodeVmStarkProof(v_vkBytes_483_, v_baselineBytes_484_, v_proofBytes_485_, v_pvBytes_486_, v_userPvsBytes_487_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_496_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_496_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_496_ == 0)
{
v___x_491_ = v___x_488_;
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_dec(v___x_488_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_494_; 
if (v_isShared_492_ == 0)
{
v___x_494_ = v___x_491_;
goto v_reusejp_493_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v_a_489_);
v___x_494_ = v_reuseFailAlloc_495_;
goto v_reusejp_493_;
}
v_reusejp_493_:
{
return v___x_494_;
}
}
}
else
{
lean_object* v_a_497_; lean_object* v_fst_498_; lean_object* v_snd_499_; lean_object* v___x_500_; 
v_a_497_ = lean_ctor_get(v___x_488_, 0);
lean_inc(v_a_497_);
lean_dec_ref_known(v___x_488_, 1);
v_fst_498_ = lean_ctor_get(v_a_497_, 0);
lean_inc(v_fst_498_);
v_snd_499_ = lean_ctor_get(v_a_497_, 1);
lean_inc(v_snd_499_);
lean_dec(v_a_497_);
v___x_500_ = lp_openvm_x2dfv_VmVerifier_verifyDecodedVmStarkProof(v_fst_498_, v_snd_499_);
if (lean_obj_tag(v___x_500_) == 0)
{
lean_object* v_a_501_; lean_object* v___x_503_; uint8_t v_isShared_504_; uint8_t v_isSharedCheck_519_; 
v_a_501_ = lean_ctor_get(v___x_500_, 0);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_500_);
if (v_isSharedCheck_519_ == 0)
{
v___x_503_ = v___x_500_;
v_isShared_504_ = v_isSharedCheck_519_;
goto v_resetjp_502_;
}
else
{
lean_inc(v_a_501_);
lean_dec(v___x_500_);
v___x_503_ = lean_box(0);
v_isShared_504_ = v_isSharedCheck_519_;
goto v_resetjp_502_;
}
v_resetjp_502_:
{
if (lean_obj_tag(v_a_501_) == 0)
{
uint8_t v_error_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_517_; 
v_error_505_ = lean_ctor_get_uint8(v_a_501_, 0);
v_isSharedCheck_517_ = !lean_is_exclusive(v_a_501_);
if (v_isSharedCheck_517_ == 0)
{
v___x_507_ = v_a_501_;
v_isShared_508_ = v_isSharedCheck_517_;
goto v_resetjp_506_;
}
else
{
lean_dec(v_a_501_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_517_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_510_; 
if (v_isShared_508_ == 0)
{
lean_ctor_set_tag(v___x_507_, 1);
v___x_510_ = v___x_507_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v_reuseFailAlloc_516_, 0, v_error_505_);
v___x_510_ = v_reuseFailAlloc_516_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
uint32_t v___x_511_; lean_object* v___x_512_; lean_object* v___x_514_; 
v___x_511_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_exitCode(v___x_510_);
lean_dec_ref(v___x_510_);
v___x_512_ = lean_box_uint32(v___x_511_);
if (v_isShared_504_ == 0)
{
lean_ctor_set(v___x_503_, 0, v___x_512_);
v___x_514_ = v___x_503_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v___x_512_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
}
}
else
{
lean_object* v___x_518_; 
lean_del_object(v___x_503_);
v___x_518_ = lean_obj_once(&lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0, &lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0_once, _init_lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0);
return v___x_518_;
}
}
}
else
{
lean_object* v_a_520_; lean_object* v___x_522_; uint8_t v_isShared_523_; uint8_t v_isSharedCheck_527_; 
v_a_520_ = lean_ctor_get(v___x_500_, 0);
v_isSharedCheck_527_ = !lean_is_exclusive(v___x_500_);
if (v_isSharedCheck_527_ == 0)
{
v___x_522_ = v___x_500_;
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
else
{
lean_inc(v_a_520_);
lean_dec(v___x_500_);
v___x_522_ = lean_box(0);
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
v_resetjp_521_:
{
lean_object* v___x_525_; 
if (v_isShared_523_ == 0)
{
v___x_525_ = v___x_522_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v_a_520_);
v___x_525_ = v_reuseFailAlloc_526_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
return v___x_525_;
}
}
}
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Runtime_RawInstances(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_VerifierBabyBearPoseidon2(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VmVerifier_Spec_Wire(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VmVerifier_Spec_Runtime(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Runtime_RawInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_VerifierBabyBearPoseidon2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VmVerifier_Spec_Wire(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VmVerifier_rawZero = _init_lp_openvm_x2dfv_VmVerifier_rawZero();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_rawZero);
lp_openvm_x2dfv_VmVerifier_rawOne = _init_lp_openvm_x2dfv_VmVerifier_rawOne();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_rawOne);
lp_openvm_x2dfv_VmVerifier_rawZeroDigest = _init_lp_openvm_x2dfv_VmVerifier_rawZeroDigest();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_rawZeroDigest);
lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError_default = _init_lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError_default();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError_default);
lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError = _init_lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_instInhabitedVmStarkProofError);
lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1 = _init_lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
