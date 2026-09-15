// Lean compiler output
// Module: VmVerifier.Spec.Runtime
// Imports: public import Init public meta import Init public import Fundamentals.Poseidon2.Raw public import Swirl.Protocol.Noninteractive.Runtime.RawInstances public import Swirl.Protocol.Noninteractive.Verifier.Runtime.Main public import Swirl.Protocol.Noninteractive.VerifierBabyBearPoseidon2 public import VmVerifier.Spec.Wire
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
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_workspace_Recursion_Spec_digestPrefixOf___redArg(lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_workspace_VmVerifier_publicValuesHeight_x3f(lean_object*);
lean_object* lp_workspace_VmVerifier_MemoryDimensions_overallHeight(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_workspace_VmVerifier_MemoryDimensions_labelToIndex(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Array_instDecidableEqImpl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_ofBase(lean_object*);
uint32_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_exitCode(lean_object*);
lean_object* lp_workspace_VmVerifier_parseVmProofData_x3f(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_ofNat(lean_object*);
uint8_t lp_workspace_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_decodeAll(lean_object*, lean_object*, lean_object*);
lean_object* lp_workspace_VmVerifier_Spec_Wire_readBaseline(lean_object*);
lean_object* lp_workspace_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
extern lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_RawInstances_rawFieldOpsEF;
lean_object* lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_hashSlice(lean_object*);
extern lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_init___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__0;
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawZero;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawOne;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawZeroDigest___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawZeroDigest___lam__0___boxed(lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_rawZeroDigest___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_VmVerifier_rawZeroDigest___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_rawZeroDigest___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_rawZeroDigest___closed__0_value;
static lean_once_cell_t lp_workspace_VmVerifier_rawZeroDigest___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_rawZeroDigest___closed__1;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawZeroDigest;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawLeafDigest(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawMerkleNode(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawMerkleNode___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_workspace_VmVerifier_rawToChunks___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_workspace_VmVerifier_rawToChunks___redArg___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_rawToChunks___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_rawPublicValuesMerkleRoot_x3f_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawComputeExeCommit___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawComputeExeCommit___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawComputeExeCommit(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instDecidableEqFin___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1))} };
static const lean_object* lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___closed__0_value;
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_decidableUserPublicValuesRawValid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_workspace_VmVerifier_decidableVmProofDataRawValid___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_decidableVmProofDataRawValid___closed__0;
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_decidableVmProofDataRawValid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decidableVmProofDataRawValid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_verifyVmProofData(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmProofData___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_verifyVmStarkProofPvs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProofPvs___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instInhabitedVmStarkProofError_default;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_instInhabitedVmStarkProofError;
static const lean_closure_object lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_hashSlice, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__0_value;
static const lean_closure_object lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__1 = (const lean_object*)&lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__1_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_vmStarkProtocolConfig(lean_object*);
static const lean_closure_object lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBearExt4_Raw_ofBase, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__0_value;
static const lean_ctor_object lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__1 = (const lean_object*)&lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__1_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProofInner(lean_object*, lean_object*);
static const lean_ctor_object lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__0 = (const lean_object*)&lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__0_value;
static const lean_ctor_object lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__1 = (const lean_object*)&lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__1_value;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyDecodedVmStarkProof(lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_workspace_VmVerifier_decodeVmStarkProof___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decodeVmStarkProof___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decodeVmStarkProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1;
static lean_once_cell_t lp_workspace_VmVerifier_verifyVmStarkProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_VmVerifier_verifyVmStarkProof___closed__0;
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_workspace_VmVerifier_rawZero(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(0u);
return v___x_1_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_rawOne(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(1u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawZeroDigest___lam__0(lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(0u);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawZeroDigest___lam__0___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_workspace_VmVerifier_rawZeroDigest___lam__0(v_x_5_);
lean_dec(v_x_5_);
return v_res_6_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_rawZeroDigest___closed__1(void){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_8_ = ((lean_object*)(lp_workspace_VmVerifier_rawZeroDigest___closed__0));
v___x_9_ = lean_unsigned_to_nat(8u);
v___x_10_ = l_Array_ofFn___redArg(v___x_9_, v___f_8_);
return v___x_10_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_rawZeroDigest(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_workspace_VmVerifier_rawZeroDigest___closed__1, &lp_workspace_VmVerifier_rawZeroDigest___closed__1_once, _init_lp_workspace_VmVerifier_rawZeroDigest___closed__1);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawLeafDigest(lean_object* v_values_12_){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lp_workspace_VmVerifier_rawZeroDigest;
v___x_14_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(v_values_12_, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawMerkleNode(lean_object* v_x_15_, lean_object* v_x_16_, lean_object* v_x_17_){
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
v___x_25_ = lp_workspace_VmVerifier_rawMerkleNode(v_n_22_, v_x_16_, v___x_24_);
v___x_26_ = lean_nat_add(v___x_24_, v_one_21_);
lean_dec(v___x_24_);
v___x_27_ = lp_workspace_VmVerifier_rawMerkleNode(v_n_22_, v_x_16_, v___x_26_);
lean_dec(v_n_22_);
v___x_28_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(v___x_25_, v___x_27_);
return v___x_28_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawMerkleNode___boxed(lean_object* v_x_29_, lean_object* v_x_30_, lean_object* v_x_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_workspace_VmVerifier_rawMerkleNode(v_x_29_, v_x_30_, v_x_31_);
lean_dec(v_x_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go___redArg(lean_object* v_chunkSize_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_){
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go___redArg___boxed(lean_object* v_chunkSize_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_workspace_VmVerifier_rawToChunks_go___redArg(v_chunkSize_51_, v_a_52_, v_a_53_, v_a_54_);
lean_dec(v_chunkSize_51_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go(lean_object* v_00_u03b1_56_, lean_object* v_chunkSize_57_, lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_workspace_VmVerifier_rawToChunks_go___redArg(v_chunkSize_57_, v_a_58_, v_a_59_, v_a_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks_go___boxed(lean_object* v_00_u03b1_62_, lean_object* v_chunkSize_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_workspace_VmVerifier_rawToChunks_go(v_00_u03b1_62_, v_chunkSize_63_, v_a_64_, v_a_65_, v_a_66_);
lean_dec(v_chunkSize_63_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks___redArg(lean_object* v_x_70_, lean_object* v_x_71_){
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
v___x_80_ = ((lean_object*)(lp_workspace_VmVerifier_rawToChunks___redArg___closed__0));
v___x_81_ = lp_workspace_VmVerifier_rawToChunks_go___redArg(v_x_70_, v_tail_74_, v___x_79_, v___x_80_);
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks___redArg___boxed(lean_object* v_x_84_, lean_object* v_x_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_workspace_VmVerifier_rawToChunks___redArg(v_x_84_, v_x_85_);
lean_dec(v_x_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks(lean_object* v_00_u03b1_87_, lean_object* v_x_88_, lean_object* v_x_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_workspace_VmVerifier_rawToChunks___redArg(v_x_88_, v_x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawToChunks___boxed(lean_object* v_00_u03b1_91_, lean_object* v_x_92_, lean_object* v_x_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_workspace_VmVerifier_rawToChunks(v_00_u03b1_91_, v_x_92_, v_x_93_);
lean_dec(v_x_92_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0(lean_object* v___x_95_, lean_object* v_index_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = l_List_get_x3fInternal___redArg(v___x_95_, v_index_96_);
if (lean_obj_tag(v___x_97_) == 0)
{
lean_object* v___x_98_; 
v___x_98_ = lp_workspace_VmVerifier_rawZeroDigest;
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0___boxed(lean_object* v___x_100_, lean_object* v_index_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0(v___x_100_, v_index_101_);
lean_dec(v___x_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_workspace_List_mapTR_loop___at___00VmVerifier_rawPublicValuesMerkleRoot_x3f_spec__0(lean_object* v_a_103_, lean_object* v_a_104_){
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
v___x_112_ = lp_workspace_Recursion_Spec_digestPrefixOf___redArg(v___x_111_, v_head_106_);
v___x_113_ = lp_workspace_VmVerifier_rawLeafDigest(v___x_112_);
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f(lean_object* v_values_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_workspace_VmVerifier_publicValuesHeight_x3f(v_values_119_);
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
v___x_127_ = lp_workspace_VmVerifier_rawToChunks___redArg(v___x_126_, v_values_119_);
v___x_128_ = lean_box(0);
v___x_129_ = lp_workspace_List_mapTR_loop___at___00VmVerifier_rawPublicValuesMerkleRoot_x3f_spec__0(v___x_127_, v___x_128_);
v___f_130_ = lean_alloc_closure((void*)(lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f___lam__0___boxed), 2, 1);
lean_closure_set(v___f_130_, 0, v___x_129_);
v___x_131_ = lean_unsigned_to_nat(0u);
v___x_132_ = lp_workspace_VmVerifier_rawMerkleNode(v_val_122_, v___f_130_, v___x_131_);
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawComputeExeCommit___lam__0(lean_object* v_initialPc_137_, lean_object* v_index_138_){
_start:
{
lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_139_ = lean_unsigned_to_nat(0u);
v___x_140_ = lean_nat_dec_eq(v_index_138_, v___x_139_);
if (v___x_140_ == 0)
{
return v___x_139_;
}
else
{
lean_inc(v_initialPc_137_);
return v_initialPc_137_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawComputeExeCommit___lam__0___boxed(lean_object* v_initialPc_141_, lean_object* v_index_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_workspace_VmVerifier_rawComputeExeCommit___lam__0(v_initialPc_141_, v_index_142_);
lean_dec(v_index_142_);
lean_dec(v_initialPc_141_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_rawComputeExeCommit(lean_object* v_programCommit_144_, lean_object* v_initialRoot_145_, lean_object* v_initialPc_146_){
_start:
{
lean_object* v___f_147_; lean_object* v___x_148_; lean_object* v_paddedPc_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___f_147_ = lean_alloc_closure((void*)(lp_workspace_VmVerifier_rawComputeExeCommit___lam__0___boxed), 2, 1);
lean_closure_set(v___f_147_, 0, v_initialPc_146_);
v___x_148_ = lean_unsigned_to_nat(8u);
v_paddedPc_149_ = l_Array_ofFn___redArg(v___x_148_, v___f_147_);
v___x_150_ = lp_workspace_VmVerifier_rawLeafDigest(v_programCommit_144_);
v___x_151_ = lp_workspace_VmVerifier_rawLeafDigest(v_initialRoot_145_);
v___x_152_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(v___x_150_, v___x_151_);
v___x_153_ = lp_workspace_VmVerifier_rawLeafDigest(v_paddedPc_149_);
v___x_154_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_compressDigest(v___x_152_, v___x_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_workspace___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter___redArg(lean_object* v_x_155_, lean_object* v_h__1_156_, lean_object* v_h__2_157_){
_start:
{
if (lean_obj_tag(v_x_155_) == 0)
{
lean_object* v___x_158_; lean_object* v___x_159_; 
lean_dec(v_h__2_157_);
v___x_158_ = lean_box(0);
v___x_159_ = lean_apply_1(v_h__1_156_, v___x_158_);
return v___x_159_;
}
else
{
lean_object* v_val_160_; lean_object* v___x_161_; 
lean_dec(v_h__1_156_);
v_val_160_ = lean_ctor_get(v_x_155_, 0);
lean_inc(v_val_160_);
lean_dec_ref_known(v_x_155_, 1);
v___x_161_ = lean_apply_1(v_h__2_157_, v_val_160_);
return v___x_161_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace___private_VmVerifier_Spec_Runtime_0__VmVerifier_UserPublicValuesRawValid_match__1_splitter(lean_object* v_motive_162_, lean_object* v_x_163_, lean_object* v_h__1_164_, lean_object* v_h__2_165_){
_start:
{
if (lean_obj_tag(v_x_163_) == 0)
{
lean_object* v___x_166_; lean_object* v___x_167_; 
lean_dec(v_h__2_165_);
v___x_166_ = lean_box(0);
v___x_167_ = lean_apply_1(v_h__1_164_, v___x_166_);
return v___x_167_;
}
else
{
lean_object* v_val_168_; lean_object* v___x_169_; 
lean_dec(v_h__1_164_);
v_val_168_ = lean_ctor_get(v_x_163_, 0);
lean_inc(v_val_168_);
lean_dec_ref_known(v_x_163_, 1);
v___x_169_ = lean_apply_1(v_h__2_165_, v_val_168_);
return v___x_169_;
}
}
}
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0(lean_object* v_a_172_, lean_object* v_b_173_){
_start:
{
lean_object* v___x_174_; uint8_t v___x_175_; 
v___x_174_ = ((lean_object*)(lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0));
v___x_175_ = l_Array_instDecidableEqImpl___redArg(v___x_174_, v_a_172_, v_b_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___boxed(lean_object* v_a_176_, lean_object* v_b_177_){
_start:
{
uint8_t v_res_178_; lean_object* v_r_179_; 
v_res_178_ = lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0(v_a_176_, v_b_177_);
lean_dec_ref(v_b_177_);
lean_dec_ref(v_a_176_);
v_r_179_ = lean_box(v_res_178_);
return v_r_179_;
}
}
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_decidableUserPublicValuesRawValid(lean_object* v_userProof_181_, lean_object* v_dimensions_182_, lean_object* v_finalRoot_183_){
_start:
{
lean_object* v_authenticationPath_184_; lean_object* v_publicValues_185_; lean_object* v_publicValuesCommit_186_; lean_object* v___x_187_; 
v_authenticationPath_184_ = lean_ctor_get(v_userProof_181_, 0);
lean_inc(v_authenticationPath_184_);
v_publicValues_185_ = lean_ctor_get(v_userProof_181_, 1);
lean_inc(v_publicValues_185_);
v_publicValuesCommit_186_ = lean_ctor_get(v_userProof_181_, 2);
lean_inc_ref(v_publicValuesCommit_186_);
lean_dec_ref(v_userProof_181_);
v___x_187_ = lp_workspace_VmVerifier_publicValuesHeight_x3f(v_publicValues_185_);
if (lean_obj_tag(v___x_187_) == 0)
{
uint8_t v___x_188_; 
lean_dec_ref(v_publicValuesCommit_186_);
lean_dec(v_publicValues_185_);
lean_dec(v_authenticationPath_184_);
lean_dec_ref(v_finalRoot_183_);
v___x_188_ = 0;
return v___x_188_;
}
else
{
lean_object* v_val_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_211_; 
v_val_189_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_211_ == 0)
{
v___x_191_ = v___x_187_;
v_isShared_192_ = v_isSharedCheck_211_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_val_189_);
lean_dec(v___x_187_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_211_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v___x_193_; uint8_t v___x_194_; 
v___x_193_ = lp_workspace_VmVerifier_MemoryDimensions_overallHeight(v_dimensions_182_);
v___x_194_ = lean_nat_dec_le(v_val_189_, v___x_193_);
if (v___x_194_ == 0)
{
lean_dec(v___x_193_);
lean_del_object(v___x_191_);
lean_dec(v_val_189_);
lean_dec_ref(v_publicValuesCommit_186_);
lean_dec(v_publicValues_185_);
lean_dec(v_authenticationPath_184_);
lean_dec_ref(v_finalRoot_183_);
return v___x_194_;
}
else
{
lean_object* v___x_195_; lean_object* v___x_196_; uint8_t v___x_197_; 
v___x_195_ = l_List_lengthTR___redArg(v_authenticationPath_184_);
v___x_196_ = lean_nat_sub(v___x_193_, v_val_189_);
lean_dec(v___x_193_);
v___x_197_ = lean_nat_dec_eq(v___x_195_, v___x_196_);
lean_dec(v___x_196_);
lean_dec(v___x_195_);
if (v___x_197_ == 0)
{
lean_del_object(v___x_191_);
lean_dec(v_val_189_);
lean_dec_ref(v_publicValuesCommit_186_);
lean_dec(v_publicValues_185_);
lean_dec(v_authenticationPath_184_);
lean_dec_ref(v_finalRoot_183_);
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; uint8_t v___x_204_; 
v___x_198_ = lean_unsigned_to_nat(3u);
v___x_199_ = lean_unsigned_to_nat(0u);
v___x_200_ = lp_workspace_VmVerifier_MemoryDimensions_labelToIndex(v_dimensions_182_, v___x_198_, v___x_199_);
v___x_201_ = lean_unsigned_to_nat(2u);
v___x_202_ = lean_nat_pow(v___x_201_, v_val_189_);
lean_dec(v_val_189_);
v___x_203_ = lean_nat_div(v___x_200_, v___x_202_);
lean_dec(v___x_202_);
lean_dec(v___x_200_);
lean_inc_ref(v_publicValuesCommit_186_);
v___x_204_ = lp_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw_merkleVerify(v_finalRoot_183_, v___x_203_, v_publicValuesCommit_186_, v_authenticationPath_184_);
if (v___x_204_ == 0)
{
lean_del_object(v___x_191_);
lean_dec_ref(v_publicValuesCommit_186_);
lean_dec(v_publicValues_185_);
return v___x_204_;
}
else
{
lean_object* v___f_205_; lean_object* v___x_206_; lean_object* v___x_208_; 
v___f_205_ = ((lean_object*)(lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___closed__0));
v___x_206_ = lp_workspace_VmVerifier_rawPublicValuesMerkleRoot_x3f(v_publicValues_185_);
if (v_isShared_192_ == 0)
{
lean_ctor_set(v___x_191_, 0, v_publicValuesCommit_186_);
v___x_208_ = v___x_191_;
goto v_reusejp_207_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v_publicValuesCommit_186_);
v___x_208_ = v_reuseFailAlloc_210_;
goto v_reusejp_207_;
}
v_reusejp_207_:
{
uint8_t v___x_209_; 
v___x_209_ = l_Option_instDecidableEq___redArg(v___f_205_, v___x_206_, v___x_208_);
return v___x_209_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___boxed(lean_object* v_userProof_212_, lean_object* v_dimensions_213_, lean_object* v_finalRoot_214_){
_start:
{
uint8_t v_res_215_; lean_object* v_r_216_; 
v_res_215_ = lp_workspace_VmVerifier_decidableUserPublicValuesRawValid(v_userProof_212_, v_dimensions_213_, v_finalRoot_214_);
lean_dec_ref(v_dimensions_213_);
v_r_216_ = lean_box(v_res_215_);
return v_r_216_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_decidableVmProofDataRawValid___closed__0(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_unsigned_to_nat(2u);
v___x_218_ = lp_swirl_x2drbr_x2dformal_Fundamentals_BabyBear_FBB_Raw_ofNat(v___x_217_);
return v___x_218_;
}
}
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_decidableVmProofDataRawValid(lean_object* v_vmPublicValues_219_, lean_object* v_userPublicValuesProof_220_, lean_object* v_baseline_221_){
_start:
{
lean_object* v_toAggregationPublicValues_222_; lean_object* v_vm_223_; lean_object* v_memory_224_; lean_object* v_programCommit_225_; lean_object* v_initialState_226_; lean_object* v_initialPc_227_; lean_object* v_memoryDimensions_228_; lean_object* v_numUserPvs_229_; lean_object* v_appVkCommit_230_; lean_object* v_leafVkCommit_231_; lean_object* v_internalForLeafVkCommit_232_; lean_object* v_internalRecursiveVkCommit_233_; lean_object* v_constraintEvalCachedCommit_234_; lean_object* v_toAggregationBasePublicValues_235_; lean_object* v_programCommit_236_; lean_object* v_connector_237_; lean_object* v_initialRoot_238_; lean_object* v_finalRoot_239_; uint8_t v___x_240_; 
v_toAggregationPublicValues_222_ = lean_ctor_get(v_vmPublicValues_219_, 0);
lean_inc_ref(v_toAggregationPublicValues_222_);
v_vm_223_ = lean_ctor_get(v_toAggregationPublicValues_222_, 1);
lean_inc_ref(v_vm_223_);
v_memory_224_ = lean_ctor_get(v_vm_223_, 2);
lean_inc_ref(v_memory_224_);
v_programCommit_225_ = lean_ctor_get(v_baseline_221_, 0);
v_initialState_226_ = lean_ctor_get(v_baseline_221_, 1);
v_initialPc_227_ = lean_ctor_get(v_baseline_221_, 2);
v_memoryDimensions_228_ = lean_ctor_get(v_baseline_221_, 3);
v_numUserPvs_229_ = lean_ctor_get(v_baseline_221_, 4);
v_appVkCommit_230_ = lean_ctor_get(v_baseline_221_, 5);
v_leafVkCommit_231_ = lean_ctor_get(v_baseline_221_, 6);
v_internalForLeafVkCommit_232_ = lean_ctor_get(v_baseline_221_, 7);
v_internalRecursiveVkCommit_233_ = lean_ctor_get(v_baseline_221_, 8);
v_constraintEvalCachedCommit_234_ = lean_ctor_get(v_vmPublicValues_219_, 1);
lean_inc_ref(v_constraintEvalCachedCommit_234_);
lean_dec_ref(v_vmPublicValues_219_);
v_toAggregationBasePublicValues_235_ = lean_ctor_get(v_toAggregationPublicValues_222_, 0);
lean_inc_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_toAggregationPublicValues_222_);
v_programCommit_236_ = lean_ctor_get(v_vm_223_, 0);
lean_inc_ref(v_programCommit_236_);
v_connector_237_ = lean_ctor_get(v_vm_223_, 1);
lean_inc_ref(v_connector_237_);
lean_dec_ref(v_vm_223_);
v_initialRoot_238_ = lean_ctor_get(v_memory_224_, 0);
lean_inc_ref(v_initialRoot_238_);
v_finalRoot_239_ = lean_ctor_get(v_memory_224_, 1);
lean_inc_ref(v_finalRoot_239_);
lean_dec_ref(v_memory_224_);
lean_inc_ref(v_userPublicValuesProof_220_);
v___x_240_ = lp_workspace_VmVerifier_decidableUserPublicValuesRawValid(v_userPublicValuesProof_220_, v_memoryDimensions_228_, v_finalRoot_239_);
if (v___x_240_ == 0)
{
lean_dec_ref(v_initialRoot_238_);
lean_dec_ref(v_connector_237_);
lean_dec_ref(v_programCommit_236_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
lean_dec_ref(v_userPublicValuesProof_220_);
return v___x_240_;
}
else
{
lean_object* v_publicValues_241_; lean_object* v___x_242_; uint8_t v___x_243_; 
v_publicValues_241_ = lean_ctor_get(v_userPublicValuesProof_220_, 1);
lean_inc(v_publicValues_241_);
lean_dec_ref(v_userPublicValuesProof_220_);
v___x_242_ = l_List_lengthTR___redArg(v_publicValues_241_);
lean_dec(v_publicValues_241_);
v___x_243_ = lean_nat_dec_eq(v___x_242_, v_numUserPvs_229_);
lean_dec(v___x_242_);
if (v___x_243_ == 0)
{
lean_dec_ref(v_initialRoot_238_);
lean_dec_ref(v_connector_237_);
lean_dec_ref(v_programCommit_236_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_243_;
}
else
{
lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_244_ = ((lean_object*)(lp_workspace_VmVerifier_decidableUserPublicValuesRawValid___lam__0___closed__0));
v___x_245_ = l_Array_instDecidableEqImpl___redArg(v___x_244_, v_programCommit_236_, v_programCommit_225_);
lean_dec_ref(v_programCommit_236_);
if (v___x_245_ == 0)
{
lean_dec_ref(v_initialRoot_238_);
lean_dec_ref(v_connector_237_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_245_;
}
else
{
uint8_t v___x_246_; 
v___x_246_ = l_Array_instDecidableEqImpl___redArg(v___x_244_, v_initialRoot_238_, v_initialState_226_);
lean_dec_ref(v_initialRoot_238_);
if (v___x_246_ == 0)
{
lean_dec_ref(v_connector_237_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_246_;
}
else
{
lean_object* v_initialPc_247_; lean_object* v_exitCode_248_; lean_object* v_isTerminate_249_; uint8_t v___x_250_; 
v_initialPc_247_ = lean_ctor_get(v_connector_237_, 0);
lean_inc(v_initialPc_247_);
v_exitCode_248_ = lean_ctor_get(v_connector_237_, 2);
lean_inc(v_exitCode_248_);
v_isTerminate_249_ = lean_ctor_get(v_connector_237_, 3);
lean_inc(v_isTerminate_249_);
lean_dec_ref(v_connector_237_);
v___x_250_ = lean_nat_dec_eq(v_initialPc_247_, v_initialPc_227_);
lean_dec(v_initialPc_247_);
if (v___x_250_ == 0)
{
lean_dec(v_isTerminate_249_);
lean_dec(v_exitCode_248_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_251_ = lean_unsigned_to_nat(0u);
v___x_252_ = lean_nat_dec_eq(v_exitCode_248_, v___x_251_);
lean_dec(v_exitCode_248_);
if (v___x_252_ == 0)
{
lean_dec(v_isTerminate_249_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_252_;
}
else
{
lean_object* v___x_253_; uint8_t v___x_254_; 
v___x_253_ = lean_unsigned_to_nat(1u);
v___x_254_ = lean_nat_dec_eq(v_isTerminate_249_, v___x_253_);
lean_dec(v_isTerminate_249_);
if (v___x_254_ == 0)
{
lean_dec_ref(v_toAggregationBasePublicValues_235_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_254_;
}
else
{
lean_object* v_internalFlag_255_; lean_object* v_appVkCommit_256_; lean_object* v_leafVkCommit_257_; lean_object* v_internalForLeafVkCommit_258_; lean_object* v_recursionDepth_259_; lean_object* v_internalRecursiveVkCommit_260_; lean_object* v___x_261_; uint8_t v___x_262_; 
v_internalFlag_255_ = lean_ctor_get(v_toAggregationBasePublicValues_235_, 0);
lean_inc(v_internalFlag_255_);
v_appVkCommit_256_ = lean_ctor_get(v_toAggregationBasePublicValues_235_, 1);
lean_inc_ref(v_appVkCommit_256_);
v_leafVkCommit_257_ = lean_ctor_get(v_toAggregationBasePublicValues_235_, 2);
lean_inc_ref(v_leafVkCommit_257_);
v_internalForLeafVkCommit_258_ = lean_ctor_get(v_toAggregationBasePublicValues_235_, 3);
lean_inc_ref(v_internalForLeafVkCommit_258_);
v_recursionDepth_259_ = lean_ctor_get(v_toAggregationBasePublicValues_235_, 4);
lean_inc(v_recursionDepth_259_);
v_internalRecursiveVkCommit_260_ = lean_ctor_get(v_toAggregationBasePublicValues_235_, 5);
lean_inc_ref(v_internalRecursiveVkCommit_260_);
lean_dec_ref(v_toAggregationBasePublicValues_235_);
v___x_261_ = lean_obj_once(&lp_workspace_VmVerifier_decidableVmProofDataRawValid___closed__0, &lp_workspace_VmVerifier_decidableVmProofDataRawValid___closed__0_once, _init_lp_workspace_VmVerifier_decidableVmProofDataRawValid___closed__0);
v___x_262_ = lean_nat_dec_eq(v_internalFlag_255_, v___x_261_);
lean_dec(v_internalFlag_255_);
if (v___x_262_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_260_);
lean_dec(v_recursionDepth_259_);
lean_dec_ref(v_internalForLeafVkCommit_258_);
lean_dec_ref(v_leafVkCommit_257_);
lean_dec_ref(v_appVkCommit_256_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_262_;
}
else
{
uint8_t v___x_263_; 
v___x_263_ = lp_workspace_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_244_, v_appVkCommit_256_, v_appVkCommit_230_);
lean_dec_ref(v_appVkCommit_256_);
if (v___x_263_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_260_);
lean_dec(v_recursionDepth_259_);
lean_dec_ref(v_internalForLeafVkCommit_258_);
lean_dec_ref(v_leafVkCommit_257_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_263_;
}
else
{
uint8_t v___x_264_; 
v___x_264_ = lp_workspace_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_244_, v_leafVkCommit_257_, v_leafVkCommit_231_);
lean_dec_ref(v_leafVkCommit_257_);
if (v___x_264_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_260_);
lean_dec(v_recursionDepth_259_);
lean_dec_ref(v_internalForLeafVkCommit_258_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_264_;
}
else
{
uint8_t v___x_265_; 
v___x_265_ = lp_workspace_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_244_, v_internalForLeafVkCommit_258_, v_internalForLeafVkCommit_232_);
lean_dec_ref(v_internalForLeafVkCommit_258_);
if (v___x_265_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_260_);
lean_dec(v_recursionDepth_259_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_265_;
}
else
{
uint8_t v___x_266_; 
v___x_266_ = lean_nat_dec_lt(v___x_251_, v_recursionDepth_259_);
if (v___x_266_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_260_);
lean_dec(v_recursionDepth_259_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_266_;
}
else
{
lean_object* v___x_267_; uint8_t v___x_268_; 
v___x_267_ = lean_unsigned_to_nat(256u);
v___x_268_ = lean_nat_dec_le(v_recursionDepth_259_, v___x_267_);
if (v___x_268_ == 0)
{
lean_dec_ref(v_internalRecursiveVkCommit_260_);
lean_dec(v_recursionDepth_259_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_268_;
}
else
{
uint8_t v___x_269_; 
v___x_269_ = lean_nat_dec_eq(v_recursionDepth_259_, v___x_253_);
lean_dec(v_recursionDepth_259_);
if (v___x_269_ == 0)
{
uint8_t v___x_270_; 
v___x_270_ = lp_workspace_Recursion_Spec_instDecidableEqVkCommitData_decEq___redArg(v___x_244_, v_internalRecursiveVkCommit_260_, v_internalRecursiveVkCommit_233_);
lean_dec_ref(v_internalRecursiveVkCommit_260_);
if (v___x_270_ == 0)
{
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_270_;
}
else
{
lean_object* v_cachedCommit_271_; uint8_t v___x_272_; 
v_cachedCommit_271_ = lean_ctor_get(v_internalRecursiveVkCommit_233_, 0);
v___x_272_ = l_Array_instDecidableEqImpl___redArg(v___x_244_, v_constraintEvalCachedCommit_234_, v_cachedCommit_271_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_272_;
}
}
else
{
lean_object* v_cachedCommit_273_; lean_object* v_vkPreHash_274_; lean_object* v___x_275_; uint8_t v___x_276_; 
v_cachedCommit_273_ = lean_ctor_get(v_internalRecursiveVkCommit_260_, 0);
lean_inc_ref(v_cachedCommit_273_);
v_vkPreHash_274_ = lean_ctor_get(v_internalRecursiveVkCommit_260_, 1);
lean_inc_ref(v_vkPreHash_274_);
lean_dec_ref(v_internalRecursiveVkCommit_260_);
v___x_275_ = lp_workspace_VmVerifier_rawZeroDigest;
v___x_276_ = l_Array_instDecidableEqImpl___redArg(v___x_244_, v_cachedCommit_273_, v___x_275_);
lean_dec_ref(v_cachedCommit_273_);
if (v___x_276_ == 0)
{
lean_dec_ref(v_vkPreHash_274_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_276_;
}
else
{
uint8_t v___x_277_; 
v___x_277_ = l_Array_instDecidableEqImpl___redArg(v___x_244_, v_vkPreHash_274_, v___x_275_);
lean_dec_ref(v_vkPreHash_274_);
if (v___x_277_ == 0)
{
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_277_;
}
else
{
lean_object* v_cachedCommit_278_; uint8_t v___x_279_; 
v_cachedCommit_278_ = lean_ctor_get(v_internalForLeafVkCommit_232_, 0);
v___x_279_ = l_Array_instDecidableEqImpl___redArg(v___x_244_, v_constraintEvalCachedCommit_234_, v_cachedCommit_278_);
lean_dec_ref(v_constraintEvalCachedCommit_234_);
return v___x_279_;
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
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decidableVmProofDataRawValid___boxed(lean_object* v_vmPublicValues_280_, lean_object* v_userPublicValuesProof_281_, lean_object* v_baseline_282_){
_start:
{
uint8_t v_res_283_; lean_object* v_r_284_; 
v_res_283_ = lp_workspace_VmVerifier_decidableVmProofDataRawValid(v_vmPublicValues_280_, v_userPublicValuesProof_281_, v_baseline_282_);
lean_dec_ref(v_baseline_282_);
v_r_284_ = lean_box(v_res_283_);
return v_r_284_;
}
}
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_verifyVmProofData(lean_object* v_vmPublicValues_285_, lean_object* v_userPublicValuesProof_286_, lean_object* v_baseline_287_){
_start:
{
uint8_t v___x_288_; 
v___x_288_ = lp_workspace_VmVerifier_decidableVmProofDataRawValid(v_vmPublicValues_285_, v_userPublicValuesProof_286_, v_baseline_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmProofData___boxed(lean_object* v_vmPublicValues_289_, lean_object* v_userPublicValuesProof_290_, lean_object* v_baseline_291_){
_start:
{
uint8_t v_res_292_; lean_object* v_r_293_; 
v_res_292_ = lp_workspace_VmVerifier_verifyVmProofData(v_vmPublicValues_289_, v_userPublicValuesProof_290_, v_baseline_291_);
lean_dec_ref(v_baseline_291_);
v_r_293_ = lean_box(v_res_292_);
return v_r_293_;
}
}
LEAN_EXPORT uint8_t lp_workspace_VmVerifier_verifyVmStarkProofPvs(lean_object* v_vk_294_, lean_object* v_proof_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_workspace_VmVerifier_parseVmProofData_x3f(v_proof_295_);
if (lean_obj_tag(v___x_296_) == 0)
{
uint8_t v___x_297_; 
v___x_297_ = 0;
return v___x_297_;
}
else
{
lean_object* v_val_298_; lean_object* v_fst_299_; lean_object* v_snd_300_; lean_object* v_baseline_301_; uint8_t v___x_302_; 
v_val_298_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_val_298_);
lean_dec_ref_known(v___x_296_, 1);
v_fst_299_ = lean_ctor_get(v_val_298_, 0);
lean_inc(v_fst_299_);
v_snd_300_ = lean_ctor_get(v_val_298_, 1);
lean_inc(v_snd_300_);
lean_dec(v_val_298_);
v_baseline_301_ = lean_ctor_get(v_vk_294_, 1);
v___x_302_ = lp_workspace_VmVerifier_decidableVmProofDataRawValid(v_fst_299_, v_snd_300_, v_baseline_301_);
return v___x_302_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProofPvs___boxed(lean_object* v_vk_303_, lean_object* v_proof_304_){
_start:
{
uint8_t v_res_305_; lean_object* v_r_306_; 
v_res_305_ = lp_workspace_VmVerifier_verifyVmStarkProofPvs(v_vk_303_, v_proof_304_);
lean_dec_ref(v_vk_303_);
v_r_306_ = lean_box(v_res_305_);
return v_r_306_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorIdx(lean_object* v_x_307_){
_start:
{
if (lean_obj_tag(v_x_307_) == 0)
{
lean_object* v___x_308_; 
v___x_308_ = lean_unsigned_to_nat(0u);
return v___x_308_;
}
else
{
lean_object* v___x_309_; 
v___x_309_ = lean_unsigned_to_nat(1u);
return v___x_309_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorIdx___boxed(lean_object* v_x_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_workspace_VmVerifier_VmStarkProofError_ctorIdx(v_x_310_);
lean_dec(v_x_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(lean_object* v_t_312_, lean_object* v_k_313_){
_start:
{
if (lean_obj_tag(v_t_312_) == 0)
{
uint8_t v_error_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v_error_314_ = lean_ctor_get_uint8(v_t_312_, 0);
v___x_315_ = lean_box(v_error_314_);
v___x_316_ = lean_apply_1(v_k_313_, v___x_315_);
return v___x_316_;
}
else
{
return v_k_313_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg___boxed(lean_object* v_t_317_, lean_object* v_k_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_317_, v_k_318_);
lean_dec(v_t_317_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim(lean_object* v_motive_320_, lean_object* v_ctorIdx_321_, lean_object* v_t_322_, lean_object* v_h_323_, lean_object* v_k_324_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_322_, v_k_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_ctorElim___boxed(lean_object* v_motive_326_, lean_object* v_ctorIdx_327_, lean_object* v_t_328_, lean_object* v_h_329_, lean_object* v_k_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim(v_motive_326_, v_ctorIdx_327_, v_t_328_, v_h_329_, v_k_330_);
lean_dec(v_t_328_);
lean_dec(v_ctorIdx_327_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim___redArg(lean_object* v_t_332_, lean_object* v_stark_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_332_, v_stark_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim___redArg___boxed(lean_object* v_t_335_, lean_object* v_stark_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_workspace_VmVerifier_VmStarkProofError_stark_elim___redArg(v_t_335_, v_stark_336_);
lean_dec(v_t_335_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim(lean_object* v_motive_338_, lean_object* v_t_339_, lean_object* v_h_340_, lean_object* v_stark_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_339_, v_stark_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_stark_elim___boxed(lean_object* v_motive_343_, lean_object* v_t_344_, lean_object* v_h_345_, lean_object* v_stark_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_workspace_VmVerifier_VmStarkProofError_stark_elim(v_motive_343_, v_t_344_, v_h_345_, v_stark_346_);
lean_dec(v_t_344_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___redArg(lean_object* v_t_348_, lean_object* v_publicValues_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_348_, v_publicValues_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___redArg___boxed(lean_object* v_t_351_, lean_object* v_publicValues_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___redArg(v_t_351_, v_publicValues_352_);
lean_dec(v_t_351_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim(lean_object* v_motive_354_, lean_object* v_t_355_, lean_object* v_h_356_, lean_object* v_publicValues_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_workspace_VmVerifier_VmStarkProofError_ctorElim___redArg(v_t_355_, v_publicValues_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim___boxed(lean_object* v_motive_359_, lean_object* v_t_360_, lean_object* v_h_361_, lean_object* v_publicValues_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_workspace_VmVerifier_VmStarkProofError_publicValues_elim(v_motive_359_, v_t_360_, v_h_361_, v_publicValues_362_);
lean_dec(v_t_360_);
return v_res_363_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instInhabitedVmStarkProofError_default(void){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lean_box(1);
return v___x_364_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_instInhabitedVmStarkProofError(void){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lean_box(1);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_vmStarkProtocolConfig(lean_object* v_vk_368_){
_start:
{
lean_object* v_internalRecursiveVk_369_; lean_object* v_inner_370_; lean_object* v_params_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_380_; 
v_internalRecursiveVk_369_ = lean_ctor_get(v_vk_368_, 0);
lean_inc_ref(v_internalRecursiveVk_369_);
lean_dec_ref(v_vk_368_);
v_inner_370_ = lean_ctor_get(v_internalRecursiveVk_369_, 0);
lean_inc_ref(v_inner_370_);
lean_dec_ref(v_internalRecursiveVk_369_);
v_params_371_ = lean_ctor_get(v_inner_370_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v_inner_370_);
if (v_isSharedCheck_380_ == 0)
{
lean_object* v_unused_381_; lean_object* v_unused_382_; 
v_unused_381_ = lean_ctor_get(v_inner_370_, 2);
lean_dec(v_unused_381_);
v_unused_382_ = lean_ctor_get(v_inner_370_, 1);
lean_dec(v_unused_382_);
v___x_373_ = v_inner_370_;
v_isShared_374_ = v_isSharedCheck_380_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_params_371_);
lean_dec(v_inner_370_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_380_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_378_; 
v___x_375_ = ((lean_object*)(lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__0));
v___x_376_ = ((lean_object*)(lp_workspace_VmVerifier_vmStarkProtocolConfig___closed__1));
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 2, v___x_376_);
lean_ctor_set(v___x_373_, 1, v___x_375_);
v___x_378_ = v___x_373_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v_params_371_);
lean_ctor_set(v_reuseFailAlloc_379_, 1, v___x_375_);
lean_ctor_set(v_reuseFailAlloc_379_, 2, v___x_376_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
return v___x_378_;
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProofInner(lean_object* v_vk_386_, lean_object* v_proof_387_){
_start:
{
lean_object* v_internalRecursiveVk_388_; lean_object* v_inner_389_; lean_object* v___f_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v_internalRecursiveVk_388_ = lean_ctor_get(v_vk_386_, 0);
lean_inc_ref(v_internalRecursiveVk_388_);
v_inner_389_ = lean_ctor_get(v_proof_387_, 0);
lean_inc_ref(v_inner_389_);
lean_dec_ref(v_proof_387_);
v___f_390_ = ((lean_object*)(lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__0));
v___x_391_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_RawInstances_rawFieldOpsEF;
v___x_392_ = lp_workspace_VmVerifier_vmStarkProtocolConfig(v_vk_386_);
v___x_393_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_init___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__0;
v___x_394_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___at___00Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2_spec__1(v___x_391_, v___f_390_, v___x_392_, v_internalRecursiveVk_388_, v_inner_389_, v___x_393_);
if (lean_obj_tag(v___x_394_) == 0)
{
lean_object* v_a_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_402_; 
v_a_395_ = lean_ctor_get(v___x_394_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_394_);
if (v_isSharedCheck_402_ == 0)
{
v___x_397_ = v___x_394_;
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_a_395_);
lean_dec(v___x_394_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_400_; 
if (v_isShared_398_ == 0)
{
v___x_400_ = v___x_397_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_a_395_);
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
lean_object* v___x_403_; 
lean_dec_ref_known(v___x_394_, 1);
v___x_403_ = ((lean_object*)(lp_workspace_VmVerifier_verifyVmStarkProofInner___closed__1));
return v___x_403_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyDecodedVmStarkProof(lean_object* v_vk_408_, lean_object* v_proof_409_){
_start:
{
lean_object* v___x_410_; 
lean_inc_ref(v_proof_409_);
lean_inc_ref(v_vk_408_);
v___x_410_ = lp_workspace_VmVerifier_verifyVmStarkProofInner(v_vk_408_, v_proof_409_);
if (lean_obj_tag(v___x_410_) == 0)
{
lean_object* v_a_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_420_; 
lean_dec_ref(v_proof_409_);
lean_dec_ref(v_vk_408_);
v_a_411_ = lean_ctor_get(v___x_410_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_420_ == 0)
{
v___x_413_ = v___x_410_;
v_isShared_414_ = v_isSharedCheck_420_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_a_411_);
lean_dec(v___x_410_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_420_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
lean_object* v___x_415_; uint8_t v___x_416_; lean_object* v___x_418_; 
v___x_415_ = lean_alloc_ctor(0, 0, 1);
v___x_416_ = lean_unbox(v_a_411_);
lean_dec(v_a_411_);
lean_ctor_set_uint8(v___x_415_, 0, v___x_416_);
if (v_isShared_414_ == 0)
{
lean_ctor_set(v___x_413_, 0, v___x_415_);
v___x_418_ = v___x_413_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___x_415_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
else
{
uint8_t v___x_421_; 
lean_dec_ref_known(v___x_410_, 1);
v___x_421_ = lp_workspace_VmVerifier_verifyVmStarkProofPvs(v_vk_408_, v_proof_409_);
lean_dec_ref(v_vk_408_);
if (v___x_421_ == 0)
{
lean_object* v___x_422_; 
v___x_422_ = ((lean_object*)(lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__0));
return v___x_422_;
}
else
{
lean_object* v___x_423_; 
v___x_423_ = ((lean_object*)(lp_workspace_VmVerifier_verifyDecodedVmStarkProof___closed__1));
return v___x_423_;
}
}
}
}
LEAN_EXPORT uint32_t lp_workspace_VmVerifier_decodeVmStarkProof___lam__0(lean_object* v_error_424_){
_start:
{
lean_object* v___x_425_; uint32_t v___x_426_; 
v___x_425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_425_, 0, v_error_424_);
v___x_426_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_exitCode(v___x_425_);
lean_dec_ref_known(v___x_425_, 1);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decodeVmStarkProof___lam__0___boxed(lean_object* v_error_427_){
_start:
{
uint32_t v_res_428_; lean_object* v_r_429_; 
v_res_428_ = lp_workspace_VmVerifier_decodeVmStarkProof___lam__0(v_error_427_);
v_r_429_ = lean_box_uint32(v_res_428_);
return v_r_429_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_decodeVmStarkProof(lean_object* v_vkBytes_430_, lean_object* v_baselineBytes_431_, lean_object* v_proofBytes_432_, lean_object* v_pvBytes_433_, lean_object* v_userPvsBytes_434_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_decodeAll(v_vkBytes_430_, v_proofBytes_432_, v_pvBytes_433_);
if (lean_obj_tag(v___x_435_) == 0)
{
lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_445_; 
lean_dec_ref(v_userPvsBytes_434_);
lean_dec_ref(v_baselineBytes_431_);
v_a_436_ = lean_ctor_get(v___x_435_, 0);
v_isSharedCheck_445_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_445_ == 0)
{
v___x_438_ = v___x_435_;
v_isShared_439_ = v_isSharedCheck_445_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_435_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_445_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
uint32_t v___x_440_; lean_object* v___x_441_; lean_object* v___x_443_; 
v___x_440_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_exitCode(v_a_436_);
lean_dec(v_a_436_);
v___x_441_ = lean_box_uint32(v___x_440_);
if (v_isShared_439_ == 0)
{
lean_ctor_set(v___x_438_, 0, v___x_441_);
v___x_443_ = v___x_438_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v___x_441_);
v___x_443_ = v_reuseFailAlloc_444_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
return v___x_443_;
}
}
}
else
{
lean_object* v_a_446_; lean_object* v_snd_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_495_; 
v_a_446_ = lean_ctor_get(v___x_435_, 0);
lean_inc(v_a_446_);
lean_dec_ref_known(v___x_435_, 1);
v_snd_447_ = lean_ctor_get(v_a_446_, 1);
v_isSharedCheck_495_ = !lean_is_exclusive(v_a_446_);
if (v_isSharedCheck_495_ == 0)
{
lean_object* v_unused_496_; 
v_unused_496_ = lean_ctor_get(v_a_446_, 0);
lean_dec(v_unused_496_);
v___x_449_ = v_a_446_;
v_isShared_450_ = v_isSharedCheck_495_;
goto v_resetjp_448_;
}
else
{
lean_inc(v_snd_447_);
lean_dec(v_a_446_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_495_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v_fst_451_; lean_object* v_snd_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_494_; 
v_fst_451_ = lean_ctor_get(v_snd_447_, 0);
v_snd_452_ = lean_ctor_get(v_snd_447_, 1);
v_isSharedCheck_494_ = !lean_is_exclusive(v_snd_447_);
if (v_isSharedCheck_494_ == 0)
{
v___x_454_ = v_snd_447_;
v_isShared_455_ = v_isSharedCheck_494_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_snd_452_);
lean_inc(v_fst_451_);
lean_dec(v_snd_447_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_494_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v___x_456_; 
v___x_456_ = lp_workspace_VmVerifier_Spec_Wire_readBaseline(v_baselineBytes_431_);
if (lean_obj_tag(v___x_456_) == 0)
{
lean_object* v_a_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_466_; 
lean_del_object(v___x_454_);
lean_dec(v_snd_452_);
lean_dec(v_fst_451_);
lean_del_object(v___x_449_);
lean_dec_ref(v_userPvsBytes_434_);
v_a_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_466_ == 0)
{
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_466_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_a_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_466_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
uint32_t v___x_461_; lean_object* v___x_462_; lean_object* v___x_464_; 
v___x_461_ = lp_workspace_VmVerifier_decodeVmStarkProof___lam__0(v_a_457_);
v___x_462_ = lean_box_uint32(v___x_461_);
if (v_isShared_460_ == 0)
{
lean_ctor_set(v___x_459_, 0, v___x_462_);
v___x_464_ = v___x_459_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v___x_462_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
}
else
{
lean_object* v_a_467_; lean_object* v___x_468_; 
v_a_467_ = lean_ctor_get(v___x_456_, 0);
lean_inc(v_a_467_);
lean_dec_ref_known(v___x_456_, 1);
v___x_468_ = lp_workspace_VmVerifier_Spec_Wire_readUserPvsProof(v_userPvsBytes_434_);
if (lean_obj_tag(v___x_468_) == 0)
{
lean_object* v_a_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_478_; 
lean_dec(v_a_467_);
lean_del_object(v___x_454_);
lean_dec(v_snd_452_);
lean_dec(v_fst_451_);
lean_del_object(v___x_449_);
v_a_469_ = lean_ctor_get(v___x_468_, 0);
v_isSharedCheck_478_ = !lean_is_exclusive(v___x_468_);
if (v_isSharedCheck_478_ == 0)
{
v___x_471_ = v___x_468_;
v_isShared_472_ = v_isSharedCheck_478_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_a_469_);
lean_dec(v___x_468_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_478_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
uint32_t v___x_473_; lean_object* v___x_474_; lean_object* v___x_476_; 
v___x_473_ = lp_workspace_VmVerifier_decodeVmStarkProof___lam__0(v_a_469_);
v___x_474_ = lean_box_uint32(v___x_473_);
if (v_isShared_472_ == 0)
{
lean_ctor_set(v___x_471_, 0, v___x_474_);
v___x_476_ = v___x_471_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v___x_474_);
v___x_476_ = v_reuseFailAlloc_477_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
return v___x_476_;
}
}
}
else
{
lean_object* v_a_479_; lean_object* v___x_481_; uint8_t v_isShared_482_; uint8_t v_isSharedCheck_493_; 
v_a_479_ = lean_ctor_get(v___x_468_, 0);
v_isSharedCheck_493_ = !lean_is_exclusive(v___x_468_);
if (v_isSharedCheck_493_ == 0)
{
v___x_481_ = v___x_468_;
v_isShared_482_ = v_isSharedCheck_493_;
goto v_resetjp_480_;
}
else
{
lean_inc(v_a_479_);
lean_dec(v___x_468_);
v___x_481_ = lean_box(0);
v_isShared_482_ = v_isSharedCheck_493_;
goto v_resetjp_480_;
}
v_resetjp_480_:
{
lean_object* v___x_484_; 
if (v_isShared_450_ == 0)
{
lean_ctor_set(v___x_449_, 1, v_a_467_);
lean_ctor_set(v___x_449_, 0, v_fst_451_);
v___x_484_ = v___x_449_;
goto v_reusejp_483_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v_fst_451_);
lean_ctor_set(v_reuseFailAlloc_492_, 1, v_a_467_);
v___x_484_ = v_reuseFailAlloc_492_;
goto v_reusejp_483_;
}
v_reusejp_483_:
{
lean_object* v___x_485_; lean_object* v___x_487_; 
v___x_485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_485_, 0, v_snd_452_);
lean_ctor_set(v___x_485_, 1, v_a_479_);
if (v_isShared_455_ == 0)
{
lean_ctor_set(v___x_454_, 1, v___x_485_);
lean_ctor_set(v___x_454_, 0, v___x_484_);
v___x_487_ = v___x_454_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v___x_484_);
lean_ctor_set(v_reuseFailAlloc_491_, 1, v___x_485_);
v___x_487_ = v_reuseFailAlloc_491_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
lean_object* v___x_489_; 
if (v_isShared_482_ == 0)
{
lean_ctor_set(v___x_481_, 0, v___x_487_);
v___x_489_ = v___x_481_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v___x_487_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
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
static lean_object* _init_lp_workspace_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1(void){
_start:
{
uint32_t v___x_497_; lean_object* v___x_498_; 
v___x_497_ = 12;
v___x_498_ = lean_box_uint32(v___x_497_);
return v___x_498_;
}
}
static lean_object* _init_lp_workspace_VmVerifier_verifyVmStarkProof___closed__0(void){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_499_ = lp_workspace_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1;
v___x_500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_workspace_VmVerifier_verifyVmStarkProof(lean_object* v_vkBytes_501_, lean_object* v_baselineBytes_502_, lean_object* v_proofBytes_503_, lean_object* v_pvBytes_504_, lean_object* v_userPvsBytes_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_workspace_VmVerifier_decodeVmStarkProof(v_vkBytes_501_, v_baselineBytes_502_, v_proofBytes_503_, v_pvBytes_504_, v_userPvsBytes_505_);
if (lean_obj_tag(v___x_506_) == 0)
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
v_a_507_ = lean_ctor_get(v___x_506_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_506_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_506_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_506_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
else
{
lean_object* v_a_515_; lean_object* v_fst_516_; lean_object* v_snd_517_; lean_object* v___x_518_; 
v_a_515_ = lean_ctor_get(v___x_506_, 0);
lean_inc(v_a_515_);
lean_dec_ref_known(v___x_506_, 1);
v_fst_516_ = lean_ctor_get(v_a_515_, 0);
lean_inc(v_fst_516_);
v_snd_517_ = lean_ctor_get(v_a_515_, 1);
lean_inc(v_snd_517_);
lean_dec(v_a_515_);
v___x_518_ = lp_workspace_VmVerifier_verifyDecodedVmStarkProof(v_fst_516_, v_snd_517_);
if (lean_obj_tag(v___x_518_) == 0)
{
lean_object* v_a_519_; lean_object* v___x_521_; uint8_t v_isShared_522_; uint8_t v_isSharedCheck_537_; 
v_a_519_ = lean_ctor_get(v___x_518_, 0);
v_isSharedCheck_537_ = !lean_is_exclusive(v___x_518_);
if (v_isSharedCheck_537_ == 0)
{
v___x_521_ = v___x_518_;
v_isShared_522_ = v_isSharedCheck_537_;
goto v_resetjp_520_;
}
else
{
lean_inc(v_a_519_);
lean_dec(v___x_518_);
v___x_521_ = lean_box(0);
v_isShared_522_ = v_isSharedCheck_537_;
goto v_resetjp_520_;
}
v_resetjp_520_:
{
if (lean_obj_tag(v_a_519_) == 0)
{
uint8_t v_error_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_535_; 
v_error_523_ = lean_ctor_get_uint8(v_a_519_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v_a_519_);
if (v_isSharedCheck_535_ == 0)
{
v___x_525_ = v_a_519_;
v_isShared_526_ = v_isSharedCheck_535_;
goto v_resetjp_524_;
}
else
{
lean_dec(v_a_519_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_535_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
lean_ctor_set_tag(v___x_525_, 1);
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v_reuseFailAlloc_534_, 0, v_error_523_);
v___x_528_ = v_reuseFailAlloc_534_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
uint32_t v___x_529_; lean_object* v___x_530_; lean_object* v___x_532_; 
v___x_529_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_exitCode(v___x_528_);
lean_dec_ref(v___x_528_);
v___x_530_ = lean_box_uint32(v___x_529_);
if (v_isShared_522_ == 0)
{
lean_ctor_set(v___x_521_, 0, v___x_530_);
v___x_532_ = v___x_521_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v___x_530_);
v___x_532_ = v_reuseFailAlloc_533_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
return v___x_532_;
}
}
}
}
else
{
lean_object* v___x_536_; 
lean_del_object(v___x_521_);
v___x_536_ = lean_obj_once(&lp_workspace_VmVerifier_verifyVmStarkProof___closed__0, &lp_workspace_VmVerifier_verifyVmStarkProof___closed__0_once, _init_lp_workspace_VmVerifier_verifyVmStarkProof___closed__0);
return v___x_536_;
}
}
}
else
{
lean_object* v_a_538_; lean_object* v___x_540_; uint8_t v_isShared_541_; uint8_t v_isSharedCheck_545_; 
v_a_538_ = lean_ctor_get(v___x_518_, 0);
v_isSharedCheck_545_ = !lean_is_exclusive(v___x_518_);
if (v_isSharedCheck_545_ == 0)
{
v___x_540_ = v___x_518_;
v_isShared_541_ = v_isSharedCheck_545_;
goto v_resetjp_539_;
}
else
{
lean_inc(v_a_538_);
lean_dec(v___x_518_);
v___x_540_ = lean_box(0);
v_isShared_541_ = v_isSharedCheck_545_;
goto v_resetjp_539_;
}
v_resetjp_539_:
{
lean_object* v___x_543_; 
if (v_isShared_541_ == 0)
{
v___x_543_ = v___x_540_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v_a_538_);
v___x_543_ = v_reuseFailAlloc_544_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
return v___x_543_;
}
}
}
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_RawInstances(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Main(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_VerifierBabyBearPoseidon2(uint8_t builtin);
lean_object* initialize_workspace_VmVerifier_Spec_Wire(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_workspace_VmVerifier_Spec_Runtime(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dformal_Fundamentals_Poseidon2_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_RawInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_VerifierBabyBearPoseidon2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_workspace_VmVerifier_Spec_Wire(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_workspace_VmVerifier_rawZero = _init_lp_workspace_VmVerifier_rawZero();
lean_mark_persistent(lp_workspace_VmVerifier_rawZero);
lp_workspace_VmVerifier_rawOne = _init_lp_workspace_VmVerifier_rawOne();
lean_mark_persistent(lp_workspace_VmVerifier_rawOne);
lp_workspace_VmVerifier_rawZeroDigest = _init_lp_workspace_VmVerifier_rawZeroDigest();
lean_mark_persistent(lp_workspace_VmVerifier_rawZeroDigest);
lp_workspace_VmVerifier_instInhabitedVmStarkProofError_default = _init_lp_workspace_VmVerifier_instInhabitedVmStarkProofError_default();
lean_mark_persistent(lp_workspace_VmVerifier_instInhabitedVmStarkProofError_default);
lp_workspace_VmVerifier_instInhabitedVmStarkProofError = _init_lp_workspace_VmVerifier_instInhabitedVmStarkProofError();
lean_mark_persistent(lp_workspace_VmVerifier_instInhabitedVmStarkProofError);
lp_workspace_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1 = _init_lp_workspace_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1();
lean_mark_persistent(lp_workspace_VmVerifier_verifyVmStarkProof___closed__0___boxed__const__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
