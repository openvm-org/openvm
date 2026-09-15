// Lean compiler output
// Module: Swirl.Protocol.Noninteractive.Verifier.Runtime.Whir
// Imports: public import Init public meta import Init public import Swirl.Protocol.Noninteractive.Core public import Swirl.Protocol.Noninteractive.Ops public import Swirl.Protocol.Noninteractive.Runtime.Core public import Swirl.Protocol.Noninteractive.Verifier.Runtime.Common
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
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_pure(lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___boxed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_ExtensionEncoding_extToWords___redArg(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observePowWitness___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_kWhir(lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleBits___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_numWhirRounds(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeCommit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_logStackedHeight(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_flatMapTR_go___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_logFinalPolyLen(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___boxed(lean_object**);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__3_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__4_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__5_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__5_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__6_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__7_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__7_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__10_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__11_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__4, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__12_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__7, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__13_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__9, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__14_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_map, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__15_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__15_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__11_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__16_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_pure, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__17_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__16_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__17_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__12_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__13_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__14_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__18_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_bind, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__19 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__19_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__18_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__19_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__20 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__20_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__22 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__22_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__23 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__23_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__1___boxed(lean_object**);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_lengthTR___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__3_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__23_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress___redArg(lean_object* v_inst_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_dec_ref(v_inst_1_);
return v_x_2_;
}
else
{
lean_object* v_tail_3_; 
v_tail_3_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_tail_3_);
if (lean_obj_tag(v_tail_3_) == 0)
{
lean_dec_ref(v_inst_1_);
return v_x_2_;
}
else
{
lean_object* v_head_4_; lean_object* v_head_5_; lean_object* v_tail_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_16_; 
v_head_4_ = lean_ctor_get(v_x_2_, 0);
lean_inc(v_head_4_);
lean_dec_ref_known(v_x_2_, 2);
v_head_5_ = lean_ctor_get(v_tail_3_, 0);
v_tail_6_ = lean_ctor_get(v_tail_3_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v_tail_3_);
if (v_isSharedCheck_16_ == 0)
{
v___x_8_ = v_tail_3_;
v_isShared_9_ = v_isSharedCheck_16_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_tail_6_);
lean_inc(v_head_5_);
lean_dec(v_tail_3_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_16_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v_compress_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_14_; 
v_compress_10_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_compress_10_);
v___x_11_ = lean_apply_2(v_compress_10_, v_head_4_, v_head_5_);
v___x_12_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress___redArg(v_inst_1_, v_tail_6_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___x_12_);
lean_ctor_set(v___x_8_, 0, v___x_11_);
v___x_14_ = v___x_8_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v___x_11_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v___x_12_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress(lean_object* v_F_17_, lean_object* v_Digest_18_, lean_object* v_inst_19_, lean_object* v_x_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress___redArg(v_inst_19_, v_x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress_match__1_splitter___redArg(lean_object* v_x_22_, lean_object* v_h__1_23_, lean_object* v_h__2_24_, lean_object* v_h__3_25_){
_start:
{
if (lean_obj_tag(v_x_22_) == 0)
{
lean_object* v___x_26_; lean_object* v___x_27_; 
lean_dec(v_h__3_25_);
lean_dec(v_h__2_24_);
v___x_26_ = lean_box(0);
v___x_27_ = lean_apply_1(v_h__1_23_, v___x_26_);
return v___x_27_;
}
else
{
lean_object* v_tail_28_; 
lean_dec(v_h__1_23_);
v_tail_28_ = lean_ctor_get(v_x_22_, 1);
if (lean_obj_tag(v_tail_28_) == 0)
{
lean_object* v_head_29_; lean_object* v___x_30_; 
lean_dec(v_h__3_25_);
v_head_29_ = lean_ctor_get(v_x_22_, 0);
lean_inc(v_head_29_);
lean_dec_ref_known(v_x_22_, 2);
v___x_30_ = lean_apply_1(v_h__2_24_, v_head_29_);
return v___x_30_;
}
else
{
lean_object* v___x_31_; 
lean_dec(v_h__2_24_);
v___x_31_ = lean_apply_3(v_h__3_25_, v_x_22_, lean_box(0), lean_box(0));
return v___x_31_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress_match__1_splitter(lean_object* v_Digest_32_, lean_object* v_motive_33_, lean_object* v_x_34_, lean_object* v_h__1_35_, lean_object* v_h__2_36_, lean_object* v_h__3_37_){
_start:
{
if (lean_obj_tag(v_x_34_) == 0)
{
lean_object* v___x_38_; lean_object* v___x_39_; 
lean_dec(v_h__3_37_);
lean_dec(v_h__2_36_);
v___x_38_ = lean_box(0);
v___x_39_ = lean_apply_1(v_h__1_35_, v___x_38_);
return v___x_39_;
}
else
{
lean_object* v_tail_40_; 
lean_dec(v_h__1_35_);
v_tail_40_ = lean_ctor_get(v_x_34_, 1);
if (lean_obj_tag(v_tail_40_) == 0)
{
lean_object* v_head_41_; lean_object* v___x_42_; 
lean_dec(v_h__3_37_);
v_head_41_ = lean_ctor_get(v_x_34_, 0);
lean_inc(v_head_41_);
lean_dec_ref_known(v_x_34_, 2);
v___x_42_ = lean_apply_1(v_h__2_36_, v_head_41_);
return v___x_42_;
}
else
{
lean_object* v___x_43_; 
lean_dec(v_h__2_36_);
v___x_43_ = lean_apply_3(v_h__3_37_, v_x_34_, lean_box(0), lean_box(0));
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress___redArg(lean_object* v_inst_44_, lean_object* v_x_45_){
_start:
{
if (lean_obj_tag(v_x_45_) == 0)
{
lean_object* v___x_46_; 
lean_dec_ref(v_inst_44_);
v___x_46_ = lean_box(0);
return v___x_46_;
}
else
{
lean_object* v_tail_47_; 
v_tail_47_ = lean_ctor_get(v_x_45_, 1);
if (lean_obj_tag(v_tail_47_) == 0)
{
lean_object* v_head_48_; lean_object* v___x_49_; 
lean_dec_ref(v_inst_44_);
v_head_48_ = lean_ctor_get(v_x_45_, 0);
lean_inc(v_head_48_);
lean_dec_ref_known(v_x_45_, 2);
v___x_49_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_49_, 0, v_head_48_);
return v___x_49_;
}
else
{
lean_object* v___x_50_; 
lean_inc_ref(v_inst_44_);
v___x_50_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_pairCompress___redArg(v_inst_44_, v_x_45_);
v_x_45_ = v___x_50_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress(lean_object* v_F_52_, lean_object* v_Digest_53_, lean_object* v_inst_54_, lean_object* v_x_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress___redArg(v_inst_54_, v_x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___lam__0(lean_object* v_inst_57_, lean_object* v_acc_58_, lean_object* v_sibling_59_){
_start:
{
lean_object* v_fst_60_; lean_object* v_snd_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_80_; 
v_fst_60_ = lean_ctor_get(v_acc_58_, 0);
v_snd_61_ = lean_ctor_get(v_acc_58_, 1);
v_isSharedCheck_80_ = !lean_is_exclusive(v_acc_58_);
if (v_isSharedCheck_80_ == 0)
{
v___x_63_ = v_acc_58_;
v_isShared_64_ = v_isSharedCheck_80_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_snd_61_);
lean_inc(v_fst_60_);
lean_dec(v_acc_58_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_80_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v___y_66_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; uint8_t v___x_75_; 
v___x_72_ = lean_unsigned_to_nat(2u);
v___x_73_ = lean_nat_mod(v_snd_61_, v___x_72_);
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = lean_nat_dec_eq(v___x_73_, v___x_74_);
lean_dec(v___x_73_);
if (v___x_75_ == 0)
{
lean_object* v_compress_76_; lean_object* v___x_77_; 
v_compress_76_ = lean_ctor_get(v_inst_57_, 1);
lean_inc(v_compress_76_);
lean_dec_ref(v_inst_57_);
v___x_77_ = lean_apply_2(v_compress_76_, v_sibling_59_, v_fst_60_);
v___y_66_ = v___x_77_;
goto v___jp_65_;
}
else
{
lean_object* v_compress_78_; lean_object* v___x_79_; 
v_compress_78_ = lean_ctor_get(v_inst_57_, 1);
lean_inc(v_compress_78_);
lean_dec_ref(v_inst_57_);
v___x_79_ = lean_apply_2(v_compress_78_, v_fst_60_, v_sibling_59_);
v___y_66_ = v___x_79_;
goto v___jp_65_;
}
v___jp_65_:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_70_; 
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = lean_nat_shiftr(v_snd_61_, v___x_67_);
lean_dec(v_snd_61_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 1, v___x_68_);
lean_ctor_set(v___x_63_, 0, v___y_66_);
v___x_70_ = v___x_63_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_71_; 
v_reuseFailAlloc_71_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_71_, 0, v___y_66_);
lean_ctor_set(v_reuseFailAlloc_71_, 1, v___x_68_);
v___x_70_ = v_reuseFailAlloc_71_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
return v___x_70_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg(lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_root_88_, lean_object* v_idx_89_, lean_object* v_leafHash_90_, lean_object* v_merkleProof_91_){
_start:
{
lean_object* v___f_92_; lean_object* v___x_93_; lean_object* v_final_94_; lean_object* v_fst_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___f_92_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___lam__0), 3, 1);
lean_closure_set(v___f_92_, 0, v_inst_86_);
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v_leafHash_90_);
lean_ctor_set(v___x_93_, 1, v_idx_89_);
v_final_94_ = l_List_foldl___redArg(v___f_92_, v___x_93_, v_merkleProof_91_);
v_fst_95_ = lean_ctor_get(v_final_94_, 0);
lean_inc(v_fst_95_);
lean_dec(v_final_94_);
v___x_96_ = lean_apply_2(v_inst_87_, v_fst_95_, v_root_88_);
v___x_97_ = lean_unbox(v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; 
v___x_98_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__0));
return v___x_98_;
}
else
{
lean_object* v___x_99_; 
v___x_99_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg___closed__1));
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify(lean_object* v_F_100_, lean_object* v_Digest_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_root_104_, lean_object* v_idx_105_, lean_object* v_leafHash_106_, lean_object* v_merkleProof_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg(v_inst_102_, v_inst_103_, v_root_104_, v_idx_105_, v_leafHash_106_, v_merkleProof_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(lean_object* v_values_112_, lean_object* v_idx_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = l_List_get_x3fInternal___redArg(v_values_112_, v_idx_113_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v___x_115_; 
v___x_115_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0));
return v___x_115_;
}
else
{
lean_object* v_val_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_123_; 
v_val_116_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_123_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_123_ == 0)
{
v___x_118_ = v___x_114_;
v_isShared_119_ = v_isSharedCheck_123_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_val_116_);
lean_dec(v___x_114_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_123_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
lean_object* v___x_121_; 
if (v_isShared_119_ == 0)
{
v___x_121_ = v___x_118_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_122_; 
v_reuseFailAlloc_122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_122_, 0, v_val_116_);
v___x_121_ = v_reuseFailAlloc_122_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
return v___x_121_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___boxed(lean_object* v_values_124_, lean_object* v_idx_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_values_124_, v_idx_125_);
lean_dec(v_values_124_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem(lean_object* v_00_u03b1_127_, lean_object* v_values_128_, lean_object* v_idx_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_values_128_, v_idx_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___boxed(lean_object* v_00_u03b1_131_, lean_object* v_values_132_, lean_object* v_idx_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem(v_00_u03b1_131_, v_values_132_, v_idx_133_);
lean_dec(v_values_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows_spec__0___redArg(lean_object* v_fo_135_, lean_object* v_algMap_136_, lean_object* v_x_137_, lean_object* v_x_138_){
_start:
{
if (lean_obj_tag(v_x_138_) == 0)
{
lean_dec(v_algMap_136_);
lean_dec_ref(v_fo_135_);
return v_x_137_;
}
else
{
lean_object* v_toRingOps_139_; lean_object* v_toSemiringOps_140_; lean_object* v_head_141_; lean_object* v_tail_142_; lean_object* v_add_143_; lean_object* v_mul_144_; lean_object* v_fst_145_; lean_object* v_snd_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v_toRingOps_139_ = lean_ctor_get(v_fo_135_, 0);
v_toSemiringOps_140_ = lean_ctor_get(v_toRingOps_139_, 0);
v_head_141_ = lean_ctor_get(v_x_138_, 0);
lean_inc(v_head_141_);
v_tail_142_ = lean_ctor_get(v_x_138_, 1);
lean_inc(v_tail_142_);
lean_dec_ref_known(v_x_138_, 2);
v_add_143_ = lean_ctor_get(v_toSemiringOps_140_, 3);
v_mul_144_ = lean_ctor_get(v_toSemiringOps_140_, 4);
v_fst_145_ = lean_ctor_get(v_head_141_, 0);
lean_inc(v_fst_145_);
v_snd_146_ = lean_ctor_get(v_head_141_, 1);
lean_inc(v_snd_146_);
lean_dec(v_head_141_);
lean_inc(v_algMap_136_);
v___x_147_ = lean_apply_1(v_algMap_136_, v_snd_146_);
lean_inc(v_mul_144_);
v___x_148_ = lean_apply_2(v_mul_144_, v___x_147_, v_fst_145_);
lean_inc(v_add_143_);
v___x_149_ = lean_apply_2(v_add_143_, v_x_137_, v___x_148_);
v_x_137_ = v___x_149_;
v_x_138_ = v_tail_142_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___lam__0(lean_object* v_muPows_151_, lean_object* v_fo_152_, lean_object* v_algMap_153_, lean_object* v_cv_154_, lean_object* v_row_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_muPows_151_, v_row_155_);
v___x_157_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows_spec__0___redArg(v_fo_152_, v_algMap_153_, v_cv_154_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg(lean_object* v_fo_160_, lean_object* v_algMap_161_, lean_object* v_codewordVals_162_, lean_object* v_muPows_163_, lean_object* v_openedRows_164_){
_start:
{
lean_object* v___f_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v___f_165_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___lam__0), 5, 3);
lean_closure_set(v___f_165_, 0, v_muPows_163_);
lean_closure_set(v___f_165_, 1, v_fo_160_);
lean_closure_set(v___f_165_, 2, v_algMap_161_);
v___x_166_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0));
v___x_167_ = l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(v___f_165_, v_codewordVals_162_, v_openedRows_164_, v___x_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows(lean_object* v_F_168_, lean_object* v_EF_169_, lean_object* v_fo_170_, lean_object* v_algMap_171_, lean_object* v_codewordVals_172_, lean_object* v_muPows_173_, lean_object* v_openedRows_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg(v_fo_170_, v_algMap_171_, v_codewordVals_172_, v_muPows_173_, v_openedRows_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows_spec__0(lean_object* v_EF_176_, lean_object* v_F_177_, lean_object* v_fo_178_, lean_object* v_algMap_179_, lean_object* v_x_180_, lean_object* v_x_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows_spec__0___redArg(v_fo_178_, v_algMap_179_, v_x_180_, v_x_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg(lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_fo_188_, lean_object* v_algMap_189_, lean_object* v_queryIdx_190_, lean_object* v_index_191_, lean_object* v_codewordVals_192_, lean_object* v_muPows_193_, lean_object* v_commitments_194_, lean_object* v_widths_195_, lean_object* v_openedRowsLists_196_, lean_object* v_merkleProofsLists_197_){
_start:
{
if (lean_obj_tag(v_commitments_194_) == 0)
{
lean_dec(v_muPows_193_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
if (lean_obj_tag(v_widths_195_) == 0)
{
if (lean_obj_tag(v_openedRowsLists_196_) == 0)
{
if (lean_obj_tag(v_merkleProofsLists_197_) == 0)
{
lean_object* v___x_200_; 
v___x_200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_200_, 0, v_codewordVals_192_);
return v___x_200_;
}
else
{
lean_dec(v_codewordVals_192_);
goto v___jp_198_;
}
}
else
{
lean_dec(v_codewordVals_192_);
goto v___jp_198_;
}
}
else
{
lean_dec(v_widths_195_);
lean_dec(v_codewordVals_192_);
goto v___jp_198_;
}
}
else
{
if (lean_obj_tag(v_widths_195_) == 1)
{
if (lean_obj_tag(v_openedRowsLists_196_) == 1)
{
if (lean_obj_tag(v_merkleProofsLists_197_) == 1)
{
lean_object* v_head_201_; lean_object* v_tail_202_; lean_object* v_head_203_; lean_object* v_tail_204_; lean_object* v_head_205_; lean_object* v_tail_206_; lean_object* v_head_207_; lean_object* v_tail_208_; lean_object* v___x_209_; 
v_head_201_ = lean_ctor_get(v_commitments_194_, 0);
lean_inc(v_head_201_);
v_tail_202_ = lean_ctor_get(v_commitments_194_, 1);
lean_inc(v_tail_202_);
lean_dec_ref_known(v_commitments_194_, 2);
v_head_203_ = lean_ctor_get(v_widths_195_, 0);
lean_inc(v_head_203_);
v_tail_204_ = lean_ctor_get(v_widths_195_, 1);
lean_inc(v_tail_204_);
lean_dec_ref_known(v_widths_195_, 2);
v_head_205_ = lean_ctor_get(v_openedRowsLists_196_, 0);
v_tail_206_ = lean_ctor_get(v_openedRowsLists_196_, 1);
v_head_207_ = lean_ctor_get(v_merkleProofsLists_197_, 0);
v_tail_208_ = lean_ctor_get(v_merkleProofsLists_197_, 1);
lean_inc(v_queryIdx_190_);
v___x_209_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_head_205_, v_queryIdx_190_);
if (lean_obj_tag(v___x_209_) == 0)
{
lean_object* v___x_210_; 
lean_dec_ref_known(v___x_209_, 1);
lean_dec(v_tail_204_);
lean_dec(v_head_203_);
lean_dec(v_tail_202_);
lean_dec(v_head_201_);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
v___x_210_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0));
return v___x_210_;
}
else
{
lean_object* v_a_211_; lean_object* v___x_212_; 
v_a_211_ = lean_ctor_get(v___x_209_, 0);
lean_inc(v_a_211_);
lean_dec_ref_known(v___x_209_, 1);
lean_inc(v_queryIdx_190_);
v___x_212_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_head_207_, v_queryIdx_190_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_object* v___x_213_; 
lean_dec_ref_known(v___x_212_, 1);
lean_dec(v_a_211_);
lean_dec(v_tail_204_);
lean_dec(v_head_203_);
lean_dec(v_tail_202_);
lean_dec(v_head_201_);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
v___x_213_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0));
return v___x_213_;
}
else
{
lean_object* v_a_214_; lean_object* v_hashSlice_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v_a_214_ = lean_ctor_get(v___x_212_, 0);
lean_inc(v_a_214_);
lean_dec_ref_known(v___x_212_, 1);
v_hashSlice_215_ = lean_ctor_get(v_inst_186_, 0);
v___x_216_ = lean_box(0);
lean_inc(v_a_211_);
lean_inc(v_hashSlice_215_);
v___x_217_ = l_List_mapTR_loop___redArg(v_hashSlice_215_, v_a_211_, v___x_216_);
lean_inc_ref(v_inst_186_);
v___x_218_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress___redArg(v_inst_186_, v___x_217_);
if (lean_obj_tag(v___x_218_) == 0)
{
lean_object* v___x_219_; 
lean_dec(v_a_214_);
lean_dec(v_a_211_);
lean_dec(v_tail_204_);
lean_dec(v_head_203_);
lean_dec(v_tail_202_);
lean_dec(v_head_201_);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
v___x_219_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0));
return v___x_219_;
}
else
{
lean_object* v_val_220_; lean_object* v___x_221_; 
v_val_220_ = lean_ctor_get(v___x_218_, 0);
lean_inc(v_val_220_);
lean_dec_ref_known(v___x_218_, 1);
lean_inc(v_index_191_);
lean_inc_ref(v_inst_187_);
lean_inc_ref(v_inst_186_);
v___x_221_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg(v_inst_186_, v_inst_187_, v_head_201_, v_index_191_, v_val_220_, v_a_214_);
if (lean_obj_tag(v___x_221_) == 0)
{
lean_object* v___x_222_; 
lean_dec_ref_known(v___x_221_, 1);
lean_dec(v_a_211_);
lean_dec(v_tail_204_);
lean_dec(v_head_203_);
lean_dec(v_tail_202_);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
v___x_222_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0));
return v___x_222_;
}
else
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
lean_dec_ref_known(v___x_221_, 1);
v___x_223_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0));
lean_inc(v_head_203_);
lean_inc(v_muPows_193_);
v___x_224_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_muPows_193_, v_muPows_193_, v_head_203_, v___x_223_);
v___x_225_ = l_List_drop___redArg(v_head_203_, v_muPows_193_);
lean_dec(v_muPows_193_);
lean_inc(v_algMap_189_);
lean_inc_ref(v_fo_188_);
v___x_226_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg(v_fo_188_, v_algMap_189_, v_codewordVals_192_, v___x_224_, v_a_211_);
v_codewordVals_192_ = v___x_226_;
v_muPows_193_ = v___x_225_;
v_commitments_194_ = v_tail_202_;
v_widths_195_ = v_tail_204_;
v_openedRowsLists_196_ = v_tail_206_;
v_merkleProofsLists_197_ = v_tail_208_;
goto _start;
}
}
}
}
}
else
{
lean_dec_ref_known(v_widths_195_, 2);
lean_dec_ref_known(v_commitments_194_, 2);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
goto v___jp_198_;
}
}
else
{
lean_dec_ref_known(v_widths_195_, 2);
lean_dec_ref_known(v_commitments_194_, 2);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
goto v___jp_198_;
}
}
else
{
lean_dec_ref_known(v_commitments_194_, 2);
lean_dec(v_widths_195_);
lean_dec(v_muPows_193_);
lean_dec(v_codewordVals_192_);
lean_dec(v_index_191_);
lean_dec(v_queryIdx_190_);
lean_dec(v_algMap_189_);
lean_dec_ref(v_fo_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
goto v___jp_198_;
}
}
v___jp_198_:
{
lean_object* v___x_199_; 
v___x_199_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___closed__0));
return v___x_199_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg___boxed(lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_fo_230_, lean_object* v_algMap_231_, lean_object* v_queryIdx_232_, lean_object* v_index_233_, lean_object* v_codewordVals_234_, lean_object* v_muPows_235_, lean_object* v_commitments_236_, lean_object* v_widths_237_, lean_object* v_openedRowsLists_238_, lean_object* v_merkleProofsLists_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg(v_inst_228_, v_inst_229_, v_fo_230_, v_algMap_231_, v_queryIdx_232_, v_index_233_, v_codewordVals_234_, v_muPows_235_, v_commitments_236_, v_widths_237_, v_openedRowsLists_238_, v_merkleProofsLists_239_);
lean_dec(v_merkleProofsLists_239_);
lean_dec(v_openedRowsLists_238_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold(lean_object* v_F_241_, lean_object* v_EF_242_, lean_object* v_Digest_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_fo_246_, lean_object* v_algMap_247_, lean_object* v_queryIdx_248_, lean_object* v_index_249_, lean_object* v_codewordVals_250_, lean_object* v_muPows_251_, lean_object* v_commitments_252_, lean_object* v_widths_253_, lean_object* v_openedRowsLists_254_, lean_object* v_merkleProofsLists_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg(v_inst_244_, v_inst_245_, v_fo_246_, v_algMap_247_, v_queryIdx_248_, v_index_249_, v_codewordVals_250_, v_muPows_251_, v_commitments_252_, v_widths_253_, v_openedRowsLists_254_, v_merkleProofsLists_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___boxed(lean_object* v_F_257_, lean_object* v_EF_258_, lean_object* v_Digest_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_fo_262_, lean_object* v_algMap_263_, lean_object* v_queryIdx_264_, lean_object* v_index_265_, lean_object* v_codewordVals_266_, lean_object* v_muPows_267_, lean_object* v_commitments_268_, lean_object* v_widths_269_, lean_object* v_openedRowsLists_270_, lean_object* v_merkleProofsLists_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold(v_F_257_, v_EF_258_, v_Digest_259_, v_inst_260_, v_inst_261_, v_fo_262_, v_algMap_263_, v_queryIdx_264_, v_index_265_, v_codewordVals_266_, v_muPows_267_, v_commitments_268_, v_widths_269_, v_openedRowsLists_270_, v_merkleProofsLists_271_);
lean_dec(v_merkleProofsLists_271_);
lean_dec(v_openedRowsLists_270_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg(lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_fo_276_, lean_object* v_algMap_277_, lean_object* v_kWhir_278_, lean_object* v_queryIdx_279_, lean_object* v_index_280_, lean_object* v_ziRoot_281_, lean_object* v_alphasRound_282_, lean_object* v_muPows_283_, lean_object* v_commitments_284_, lean_object* v_widths_285_, lean_object* v_initialRoundOpenedRows_286_, lean_object* v_initialRoundMerkleProofs_287_){
_start:
{
lean_object* v_toRingOps_288_; lean_object* v_toSemiringOps_289_; lean_object* v_zero_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v_initialCodewordVals_293_; lean_object* v___x_294_; 
v_toRingOps_288_ = lean_ctor_get(v_fo_276_, 0);
v_toSemiringOps_289_ = lean_ctor_get(v_toRingOps_288_, 0);
v_zero_290_ = lean_ctor_get(v_toSemiringOps_289_, 0);
v___x_291_ = lean_unsigned_to_nat(2u);
v___x_292_ = lean_nat_pow(v___x_291_, v_kWhir_278_);
lean_inc(v_zero_290_);
v_initialCodewordVals_293_ = l_List_replicateTR___redArg(v___x_292_, v_zero_290_);
lean_inc_ref(v_fo_276_);
v___x_294_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQueryFold___redArg(v_inst_273_, v_inst_274_, v_fo_276_, v_algMap_277_, v_queryIdx_279_, v_index_280_, v_initialCodewordVals_293_, v_muPows_283_, v_commitments_284_, v_widths_285_, v_initialRoundOpenedRows_286_, v_initialRoundMerkleProofs_287_);
if (lean_obj_tag(v___x_294_) == 0)
{
lean_object* v___x_295_; 
lean_dec_ref_known(v___x_294_, 1);
lean_dec(v_alphasRound_282_);
lean_dec(v_ziRoot_281_);
lean_dec_ref(v_fo_276_);
lean_dec(v_inst_275_);
v___x_295_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0));
return v___x_295_;
}
else
{
lean_object* v_a_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_304_; 
v_a_296_ = lean_ctor_get(v___x_294_, 0);
v_isSharedCheck_304_ = !lean_is_exclusive(v___x_294_);
if (v_isSharedCheck_304_ == 0)
{
v___x_298_ = v___x_294_;
v_isShared_299_ = v_isSharedCheck_304_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_a_296_);
lean_dec(v___x_294_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_304_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v___x_300_; lean_object* v___x_302_; 
v___x_300_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg(v_fo_276_, v_inst_275_, v_a_296_, v_alphasRound_282_, v_ziRoot_281_);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 0, v___x_300_);
v___x_302_ = v___x_298_;
goto v_reusejp_301_;
}
else
{
lean_object* v_reuseFailAlloc_303_; 
v_reuseFailAlloc_303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_303_, 0, v___x_300_);
v___x_302_ = v_reuseFailAlloc_303_;
goto v_reusejp_301_;
}
v_reusejp_301_:
{
return v___x_302_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg___boxed(lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_fo_308_, lean_object* v_algMap_309_, lean_object* v_kWhir_310_, lean_object* v_queryIdx_311_, lean_object* v_index_312_, lean_object* v_ziRoot_313_, lean_object* v_alphasRound_314_, lean_object* v_muPows_315_, lean_object* v_commitments_316_, lean_object* v_widths_317_, lean_object* v_initialRoundOpenedRows_318_, lean_object* v_initialRoundMerkleProofs_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg(v_inst_305_, v_inst_306_, v_inst_307_, v_fo_308_, v_algMap_309_, v_kWhir_310_, v_queryIdx_311_, v_index_312_, v_ziRoot_313_, v_alphasRound_314_, v_muPows_315_, v_commitments_316_, v_widths_317_, v_initialRoundOpenedRows_318_, v_initialRoundMerkleProofs_319_);
lean_dec(v_initialRoundMerkleProofs_319_);
lean_dec(v_initialRoundOpenedRows_318_);
lean_dec(v_kWhir_310_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery(lean_object* v_F_321_, lean_object* v_EF_322_, lean_object* v_Digest_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_fo_327_, lean_object* v_algMap_328_, lean_object* v_kWhir_329_, lean_object* v_queryIdx_330_, lean_object* v_index_331_, lean_object* v_ziRoot_332_, lean_object* v_alphasRound_333_, lean_object* v_muPows_334_, lean_object* v_commitments_335_, lean_object* v_widths_336_, lean_object* v_initialRoundOpenedRows_337_, lean_object* v_initialRoundMerkleProofs_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg(v_inst_324_, v_inst_325_, v_inst_326_, v_fo_327_, v_algMap_328_, v_kWhir_329_, v_queryIdx_330_, v_index_331_, v_ziRoot_332_, v_alphasRound_333_, v_muPows_334_, v_commitments_335_, v_widths_336_, v_initialRoundOpenedRows_337_, v_initialRoundMerkleProofs_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___boxed(lean_object** _args){
lean_object* v_F_340_ = _args[0];
lean_object* v_EF_341_ = _args[1];
lean_object* v_Digest_342_ = _args[2];
lean_object* v_inst_343_ = _args[3];
lean_object* v_inst_344_ = _args[4];
lean_object* v_inst_345_ = _args[5];
lean_object* v_fo_346_ = _args[6];
lean_object* v_algMap_347_ = _args[7];
lean_object* v_kWhir_348_ = _args[8];
lean_object* v_queryIdx_349_ = _args[9];
lean_object* v_index_350_ = _args[10];
lean_object* v_ziRoot_351_ = _args[11];
lean_object* v_alphasRound_352_ = _args[12];
lean_object* v_muPows_353_ = _args[13];
lean_object* v_commitments_354_ = _args[14];
lean_object* v_widths_355_ = _args[15];
lean_object* v_initialRoundOpenedRows_356_ = _args[16];
lean_object* v_initialRoundMerkleProofs_357_ = _args[17];
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery(v_F_340_, v_EF_341_, v_Digest_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_fo_346_, v_algMap_347_, v_kWhir_348_, v_queryIdx_349_, v_index_350_, v_ziRoot_351_, v_alphasRound_352_, v_muPows_353_, v_commitments_354_, v_widths_355_, v_initialRoundOpenedRows_356_, v_initialRoundMerkleProofs_357_);
lean_dec(v_initialRoundMerkleProofs_357_);
lean_dec(v_initialRoundOpenedRows_356_);
lean_dec(v_kWhir_348_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg___lam__0(lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_v_361_){
_start:
{
lean_object* v_hashSlice_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v_hashSlice_362_ = lean_ctor_get(v_inst_359_, 0);
lean_inc(v_hashSlice_362_);
lean_dec_ref(v_inst_359_);
v___x_363_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_ExtensionEncoding_extToWords___redArg(v_inst_360_, v_v_361_);
v___x_364_ = lean_apply_1(v_hashSlice_362_, v___x_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg(lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_fo_369_, lean_object* v_index_370_, lean_object* v_ziRoot_371_, lean_object* v_alphasRound_372_, lean_object* v_codewordCommit_373_, lean_object* v_openedValues_374_, lean_object* v_merkleProof_375_){
_start:
{
lean_object* v___f_376_; lean_object* v___x_377_; lean_object* v_leafHashes_378_; lean_object* v___x_379_; 
lean_inc_ref_n(v_inst_365_, 2);
v___f_376_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg___lam__0), 3, 2);
lean_closure_set(v___f_376_, 0, v_inst_365_);
lean_closure_set(v___f_376_, 1, v_inst_367_);
v___x_377_ = lean_box(0);
lean_inc(v_openedValues_374_);
v_leafHashes_378_ = l_List_mapTR_loop___redArg(v___f_376_, v_openedValues_374_, v___x_377_);
v___x_379_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_MerkleHasher_treeCompress___redArg(v_inst_365_, v_leafHashes_378_);
if (lean_obj_tag(v___x_379_) == 0)
{
lean_object* v___x_380_; 
lean_dec(v_merkleProof_375_);
lean_dec(v_openedValues_374_);
lean_dec(v_codewordCommit_373_);
lean_dec(v_alphasRound_372_);
lean_dec(v_ziRoot_371_);
lean_dec(v_index_370_);
lean_dec_ref(v_fo_369_);
lean_dec(v_inst_368_);
lean_dec_ref(v_inst_366_);
lean_dec_ref(v_inst_365_);
v___x_380_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0));
return v___x_380_;
}
else
{
lean_object* v_val_381_; lean_object* v___x_382_; 
v_val_381_ = lean_ctor_get(v___x_379_, 0);
lean_inc(v_val_381_);
lean_dec_ref_known(v___x_379_, 1);
v___x_382_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_merkleVerify___redArg(v_inst_365_, v_inst_366_, v_codewordCommit_373_, v_index_370_, v_val_381_, v_merkleProof_375_);
if (lean_obj_tag(v___x_382_) == 0)
{
lean_object* v___x_383_; 
lean_dec_ref_known(v___x_382_, 1);
lean_dec(v_openedValues_374_);
lean_dec(v_alphasRound_372_);
lean_dec(v_ziRoot_371_);
lean_dec_ref(v_fo_369_);
lean_dec(v_inst_368_);
v___x_383_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg___closed__0));
return v___x_383_;
}
else
{
lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_391_; 
v_isSharedCheck_391_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_391_ == 0)
{
lean_object* v_unused_392_; 
v_unused_392_ = lean_ctor_get(v___x_382_, 0);
lean_dec(v_unused_392_);
v___x_385_ = v___x_382_;
v_isShared_386_ = v_isSharedCheck_391_;
goto v_resetjp_384_;
}
else
{
lean_dec(v___x_382_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_391_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_387_; lean_object* v___x_389_; 
v___x_387_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_binaryKFold___redArg(v_fo_369_, v_inst_368_, v_openedValues_374_, v_alphasRound_372_, v_ziRoot_371_);
if (v_isShared_386_ == 0)
{
lean_ctor_set(v___x_385_, 0, v___x_387_);
v___x_389_ = v___x_385_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v___x_387_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery(lean_object* v_F_393_, lean_object* v_EF_394_, lean_object* v_Digest_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_fo_400_, lean_object* v_index_401_, lean_object* v_ziRoot_402_, lean_object* v_alphasRound_403_, lean_object* v_codewordCommit_404_, lean_object* v_openedValues_405_, lean_object* v_merkleProof_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg(v_inst_396_, v_inst_397_, v_inst_398_, v_inst_399_, v_fo_400_, v_index_401_, v_ziRoot_402_, v_alphasRound_403_, v_codewordCommit_404_, v_openedValues_405_, v_merkleProof_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg(lean_object* v_fo_408_, lean_object* v_acc_409_, lean_object* v_gammaPow_410_, lean_object* v_z_411_, lean_object* v_slcLen_412_, lean_object* v_alphaSlc_413_, lean_object* v_finalPoly_414_){
_start:
{
lean_object* v_toRingOps_415_; lean_object* v_toSemiringOps_416_; lean_object* v_zPow_417_; lean_object* v___x_418_; 
v_toRingOps_415_ = lean_ctor_get(v_fo_408_, 0);
lean_inc_ref(v_toRingOps_415_);
lean_dec_ref(v_fo_408_);
v_toSemiringOps_416_ = lean_ctor_get(v_toRingOps_415_, 0);
lean_inc_ref_n(v_toSemiringOps_416_, 2);
v_zPow_417_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_416_, v_z_411_, v_slcLen_412_);
v___x_418_ = l_List_reverse___redArg(v_zPow_417_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_dec_ref(v_toSemiringOps_416_);
lean_dec_ref(v_toRingOps_415_);
lean_dec(v_finalPoly_414_);
lean_dec(v_alphaSlc_413_);
lean_dec(v_gammaPow_410_);
return v_acc_409_;
}
else
{
lean_object* v_head_419_; lean_object* v_tail_420_; lean_object* v_add_421_; lean_object* v_mul_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v_head_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc(v_head_419_);
v_tail_420_ = lean_ctor_get(v___x_418_, 1);
lean_inc(v_tail_420_);
lean_dec_ref_known(v___x_418_, 2);
v_add_421_ = lean_ctor_get(v_toSemiringOps_416_, 3);
lean_inc(v_add_421_);
v_mul_422_ = lean_ctor_get(v_toSemiringOps_416_, 4);
lean_inc_n(v_mul_422_, 2);
v___x_423_ = l_List_reverse___redArg(v_tail_420_);
v___x_424_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_415_, v_alphaSlc_413_, v___x_423_);
v___x_425_ = lean_apply_2(v_mul_422_, v_gammaPow_410_, v___x_424_);
v___x_426_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_toSemiringOps_416_, v_finalPoly_414_, v_head_419_);
v___x_427_ = lean_apply_2(v_mul_422_, v___x_425_, v___x_426_);
v___x_428_ = lean_apply_2(v_add_421_, v_acc_409_, v___x_427_);
return v___x_428_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg___boxed(lean_object* v_fo_429_, lean_object* v_acc_430_, lean_object* v_gammaPow_431_, lean_object* v_z_432_, lean_object* v_slcLen_433_, lean_object* v_alphaSlc_434_, lean_object* v_finalPoly_435_){
_start:
{
lean_object* v_res_436_; 
v_res_436_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg(v_fo_429_, v_acc_430_, v_gammaPow_431_, v_z_432_, v_slcLen_433_, v_alphaSlc_434_, v_finalPoly_435_);
lean_dec(v_z_432_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution(lean_object* v_EF_437_, lean_object* v_fo_438_, lean_object* v_acc_439_, lean_object* v_gammaPow_440_, lean_object* v_z_441_, lean_object* v_slcLen_442_, lean_object* v_alphaSlc_443_, lean_object* v_finalPoly_444_){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg(v_fo_438_, v_acc_439_, v_gammaPow_440_, v_z_441_, v_slcLen_442_, v_alphaSlc_443_, v_finalPoly_444_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___boxed(lean_object* v_EF_446_, lean_object* v_fo_447_, lean_object* v_acc_448_, lean_object* v_gammaPow_449_, lean_object* v_z_450_, lean_object* v_slcLen_451_, lean_object* v_alphaSlc_452_, lean_object* v_finalPoly_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution(v_EF_446_, v_fo_447_, v_acc_448_, v_gammaPow_449_, v_z_450_, v_slcLen_451_, v_alphaSlc_452_, v_finalPoly_453_);
lean_dec(v_z_450_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg(lean_object* v_fo_455_, lean_object* v_acc_456_, lean_object* v_gamma_457_, lean_object* v_slcLen_458_, lean_object* v_alphaSlc_459_, lean_object* v_finalPoly_460_, lean_object* v_gammaPows_461_, lean_object* v_zs_462_){
_start:
{
if (lean_obj_tag(v_gammaPows_461_) == 0)
{
lean_dec(v_finalPoly_460_);
lean_dec(v_alphaSlc_459_);
lean_dec(v_slcLen_458_);
lean_dec(v_gamma_457_);
lean_dec_ref(v_fo_455_);
return v_acc_456_;
}
else
{
if (lean_obj_tag(v_zs_462_) == 0)
{
lean_dec_ref_known(v_gammaPows_461_, 2);
lean_dec(v_finalPoly_460_);
lean_dec(v_alphaSlc_459_);
lean_dec(v_slcLen_458_);
lean_dec(v_gamma_457_);
lean_dec_ref(v_fo_455_);
return v_acc_456_;
}
else
{
lean_object* v_toRingOps_463_; lean_object* v_toSemiringOps_464_; lean_object* v_head_465_; lean_object* v_tail_466_; lean_object* v_head_467_; lean_object* v_tail_468_; lean_object* v_mul_469_; lean_object* v___x_470_; lean_object* v_acc_x27_471_; 
v_toRingOps_463_ = lean_ctor_get(v_fo_455_, 0);
v_toSemiringOps_464_ = lean_ctor_get(v_toRingOps_463_, 0);
v_head_465_ = lean_ctor_get(v_gammaPows_461_, 0);
lean_inc(v_head_465_);
v_tail_466_ = lean_ctor_get(v_gammaPows_461_, 1);
lean_inc(v_tail_466_);
lean_dec_ref_known(v_gammaPows_461_, 2);
v_head_467_ = lean_ctor_get(v_zs_462_, 0);
v_tail_468_ = lean_ctor_get(v_zs_462_, 1);
v_mul_469_ = lean_ctor_get(v_toSemiringOps_464_, 4);
lean_inc(v_mul_469_);
lean_inc(v_gamma_457_);
v___x_470_ = lean_apply_2(v_mul_469_, v_gamma_457_, v_head_465_);
lean_inc(v_finalPoly_460_);
lean_inc(v_alphaSlc_459_);
lean_inc(v_slcLen_458_);
lean_inc_ref(v_fo_455_);
v_acc_x27_471_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg(v_fo_455_, v_acc_456_, v___x_470_, v_head_467_, v_slcLen_458_, v_alphaSlc_459_, v_finalPoly_460_);
v_acc_456_ = v_acc_x27_471_;
v_gammaPows_461_ = v_tail_466_;
v_zs_462_ = v_tail_468_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg___boxed(lean_object* v_fo_473_, lean_object* v_acc_474_, lean_object* v_gamma_475_, lean_object* v_slcLen_476_, lean_object* v_alphaSlc_477_, lean_object* v_finalPoly_478_, lean_object* v_gammaPows_479_, lean_object* v_zs_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg(v_fo_473_, v_acc_474_, v_gamma_475_, v_slcLen_476_, v_alphaSlc_477_, v_finalPoly_478_, v_gammaPows_479_, v_zs_480_);
lean_dec(v_zs_480_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound(lean_object* v_EF_482_, lean_object* v_fo_483_, lean_object* v_acc_484_, lean_object* v_gamma_485_, lean_object* v_slcLen_486_, lean_object* v_alphaSlc_487_, lean_object* v_finalPoly_488_, lean_object* v_gammaPows_489_, lean_object* v_zs_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg(v_fo_483_, v_acc_484_, v_gamma_485_, v_slcLen_486_, v_alphaSlc_487_, v_finalPoly_488_, v_gammaPows_489_, v_zs_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___boxed(lean_object* v_EF_492_, lean_object* v_fo_493_, lean_object* v_acc_494_, lean_object* v_gamma_495_, lean_object* v_slcLen_496_, lean_object* v_alphaSlc_497_, lean_object* v_finalPoly_498_, lean_object* v_gammaPows_499_, lean_object* v_zs_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound(v_EF_492_, v_fo_493_, v_acc_494_, v_gamma_495_, v_slcLen_496_, v_alphaSlc_497_, v_finalPoly_498_, v_gammaPows_499_, v_zs_500_);
lean_dec(v_zs_500_);
return v_res_501_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg(lean_object* v_fo_502_, lean_object* v_acc_503_, lean_object* v_roundIdx_504_, lean_object* v_numWhirRounds_505_, lean_object* v_kWhir_506_, lean_object* v_alphas_507_, lean_object* v_gamma_508_, lean_object* v_zsRound_509_, lean_object* v_z0_510_, lean_object* v_finalPoly_511_){
_start:
{
lean_object* v_t_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v_j_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v_alphaSlc_519_; lean_object* v_slcLen_520_; lean_object* v___y_522_; uint8_t v___x_528_; 
v_t_512_ = lean_nat_mul(v_kWhir_506_, v_numWhirRounds_505_);
v___x_513_ = lean_unsigned_to_nat(1u);
v___x_514_ = lean_nat_add(v_roundIdx_504_, v___x_513_);
v_j_515_ = lean_nat_mul(v___x_514_, v_kWhir_506_);
v___x_516_ = lean_nat_sub(v_t_512_, v_j_515_);
lean_dec(v_t_512_);
v___x_517_ = l_List_drop___redArg(v_j_515_, v_alphas_507_);
v___x_518_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0));
lean_inc(v___x_516_);
lean_inc(v___x_517_);
v_alphaSlc_519_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v___x_517_, v___x_517_, v___x_516_, v___x_518_);
lean_dec(v___x_517_);
v_slcLen_520_ = lean_nat_add(v___x_516_, v___x_513_);
lean_dec(v___x_516_);
v___x_528_ = lean_nat_dec_eq(v___x_514_, v_numWhirRounds_505_);
lean_dec(v___x_514_);
if (v___x_528_ == 0)
{
if (lean_obj_tag(v_z0_510_) == 0)
{
v___y_522_ = v_acc_503_;
goto v___jp_521_;
}
else
{
lean_object* v_val_529_; lean_object* v___x_530_; 
v_val_529_ = lean_ctor_get(v_z0_510_, 0);
lean_inc(v_finalPoly_511_);
lean_inc(v_alphaSlc_519_);
lean_inc(v_slcLen_520_);
lean_inc(v_gamma_508_);
lean_inc_ref(v_fo_502_);
v___x_530_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZContribution___redArg(v_fo_502_, v_acc_503_, v_gamma_508_, v_val_529_, v_slcLen_520_, v_alphaSlc_519_, v_finalPoly_511_);
v___y_522_ = v___x_530_;
goto v___jp_521_;
}
}
else
{
v___y_522_ = v_acc_503_;
goto v___jp_521_;
}
v___jp_521_:
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v_gammaPows_526_; lean_object* v___x_527_; 
v___x_523_ = l_List_lengthTR___redArg(v_zsRound_509_);
v___x_524_ = lean_nat_add(v___x_523_, v___x_513_);
lean_dec(v___x_523_);
lean_inc(v_gamma_508_);
lean_inc_ref(v_fo_502_);
v___x_525_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_502_, v_gamma_508_, v___x_524_);
v_gammaPows_526_ = l_List_drop___redArg(v___x_513_, v___x_525_);
lean_dec(v___x_525_);
v___x_527_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckZsRound___redArg(v_fo_502_, v___y_522_, v_gamma_508_, v_slcLen_520_, v_alphaSlc_519_, v_finalPoly_511_, v_gammaPows_526_, v_zsRound_509_);
return v___x_527_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg___boxed(lean_object* v_fo_531_, lean_object* v_acc_532_, lean_object* v_roundIdx_533_, lean_object* v_numWhirRounds_534_, lean_object* v_kWhir_535_, lean_object* v_alphas_536_, lean_object* v_gamma_537_, lean_object* v_zsRound_538_, lean_object* v_z0_539_, lean_object* v_finalPoly_540_){
_start:
{
lean_object* v_res_541_; 
v_res_541_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg(v_fo_531_, v_acc_532_, v_roundIdx_533_, v_numWhirRounds_534_, v_kWhir_535_, v_alphas_536_, v_gamma_537_, v_zsRound_538_, v_z0_539_, v_finalPoly_540_);
lean_dec(v_z0_539_);
lean_dec(v_zsRound_538_);
lean_dec(v_alphas_536_);
lean_dec(v_kWhir_535_);
lean_dec(v_numWhirRounds_534_);
lean_dec(v_roundIdx_533_);
return v_res_541_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound(lean_object* v_EF_542_, lean_object* v_fo_543_, lean_object* v_acc_544_, lean_object* v_roundIdx_545_, lean_object* v_numWhirRounds_546_, lean_object* v_kWhir_547_, lean_object* v_alphas_548_, lean_object* v_gamma_549_, lean_object* v_zsRound_550_, lean_object* v_z0_551_, lean_object* v_finalPoly_552_){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg(v_fo_543_, v_acc_544_, v_roundIdx_545_, v_numWhirRounds_546_, v_kWhir_547_, v_alphas_548_, v_gamma_549_, v_zsRound_550_, v_z0_551_, v_finalPoly_552_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___boxed(lean_object* v_EF_554_, lean_object* v_fo_555_, lean_object* v_acc_556_, lean_object* v_roundIdx_557_, lean_object* v_numWhirRounds_558_, lean_object* v_kWhir_559_, lean_object* v_alphas_560_, lean_object* v_gamma_561_, lean_object* v_zsRound_562_, lean_object* v_z0_563_, lean_object* v_finalPoly_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound(v_EF_554_, v_fo_555_, v_acc_556_, v_roundIdx_557_, v_numWhirRounds_558_, v_kWhir_559_, v_alphas_560_, v_gamma_561_, v_zsRound_562_, v_z0_563_, v_finalPoly_564_);
lean_dec(v_z0_563_);
lean_dec(v_zsRound_562_);
lean_dec(v_alphas_560_);
lean_dec(v_kWhir_559_);
lean_dec(v_numWhirRounds_558_);
lean_dec(v_roundIdx_557_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg(lean_object* v_inst_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_fo_572_, lean_object* v_params_573_, lean_object* v_claim_574_, lean_object* v_polyEvs_575_, lean_object* v_powWitnesses_576_, lean_object* v_a_577_){
_start:
{
if (lean_obj_tag(v_polyEvs_575_) == 0)
{
lean_dec_ref(v_fo_572_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
lean_dec_ref(v_inst_569_);
if (lean_obj_tag(v_powWitnesses_576_) == 0)
{
lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_580_ = lean_box(0);
v___x_581_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_581_, 0, v_claim_574_);
lean_ctor_set(v___x_581_, 1, v___x_580_);
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set(v___x_582_, 1, v_a_577_);
v___x_583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_583_, 0, v___x_582_);
return v___x_583_;
}
else
{
lean_dec_ref(v_a_577_);
lean_dec(v_powWitnesses_576_);
lean_dec(v_claim_574_);
goto v___jp_578_;
}
}
else
{
if (lean_obj_tag(v_powWitnesses_576_) == 1)
{
lean_object* v_head_584_; lean_object* v_tail_585_; lean_object* v_head_586_; lean_object* v_tail_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_652_; 
v_head_584_ = lean_ctor_get(v_polyEvs_575_, 0);
v_tail_585_ = lean_ctor_get(v_polyEvs_575_, 1);
v_head_586_ = lean_ctor_get(v_powWitnesses_576_, 0);
v_tail_587_ = lean_ctor_get(v_powWitnesses_576_, 1);
v_isSharedCheck_652_ = !lean_is_exclusive(v_powWitnesses_576_);
if (v_isSharedCheck_652_ == 0)
{
v___x_589_ = v_powWitnesses_576_;
v_isShared_590_ = v_isSharedCheck_652_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_tail_587_);
lean_inc(v_head_586_);
lean_dec(v_powWitnesses_576_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_652_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___x_591_; lean_object* v_ev1_592_; lean_object* v___x_593_; lean_object* v_a_594_; lean_object* v_snd_595_; lean_object* v___x_596_; lean_object* v_ev2_597_; lean_object* v___x_598_; lean_object* v_a_599_; lean_object* v_whir_600_; lean_object* v_snd_601_; lean_object* v_foldingPowBits_602_; lean_object* v___x_603_; 
v___x_591_ = lean_unsigned_to_nat(0u);
v_ev1_592_ = lean_array_fget_borrowed(v_head_584_, v___x_591_);
lean_inc(v_ev1_592_);
lean_inc_ref_n(v_inst_569_, 2);
lean_inc_ref_n(v_inst_570_, 3);
v___x_593_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_570_, v_inst_569_, v_ev1_592_, v_a_577_);
v_a_594_ = lean_ctor_get(v___x_593_, 0);
lean_inc(v_a_594_);
lean_dec_ref(v___x_593_);
v_snd_595_ = lean_ctor_get(v_a_594_, 1);
lean_inc(v_snd_595_);
lean_dec(v_a_594_);
v___x_596_ = lean_unsigned_to_nat(1u);
v_ev2_597_ = lean_array_fget_borrowed(v_head_584_, v___x_596_);
lean_inc(v_ev2_597_);
v___x_598_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_570_, v_inst_569_, v_ev2_597_, v_snd_595_);
v_a_599_ = lean_ctor_get(v___x_598_, 0);
lean_inc(v_a_599_);
lean_dec_ref(v___x_598_);
v_whir_600_ = lean_ctor_get(v_params_573_, 4);
v_snd_601_ = lean_ctor_get(v_a_599_, 1);
lean_inc(v_snd_601_);
lean_dec(v_a_599_);
v_foldingPowBits_602_ = lean_ctor_get(v_whir_600_, 4);
lean_inc_ref(v_inst_571_);
v___x_603_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observePowWitness___redArg(v_inst_570_, v_inst_571_, v_foldingPowBits_602_, v_head_586_, v_snd_601_);
if (lean_obj_tag(v___x_603_) == 0)
{
lean_object* v_a_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_611_; 
lean_del_object(v___x_589_);
lean_dec(v_tail_587_);
lean_dec(v_claim_574_);
lean_dec_ref(v_fo_572_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
lean_dec_ref(v_inst_569_);
v_a_604_ = lean_ctor_get(v___x_603_, 0);
v_isSharedCheck_611_ = !lean_is_exclusive(v___x_603_);
if (v_isSharedCheck_611_ == 0)
{
v___x_606_ = v___x_603_;
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_a_604_);
lean_dec(v___x_603_);
v___x_606_ = lean_box(0);
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
v_resetjp_605_:
{
lean_object* v___x_609_; 
if (v_isShared_607_ == 0)
{
v___x_609_ = v___x_606_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v_a_604_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
return v___x_609_;
}
}
}
else
{
lean_object* v_a_612_; lean_object* v_snd_613_; lean_object* v___x_614_; lean_object* v_a_615_; lean_object* v_toRingOps_616_; lean_object* v_fst_617_; lean_object* v_snd_618_; lean_object* v_sub_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v_a_612_ = lean_ctor_get(v___x_603_, 0);
lean_inc(v_a_612_);
lean_dec_ref_known(v___x_603_, 1);
v_snd_613_ = lean_ctor_get(v_a_612_, 1);
lean_inc(v_snd_613_);
lean_dec(v_a_612_);
lean_inc_ref(v_inst_569_);
lean_inc_ref(v_inst_570_);
v___x_614_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_570_, v_inst_569_, v_snd_613_);
v_a_615_ = lean_ctor_get(v___x_614_, 0);
lean_inc(v_a_615_);
lean_dec_ref(v___x_614_);
v_toRingOps_616_ = lean_ctor_get(v_fo_572_, 0);
v_fst_617_ = lean_ctor_get(v_a_615_, 0);
lean_inc_n(v_fst_617_, 2);
v_snd_618_ = lean_ctor_get(v_a_615_, 1);
lean_inc(v_snd_618_);
lean_dec(v_a_615_);
v_sub_619_ = lean_ctor_get(v_toRingOps_616_, 1);
lean_inc(v_sub_619_);
lean_inc_n(v_ev1_592_, 2);
v___x_620_ = lean_apply_2(v_sub_619_, v_claim_574_, v_ev1_592_);
lean_inc(v_ev2_597_);
lean_inc_ref(v_fo_572_);
v___x_621_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(v_fo_572_, v___x_620_, v_ev1_592_, v_ev2_597_, v_fst_617_);
v___x_622_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg(v_inst_569_, v_inst_570_, v_inst_571_, v_fo_572_, v_params_573_, v___x_621_, v_tail_585_, v_tail_587_, v_snd_618_);
if (lean_obj_tag(v___x_622_) == 0)
{
lean_dec(v_fst_617_);
lean_del_object(v___x_589_);
return v___x_622_;
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_651_; 
v_a_623_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_651_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_651_ == 0)
{
v___x_625_ = v___x_622_;
v_isShared_626_ = v_isSharedCheck_651_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_622_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_651_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v_fst_627_; lean_object* v_snd_628_; lean_object* v___x_630_; uint8_t v_isShared_631_; uint8_t v_isSharedCheck_650_; 
v_fst_627_ = lean_ctor_get(v_a_623_, 0);
v_snd_628_ = lean_ctor_get(v_a_623_, 1);
v_isSharedCheck_650_ = !lean_is_exclusive(v_a_623_);
if (v_isSharedCheck_650_ == 0)
{
v___x_630_ = v_a_623_;
v_isShared_631_ = v_isSharedCheck_650_;
goto v_resetjp_629_;
}
else
{
lean_inc(v_snd_628_);
lean_inc(v_fst_627_);
lean_dec(v_a_623_);
v___x_630_ = lean_box(0);
v_isShared_631_ = v_isSharedCheck_650_;
goto v_resetjp_629_;
}
v_resetjp_629_:
{
lean_object* v_fst_632_; lean_object* v_snd_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_649_; 
v_fst_632_ = lean_ctor_get(v_fst_627_, 0);
v_snd_633_ = lean_ctor_get(v_fst_627_, 1);
v_isSharedCheck_649_ = !lean_is_exclusive(v_fst_627_);
if (v_isSharedCheck_649_ == 0)
{
v___x_635_ = v_fst_627_;
v_isShared_636_ = v_isSharedCheck_649_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_snd_633_);
lean_inc(v_fst_632_);
lean_dec(v_fst_627_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_649_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_590_ == 0)
{
lean_ctor_set(v___x_589_, 1, v_snd_633_);
lean_ctor_set(v___x_589_, 0, v_fst_617_);
v___x_638_ = v___x_589_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_648_; 
v_reuseFailAlloc_648_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_648_, 0, v_fst_617_);
lean_ctor_set(v_reuseFailAlloc_648_, 1, v_snd_633_);
v___x_638_ = v_reuseFailAlloc_648_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
lean_object* v___x_640_; 
if (v_isShared_636_ == 0)
{
lean_ctor_set(v___x_635_, 1, v___x_638_);
v___x_640_ = v___x_635_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v_fst_632_);
lean_ctor_set(v_reuseFailAlloc_647_, 1, v___x_638_);
v___x_640_ = v_reuseFailAlloc_647_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
lean_object* v___x_642_; 
if (v_isShared_631_ == 0)
{
lean_ctor_set(v___x_630_, 0, v___x_640_);
v___x_642_ = v___x_630_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v___x_640_);
lean_ctor_set(v_reuseFailAlloc_646_, 1, v_snd_628_);
v___x_642_ = v_reuseFailAlloc_646_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
lean_object* v___x_644_; 
if (v_isShared_626_ == 0)
{
lean_ctor_set(v___x_625_, 0, v___x_642_);
v___x_644_ = v___x_625_;
goto v_reusejp_643_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v___x_642_);
v___x_644_ = v_reuseFailAlloc_645_;
goto v_reusejp_643_;
}
v_reusejp_643_:
{
return v___x_644_;
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
else
{
lean_dec_ref(v_a_577_);
lean_dec(v_powWitnesses_576_);
lean_dec(v_claim_574_);
lean_dec_ref(v_fo_572_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
lean_dec_ref(v_inst_569_);
goto v___jp_578_;
}
}
v___jp_578_:
{
lean_object* v___x_579_; 
v___x_579_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg___closed__0));
return v___x_579_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg___boxed(lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_fo_656_, lean_object* v_params_657_, lean_object* v_claim_658_, lean_object* v_polyEvs_659_, lean_object* v_powWitnesses_660_, lean_object* v_a_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg(v_inst_653_, v_inst_654_, v_inst_655_, v_fo_656_, v_params_657_, v_claim_658_, v_polyEvs_659_, v_powWitnesses_660_, v_a_661_);
lean_dec(v_polyEvs_659_);
lean_dec_ref(v_params_657_);
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM(lean_object* v_F_663_, lean_object* v_EF_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_fo_668_, lean_object* v_params_669_, lean_object* v_claim_670_, lean_object* v_polyEvs_671_, lean_object* v_powWitnesses_672_, lean_object* v_a_673_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg(v_inst_665_, v_inst_666_, v_inst_667_, v_fo_668_, v_params_669_, v_claim_670_, v_polyEvs_671_, v_powWitnesses_672_, v_a_673_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___boxed(lean_object* v_F_675_, lean_object* v_EF_676_, lean_object* v_inst_677_, lean_object* v_inst_678_, lean_object* v_inst_679_, lean_object* v_fo_680_, lean_object* v_params_681_, lean_object* v_claim_682_, lean_object* v_polyEvs_683_, lean_object* v_powWitnesses_684_, lean_object* v_a_685_){
_start:
{
lean_object* v_res_686_; 
v_res_686_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM(v_F_675_, v_EF_676_, v_inst_677_, v_inst_678_, v_inst_679_, v_fo_680_, v_params_681_, v_claim_682_, v_polyEvs_683_, v_powWitnesses_684_, v_a_685_);
lean_dec(v_polyEvs_683_);
lean_dec_ref(v_params_681_);
return v_res_686_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__0(lean_object* v_fo_687_, lean_object* v_acc_688_, lean_object* v_entry_689_){
_start:
{
lean_object* v_toRingOps_690_; lean_object* v_toSemiringOps_691_; lean_object* v_add_692_; lean_object* v_mul_693_; lean_object* v_fst_694_; lean_object* v_snd_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
v_toRingOps_690_ = lean_ctor_get(v_fo_687_, 0);
lean_inc_ref(v_toRingOps_690_);
lean_dec_ref(v_fo_687_);
v_toSemiringOps_691_ = lean_ctor_get(v_toRingOps_690_, 0);
lean_inc_ref(v_toSemiringOps_691_);
lean_dec_ref(v_toRingOps_690_);
v_add_692_ = lean_ctor_get(v_toSemiringOps_691_, 3);
lean_inc(v_add_692_);
v_mul_693_ = lean_ctor_get(v_toSemiringOps_691_, 4);
lean_inc(v_mul_693_);
lean_dec_ref(v_toSemiringOps_691_);
v_fst_694_ = lean_ctor_get(v_entry_689_, 0);
lean_inc(v_fst_694_);
v_snd_695_ = lean_ctor_get(v_entry_689_, 1);
lean_inc(v_snd_695_);
lean_dec_ref(v_entry_689_);
v___x_696_ = lean_apply_2(v_mul_693_, v_fst_694_, v_snd_695_);
v___x_697_ = lean_apply_2(v_add_692_, v_acc_688_, v___x_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__1(lean_object* v_inst_698_, lean_object* v_inst_699_, lean_object* v___x_700_, lean_object* v_acc_701_, lean_object* v_x_702_, lean_object* v___y_703_){
_start:
{
lean_object* v_fst_704_; lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_739_; 
v_fst_704_ = lean_ctor_get(v_acc_701_, 0);
v_isSharedCheck_739_ = !lean_is_exclusive(v_acc_701_);
if (v_isSharedCheck_739_ == 0)
{
lean_object* v_unused_740_; 
v_unused_740_ = lean_ctor_get(v_acc_701_, 1);
lean_dec(v_unused_740_);
v___x_706_ = v_acc_701_;
v_isShared_707_ = v_isSharedCheck_739_;
goto v_resetjp_705_;
}
else
{
lean_inc(v_fst_704_);
lean_dec(v_acc_701_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_739_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_708_; 
v___x_708_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleBits___redArg(v_inst_698_, v_inst_699_, v___x_700_, v___y_703_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_object* v_a_709_; lean_object* v___x_711_; uint8_t v_isShared_712_; uint8_t v_isSharedCheck_716_; 
lean_del_object(v___x_706_);
lean_dec(v_fst_704_);
v_a_709_ = lean_ctor_get(v___x_708_, 0);
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_716_ == 0)
{
v___x_711_ = v___x_708_;
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
else
{
lean_inc(v_a_709_);
lean_dec(v___x_708_);
v___x_711_ = lean_box(0);
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
v_resetjp_710_:
{
lean_object* v___x_714_; 
if (v_isShared_712_ == 0)
{
v___x_714_ = v___x_711_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_a_709_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
else
{
lean_object* v_a_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_738_; 
v_a_717_ = lean_ctor_get(v___x_708_, 0);
v_isSharedCheck_738_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_738_ == 0)
{
v___x_719_ = v___x_708_;
v_isShared_720_ = v_isSharedCheck_738_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_a_717_);
lean_dec(v___x_708_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_738_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v_fst_721_; lean_object* v_snd_722_; lean_object* v___x_724_; uint8_t v_isShared_725_; uint8_t v_isSharedCheck_737_; 
v_fst_721_ = lean_ctor_get(v_a_717_, 0);
v_snd_722_ = lean_ctor_get(v_a_717_, 1);
v_isSharedCheck_737_ = !lean_is_exclusive(v_a_717_);
if (v_isSharedCheck_737_ == 0)
{
v___x_724_ = v_a_717_;
v_isShared_725_ = v_isSharedCheck_737_;
goto v_resetjp_723_;
}
else
{
lean_inc(v_snd_722_);
lean_inc(v_fst_721_);
lean_dec(v_a_717_);
v___x_724_ = lean_box(0);
v_isShared_725_ = v_isSharedCheck_737_;
goto v_resetjp_723_;
}
v_resetjp_723_:
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_729_; 
v___x_726_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_726_, 0, v_fst_721_);
lean_ctor_set(v___x_726_, 1, v_fst_704_);
v___x_727_ = lean_box(0);
if (v_isShared_725_ == 0)
{
lean_ctor_set(v___x_724_, 1, v___x_727_);
lean_ctor_set(v___x_724_, 0, v___x_726_);
v___x_729_ = v___x_724_;
goto v_reusejp_728_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v___x_726_);
lean_ctor_set(v_reuseFailAlloc_736_, 1, v___x_727_);
v___x_729_ = v_reuseFailAlloc_736_;
goto v_reusejp_728_;
}
v_reusejp_728_:
{
lean_object* v___x_731_; 
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 1, v_snd_722_);
lean_ctor_set(v___x_706_, 0, v___x_729_);
v___x_731_ = v___x_706_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v___x_729_);
lean_ctor_set(v_reuseFailAlloc_735_, 1, v_snd_722_);
v___x_731_ = v_reuseFailAlloc_735_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
lean_object* v___x_733_; 
if (v_isShared_720_ == 0)
{
lean_ctor_set(v___x_719_, 0, v___x_731_);
v___x_733_ = v___x_719_;
goto v_reusejp_732_;
}
else
{
lean_object* v_reuseFailAlloc_734_; 
v_reuseFailAlloc_734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_734_, 0, v___x_731_);
v___x_733_ = v_reuseFailAlloc_734_;
goto v_reusejp_732_;
}
v_reusejp_732_:
{
return v___x_733_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__1___boxed(lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v___x_743_, lean_object* v_acc_744_, lean_object* v_x_745_, lean_object* v___y_746_){
_start:
{
lean_object* v_res_747_; 
v_res_747_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__1(v_inst_741_, v_inst_742_, v___x_743_, v_acc_744_, v_x_745_, v___y_746_);
lean_dec(v_x_745_);
lean_dec(v___x_743_);
return v_res_747_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2(lean_object* v_fo_751_, lean_object* v___x_752_, lean_object* v_kWhir_753_, lean_object* v_roundIdx_754_, lean_object* v___x_755_, lean_object* v_codewordOpenedValues_756_, lean_object* v_codewordMerkleProofs_757_, lean_object* v_codewordCommits_758_, lean_object* v_inst_759_, lean_object* v_inst_760_, lean_object* v_inst_761_, lean_object* v_inst_762_, lean_object* v_snd_763_, lean_object* v_algMap_764_, lean_object* v_muPows_765_, lean_object* v_commitments_766_, lean_object* v_widths_767_, lean_object* v_initialRoundOpenedRows_768_, lean_object* v_initialRoundMerkleProofs_769_, lean_object* v_acc_770_, lean_object* v_pair_771_, lean_object* v___y_772_){
_start:
{
lean_object* v_toRingOps_773_; lean_object* v_toSemiringOps_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_831_; 
v_toRingOps_773_ = lean_ctor_get(v_fo_751_, 0);
lean_inc_ref(v_toRingOps_773_);
v_toSemiringOps_774_ = lean_ctor_get(v_toRingOps_773_, 0);
v_isSharedCheck_831_ = !lean_is_exclusive(v_toRingOps_773_);
if (v_isSharedCheck_831_ == 0)
{
lean_object* v_unused_832_; 
v_unused_832_ = lean_ctor_get(v_toRingOps_773_, 1);
lean_dec(v_unused_832_);
v___x_776_ = v_toRingOps_773_;
v_isShared_777_ = v_isSharedCheck_831_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_toSemiringOps_774_);
lean_dec(v_toRingOps_773_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_831_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v_fst_778_; lean_object* v_snd_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_830_; 
v_fst_778_ = lean_ctor_get(v_acc_770_, 0);
v_snd_779_ = lean_ctor_get(v_acc_770_, 1);
v_isSharedCheck_830_ = !lean_is_exclusive(v_acc_770_);
if (v_isSharedCheck_830_ == 0)
{
v___x_781_ = v_acc_770_;
v_isShared_782_ = v_isSharedCheck_830_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_snd_779_);
lean_inc(v_fst_778_);
lean_dec(v_acc_770_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_830_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v_fst_783_; lean_object* v_snd_784_; lean_object* v___x_786_; uint8_t v_isShared_787_; uint8_t v_isSharedCheck_829_; 
v_fst_783_ = lean_ctor_get(v_pair_771_, 0);
v_snd_784_ = lean_ctor_get(v_pair_771_, 1);
v_isSharedCheck_829_ = !lean_is_exclusive(v_pair_771_);
if (v_isSharedCheck_829_ == 0)
{
v___x_786_ = v_pair_771_;
v_isShared_787_ = v_isSharedCheck_829_;
goto v_resetjp_785_;
}
else
{
lean_inc(v_snd_784_);
lean_inc(v_fst_783_);
lean_dec(v_pair_771_);
v___x_786_ = lean_box(0);
v_isShared_787_ = v_isSharedCheck_829_;
goto v_resetjp_785_;
}
v_resetjp_785_:
{
lean_object* v_pow_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v_yi_792_; lean_object* v___y_793_; uint8_t v___x_805_; 
v_pow_788_ = lean_ctor_get(v_toSemiringOps_774_, 5);
lean_inc(v_pow_788_);
lean_inc(v_snd_784_);
v___x_789_ = lean_apply_2(v_pow_788_, v___x_752_, v_snd_784_);
v___x_790_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_toSemiringOps_774_, v___x_789_, v_kWhir_753_);
v___x_805_ = lean_nat_dec_eq(v_roundIdx_754_, v___x_755_);
if (v___x_805_ == 0)
{
lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; 
lean_dec(v_widths_767_);
lean_dec(v_commitments_766_);
lean_dec(v_muPows_765_);
lean_dec(v_algMap_764_);
v___x_806_ = lean_unsigned_to_nat(1u);
v___x_807_ = lean_nat_sub(v_roundIdx_754_, v___x_806_);
lean_inc(v___x_807_);
v___x_808_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_codewordOpenedValues_756_, v___x_807_);
if (lean_obj_tag(v___x_808_) == 0)
{
lean_object* v___x_809_; 
lean_dec_ref_known(v___x_808_, 1);
lean_dec(v___x_807_);
lean_dec(v___x_790_);
lean_dec(v___x_789_);
lean_del_object(v___x_786_);
lean_dec(v_snd_784_);
lean_dec(v_fst_783_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
lean_dec(v_snd_763_);
lean_dec(v_inst_762_);
lean_dec_ref(v_inst_761_);
lean_dec_ref(v_inst_760_);
lean_dec_ref(v_inst_759_);
lean_dec_ref(v_fo_751_);
v___x_809_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_809_;
}
else
{
lean_object* v_a_810_; lean_object* v___x_811_; 
v_a_810_ = lean_ctor_get(v___x_808_, 0);
lean_inc(v_a_810_);
lean_dec_ref_known(v___x_808_, 1);
lean_inc(v___x_807_);
v___x_811_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_codewordMerkleProofs_757_, v___x_807_);
if (lean_obj_tag(v___x_811_) == 0)
{
lean_object* v___x_812_; 
lean_dec_ref_known(v___x_811_, 1);
lean_dec(v_a_810_);
lean_dec(v___x_807_);
lean_dec(v___x_790_);
lean_dec(v___x_789_);
lean_del_object(v___x_786_);
lean_dec(v_snd_784_);
lean_dec(v_fst_783_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
lean_dec(v_snd_763_);
lean_dec(v_inst_762_);
lean_dec_ref(v_inst_761_);
lean_dec_ref(v_inst_760_);
lean_dec_ref(v_inst_759_);
lean_dec_ref(v_fo_751_);
v___x_812_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_812_;
}
else
{
lean_object* v_a_813_; lean_object* v___x_814_; 
v_a_813_ = lean_ctor_get(v___x_811_, 0);
lean_inc(v_a_813_);
lean_dec_ref_known(v___x_811_, 1);
lean_inc(v_fst_783_);
v___x_814_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_a_810_, v_fst_783_);
lean_dec(v_a_810_);
if (lean_obj_tag(v___x_814_) == 0)
{
lean_object* v___x_815_; 
lean_dec_ref_known(v___x_814_, 1);
lean_dec(v_a_813_);
lean_dec(v___x_807_);
lean_dec(v___x_790_);
lean_dec(v___x_789_);
lean_del_object(v___x_786_);
lean_dec(v_snd_784_);
lean_dec(v_fst_783_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
lean_dec(v_snd_763_);
lean_dec(v_inst_762_);
lean_dec_ref(v_inst_761_);
lean_dec_ref(v_inst_760_);
lean_dec_ref(v_inst_759_);
lean_dec_ref(v_fo_751_);
v___x_815_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_815_;
}
else
{
lean_object* v_a_816_; lean_object* v___x_817_; 
v_a_816_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_a_816_);
lean_dec_ref_known(v___x_814_, 1);
v___x_817_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_a_813_, v_fst_783_);
lean_dec(v_a_813_);
if (lean_obj_tag(v___x_817_) == 0)
{
lean_object* v___x_818_; 
lean_dec_ref_known(v___x_817_, 1);
lean_dec(v_a_816_);
lean_dec(v___x_807_);
lean_dec(v___x_790_);
lean_dec(v___x_789_);
lean_del_object(v___x_786_);
lean_dec(v_snd_784_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
lean_dec(v_snd_763_);
lean_dec(v_inst_762_);
lean_dec_ref(v_inst_761_);
lean_dec_ref(v_inst_760_);
lean_dec_ref(v_inst_759_);
lean_dec_ref(v_fo_751_);
v___x_818_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_818_;
}
else
{
lean_object* v_a_819_; lean_object* v___x_820_; 
v_a_819_ = lean_ctor_get(v___x_817_, 0);
lean_inc(v_a_819_);
lean_dec_ref_known(v___x_817_, 1);
v___x_820_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_codewordCommits_758_, v___x_807_);
if (lean_obj_tag(v___x_820_) == 0)
{
lean_object* v___x_821_; 
lean_dec_ref_known(v___x_820_, 1);
lean_dec(v_a_819_);
lean_dec(v_a_816_);
lean_dec(v___x_790_);
lean_dec(v___x_789_);
lean_del_object(v___x_786_);
lean_dec(v_snd_784_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
lean_dec(v_snd_763_);
lean_dec(v_inst_762_);
lean_dec_ref(v_inst_761_);
lean_dec_ref(v_inst_760_);
lean_dec_ref(v_inst_759_);
lean_dec_ref(v_fo_751_);
v___x_821_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_821_;
}
else
{
lean_object* v_a_822_; lean_object* v___x_823_; 
v_a_822_ = lean_ctor_get(v___x_820_, 0);
lean_inc(v_a_822_);
lean_dec_ref_known(v___x_820_, 1);
v___x_823_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyNonInitialQuery___redArg(v_inst_759_, v_inst_760_, v_inst_761_, v_inst_762_, v_fo_751_, v_snd_784_, v___x_789_, v_snd_763_, v_a_822_, v_a_816_, v_a_819_);
if (lean_obj_tag(v___x_823_) == 0)
{
lean_object* v___x_824_; 
lean_dec_ref_known(v___x_823_, 1);
lean_dec(v___x_790_);
lean_del_object(v___x_786_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
v___x_824_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_824_;
}
else
{
lean_object* v_a_825_; 
v_a_825_ = lean_ctor_get(v___x_823_, 0);
lean_inc(v_a_825_);
lean_dec_ref_known(v___x_823_, 1);
v_yi_792_ = v_a_825_;
v___y_793_ = v___y_772_;
goto v___jp_791_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_826_; 
lean_dec_ref(v_inst_761_);
v___x_826_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInitialQuery___redArg(v_inst_759_, v_inst_760_, v_inst_762_, v_fo_751_, v_algMap_764_, v_kWhir_753_, v_fst_783_, v_snd_784_, v___x_789_, v_snd_763_, v_muPows_765_, v_commitments_766_, v_widths_767_, v_initialRoundOpenedRows_768_, v_initialRoundMerkleProofs_769_);
if (lean_obj_tag(v___x_826_) == 0)
{
lean_object* v___x_827_; 
lean_dec_ref_known(v___x_826_, 1);
lean_dec(v___x_790_);
lean_del_object(v___x_786_);
lean_del_object(v___x_781_);
lean_dec(v_snd_779_);
lean_dec(v_fst_778_);
lean_del_object(v___x_776_);
lean_dec_ref(v___y_772_);
v___x_827_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___closed__0));
return v___x_827_;
}
else
{
lean_object* v_a_828_; 
v_a_828_ = lean_ctor_get(v___x_826_, 0);
lean_inc(v_a_828_);
lean_dec_ref_known(v___x_826_, 1);
v_yi_792_ = v_a_828_;
v___y_793_ = v___y_772_;
goto v___jp_791_;
}
}
v___jp_791_:
{
lean_object* v___x_795_; 
if (v_isShared_777_ == 0)
{
lean_ctor_set_tag(v___x_776_, 1);
lean_ctor_set(v___x_776_, 1, v_fst_778_);
lean_ctor_set(v___x_776_, 0, v___x_790_);
v___x_795_ = v___x_776_;
goto v_reusejp_794_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v___x_790_);
lean_ctor_set(v_reuseFailAlloc_804_, 1, v_fst_778_);
v___x_795_ = v_reuseFailAlloc_804_;
goto v_reusejp_794_;
}
v_reusejp_794_:
{
lean_object* v___x_796_; lean_object* v___x_798_; 
v___x_796_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_796_, 0, v_yi_792_);
lean_ctor_set(v___x_796_, 1, v_snd_779_);
if (v_isShared_787_ == 0)
{
lean_ctor_set(v___x_786_, 1, v___x_796_);
lean_ctor_set(v___x_786_, 0, v___x_795_);
v___x_798_ = v___x_786_;
goto v_reusejp_797_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v___x_795_);
lean_ctor_set(v_reuseFailAlloc_803_, 1, v___x_796_);
v___x_798_ = v_reuseFailAlloc_803_;
goto v_reusejp_797_;
}
v_reusejp_797_:
{
lean_object* v___x_800_; 
if (v_isShared_782_ == 0)
{
lean_ctor_set(v___x_781_, 1, v___y_793_);
lean_ctor_set(v___x_781_, 0, v___x_798_);
v___x_800_ = v___x_781_;
goto v_reusejp_799_;
}
else
{
lean_object* v_reuseFailAlloc_802_; 
v_reuseFailAlloc_802_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_802_, 0, v___x_798_);
lean_ctor_set(v_reuseFailAlloc_802_, 1, v___y_793_);
v___x_800_ = v_reuseFailAlloc_802_;
goto v_reusejp_799_;
}
v_reusejp_799_:
{
lean_object* v___x_801_; 
v___x_801_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_801_, 0, v___x_800_);
return v___x_801_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___boxed(lean_object** _args){
lean_object* v_fo_833_ = _args[0];
lean_object* v___x_834_ = _args[1];
lean_object* v_kWhir_835_ = _args[2];
lean_object* v_roundIdx_836_ = _args[3];
lean_object* v___x_837_ = _args[4];
lean_object* v_codewordOpenedValues_838_ = _args[5];
lean_object* v_codewordMerkleProofs_839_ = _args[6];
lean_object* v_codewordCommits_840_ = _args[7];
lean_object* v_inst_841_ = _args[8];
lean_object* v_inst_842_ = _args[9];
lean_object* v_inst_843_ = _args[10];
lean_object* v_inst_844_ = _args[11];
lean_object* v_snd_845_ = _args[12];
lean_object* v_algMap_846_ = _args[13];
lean_object* v_muPows_847_ = _args[14];
lean_object* v_commitments_848_ = _args[15];
lean_object* v_widths_849_ = _args[16];
lean_object* v_initialRoundOpenedRows_850_ = _args[17];
lean_object* v_initialRoundMerkleProofs_851_ = _args[18];
lean_object* v_acc_852_ = _args[19];
lean_object* v_pair_853_ = _args[20];
lean_object* v___y_854_ = _args[21];
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2(v_fo_833_, v___x_834_, v_kWhir_835_, v_roundIdx_836_, v___x_837_, v_codewordOpenedValues_838_, v_codewordMerkleProofs_839_, v_codewordCommits_840_, v_inst_841_, v_inst_842_, v_inst_843_, v_inst_844_, v_snd_845_, v_algMap_846_, v_muPows_847_, v_commitments_848_, v_widths_849_, v_initialRoundOpenedRows_850_, v_initialRoundMerkleProofs_851_, v_acc_852_, v_pair_853_, v___y_854_);
lean_dec(v_initialRoundMerkleProofs_851_);
lean_dec(v_initialRoundOpenedRows_850_);
lean_dec(v_codewordCommits_840_);
lean_dec(v_codewordMerkleProofs_839_);
lean_dec(v_codewordOpenedValues_838_);
lean_dec(v___x_837_);
lean_dec(v_roundIdx_836_);
lean_dec(v_kWhir_835_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg(lean_object* v_inst_911_, lean_object* v_inst_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_fo_918_, lean_object* v_algMap_919_, lean_object* v_params_920_, lean_object* v_whirProof_921_, lean_object* v_roundIdx_922_, lean_object* v_logRsDomainSize_923_, lean_object* v_claim_924_, lean_object* v_muPows_925_, lean_object* v_commitments_926_, lean_object* v_widths_927_, lean_object* v_a_928_){
_start:
{
lean_object* v_whirSumcheckPolys_929_; lean_object* v_codewordCommits_930_; lean_object* v_oodValues_931_; lean_object* v_foldingPowWitnesses_932_; lean_object* v_queryPhasePowWitnesses_933_; lean_object* v_initialRoundOpenedRows_934_; lean_object* v_initialRoundMerkleProofs_935_; lean_object* v_codewordOpenedValues_936_; lean_object* v_codewordMerkleProofs_937_; lean_object* v_finalPoly_938_; lean_object* v_kWhir_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v_polysSlice_944_; lean_object* v___x_945_; lean_object* v_powSlice_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v_whirSumcheckPolys_929_ = lean_ctor_get(v_whirProof_921_, 1);
lean_inc(v_whirSumcheckPolys_929_);
v_codewordCommits_930_ = lean_ctor_get(v_whirProof_921_, 2);
lean_inc(v_codewordCommits_930_);
v_oodValues_931_ = lean_ctor_get(v_whirProof_921_, 3);
lean_inc(v_oodValues_931_);
v_foldingPowWitnesses_932_ = lean_ctor_get(v_whirProof_921_, 4);
lean_inc(v_foldingPowWitnesses_932_);
v_queryPhasePowWitnesses_933_ = lean_ctor_get(v_whirProof_921_, 5);
lean_inc(v_queryPhasePowWitnesses_933_);
v_initialRoundOpenedRows_934_ = lean_ctor_get(v_whirProof_921_, 6);
lean_inc(v_initialRoundOpenedRows_934_);
v_initialRoundMerkleProofs_935_ = lean_ctor_get(v_whirProof_921_, 7);
lean_inc(v_initialRoundMerkleProofs_935_);
v_codewordOpenedValues_936_ = lean_ctor_get(v_whirProof_921_, 8);
lean_inc(v_codewordOpenedValues_936_);
v_codewordMerkleProofs_937_ = lean_ctor_get(v_whirProof_921_, 9);
lean_inc(v_codewordMerkleProofs_937_);
v_finalPoly_938_ = lean_ctor_get(v_whirProof_921_, 10);
lean_inc(v_finalPoly_938_);
lean_dec_ref(v_whirProof_921_);
v_kWhir_939_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_kWhir(v_params_920_);
v___x_940_ = lean_nat_mul(v_roundIdx_922_, v_kWhir_939_);
lean_inc(v___x_940_);
v___x_941_ = l_List_drop___redArg(v___x_940_, v_whirSumcheckPolys_929_);
lean_dec(v_whirSumcheckPolys_929_);
v___x_942_ = lean_unsigned_to_nat(0u);
v___x_943_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__10));
lean_inc_n(v_kWhir_939_, 2);
lean_inc(v___x_941_);
v_polysSlice_944_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v___x_941_, v___x_941_, v_kWhir_939_, v___x_943_);
lean_dec(v___x_941_);
v___x_945_ = l_List_drop___redArg(v___x_940_, v_foldingPowWitnesses_932_);
lean_dec(v_foldingPowWitnesses_932_);
lean_inc(v___x_945_);
v_powSlice_946_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v___x_945_, v___x_945_, v_kWhir_939_, v___x_943_);
lean_dec(v___x_945_);
v___x_947_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__20));
lean_inc_ref(v_fo_918_);
lean_inc_ref(v_inst_917_);
lean_inc_ref(v_inst_916_);
lean_inc_ref(v_inst_913_);
v___x_948_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyInnerSumcheckM___redArg(v_inst_913_, v_inst_916_, v_inst_917_, v_fo_918_, v_params_920_, v_claim_924_, v_polysSlice_944_, v_powSlice_946_, v_a_928_);
lean_dec(v_polysSlice_944_);
if (lean_obj_tag(v___x_948_) == 0)
{
lean_object* v_a_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_956_; 
lean_dec(v_kWhir_939_);
lean_dec(v_finalPoly_938_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_queryPhasePowWitnesses_933_);
lean_dec(v_oodValues_931_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec_ref(v_inst_915_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v_a_949_ = lean_ctor_get(v___x_948_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_948_);
if (v_isSharedCheck_956_ == 0)
{
v___x_951_ = v___x_948_;
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_a_949_);
lean_dec(v___x_948_);
v___x_951_ = lean_box(0);
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
v_resetjp_950_:
{
lean_object* v___x_954_; 
if (v_isShared_952_ == 0)
{
v___x_954_ = v___x_951_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v_a_949_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
else
{
lean_object* v_a_957_; lean_object* v___x_959_; uint8_t v_isShared_960_; uint8_t v_isSharedCheck_1123_; 
v_a_957_ = lean_ctor_get(v___x_948_, 0);
v_isSharedCheck_1123_ = !lean_is_exclusive(v___x_948_);
if (v_isSharedCheck_1123_ == 0)
{
v___x_959_ = v___x_948_;
v_isShared_960_ = v_isSharedCheck_1123_;
goto v_resetjp_958_;
}
else
{
lean_inc(v_a_957_);
lean_dec(v___x_948_);
v___x_959_ = lean_box(0);
v_isShared_960_ = v_isSharedCheck_1123_;
goto v_resetjp_958_;
}
v_resetjp_958_:
{
lean_object* v_fst_961_; lean_object* v_snd_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_1122_; 
v_fst_961_ = lean_ctor_get(v_a_957_, 0);
v_snd_962_ = lean_ctor_get(v_a_957_, 1);
v_isSharedCheck_1122_ = !lean_is_exclusive(v_a_957_);
if (v_isSharedCheck_1122_ == 0)
{
v___x_964_ = v_a_957_;
v_isShared_965_ = v_isSharedCheck_1122_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_snd_962_);
lean_inc(v_fst_961_);
lean_dec(v_a_957_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_1122_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v_fst_966_; lean_object* v_snd_967_; lean_object* v___x_969_; uint8_t v_isShared_970_; uint8_t v_isSharedCheck_1121_; 
v_fst_966_ = lean_ctor_get(v_fst_961_, 0);
v_snd_967_ = lean_ctor_get(v_fst_961_, 1);
v_isSharedCheck_1121_ = !lean_is_exclusive(v_fst_961_);
if (v_isSharedCheck_1121_ == 0)
{
v___x_969_ = v_fst_961_;
v_isShared_970_ = v_isSharedCheck_1121_;
goto v_resetjp_968_;
}
else
{
lean_inc(v_snd_967_);
lean_inc(v_fst_966_);
lean_dec(v_fst_961_);
v___x_969_ = lean_box(0);
v_isShared_970_ = v_isSharedCheck_1121_;
goto v_resetjp_968_;
}
v_resetjp_968_:
{
lean_object* v___f_971_; lean_object* v___y_973_; lean_object* v___y_974_; lean_object* v___y_975_; lean_object* v___y_976_; lean_object* v___y_977_; lean_object* v___y_978_; lean_object* v_fst_999_; lean_object* v_snd_1000_; lean_object* v___y_1001_; lean_object* v_numWhirRounds_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; uint8_t v___x_1078_; 
lean_inc_ref(v_fo_918_);
v___f_971_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__0), 3, 1);
lean_closure_set(v___f_971_, 0, v_fo_918_);
v_numWhirRounds_1075_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_numWhirRounds(v_params_920_);
v___x_1076_ = lean_unsigned_to_nat(1u);
v___x_1077_ = lean_nat_add(v_roundIdx_922_, v___x_1076_);
v___x_1078_ = lean_nat_dec_eq(v___x_1077_, v_numWhirRounds_1075_);
lean_dec(v_numWhirRounds_1075_);
lean_dec(v___x_1077_);
if (v___x_1078_ == 0)
{
lean_object* v___x_1079_; 
lean_dec(v_finalPoly_938_);
lean_inc(v_roundIdx_922_);
v___x_1079_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_codewordCommits_930_, v_roundIdx_922_);
if (lean_obj_tag(v___x_1079_) == 0)
{
lean_object* v___x_1080_; 
lean_dec_ref_known(v___x_1079_, 1);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_dec(v_snd_962_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_queryPhasePowWitnesses_933_);
lean_dec(v_oodValues_931_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec_ref(v_inst_915_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v___x_1080_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21));
return v___x_1080_;
}
else
{
lean_object* v_a_1081_; lean_object* v___x_1082_; lean_object* v_a_1083_; lean_object* v_snd_1084_; lean_object* v___x_1085_; lean_object* v_a_1086_; lean_object* v_fst_1087_; lean_object* v_snd_1088_; lean_object* v___x_1089_; 
v_a_1081_ = lean_ctor_get(v___x_1079_, 0);
lean_inc(v_a_1081_);
lean_dec_ref_known(v___x_1079_, 1);
lean_inc_ref_n(v_inst_916_, 2);
v___x_1082_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeCommit___redArg(v_inst_916_, v_inst_915_, v_a_1081_, v_snd_962_);
v_a_1083_ = lean_ctor_get(v___x_1082_, 0);
lean_inc(v_a_1083_);
lean_dec_ref(v___x_1082_);
v_snd_1084_ = lean_ctor_get(v_a_1083_, 1);
lean_inc(v_snd_1084_);
lean_dec(v_a_1083_);
lean_inc_ref(v_inst_913_);
v___x_1085_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_916_, v_inst_913_, v_snd_1084_);
v_a_1086_ = lean_ctor_get(v___x_1085_, 0);
lean_inc(v_a_1086_);
lean_dec_ref(v___x_1085_);
v_fst_1087_ = lean_ctor_get(v_a_1086_, 0);
lean_inc(v_fst_1087_);
v_snd_1088_ = lean_ctor_get(v_a_1086_, 1);
lean_inc(v_snd_1088_);
lean_dec(v_a_1086_);
lean_inc(v_roundIdx_922_);
v___x_1089_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_oodValues_931_, v_roundIdx_922_);
lean_dec(v_oodValues_931_);
if (lean_obj_tag(v___x_1089_) == 0)
{
lean_object* v___x_1090_; 
lean_dec_ref_known(v___x_1089_, 1);
lean_dec(v_snd_1088_);
lean_dec(v_fst_1087_);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_queryPhasePowWitnesses_933_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v___x_1090_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21));
return v___x_1090_;
}
else
{
lean_object* v_a_1091_; lean_object* v___x_1093_; uint8_t v_isShared_1094_; uint8_t v_isSharedCheck_1108_; 
v_a_1091_ = lean_ctor_get(v___x_1089_, 0);
v_isSharedCheck_1108_ = !lean_is_exclusive(v___x_1089_);
if (v_isSharedCheck_1108_ == 0)
{
v___x_1093_ = v___x_1089_;
v_isShared_1094_ = v_isSharedCheck_1108_;
goto v_resetjp_1092_;
}
else
{
lean_inc(v_a_1091_);
lean_dec(v___x_1089_);
v___x_1093_ = lean_box(0);
v_isShared_1094_ = v_isSharedCheck_1108_;
goto v_resetjp_1092_;
}
v_resetjp_1092_:
{
lean_object* v___x_1095_; lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1107_; 
lean_inc(v_a_1091_);
lean_inc_ref(v_inst_913_);
lean_inc_ref(v_inst_916_);
v___x_1095_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_916_, v_inst_913_, v_a_1091_, v_snd_1088_);
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1107_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1107_ == 0)
{
v___x_1098_ = v___x_1095_;
v_isShared_1099_ = v_isSharedCheck_1107_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v___x_1095_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1107_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v_snd_1100_; lean_object* v___x_1102_; 
v_snd_1100_ = lean_ctor_get(v_a_1096_, 1);
lean_inc(v_snd_1100_);
lean_dec(v_a_1096_);
if (v_isShared_1099_ == 0)
{
lean_ctor_set(v___x_1098_, 0, v_a_1091_);
v___x_1102_ = v___x_1098_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1106_; 
v_reuseFailAlloc_1106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1106_, 0, v_a_1091_);
v___x_1102_ = v_reuseFailAlloc_1106_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
lean_object* v___x_1104_; 
if (v_isShared_1094_ == 0)
{
lean_ctor_set(v___x_1093_, 0, v_fst_1087_);
v___x_1104_ = v___x_1093_;
goto v_reusejp_1103_;
}
else
{
lean_object* v_reuseFailAlloc_1105_; 
v_reuseFailAlloc_1105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1105_, 0, v_fst_1087_);
v___x_1104_ = v_reuseFailAlloc_1105_;
goto v_reusejp_1103_;
}
v_reusejp_1103_:
{
v_fst_999_ = v___x_1102_;
v_snd_1000_ = v___x_1104_;
v___y_1001_ = v_snd_1100_;
goto v___jp_998_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1109_; 
lean_dec(v_oodValues_931_);
lean_dec_ref(v_inst_915_);
lean_inc_ref(v_inst_913_);
lean_inc_ref(v_inst_916_);
v___x_1109_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(v_inst_916_, v_inst_913_, v_finalPoly_938_, v_snd_962_);
if (lean_obj_tag(v___x_1109_) == 0)
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_queryPhasePowWitnesses_933_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v_a_1110_ = lean_ctor_get(v___x_1109_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1109_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1109_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1109_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
else
{
lean_object* v_a_1118_; lean_object* v_snd_1119_; lean_object* v___x_1120_; 
v_a_1118_ = lean_ctor_get(v___x_1109_, 0);
lean_inc(v_a_1118_);
lean_dec_ref_known(v___x_1109_, 1);
v_snd_1119_ = lean_ctor_get(v_a_1118_, 1);
lean_inc(v_snd_1119_);
lean_dec(v_a_1118_);
v___x_1120_ = lean_box(0);
v_fst_999_ = v___x_1120_;
v_snd_1000_ = v___x_1120_;
v___y_1001_ = v_snd_1119_;
goto v___jp_998_;
}
}
v___jp_972_:
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_987_; 
v___x_979_ = lean_unsigned_to_nat(2u);
v___x_980_ = l_List_lengthTR___redArg(v___y_976_);
v___x_981_ = lean_nat_add(v___x_980_, v___x_979_);
lean_dec(v___x_980_);
lean_inc(v___y_975_);
v___x_982_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_918_, v___y_975_, v___x_981_);
v___x_983_ = l_List_drop___redArg(v___x_979_, v___x_982_);
lean_dec(v___x_982_);
v___x_984_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___y_976_, v___x_983_);
v___x_985_ = l_List_foldl___redArg(v___f_971_, v___y_978_, v___x_984_);
if (v_isShared_970_ == 0)
{
lean_ctor_set(v___x_969_, 1, v___y_974_);
lean_ctor_set(v___x_969_, 0, v___y_975_);
v___x_987_ = v___x_969_;
goto v_reusejp_986_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v___y_975_);
lean_ctor_set(v_reuseFailAlloc_997_, 1, v___y_974_);
v___x_987_ = v_reuseFailAlloc_997_;
goto v_reusejp_986_;
}
v_reusejp_986_:
{
lean_object* v___x_989_; 
if (v_isShared_965_ == 0)
{
lean_ctor_set(v___x_964_, 1, v___x_987_);
lean_ctor_set(v___x_964_, 0, v_snd_967_);
v___x_989_ = v___x_964_;
goto v_reusejp_988_;
}
else
{
lean_object* v_reuseFailAlloc_996_; 
v_reuseFailAlloc_996_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_996_, 0, v_snd_967_);
lean_ctor_set(v_reuseFailAlloc_996_, 1, v___x_987_);
v___x_989_ = v_reuseFailAlloc_996_;
goto v_reusejp_988_;
}
v_reusejp_988_:
{
lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_994_; 
v___x_990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_990_, 0, v___y_973_);
lean_ctor_set(v___x_990_, 1, v___x_989_);
v___x_991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_991_, 0, v___x_985_);
lean_ctor_set(v___x_991_, 1, v___x_990_);
v___x_992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_992_, 0, v___x_991_);
lean_ctor_set(v___x_992_, 1, v___y_977_);
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 0, v___x_992_);
v___x_994_ = v___x_959_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_995_; 
v_reuseFailAlloc_995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_995_, 0, v___x_992_);
v___x_994_ = v_reuseFailAlloc_995_;
goto v_reusejp_993_;
}
v_reusejp_993_:
{
return v___x_994_;
}
}
}
}
v___jp_998_:
{
lean_object* v___x_1002_; 
lean_inc(v_roundIdx_922_);
v___x_1002_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_queryPhasePowWitnesses_933_, v_roundIdx_922_);
lean_dec(v_queryPhasePowWitnesses_933_);
if (lean_obj_tag(v___x_1002_) == 0)
{
lean_object* v___x_1003_; 
lean_dec_ref_known(v___x_1002_, 1);
lean_dec_ref(v___y_1001_);
lean_dec(v_snd_1000_);
lean_dec(v_fst_999_);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v___x_1003_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21));
return v___x_1003_;
}
else
{
lean_object* v_whir_1004_; lean_object* v_a_1005_; lean_object* v_rounds_1006_; lean_object* v_queryPhasePowBits_1007_; lean_object* v___x_1008_; 
v_whir_1004_ = lean_ctor_get(v_params_920_, 4);
v_a_1005_ = lean_ctor_get(v___x_1002_, 0);
lean_inc(v_a_1005_);
lean_dec_ref_known(v___x_1002_, 1);
v_rounds_1006_ = lean_ctor_get(v_whir_1004_, 1);
v_queryPhasePowBits_1007_ = lean_ctor_get(v_whir_1004_, 3);
lean_inc_ref(v_inst_917_);
lean_inc_ref(v_inst_916_);
v___x_1008_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observePowWitness___redArg(v_inst_916_, v_inst_917_, v_queryPhasePowBits_1007_, v_a_1005_, v___y_1001_);
if (lean_obj_tag(v___x_1008_) == 0)
{
lean_object* v_a_1009_; lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1016_; 
lean_dec(v_snd_1000_);
lean_dec(v_fst_999_);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v_a_1009_ = lean_ctor_get(v___x_1008_, 0);
v_isSharedCheck_1016_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1016_ == 0)
{
v___x_1011_ = v___x_1008_;
v_isShared_1012_ = v_isSharedCheck_1016_;
goto v_resetjp_1010_;
}
else
{
lean_inc(v_a_1009_);
lean_dec(v___x_1008_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1016_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
lean_object* v___x_1014_; 
if (v_isShared_1012_ == 0)
{
v___x_1014_ = v___x_1011_;
goto v_reusejp_1013_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v_a_1009_);
v___x_1014_ = v_reuseFailAlloc_1015_;
goto v_reusejp_1013_;
}
v_reusejp_1013_:
{
return v___x_1014_;
}
}
}
else
{
lean_object* v_a_1017_; lean_object* v_snd_1018_; lean_object* v___x_1019_; 
v_a_1017_ = lean_ctor_get(v___x_1008_, 0);
lean_inc(v_a_1017_);
lean_dec_ref_known(v___x_1008_, 1);
v_snd_1018_ = lean_ctor_get(v_a_1017_, 1);
lean_inc(v_snd_1018_);
lean_dec(v_a_1017_);
lean_inc(v_roundIdx_922_);
v___x_1019_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_rounds_1006_, v_roundIdx_922_);
if (lean_obj_tag(v___x_1019_) == 0)
{
lean_object* v___x_1020_; 
lean_dec_ref_known(v___x_1019_, 1);
lean_dec(v_snd_1018_);
lean_dec(v_snd_1000_);
lean_dec(v_fst_999_);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v___x_1020_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__21));
return v___x_1020_;
}
else
{
lean_object* v_a_1021_; lean_object* v___x_1022_; lean_object* v___f_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_10632__overap_1026_; lean_object* v___x_1027_; 
v_a_1021_ = lean_ctor_get(v___x_1019_, 0);
lean_inc(v_a_1021_);
lean_dec_ref_known(v___x_1019_, 1);
v___x_1022_ = lean_nat_sub(v_logRsDomainSize_923_, v_kWhir_939_);
lean_inc_ref(v_inst_916_);
v___f_1023_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__1___boxed), 6, 3);
lean_closure_set(v___f_1023_, 0, v_inst_916_);
lean_closure_set(v___f_1023_, 1, v_inst_917_);
lean_closure_set(v___f_1023_, 2, v___x_1022_);
v___x_1024_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__22));
v___x_1025_ = l_List_range(v_a_1021_);
v___x_10632__overap_1026_ = l_List_foldlM___redArg(v___x_947_, v___f_1023_, v___x_1024_, v___x_1025_);
v___x_1027_ = lean_apply_1(v___x_10632__overap_1026_, v_snd_1018_);
if (lean_obj_tag(v___x_1027_) == 0)
{
lean_object* v_a_1028_; lean_object* v___x_1030_; uint8_t v_isShared_1031_; uint8_t v_isSharedCheck_1035_; 
lean_dec(v_snd_1000_);
lean_dec(v_fst_999_);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec(v_kWhir_939_);
lean_dec(v_codewordMerkleProofs_937_);
lean_dec(v_codewordOpenedValues_936_);
lean_dec(v_initialRoundMerkleProofs_935_);
lean_dec(v_initialRoundOpenedRows_934_);
lean_dec(v_codewordCommits_930_);
lean_dec(v_widths_927_);
lean_dec(v_commitments_926_);
lean_dec(v_muPows_925_);
lean_dec(v_logRsDomainSize_923_);
lean_dec(v_roundIdx_922_);
lean_dec(v_algMap_919_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_916_);
lean_dec(v_inst_914_);
lean_dec_ref(v_inst_913_);
lean_dec_ref(v_inst_912_);
lean_dec_ref(v_inst_911_);
v_a_1028_ = lean_ctor_get(v___x_1027_, 0);
v_isSharedCheck_1035_ = !lean_is_exclusive(v___x_1027_);
if (v_isSharedCheck_1035_ == 0)
{
v___x_1030_ = v___x_1027_;
v_isShared_1031_ = v_isSharedCheck_1035_;
goto v_resetjp_1029_;
}
else
{
lean_inc(v_a_1028_);
lean_dec(v___x_1027_);
v___x_1030_ = lean_box(0);
v_isShared_1031_ = v_isSharedCheck_1035_;
goto v_resetjp_1029_;
}
v_resetjp_1029_:
{
lean_object* v___x_1033_; 
if (v_isShared_1031_ == 0)
{
v___x_1033_ = v___x_1030_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v_a_1028_);
v___x_1033_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
return v___x_1033_;
}
}
}
else
{
lean_object* v_a_1036_; lean_object* v_fst_1037_; lean_object* v_snd_1038_; lean_object* v_fst_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___f_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_10705__overap_1047_; lean_object* v___x_1048_; 
v_a_1036_ = lean_ctor_get(v___x_1027_, 0);
lean_inc(v_a_1036_);
lean_dec_ref_known(v___x_1027_, 1);
v_fst_1037_ = lean_ctor_get(v_a_1036_, 0);
lean_inc(v_fst_1037_);
v_snd_1038_ = lean_ctor_get(v_a_1036_, 1);
lean_inc(v_snd_1038_);
lean_dec(v_a_1036_);
v_fst_1039_ = lean_ctor_get(v_fst_1037_, 0);
lean_inc(v_fst_1039_);
lean_dec(v_fst_1037_);
v___x_1040_ = l_List_reverse___redArg(v_fst_1039_);
lean_inc(v_inst_914_);
v___x_1041_ = lean_apply_1(v_inst_914_, v_logRsDomainSize_923_);
lean_inc(v_snd_967_);
lean_inc_ref(v_inst_913_);
lean_inc_ref(v_fo_918_);
v___f_1042_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___lam__2___boxed), 22, 19);
lean_closure_set(v___f_1042_, 0, v_fo_918_);
lean_closure_set(v___f_1042_, 1, v___x_1041_);
lean_closure_set(v___f_1042_, 2, v_kWhir_939_);
lean_closure_set(v___f_1042_, 3, v_roundIdx_922_);
lean_closure_set(v___f_1042_, 4, v___x_942_);
lean_closure_set(v___f_1042_, 5, v_codewordOpenedValues_936_);
lean_closure_set(v___f_1042_, 6, v_codewordMerkleProofs_937_);
lean_closure_set(v___f_1042_, 7, v_codewordCommits_930_);
lean_closure_set(v___f_1042_, 8, v_inst_911_);
lean_closure_set(v___f_1042_, 9, v_inst_912_);
lean_closure_set(v___f_1042_, 10, v_inst_913_);
lean_closure_set(v___f_1042_, 11, v_inst_914_);
lean_closure_set(v___f_1042_, 12, v_snd_967_);
lean_closure_set(v___f_1042_, 13, v_algMap_919_);
lean_closure_set(v___f_1042_, 14, v_muPows_925_);
lean_closure_set(v___f_1042_, 15, v_commitments_926_);
lean_closure_set(v___f_1042_, 16, v_widths_927_);
lean_closure_set(v___f_1042_, 17, v_initialRoundOpenedRows_934_);
lean_closure_set(v___f_1042_, 18, v_initialRoundMerkleProofs_935_);
v___x_1043_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__23));
v___x_1044_ = l_List_lengthTR___redArg(v___x_1040_);
v___x_1045_ = l_List_range(v___x_1044_);
v___x_1046_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1045_, v___x_1040_);
v___x_10705__overap_1047_ = l_List_foldlM___redArg(v___x_947_, v___f_1042_, v___x_1043_, v___x_1046_);
v___x_1048_ = lean_apply_1(v___x_10705__overap_1047_, v_snd_1038_);
if (lean_obj_tag(v___x_1048_) == 0)
{
lean_object* v_a_1049_; lean_object* v___x_1051_; uint8_t v_isShared_1052_; uint8_t v_isSharedCheck_1056_; 
lean_dec(v_snd_1000_);
lean_dec(v_fst_999_);
lean_dec_ref(v___f_971_);
lean_del_object(v___x_969_);
lean_dec(v_snd_967_);
lean_dec(v_fst_966_);
lean_del_object(v___x_964_);
lean_del_object(v___x_959_);
lean_dec_ref(v_fo_918_);
lean_dec_ref(v_inst_916_);
lean_dec_ref(v_inst_913_);
v_a_1049_ = lean_ctor_get(v___x_1048_, 0);
v_isSharedCheck_1056_ = !lean_is_exclusive(v___x_1048_);
if (v_isSharedCheck_1056_ == 0)
{
v___x_1051_ = v___x_1048_;
v_isShared_1052_ = v_isSharedCheck_1056_;
goto v_resetjp_1050_;
}
else
{
lean_inc(v_a_1049_);
lean_dec(v___x_1048_);
v___x_1051_ = lean_box(0);
v_isShared_1052_ = v_isSharedCheck_1056_;
goto v_resetjp_1050_;
}
v_resetjp_1050_:
{
lean_object* v___x_1054_; 
if (v_isShared_1052_ == 0)
{
v___x_1054_ = v___x_1051_;
goto v_reusejp_1053_;
}
else
{
lean_object* v_reuseFailAlloc_1055_; 
v_reuseFailAlloc_1055_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1055_, 0, v_a_1049_);
v___x_1054_ = v_reuseFailAlloc_1055_;
goto v_reusejp_1053_;
}
v_reusejp_1053_:
{
return v___x_1054_;
}
}
}
else
{
lean_object* v_a_1057_; lean_object* v_fst_1058_; lean_object* v_snd_1059_; lean_object* v_fst_1060_; lean_object* v_snd_1061_; lean_object* v___x_1062_; lean_object* v_a_1063_; lean_object* v_fst_1064_; lean_object* v_snd_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; 
v_a_1057_ = lean_ctor_get(v___x_1048_, 0);
lean_inc(v_a_1057_);
lean_dec_ref_known(v___x_1048_, 1);
v_fst_1058_ = lean_ctor_get(v_a_1057_, 0);
lean_inc(v_fst_1058_);
v_snd_1059_ = lean_ctor_get(v_a_1057_, 1);
lean_inc(v_snd_1059_);
lean_dec(v_a_1057_);
v_fst_1060_ = lean_ctor_get(v_fst_1058_, 0);
lean_inc(v_fst_1060_);
v_snd_1061_ = lean_ctor_get(v_fst_1058_, 1);
lean_inc(v_snd_1061_);
lean_dec(v_fst_1058_);
v___x_1062_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_916_, v_inst_913_, v_snd_1059_);
v_a_1063_ = lean_ctor_get(v___x_1062_, 0);
lean_inc(v_a_1063_);
lean_dec_ref(v___x_1062_);
v_fst_1064_ = lean_ctor_get(v_a_1063_, 0);
lean_inc(v_fst_1064_);
v_snd_1065_ = lean_ctor_get(v_a_1063_, 1);
lean_inc(v_snd_1065_);
lean_dec(v_a_1063_);
v___x_1066_ = l_List_reverse___redArg(v_fst_1060_);
v___x_1067_ = l_List_reverse___redArg(v_snd_1061_);
if (lean_obj_tag(v_fst_999_) == 0)
{
v___y_973_ = v___x_1066_;
v___y_974_ = v_snd_1000_;
v___y_975_ = v_fst_1064_;
v___y_976_ = v___x_1067_;
v___y_977_ = v_snd_1065_;
v___y_978_ = v_fst_966_;
goto v___jp_972_;
}
else
{
lean_object* v_toRingOps_1068_; lean_object* v_toSemiringOps_1069_; lean_object* v_val_1070_; lean_object* v_add_1071_; lean_object* v_mul_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
v_toRingOps_1068_ = lean_ctor_get(v_fo_918_, 0);
v_toSemiringOps_1069_ = lean_ctor_get(v_toRingOps_1068_, 0);
v_val_1070_ = lean_ctor_get(v_fst_999_, 0);
lean_inc(v_val_1070_);
lean_dec_ref_known(v_fst_999_, 1);
v_add_1071_ = lean_ctor_get(v_toSemiringOps_1069_, 3);
v_mul_1072_ = lean_ctor_get(v_toSemiringOps_1069_, 4);
lean_inc(v_mul_1072_);
lean_inc(v_fst_1064_);
v___x_1073_ = lean_apply_2(v_mul_1072_, v_val_1070_, v_fst_1064_);
lean_inc(v_add_1071_);
v___x_1074_ = lean_apply_2(v_add_1071_, v_fst_966_, v___x_1073_);
v___y_973_ = v___x_1066_;
v___y_974_ = v_snd_1000_;
v___y_975_ = v_fst_1064_;
v___y_976_ = v___x_1067_;
v___y_977_ = v_snd_1065_;
v___y_978_ = v___x_1074_;
goto v___jp_972_;
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___boxed(lean_object** _args){
lean_object* v_inst_1124_ = _args[0];
lean_object* v_inst_1125_ = _args[1];
lean_object* v_inst_1126_ = _args[2];
lean_object* v_inst_1127_ = _args[3];
lean_object* v_inst_1128_ = _args[4];
lean_object* v_inst_1129_ = _args[5];
lean_object* v_inst_1130_ = _args[6];
lean_object* v_fo_1131_ = _args[7];
lean_object* v_algMap_1132_ = _args[8];
lean_object* v_params_1133_ = _args[9];
lean_object* v_whirProof_1134_ = _args[10];
lean_object* v_roundIdx_1135_ = _args[11];
lean_object* v_logRsDomainSize_1136_ = _args[12];
lean_object* v_claim_1137_ = _args[13];
lean_object* v_muPows_1138_ = _args[14];
lean_object* v_commitments_1139_ = _args[15];
lean_object* v_widths_1140_ = _args[16];
lean_object* v_a_1141_ = _args[17];
_start:
{
lean_object* v_res_1142_; 
v_res_1142_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg(v_inst_1124_, v_inst_1125_, v_inst_1126_, v_inst_1127_, v_inst_1128_, v_inst_1129_, v_inst_1130_, v_fo_1131_, v_algMap_1132_, v_params_1133_, v_whirProof_1134_, v_roundIdx_1135_, v_logRsDomainSize_1136_, v_claim_1137_, v_muPows_1138_, v_commitments_1139_, v_widths_1140_, v_a_1141_);
lean_dec_ref(v_params_1133_);
return v_res_1142_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM(lean_object* v_F_1143_, lean_object* v_EF_1144_, lean_object* v_Digest_1145_, lean_object* v_inst_1146_, lean_object* v_inst_1147_, lean_object* v_inst_1148_, lean_object* v_inst_1149_, lean_object* v_inst_1150_, lean_object* v_inst_1151_, lean_object* v_inst_1152_, lean_object* v_fo_1153_, lean_object* v_algMap_1154_, lean_object* v_params_1155_, lean_object* v_whirProof_1156_, lean_object* v_roundIdx_1157_, lean_object* v_logRsDomainSize_1158_, lean_object* v_claim_1159_, lean_object* v_muPows_1160_, lean_object* v_commitments_1161_, lean_object* v_widths_1162_, lean_object* v_a_1163_){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg(v_inst_1146_, v_inst_1147_, v_inst_1148_, v_inst_1149_, v_inst_1150_, v_inst_1151_, v_inst_1152_, v_fo_1153_, v_algMap_1154_, v_params_1155_, v_whirProof_1156_, v_roundIdx_1157_, v_logRsDomainSize_1158_, v_claim_1159_, v_muPows_1160_, v_commitments_1161_, v_widths_1162_, v_a_1163_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___boxed(lean_object** _args){
lean_object* v_F_1165_ = _args[0];
lean_object* v_EF_1166_ = _args[1];
lean_object* v_Digest_1167_ = _args[2];
lean_object* v_inst_1168_ = _args[3];
lean_object* v_inst_1169_ = _args[4];
lean_object* v_inst_1170_ = _args[5];
lean_object* v_inst_1171_ = _args[6];
lean_object* v_inst_1172_ = _args[7];
lean_object* v_inst_1173_ = _args[8];
lean_object* v_inst_1174_ = _args[9];
lean_object* v_fo_1175_ = _args[10];
lean_object* v_algMap_1176_ = _args[11];
lean_object* v_params_1177_ = _args[12];
lean_object* v_whirProof_1178_ = _args[13];
lean_object* v_roundIdx_1179_ = _args[14];
lean_object* v_logRsDomainSize_1180_ = _args[15];
lean_object* v_claim_1181_ = _args[16];
lean_object* v_muPows_1182_ = _args[17];
lean_object* v_commitments_1183_ = _args[18];
lean_object* v_widths_1184_ = _args[19];
lean_object* v_a_1185_ = _args[20];
_start:
{
lean_object* v_res_1186_; 
v_res_1186_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM(v_F_1165_, v_EF_1166_, v_Digest_1167_, v_inst_1168_, v_inst_1169_, v_inst_1170_, v_inst_1171_, v_inst_1172_, v_inst_1173_, v_inst_1174_, v_fo_1175_, v_algMap_1176_, v_params_1177_, v_whirProof_1178_, v_roundIdx_1179_, v_logRsDomainSize_1180_, v_claim_1181_, v_muPows_1182_, v_commitments_1183_, v_widths_1184_, v_a_1185_);
lean_dec_ref(v_params_1177_);
return v_res_1186_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__0(lean_object* v_mul_1187_, lean_object* v_add_1188_, lean_object* v_acc_1189_, lean_object* v_entry_1190_){
_start:
{
lean_object* v_fst_1191_; lean_object* v_snd_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; 
v_fst_1191_ = lean_ctor_get(v_entry_1190_, 0);
lean_inc(v_fst_1191_);
v_snd_1192_ = lean_ctor_get(v_entry_1190_, 1);
lean_inc(v_snd_1192_);
lean_dec_ref(v_entry_1190_);
v___x_1193_ = lean_apply_2(v_mul_1187_, v_fst_1191_, v_snd_1192_);
v___x_1194_ = lean_apply_2(v_add_1188_, v_acc_1189_, v___x_1193_);
return v___x_1194_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__1(lean_object* v___x_1195_, lean_object* v_inst_1196_, lean_object* v_inst_1197_, lean_object* v_inst_1198_, lean_object* v_inst_1199_, lean_object* v_inst_1200_, lean_object* v_inst_1201_, lean_object* v_inst_1202_, lean_object* v_fo_1203_, lean_object* v_algMap_1204_, lean_object* v_params_1205_, lean_object* v_whirProof_1206_, lean_object* v___x_1207_, lean_object* v_commits_1208_, lean_object* v_widths_1209_, lean_object* v_acc_1210_, lean_object* v_roundIdx_1211_, lean_object* v___y_1212_){
_start:
{
lean_object* v_snd_1213_; lean_object* v_snd_1214_; lean_object* v_snd_1215_; lean_object* v_fst_1216_; lean_object* v_fst_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1318_; 
v_snd_1213_ = lean_ctor_get(v_acc_1210_, 1);
lean_inc(v_snd_1213_);
v_snd_1214_ = lean_ctor_get(v_snd_1213_, 1);
lean_inc(v_snd_1214_);
v_snd_1215_ = lean_ctor_get(v_snd_1214_, 1);
lean_inc(v_snd_1215_);
v_fst_1216_ = lean_ctor_get(v_acc_1210_, 0);
lean_inc(v_fst_1216_);
lean_dec_ref(v_acc_1210_);
v_fst_1217_ = lean_ctor_get(v_snd_1213_, 0);
v_isSharedCheck_1318_ = !lean_is_exclusive(v_snd_1213_);
if (v_isSharedCheck_1318_ == 0)
{
lean_object* v_unused_1319_; 
v_unused_1319_ = lean_ctor_get(v_snd_1213_, 1);
lean_dec(v_unused_1319_);
v___x_1219_ = v_snd_1213_;
v_isShared_1220_ = v_isSharedCheck_1318_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_fst_1217_);
lean_dec(v_snd_1213_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1318_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v_fst_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1316_; 
v_fst_1221_ = lean_ctor_get(v_snd_1214_, 0);
v_isSharedCheck_1316_ = !lean_is_exclusive(v_snd_1214_);
if (v_isSharedCheck_1316_ == 0)
{
lean_object* v_unused_1317_; 
v_unused_1317_ = lean_ctor_get(v_snd_1214_, 1);
lean_dec(v_unused_1317_);
v___x_1223_ = v_snd_1214_;
v_isShared_1224_ = v_isSharedCheck_1316_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_fst_1221_);
lean_dec(v_snd_1214_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1316_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v_fst_1225_; lean_object* v_snd_1226_; lean_object* v___x_1228_; uint8_t v_isShared_1229_; uint8_t v_isSharedCheck_1315_; 
v_fst_1225_ = lean_ctor_get(v_snd_1215_, 0);
v_snd_1226_ = lean_ctor_get(v_snd_1215_, 1);
v_isSharedCheck_1315_ = !lean_is_exclusive(v_snd_1215_);
if (v_isSharedCheck_1315_ == 0)
{
v___x_1228_ = v_snd_1215_;
v_isShared_1229_ = v_isSharedCheck_1315_;
goto v_resetjp_1227_;
}
else
{
lean_inc(v_snd_1226_);
lean_inc(v_fst_1225_);
lean_dec(v_snd_1215_);
v___x_1228_ = lean_box(0);
v_isShared_1229_ = v_isSharedCheck_1315_;
goto v_resetjp_1227_;
}
v_resetjp_1227_:
{
lean_object* v___x_1230_; lean_object* v___x_1231_; 
v___x_1230_ = lean_nat_sub(v___x_1195_, v_roundIdx_1211_);
v___x_1231_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg(v_inst_1196_, v_inst_1197_, v_inst_1198_, v_inst_1199_, v_inst_1200_, v_inst_1201_, v_inst_1202_, v_fo_1203_, v_algMap_1204_, v_params_1205_, v_whirProof_1206_, v_roundIdx_1211_, v___x_1230_, v_fst_1216_, v___x_1207_, v_commits_1208_, v_widths_1209_, v___y_1212_);
if (lean_obj_tag(v___x_1231_) == 0)
{
lean_object* v_a_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1239_; 
lean_del_object(v___x_1228_);
lean_dec(v_snd_1226_);
lean_dec(v_fst_1225_);
lean_del_object(v___x_1223_);
lean_dec(v_fst_1221_);
lean_del_object(v___x_1219_);
lean_dec(v_fst_1217_);
v_a_1232_ = lean_ctor_get(v___x_1231_, 0);
v_isSharedCheck_1239_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1239_ == 0)
{
v___x_1234_ = v___x_1231_;
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_a_1232_);
lean_dec(v___x_1231_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v___x_1237_; 
if (v_isShared_1235_ == 0)
{
v___x_1237_ = v___x_1234_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1238_; 
v_reuseFailAlloc_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1238_, 0, v_a_1232_);
v___x_1237_ = v_reuseFailAlloc_1238_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
return v___x_1237_;
}
}
}
else
{
lean_object* v_a_1240_; lean_object* v___x_1242_; uint8_t v_isShared_1243_; uint8_t v_isSharedCheck_1314_; 
v_a_1240_ = lean_ctor_get(v___x_1231_, 0);
v_isSharedCheck_1314_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1314_ == 0)
{
v___x_1242_ = v___x_1231_;
v_isShared_1243_ = v_isSharedCheck_1314_;
goto v_resetjp_1241_;
}
else
{
lean_inc(v_a_1240_);
lean_dec(v___x_1231_);
v___x_1242_ = lean_box(0);
v_isShared_1243_ = v_isSharedCheck_1314_;
goto v_resetjp_1241_;
}
v_resetjp_1241_:
{
lean_object* v_fst_1244_; lean_object* v_snd_1245_; lean_object* v_snd_1246_; lean_object* v_snd_1247_; lean_object* v_snd_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1312_; 
v_fst_1244_ = lean_ctor_get(v_a_1240_, 0);
lean_inc(v_fst_1244_);
v_snd_1245_ = lean_ctor_get(v_fst_1244_, 1);
lean_inc(v_snd_1245_);
v_snd_1246_ = lean_ctor_get(v_snd_1245_, 1);
lean_inc(v_snd_1246_);
v_snd_1247_ = lean_ctor_get(v_snd_1246_, 1);
lean_inc(v_snd_1247_);
v_snd_1248_ = lean_ctor_get(v_a_1240_, 1);
v_isSharedCheck_1312_ = !lean_is_exclusive(v_a_1240_);
if (v_isSharedCheck_1312_ == 0)
{
lean_object* v_unused_1313_; 
v_unused_1313_ = lean_ctor_get(v_a_1240_, 0);
lean_dec(v_unused_1313_);
v___x_1250_ = v_a_1240_;
v_isShared_1251_ = v_isSharedCheck_1312_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_snd_1248_);
lean_dec(v_a_1240_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1312_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v_fst_1252_; lean_object* v___x_1254_; uint8_t v_isShared_1255_; uint8_t v_isSharedCheck_1310_; 
v_fst_1252_ = lean_ctor_get(v_fst_1244_, 0);
v_isSharedCheck_1310_ = !lean_is_exclusive(v_fst_1244_);
if (v_isSharedCheck_1310_ == 0)
{
lean_object* v_unused_1311_; 
v_unused_1311_ = lean_ctor_get(v_fst_1244_, 1);
lean_dec(v_unused_1311_);
v___x_1254_ = v_fst_1244_;
v_isShared_1255_ = v_isSharedCheck_1310_;
goto v_resetjp_1253_;
}
else
{
lean_inc(v_fst_1252_);
lean_dec(v_fst_1244_);
v___x_1254_ = lean_box(0);
v_isShared_1255_ = v_isSharedCheck_1310_;
goto v_resetjp_1253_;
}
v_resetjp_1253_:
{
lean_object* v_fst_1256_; lean_object* v___x_1258_; uint8_t v_isShared_1259_; uint8_t v_isSharedCheck_1308_; 
v_fst_1256_ = lean_ctor_get(v_snd_1245_, 0);
v_isSharedCheck_1308_ = !lean_is_exclusive(v_snd_1245_);
if (v_isSharedCheck_1308_ == 0)
{
lean_object* v_unused_1309_; 
v_unused_1309_ = lean_ctor_get(v_snd_1245_, 1);
lean_dec(v_unused_1309_);
v___x_1258_ = v_snd_1245_;
v_isShared_1259_ = v_isSharedCheck_1308_;
goto v_resetjp_1257_;
}
else
{
lean_inc(v_fst_1256_);
lean_dec(v_snd_1245_);
v___x_1258_ = lean_box(0);
v_isShared_1259_ = v_isSharedCheck_1308_;
goto v_resetjp_1257_;
}
v_resetjp_1257_:
{
lean_object* v_fst_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1306_; 
v_fst_1260_ = lean_ctor_get(v_snd_1246_, 0);
v_isSharedCheck_1306_ = !lean_is_exclusive(v_snd_1246_);
if (v_isSharedCheck_1306_ == 0)
{
lean_object* v_unused_1307_; 
v_unused_1307_ = lean_ctor_get(v_snd_1246_, 1);
lean_dec(v_unused_1307_);
v___x_1262_ = v_snd_1246_;
v_isShared_1263_ = v_isSharedCheck_1306_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_fst_1260_);
lean_dec(v_snd_1246_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1306_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v_fst_1264_; lean_object* v_snd_1265_; lean_object* v___x_1267_; uint8_t v_isShared_1268_; uint8_t v_isSharedCheck_1305_; 
v_fst_1264_ = lean_ctor_get(v_snd_1247_, 0);
v_snd_1265_ = lean_ctor_get(v_snd_1247_, 1);
v_isSharedCheck_1305_ = !lean_is_exclusive(v_snd_1247_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1267_ = v_snd_1247_;
v_isShared_1268_ = v_isSharedCheck_1305_;
goto v_resetjp_1266_;
}
else
{
lean_inc(v_snd_1265_);
lean_inc(v_fst_1264_);
lean_dec(v_snd_1247_);
v___x_1267_ = lean_box(0);
v_isShared_1268_ = v_isSharedCheck_1305_;
goto v_resetjp_1266_;
}
v_resetjp_1266_:
{
lean_object* v___y_1270_; 
if (lean_obj_tag(v_snd_1265_) == 0)
{
lean_del_object(v___x_1219_);
v___y_1270_ = v_snd_1226_;
goto v___jp_1269_;
}
else
{
lean_object* v_val_1299_; lean_object* v___x_1300_; lean_object* v___x_1302_; 
v_val_1299_ = lean_ctor_get(v_snd_1265_, 0);
lean_inc(v_val_1299_);
lean_dec_ref_known(v_snd_1265_, 1);
v___x_1300_ = lean_box(0);
if (v_isShared_1220_ == 0)
{
lean_ctor_set_tag(v___x_1219_, 1);
lean_ctor_set(v___x_1219_, 1, v___x_1300_);
lean_ctor_set(v___x_1219_, 0, v_val_1299_);
v___x_1302_ = v___x_1219_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1304_; 
v_reuseFailAlloc_1304_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1304_, 0, v_val_1299_);
lean_ctor_set(v_reuseFailAlloc_1304_, 1, v___x_1300_);
v___x_1302_ = v_reuseFailAlloc_1304_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
lean_object* v___x_1303_; 
v___x_1303_ = l_List_appendTR___redArg(v_snd_1226_, v___x_1302_);
v___y_1270_ = v___x_1303_;
goto v___jp_1269_;
}
}
v___jp_1269_:
{
lean_object* v___x_1271_; lean_object* v___x_1273_; 
v___x_1271_ = lean_box(0);
if (v_isShared_1229_ == 0)
{
lean_ctor_set_tag(v___x_1228_, 1);
lean_ctor_set(v___x_1228_, 1, v___x_1271_);
lean_ctor_set(v___x_1228_, 0, v_fst_1256_);
v___x_1273_ = v___x_1228_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1298_; 
v_reuseFailAlloc_1298_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1298_, 0, v_fst_1256_);
lean_ctor_set(v_reuseFailAlloc_1298_, 1, v___x_1271_);
v___x_1273_ = v_reuseFailAlloc_1298_;
goto v_reusejp_1272_;
}
v_reusejp_1272_:
{
lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1277_; 
v___x_1274_ = l_List_appendTR___redArg(v_fst_1217_, v___x_1273_);
v___x_1275_ = l_List_appendTR___redArg(v_fst_1221_, v_fst_1260_);
if (v_isShared_1224_ == 0)
{
lean_ctor_set_tag(v___x_1223_, 1);
lean_ctor_set(v___x_1223_, 1, v___x_1271_);
lean_ctor_set(v___x_1223_, 0, v_fst_1264_);
v___x_1277_ = v___x_1223_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1297_; 
v_reuseFailAlloc_1297_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1297_, 0, v_fst_1264_);
lean_ctor_set(v_reuseFailAlloc_1297_, 1, v___x_1271_);
v___x_1277_ = v_reuseFailAlloc_1297_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
lean_object* v___x_1278_; lean_object* v___x_1280_; 
v___x_1278_ = l_List_appendTR___redArg(v_fst_1225_, v___x_1277_);
if (v_isShared_1268_ == 0)
{
lean_ctor_set(v___x_1267_, 1, v___y_1270_);
lean_ctor_set(v___x_1267_, 0, v___x_1278_);
v___x_1280_ = v___x_1267_;
goto v_reusejp_1279_;
}
else
{
lean_object* v_reuseFailAlloc_1296_; 
v_reuseFailAlloc_1296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1296_, 0, v___x_1278_);
lean_ctor_set(v_reuseFailAlloc_1296_, 1, v___y_1270_);
v___x_1280_ = v_reuseFailAlloc_1296_;
goto v_reusejp_1279_;
}
v_reusejp_1279_:
{
lean_object* v___x_1282_; 
if (v_isShared_1263_ == 0)
{
lean_ctor_set(v___x_1262_, 1, v___x_1280_);
lean_ctor_set(v___x_1262_, 0, v___x_1275_);
v___x_1282_ = v___x_1262_;
goto v_reusejp_1281_;
}
else
{
lean_object* v_reuseFailAlloc_1295_; 
v_reuseFailAlloc_1295_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1295_, 0, v___x_1275_);
lean_ctor_set(v_reuseFailAlloc_1295_, 1, v___x_1280_);
v___x_1282_ = v_reuseFailAlloc_1295_;
goto v_reusejp_1281_;
}
v_reusejp_1281_:
{
lean_object* v___x_1284_; 
if (v_isShared_1259_ == 0)
{
lean_ctor_set(v___x_1258_, 1, v___x_1282_);
lean_ctor_set(v___x_1258_, 0, v___x_1274_);
v___x_1284_ = v___x_1258_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v___x_1274_);
lean_ctor_set(v_reuseFailAlloc_1294_, 1, v___x_1282_);
v___x_1284_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
lean_object* v___x_1286_; 
if (v_isShared_1255_ == 0)
{
lean_ctor_set(v___x_1254_, 1, v___x_1284_);
v___x_1286_ = v___x_1254_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1293_; 
v_reuseFailAlloc_1293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1293_, 0, v_fst_1252_);
lean_ctor_set(v_reuseFailAlloc_1293_, 1, v___x_1284_);
v___x_1286_ = v_reuseFailAlloc_1293_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
lean_object* v___x_1288_; 
if (v_isShared_1251_ == 0)
{
lean_ctor_set(v___x_1250_, 0, v___x_1286_);
v___x_1288_ = v___x_1250_;
goto v_reusejp_1287_;
}
else
{
lean_object* v_reuseFailAlloc_1292_; 
v_reuseFailAlloc_1292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1292_, 0, v___x_1286_);
lean_ctor_set(v_reuseFailAlloc_1292_, 1, v_snd_1248_);
v___x_1288_ = v_reuseFailAlloc_1292_;
goto v_reusejp_1287_;
}
v_reusejp_1287_:
{
lean_object* v___x_1290_; 
if (v_isShared_1243_ == 0)
{
lean_ctor_set(v___x_1242_, 0, v___x_1288_);
v___x_1290_ = v___x_1242_;
goto v_reusejp_1289_;
}
else
{
lean_object* v_reuseFailAlloc_1291_; 
v_reuseFailAlloc_1291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1291_, 0, v___x_1288_);
v___x_1290_ = v_reuseFailAlloc_1291_;
goto v_reusejp_1289_;
}
v_reusejp_1289_:
{
return v___x_1290_;
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
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__1___boxed(lean_object** _args){
lean_object* v___x_1320_ = _args[0];
lean_object* v_inst_1321_ = _args[1];
lean_object* v_inst_1322_ = _args[2];
lean_object* v_inst_1323_ = _args[3];
lean_object* v_inst_1324_ = _args[4];
lean_object* v_inst_1325_ = _args[5];
lean_object* v_inst_1326_ = _args[6];
lean_object* v_inst_1327_ = _args[7];
lean_object* v_fo_1328_ = _args[8];
lean_object* v_algMap_1329_ = _args[9];
lean_object* v_params_1330_ = _args[10];
lean_object* v_whirProof_1331_ = _args[11];
lean_object* v___x_1332_ = _args[12];
lean_object* v_commits_1333_ = _args[13];
lean_object* v_widths_1334_ = _args[14];
lean_object* v_acc_1335_ = _args[15];
lean_object* v_roundIdx_1336_ = _args[16];
lean_object* v___y_1337_ = _args[17];
_start:
{
lean_object* v_res_1338_; 
v_res_1338_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__1(v___x_1320_, v_inst_1321_, v_inst_1322_, v_inst_1323_, v_inst_1324_, v_inst_1325_, v_inst_1326_, v_inst_1327_, v_fo_1328_, v_algMap_1329_, v_params_1330_, v_whirProof_1331_, v___x_1332_, v_commits_1333_, v_widths_1334_, v_acc_1335_, v_roundIdx_1336_, v___y_1337_);
lean_dec_ref(v_params_1330_);
lean_dec(v___x_1320_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2(lean_object* v_fst_1342_, lean_object* v_fst_1343_, lean_object* v_snd_1344_, lean_object* v_fo_1345_, lean_object* v_numWhirRounds_1346_, lean_object* v_kWhir_1347_, lean_object* v_fst_1348_, lean_object* v_finalPoly_1349_, lean_object* v_acc_1350_, lean_object* v_roundIdx_1351_, lean_object* v___y_1352_){
_start:
{
lean_object* v___x_1353_; 
lean_inc(v_roundIdx_1351_);
v___x_1353_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_fst_1342_, v_roundIdx_1351_);
if (lean_obj_tag(v___x_1353_) == 0)
{
lean_object* v___x_1354_; 
lean_dec_ref_known(v___x_1353_, 1);
lean_dec_ref(v___y_1352_);
lean_dec(v_roundIdx_1351_);
lean_dec(v_acc_1350_);
lean_dec(v_finalPoly_1349_);
lean_dec_ref(v_fo_1345_);
v___x_1354_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___closed__0));
return v___x_1354_;
}
else
{
lean_object* v_a_1355_; lean_object* v___x_1356_; 
v_a_1355_ = lean_ctor_get(v___x_1353_, 0);
lean_inc(v_a_1355_);
lean_dec_ref_known(v___x_1353_, 1);
lean_inc(v_roundIdx_1351_);
v___x_1356_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_requireWhirElem___redArg(v_fst_1343_, v_roundIdx_1351_);
if (lean_obj_tag(v___x_1356_) == 0)
{
lean_object* v___x_1357_; 
lean_dec_ref_known(v___x_1356_, 1);
lean_dec(v_a_1355_);
lean_dec_ref(v___y_1352_);
lean_dec(v_roundIdx_1351_);
lean_dec(v_acc_1350_);
lean_dec(v_finalPoly_1349_);
lean_dec_ref(v_fo_1345_);
v___x_1357_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___closed__0));
return v___x_1357_;
}
else
{
lean_object* v_a_1358_; lean_object* v___x_1360_; uint8_t v_isShared_1361_; uint8_t v_isSharedCheck_1368_; 
v_a_1358_ = lean_ctor_get(v___x_1356_, 0);
v_isSharedCheck_1368_ = !lean_is_exclusive(v___x_1356_);
if (v_isSharedCheck_1368_ == 0)
{
v___x_1360_ = v___x_1356_;
v_isShared_1361_ = v_isSharedCheck_1368_;
goto v_resetjp_1359_;
}
else
{
lean_inc(v_a_1358_);
lean_dec(v___x_1356_);
v___x_1360_ = lean_box(0);
v_isShared_1361_ = v_isSharedCheck_1368_;
goto v_resetjp_1359_;
}
v_resetjp_1359_:
{
lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1366_; 
lean_inc(v_roundIdx_1351_);
v___x_1362_ = l_List_get_x3fInternal___redArg(v_snd_1344_, v_roundIdx_1351_);
v___x_1363_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_finalCheckRound___redArg(v_fo_1345_, v_acc_1350_, v_roundIdx_1351_, v_numWhirRounds_1346_, v_kWhir_1347_, v_fst_1348_, v_a_1358_, v_a_1355_, v___x_1362_, v_finalPoly_1349_);
lean_dec(v___x_1362_);
lean_dec(v_a_1355_);
lean_dec(v_roundIdx_1351_);
v___x_1364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1364_, 0, v___x_1363_);
lean_ctor_set(v___x_1364_, 1, v___y_1352_);
if (v_isShared_1361_ == 0)
{
lean_ctor_set(v___x_1360_, 0, v___x_1364_);
v___x_1366_ = v___x_1360_;
goto v_reusejp_1365_;
}
else
{
lean_object* v_reuseFailAlloc_1367_; 
v_reuseFailAlloc_1367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1367_, 0, v___x_1364_);
v___x_1366_ = v_reuseFailAlloc_1367_;
goto v_reusejp_1365_;
}
v_reusejp_1365_:
{
return v___x_1366_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___boxed(lean_object* v_fst_1369_, lean_object* v_fst_1370_, lean_object* v_snd_1371_, lean_object* v_fo_1372_, lean_object* v_numWhirRounds_1373_, lean_object* v_kWhir_1374_, lean_object* v_fst_1375_, lean_object* v_finalPoly_1376_, lean_object* v_acc_1377_, lean_object* v_roundIdx_1378_, lean_object* v___y_1379_){
_start:
{
lean_object* v_res_1380_; 
v_res_1380_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2(v_fst_1369_, v_fst_1370_, v_snd_1371_, v_fo_1372_, v_numWhirRounds_1373_, v_kWhir_1374_, v_fst_1375_, v_finalPoly_1376_, v_acc_1377_, v_roundIdx_1378_, v___y_1379_);
lean_dec(v_fst_1375_);
lean_dec(v_kWhir_1374_);
lean_dec(v_numWhirRounds_1373_);
lean_dec(v_snd_1371_);
lean_dec(v_fst_1370_);
lean_dec(v_fst_1369_);
return v_res_1380_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg(lean_object* v_inst_1393_, lean_object* v_inst_1394_, lean_object* v_inst_1395_, lean_object* v_inst_1396_, lean_object* v_inst_1397_, lean_object* v_inst_1398_, lean_object* v_inst_1399_, lean_object* v_inst_1400_, lean_object* v_fo_1401_, lean_object* v_algMap_1402_, lean_object* v_params_1403_, lean_object* v_stackingOpenings_1404_, lean_object* v_whirProof_1405_, lean_object* v_commits_1406_, lean_object* v_uCube_1407_, lean_object* v_a_1408_){
_start:
{
lean_object* v_numWhirRounds_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; uint8_t v___x_1414_; 
v_numWhirRounds_1411_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_numWhirRounds(v_params_1403_);
v___x_1412_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyWhirRoundM___redArg___closed__20));
v___x_1413_ = lean_unsigned_to_nat(0u);
v___x_1414_ = lean_nat_dec_eq(v_numWhirRounds_1411_, v___x_1413_);
if (v___x_1414_ == 0)
{
lean_object* v_muPowWitness_1415_; lean_object* v_whirSumcheckPolys_1416_; lean_object* v_foldingPowWitnesses_1417_; lean_object* v_queryPhasePowWitnesses_1418_; lean_object* v_finalPoly_1419_; lean_object* v_kWhir_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; uint8_t v___x_1423_; 
v_muPowWitness_1415_ = lean_ctor_get(v_whirProof_1405_, 0);
v_whirSumcheckPolys_1416_ = lean_ctor_get(v_whirProof_1405_, 1);
v_foldingPowWitnesses_1417_ = lean_ctor_get(v_whirProof_1405_, 4);
v_queryPhasePowWitnesses_1418_ = lean_ctor_get(v_whirProof_1405_, 5);
v_finalPoly_1419_ = lean_ctor_get(v_whirProof_1405_, 10);
lean_inc(v_finalPoly_1419_);
v_kWhir_1420_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_kWhir(v_params_1403_);
v___x_1421_ = l_List_lengthTR___redArg(v_whirSumcheckPolys_1416_);
v___x_1422_ = lean_nat_mul(v_kWhir_1420_, v_numWhirRounds_1411_);
v___x_1423_ = lean_nat_dec_eq(v___x_1421_, v___x_1422_);
lean_dec(v___x_1421_);
if (v___x_1423_ == 0)
{
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
goto v___jp_1409_;
}
else
{
if (v___x_1414_ == 0)
{
lean_object* v___x_1424_; uint8_t v___x_1425_; 
v___x_1424_ = l_List_lengthTR___redArg(v_foldingPowWitnesses_1417_);
v___x_1425_ = lean_nat_dec_eq(v___x_1424_, v___x_1422_);
lean_dec(v___x_1424_);
if (v___x_1425_ == 0)
{
lean_object* v___x_1426_; 
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
v___x_1426_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1426_;
}
else
{
lean_object* v___x_1427_; uint8_t v___x_1428_; 
v___x_1427_ = l_List_lengthTR___redArg(v_queryPhasePowWitnesses_1418_);
v___x_1428_ = lean_nat_dec_eq(v___x_1427_, v_numWhirRounds_1411_);
lean_dec(v___x_1427_);
if (v___x_1428_ == 0)
{
lean_object* v___x_1429_; 
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
v___x_1429_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1429_;
}
else
{
lean_object* v_whir_1430_; lean_object* v_logBlowup_1431_; lean_object* v_rounds_1432_; lean_object* v_muPowBits_1433_; lean_object* v___x_1434_; uint8_t v___x_1435_; 
v_whir_1430_ = lean_ctor_get(v_params_1403_, 4);
v_logBlowup_1431_ = lean_ctor_get(v_params_1403_, 3);
v_rounds_1432_ = lean_ctor_get(v_whir_1430_, 1);
v_muPowBits_1433_ = lean_ctor_get(v_whir_1430_, 2);
v___x_1434_ = l_List_lengthTR___redArg(v_rounds_1432_);
v___x_1435_ = lean_nat_dec_eq(v___x_1434_, v_numWhirRounds_1411_);
lean_dec(v___x_1434_);
if (v___x_1435_ == 0)
{
lean_object* v___x_1436_; 
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
v___x_1436_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1436_;
}
else
{
lean_object* v___x_1437_; lean_object* v___x_1438_; uint8_t v___x_1439_; 
v___x_1437_ = l_List_lengthTR___redArg(v_commits_1406_);
v___x_1438_ = l_List_lengthTR___redArg(v_stackingOpenings_1404_);
v___x_1439_ = lean_nat_dec_eq(v___x_1437_, v___x_1438_);
lean_dec(v___x_1438_);
lean_dec(v___x_1437_);
if (v___x_1439_ == 0)
{
lean_object* v___x_1440_; 
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
v___x_1440_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1440_;
}
else
{
lean_object* v___x_1441_; 
lean_inc(v_muPowWitness_1415_);
lean_inc_ref(v_inst_1400_);
lean_inc_ref(v_inst_1399_);
v___x_1441_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observePowWitness___redArg(v_inst_1399_, v_inst_1400_, v_muPowBits_1433_, v_muPowWitness_1415_, v_a_1408_);
if (lean_obj_tag(v___x_1441_) == 0)
{
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
return v___x_1441_;
}
else
{
lean_object* v_a_1442_; lean_object* v_snd_1443_; lean_object* v___x_1444_; lean_object* v_a_1445_; lean_object* v_toRingOps_1446_; lean_object* v_toSemiringOps_1447_; lean_object* v_fst_1448_; lean_object* v_snd_1449_; lean_object* v___x_1451_; uint8_t v_isShared_1452_; uint8_t v_isSharedCheck_1552_; 
v_a_1442_ = lean_ctor_get(v___x_1441_, 0);
lean_inc(v_a_1442_);
lean_dec_ref_known(v___x_1441_, 1);
v_snd_1443_ = lean_ctor_get(v_a_1442_, 1);
lean_inc(v_snd_1443_);
lean_dec(v_a_1442_);
lean_inc_ref(v_inst_1397_);
lean_inc_ref(v_inst_1399_);
v___x_1444_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_1399_, v_inst_1397_, v_snd_1443_);
v_a_1445_ = lean_ctor_get(v___x_1444_, 0);
lean_inc(v_a_1445_);
lean_dec_ref(v___x_1444_);
v_toRingOps_1446_ = lean_ctor_get(v_fo_1401_, 0);
lean_inc_ref(v_toRingOps_1446_);
v_toSemiringOps_1447_ = lean_ctor_get(v_toRingOps_1446_, 0);
v_fst_1448_ = lean_ctor_get(v_a_1445_, 0);
v_snd_1449_ = lean_ctor_get(v_a_1445_, 1);
v_isSharedCheck_1552_ = !lean_is_exclusive(v_a_1445_);
if (v_isSharedCheck_1552_ == 0)
{
v___x_1451_ = v_a_1445_;
v_isShared_1452_ = v_isSharedCheck_1552_;
goto v_resetjp_1450_;
}
else
{
lean_inc(v_snd_1449_);
lean_inc(v_fst_1448_);
lean_dec(v_a_1445_);
v___x_1451_ = lean_box(0);
v_isShared_1452_ = v_isSharedCheck_1552_;
goto v_resetjp_1450_;
}
v_resetjp_1450_:
{
lean_object* v_zero_1453_; lean_object* v_add_1454_; lean_object* v_mul_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v_widths_1458_; lean_object* v___f_1459_; lean_object* v___f_1460_; lean_object* v_logStackedHeight_1461_; lean_object* v_totalWidth_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___f_1470_; lean_object* v___x_1471_; lean_object* v___x_1473_; 
v_zero_1453_ = lean_ctor_get(v_toSemiringOps_1447_, 0);
v_add_1454_ = lean_ctor_get(v_toSemiringOps_1447_, 3);
v_mul_1455_ = lean_ctor_get(v_toSemiringOps_1447_, 4);
lean_inc_n(v_mul_1455_, 2);
v___x_1456_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__1));
v___x_1457_ = lean_box(0);
lean_inc(v_stackingOpenings_1404_);
v_widths_1458_ = l_List_mapTR_loop___redArg(v___x_1456_, v_stackingOpenings_1404_, v___x_1457_);
v___f_1459_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__2));
lean_inc(v_add_1454_);
v___f_1460_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1460_, 0, v_mul_1455_);
lean_closure_set(v___f_1460_, 1, v_add_1454_);
v_logStackedHeight_1461_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_logStackedHeight(v_params_1403_);
lean_inc(v_widths_1458_);
v_totalWidth_1462_ = l_List_foldl___redArg(v___f_1459_, v___x_1413_, v_widths_1458_);
lean_inc_ref_n(v_fo_1401_, 2);
v___x_1463_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_1401_, v_fst_1448_, v_totalWidth_1462_);
v___x_1464_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__3));
v___x_1465_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_accumulateInitialOpenedRows___redArg___closed__0));
v___x_1466_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go___redArg(v___x_1464_, v_stackingOpenings_1404_, v___x_1465_);
lean_inc(v___x_1463_);
v___x_1467_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1466_, v___x_1463_);
lean_inc(v_zero_1453_);
v___x_1468_ = l_List_foldl___redArg(v___f_1460_, v_zero_1453_, v___x_1467_);
v___x_1469_ = lean_nat_add(v_logStackedHeight_1461_, v_logBlowup_1431_);
lean_dec(v_logStackedHeight_1461_);
lean_inc_ref(v_params_1403_);
v___f_1470_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__1___boxed), 18, 15);
lean_closure_set(v___f_1470_, 0, v___x_1469_);
lean_closure_set(v___f_1470_, 1, v_inst_1395_);
lean_closure_set(v___f_1470_, 2, v_inst_1396_);
lean_closure_set(v___f_1470_, 3, v_inst_1397_);
lean_closure_set(v___f_1470_, 4, v_inst_1394_);
lean_closure_set(v___f_1470_, 5, v_inst_1398_);
lean_closure_set(v___f_1470_, 6, v_inst_1399_);
lean_closure_set(v___f_1470_, 7, v_inst_1400_);
lean_closure_set(v___f_1470_, 8, v_fo_1401_);
lean_closure_set(v___f_1470_, 9, v_algMap_1402_);
lean_closure_set(v___f_1470_, 10, v_params_1403_);
lean_closure_set(v___f_1470_, 11, v_whirProof_1405_);
lean_closure_set(v___f_1470_, 12, v___x_1463_);
lean_closure_set(v___f_1470_, 13, v_commits_1406_);
lean_closure_set(v___f_1470_, 14, v_widths_1458_);
v___x_1471_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__5));
if (v_isShared_1452_ == 0)
{
lean_ctor_set(v___x_1451_, 1, v___x_1471_);
lean_ctor_set(v___x_1451_, 0, v___x_1468_);
v___x_1473_ = v___x_1451_;
goto v_reusejp_1472_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v___x_1468_);
lean_ctor_set(v_reuseFailAlloc_1551_, 1, v___x_1471_);
v___x_1473_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1472_;
}
v_reusejp_1472_:
{
lean_object* v___x_1474_; lean_object* v___x_9949__overap_1475_; lean_object* v___x_1476_; 
lean_inc(v_numWhirRounds_1411_);
v___x_1474_ = l_List_range(v_numWhirRounds_1411_);
lean_inc(v___x_1474_);
v___x_9949__overap_1475_ = l_List_foldlM___redArg(v___x_1412_, v___f_1470_, v___x_1473_, v___x_1474_);
v___x_1476_ = lean_apply_1(v___x_9949__overap_1475_, v_snd_1449_);
if (lean_obj_tag(v___x_1476_) == 0)
{
lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1484_; 
lean_dec(v___x_1474_);
lean_dec(v_mul_1455_);
lean_dec_ref(v_toRingOps_1446_);
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec(v_uCube_1407_);
lean_dec_ref(v_params_1403_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1393_);
v_a_1477_ = lean_ctor_get(v___x_1476_, 0);
v_isSharedCheck_1484_ = !lean_is_exclusive(v___x_1476_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1479_ = v___x_1476_;
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1476_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v___x_1482_; 
if (v_isShared_1480_ == 0)
{
v___x_1482_ = v___x_1479_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v_a_1477_);
v___x_1482_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1481_;
}
v_reusejp_1481_:
{
return v___x_1482_;
}
}
}
else
{
lean_object* v_a_1485_; lean_object* v_fst_1486_; lean_object* v_snd_1487_; lean_object* v_snd_1488_; lean_object* v_snd_1489_; lean_object* v_snd_1490_; lean_object* v_fst_1491_; lean_object* v_fst_1492_; lean_object* v_fst_1493_; lean_object* v_fst_1494_; lean_object* v_snd_1495_; lean_object* v_logFinalPolyLen_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; uint8_t v___x_1500_; 
v_a_1485_ = lean_ctor_get(v___x_1476_, 0);
lean_inc(v_a_1485_);
lean_dec_ref_known(v___x_1476_, 1);
v_fst_1486_ = lean_ctor_get(v_a_1485_, 0);
lean_inc(v_fst_1486_);
v_snd_1487_ = lean_ctor_get(v_fst_1486_, 1);
lean_inc(v_snd_1487_);
v_snd_1488_ = lean_ctor_get(v_snd_1487_, 1);
lean_inc(v_snd_1488_);
v_snd_1489_ = lean_ctor_get(v_snd_1488_, 1);
lean_inc(v_snd_1489_);
v_snd_1490_ = lean_ctor_get(v_a_1485_, 1);
lean_inc(v_snd_1490_);
lean_dec(v_a_1485_);
v_fst_1491_ = lean_ctor_get(v_fst_1486_, 0);
lean_inc(v_fst_1491_);
lean_dec(v_fst_1486_);
v_fst_1492_ = lean_ctor_get(v_snd_1487_, 0);
lean_inc(v_fst_1492_);
lean_dec(v_snd_1487_);
v_fst_1493_ = lean_ctor_get(v_snd_1488_, 0);
lean_inc(v_fst_1493_);
lean_dec(v_snd_1488_);
v_fst_1494_ = lean_ctor_get(v_snd_1489_, 0);
lean_inc(v_fst_1494_);
v_snd_1495_ = lean_ctor_get(v_snd_1489_, 1);
lean_inc(v_snd_1495_);
lean_dec(v_snd_1489_);
v_logFinalPolyLen_1496_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_logFinalPolyLen(v_params_1403_);
lean_dec_ref(v_params_1403_);
v___x_1497_ = l_List_lengthTR___redArg(v_finalPoly_1419_);
v___x_1498_ = lean_unsigned_to_nat(2u);
v___x_1499_ = lean_nat_pow(v___x_1498_, v_logFinalPolyLen_1496_);
lean_dec(v_logFinalPolyLen_1496_);
v___x_1500_ = lean_nat_dec_eq(v___x_1497_, v___x_1499_);
lean_dec(v___x_1499_);
lean_dec(v___x_1497_);
if (v___x_1500_ == 0)
{
lean_object* v___x_1501_; 
lean_dec(v_snd_1495_);
lean_dec(v_fst_1494_);
lean_dec(v_fst_1493_);
lean_dec(v_fst_1492_);
lean_dec(v_fst_1491_);
lean_dec(v_snd_1490_);
lean_dec(v___x_1474_);
lean_dec(v_mul_1455_);
lean_dec_ref(v_toRingOps_1446_);
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec(v_uCube_1407_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1393_);
v___x_1501_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1501_;
}
else
{
lean_object* v___x_1502_; uint8_t v___x_1503_; 
v___x_1502_ = l_List_lengthTR___redArg(v_fst_1493_);
v___x_1503_ = lean_nat_dec_eq(v___x_1502_, v___x_1422_);
lean_dec(v___x_1502_);
if (v___x_1503_ == 0)
{
lean_object* v___x_1504_; 
lean_dec(v_snd_1495_);
lean_dec(v_fst_1494_);
lean_dec(v_fst_1493_);
lean_dec(v_fst_1492_);
lean_dec(v_fst_1491_);
lean_dec(v_snd_1490_);
lean_dec(v___x_1474_);
lean_dec(v_mul_1455_);
lean_dec_ref(v_toRingOps_1446_);
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec(v_uCube_1407_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1393_);
v___x_1504_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1504_;
}
else
{
lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; uint8_t v___x_1508_; 
v___x_1505_ = l_List_lengthTR___redArg(v_snd_1495_);
v___x_1506_ = lean_unsigned_to_nat(1u);
v___x_1507_ = lean_nat_add(v___x_1505_, v___x_1506_);
lean_dec(v___x_1505_);
v___x_1508_ = lean_nat_dec_eq(v___x_1507_, v_numWhirRounds_1411_);
lean_dec(v___x_1507_);
if (v___x_1508_ == 0)
{
lean_object* v___x_1509_; 
lean_dec(v_snd_1495_);
lean_dec(v_fst_1494_);
lean_dec(v_fst_1493_);
lean_dec(v_fst_1492_);
lean_dec(v_fst_1491_);
lean_dec(v_snd_1490_);
lean_dec(v___x_1474_);
lean_dec(v_mul_1455_);
lean_dec_ref(v_toRingOps_1446_);
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec(v_uCube_1407_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1393_);
v___x_1509_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1509_;
}
else
{
lean_object* v___x_1510_; uint8_t v___x_1511_; 
v___x_1510_ = l_List_lengthTR___redArg(v_fst_1494_);
v___x_1511_ = lean_nat_dec_eq(v___x_1510_, v_numWhirRounds_1411_);
lean_dec(v___x_1510_);
if (v___x_1511_ == 0)
{
lean_object* v___x_1512_; 
lean_dec(v_snd_1495_);
lean_dec(v_fst_1494_);
lean_dec(v_fst_1493_);
lean_dec(v_fst_1492_);
lean_dec(v_fst_1491_);
lean_dec(v_snd_1490_);
lean_dec(v___x_1474_);
lean_dec(v_mul_1455_);
lean_dec_ref(v_toRingOps_1446_);
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec(v_uCube_1407_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1393_);
v___x_1512_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1512_;
}
else
{
lean_object* v___f_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_10193__overap_1520_; lean_object* v___x_1521_; 
lean_inc(v_finalPoly_1419_);
lean_inc_n(v_fst_1493_, 2);
v___f_1513_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___lam__2___boxed), 11, 8);
lean_closure_set(v___f_1513_, 0, v_fst_1492_);
lean_closure_set(v___f_1513_, 1, v_fst_1494_);
lean_closure_set(v___f_1513_, 2, v_snd_1495_);
lean_closure_set(v___f_1513_, 3, v_fo_1401_);
lean_closure_set(v___f_1513_, 4, v_numWhirRounds_1411_);
lean_closure_set(v___f_1513_, 5, v_kWhir_1420_);
lean_closure_set(v___f_1513_, 6, v_fst_1493_);
lean_closure_set(v___f_1513_, 7, v_finalPoly_1419_);
lean_inc_n(v___x_1422_, 2);
lean_inc(v_uCube_1407_);
v___x_1514_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_uCube_1407_, v_uCube_1407_, v___x_1422_, v___x_1465_);
v___x_1515_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_fst_1493_, v_fst_1493_, v___x_1422_, v___x_1465_);
lean_dec(v_fst_1493_);
lean_inc_ref(v_toRingOps_1446_);
v___x_1516_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle___redArg(v_toRingOps_1446_, v___x_1514_, v___x_1515_);
v___x_1517_ = l_List_drop___redArg(v___x_1422_, v_uCube_1407_);
lean_dec(v_uCube_1407_);
v___x_1518_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint___redArg(v_toRingOps_1446_, v_finalPoly_1419_, v___x_1517_);
v___x_1519_ = lean_apply_2(v_mul_1455_, v___x_1516_, v___x_1518_);
v___x_10193__overap_1520_ = l_List_foldlM___redArg(v___x_1412_, v___f_1513_, v___x_1519_, v___x_1474_);
v___x_1521_ = lean_apply_1(v___x_10193__overap_1520_, v_snd_1490_);
if (lean_obj_tag(v___x_1521_) == 0)
{
lean_object* v_a_1522_; lean_object* v___x_1524_; uint8_t v_isShared_1525_; uint8_t v_isSharedCheck_1529_; 
lean_dec(v_fst_1491_);
lean_dec_ref(v_inst_1393_);
v_a_1522_ = lean_ctor_get(v___x_1521_, 0);
v_isSharedCheck_1529_ = !lean_is_exclusive(v___x_1521_);
if (v_isSharedCheck_1529_ == 0)
{
v___x_1524_ = v___x_1521_;
v_isShared_1525_ = v_isSharedCheck_1529_;
goto v_resetjp_1523_;
}
else
{
lean_inc(v_a_1522_);
lean_dec(v___x_1521_);
v___x_1524_ = lean_box(0);
v_isShared_1525_ = v_isSharedCheck_1529_;
goto v_resetjp_1523_;
}
v_resetjp_1523_:
{
lean_object* v___x_1527_; 
if (v_isShared_1525_ == 0)
{
v___x_1527_ = v___x_1524_;
goto v_reusejp_1526_;
}
else
{
lean_object* v_reuseFailAlloc_1528_; 
v_reuseFailAlloc_1528_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1528_, 0, v_a_1522_);
v___x_1527_ = v_reuseFailAlloc_1528_;
goto v_reusejp_1526_;
}
v_reusejp_1526_:
{
return v___x_1527_;
}
}
}
else
{
lean_object* v_a_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1550_; 
v_a_1530_ = lean_ctor_get(v___x_1521_, 0);
v_isSharedCheck_1550_ = !lean_is_exclusive(v___x_1521_);
if (v_isSharedCheck_1550_ == 0)
{
v___x_1532_ = v___x_1521_;
v_isShared_1533_ = v_isSharedCheck_1550_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_a_1530_);
lean_dec(v___x_1521_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1550_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v_fst_1534_; lean_object* v_snd_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1549_; 
v_fst_1534_ = lean_ctor_get(v_a_1530_, 0);
v_snd_1535_ = lean_ctor_get(v_a_1530_, 1);
v_isSharedCheck_1549_ = !lean_is_exclusive(v_a_1530_);
if (v_isSharedCheck_1549_ == 0)
{
v___x_1537_ = v_a_1530_;
v_isShared_1538_ = v_isSharedCheck_1549_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_snd_1535_);
lean_inc(v_fst_1534_);
lean_dec(v_a_1530_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1549_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1539_; uint8_t v___x_1540_; 
v___x_1539_ = lean_apply_2(v_inst_1393_, v_fst_1534_, v_fst_1491_);
v___x_1540_ = lean_unbox(v___x_1539_);
if (v___x_1540_ == 0)
{
lean_object* v___x_1541_; 
lean_del_object(v___x_1537_);
lean_dec(v_snd_1535_);
lean_del_object(v___x_1532_);
v___x_1541_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1541_;
}
else
{
lean_object* v___x_1542_; lean_object* v___x_1544_; 
v___x_1542_ = lean_box(0);
if (v_isShared_1538_ == 0)
{
lean_ctor_set(v___x_1537_, 0, v___x_1542_);
v___x_1544_ = v___x_1537_;
goto v_reusejp_1543_;
}
else
{
lean_object* v_reuseFailAlloc_1548_; 
v_reuseFailAlloc_1548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1548_, 0, v___x_1542_);
lean_ctor_set(v_reuseFailAlloc_1548_, 1, v_snd_1535_);
v___x_1544_ = v_reuseFailAlloc_1548_;
goto v_reusejp_1543_;
}
v_reusejp_1543_:
{
lean_object* v___x_1546_; 
if (v_isShared_1533_ == 0)
{
lean_ctor_set(v___x_1532_, 0, v___x_1544_);
v___x_1546_ = v___x_1532_;
goto v_reusejp_1545_;
}
else
{
lean_object* v_reuseFailAlloc_1547_; 
v_reuseFailAlloc_1547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1547_, 0, v___x_1544_);
v___x_1546_ = v_reuseFailAlloc_1547_;
goto v_reusejp_1545_;
}
v_reusejp_1545_:
{
return v___x_1546_;
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
}
else
{
lean_dec(v___x_1422_);
lean_dec(v_kWhir_1420_);
lean_dec(v_finalPoly_1419_);
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
goto v___jp_1409_;
}
}
}
else
{
lean_object* v___x_1553_; 
lean_dec(v_numWhirRounds_1411_);
lean_dec_ref(v_a_1408_);
lean_dec(v_uCube_1407_);
lean_dec(v_commits_1406_);
lean_dec_ref(v_whirProof_1405_);
lean_dec(v_stackingOpenings_1404_);
lean_dec_ref(v_params_1403_);
lean_dec(v_algMap_1402_);
lean_dec_ref(v_fo_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v_inst_1399_);
lean_dec_ref(v_inst_1398_);
lean_dec_ref(v_inst_1397_);
lean_dec_ref(v_inst_1396_);
lean_dec_ref(v_inst_1395_);
lean_dec(v_inst_1394_);
lean_dec_ref(v_inst_1393_);
v___x_1553_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1553_;
}
v___jp_1409_:
{
lean_object* v___x_1410_; 
v___x_1410_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg___closed__0));
return v___x_1410_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM(lean_object* v_F_1554_, lean_object* v_EF_1555_, lean_object* v_Digest_1556_, lean_object* v_inst_1557_, lean_object* v_inst_1558_, lean_object* v_inst_1559_, lean_object* v_inst_1560_, lean_object* v_inst_1561_, lean_object* v_inst_1562_, lean_object* v_inst_1563_, lean_object* v_inst_1564_, lean_object* v_fo_1565_, lean_object* v_algMap_1566_, lean_object* v_params_1567_, lean_object* v_stackingOpenings_1568_, lean_object* v_whirProof_1569_, lean_object* v_commits_1570_, lean_object* v_uCube_1571_, lean_object* v_a_1572_){
_start:
{
lean_object* v___x_1573_; 
v___x_1573_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg(v_inst_1557_, v_inst_1558_, v_inst_1559_, v_inst_1560_, v_inst_1561_, v_inst_1562_, v_inst_1563_, v_inst_1564_, v_fo_1565_, v_algMap_1566_, v_params_1567_, v_stackingOpenings_1568_, v_whirProof_1569_, v_commits_1570_, v_uCube_1571_, v_a_1572_);
return v___x_1573_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___boxed(lean_object** _args){
lean_object* v_F_1574_ = _args[0];
lean_object* v_EF_1575_ = _args[1];
lean_object* v_Digest_1576_ = _args[2];
lean_object* v_inst_1577_ = _args[3];
lean_object* v_inst_1578_ = _args[4];
lean_object* v_inst_1579_ = _args[5];
lean_object* v_inst_1580_ = _args[6];
lean_object* v_inst_1581_ = _args[7];
lean_object* v_inst_1582_ = _args[8];
lean_object* v_inst_1583_ = _args[9];
lean_object* v_inst_1584_ = _args[10];
lean_object* v_fo_1585_ = _args[11];
lean_object* v_algMap_1586_ = _args[12];
lean_object* v_params_1587_ = _args[13];
lean_object* v_stackingOpenings_1588_ = _args[14];
lean_object* v_whirProof_1589_ = _args[15];
lean_object* v_commits_1590_ = _args[16];
lean_object* v_uCube_1591_ = _args[17];
lean_object* v_a_1592_ = _args[18];
_start:
{
lean_object* v_res_1593_; 
v_res_1593_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM(v_F_1574_, v_EF_1575_, v_Digest_1576_, v_inst_1577_, v_inst_1578_, v_inst_1579_, v_inst_1580_, v_inst_1581_, v_inst_1582_, v_inst_1583_, v_inst_1584_, v_fo_1585_, v_algMap_1586_, v_params_1587_, v_stackingOpenings_1588_, v_whirProof_1589_, v_commits_1590_, v_uCube_1591_, v_a_1592_);
return v_res_1593_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___redArg(lean_object* v_inst_1594_, lean_object* v_inst_1595_, lean_object* v_inst_1596_, lean_object* v_inst_1597_, lean_object* v_inst_1598_, lean_object* v_inst_1599_, lean_object* v_inst_1600_, lean_object* v_fo_1601_, lean_object* v_algMap_1602_, lean_object* v_transcript_1603_, lean_object* v_config_1604_, lean_object* v_whirProof_1605_, lean_object* v_stackingOpenings_1606_, lean_object* v_commits_1607_, lean_object* v_uCube_1608_){
_start:
{
lean_object* v_params_1609_; lean_object* v_hashSlice_1610_; lean_object* v_compress_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; 
v_params_1609_ = lean_ctor_get(v_config_1604_, 0);
lean_inc_ref(v_params_1609_);
v_hashSlice_1610_ = lean_ctor_get(v_config_1604_, 1);
lean_inc(v_hashSlice_1610_);
v_compress_1611_ = lean_ctor_get(v_config_1604_, 2);
lean_inc(v_compress_1611_);
lean_dec_ref(v_config_1604_);
v___x_1612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1612_, 0, v_hashSlice_1610_);
lean_ctor_set(v___x_1612_, 1, v_compress_1611_);
v___x_1613_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verifyM___redArg(v_inst_1594_, v_inst_1595_, v___x_1612_, v_inst_1596_, v_inst_1597_, v_inst_1598_, v_inst_1599_, v_inst_1600_, v_fo_1601_, v_algMap_1602_, v_params_1609_, v_stackingOpenings_1606_, v_whirProof_1605_, v_commits_1607_, v_uCube_1608_, v_transcript_1603_);
if (lean_obj_tag(v___x_1613_) == 0)
{
lean_object* v_a_1614_; lean_object* v___x_1616_; uint8_t v_isShared_1617_; uint8_t v_isSharedCheck_1621_; 
v_a_1614_ = lean_ctor_get(v___x_1613_, 0);
v_isSharedCheck_1621_ = !lean_is_exclusive(v___x_1613_);
if (v_isSharedCheck_1621_ == 0)
{
v___x_1616_ = v___x_1613_;
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
else
{
lean_inc(v_a_1614_);
lean_dec(v___x_1613_);
v___x_1616_ = lean_box(0);
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
v_resetjp_1615_:
{
lean_object* v___x_1619_; 
if (v_isShared_1617_ == 0)
{
v___x_1619_ = v___x_1616_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v_a_1614_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
return v___x_1619_;
}
}
}
else
{
lean_object* v_a_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1630_; 
v_a_1622_ = lean_ctor_get(v___x_1613_, 0);
v_isSharedCheck_1630_ = !lean_is_exclusive(v___x_1613_);
if (v_isSharedCheck_1630_ == 0)
{
v___x_1624_ = v___x_1613_;
v_isShared_1625_ = v_isSharedCheck_1630_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_a_1622_);
lean_dec(v___x_1613_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1630_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v_snd_1626_; lean_object* v___x_1628_; 
v_snd_1626_ = lean_ctor_get(v_a_1622_, 1);
lean_inc(v_snd_1626_);
lean_dec(v_a_1622_);
if (v_isShared_1625_ == 0)
{
lean_ctor_set(v___x_1624_, 0, v_snd_1626_);
v___x_1628_ = v___x_1624_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v_snd_1626_);
v___x_1628_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
return v___x_1628_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify(lean_object* v_F_1631_, lean_object* v_EF_1632_, lean_object* v_Digest_1633_, lean_object* v_inst_1634_, lean_object* v_inst_1635_, lean_object* v_inst_1636_, lean_object* v_inst_1637_, lean_object* v_inst_1638_, lean_object* v_inst_1639_, lean_object* v_inst_1640_, lean_object* v_fo_1641_, lean_object* v_algMap_1642_, lean_object* v_transcript_1643_, lean_object* v_config_1644_, lean_object* v_whirProof_1645_, lean_object* v_stackingOpenings_1646_, lean_object* v_commits_1647_, lean_object* v_uCube_1648_){
_start:
{
lean_object* v___x_1649_; 
v___x_1649_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___redArg(v_inst_1634_, v_inst_1635_, v_inst_1636_, v_inst_1637_, v_inst_1638_, v_inst_1639_, v_inst_1640_, v_fo_1641_, v_algMap_1642_, v_transcript_1643_, v_config_1644_, v_whirProof_1645_, v_stackingOpenings_1646_, v_commits_1647_, v_uCube_1648_);
return v___x_1649_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___boxed(lean_object** _args){
lean_object* v_F_1650_ = _args[0];
lean_object* v_EF_1651_ = _args[1];
lean_object* v_Digest_1652_ = _args[2];
lean_object* v_inst_1653_ = _args[3];
lean_object* v_inst_1654_ = _args[4];
lean_object* v_inst_1655_ = _args[5];
lean_object* v_inst_1656_ = _args[6];
lean_object* v_inst_1657_ = _args[7];
lean_object* v_inst_1658_ = _args[8];
lean_object* v_inst_1659_ = _args[9];
lean_object* v_fo_1660_ = _args[10];
lean_object* v_algMap_1661_ = _args[11];
lean_object* v_transcript_1662_ = _args[12];
lean_object* v_config_1663_ = _args[13];
lean_object* v_whirProof_1664_ = _args[14];
lean_object* v_stackingOpenings_1665_ = _args[15];
lean_object* v_commits_1666_ = _args[16];
lean_object* v_uCube_1667_ = _args[17];
_start:
{
lean_object* v_res_1668_; 
v_res_1668_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify(v_F_1650_, v_EF_1651_, v_Digest_1652_, v_inst_1653_, v_inst_1654_, v_inst_1655_, v_inst_1656_, v_inst_1657_, v_inst_1658_, v_inst_1659_, v_fo_1660_, v_algMap_1661_, v_transcript_1662_, v_config_1663_, v_whirProof_1664_, v_stackingOpenings_1665_, v_commits_1666_, v_uCube_1667_);
return v_res_1668_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Core(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Ops(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_Core(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Common(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Ops(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
