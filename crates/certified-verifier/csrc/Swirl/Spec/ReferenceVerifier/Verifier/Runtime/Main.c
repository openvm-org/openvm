// Lean compiler output
// Module: Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Main
// Imports: public import Init public meta import Init public import Fundamentals.Spec.Runtime.Core public import Swirl.Spec.ReferenceVerifier.Ops public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Batch public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Stacking public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Whir
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
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observe(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_pure(lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_swirl_x2dfv_Fundamentals_Runtime_instDecidableEqSystemParams_decEq(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_verify___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_preprocessed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_preprocessed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_main_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_main_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_cache_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_cache_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "Swirl.Protocol.Noninteractive.Verifier.Runtime.ConcreteInitialRootVectorId.main"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 88, .m_capacity = 88, .m_length = 87, .m_data = "Swirl.Protocol.Noninteractive.Verifier.Runtime.ConcreteInitialRootVectorId.preprocessed"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__4_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 81, .m_capacity = 81, .m_length = 80, .m_data = "Swirl.Protocol.Noninteractive.Verifier.Runtime.ConcreteInitialRootVectorId.cache"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__8_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__9_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId___closed__0_value;
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "commitmentId"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__9_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "digest"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__11_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__12;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__13_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__14;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__15;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__3_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__4_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__5_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__5_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__6_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__1_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__7_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__7_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__10_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__4, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__11_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__7, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__12_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__9, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__13_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_map, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__14_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__15_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_pure, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__15_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__16_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__11_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__12_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__17_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_bind, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__18 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__18_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__17_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__18_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__19 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__19_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__20 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(9) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorFromVerifyingKeyAndTraceData___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorFromVerifyingKeyAndTraceData(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootDigests___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootDigests(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootDigests___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootDigests(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorDigests___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorDigests(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_TranscriptEvent_observe, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg(lean_object* v_traceVdata_1_, lean_object* v_lSkip_2_, lean_object* v_a_3_, lean_object* v_a_4_){
_start:
{
if (lean_obj_tag(v_a_3_) == 0)
{
lean_object* v___x_5_; 
lean_dec(v_lSkip_2_);
v___x_5_ = l_List_reverse___redArg(v_a_4_);
return v___x_5_;
}
else
{
lean_object* v_head_6_; lean_object* v_tail_7_; lean_object* v___x_9_; uint8_t v_isShared_10_; uint8_t v_isSharedCheck_25_; 
v_head_6_ = lean_ctor_get(v_a_3_, 0);
v_tail_7_ = lean_ctor_get(v_a_3_, 1);
v_isSharedCheck_25_ = !lean_is_exclusive(v_a_3_);
if (v_isSharedCheck_25_ == 0)
{
v___x_9_ = v_a_3_;
v_isShared_10_ = v_isSharedCheck_25_;
goto v_resetjp_8_;
}
else
{
lean_inc(v_tail_7_);
lean_inc(v_head_6_);
lean_dec(v_a_3_);
v___x_9_ = lean_box(0);
v_isShared_10_ = v_isSharedCheck_25_;
goto v_resetjp_8_;
}
v_resetjp_8_:
{
lean_object* v___y_12_; lean_object* v___x_17_; 
v___x_17_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_1_, v_head_6_);
if (lean_obj_tag(v___x_17_) == 0)
{
lean_object* v___x_18_; lean_object* v___x_19_; 
lean_inc(v_lSkip_2_);
v___x_18_ = lean_nat_to_int(v_lSkip_2_);
v___x_19_ = lean_int_neg(v___x_18_);
lean_dec(v___x_18_);
v___y_12_ = v___x_19_;
goto v___jp_11_;
}
else
{
lean_object* v_val_20_; lean_object* v_logHeight_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v_val_20_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_val_20_);
lean_dec_ref_known(v___x_17_, 1);
v_logHeight_21_ = lean_ctor_get(v_val_20_, 0);
lean_inc(v_logHeight_21_);
lean_dec(v_val_20_);
v___x_22_ = lean_nat_to_int(v_logHeight_21_);
lean_inc(v_lSkip_2_);
v___x_23_ = lean_nat_to_int(v_lSkip_2_);
v___x_24_ = lean_int_sub(v___x_22_, v___x_23_);
lean_dec(v___x_23_);
lean_dec(v___x_22_);
v___y_12_ = v___x_24_;
goto v___jp_11_;
}
v___jp_11_:
{
lean_object* v___x_14_; 
if (v_isShared_10_ == 0)
{
lean_ctor_set(v___x_9_, 1, v_a_4_);
lean_ctor_set(v___x_9_, 0, v___y_12_);
v___x_14_ = v___x_9_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___y_12_);
lean_ctor_set(v_reuseFailAlloc_16_, 1, v_a_4_);
v___x_14_ = v_reuseFailAlloc_16_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
v_a_3_ = v_tail_7_;
v_a_4_ = v___x_14_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg___boxed(lean_object* v_traceVdata_26_, lean_object* v_lSkip_27_, lean_object* v_a_28_, lean_object* v_a_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg(v_traceVdata_26_, v_lSkip_27_, v_a_28_, v_a_29_);
lean_dec(v_traceVdata_26_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg(lean_object* v_lSkip_31_, lean_object* v_traceVdata_32_, lean_object* v_traceIdToAirId_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_34_ = lean_box(0);
v___x_35_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg(v_traceVdata_32_, v_lSkip_31_, v_traceIdToAirId_33_, v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg___boxed(lean_object* v_lSkip_36_, lean_object* v_traceVdata_37_, lean_object* v_traceIdToAirId_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg(v_lSkip_36_, v_traceVdata_37_, v_traceIdToAirId_38_);
lean_dec(v_traceVdata_37_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace(lean_object* v_Digest_40_, lean_object* v_lSkip_41_, lean_object* v_traceVdata_42_, lean_object* v_traceIdToAirId_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg(v_lSkip_41_, v_traceVdata_42_, v_traceIdToAirId_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___boxed(lean_object* v_Digest_45_, lean_object* v_lSkip_46_, lean_object* v_traceVdata_47_, lean_object* v_traceIdToAirId_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace(v_Digest_45_, v_lSkip_46_, v_traceVdata_47_, v_traceIdToAirId_48_);
lean_dec(v_traceVdata_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0(lean_object* v_Digest_50_, lean_object* v_traceVdata_51_, lean_object* v_lSkip_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___redArg(v_traceVdata_51_, v_lSkip_52_, v_a_53_, v_a_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0___boxed(lean_object* v_Digest_56_, lean_object* v_traceVdata_57_, lean_object* v_lSkip_58_, lean_object* v_a_59_, lean_object* v_a_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace_spec__0(v_Digest_56_, v_traceVdata_57_, v_lSkip_58_, v_a_59_, v_a_60_);
lean_dec(v_traceVdata_57_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg(lean_object* v_lSkip_65_, lean_object* v_constraint_66_, lean_object* v_traceVdata_67_, lean_object* v_airId_68_){
_start:
{
lean_object* v_coefficients_69_; lean_object* v___x_70_; 
v_coefficients_69_ = lean_ctor_get(v_constraint_66_, 0);
lean_inc(v_airId_68_);
v___x_70_ = l_List_get_x3fInternal___redArg(v_coefficients_69_, v_airId_68_);
if (lean_obj_tag(v___x_70_) == 0)
{
lean_object* v___x_71_; 
lean_dec(v_airId_68_);
lean_dec(v_lSkip_65_);
v___x_71_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg___closed__0));
return v___x_71_;
}
else
{
lean_object* v_val_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_92_; 
v_val_72_ = lean_ctor_get(v___x_70_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_70_);
if (v_isSharedCheck_92_ == 0)
{
v___x_74_ = v___x_70_;
v_isShared_75_ = v_isSharedCheck_92_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_val_72_);
lean_dec(v___x_70_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_92_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___y_77_; lean_object* v___y_78_; lean_object* v___y_85_; lean_object* v___x_88_; 
v___x_88_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_67_, v_airId_68_);
if (lean_obj_tag(v___x_88_) == 0)
{
lean_object* v___x_89_; 
v___x_89_ = lean_unsigned_to_nat(0u);
v___y_85_ = v___x_89_;
goto v___jp_84_;
}
else
{
lean_object* v_val_90_; lean_object* v_logHeight_91_; 
v_val_90_ = lean_ctor_get(v___x_88_, 0);
lean_inc(v_val_90_);
lean_dec_ref_known(v___x_88_, 1);
v_logHeight_91_ = lean_ctor_get(v_val_90_, 0);
lean_inc(v_logHeight_91_);
lean_dec(v_val_90_);
v___y_85_ = v_logHeight_91_;
goto v___jp_84_;
}
v___jp_76_:
{
lean_object* v_liftedHeight_79_; lean_object* v___x_80_; lean_object* v___x_82_; 
v_liftedHeight_79_ = lean_nat_pow(v___y_77_, v___y_78_);
lean_dec(v___y_78_);
v___x_80_ = lean_nat_mul(v_liftedHeight_79_, v_val_72_);
lean_dec(v_val_72_);
lean_dec(v_liftedHeight_79_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_80_);
v___x_82_ = v___x_74_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v___x_80_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
v___jp_84_:
{
lean_object* v___x_86_; uint8_t v___x_87_; 
v___x_86_ = lean_unsigned_to_nat(2u);
v___x_87_ = lean_nat_dec_le(v___y_85_, v_lSkip_65_);
if (v___x_87_ == 0)
{
lean_dec(v_lSkip_65_);
v___y_77_ = v___x_86_;
v___y_78_ = v___y_85_;
goto v___jp_76_;
}
else
{
lean_dec(v___y_85_);
v___y_77_ = v___x_86_;
v___y_78_ = v_lSkip_65_;
goto v___jp_76_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg___boxed(lean_object* v_lSkip_93_, lean_object* v_constraint_94_, lean_object* v_traceVdata_95_, lean_object* v_airId_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg(v_lSkip_93_, v_constraint_94_, v_traceVdata_95_, v_airId_96_);
lean_dec(v_traceVdata_95_);
lean_dec_ref(v_constraint_94_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM(lean_object* v_Digest_98_, lean_object* v_lSkip_99_, lean_object* v_constraint_100_, lean_object* v_traceVdata_101_, lean_object* v_airId_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg(v_lSkip_99_, v_constraint_100_, v_traceVdata_101_, v_airId_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___boxed(lean_object* v_Digest_104_, lean_object* v_lSkip_105_, lean_object* v_constraint_106_, lean_object* v_traceVdata_107_, lean_object* v_airId_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM(v_Digest_104_, v_lSkip_105_, v_constraint_106_, v_traceVdata_107_, v_airId_108_);
lean_dec(v_traceVdata_107_);
lean_dec_ref(v_constraint_106_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg(lean_object* v_lSkip_110_, lean_object* v_head_111_, lean_object* v_traceVdata_112_, lean_object* v_x_113_, lean_object* v_x_114_){
_start:
{
if (lean_obj_tag(v_x_114_) == 0)
{
lean_object* v___x_115_; 
lean_dec(v_lSkip_110_);
v___x_115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_115_, 0, v_x_113_);
return v___x_115_;
}
else
{
lean_object* v_head_116_; lean_object* v_tail_117_; lean_object* v___x_118_; 
v_head_116_ = lean_ctor_get(v_x_114_, 0);
lean_inc(v_head_116_);
v_tail_117_ = lean_ctor_get(v_x_114_, 1);
lean_inc(v_tail_117_);
lean_dec_ref_known(v_x_114_, 2);
lean_inc(v_lSkip_110_);
v___x_118_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightContributionM___redArg(v_lSkip_110_, v_head_111_, v_traceVdata_112_, v_head_116_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_dec(v_x_113_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_dec(v_tail_117_);
lean_dec(v_lSkip_110_);
return v___x_118_;
}
else
{
lean_object* v_a_119_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
lean_inc(v_a_119_);
lean_dec_ref_known(v___x_118_, 1);
v_x_113_ = v_a_119_;
v_x_114_ = v_tail_117_;
goto _start;
}
}
else
{
lean_object* v_a_121_; lean_object* v___x_122_; 
v_a_121_ = lean_ctor_get(v___x_118_, 0);
lean_inc(v_a_121_);
lean_dec_ref_known(v___x_118_, 1);
v___x_122_ = lean_nat_add(v_x_113_, v_a_121_);
lean_dec(v_a_121_);
lean_dec(v_x_113_);
v_x_113_ = v___x_122_;
v_x_114_ = v_tail_117_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg___boxed(lean_object* v_lSkip_124_, lean_object* v_head_125_, lean_object* v_traceVdata_126_, lean_object* v_x_127_, lean_object* v_x_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg(v_lSkip_124_, v_head_125_, v_traceVdata_126_, v_x_127_, v_x_128_);
lean_dec(v_traceVdata_126_);
lean_dec_ref(v_head_125_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg(lean_object* v_lSkip_136_, lean_object* v_traceVdata_137_, lean_object* v_traceIdToAirId_138_, lean_object* v_a_139_){
_start:
{
if (lean_obj_tag(v_a_139_) == 0)
{
lean_object* v___x_140_; 
lean_dec(v_traceIdToAirId_138_);
lean_dec(v_lSkip_136_);
v___x_140_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__0));
return v___x_140_;
}
else
{
lean_object* v_head_141_; lean_object* v_tail_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v_head_141_ = lean_ctor_get(v_a_139_, 0);
v_tail_142_ = lean_ctor_get(v_a_139_, 1);
v___x_143_ = lean_unsigned_to_nat(0u);
lean_inc(v_traceIdToAirId_138_);
lean_inc(v_lSkip_136_);
v___x_144_ = lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg(v_lSkip_136_, v_head_141_, v_traceVdata_137_, v___x_143_, v_traceIdToAirId_138_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v___x_145_; 
lean_dec_ref_known(v___x_144_, 1);
lean_dec(v_traceIdToAirId_138_);
lean_dec(v_lSkip_136_);
v___x_145_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___closed__1));
return v___x_145_;
}
else
{
lean_object* v_a_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_157_; 
v_a_146_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_157_ == 0)
{
v___x_148_ = v___x_144_;
v_isShared_149_ = v_isSharedCheck_157_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_a_146_);
lean_dec(v___x_144_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_157_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v_threshold_150_; uint8_t v___x_151_; 
v_threshold_150_ = lean_ctor_get(v_head_141_, 1);
v___x_151_ = lean_nat_dec_lt(v_a_146_, v_threshold_150_);
lean_dec(v_a_146_);
if (v___x_151_ == 0)
{
lean_object* v___x_152_; lean_object* v___x_154_; 
lean_dec(v_traceIdToAirId_138_);
lean_dec(v_lSkip_136_);
v___x_152_ = lean_box(v___x_151_);
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 0, v___x_152_);
v___x_154_ = v___x_148_;
goto v_reusejp_153_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v___x_152_);
v___x_154_ = v_reuseFailAlloc_155_;
goto v_reusejp_153_;
}
v_reusejp_153_:
{
return v___x_154_;
}
}
else
{
lean_del_object(v___x_148_);
v_a_139_ = v_tail_142_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg___boxed(lean_object* v_lSkip_158_, lean_object* v_traceVdata_159_, lean_object* v_traceIdToAirId_160_, lean_object* v_a_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg(v_lSkip_158_, v_traceVdata_159_, v_traceIdToAirId_160_, v_a_161_);
lean_dec(v_a_161_);
lean_dec(v_traceVdata_159_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk(lean_object* v_Digest_163_, lean_object* v_lSkip_164_, lean_object* v_traceVdata_165_, lean_object* v_traceIdToAirId_166_, lean_object* v_a_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg(v_lSkip_164_, v_traceVdata_165_, v_traceIdToAirId_166_, v_a_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___boxed(lean_object* v_Digest_169_, lean_object* v_lSkip_170_, lean_object* v_traceVdata_171_, lean_object* v_traceIdToAirId_172_, lean_object* v_a_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk(v_Digest_169_, v_lSkip_170_, v_traceVdata_171_, v_traceIdToAirId_172_, v_a_173_);
lean_dec(v_a_173_);
lean_dec(v_traceVdata_171_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0(lean_object* v_Digest_175_, lean_object* v_lSkip_176_, lean_object* v_head_177_, lean_object* v_traceVdata_178_, lean_object* v_x_179_, lean_object* v_x_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___redArg(v_lSkip_176_, v_head_177_, v_traceVdata_178_, v_x_179_, v_x_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0___boxed(lean_object* v_Digest_182_, lean_object* v_lSkip_183_, lean_object* v_head_184_, lean_object* v_traceVdata_185_, lean_object* v_x_186_, lean_object* v_x_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk_spec__0(v_Digest_182_, v_lSkip_183_, v_head_184_, v_traceVdata_185_, v_x_186_, v_x_187_);
lean_dec(v_traceVdata_185_);
lean_dec_ref(v_head_184_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg(lean_object* v_lSkip_189_, lean_object* v_vk_190_, lean_object* v_traceVdata_191_, lean_object* v_traceIdToAirId_192_){
_start:
{
lean_object* v_inner_193_; lean_object* v_traceHeightConstraints_194_; lean_object* v___x_195_; 
v_inner_193_ = lean_ctor_get(v_vk_190_, 0);
v_traceHeightConstraints_194_ = lean_ctor_get(v_inner_193_, 2);
v___x_195_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM_constraintsOk___redArg(v_lSkip_189_, v_traceVdata_191_, v_traceIdToAirId_192_, v_traceHeightConstraints_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg___boxed(lean_object* v_lSkip_196_, lean_object* v_vk_197_, lean_object* v_traceVdata_198_, lean_object* v_traceIdToAirId_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg(v_lSkip_196_, v_vk_197_, v_traceVdata_198_, v_traceIdToAirId_199_);
lean_dec(v_traceVdata_198_);
lean_dec_ref(v_vk_197_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM(lean_object* v_F_201_, lean_object* v_Digest_202_, lean_object* v_lSkip_203_, lean_object* v_vk_204_, lean_object* v_traceVdata_205_, lean_object* v_traceIdToAirId_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg(v_lSkip_203_, v_vk_204_, v_traceVdata_205_, v_traceIdToAirId_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___boxed(lean_object* v_F_208_, lean_object* v_Digest_209_, lean_object* v_lSkip_210_, lean_object* v_vk_211_, lean_object* v_traceVdata_212_, lean_object* v_traceIdToAirId_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM(v_F_208_, v_Digest_209_, v_lSkip_210_, v_vk_211_, v_traceVdata_212_, v_traceIdToAirId_213_);
lean_dec(v_traceVdata_212_);
lean_dec_ref(v_vk_211_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg(lean_object* v_vk_215_, lean_object* v_traceVdata_216_, lean_object* v_x_217_, lean_object* v_x_218_){
_start:
{
if (lean_obj_tag(v_x_218_) == 0)
{
return v_x_217_;
}
else
{
lean_object* v_head_219_; lean_object* v_tail_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_258_; 
v_head_219_ = lean_ctor_get(v_x_218_, 0);
v_tail_220_ = lean_ctor_get(v_x_218_, 1);
v_isSharedCheck_258_ = !lean_is_exclusive(v_x_218_);
if (v_isSharedCheck_258_ == 0)
{
v___x_222_ = v_x_218_;
v_isShared_223_ = v_isSharedCheck_258_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_tail_220_);
lean_inc(v_head_219_);
lean_dec(v_x_218_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_258_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___y_225_; uint8_t v___y_226_; lean_object* v___y_227_; uint8_t v___y_237_; lean_object* v___y_238_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___y_247_; 
v___x_244_ = lean_box(0);
lean_inc(v_head_219_);
v___x_245_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_215_, v_head_219_);
if (lean_obj_tag(v___x_245_) == 0)
{
uint8_t v___x_254_; 
v___x_254_ = 0;
v___y_247_ = v___x_254_;
goto v___jp_246_;
}
else
{
lean_object* v_val_255_; lean_object* v_params_256_; uint8_t v_needRot_257_; 
v_val_255_ = lean_ctor_get(v___x_245_, 0);
lean_inc(v_val_255_);
v_params_256_ = lean_ctor_get(v_val_255_, 1);
lean_inc_ref(v_params_256_);
lean_dec(v_val_255_);
v_needRot_257_ = lean_ctor_get_uint8(v_params_256_, sizeof(void*)*2);
lean_dec_ref(v_params_256_);
v___y_247_ = v_needRot_257_;
goto v___jp_246_;
}
v___jp_224_:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_231_; 
v___x_228_ = lean_box(0);
v___x_229_ = lean_box(v___y_226_);
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 1, v___x_228_);
lean_ctor_set(v___x_222_, 0, v___x_229_);
v___x_231_ = v___x_222_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_229_);
lean_ctor_set(v_reuseFailAlloc_235_, 1, v___x_228_);
v___x_231_ = v_reuseFailAlloc_235_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = l_List_replicateTR___redArg(v___y_227_, v___x_231_);
v___x_233_ = l_List_appendTR___redArg(v___y_225_, v___x_232_);
v_x_217_ = v___x_233_;
v_x_218_ = v_tail_220_;
goto _start;
}
}
v___jp_236_:
{
lean_object* v___x_239_; 
v___x_239_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_216_, v_head_219_);
if (lean_obj_tag(v___x_239_) == 0)
{
lean_object* v___x_240_; 
v___x_240_ = lean_unsigned_to_nat(0u);
v___y_225_ = v___y_238_;
v___y_226_ = v___y_237_;
v___y_227_ = v___x_240_;
goto v___jp_224_;
}
else
{
lean_object* v_val_241_; lean_object* v_cachedCommitments_242_; lean_object* v___x_243_; 
v_val_241_ = lean_ctor_get(v___x_239_, 0);
lean_inc(v_val_241_);
lean_dec_ref_known(v___x_239_, 1);
v_cachedCommitments_242_ = lean_ctor_get(v_val_241_, 1);
lean_inc(v_cachedCommitments_242_);
lean_dec(v_val_241_);
v___x_243_ = l_List_lengthTR___redArg(v_cachedCommitments_242_);
lean_dec(v_cachedCommitments_242_);
v___y_225_ = v___y_238_;
v___y_226_ = v___y_237_;
v___y_227_ = v___x_243_;
goto v___jp_224_;
}
}
v___jp_246_:
{
if (lean_obj_tag(v___x_245_) == 0)
{
v___y_237_ = v___y_247_;
v___y_238_ = v_x_217_;
goto v___jp_236_;
}
else
{
lean_object* v_val_248_; lean_object* v_preprocessedData_249_; 
v_val_248_ = lean_ctor_get(v___x_245_, 0);
lean_inc(v_val_248_);
lean_dec_ref_known(v___x_245_, 1);
v_preprocessedData_249_ = lean_ctor_get(v_val_248_, 0);
lean_inc(v_preprocessedData_249_);
lean_dec(v_val_248_);
if (lean_obj_tag(v_preprocessedData_249_) == 0)
{
v___y_237_ = v___y_247_;
v___y_238_ = v_x_217_;
goto v___jp_236_;
}
else
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
lean_dec_ref_known(v_preprocessedData_249_, 1);
v___x_250_ = lean_box(v___y_247_);
v___x_251_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___x_244_);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v___x_244_);
v___x_253_ = l_List_appendTR___redArg(v_x_217_, v___x_252_);
v___y_237_ = v___y_247_;
v___y_238_ = v___x_253_;
goto v___jp_236_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg___boxed(lean_object* v_vk_259_, lean_object* v_traceVdata_260_, lean_object* v_x_261_, lean_object* v_x_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg(v_vk_259_, v_traceVdata_260_, v_x_261_, v_x_262_);
lean_dec(v_traceVdata_260_);
lean_dec_ref(v_vk_259_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg(lean_object* v_vk_264_, lean_object* v_a_265_, lean_object* v_a_266_){
_start:
{
if (lean_obj_tag(v_a_265_) == 0)
{
lean_object* v___x_267_; 
v___x_267_ = l_List_reverse___redArg(v_a_266_);
return v___x_267_;
}
else
{
lean_object* v_head_268_; lean_object* v_tail_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_285_; 
v_head_268_ = lean_ctor_get(v_a_265_, 0);
v_tail_269_ = lean_ctor_get(v_a_265_, 1);
v_isSharedCheck_285_ = !lean_is_exclusive(v_a_265_);
if (v_isSharedCheck_285_ == 0)
{
v___x_271_ = v_a_265_;
v_isShared_272_ = v_isSharedCheck_285_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_tail_269_);
lean_inc(v_head_268_);
lean_dec(v_a_265_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_285_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
uint8_t v___y_274_; lean_object* v___x_280_; 
v___x_280_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_264_, v_head_268_);
if (lean_obj_tag(v___x_280_) == 0)
{
uint8_t v___x_281_; 
v___x_281_ = 0;
v___y_274_ = v___x_281_;
goto v___jp_273_;
}
else
{
lean_object* v_val_282_; lean_object* v_params_283_; uint8_t v_needRot_284_; 
v_val_282_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_val_282_);
lean_dec_ref_known(v___x_280_, 1);
v_params_283_ = lean_ctor_get(v_val_282_, 1);
lean_inc_ref(v_params_283_);
lean_dec(v_val_282_);
v_needRot_284_ = lean_ctor_get_uint8(v_params_283_, sizeof(void*)*2);
lean_dec_ref(v_params_283_);
v___y_274_ = v_needRot_284_;
goto v___jp_273_;
}
v___jp_273_:
{
lean_object* v___x_275_; lean_object* v___x_277_; 
v___x_275_ = lean_box(v___y_274_);
if (v_isShared_272_ == 0)
{
lean_ctor_set(v___x_271_, 1, v_a_266_);
lean_ctor_set(v___x_271_, 0, v___x_275_);
v___x_277_ = v___x_271_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v___x_275_);
lean_ctor_set(v_reuseFailAlloc_279_, 1, v_a_266_);
v___x_277_ = v_reuseFailAlloc_279_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
v_a_265_ = v_tail_269_;
v_a_266_ = v___x_277_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg___boxed(lean_object* v_vk_286_, lean_object* v_a_287_, lean_object* v_a_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg(v_vk_286_, v_a_287_, v_a_288_);
lean_dec_ref(v_vk_286_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg(lean_object* v_vk_290_, lean_object* v_traceVdata_291_, lean_object* v_traceIdToAirId_292_){
_start:
{
lean_object* v___x_293_; lean_object* v_needRotPerTrace_294_; lean_object* v_base_295_; lean_object* v___x_296_; 
v___x_293_ = lean_box(0);
lean_inc(v_traceIdToAirId_292_);
v_needRotPerTrace_294_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg(v_vk_290_, v_traceIdToAirId_292_, v___x_293_);
v_base_295_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_base_295_, 0, v_needRotPerTrace_294_);
lean_ctor_set(v_base_295_, 1, v___x_293_);
v___x_296_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg(v_vk_290_, v_traceVdata_291_, v_base_295_, v_traceIdToAirId_292_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg___boxed(lean_object* v_vk_297_, lean_object* v_traceVdata_298_, lean_object* v_traceIdToAirId_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg(v_vk_297_, v_traceVdata_298_, v_traceIdToAirId_299_);
lean_dec(v_traceVdata_298_);
lean_dec_ref(v_vk_297_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit(lean_object* v_F_301_, lean_object* v_Digest_302_, lean_object* v_vk_303_, lean_object* v_traceVdata_304_, lean_object* v_traceIdToAirId_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg(v_vk_303_, v_traceVdata_304_, v_traceIdToAirId_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___boxed(lean_object* v_F_307_, lean_object* v_Digest_308_, lean_object* v_vk_309_, lean_object* v_traceVdata_310_, lean_object* v_traceIdToAirId_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit(v_F_307_, v_Digest_308_, v_vk_309_, v_traceVdata_310_, v_traceIdToAirId_311_);
lean_dec(v_traceVdata_310_);
lean_dec_ref(v_vk_309_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0(lean_object* v_F_313_, lean_object* v_Digest_314_, lean_object* v_vk_315_, lean_object* v_a_316_, lean_object* v_a_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___redArg(v_vk_315_, v_a_316_, v_a_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0___boxed(lean_object* v_F_319_, lean_object* v_Digest_320_, lean_object* v_vk_321_, lean_object* v_a_322_, lean_object* v_a_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__0(v_F_319_, v_Digest_320_, v_vk_321_, v_a_322_, v_a_323_);
lean_dec_ref(v_vk_321_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1(lean_object* v_F_325_, lean_object* v_Digest_326_, lean_object* v_vk_327_, lean_object* v_traceVdata_328_, lean_object* v_x_329_, lean_object* v_x_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___redArg(v_vk_327_, v_traceVdata_328_, v_x_329_, v_x_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1___boxed(lean_object* v_F_332_, lean_object* v_Digest_333_, lean_object* v_vk_334_, lean_object* v_traceVdata_335_, lean_object* v_x_336_, lean_object* v_x_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit_spec__1(v_F_332_, v_Digest_333_, v_vk_334_, v_traceVdata_335_, v_x_336_, v_x_337_);
lean_dec(v_traceVdata_335_);
lean_dec_ref(v_vk_334_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorIdx(lean_object* v_x_339_){
_start:
{
switch(lean_obj_tag(v_x_339_))
{
case 0:
{
lean_object* v___x_340_; 
v___x_340_ = lean_unsigned_to_nat(0u);
return v___x_340_;
}
case 1:
{
lean_object* v___x_341_; 
v___x_341_ = lean_unsigned_to_nat(1u);
return v___x_341_;
}
default: 
{
lean_object* v___x_342_; 
v___x_342_ = lean_unsigned_to_nat(2u);
return v___x_342_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorIdx___boxed(lean_object* v_x_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorIdx(v_x_343_);
lean_dec(v_x_343_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(lean_object* v_t_345_, lean_object* v_k_346_){
_start:
{
switch(lean_obj_tag(v_t_345_))
{
case 0:
{
lean_object* v_airId_347_; lean_object* v___x_348_; 
v_airId_347_ = lean_ctor_get(v_t_345_, 0);
lean_inc(v_airId_347_);
lean_dec_ref_known(v_t_345_, 1);
v___x_348_ = lean_apply_1(v_k_346_, v_airId_347_);
return v___x_348_;
}
case 1:
{
return v_k_346_;
}
default: 
{
lean_object* v_airId_349_; lean_object* v_cacheIndex_350_; lean_object* v___x_351_; 
v_airId_349_ = lean_ctor_get(v_t_345_, 0);
lean_inc(v_airId_349_);
v_cacheIndex_350_ = lean_ctor_get(v_t_345_, 1);
lean_inc(v_cacheIndex_350_);
lean_dec_ref_known(v_t_345_, 2);
v___x_351_ = lean_apply_2(v_k_346_, v_airId_349_, v_cacheIndex_350_);
return v___x_351_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim(lean_object* v_motive_352_, lean_object* v_ctorIdx_353_, lean_object* v_t_354_, lean_object* v_h_355_, lean_object* v_k_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_354_, v_k_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___boxed(lean_object* v_motive_358_, lean_object* v_ctorIdx_359_, lean_object* v_t_360_, lean_object* v_h_361_, lean_object* v_k_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim(v_motive_358_, v_ctorIdx_359_, v_t_360_, v_h_361_, v_k_362_);
lean_dec(v_ctorIdx_359_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_preprocessed_elim___redArg(lean_object* v_t_364_, lean_object* v_preprocessed_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_364_, v_preprocessed_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_preprocessed_elim(lean_object* v_motive_367_, lean_object* v_t_368_, lean_object* v_h_369_, lean_object* v_preprocessed_370_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_368_, v_preprocessed_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_main_elim___redArg(lean_object* v_t_372_, lean_object* v_main_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_372_, v_main_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_main_elim(lean_object* v_motive_375_, lean_object* v_t_376_, lean_object* v_h_377_, lean_object* v_main_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_376_, v_main_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_cache_elim___redArg(lean_object* v_t_380_, lean_object* v_cache_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_380_, v_cache_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_cache_elim(lean_object* v_motive_383_, lean_object* v_t_384_, lean_object* v_h_385_, lean_object* v_cache_386_){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_ConcreteInitialRootVectorId_ctorElim___redArg(v_t_384_, v_cache_386_);
return v___x_387_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId_decEq(lean_object* v_x_388_, lean_object* v_x_389_){
_start:
{
switch(lean_obj_tag(v_x_388_))
{
case 0:
{
lean_object* v_airId_390_; uint8_t v___x_391_; 
v_airId_390_ = lean_ctor_get(v_x_388_, 0);
v___x_391_ = 0;
if (lean_obj_tag(v_x_389_) == 0)
{
lean_object* v_airId_392_; uint8_t v___x_393_; 
v_airId_392_ = lean_ctor_get(v_x_389_, 0);
v___x_393_ = lean_nat_dec_eq(v_airId_390_, v_airId_392_);
if (v___x_393_ == 0)
{
return v___x_391_;
}
else
{
return v___x_393_;
}
}
else
{
return v___x_391_;
}
}
case 1:
{
if (lean_obj_tag(v_x_389_) == 1)
{
uint8_t v___x_394_; 
v___x_394_ = 1;
return v___x_394_;
}
else
{
uint8_t v___x_395_; 
v___x_395_ = 0;
return v___x_395_;
}
}
default: 
{
lean_object* v_airId_396_; lean_object* v_cacheIndex_397_; uint8_t v___x_398_; 
v_airId_396_ = lean_ctor_get(v_x_388_, 0);
v_cacheIndex_397_ = lean_ctor_get(v_x_388_, 1);
v___x_398_ = 0;
if (lean_obj_tag(v_x_389_) == 2)
{
lean_object* v_airId_399_; lean_object* v_cacheIndex_400_; uint8_t v___x_401_; 
v_airId_399_ = lean_ctor_get(v_x_389_, 0);
v_cacheIndex_400_ = lean_ctor_get(v_x_389_, 1);
v___x_401_ = lean_nat_dec_eq(v_airId_396_, v_airId_399_);
if (v___x_401_ == 0)
{
return v___x_398_;
}
else
{
uint8_t v___x_402_; 
v___x_402_ = lean_nat_dec_eq(v_cacheIndex_397_, v_cacheIndex_400_);
if (v___x_402_ == 0)
{
return v___x_398_;
}
else
{
return v___x_402_;
}
}
}
else
{
return v___x_398_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId_decEq___boxed(lean_object* v_x_403_, lean_object* v_x_404_){
_start:
{
uint8_t v_res_405_; lean_object* v_r_406_; 
v_res_405_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId_decEq(v_x_403_, v_x_404_);
lean_dec(v_x_404_);
lean_dec(v_x_403_);
v_r_406_ = lean_box(v_res_405_);
return v_r_406_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId(lean_object* v_x_407_, lean_object* v_x_408_){
_start:
{
uint8_t v___x_409_; 
v___x_409_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId_decEq(v_x_407_, v_x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId___boxed(lean_object* v_x_410_, lean_object* v_x_411_){
_start:
{
uint8_t v_res_412_; lean_object* v_r_413_; 
v_res_412_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqConcreteInitialRootVectorId(v_x_410_, v_x_411_);
lean_dec(v_x_411_);
lean_dec(v_x_410_);
v_r_413_ = lean_box(v_res_412_);
return v_r_413_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5(void){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_423_ = lean_unsigned_to_nat(2u);
v___x_424_ = lean_nat_to_int(v___x_423_);
return v___x_424_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6(void){
_start:
{
lean_object* v___x_425_; lean_object* v___x_426_; 
v___x_425_ = lean_unsigned_to_nat(1u);
v___x_426_ = lean_nat_to_int(v___x_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr(lean_object* v_x_433_, lean_object* v_prec_434_){
_start:
{
lean_object* v___y_436_; 
switch(lean_obj_tag(v_x_433_))
{
case 0:
{
lean_object* v_airId_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_462_; 
v_airId_442_ = lean_ctor_get(v_x_433_, 0);
v_isSharedCheck_462_ = !lean_is_exclusive(v_x_433_);
if (v_isSharedCheck_462_ == 0)
{
v___x_444_ = v_x_433_;
v_isShared_445_ = v_isSharedCheck_462_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_airId_442_);
lean_dec(v_x_433_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_462_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___y_447_; lean_object* v___x_458_; uint8_t v___x_459_; 
v___x_458_ = lean_unsigned_to_nat(1024u);
v___x_459_ = lean_nat_dec_le(v___x_458_, v_prec_434_);
if (v___x_459_ == 0)
{
lean_object* v___x_460_; 
v___x_460_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5);
v___y_447_ = v___x_460_;
goto v___jp_446_;
}
else
{
lean_object* v___x_461_; 
v___x_461_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6);
v___y_447_ = v___x_461_;
goto v___jp_446_;
}
v___jp_446_:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_451_; 
v___x_448_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__4));
v___x_449_ = l_Nat_reprFast(v_airId_442_);
if (v_isShared_445_ == 0)
{
lean_ctor_set_tag(v___x_444_, 3);
lean_ctor_set(v___x_444_, 0, v___x_449_);
v___x_451_ = v___x_444_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_457_; 
v_reuseFailAlloc_457_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_457_, 0, v___x_449_);
v___x_451_ = v_reuseFailAlloc_457_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
lean_object* v___x_452_; lean_object* v___x_453_; uint8_t v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_452_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_448_);
lean_ctor_set(v___x_452_, 1, v___x_451_);
lean_inc(v___y_447_);
v___x_453_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_453_, 0, v___y_447_);
lean_ctor_set(v___x_453_, 1, v___x_452_);
v___x_454_ = 0;
v___x_455_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_455_, 0, v___x_453_);
lean_ctor_set_uint8(v___x_455_, sizeof(void*)*1, v___x_454_);
v___x_456_ = l_Repr_addAppParen(v___x_455_, v_prec_434_);
return v___x_456_;
}
}
}
}
case 1:
{
lean_object* v___x_463_; uint8_t v___x_464_; 
v___x_463_ = lean_unsigned_to_nat(1024u);
v___x_464_ = lean_nat_dec_le(v___x_463_, v_prec_434_);
if (v___x_464_ == 0)
{
lean_object* v___x_465_; 
v___x_465_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5);
v___y_436_ = v___x_465_;
goto v___jp_435_;
}
else
{
lean_object* v___x_466_; 
v___x_466_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6);
v___y_436_ = v___x_466_;
goto v___jp_435_;
}
}
default: 
{
lean_object* v_airId_467_; lean_object* v_cacheIndex_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_493_; 
v_airId_467_ = lean_ctor_get(v_x_433_, 0);
v_cacheIndex_468_ = lean_ctor_get(v_x_433_, 1);
v_isSharedCheck_493_ = !lean_is_exclusive(v_x_433_);
if (v_isSharedCheck_493_ == 0)
{
v___x_470_ = v_x_433_;
v_isShared_471_ = v_isSharedCheck_493_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_cacheIndex_468_);
lean_inc(v_airId_467_);
lean_dec(v_x_433_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_493_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
lean_object* v___y_473_; lean_object* v___x_489_; uint8_t v___x_490_; 
v___x_489_ = lean_unsigned_to_nat(1024u);
v___x_490_ = lean_nat_dec_le(v___x_489_, v_prec_434_);
if (v___x_490_ == 0)
{
lean_object* v___x_491_; 
v___x_491_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__5);
v___y_473_ = v___x_491_;
goto v___jp_472_;
}
else
{
lean_object* v___x_492_; 
v___x_492_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__6);
v___y_473_ = v___x_492_;
goto v___jp_472_;
}
v___jp_472_:
{
lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_479_; 
v___x_474_ = lean_box(1);
v___x_475_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__9));
v___x_476_ = l_Nat_reprFast(v_airId_467_);
v___x_477_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_477_, 0, v___x_476_);
if (v_isShared_471_ == 0)
{
lean_ctor_set_tag(v___x_470_, 5);
lean_ctor_set(v___x_470_, 1, v___x_477_);
lean_ctor_set(v___x_470_, 0, v___x_475_);
v___x_479_ = v___x_470_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v___x_475_);
lean_ctor_set(v_reuseFailAlloc_488_, 1, v___x_477_);
v___x_479_ = v_reuseFailAlloc_488_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; uint8_t v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_480_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_480_, 0, v___x_479_);
lean_ctor_set(v___x_480_, 1, v___x_474_);
v___x_481_ = l_Nat_reprFast(v_cacheIndex_468_);
v___x_482_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_482_, 0, v___x_481_);
v___x_483_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_483_, 0, v___x_480_);
lean_ctor_set(v___x_483_, 1, v___x_482_);
lean_inc(v___y_473_);
v___x_484_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_484_, 0, v___y_473_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
v___x_485_ = 0;
v___x_486_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_486_, 0, v___x_484_);
lean_ctor_set_uint8(v___x_486_, sizeof(void*)*1, v___x_485_);
v___x_487_ = l_Repr_addAppParen(v___x_486_, v_prec_434_);
return v___x_487_;
}
}
}
}
}
v___jp_435_:
{
lean_object* v___x_437_; lean_object* v___x_438_; uint8_t v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_437_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___closed__1));
lean_inc(v___y_436_);
v___x_438_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_438_, 0, v___y_436_);
lean_ctor_set(v___x_438_, 1, v___x_437_);
v___x_439_ = 0;
v___x_440_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_440_, 0, v___x_438_);
lean_ctor_set_uint8(v___x_440_, sizeof(void*)*1, v___x_439_);
v___x_441_ = l_Repr_addAppParen(v___x_440_, v_prec_434_);
return v___x_441_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr___boxed(lean_object* v_x_494_, lean_object* v_prec_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprConcreteInitialRootVectorId_repr(v_x_494_, v_prec_495_);
lean_dec(v_prec_495_);
return v_res_496_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg(lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_x_501_, lean_object* v_x_502_){
_start:
{
lean_object* v_commitmentId_503_; lean_object* v_digest_504_; lean_object* v_commitmentId_505_; lean_object* v_digest_506_; lean_object* v___x_507_; uint8_t v___x_508_; 
v_commitmentId_503_ = lean_ctor_get(v_x_501_, 0);
lean_inc(v_commitmentId_503_);
v_digest_504_ = lean_ctor_get(v_x_501_, 1);
lean_inc(v_digest_504_);
lean_dec_ref(v_x_501_);
v_commitmentId_505_ = lean_ctor_get(v_x_502_, 0);
lean_inc(v_commitmentId_505_);
v_digest_506_ = lean_ctor_get(v_x_502_, 1);
lean_inc(v_digest_506_);
lean_dec_ref(v_x_502_);
v___x_507_ = lean_apply_2(v_inst_499_, v_commitmentId_503_, v_commitmentId_505_);
v___x_508_ = lean_unbox(v___x_507_);
if (v___x_508_ == 0)
{
uint8_t v___x_509_; 
lean_dec(v_digest_506_);
lean_dec(v_digest_504_);
lean_dec_ref(v_inst_500_);
v___x_509_ = lean_unbox(v___x_507_);
return v___x_509_;
}
else
{
lean_object* v___x_510_; uint8_t v___x_511_; 
v___x_510_ = lean_apply_2(v_inst_500_, v_digest_504_, v_digest_506_);
v___x_511_ = lean_unbox(v___x_510_);
return v___x_511_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg___boxed(lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_x_514_, lean_object* v_x_515_){
_start:
{
uint8_t v_res_516_; lean_object* v_r_517_; 
v_res_516_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg(v_inst_512_, v_inst_513_, v_x_514_, v_x_515_);
v_r_517_ = lean_box(v_res_516_);
return v_r_517_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq(lean_object* v_CommitmentId_518_, lean_object* v_Digest_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_x_522_, lean_object* v_x_523_){
_start:
{
uint8_t v___x_524_; 
v___x_524_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg(v_inst_520_, v_inst_521_, v_x_522_, v_x_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___boxed(lean_object* v_CommitmentId_525_, lean_object* v_Digest_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_x_529_, lean_object* v_x_530_){
_start:
{
uint8_t v_res_531_; lean_object* v_r_532_; 
v_res_531_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq(v_CommitmentId_525_, v_Digest_526_, v_inst_527_, v_inst_528_, v_x_529_, v_x_530_);
v_r_532_ = lean_box(v_res_531_);
return v_r_532_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___redArg(lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_x_535_, lean_object* v_x_536_){
_start:
{
uint8_t v___x_537_; 
v___x_537_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg(v_inst_533_, v_inst_534_, v_x_535_, v_x_536_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___redArg___boxed(lean_object* v_inst_538_, lean_object* v_inst_539_, lean_object* v_x_540_, lean_object* v_x_541_){
_start:
{
uint8_t v_res_542_; lean_object* v_r_543_; 
v_res_542_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___redArg(v_inst_538_, v_inst_539_, v_x_540_, v_x_541_);
v_r_543_ = lean_box(v_res_542_);
return v_r_543_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry(lean_object* v_CommitmentId_544_, lean_object* v_Digest_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_x_548_, lean_object* v_x_549_){
_start:
{
uint8_t v___x_550_; 
v___x_550_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry_decEq___redArg(v_inst_546_, v_inst_547_, v_x_548_, v_x_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry___boxed(lean_object* v_CommitmentId_551_, lean_object* v_Digest_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_x_555_, lean_object* v_x_556_){
_start:
{
uint8_t v_res_557_; lean_object* v_r_558_; 
v_res_557_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instDecidableEqInitialRootVectorEntry(v_CommitmentId_551_, v_Digest_552_, v_inst_553_, v_inst_554_, v_x_555_, v_x_556_);
v_r_558_ = lean_box(v_res_557_);
return v_r_558_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_572_ = lean_unsigned_to_nat(16u);
v___x_573_ = lean_nat_to_int(v___x_572_);
return v___x_573_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_580_ = lean_unsigned_to_nat(10u);
v___x_581_ = lean_nat_to_int(v___x_580_);
return v___x_581_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__14(void){
_start:
{
lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_583_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__0));
v___x_584_ = lean_string_length(v___x_583_);
return v___x_584_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_585_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__14, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__14_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__14);
v___x_586_ = lean_nat_to_int(v___x_585_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg(lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_x_593_){
_start:
{
lean_object* v_commitmentId_594_; lean_object* v_digest_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_629_; 
v_commitmentId_594_ = lean_ctor_get(v_x_593_, 0);
v_digest_595_ = lean_ctor_get(v_x_593_, 1);
v_isSharedCheck_629_ = !lean_is_exclusive(v_x_593_);
if (v_isSharedCheck_629_ == 0)
{
v___x_597_ = v_x_593_;
v_isShared_598_ = v_isSharedCheck_629_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_digest_595_);
lean_inc(v_commitmentId_594_);
lean_dec(v_x_593_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_629_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_605_; 
v___x_599_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__5));
v___x_600_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__6));
v___x_601_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__7, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__7);
v___x_602_ = lean_unsigned_to_nat(0u);
v___x_603_ = lean_apply_2(v_inst_591_, v_commitmentId_594_, v___x_602_);
if (v_isShared_598_ == 0)
{
lean_ctor_set_tag(v___x_597_, 4);
lean_ctor_set(v___x_597_, 1, v___x_603_);
lean_ctor_set(v___x_597_, 0, v___x_601_);
v___x_605_ = v___x_597_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_628_; 
v_reuseFailAlloc_628_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_628_, 0, v___x_601_);
lean_ctor_set(v_reuseFailAlloc_628_, 1, v___x_603_);
v___x_605_ = v_reuseFailAlloc_628_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
uint8_t v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_606_ = 0;
v___x_607_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_607_, 0, v___x_605_);
lean_ctor_set_uint8(v___x_607_, sizeof(void*)*1, v___x_606_);
v___x_608_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_608_, 0, v___x_600_);
lean_ctor_set(v___x_608_, 1, v___x_607_);
v___x_609_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__9));
v___x_610_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_610_, 0, v___x_608_);
lean_ctor_set(v___x_610_, 1, v___x_609_);
v___x_611_ = lean_box(1);
v___x_612_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_612_, 0, v___x_610_);
lean_ctor_set(v___x_612_, 1, v___x_611_);
v___x_613_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__11));
v___x_614_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_614_, 0, v___x_612_);
lean_ctor_set(v___x_614_, 1, v___x_613_);
v___x_615_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_615_, 0, v___x_614_);
lean_ctor_set(v___x_615_, 1, v___x_599_);
v___x_616_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__12, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__12_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__12);
v___x_617_ = lean_apply_2(v_inst_592_, v_digest_595_, v___x_602_);
v___x_618_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_618_, 0, v___x_616_);
lean_ctor_set(v___x_618_, 1, v___x_617_);
v___x_619_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_619_, 0, v___x_618_);
lean_ctor_set_uint8(v___x_619_, sizeof(void*)*1, v___x_606_);
v___x_620_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_615_);
lean_ctor_set(v___x_620_, 1, v___x_619_);
v___x_621_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__15, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__15_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__15);
v___x_622_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__16));
v___x_623_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_623_, 0, v___x_622_);
lean_ctor_set(v___x_623_, 1, v___x_620_);
v___x_624_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg___closed__17));
v___x_625_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_625_, 0, v___x_623_);
lean_ctor_set(v___x_625_, 1, v___x_624_);
v___x_626_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_621_);
lean_ctor_set(v___x_626_, 1, v___x_625_);
v___x_627_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_627_, 0, v___x_626_);
lean_ctor_set_uint8(v___x_627_, sizeof(void*)*1, v___x_606_);
return v___x_627_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr(lean_object* v_CommitmentId_630_, lean_object* v_Digest_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_x_634_, lean_object* v_prec_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___redArg(v_inst_632_, v_inst_633_, v_x_634_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___boxed(lean_object* v_CommitmentId_637_, lean_object* v_Digest_638_, lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_x_641_, lean_object* v_prec_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr(v_CommitmentId_637_, v_Digest_638_, v_inst_639_, v_inst_640_, v_x_641_, v_prec_642_);
lean_dec(v_prec_642_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry___redArg(lean_object* v_inst_644_, lean_object* v_inst_645_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___boxed), 6, 4);
lean_closure_set(v___x_646_, 0, lean_box(0));
lean_closure_set(v___x_646_, 1, lean_box(0));
lean_closure_set(v___x_646_, 2, v_inst_644_);
lean_closure_set(v___x_646_, 3, v_inst_645_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry(lean_object* v_CommitmentId_647_, lean_object* v_Digest_648_, lean_object* v_inst_649_, lean_object* v_inst_650_){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_instReprInitialRootVectorEntry_repr___boxed), 6, 4);
lean_closure_set(v___x_651_, 0, lean_box(0));
lean_closure_set(v___x_651_, 1, lean_box(0));
lean_closure_set(v___x_651_, 2, v_inst_649_);
lean_closure_set(v___x_651_, 3, v_inst_650_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir___redArg(lean_object* v_airId_652_, lean_object* v_avk_653_){
_start:
{
lean_object* v_preprocessedData_654_; 
v_preprocessedData_654_ = lean_ctor_get(v_avk_653_, 0);
lean_inc(v_preprocessedData_654_);
lean_dec_ref(v_avk_653_);
if (lean_obj_tag(v_preprocessedData_654_) == 0)
{
lean_object* v___x_655_; 
lean_dec(v_airId_652_);
v___x_655_ = lean_box(0);
return v___x_655_;
}
else
{
lean_object* v_val_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_667_; 
v_val_656_ = lean_ctor_get(v_preprocessedData_654_, 0);
v_isSharedCheck_667_ = !lean_is_exclusive(v_preprocessedData_654_);
if (v_isSharedCheck_667_ == 0)
{
v___x_658_ = v_preprocessedData_654_;
v_isShared_659_ = v_isSharedCheck_667_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_val_656_);
lean_dec(v_preprocessedData_654_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_667_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v_commit_660_; lean_object* v___x_662_; 
v_commit_660_ = lean_ctor_get(v_val_656_, 0);
lean_inc(v_commit_660_);
lean_dec(v_val_656_);
if (v_isShared_659_ == 0)
{
lean_ctor_set_tag(v___x_658_, 0);
lean_ctor_set(v___x_658_, 0, v_airId_652_);
v___x_662_ = v___x_658_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_666_; 
v_reuseFailAlloc_666_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_666_, 0, v_airId_652_);
v___x_662_ = v_reuseFailAlloc_666_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; 
v___x_663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_663_, 0, v___x_662_);
lean_ctor_set(v___x_663_, 1, v_commit_660_);
v___x_664_ = lean_box(0);
v___x_665_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_665_, 0, v___x_663_);
lean_ctor_set(v___x_665_, 1, v___x_664_);
return v___x_665_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir(lean_object* v_F_668_, lean_object* v_Digest_669_, lean_object* v_airId_670_, lean_object* v_avk_671_){
_start:
{
lean_object* v___x_672_; 
v___x_672_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir___redArg(v_airId_670_, v_avk_671_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir_spec__0___redArg(lean_object* v_airId_673_, lean_object* v_a_674_, lean_object* v_a_675_){
_start:
{
if (lean_obj_tag(v_a_674_) == 0)
{
lean_object* v___x_676_; 
lean_dec(v_airId_673_);
v___x_676_ = l_List_reverse___redArg(v_a_675_);
return v___x_676_;
}
else
{
lean_object* v_head_677_; lean_object* v_tail_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_696_; 
v_head_677_ = lean_ctor_get(v_a_674_, 0);
v_tail_678_ = lean_ctor_get(v_a_674_, 1);
v_isSharedCheck_696_ = !lean_is_exclusive(v_a_674_);
if (v_isSharedCheck_696_ == 0)
{
v___x_680_ = v_a_674_;
v_isShared_681_ = v_isSharedCheck_696_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_tail_678_);
lean_inc(v_head_677_);
lean_dec(v_a_674_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_696_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v_fst_682_; lean_object* v_snd_683_; lean_object* v___x_685_; uint8_t v_isShared_686_; uint8_t v_isSharedCheck_695_; 
v_fst_682_ = lean_ctor_get(v_head_677_, 0);
v_snd_683_ = lean_ctor_get(v_head_677_, 1);
v_isSharedCheck_695_ = !lean_is_exclusive(v_head_677_);
if (v_isSharedCheck_695_ == 0)
{
v___x_685_ = v_head_677_;
v_isShared_686_ = v_isSharedCheck_695_;
goto v_resetjp_684_;
}
else
{
lean_inc(v_snd_683_);
lean_inc(v_fst_682_);
lean_dec(v_head_677_);
v___x_685_ = lean_box(0);
v_isShared_686_ = v_isSharedCheck_695_;
goto v_resetjp_684_;
}
v_resetjp_684_:
{
lean_object* v___x_688_; 
lean_inc(v_airId_673_);
if (v_isShared_686_ == 0)
{
lean_ctor_set_tag(v___x_685_, 2);
lean_ctor_set(v___x_685_, 1, v_fst_682_);
lean_ctor_set(v___x_685_, 0, v_airId_673_);
v___x_688_ = v___x_685_;
goto v_reusejp_687_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v_airId_673_);
lean_ctor_set(v_reuseFailAlloc_694_, 1, v_fst_682_);
v___x_688_ = v_reuseFailAlloc_694_;
goto v_reusejp_687_;
}
v_reusejp_687_:
{
lean_object* v___x_689_; lean_object* v___x_691_; 
v___x_689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_689_, 0, v___x_688_);
lean_ctor_set(v___x_689_, 1, v_snd_683_);
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 1, v_a_675_);
lean_ctor_set(v___x_680_, 0, v___x_689_);
v___x_691_ = v___x_680_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v___x_689_);
lean_ctor_set(v_reuseFailAlloc_693_, 1, v_a_675_);
v___x_691_ = v_reuseFailAlloc_693_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
v_a_674_ = v_tail_678_;
v_a_675_ = v___x_691_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir___redArg(lean_object* v_airId_697_, lean_object* v_traceVData_698_){
_start:
{
if (lean_obj_tag(v_traceVData_698_) == 0)
{
lean_object* v___x_699_; 
lean_dec(v_airId_697_);
v___x_699_ = lean_box(0);
return v___x_699_;
}
else
{
lean_object* v_val_700_; lean_object* v_cachedCommitments_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; 
v_val_700_ = lean_ctor_get(v_traceVData_698_, 0);
lean_inc(v_val_700_);
lean_dec_ref_known(v_traceVData_698_, 1);
v_cachedCommitments_701_ = lean_ctor_get(v_val_700_, 1);
lean_inc(v_cachedCommitments_701_);
lean_dec(v_val_700_);
v___x_702_ = l_List_lengthTR___redArg(v_cachedCommitments_701_);
v___x_703_ = l_List_range(v___x_702_);
v___x_704_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_703_, v_cachedCommitments_701_);
v___x_705_ = lean_box(0);
v___x_706_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir_spec__0___redArg(v_airId_697_, v___x_704_, v___x_705_);
return v___x_706_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir(lean_object* v_Digest_707_, lean_object* v_airId_708_, lean_object* v_traceVData_709_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir___redArg(v_airId_708_, v_traceVData_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir_spec__0(lean_object* v_Digest_711_, lean_object* v_airId_712_, lean_object* v_a_713_, lean_object* v_a_714_){
_start:
{
lean_object* v___x_715_; 
v___x_715_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir_spec__0___redArg(v_airId_712_, v_a_713_, v_a_714_);
return v___x_715_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___redArg(lean_object* v_vk_716_, lean_object* v_traceVdata_717_, lean_object* v_commits_718_, lean_object* v_airId_719_){
_start:
{
lean_object* v___y_721_; lean_object* v___x_725_; 
lean_inc(v_airId_719_);
v___x_725_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_716_, v_airId_719_);
if (lean_obj_tag(v___x_725_) == 0)
{
v___y_721_ = v_commits_718_;
goto v___jp_720_;
}
else
{
lean_object* v_val_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
v_val_726_ = lean_ctor_get(v___x_725_, 0);
lean_inc(v_val_726_);
lean_dec_ref_known(v___x_725_, 1);
lean_inc(v_airId_719_);
v___x_727_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir___redArg(v_airId_719_, v_val_726_);
v___x_728_ = l_List_appendTR___redArg(v_commits_718_, v___x_727_);
v___y_721_ = v___x_728_;
goto v___jp_720_;
}
v___jp_720_:
{
lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; 
lean_inc(v_airId_719_);
v___x_722_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_717_, v_airId_719_);
v___x_723_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir___redArg(v_airId_719_, v___x_722_);
v___x_724_ = l_List_appendTR___redArg(v___y_721_, v___x_723_);
return v___x_724_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___redArg___boxed(lean_object* v_vk_729_, lean_object* v_traceVdata_730_, lean_object* v_commits_731_, lean_object* v_airId_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___redArg(v_vk_729_, v_traceVdata_730_, v_commits_731_, v_airId_732_);
lean_dec(v_traceVdata_730_);
lean_dec_ref(v_vk_729_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir(lean_object* v_F_734_, lean_object* v_Digest_735_, lean_object* v_vk_736_, lean_object* v_traceVdata_737_, lean_object* v_commits_738_, lean_object* v_airId_739_){
_start:
{
lean_object* v___x_740_; 
v___x_740_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___redArg(v_vk_736_, v_traceVdata_737_, v_commits_738_, v_airId_739_);
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir___boxed(lean_object* v_F_741_, lean_object* v_Digest_742_, lean_object* v_vk_743_, lean_object* v_traceVdata_744_, lean_object* v_commits_745_, lean_object* v_airId_746_){
_start:
{
lean_object* v_res_747_; 
v_res_747_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_appendInitialRootEntriesForAir(v_F_741_, v_Digest_742_, v_vk_743_, v_traceVdata_744_, v_commits_745_, v_airId_746_);
lean_dec(v_traceVdata_744_);
lean_dec_ref(v_vk_743_);
return v_res_747_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg(lean_object* v_traceVdata_748_, lean_object* v_vk_749_, lean_object* v_x_750_, lean_object* v_x_751_){
_start:
{
if (lean_obj_tag(v_x_751_) == 0)
{
return v_x_750_;
}
else
{
lean_object* v_head_752_; lean_object* v_tail_753_; lean_object* v___y_755_; lean_object* v___x_760_; 
v_head_752_ = lean_ctor_get(v_x_751_, 0);
lean_inc_n(v_head_752_, 2);
v_tail_753_ = lean_ctor_get(v_x_751_, 1);
lean_inc(v_tail_753_);
lean_dec_ref_known(v_x_751_, 2);
v___x_760_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_749_, v_head_752_);
if (lean_obj_tag(v___x_760_) == 0)
{
v___y_755_ = v_x_750_;
goto v___jp_754_;
}
else
{
lean_object* v_val_761_; lean_object* v___x_762_; lean_object* v___x_763_; 
v_val_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc(v_val_761_);
lean_dec_ref_known(v___x_760_, 1);
lean_inc(v_head_752_);
v___x_762_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir___redArg(v_head_752_, v_val_761_);
v___x_763_ = l_List_appendTR___redArg(v_x_750_, v___x_762_);
v___y_755_ = v___x_763_;
goto v___jp_754_;
}
v___jp_754_:
{
lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; 
lean_inc(v_head_752_);
v___x_756_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceVDataAt___redArg(v_traceVdata_748_, v_head_752_);
v___x_757_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir___redArg(v_head_752_, v___x_756_);
v___x_758_ = l_List_appendTR___redArg(v___y_755_, v___x_757_);
v_x_750_ = v___x_758_;
v_x_751_ = v_tail_753_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg___boxed(lean_object* v_traceVdata_764_, lean_object* v_vk_765_, lean_object* v_x_766_, lean_object* v_x_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg(v_traceVdata_764_, v_vk_765_, v_x_766_, v_x_767_);
lean_dec_ref(v_vk_765_);
lean_dec(v_traceVdata_764_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg(lean_object* v_vk_769_, lean_object* v_commonMainCommit_770_, lean_object* v_traceVdata_771_, lean_object* v_traceIdToAirId_772_){
_start:
{
lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v___x_773_ = lean_box(1);
v___x_774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
lean_ctor_set(v___x_774_, 1, v_commonMainCommit_770_);
v___x_775_ = lean_box(0);
v___x_776_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_776_, 0, v___x_774_);
lean_ctor_set(v___x_776_, 1, v___x_775_);
v___x_777_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg(v_traceVdata_771_, v_vk_769_, v___x_776_, v_traceIdToAirId_772_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg___boxed(lean_object* v_vk_778_, lean_object* v_commonMainCommit_779_, lean_object* v_traceVdata_780_, lean_object* v_traceIdToAirId_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg(v_vk_778_, v_commonMainCommit_779_, v_traceVdata_780_, v_traceIdToAirId_781_);
lean_dec(v_traceVdata_780_);
lean_dec_ref(v_vk_778_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView(lean_object* v_F_783_, lean_object* v_Digest_784_, lean_object* v_vk_785_, lean_object* v_commonMainCommit_786_, lean_object* v_traceVdata_787_, lean_object* v_traceIdToAirId_788_){
_start:
{
lean_object* v___x_789_; 
v___x_789_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg(v_vk_785_, v_commonMainCommit_786_, v_traceVdata_787_, v_traceIdToAirId_788_);
return v___x_789_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___boxed(lean_object* v_F_790_, lean_object* v_Digest_791_, lean_object* v_vk_792_, lean_object* v_commonMainCommit_793_, lean_object* v_traceVdata_794_, lean_object* v_traceIdToAirId_795_){
_start:
{
lean_object* v_res_796_; 
v_res_796_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView(v_F_790_, v_Digest_791_, v_vk_792_, v_commonMainCommit_793_, v_traceVdata_794_, v_traceIdToAirId_795_);
lean_dec(v_traceVdata_794_);
lean_dec_ref(v_vk_792_);
return v_res_796_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0(lean_object* v_Digest_797_, lean_object* v_traceVdata_798_, lean_object* v_F_799_, lean_object* v_vk_800_, lean_object* v_x_801_, lean_object* v_x_802_){
_start:
{
lean_object* v___x_803_; 
v___x_803_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___redArg(v_traceVdata_798_, v_vk_800_, v_x_801_, v_x_802_);
return v___x_803_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0___boxed(lean_object* v_Digest_804_, lean_object* v_traceVdata_805_, lean_object* v_F_806_, lean_object* v_vk_807_, lean_object* v_x_808_, lean_object* v_x_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView_spec__0(v_Digest_804_, v_traceVdata_805_, v_F_806_, v_vk_807_, v_x_808_, v_x_809_);
lean_dec_ref(v_vk_807_);
lean_dec(v_traceVdata_805_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(lean_object* v_a_811_, lean_object* v_a_812_){
_start:
{
if (lean_obj_tag(v_a_811_) == 0)
{
lean_object* v___x_813_; 
v___x_813_ = l_List_reverse___redArg(v_a_812_);
return v___x_813_;
}
else
{
lean_object* v_head_814_; lean_object* v_tail_815_; lean_object* v___x_817_; uint8_t v_isShared_818_; uint8_t v_isSharedCheck_824_; 
v_head_814_ = lean_ctor_get(v_a_811_, 0);
v_tail_815_ = lean_ctor_get(v_a_811_, 1);
v_isSharedCheck_824_ = !lean_is_exclusive(v_a_811_);
if (v_isSharedCheck_824_ == 0)
{
v___x_817_ = v_a_811_;
v_isShared_818_ = v_isSharedCheck_824_;
goto v_resetjp_816_;
}
else
{
lean_inc(v_tail_815_);
lean_inc(v_head_814_);
lean_dec(v_a_811_);
v___x_817_ = lean_box(0);
v_isShared_818_ = v_isSharedCheck_824_;
goto v_resetjp_816_;
}
v_resetjp_816_:
{
lean_object* v_digest_819_; lean_object* v___x_821_; 
v_digest_819_ = lean_ctor_get(v_head_814_, 1);
lean_inc(v_digest_819_);
lean_dec(v_head_814_);
if (v_isShared_818_ == 0)
{
lean_ctor_set(v___x_817_, 1, v_a_812_);
lean_ctor_set(v___x_817_, 0, v_digest_819_);
v___x_821_ = v___x_817_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_digest_819_);
lean_ctor_set(v_reuseFailAlloc_823_, 1, v_a_812_);
v___x_821_ = v_reuseFailAlloc_823_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
v_a_811_ = v_tail_815_;
v_a_812_ = v___x_821_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg(lean_object* v_vk_825_, lean_object* v_commonMainCommit_826_, lean_object* v_traceVdata_827_, lean_object* v_traceIdToAirId_828_){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_829_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_whirInitialCommitmentView___redArg(v_vk_825_, v_commonMainCommit_826_, v_traceVdata_827_, v_traceIdToAirId_828_);
v___x_830_ = lean_box(0);
v___x_831_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(v___x_829_, v___x_830_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg___boxed(lean_object* v_vk_832_, lean_object* v_commonMainCommit_833_, lean_object* v_traceVdata_834_, lean_object* v_traceIdToAirId_835_){
_start:
{
lean_object* v_res_836_; 
v_res_836_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg(v_vk_832_, v_commonMainCommit_833_, v_traceVdata_834_, v_traceIdToAirId_835_);
lean_dec(v_traceVdata_834_);
lean_dec_ref(v_vk_832_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits(lean_object* v_F_837_, lean_object* v_Digest_838_, lean_object* v_vk_839_, lean_object* v_commonMainCommit_840_, lean_object* v_traceVdata_841_, lean_object* v_traceIdToAirId_842_){
_start:
{
lean_object* v___x_843_; 
v___x_843_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg(v_vk_839_, v_commonMainCommit_840_, v_traceVdata_841_, v_traceIdToAirId_842_);
return v___x_843_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___boxed(lean_object* v_F_844_, lean_object* v_Digest_845_, lean_object* v_vk_846_, lean_object* v_commonMainCommit_847_, lean_object* v_traceVdata_848_, lean_object* v_traceIdToAirId_849_){
_start:
{
lean_object* v_res_850_; 
v_res_850_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits(v_F_844_, v_Digest_845_, v_vk_846_, v_commonMainCommit_847_, v_traceVdata_848_, v_traceIdToAirId_849_);
lean_dec(v_traceVdata_848_);
lean_dec_ref(v_vk_846_);
return v_res_850_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0(lean_object* v_Digest_851_, lean_object* v_a_852_, lean_object* v_a_853_){
_start:
{
lean_object* v___x_854_; 
v___x_854_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(v_a_852_, v_a_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___redArg(lean_object* v_publicValues_855_, lean_object* v_airId_856_){
_start:
{
lean_object* v___x_857_; 
v___x_857_ = l_List_get_x3fInternal___redArg(v_publicValues_855_, v_airId_856_);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___redArg___boxed(lean_object* v_publicValues_858_, lean_object* v_airId_859_){
_start:
{
lean_object* v_res_860_; 
v_res_860_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___redArg(v_publicValues_858_, v_airId_859_);
lean_dec(v_publicValues_858_);
return v_res_860_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt(lean_object* v_F_861_, lean_object* v_publicValues_862_, lean_object* v_airId_863_){
_start:
{
lean_object* v___x_864_; 
v___x_864_ = l_List_get_x3fInternal___redArg(v_publicValues_862_, v_airId_863_);
return v___x_864_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt___boxed(lean_object* v_F_865_, lean_object* v_publicValues_866_, lean_object* v_airId_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_publicValuesAt(v_F_865_, v_publicValues_866_, v_airId_867_);
lean_dec(v_publicValues_866_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___lam__0(lean_object* v_inst_869_, lean_object* v_x_870_, lean_object* v_value_871_, lean_object* v___y_872_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(v_inst_869_, v_value_871_, v___y_872_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___lam__1(lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_x_876_, lean_object* v_commit_877_, lean_object* v___y_878_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(v_inst_874_, v_inst_875_, v_commit_877_, v___y_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg(lean_object* v_inst_928_, lean_object* v_inst_929_, lean_object* v_lSkip_930_, lean_object* v_avk_931_, lean_object* v_traceVData_932_, lean_object* v_publicValues_933_, lean_object* v_a_934_){
_start:
{
lean_object* v___x_935_; lean_object* v_preprocessedData_936_; uint8_t v_isRequired_937_; lean_object* v___f_938_; lean_object* v___y_940_; lean_object* v___f_961_; lean_object* v_cachedCommitments_963_; lean_object* v___y_964_; lean_object* v___y_971_; 
v___x_935_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__19));
v_preprocessedData_936_ = lean_ctor_get(v_avk_931_, 0);
lean_inc(v_preprocessedData_936_);
v_isRequired_937_ = lean_ctor_get_uint8(v_avk_931_, sizeof(void*)*5);
lean_dec_ref(v_avk_931_);
lean_inc_ref_n(v_inst_929_, 2);
v___f_938_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___lam__0), 4, 1);
lean_closure_set(v___f_938_, 0, v_inst_929_);
lean_inc_ref(v_inst_928_);
v___f_961_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___lam__1), 5, 2);
lean_closure_set(v___f_961_, 0, v_inst_929_);
lean_closure_set(v___f_961_, 1, v_inst_928_);
if (v_isRequired_937_ == 0)
{
lean_object* v_ofBool_994_; uint8_t v___y_996_; 
v_ofBool_994_ = lean_ctor_get(v_inst_928_, 0);
if (lean_obj_tag(v_traceVData_932_) == 0)
{
v___y_996_ = v_isRequired_937_;
goto v___jp_995_;
}
else
{
uint8_t v___x_1002_; 
v___x_1002_ = 1;
v___y_996_ = v___x_1002_;
goto v___jp_995_;
}
v___jp_995_:
{
lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v_a_1000_; lean_object* v_snd_1001_; 
v___x_997_ = lean_box(v___y_996_);
lean_inc(v_ofBool_994_);
v___x_998_ = lean_apply_1(v_ofBool_994_, v___x_997_);
lean_inc_ref(v_inst_929_);
v___x_999_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(v_inst_929_, v___x_998_, v_a_934_);
v_a_1000_ = lean_ctor_get(v___x_999_, 0);
lean_inc(v_a_1000_);
lean_dec_ref(v___x_999_);
v_snd_1001_ = lean_ctor_get(v_a_1000_, 1);
lean_inc(v_snd_1001_);
lean_dec(v_a_1000_);
v___y_971_ = v_snd_1001_;
goto v___jp_970_;
}
}
else
{
v___y_971_ = v_a_934_;
goto v___jp_970_;
}
v___jp_939_:
{
lean_object* v___x_941_; lean_object* v___x_406__overap_942_; lean_object* v___x_943_; 
v___x_941_ = lean_box(0);
v___x_406__overap_942_ = l_List_foldlM___redArg(v___x_935_, v___f_938_, v___x_941_, v_publicValues_933_);
v___x_943_ = lean_apply_1(v___x_406__overap_942_, v___y_940_);
if (lean_obj_tag(v___x_943_) == 0)
{
return v___x_943_;
}
else
{
lean_object* v_a_944_; lean_object* v___x_946_; uint8_t v_isShared_947_; uint8_t v_isSharedCheck_960_; 
v_a_944_ = lean_ctor_get(v___x_943_, 0);
v_isSharedCheck_960_ = !lean_is_exclusive(v___x_943_);
if (v_isSharedCheck_960_ == 0)
{
v___x_946_ = v___x_943_;
v_isShared_947_ = v_isSharedCheck_960_;
goto v_resetjp_945_;
}
else
{
lean_inc(v_a_944_);
lean_dec(v___x_943_);
v___x_946_ = lean_box(0);
v_isShared_947_ = v_isSharedCheck_960_;
goto v_resetjp_945_;
}
v_resetjp_945_:
{
lean_object* v_snd_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_958_; 
v_snd_948_ = lean_ctor_get(v_a_944_, 1);
v_isSharedCheck_958_ = !lean_is_exclusive(v_a_944_);
if (v_isSharedCheck_958_ == 0)
{
lean_object* v_unused_959_; 
v_unused_959_ = lean_ctor_get(v_a_944_, 0);
lean_dec(v_unused_959_);
v___x_950_ = v_a_944_;
v_isShared_951_ = v_isSharedCheck_958_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_snd_948_);
lean_dec(v_a_944_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_958_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_953_; 
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 0, v___x_941_);
v___x_953_ = v___x_950_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_957_; 
v_reuseFailAlloc_957_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_957_, 0, v___x_941_);
lean_ctor_set(v_reuseFailAlloc_957_, 1, v_snd_948_);
v___x_953_ = v_reuseFailAlloc_957_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
lean_object* v___x_955_; 
if (v_isShared_947_ == 0)
{
lean_ctor_set(v___x_946_, 0, v___x_953_);
v___x_955_ = v___x_946_;
goto v_reusejp_954_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v___x_953_);
v___x_955_ = v_reuseFailAlloc_956_;
goto v_reusejp_954_;
}
v_reusejp_954_:
{
return v___x_955_;
}
}
}
}
}
}
v___jp_962_:
{
lean_object* v___x_965_; lean_object* v___x_1699__overap_966_; lean_object* v___x_967_; 
v___x_965_ = lean_box(0);
v___x_1699__overap_966_ = l_List_foldlM___redArg(v___x_935_, v___f_961_, v___x_965_, v_cachedCommitments_963_);
v___x_967_ = lean_apply_1(v___x_1699__overap_966_, v___y_964_);
if (lean_obj_tag(v___x_967_) == 0)
{
lean_dec_ref(v___f_938_);
lean_dec(v_publicValues_933_);
return v___x_967_;
}
else
{
lean_object* v_a_968_; lean_object* v_snd_969_; 
v_a_968_ = lean_ctor_get(v___x_967_, 0);
lean_inc(v_a_968_);
lean_dec_ref_known(v___x_967_, 1);
v_snd_969_ = lean_ctor_get(v_a_968_, 1);
lean_inc(v_snd_969_);
lean_dec(v_a_968_);
v___y_940_ = v_snd_969_;
goto v___jp_939_;
}
}
v___jp_970_:
{
if (lean_obj_tag(v_traceVData_932_) == 0)
{
lean_dec_ref(v___f_961_);
lean_dec(v_preprocessedData_936_);
lean_dec(v_lSkip_930_);
lean_dec_ref(v_inst_929_);
lean_dec_ref(v_inst_928_);
v___y_940_ = v___y_971_;
goto v___jp_939_;
}
else
{
if (lean_obj_tag(v_preprocessedData_936_) == 0)
{
lean_object* v_val_972_; lean_object* v_ofNat_973_; lean_object* v_logHeight_974_; lean_object* v_cachedCommitments_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v_a_978_; lean_object* v_snd_979_; 
lean_dec(v_lSkip_930_);
v_val_972_ = lean_ctor_get(v_traceVData_932_, 0);
lean_inc(v_val_972_);
lean_dec_ref_known(v_traceVData_932_, 1);
v_ofNat_973_ = lean_ctor_get(v_inst_928_, 1);
lean_inc(v_ofNat_973_);
lean_dec_ref(v_inst_928_);
v_logHeight_974_ = lean_ctor_get(v_val_972_, 0);
lean_inc(v_logHeight_974_);
v_cachedCommitments_975_ = lean_ctor_get(v_val_972_, 1);
lean_inc(v_cachedCommitments_975_);
lean_dec(v_val_972_);
v___x_976_ = lean_apply_1(v_ofNat_973_, v_logHeight_974_);
v___x_977_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(v_inst_929_, v___x_976_, v___y_971_);
v_a_978_ = lean_ctor_get(v___x_977_, 0);
lean_inc(v_a_978_);
lean_dec_ref(v___x_977_);
v_snd_979_ = lean_ctor_get(v_a_978_, 1);
lean_inc(v_snd_979_);
lean_dec(v_a_978_);
v_cachedCommitments_963_ = v_cachedCommitments_975_;
v___y_964_ = v_snd_979_;
goto v___jp_962_;
}
else
{
lean_object* v_val_980_; lean_object* v_val_981_; lean_object* v_commit_982_; lean_object* v_hypercubeDim_983_; lean_object* v_logHeight_984_; lean_object* v_cachedCommitments_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; uint8_t v___x_989_; 
v_val_980_ = lean_ctor_get(v_preprocessedData_936_, 0);
lean_inc(v_val_980_);
lean_dec_ref_known(v_preprocessedData_936_, 1);
v_val_981_ = lean_ctor_get(v_traceVData_932_, 0);
lean_inc(v_val_981_);
lean_dec_ref_known(v_traceVData_932_, 1);
v_commit_982_ = lean_ctor_get(v_val_980_, 0);
lean_inc(v_commit_982_);
v_hypercubeDim_983_ = lean_ctor_get(v_val_980_, 1);
lean_inc(v_hypercubeDim_983_);
lean_dec(v_val_980_);
v_logHeight_984_ = lean_ctor_get(v_val_981_, 0);
lean_inc(v_logHeight_984_);
v_cachedCommitments_985_ = lean_ctor_get(v_val_981_, 1);
lean_inc(v_cachedCommitments_985_);
lean_dec(v_val_981_);
v___x_986_ = lean_nat_to_int(v_lSkip_930_);
v___x_987_ = lean_int_add(v_hypercubeDim_983_, v___x_986_);
lean_dec(v___x_986_);
lean_dec(v_hypercubeDim_983_);
v___x_988_ = lean_nat_to_int(v_logHeight_984_);
v___x_989_ = lean_int_dec_eq(v___x_987_, v___x_988_);
lean_dec(v___x_988_);
lean_dec(v___x_987_);
if (v___x_989_ == 0)
{
lean_object* v___x_990_; 
lean_dec(v_cachedCommitments_985_);
lean_dec(v_commit_982_);
lean_dec_ref(v___y_971_);
lean_dec_ref(v___f_961_);
lean_dec_ref(v___f_938_);
lean_dec(v_publicValues_933_);
lean_dec_ref(v_inst_929_);
lean_dec_ref(v_inst_928_);
v___x_990_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg___closed__20));
return v___x_990_;
}
else
{
lean_object* v___x_991_; lean_object* v_a_992_; lean_object* v_snd_993_; 
v___x_991_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(v_inst_929_, v_inst_928_, v_commit_982_, v___y_971_);
v_a_992_ = lean_ctor_get(v___x_991_, 0);
lean_inc(v_a_992_);
lean_dec_ref(v___x_991_);
v_snd_993_ = lean_ctor_get(v_a_992_, 1);
lean_inc(v_snd_993_);
lean_dec(v_a_992_);
v_cachedCommitments_963_ = v_cachedCommitments_985_;
v___y_964_ = v_snd_993_;
goto v___jp_962_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM(lean_object* v_F_1003_, lean_object* v_Digest_1004_, lean_object* v_inst_1005_, lean_object* v_inst_1006_, lean_object* v_lSkip_1007_, lean_object* v_avk_1008_, lean_object* v_traceVData_1009_, lean_object* v_publicValues_1010_, lean_object* v_a_1011_){
_start:
{
lean_object* v___x_1012_; 
v___x_1012_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg(v_inst_1005_, v_inst_1006_, v_lSkip_1007_, v_avk_1008_, v_traceVData_1009_, v_publicValues_1010_, v_a_1011_);
return v___x_1012_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg(lean_object* v_inst_1016_, lean_object* v_inst_1017_, lean_object* v_lSkip_1018_, lean_object* v_publicValues_1019_, lean_object* v_x_1020_, lean_object* v_x_1021_, lean_object* v_x_1022_, lean_object* v_a_1023_){
_start:
{
if (lean_obj_tag(v_x_1021_) == 0)
{
lean_dec(v_x_1020_);
lean_dec(v_lSkip_1018_);
lean_dec_ref(v_inst_1017_);
lean_dec_ref(v_inst_1016_);
if (lean_obj_tag(v_x_1022_) == 0)
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; 
v___x_1026_ = lean_box(0);
v___x_1027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1027_, 0, v___x_1026_);
lean_ctor_set(v___x_1027_, 1, v_a_1023_);
v___x_1028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1027_);
return v___x_1028_;
}
else
{
lean_dec_ref(v_a_1023_);
lean_dec(v_x_1022_);
goto v___jp_1024_;
}
}
else
{
if (lean_obj_tag(v_x_1022_) == 1)
{
lean_object* v_head_1029_; lean_object* v_tail_1030_; lean_object* v_head_1031_; lean_object* v_tail_1032_; lean_object* v___x_1033_; 
v_head_1029_ = lean_ctor_get(v_x_1021_, 0);
lean_inc(v_head_1029_);
v_tail_1030_ = lean_ctor_get(v_x_1021_, 1);
lean_inc(v_tail_1030_);
lean_dec_ref_known(v_x_1021_, 2);
v_head_1031_ = lean_ctor_get(v_x_1022_, 0);
lean_inc(v_head_1031_);
v_tail_1032_ = lean_ctor_get(v_x_1022_, 1);
lean_inc(v_tail_1032_);
lean_dec_ref_known(v_x_1022_, 2);
lean_inc(v_x_1020_);
v___x_1033_ = l_List_get_x3fInternal___redArg(v_publicValues_1019_, v_x_1020_);
if (lean_obj_tag(v___x_1033_) == 0)
{
lean_object* v___x_1034_; 
lean_dec(v_tail_1032_);
lean_dec(v_head_1031_);
lean_dec(v_tail_1030_);
lean_dec(v_head_1029_);
lean_dec_ref(v_a_1023_);
lean_dec(v_x_1020_);
lean_dec(v_lSkip_1018_);
lean_dec_ref(v_inst_1017_);
lean_dec_ref(v_inst_1016_);
v___x_1034_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___closed__0));
return v___x_1034_;
}
else
{
lean_object* v_val_1035_; lean_object* v___x_1036_; 
v_val_1035_ = lean_ctor_get(v___x_1033_, 0);
lean_inc(v_val_1035_);
lean_dec_ref_known(v___x_1033_, 1);
lean_inc(v_lSkip_1018_);
lean_inc_ref(v_inst_1017_);
lean_inc_ref(v_inst_1016_);
v___x_1036_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleAirM___redArg(v_inst_1016_, v_inst_1017_, v_lSkip_1018_, v_head_1029_, v_head_1031_, v_val_1035_, v_a_1023_);
if (lean_obj_tag(v___x_1036_) == 0)
{
lean_dec(v_tail_1032_);
lean_dec(v_tail_1030_);
lean_dec(v_x_1020_);
lean_dec(v_lSkip_1018_);
lean_dec_ref(v_inst_1017_);
lean_dec_ref(v_inst_1016_);
return v___x_1036_;
}
else
{
lean_object* v_a_1037_; lean_object* v_snd_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v_a_1037_ = lean_ctor_get(v___x_1036_, 0);
lean_inc(v_a_1037_);
lean_dec_ref_known(v___x_1036_, 1);
v_snd_1038_ = lean_ctor_get(v_a_1037_, 1);
lean_inc(v_snd_1038_);
lean_dec(v_a_1037_);
v___x_1039_ = lean_unsigned_to_nat(1u);
v___x_1040_ = lean_nat_add(v_x_1020_, v___x_1039_);
lean_dec(v_x_1020_);
v_x_1020_ = v___x_1040_;
v_x_1021_ = v_tail_1030_;
v_x_1022_ = v_tail_1032_;
v_a_1023_ = v_snd_1038_;
goto _start;
}
}
}
else
{
lean_dec_ref_known(v_x_1021_, 2);
lean_dec_ref(v_a_1023_);
lean_dec(v_x_1022_);
lean_dec(v_x_1020_);
lean_dec(v_lSkip_1018_);
lean_dec_ref(v_inst_1017_);
lean_dec_ref(v_inst_1016_);
goto v___jp_1024_;
}
}
v___jp_1024_:
{
lean_object* v___x_1025_; 
v___x_1025_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___closed__0));
return v___x_1025_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg___boxed(lean_object* v_inst_1042_, lean_object* v_inst_1043_, lean_object* v_lSkip_1044_, lean_object* v_publicValues_1045_, lean_object* v_x_1046_, lean_object* v_x_1047_, lean_object* v_x_1048_, lean_object* v_a_1049_){
_start:
{
lean_object* v_res_1050_; 
v_res_1050_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg(v_inst_1042_, v_inst_1043_, v_lSkip_1044_, v_publicValues_1045_, v_x_1046_, v_x_1047_, v_x_1048_, v_a_1049_);
lean_dec(v_publicValues_1045_);
return v_res_1050_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM(lean_object* v_F_1051_, lean_object* v_Digest_1052_, lean_object* v_inst_1053_, lean_object* v_inst_1054_, lean_object* v_lSkip_1055_, lean_object* v_vk_1056_, lean_object* v_publicValues_1057_, lean_object* v_x_1058_, lean_object* v_x_1059_, lean_object* v_x_1060_, lean_object* v_a_1061_){
_start:
{
lean_object* v___x_1062_; 
v___x_1062_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg(v_inst_1053_, v_inst_1054_, v_lSkip_1055_, v_publicValues_1057_, v_x_1058_, v_x_1059_, v_x_1060_, v_a_1061_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___boxed(lean_object* v_F_1063_, lean_object* v_Digest_1064_, lean_object* v_inst_1065_, lean_object* v_inst_1066_, lean_object* v_lSkip_1067_, lean_object* v_vk_1068_, lean_object* v_publicValues_1069_, lean_object* v_x_1070_, lean_object* v_x_1071_, lean_object* v_x_1072_, lean_object* v_a_1073_){
_start:
{
lean_object* v_res_1074_; 
v_res_1074_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM(v_F_1063_, v_Digest_1064_, v_inst_1065_, v_inst_1066_, v_lSkip_1067_, v_vk_1068_, v_publicValues_1069_, v_x_1070_, v_x_1071_, v_x_1072_, v_a_1073_);
lean_dec(v_publicValues_1069_);
lean_dec_ref(v_vk_1068_);
return v_res_1074_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg(lean_object* v_inst_1075_, lean_object* v_inst_1076_, lean_object* v_lSkip_1077_, lean_object* v_vk_1078_, lean_object* v_commonMainCommit_1079_, lean_object* v_traceVdata_1080_, lean_object* v_publicValues_1081_, lean_object* v_a_1082_){
_start:
{
lean_object* v_inner_1083_; lean_object* v_preHash_1084_; lean_object* v___x_1085_; lean_object* v_a_1086_; lean_object* v_snd_1087_; lean_object* v___x_1088_; lean_object* v_a_1089_; lean_object* v_snd_1090_; lean_object* v_perAir_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; 
v_inner_1083_ = lean_ctor_get(v_vk_1078_, 0);
lean_inc_ref(v_inner_1083_);
v_preHash_1084_ = lean_ctor_get(v_vk_1078_, 1);
lean_inc(v_preHash_1084_);
lean_dec_ref(v_vk_1078_);
lean_inc_ref_n(v_inst_1075_, 2);
lean_inc_ref_n(v_inst_1076_, 2);
v___x_1085_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(v_inst_1076_, v_inst_1075_, v_preHash_1084_, v_a_1082_);
v_a_1086_ = lean_ctor_get(v___x_1085_, 0);
lean_inc(v_a_1086_);
lean_dec_ref(v___x_1085_);
v_snd_1087_ = lean_ctor_get(v_a_1086_, 1);
lean_inc(v_snd_1087_);
lean_dec(v_a_1086_);
v___x_1088_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(v_inst_1076_, v_inst_1075_, v_commonMainCommit_1079_, v_snd_1087_);
v_a_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc(v_a_1089_);
lean_dec_ref(v___x_1088_);
v_snd_1090_ = lean_ctor_get(v_a_1089_, 1);
lean_inc(v_snd_1090_);
lean_dec(v_a_1089_);
v_perAir_1091_ = lean_ctor_get(v_inner_1083_, 1);
lean_inc(v_perAir_1091_);
lean_dec_ref(v_inner_1083_);
v___x_1092_ = lean_unsigned_to_nat(0u);
v___x_1093_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTailM___redArg(v_inst_1075_, v_inst_1076_, v_lSkip_1077_, v_publicValues_1081_, v___x_1092_, v_perAir_1091_, v_traceVdata_1080_, v_snd_1090_);
return v___x_1093_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg___boxed(lean_object* v_inst_1094_, lean_object* v_inst_1095_, lean_object* v_lSkip_1096_, lean_object* v_vk_1097_, lean_object* v_commonMainCommit_1098_, lean_object* v_traceVdata_1099_, lean_object* v_publicValues_1100_, lean_object* v_a_1101_){
_start:
{
lean_object* v_res_1102_; 
v_res_1102_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg(v_inst_1094_, v_inst_1095_, v_lSkip_1096_, v_vk_1097_, v_commonMainCommit_1098_, v_traceVdata_1099_, v_publicValues_1100_, v_a_1101_);
lean_dec(v_publicValues_1100_);
return v_res_1102_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM(lean_object* v_F_1103_, lean_object* v_Digest_1104_, lean_object* v_inst_1105_, lean_object* v_inst_1106_, lean_object* v_lSkip_1107_, lean_object* v_vk_1108_, lean_object* v_commonMainCommit_1109_, lean_object* v_traceVdata_1110_, lean_object* v_publicValues_1111_, lean_object* v_a_1112_){
_start:
{
lean_object* v___x_1113_; 
v___x_1113_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg(v_inst_1105_, v_inst_1106_, v_lSkip_1107_, v_vk_1108_, v_commonMainCommit_1109_, v_traceVdata_1110_, v_publicValues_1111_, v_a_1112_);
return v___x_1113_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___boxed(lean_object* v_F_1114_, lean_object* v_Digest_1115_, lean_object* v_inst_1116_, lean_object* v_inst_1117_, lean_object* v_lSkip_1118_, lean_object* v_vk_1119_, lean_object* v_commonMainCommit_1120_, lean_object* v_traceVdata_1121_, lean_object* v_publicValues_1122_, lean_object* v_a_1123_){
_start:
{
lean_object* v_res_1124_; 
v_res_1124_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM(v_F_1114_, v_Digest_1115_, v_inst_1116_, v_inst_1117_, v_lSkip_1118_, v_vk_1119_, v_commonMainCommit_1120_, v_traceVdata_1121_, v_publicValues_1122_, v_a_1123_);
lean_dec(v_publicValues_1122_);
return v_res_1124_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg(lean_object* v_inst_1140_, lean_object* v_inst_1141_, lean_object* v_inst_1142_, lean_object* v_inst_1143_, lean_object* v_inst_1144_, lean_object* v_inst_1145_, lean_object* v_inst_1146_, lean_object* v_fo_1147_, lean_object* v_algMap_1148_, lean_object* v_config_1149_, lean_object* v_vk_1150_, lean_object* v_00_u03c0_1151_, lean_object* v_transcript_1152_){
_start:
{
lean_object* v_inner_1155_; lean_object* v_params_1156_; lean_object* v_params_1157_; uint8_t v___x_1158_; 
v_inner_1155_ = lean_ctor_get(v_vk_1150_, 0);
v_params_1156_ = lean_ctor_get(v_config_1149_, 0);
v_params_1157_ = lean_ctor_get(v_inner_1155_, 0);
lean_inc_ref(v_params_1157_);
lean_inc_ref(v_params_1156_);
v___x_1158_ = lp_swirl_x2dfv_Fundamentals_Runtime_instDecidableEqSystemParams_decEq(v_params_1156_, v_params_1157_);
if (v___x_1158_ == 0)
{
lean_object* v___x_1159_; 
lean_dec_ref(v_transcript_1152_);
lean_dec_ref(v_00_u03c0_1151_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v___x_1159_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__1));
return v___x_1159_;
}
else
{
lean_object* v_commonMainCommit_1160_; lean_object* v_traceVdata_1161_; lean_object* v_publicValues_1162_; lean_object* v_gkrProof_1163_; lean_object* v_batchConstraintProof_1164_; lean_object* v_stackingProof_1165_; lean_object* v_whirProof_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; uint8_t v___x_1169_; 
v_commonMainCommit_1160_ = lean_ctor_get(v_00_u03c0_1151_, 0);
lean_inc(v_commonMainCommit_1160_);
v_traceVdata_1161_ = lean_ctor_get(v_00_u03c0_1151_, 1);
lean_inc(v_traceVdata_1161_);
v_publicValues_1162_ = lean_ctor_get(v_00_u03c0_1151_, 2);
lean_inc(v_publicValues_1162_);
v_gkrProof_1163_ = lean_ctor_get(v_00_u03c0_1151_, 3);
lean_inc_ref(v_gkrProof_1163_);
v_batchConstraintProof_1164_ = lean_ctor_get(v_00_u03c0_1151_, 4);
lean_inc_ref(v_batchConstraintProof_1164_);
v_stackingProof_1165_ = lean_ctor_get(v_00_u03c0_1151_, 5);
lean_inc_ref(v_stackingProof_1165_);
v_whirProof_1166_ = lean_ctor_get(v_00_u03c0_1151_, 6);
lean_inc_ref(v_whirProof_1166_);
v___x_1167_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_numPresentTraces___redArg(v_traceVdata_1161_);
v___x_1168_ = lean_unsigned_to_nat(0u);
v___x_1169_ = lean_nat_dec_eq(v___x_1167_, v___x_1168_);
lean_dec(v___x_1167_);
if (v___x_1169_ == 0)
{
lean_object* v___x_1170_; 
lean_inc_ref(v_vk_1150_);
v___x_1170_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_verify___redArg(v_vk_1150_, v_00_u03c0_1151_);
if (lean_obj_tag(v___x_1170_) == 0)
{
lean_object* v___x_1171_; 
lean_dec_ref_known(v___x_1170_, 1);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec_ref(v_gkrProof_1163_);
lean_dec(v_publicValues_1162_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_transcript_1152_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v___x_1171_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__2));
return v___x_1171_;
}
else
{
lean_object* v_a_1172_; lean_object* v_lSkip_1173_; lean_object* v_nStack_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; 
v_a_1172_ = lean_ctor_get(v___x_1170_, 0);
lean_inc(v_a_1172_);
lean_dec_ref_known(v___x_1170_, 1);
v_lSkip_1173_ = lean_ctor_get(v_params_1156_, 0);
v_nStack_1174_ = lean_ctor_get(v_params_1156_, 1);
lean_inc(v_traceVdata_1161_);
v___x_1175_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeTraceIdToAirId___redArg(v_vk_1150_, v_traceVdata_1161_);
lean_inc(v___x_1175_);
lean_inc(v_lSkip_1173_);
v___x_1176_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_traceHeightConstraintsHoldM___redArg(v_lSkip_1173_, v_vk_1150_, v_traceVdata_1161_, v___x_1175_);
if (lean_obj_tag(v___x_1176_) == 0)
{
lean_object* v___x_1177_; 
lean_dec_ref_known(v___x_1176_, 1);
lean_dec(v___x_1175_);
lean_dec(v_a_1172_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec_ref(v_gkrProof_1163_);
lean_dec(v_publicValues_1162_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_transcript_1152_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v___x_1177_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__2));
return v___x_1177_;
}
else
{
lean_object* v_a_1178_; uint8_t v___x_1179_; 
v_a_1178_ = lean_ctor_get(v___x_1176_, 0);
lean_inc(v_a_1178_);
lean_dec_ref_known(v___x_1176_, 1);
v___x_1179_ = lean_unbox(v_a_1178_);
lean_dec(v_a_1178_);
if (v___x_1179_ == 0)
{
lean_dec(v___x_1175_);
lean_dec(v_a_1172_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec_ref(v_gkrProof_1163_);
lean_dec(v_publicValues_1162_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_transcript_1152_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
goto v___jp_1153_;
}
else
{
if (v___x_1169_ == 0)
{
lean_object* v___x_1180_; 
lean_inc(v_traceVdata_1161_);
lean_inc(v_commonMainCommit_1160_);
lean_inc_ref(v_vk_1150_);
lean_inc(v_lSkip_1173_);
lean_inc_ref(v_inst_1144_);
lean_inc_ref(v_inst_1142_);
v___x_1180_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleM___redArg(v_inst_1142_, v_inst_1144_, v_lSkip_1173_, v_vk_1150_, v_commonMainCommit_1160_, v_traceVdata_1161_, v_publicValues_1162_, v_transcript_1152_);
if (lean_obj_tag(v___x_1180_) == 0)
{
lean_object* v_a_1181_; lean_object* v___x_1183_; uint8_t v_isShared_1184_; uint8_t v_isSharedCheck_1188_; 
lean_dec(v___x_1175_);
lean_dec(v_a_1172_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec_ref(v_gkrProof_1163_);
lean_dec(v_publicValues_1162_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v_a_1181_ = lean_ctor_get(v___x_1180_, 0);
v_isSharedCheck_1188_ = !lean_is_exclusive(v___x_1180_);
if (v_isSharedCheck_1188_ == 0)
{
v___x_1183_ = v___x_1180_;
v_isShared_1184_ = v_isSharedCheck_1188_;
goto v_resetjp_1182_;
}
else
{
lean_inc(v_a_1181_);
lean_dec(v___x_1180_);
v___x_1183_ = lean_box(0);
v_isShared_1184_ = v_isSharedCheck_1188_;
goto v_resetjp_1182_;
}
v_resetjp_1182_:
{
lean_object* v___x_1186_; 
if (v_isShared_1184_ == 0)
{
v___x_1186_ = v___x_1183_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1187_; 
v_reuseFailAlloc_1187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1187_, 0, v_a_1181_);
v___x_1186_ = v_reuseFailAlloc_1187_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
return v___x_1186_;
}
}
}
else
{
lean_object* v_a_1189_; lean_object* v_snd_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; 
v_a_1189_ = lean_ctor_get(v___x_1180_, 0);
lean_inc(v_a_1189_);
lean_dec_ref_known(v___x_1180_, 1);
v_snd_1190_ = lean_ctor_get(v_a_1189_, 1);
lean_inc(v_snd_1190_);
lean_dec(v_a_1189_);
lean_inc_n(v___x_1175_, 3);
lean_inc(v_lSkip_1173_);
v___x_1191_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNPerTrace___redArg(v_lSkip_1173_, v_traceVdata_1161_, v___x_1175_);
v___x_1192_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeNeedRotPerCommit___redArg(v_vk_1150_, v_traceVdata_1161_, v___x_1175_);
lean_inc_ref(v_batchConstraintProof_1164_);
lean_inc_ref(v_vk_1150_);
lean_inc(v_algMap_1148_);
lean_inc_ref(v_fo_1147_);
lean_inc_ref(v_inst_1143_);
lean_inc_ref(v_inst_1145_);
lean_inc_ref(v_inst_1144_);
lean_inc(v_inst_1141_);
lean_inc_ref(v_inst_1140_);
v___x_1193_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(v_inst_1140_, v_inst_1141_, v_inst_1144_, v_inst_1145_, v_inst_1143_, v_fo_1147_, v_algMap_1148_, v_snd_1190_, v_vk_1150_, v_publicValues_1162_, v_gkrProof_1163_, v_batchConstraintProof_1164_, v_traceVdata_1161_, v___x_1175_, v___x_1191_);
if (lean_obj_tag(v___x_1193_) == 0)
{
lean_object* v_a_1194_; lean_object* v___x_1196_; uint8_t v_isShared_1197_; uint8_t v_isSharedCheck_1201_; 
lean_dec(v___x_1192_);
lean_dec(v___x_1175_);
lean_dec(v_a_1172_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v_a_1194_ = lean_ctor_get(v___x_1193_, 0);
v_isSharedCheck_1201_ = !lean_is_exclusive(v___x_1193_);
if (v_isSharedCheck_1201_ == 0)
{
v___x_1196_ = v___x_1193_;
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
else
{
lean_inc(v_a_1194_);
lean_dec(v___x_1193_);
v___x_1196_ = lean_box(0);
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
v_resetjp_1195_:
{
lean_object* v___x_1199_; 
if (v_isShared_1197_ == 0)
{
v___x_1199_ = v___x_1196_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v_a_1194_);
v___x_1199_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
return v___x_1199_;
}
}
}
else
{
lean_object* v_a_1202_; lean_object* v_fst_1203_; lean_object* v_snd_1204_; lean_object* v_columnOpenings_1205_; lean_object* v___x_1206_; 
v_a_1202_ = lean_ctor_get(v___x_1193_, 0);
lean_inc(v_a_1202_);
lean_dec_ref_known(v___x_1193_, 1);
v_fst_1203_ = lean_ctor_get(v_a_1202_, 0);
lean_inc(v_fst_1203_);
v_snd_1204_ = lean_ctor_get(v_a_1202_, 1);
lean_inc(v_snd_1204_);
lean_dec(v_a_1202_);
v_columnOpenings_1205_ = lean_ctor_get(v_batchConstraintProof_1164_, 4);
lean_inc(v_columnOpenings_1205_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_inc(v_nStack_1174_);
lean_inc(v_lSkip_1173_);
lean_inc_ref(v_stackingProof_1165_);
lean_inc_ref(v_fo_1147_);
lean_inc_ref(v_inst_1143_);
lean_inc_ref(v_inst_1144_);
lean_inc_ref(v_inst_1140_);
lean_inc(v_inst_1141_);
v___x_1206_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify___redArg(v_inst_1141_, v_inst_1140_, v_inst_1144_, v_inst_1143_, v_fo_1147_, v_snd_1204_, v_stackingProof_1165_, v_a_1172_, v___x_1192_, v_lSkip_1173_, v_nStack_1174_, v_columnOpenings_1205_, v_fst_1203_);
if (lean_obj_tag(v___x_1206_) == 0)
{
lean_object* v_a_1207_; lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1214_; 
lean_dec(v___x_1175_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v_a_1207_ = lean_ctor_get(v___x_1206_, 0);
v_isSharedCheck_1214_ = !lean_is_exclusive(v___x_1206_);
if (v_isSharedCheck_1214_ == 0)
{
v___x_1209_ = v___x_1206_;
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
else
{
lean_inc(v_a_1207_);
lean_dec(v___x_1206_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1212_; 
if (v_isShared_1210_ == 0)
{
v___x_1212_ = v___x_1209_;
goto v_reusejp_1211_;
}
else
{
lean_object* v_reuseFailAlloc_1213_; 
v_reuseFailAlloc_1213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1213_, 0, v_a_1207_);
v___x_1212_ = v_reuseFailAlloc_1213_;
goto v_reusejp_1211_;
}
v_reusejp_1211_:
{
return v___x_1212_;
}
}
}
else
{
lean_object* v_a_1215_; lean_object* v_fst_1216_; lean_object* v_snd_1217_; lean_object* v___x_1218_; 
v_a_1215_ = lean_ctor_get(v___x_1206_, 0);
lean_inc(v_a_1215_);
lean_dec_ref_known(v___x_1206_, 1);
v_fst_1216_ = lean_ctor_get(v_a_1215_, 0);
lean_inc(v_fst_1216_);
v_snd_1217_ = lean_ctor_get(v_a_1215_, 1);
lean_inc(v_snd_1217_);
lean_dec(v_a_1215_);
lean_inc(v_lSkip_1173_);
lean_inc_ref(v_fo_1147_);
v___x_1218_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube___redArg(v_fo_1147_, v_lSkip_1173_, v_fst_1216_);
if (lean_obj_tag(v___x_1218_) == 0)
{
lean_object* v___x_1219_; 
lean_dec_ref_known(v___x_1218_, 1);
lean_dec(v_snd_1217_);
lean_dec(v___x_1175_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v___x_1219_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__3));
return v___x_1219_;
}
else
{
lean_object* v_a_1220_; lean_object* v_stackingOpenings_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; 
v_a_1220_ = lean_ctor_get(v___x_1218_, 0);
lean_inc(v_a_1220_);
lean_dec_ref_known(v___x_1218_, 1);
v_stackingOpenings_1221_ = lean_ctor_get(v_stackingProof_1165_, 2);
lean_inc(v_stackingOpenings_1221_);
lean_dec_ref(v_stackingProof_1165_);
v___x_1222_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits___redArg(v_vk_1150_, v_commonMainCommit_1160_, v_traceVdata_1161_, v___x_1175_);
lean_dec(v_traceVdata_1161_);
lean_dec_ref(v_vk_1150_);
v___x_1223_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Whir_verify___redArg(v_inst_1140_, v_inst_1141_, v_inst_1146_, v_inst_1143_, v_inst_1142_, v_inst_1144_, v_inst_1145_, v_fo_1147_, v_algMap_1148_, v_snd_1217_, v_config_1149_, v_whirProof_1166_, v_stackingOpenings_1221_, v___x_1222_, v_a_1220_);
return v___x_1223_;
}
}
}
}
}
else
{
lean_dec(v___x_1175_);
lean_dec(v_a_1172_);
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec_ref(v_gkrProof_1163_);
lean_dec(v_publicValues_1162_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_transcript_1152_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
goto v___jp_1153_;
}
}
}
}
}
else
{
lean_object* v___x_1224_; 
lean_dec_ref(v_whirProof_1166_);
lean_dec_ref(v_stackingProof_1165_);
lean_dec_ref(v_batchConstraintProof_1164_);
lean_dec_ref(v_gkrProof_1163_);
lean_dec(v_publicValues_1162_);
lean_dec(v_traceVdata_1161_);
lean_dec(v_commonMainCommit_1160_);
lean_dec_ref(v_transcript_1152_);
lean_dec_ref(v_00_u03c0_1151_);
lean_dec_ref(v_vk_1150_);
lean_dec_ref(v_config_1149_);
lean_dec(v_algMap_1148_);
lean_dec_ref(v_fo_1147_);
lean_dec_ref(v_inst_1146_);
lean_dec_ref(v_inst_1145_);
lean_dec_ref(v_inst_1144_);
lean_dec_ref(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec(v_inst_1141_);
lean_dec_ref(v_inst_1140_);
v___x_1224_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__4));
return v___x_1224_;
}
}
v___jp_1153_:
{
lean_object* v___x_1154_; 
v___x_1154_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg___closed__0));
return v___x_1154_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify(lean_object* v_F_1225_, lean_object* v_EF_1226_, lean_object* v_Digest_1227_, lean_object* v_inst_1228_, lean_object* v_inst_1229_, lean_object* v_inst_1230_, lean_object* v_inst_1231_, lean_object* v_inst_1232_, lean_object* v_inst_1233_, lean_object* v_inst_1234_, lean_object* v_fo_1235_, lean_object* v_algMap_1236_, lean_object* v_config_1237_, lean_object* v_vk_1238_, lean_object* v_00_u03c0_1239_, lean_object* v_transcript_1240_){
_start:
{
lean_object* v___x_1241_; 
v___x_1241_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_verify___redArg(v_inst_1228_, v_inst_1229_, v_inst_1230_, v_inst_1231_, v_inst_1232_, v_inst_1233_, v_inst_1234_, v_fo_1235_, v_algMap_1236_, v_config_1237_, v_vk_1238_, v_00_u03c0_1239_, v_transcript_1240_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries_spec__0___redArg(lean_object* v_x_1242_, lean_object* v_x_1243_){
_start:
{
if (lean_obj_tag(v_x_1243_) == 0)
{
return v_x_1242_;
}
else
{
lean_object* v_head_1244_; lean_object* v_tail_1245_; lean_object* v_fst_1246_; lean_object* v_snd_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; 
v_head_1244_ = lean_ctor_get(v_x_1243_, 0);
lean_inc(v_head_1244_);
v_tail_1245_ = lean_ctor_get(v_x_1243_, 1);
lean_inc(v_tail_1245_);
lean_dec_ref_known(v_x_1243_, 2);
v_fst_1246_ = lean_ctor_get(v_head_1244_, 0);
lean_inc(v_fst_1246_);
v_snd_1247_ = lean_ctor_get(v_head_1244_, 1);
lean_inc(v_snd_1247_);
lean_dec(v_head_1244_);
v___x_1248_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntriesForAir___redArg(v_fst_1246_, v_snd_1247_);
v___x_1249_ = l_List_appendTR___redArg(v_x_1242_, v___x_1248_);
v_x_1242_ = v___x_1249_;
v_x_1243_ = v_tail_1245_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries___redArg(lean_object* v_vk_1251_){
_start:
{
lean_object* v_inner_1252_; lean_object* v_perAir_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; 
v_inner_1252_ = lean_ctor_get(v_vk_1251_, 0);
lean_inc_ref(v_inner_1252_);
lean_dec_ref(v_vk_1251_);
v_perAir_1253_ = lean_ctor_get(v_inner_1252_, 1);
lean_inc(v_perAir_1253_);
lean_dec_ref(v_inner_1252_);
v___x_1254_ = lean_box(0);
v___x_1255_ = l_List_lengthTR___redArg(v_perAir_1253_);
v___x_1256_ = l_List_range(v___x_1255_);
v___x_1257_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1256_, v_perAir_1253_);
v___x_1258_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries_spec__0___redArg(v___x_1254_, v___x_1257_);
return v___x_1258_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries(lean_object* v_F_1259_, lean_object* v_Digest_1260_, lean_object* v_vk_1261_){
_start:
{
lean_object* v___x_1262_; 
v___x_1262_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries___redArg(v_vk_1261_);
return v___x_1262_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries_spec__0(lean_object* v_Digest_1263_, lean_object* v_F_1264_, lean_object* v_x_1265_, lean_object* v_x_1266_){
_start:
{
lean_object* v___x_1267_; 
v___x_1267_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries_spec__0___redArg(v_x_1265_, v_x_1266_);
return v___x_1267_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries_spec__0___redArg(lean_object* v_x_1268_, lean_object* v_x_1269_){
_start:
{
if (lean_obj_tag(v_x_1269_) == 0)
{
return v_x_1268_;
}
else
{
lean_object* v_head_1270_; lean_object* v_tail_1271_; lean_object* v_fst_1272_; lean_object* v_snd_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; 
v_head_1270_ = lean_ctor_get(v_x_1269_, 0);
lean_inc(v_head_1270_);
v_tail_1271_ = lean_ctor_get(v_x_1269_, 1);
lean_inc(v_tail_1271_);
lean_dec_ref_known(v_x_1269_, 2);
v_fst_1272_ = lean_ctor_get(v_head_1270_, 0);
lean_inc(v_fst_1272_);
v_snd_1273_ = lean_ctor_get(v_head_1270_, 1);
lean_inc(v_snd_1273_);
lean_dec(v_head_1270_);
v___x_1274_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntriesForAir___redArg(v_fst_1272_, v_snd_1273_);
v___x_1275_ = l_List_appendTR___redArg(v_x_1268_, v___x_1274_);
v_x_1268_ = v___x_1275_;
v_x_1269_ = v_tail_1271_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries___redArg(lean_object* v_traceVdata_1277_){
_start:
{
lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; 
v___x_1278_ = lean_box(0);
v___x_1279_ = l_List_lengthTR___redArg(v_traceVdata_1277_);
v___x_1280_ = l_List_range(v___x_1279_);
v___x_1281_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1280_, v_traceVdata_1277_);
v___x_1282_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries_spec__0___redArg(v___x_1278_, v___x_1281_);
return v___x_1282_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries(lean_object* v_Digest_1283_, lean_object* v_traceVdata_1284_){
_start:
{
lean_object* v___x_1285_; 
v___x_1285_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries___redArg(v_traceVdata_1284_);
return v___x_1285_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries_spec__0(lean_object* v_Digest_1286_, lean_object* v_x_1287_, lean_object* v_x_1288_){
_start:
{
lean_object* v___x_1289_; 
v___x_1289_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries_spec__0___redArg(v_x_1287_, v_x_1288_);
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorFromVerifyingKeyAndTraceData___redArg(lean_object* v_vk_1290_, lean_object* v_commonMainCommit_1291_, lean_object* v_traceVdata_1292_){
_start:
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; 
v___x_1293_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries___redArg(v_vk_1290_);
v___x_1294_ = lean_box(1);
v___x_1295_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1295_, 0, v___x_1294_);
lean_ctor_set(v___x_1295_, 1, v_commonMainCommit_1291_);
v___x_1296_ = lean_box(0);
v___x_1297_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1297_, 0, v___x_1295_);
lean_ctor_set(v___x_1297_, 1, v___x_1296_);
v___x_1298_ = l_List_appendTR___redArg(v___x_1293_, v___x_1297_);
v___x_1299_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries___redArg(v_traceVdata_1292_);
v___x_1300_ = l_List_appendTR___redArg(v___x_1298_, v___x_1299_);
return v___x_1300_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorFromVerifyingKeyAndTraceData(lean_object* v_F_1301_, lean_object* v_Digest_1302_, lean_object* v_vk_1303_, lean_object* v_commonMainCommit_1304_, lean_object* v_traceVdata_1305_){
_start:
{
lean_object* v___x_1306_; 
v___x_1306_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorFromVerifyingKeyAndTraceData___redArg(v_vk_1303_, v_commonMainCommit_1304_, v_traceVdata_1305_);
return v___x_1306_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootDigests___redArg(lean_object* v_vk_1307_){
_start:
{
lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; 
v___x_1308_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootEntries___redArg(v_vk_1307_);
v___x_1309_ = lean_box(0);
v___x_1310_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(v___x_1308_, v___x_1309_);
return v___x_1310_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootDigests(lean_object* v_F_1311_, lean_object* v_Digest_1312_, lean_object* v_vk_1313_){
_start:
{
lean_object* v___x_1314_; 
v___x_1314_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preprocessedInitialRootDigests___redArg(v_vk_1313_);
return v___x_1314_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootDigests___redArg(lean_object* v_traceVdata_1315_){
_start:
{
lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
v___x_1316_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootEntries___redArg(v_traceVdata_1315_);
v___x_1317_ = lean_box(0);
v___x_1318_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(v___x_1316_, v___x_1317_);
return v___x_1318_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootDigests(lean_object* v_Digest_1319_, lean_object* v_traceVdata_1320_){
_start:
{
lean_object* v___x_1321_; 
v___x_1321_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_cachedInitialRootDigests___redArg(v_traceVdata_1320_);
return v___x_1321_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorDigests___redArg(lean_object* v_vk_1322_, lean_object* v_commonMainCommit_1323_, lean_object* v_traceVdata_1324_){
_start:
{
lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; 
v___x_1325_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorFromVerifyingKeyAndTraceData___redArg(v_vk_1322_, v_commonMainCommit_1323_, v_traceVdata_1324_);
v___x_1326_ = lean_box(0);
v___x_1327_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_computeWhirCommits_spec__0___redArg(v___x_1325_, v___x_1326_);
return v___x_1327_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorDigests(lean_object* v_F_1328_, lean_object* v_Digest_1329_, lean_object* v_vk_1330_, lean_object* v_commonMainCommit_1331_, lean_object* v_traceVdata_1332_){
_start:
{
lean_object* v___x_1333_; 
v___x_1333_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorDigests___redArg(v_vk_1330_, v_commonMainCommit_1331_, v_traceVdata_1332_);
return v___x_1333_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg(lean_object* v_vk_1335_, lean_object* v_commonMainCommit_1336_, lean_object* v_traceVdata_1337_){
_start:
{
lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; 
v___x_1338_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg___closed__0));
v___x_1339_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorDigests___redArg(v_vk_1335_, v_commonMainCommit_1336_, v_traceVdata_1337_);
v___x_1340_ = lean_box(0);
v___x_1341_ = l_List_mapTR_loop___redArg(v___x_1338_, v___x_1339_, v___x_1340_);
return v___x_1341_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents(lean_object* v_F_1342_, lean_object* v_Digest_1343_, lean_object* v_inst_1344_, lean_object* v_vk_1345_, lean_object* v_commonMainCommit_1346_, lean_object* v_traceVdata_1347_){
_start:
{
lean_object* v___x_1348_; 
v___x_1348_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg(v_vk_1345_, v_commonMainCommit_1346_, v_traceVdata_1347_);
return v___x_1348_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___boxed(lean_object* v_F_1349_, lean_object* v_Digest_1350_, lean_object* v_inst_1351_, lean_object* v_vk_1352_, lean_object* v_commonMainCommit_1353_, lean_object* v_traceVdata_1354_){
_start:
{
lean_object* v_res_1355_; 
v_res_1355_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents(v_F_1349_, v_Digest_1350_, v_inst_1351_, v_vk_1352_, v_commonMainCommit_1353_, v_traceVdata_1354_);
lean_dec_ref(v_inst_1351_);
return v_res_1355_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg(lean_object* v_inst_1360_, lean_object* v_lSkip_1361_, lean_object* v_avk_1362_, lean_object* v_traceVData_1363_, lean_object* v_publicValues_1364_){
_start:
{
lean_object* v___y_1366_; lean_object* v_traceEvents_1367_; lean_object* v_preprocessedData_1374_; uint8_t v_isRequired_1375_; lean_object* v___y_1377_; 
v_preprocessedData_1374_ = lean_ctor_get(v_avk_1362_, 0);
v_isRequired_1375_ = lean_ctor_get_uint8(v_avk_1362_, sizeof(void*)*5);
if (v_isRequired_1375_ == 0)
{
lean_object* v_ofBool_1409_; uint8_t v___y_1411_; 
v_ofBool_1409_ = lean_ctor_get(v_inst_1360_, 0);
if (lean_obj_tag(v_traceVData_1363_) == 0)
{
v___y_1411_ = v_isRequired_1375_;
goto v___jp_1410_;
}
else
{
uint8_t v___x_1417_; 
v___x_1417_ = 1;
v___y_1411_ = v___x_1417_;
goto v___jp_1410_;
}
v___jp_1410_:
{
lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; 
v___x_1412_ = lean_box(v___y_1411_);
lean_inc(v_ofBool_1409_);
v___x_1413_ = lean_apply_1(v_ofBool_1409_, v___x_1412_);
v___x_1414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1414_, 0, v___x_1413_);
v___x_1415_ = lean_box(0);
v___x_1416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1416_, 0, v___x_1414_);
lean_ctor_set(v___x_1416_, 1, v___x_1415_);
v___y_1377_ = v___x_1416_;
goto v___jp_1376_;
}
}
else
{
lean_object* v___x_1418_; 
v___x_1418_ = lean_box(0);
v___y_1377_ = v___x_1418_;
goto v___jp_1376_;
}
v___jp_1365_:
{
lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; 
v___x_1368_ = l_List_appendTR___redArg(v___y_1366_, v_traceEvents_1367_);
v___x_1369_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__0));
v___x_1370_ = lean_box(0);
v___x_1371_ = l_List_mapTR_loop___redArg(v___x_1369_, v_publicValues_1364_, v___x_1370_);
v___x_1372_ = l_List_appendTR___redArg(v___x_1368_, v___x_1371_);
v___x_1373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1372_);
return v___x_1373_;
}
v___jp_1376_:
{
if (lean_obj_tag(v_traceVData_1363_) == 0)
{
lean_object* v___x_1378_; 
lean_dec(v_lSkip_1361_);
lean_dec_ref(v_inst_1360_);
v___x_1378_ = lean_box(0);
v___y_1366_ = v___y_1377_;
v_traceEvents_1367_ = v___x_1378_;
goto v___jp_1365_;
}
else
{
if (lean_obj_tag(v_preprocessedData_1374_) == 0)
{
lean_object* v_val_1379_; lean_object* v___x_1381_; uint8_t v_isShared_1382_; uint8_t v_isSharedCheck_1398_; 
lean_dec(v_lSkip_1361_);
v_val_1379_ = lean_ctor_get(v_traceVData_1363_, 0);
v_isSharedCheck_1398_ = !lean_is_exclusive(v_traceVData_1363_);
if (v_isSharedCheck_1398_ == 0)
{
v___x_1381_ = v_traceVData_1363_;
v_isShared_1382_ = v_isSharedCheck_1398_;
goto v_resetjp_1380_;
}
else
{
lean_inc(v_val_1379_);
lean_dec(v_traceVData_1363_);
v___x_1381_ = lean_box(0);
v_isShared_1382_ = v_isSharedCheck_1398_;
goto v_resetjp_1380_;
}
v_resetjp_1380_:
{
lean_object* v_ofNat_1383_; lean_object* v_logHeight_1384_; lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1396_; 
v_ofNat_1383_ = lean_ctor_get(v_inst_1360_, 1);
lean_inc(v_ofNat_1383_);
lean_dec_ref(v_inst_1360_);
v_logHeight_1384_ = lean_ctor_get(v_val_1379_, 0);
v_isSharedCheck_1396_ = !lean_is_exclusive(v_val_1379_);
if (v_isSharedCheck_1396_ == 0)
{
lean_object* v_unused_1397_; 
v_unused_1397_ = lean_ctor_get(v_val_1379_, 1);
lean_dec(v_unused_1397_);
v___x_1386_ = v_val_1379_;
v_isShared_1387_ = v_isSharedCheck_1396_;
goto v_resetjp_1385_;
}
else
{
lean_inc(v_logHeight_1384_);
lean_dec(v_val_1379_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1396_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v___x_1388_; lean_object* v___x_1390_; 
v___x_1388_ = lean_apply_1(v_ofNat_1383_, v_logHeight_1384_);
if (v_isShared_1382_ == 0)
{
lean_ctor_set_tag(v___x_1381_, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1388_);
v___x_1390_ = v___x_1381_;
goto v_reusejp_1389_;
}
else
{
lean_object* v_reuseFailAlloc_1395_; 
v_reuseFailAlloc_1395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1395_, 0, v___x_1388_);
v___x_1390_ = v_reuseFailAlloc_1395_;
goto v_reusejp_1389_;
}
v_reusejp_1389_:
{
lean_object* v___x_1391_; lean_object* v___x_1393_; 
v___x_1391_ = lean_box(0);
if (v_isShared_1387_ == 0)
{
lean_ctor_set_tag(v___x_1386_, 1);
lean_ctor_set(v___x_1386_, 1, v___x_1391_);
lean_ctor_set(v___x_1386_, 0, v___x_1390_);
v___x_1393_ = v___x_1386_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v___x_1390_);
lean_ctor_set(v_reuseFailAlloc_1394_, 1, v___x_1391_);
v___x_1393_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
v___y_1366_ = v___y_1377_;
v_traceEvents_1367_ = v___x_1393_;
goto v___jp_1365_;
}
}
}
}
}
else
{
lean_object* v_val_1399_; lean_object* v_val_1400_; lean_object* v_hypercubeDim_1401_; lean_object* v_logHeight_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; uint8_t v___x_1406_; 
lean_dec_ref(v_inst_1360_);
v_val_1399_ = lean_ctor_get(v_preprocessedData_1374_, 0);
v_val_1400_ = lean_ctor_get(v_traceVData_1363_, 0);
lean_inc(v_val_1400_);
lean_dec_ref_known(v_traceVData_1363_, 1);
v_hypercubeDim_1401_ = lean_ctor_get(v_val_1399_, 1);
v_logHeight_1402_ = lean_ctor_get(v_val_1400_, 0);
lean_inc(v_logHeight_1402_);
lean_dec(v_val_1400_);
v___x_1403_ = lean_nat_to_int(v_lSkip_1361_);
v___x_1404_ = lean_int_add(v_hypercubeDim_1401_, v___x_1403_);
lean_dec(v___x_1403_);
v___x_1405_ = lean_nat_to_int(v_logHeight_1402_);
v___x_1406_ = lean_int_dec_eq(v___x_1404_, v___x_1405_);
lean_dec(v___x_1405_);
lean_dec(v___x_1404_);
if (v___x_1406_ == 0)
{
lean_object* v___x_1407_; 
lean_dec(v___y_1377_);
lean_dec(v_publicValues_1364_);
v___x_1407_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___closed__1));
return v___x_1407_;
}
else
{
lean_object* v___x_1408_; 
v___x_1408_ = lean_box(0);
v___y_1366_ = v___y_1377_;
v_traceEvents_1367_ = v___x_1408_;
goto v___jp_1365_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg___boxed(lean_object* v_inst_1419_, lean_object* v_lSkip_1420_, lean_object* v_avk_1421_, lean_object* v_traceVData_1422_, lean_object* v_publicValues_1423_){
_start:
{
lean_object* v_res_1424_; 
v_res_1424_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg(v_inst_1419_, v_lSkip_1420_, v_avk_1421_, v_traceVData_1422_, v_publicValues_1423_);
lean_dec_ref(v_avk_1421_);
return v_res_1424_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir(lean_object* v_F_1425_, lean_object* v_Digest_1426_, lean_object* v_inst_1427_, lean_object* v_lSkip_1428_, lean_object* v_avk_1429_, lean_object* v_traceVData_1430_, lean_object* v_publicValues_1431_){
_start:
{
lean_object* v___x_1432_; 
v___x_1432_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg(v_inst_1427_, v_lSkip_1428_, v_avk_1429_, v_traceVData_1430_, v_publicValues_1431_);
return v___x_1432_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___boxed(lean_object* v_F_1433_, lean_object* v_Digest_1434_, lean_object* v_inst_1435_, lean_object* v_lSkip_1436_, lean_object* v_avk_1437_, lean_object* v_traceVData_1438_, lean_object* v_publicValues_1439_){
_start:
{
lean_object* v_res_1440_; 
v_res_1440_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir(v_F_1433_, v_Digest_1434_, v_inst_1435_, v_lSkip_1436_, v_avk_1437_, v_traceVData_1438_, v_publicValues_1439_);
lean_dec_ref(v_avk_1437_);
return v_res_1440_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg(lean_object* v_inst_1446_, lean_object* v_lSkip_1447_, lean_object* v_publicValues_1448_, lean_object* v_x_1449_, lean_object* v_x_1450_, lean_object* v_x_1451_){
_start:
{
if (lean_obj_tag(v_x_1450_) == 0)
{
lean_dec(v_x_1449_);
lean_dec(v_lSkip_1447_);
lean_dec_ref(v_inst_1446_);
if (lean_obj_tag(v_x_1451_) == 0)
{
lean_object* v___x_1454_; 
v___x_1454_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__1));
return v___x_1454_;
}
else
{
lean_dec(v_x_1451_);
goto v___jp_1452_;
}
}
else
{
if (lean_obj_tag(v_x_1451_) == 1)
{
lean_object* v_head_1455_; lean_object* v_tail_1456_; lean_object* v_head_1457_; lean_object* v_tail_1458_; lean_object* v___x_1459_; 
v_head_1455_ = lean_ctor_get(v_x_1450_, 0);
v_tail_1456_ = lean_ctor_get(v_x_1450_, 1);
v_head_1457_ = lean_ctor_get(v_x_1451_, 0);
lean_inc(v_head_1457_);
v_tail_1458_ = lean_ctor_get(v_x_1451_, 1);
lean_inc(v_tail_1458_);
lean_dec_ref_known(v_x_1451_, 2);
lean_inc(v_x_1449_);
v___x_1459_ = l_List_get_x3fInternal___redArg(v_publicValues_1448_, v_x_1449_);
if (lean_obj_tag(v___x_1459_) == 0)
{
lean_object* v___x_1460_; 
lean_dec(v_tail_1458_);
lean_dec(v_head_1457_);
lean_dec(v_x_1449_);
lean_dec(v_lSkip_1447_);
lean_dec_ref(v_inst_1446_);
v___x_1460_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__0));
return v___x_1460_;
}
else
{
lean_object* v_val_1461_; lean_object* v___x_1462_; 
v_val_1461_ = lean_ctor_get(v___x_1459_, 0);
lean_inc(v_val_1461_);
lean_dec_ref_known(v___x_1459_, 1);
lean_inc(v_lSkip_1447_);
lean_inc_ref(v_inst_1446_);
v___x_1462_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_preambleEventsForAir___redArg(v_inst_1446_, v_lSkip_1447_, v_head_1455_, v_head_1457_, v_val_1461_);
if (lean_obj_tag(v___x_1462_) == 0)
{
lean_dec(v_tail_1458_);
lean_dec(v_x_1449_);
lean_dec(v_lSkip_1447_);
lean_dec_ref(v_inst_1446_);
return v___x_1462_;
}
else
{
lean_object* v_a_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; 
v_a_1463_ = lean_ctor_get(v___x_1462_, 0);
lean_inc(v_a_1463_);
lean_dec_ref_known(v___x_1462_, 1);
v___x_1464_ = lean_unsigned_to_nat(1u);
v___x_1465_ = lean_nat_add(v_x_1449_, v___x_1464_);
lean_dec(v_x_1449_);
v___x_1466_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg(v_inst_1446_, v_lSkip_1447_, v_publicValues_1448_, v___x_1465_, v_tail_1456_, v_tail_1458_);
if (lean_obj_tag(v___x_1466_) == 0)
{
lean_dec(v_a_1463_);
return v___x_1466_;
}
else
{
lean_object* v_a_1467_; lean_object* v___x_1469_; uint8_t v_isShared_1470_; uint8_t v_isSharedCheck_1475_; 
v_a_1467_ = lean_ctor_get(v___x_1466_, 0);
v_isSharedCheck_1475_ = !lean_is_exclusive(v___x_1466_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1469_ = v___x_1466_;
v_isShared_1470_ = v_isSharedCheck_1475_;
goto v_resetjp_1468_;
}
else
{
lean_inc(v_a_1467_);
lean_dec(v___x_1466_);
v___x_1469_ = lean_box(0);
v_isShared_1470_ = v_isSharedCheck_1475_;
goto v_resetjp_1468_;
}
v_resetjp_1468_:
{
lean_object* v___x_1471_; lean_object* v___x_1473_; 
v___x_1471_ = l_List_appendTR___redArg(v_a_1463_, v_a_1467_);
if (v_isShared_1470_ == 0)
{
lean_ctor_set(v___x_1469_, 0, v___x_1471_);
v___x_1473_ = v___x_1469_;
goto v_reusejp_1472_;
}
else
{
lean_object* v_reuseFailAlloc_1474_; 
v_reuseFailAlloc_1474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1474_, 0, v___x_1471_);
v___x_1473_ = v_reuseFailAlloc_1474_;
goto v_reusejp_1472_;
}
v_reusejp_1472_:
{
return v___x_1473_;
}
}
}
}
}
}
else
{
lean_dec(v_x_1451_);
lean_dec(v_x_1449_);
lean_dec(v_lSkip_1447_);
lean_dec_ref(v_inst_1446_);
goto v___jp_1452_;
}
}
v___jp_1452_:
{
lean_object* v___x_1453_; 
v___x_1453_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___closed__0));
return v___x_1453_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg___boxed(lean_object* v_inst_1476_, lean_object* v_lSkip_1477_, lean_object* v_publicValues_1478_, lean_object* v_x_1479_, lean_object* v_x_1480_, lean_object* v_x_1481_){
_start:
{
lean_object* v_res_1482_; 
v_res_1482_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg(v_inst_1476_, v_lSkip_1477_, v_publicValues_1478_, v_x_1479_, v_x_1480_, v_x_1481_);
lean_dec(v_x_1480_);
lean_dec(v_publicValues_1478_);
return v_res_1482_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail(lean_object* v_F_1483_, lean_object* v_Digest_1484_, lean_object* v_inst_1485_, lean_object* v_lSkip_1486_, lean_object* v_vk_1487_, lean_object* v_publicValues_1488_, lean_object* v_x_1489_, lean_object* v_x_1490_, lean_object* v_x_1491_){
_start:
{
lean_object* v___x_1492_; 
v___x_1492_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg(v_inst_1485_, v_lSkip_1486_, v_publicValues_1488_, v_x_1489_, v_x_1490_, v_x_1491_);
return v___x_1492_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___boxed(lean_object* v_F_1493_, lean_object* v_Digest_1494_, lean_object* v_inst_1495_, lean_object* v_lSkip_1496_, lean_object* v_vk_1497_, lean_object* v_publicValues_1498_, lean_object* v_x_1499_, lean_object* v_x_1500_, lean_object* v_x_1501_){
_start:
{
lean_object* v_res_1502_; 
v_res_1502_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail(v_F_1493_, v_Digest_1494_, v_inst_1495_, v_lSkip_1496_, v_vk_1497_, v_publicValues_1498_, v_x_1499_, v_x_1500_, v_x_1501_);
lean_dec(v_x_1500_);
lean_dec(v_publicValues_1498_);
lean_dec_ref(v_vk_1497_);
return v_res_1502_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___redArg(lean_object* v_inst_1503_, lean_object* v_lSkip_1504_, lean_object* v_vk_1505_, lean_object* v_commonMainCommit_1506_, lean_object* v_traceVdata_1507_, lean_object* v_publicValues_1508_){
_start:
{
lean_object* v_inner_1509_; lean_object* v_preHash_1510_; lean_object* v_perAir_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; 
v_inner_1509_ = lean_ctor_get(v_vk_1505_, 0);
v_preHash_1510_ = lean_ctor_get(v_vk_1505_, 1);
lean_inc(v_preHash_1510_);
v_perAir_1511_ = lean_ctor_get(v_inner_1509_, 1);
v___x_1512_ = lean_unsigned_to_nat(0u);
lean_inc(v_traceVdata_1507_);
v___x_1513_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreambleTail___redArg(v_inst_1503_, v_lSkip_1504_, v_publicValues_1508_, v___x_1512_, v_perAir_1511_, v_traceVdata_1507_);
if (lean_obj_tag(v___x_1513_) == 0)
{
lean_dec(v_preHash_1510_);
lean_dec(v_traceVdata_1507_);
lean_dec(v_commonMainCommit_1506_);
lean_dec_ref(v_vk_1505_);
return v___x_1513_;
}
else
{
lean_object* v_a_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1527_; 
v_a_1514_ = lean_ctor_get(v___x_1513_, 0);
v_isSharedCheck_1527_ = !lean_is_exclusive(v___x_1513_);
if (v_isSharedCheck_1527_ == 0)
{
v___x_1516_ = v___x_1513_;
v_isShared_1517_ = v_isSharedCheck_1527_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_a_1514_);
lean_dec(v___x_1513_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1527_;
goto v_resetjp_1515_;
}
v_resetjp_1515_:
{
lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1525_; 
v___x_1518_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_initialRootVectorTranscriptEvents___redArg(v_vk_1505_, v_commonMainCommit_1506_, v_traceVdata_1507_);
v___x_1519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1519_, 0, v_preHash_1510_);
v___x_1520_ = lean_box(0);
v___x_1521_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1521_, 0, v___x_1519_);
lean_ctor_set(v___x_1521_, 1, v___x_1520_);
v___x_1522_ = l_List_appendTR___redArg(v___x_1521_, v___x_1518_);
v___x_1523_ = l_List_appendTR___redArg(v___x_1522_, v_a_1514_);
if (v_isShared_1517_ == 0)
{
lean_ctor_set(v___x_1516_, 0, v___x_1523_);
v___x_1525_ = v___x_1516_;
goto v_reusejp_1524_;
}
else
{
lean_object* v_reuseFailAlloc_1526_; 
v_reuseFailAlloc_1526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1526_, 0, v___x_1523_);
v___x_1525_ = v_reuseFailAlloc_1526_;
goto v_reusejp_1524_;
}
v_reusejp_1524_:
{
return v___x_1525_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___redArg___boxed(lean_object* v_inst_1528_, lean_object* v_lSkip_1529_, lean_object* v_vk_1530_, lean_object* v_commonMainCommit_1531_, lean_object* v_traceVdata_1532_, lean_object* v_publicValues_1533_){
_start:
{
lean_object* v_res_1534_; 
v_res_1534_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___redArg(v_inst_1528_, v_lSkip_1529_, v_vk_1530_, v_commonMainCommit_1531_, v_traceVdata_1532_, v_publicValues_1533_);
lean_dec(v_publicValues_1533_);
return v_res_1534_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble(lean_object* v_F_1535_, lean_object* v_Digest_1536_, lean_object* v_inst_1537_, lean_object* v_lSkip_1538_, lean_object* v_vk_1539_, lean_object* v_commonMainCommit_1540_, lean_object* v_traceVdata_1541_, lean_object* v_publicValues_1542_){
_start:
{
lean_object* v___x_1543_; 
v___x_1543_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___redArg(v_inst_1537_, v_lSkip_1538_, v_vk_1539_, v_commonMainCommit_1540_, v_traceVdata_1541_, v_publicValues_1542_);
return v___x_1543_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble___boxed(lean_object* v_F_1544_, lean_object* v_Digest_1545_, lean_object* v_inst_1546_, lean_object* v_lSkip_1547_, lean_object* v_vk_1548_, lean_object* v_commonMainCommit_1549_, lean_object* v_traceVdata_1550_, lean_object* v_publicValues_1551_){
_start:
{
lean_object* v_res_1552_; 
v_res_1552_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_observePreamble(v_F_1544_, v_Digest_1545_, v_inst_1546_, v_lSkip_1547_, v_vk_1548_, v_commonMainCommit_1549_, v_traceVdata_1550_, v_publicValues_1551_);
lean_dec(v_publicValues_1551_);
return v_res_1552_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Core(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Ops(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Stacking(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Whir(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Main(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Ops(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Stacking(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Whir(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
