// Lean compiler output
// Module: Fundamentals.Spec.Runtime.Core
// Imports: public import Init public meta import Init
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
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_shiftl(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* l_Except_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_pure(lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_ExtensionEncoding_extToWords___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_ExtensionEncoding_extToWords(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observe_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observeCommit_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observeCommit_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Fundamentals.Runtime.TranscriptEvent.observe"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__2_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3;
static lean_once_cell_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "Fundamentals.Runtime.TranscriptEvent.observeCommit"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__6_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorbWords___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorbWords(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvent___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeExt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeCommit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeCommit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_prepareForSqueeze___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_prepareForSqueeze(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqVerifierError(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqVerifierError___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Fundamentals.Runtime.VerifierError.traceHeightsTooLarge"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__1_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 67, .m_capacity = 67, .m_length = 66, .m_data = "Fundamentals.Runtime.VerifierError.preprocessedTraceHeightMismatch"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "Fundamentals.Runtime.VerifierError.emptyTraces"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__5_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "Fundamentals.Runtime.VerifierError.proofShapeError"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__6_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__7_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Fundamentals.Runtime.VerifierError.challengeDerivationError"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__9_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Fundamentals.Runtime.VerifierError.batchConstraintError"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__10_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__10_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__11_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Fundamentals.Runtime.VerifierError.stackedReductionError"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__12_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__12_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__13_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Fundamentals.Runtime.VerifierError.invalidPrismPoint"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__14_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__14_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__15_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Fundamentals.Runtime.VerifierError.whirError"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__16_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__17_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Fundamentals.Runtime.VerifierError.systemParamsMismatch"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__18_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__18_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__19 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__19_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_init___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_init(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__3_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__4_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__5_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__5_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__6_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__7_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__7_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__10_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__4, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__11_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__7, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__12_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__9, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__13_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_map, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__14_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__15_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_pure, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__15_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__16_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__11_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__12_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__17_value;
static const lean_closure_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_bind, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__18_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__17_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__18_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__19 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBase___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBase(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_ExtensionEncoding_extToWords___redArg(lean_object* v_inst_1_, lean_object* v_e_2_){
_start:
{
lean_object* v_extWordCount_3_; lean_object* v_extToWord_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v_extWordCount_3_ = lean_ctor_get(v_inst_1_, 0);
lean_inc(v_extWordCount_3_);
v_extToWord_4_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_extToWord_4_);
lean_dec_ref(v_inst_1_);
v___x_5_ = lean_apply_1(v_extToWord_4_, v_e_2_);
v___x_6_ = l_List_finRange(v_extWordCount_3_);
v___x_7_ = lean_box(0);
v___x_8_ = l_List_mapTR_loop___redArg(v___x_5_, v___x_6_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_ExtensionEncoding_extToWords(lean_object* v_F_9_, lean_object* v_EF_10_, lean_object* v_inst_11_, lean_object* v_e_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_ExtensionEncoding_extToWords___redArg(v_inst_11_, v_e_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___redArg(lean_object* v_x_14_){
_start:
{
if (lean_obj_tag(v_x_14_) == 0)
{
lean_object* v___x_15_; 
v___x_15_ = lean_unsigned_to_nat(0u);
return v___x_15_;
}
else
{
lean_object* v___x_16_; 
v___x_16_ = lean_unsigned_to_nat(1u);
return v___x_16_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___redArg___boxed(lean_object* v_x_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___redArg(v_x_17_);
lean_dec_ref(v_x_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx(lean_object* v_F_19_, lean_object* v_Digest_20_, lean_object* v_x_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___redArg(v_x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx___boxed(lean_object* v_F_23_, lean_object* v_Digest_24_, lean_object* v_x_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorIdx(v_F_23_, v_Digest_24_, v_x_25_);
lean_dec_ref(v_x_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(lean_object* v_t_27_, lean_object* v_k_28_){
_start:
{
lean_object* v_value_29_; lean_object* v___x_30_; 
v_value_29_ = lean_ctor_get(v_t_27_, 0);
lean_inc(v_value_29_);
lean_dec_ref(v_t_27_);
v___x_30_ = lean_apply_1(v_k_28_, v_value_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim(lean_object* v_F_31_, lean_object* v_Digest_32_, lean_object* v_motive_33_, lean_object* v_ctorIdx_34_, lean_object* v_t_35_, lean_object* v_h_36_, lean_object* v_k_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(v_t_35_, v_k_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___boxed(lean_object* v_F_39_, lean_object* v_Digest_40_, lean_object* v_motive_41_, lean_object* v_ctorIdx_42_, lean_object* v_t_43_, lean_object* v_h_44_, lean_object* v_k_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim(v_F_39_, v_Digest_40_, v_motive_41_, v_ctorIdx_42_, v_t_43_, v_h_44_, v_k_45_);
lean_dec(v_ctorIdx_42_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observe_elim___redArg(lean_object* v_t_47_, lean_object* v_observe_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(v_t_47_, v_observe_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observe_elim(lean_object* v_F_50_, lean_object* v_Digest_51_, lean_object* v_motive_52_, lean_object* v_t_53_, lean_object* v_h_54_, lean_object* v_observe_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(v_t_53_, v_observe_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observeCommit_elim___redArg(lean_object* v_t_57_, lean_object* v_observeCommit_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(v_t_57_, v_observeCommit_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_observeCommit_elim(lean_object* v_F_60_, lean_object* v_Digest_61_, lean_object* v_motive_62_, lean_object* v_t_63_, lean_object* v_h_64_, lean_object* v_observeCommit_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptEvent_ctorElim___redArg(v_t_63_, v_observeCommit_65_);
return v___x_66_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg(lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_x_69_, lean_object* v_x_70_){
_start:
{
if (lean_obj_tag(v_x_69_) == 0)
{
lean_dec_ref(v_inst_68_);
if (lean_obj_tag(v_x_70_) == 0)
{
lean_object* v_value_71_; lean_object* v_value_72_; lean_object* v___x_73_; uint8_t v___x_74_; 
v_value_71_ = lean_ctor_get(v_x_69_, 0);
lean_inc(v_value_71_);
lean_dec_ref_known(v_x_69_, 1);
v_value_72_ = lean_ctor_get(v_x_70_, 0);
lean_inc(v_value_72_);
lean_dec_ref_known(v_x_70_, 1);
v___x_73_ = lean_apply_2(v_inst_67_, v_value_71_, v_value_72_);
v___x_74_ = lean_unbox(v___x_73_);
return v___x_74_;
}
else
{
uint8_t v___x_75_; 
lean_dec_ref_known(v_x_70_, 1);
lean_dec_ref_known(v_x_69_, 1);
lean_dec_ref(v_inst_67_);
v___x_75_ = 0;
return v___x_75_;
}
}
else
{
lean_dec_ref(v_inst_67_);
if (lean_obj_tag(v_x_70_) == 0)
{
uint8_t v___x_76_; 
lean_dec_ref_known(v_x_70_, 1);
lean_dec_ref_known(v_x_69_, 1);
lean_dec_ref(v_inst_68_);
v___x_76_ = 0;
return v___x_76_;
}
else
{
lean_object* v_digest_77_; lean_object* v_digest_78_; lean_object* v___x_79_; uint8_t v___x_80_; 
v_digest_77_ = lean_ctor_get(v_x_69_, 0);
lean_inc(v_digest_77_);
lean_dec_ref_known(v_x_69_, 1);
v_digest_78_ = lean_ctor_get(v_x_70_, 0);
lean_inc(v_digest_78_);
lean_dec_ref_known(v_x_70_, 1);
v___x_79_ = lean_apply_2(v_inst_68_, v_digest_77_, v_digest_78_);
v___x_80_ = lean_unbox(v___x_79_);
return v___x_80_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg___boxed(lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_x_83_, lean_object* v_x_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg(v_inst_81_, v_inst_82_, v_x_83_, v_x_84_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq(lean_object* v_F_87_, lean_object* v_Digest_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_x_91_, lean_object* v_x_92_){
_start:
{
uint8_t v___x_93_; 
v___x_93_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg(v_inst_89_, v_inst_90_, v_x_91_, v_x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___boxed(lean_object* v_F_94_, lean_object* v_Digest_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_x_98_, lean_object* v_x_99_){
_start:
{
uint8_t v_res_100_; lean_object* v_r_101_; 
v_res_100_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq(v_F_94_, v_Digest_95_, v_inst_96_, v_inst_97_, v_x_98_, v_x_99_);
v_r_101_ = lean_box(v_res_100_);
return v_r_101_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_x_104_, lean_object* v_x_105_){
_start:
{
uint8_t v___x_106_; 
v___x_106_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg(v_inst_102_, v_inst_103_, v_x_104_, v_x_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___redArg___boxed(lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_x_109_, lean_object* v_x_110_){
_start:
{
uint8_t v_res_111_; lean_object* v_r_112_; 
v_res_111_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___redArg(v_inst_107_, v_inst_108_, v_x_109_, v_x_110_);
v_r_112_ = lean_box(v_res_111_);
return v_r_112_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent(lean_object* v_F_113_, lean_object* v_Digest_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_x_117_, lean_object* v_x_118_){
_start:
{
uint8_t v___x_119_; 
v___x_119_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent_decEq___redArg(v_inst_115_, v_inst_116_, v_x_117_, v_x_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent___boxed(lean_object* v_F_120_, lean_object* v_Digest_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_x_124_, lean_object* v_x_125_){
_start:
{
uint8_t v_res_126_; lean_object* v_r_127_; 
v_res_126_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqTranscriptEvent(v_F_120_, v_Digest_121_, v_inst_122_, v_inst_123_, v_x_124_, v_x_125_);
v_r_127_ = lean_box(v_res_126_);
return v_r_127_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = lean_unsigned_to_nat(2u);
v___x_135_ = lean_nat_to_int(v___x_134_);
return v___x_135_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_136_ = lean_unsigned_to_nat(1u);
v___x_137_ = lean_nat_to_int(v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg(lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_x_146_, lean_object* v_prec_147_){
_start:
{
if (lean_obj_tag(v_x_146_) == 0)
{
lean_object* v_value_148_; lean_object* v___y_150_; lean_object* v___x_159_; uint8_t v___x_160_; 
lean_dec_ref(v_inst_145_);
v_value_148_ = lean_ctor_get(v_x_146_, 0);
lean_inc(v_value_148_);
lean_dec_ref_known(v_x_146_, 1);
v___x_159_ = lean_unsigned_to_nat(1024u);
v___x_160_ = lean_nat_dec_le(v___x_159_, v_prec_147_);
if (v___x_160_ == 0)
{
lean_object* v___x_161_; 
v___x_161_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_150_ = v___x_161_;
goto v___jp_149_;
}
else
{
lean_object* v___x_162_; 
v___x_162_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_150_ = v___x_162_;
goto v___jp_149_;
}
v___jp_149_:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; uint8_t v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_151_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__2));
v___x_152_ = lean_unsigned_to_nat(1024u);
v___x_153_ = lean_apply_2(v_inst_144_, v_value_148_, v___x_152_);
v___x_154_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_151_);
lean_ctor_set(v___x_154_, 1, v___x_153_);
lean_inc(v___y_150_);
v___x_155_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_155_, 0, v___y_150_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = 0;
v___x_157_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_157_, 0, v___x_155_);
lean_ctor_set_uint8(v___x_157_, sizeof(void*)*1, v___x_156_);
v___x_158_ = l_Repr_addAppParen(v___x_157_, v_prec_147_);
return v___x_158_;
}
}
else
{
lean_object* v_digest_163_; lean_object* v___y_165_; lean_object* v___x_174_; uint8_t v___x_175_; 
lean_dec_ref(v_inst_144_);
v_digest_163_ = lean_ctor_get(v_x_146_, 0);
lean_inc(v_digest_163_);
lean_dec_ref_known(v_x_146_, 1);
v___x_174_ = lean_unsigned_to_nat(1024u);
v___x_175_ = lean_nat_dec_le(v___x_174_, v_prec_147_);
if (v___x_175_ == 0)
{
lean_object* v___x_176_; 
v___x_176_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_165_ = v___x_176_;
goto v___jp_164_;
}
else
{
lean_object* v___x_177_; 
v___x_177_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_165_ = v___x_177_;
goto v___jp_164_;
}
v___jp_164_:
{
lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; uint8_t v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_166_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__7));
v___x_167_ = lean_unsigned_to_nat(1024u);
v___x_168_ = lean_apply_2(v_inst_145_, v_digest_163_, v___x_167_);
v___x_169_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_169_, 0, v___x_166_);
lean_ctor_set(v___x_169_, 1, v___x_168_);
lean_inc(v___y_165_);
v___x_170_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_170_, 0, v___y_165_);
lean_ctor_set(v___x_170_, 1, v___x_169_);
v___x_171_ = 0;
v___x_172_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_172_, 0, v___x_170_);
lean_ctor_set_uint8(v___x_172_, sizeof(void*)*1, v___x_171_);
v___x_173_ = l_Repr_addAppParen(v___x_172_, v_prec_147_);
return v___x_173_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___boxed(lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_x_180_, lean_object* v_prec_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg(v_inst_178_, v_inst_179_, v_x_180_, v_prec_181_);
lean_dec(v_prec_181_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr(lean_object* v_F_183_, lean_object* v_Digest_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_x_187_, lean_object* v_prec_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg(v_inst_185_, v_inst_186_, v_x_187_, v_prec_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___boxed(lean_object* v_F_190_, lean_object* v_Digest_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_x_194_, lean_object* v_prec_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr(v_F_190_, v_Digest_191_, v_inst_192_, v_inst_193_, v_x_194_, v_prec_195_);
lean_dec(v_prec_195_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent___redArg(lean_object* v_inst_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___boxed), 6, 4);
lean_closure_set(v___x_199_, 0, lean_box(0));
lean_closure_set(v___x_199_, 1, lean_box(0));
lean_closure_set(v___x_199_, 2, v_inst_197_);
lean_closure_set(v___x_199_, 3, v_inst_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent(lean_object* v_F_200_, lean_object* v_Digest_201_, lean_object* v_inst_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___boxed), 6, 4);
lean_closure_set(v___x_204_, 0, lean_box(0));
lean_closure_set(v___x_204_, 1, lean_box(0));
lean_closure_set(v___x_204_, 2, v_inst_202_);
lean_closure_set(v___x_204_, 3, v_inst_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg___lam__0(lean_object* v_zero_205_, lean_object* v_x_206_){
_start:
{
lean_inc(v_zero_205_);
return v_zero_205_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg___lam__0___boxed(lean_object* v_zero_207_, lean_object* v_x_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg___lam__0(v_zero_207_, v_x_208_);
lean_dec(v_x_208_);
lean_dec(v_zero_207_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg(lean_object* v_cfg_210_){
_start:
{
lean_object* v_width_211_; lean_object* v_zero_212_; lean_object* v___f_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v_width_211_ = lean_ctor_get(v_cfg_210_, 0);
lean_inc(v_width_211_);
v_zero_212_ = lean_ctor_get(v_cfg_210_, 2);
lean_inc(v_zero_212_);
lean_dec_ref(v_cfg_210_);
v___f_213_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_213_, 0, v_zero_212_);
v___x_214_ = l_Array_ofFn___redArg(v_width_211_, v___f_213_);
v___x_215_ = lean_unsigned_to_nat(0u);
v___x_216_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_216_, 0, v___x_214_);
lean_ctor_set(v___x_216_, 1, v___x_215_);
lean_ctor_set(v___x_216_, 2, v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init(lean_object* v_F_217_, lean_object* v_cfg_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg(v_cfg_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg___lam__0(lean_object* v_absorbIdx_220_, lean_object* v_state_221_, lean_object* v_value_222_, lean_object* v_idx_223_){
_start:
{
uint8_t v___x_224_; 
v___x_224_ = lean_nat_dec_eq(v_idx_223_, v_absorbIdx_220_);
if (v___x_224_ == 0)
{
lean_object* v___x_225_; 
v___x_225_ = lean_array_fget_borrowed(v_state_221_, v_idx_223_);
lean_inc(v___x_225_);
return v___x_225_;
}
else
{
lean_inc(v_value_222_);
return v_value_222_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg___lam__0___boxed(lean_object* v_absorbIdx_226_, lean_object* v_state_227_, lean_object* v_value_228_, lean_object* v_idx_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg___lam__0(v_absorbIdx_226_, v_state_227_, v_value_228_, v_idx_229_);
lean_dec(v_idx_229_);
lean_dec(v_value_228_);
lean_dec_ref(v_state_227_);
lean_dec(v_absorbIdx_226_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg(lean_object* v_cfg_231_, lean_object* v_sponge_232_, lean_object* v_value_233_){
_start:
{
lean_object* v_width_234_; lean_object* v_rate_235_; lean_object* v_permute_236_; lean_object* v_state_237_; lean_object* v_absorbIdx_238_; lean_object* v_sampleIdx_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_256_; 
v_width_234_ = lean_ctor_get(v_cfg_231_, 0);
lean_inc(v_width_234_);
v_rate_235_ = lean_ctor_get(v_cfg_231_, 1);
lean_inc(v_rate_235_);
v_permute_236_ = lean_ctor_get(v_cfg_231_, 3);
lean_inc_ref(v_permute_236_);
lean_dec_ref(v_cfg_231_);
v_state_237_ = lean_ctor_get(v_sponge_232_, 0);
v_absorbIdx_238_ = lean_ctor_get(v_sponge_232_, 1);
v_sampleIdx_239_ = lean_ctor_get(v_sponge_232_, 2);
v_isSharedCheck_256_ = !lean_is_exclusive(v_sponge_232_);
if (v_isSharedCheck_256_ == 0)
{
v___x_241_ = v_sponge_232_;
v_isShared_242_ = v_isSharedCheck_256_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_sampleIdx_239_);
lean_inc(v_absorbIdx_238_);
lean_inc(v_state_237_);
lean_dec(v_sponge_232_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_256_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___f_243_; lean_object* v_overwritten_244_; lean_object* v___x_245_; lean_object* v___x_246_; uint8_t v___x_247_; 
lean_inc(v_absorbIdx_238_);
v___f_243_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_243_, 0, v_absorbIdx_238_);
lean_closure_set(v___f_243_, 1, v_state_237_);
lean_closure_set(v___f_243_, 2, v_value_233_);
v_overwritten_244_ = l_Array_ofFn___redArg(v_width_234_, v___f_243_);
v___x_245_ = lean_unsigned_to_nat(1u);
v___x_246_ = lean_nat_add(v_absorbIdx_238_, v___x_245_);
lean_dec(v_absorbIdx_238_);
v___x_247_ = lean_nat_dec_eq(v___x_246_, v_rate_235_);
if (v___x_247_ == 0)
{
lean_object* v___x_249_; 
lean_dec_ref(v_permute_236_);
lean_dec(v_rate_235_);
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 1, v___x_246_);
lean_ctor_set(v___x_241_, 0, v_overwritten_244_);
v___x_249_ = v___x_241_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_overwritten_244_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_250_, 2, v_sampleIdx_239_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
else
{
lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_254_; 
lean_dec(v___x_246_);
lean_dec(v_sampleIdx_239_);
v___x_251_ = lean_apply_1(v_permute_236_, v_overwritten_244_);
v___x_252_ = lean_unsigned_to_nat(0u);
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 2, v_rate_235_);
lean_ctor_set(v___x_241_, 1, v___x_252_);
lean_ctor_set(v___x_241_, 0, v___x_251_);
v___x_254_ = v___x_241_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v___x_251_);
lean_ctor_set(v_reuseFailAlloc_255_, 1, v___x_252_);
lean_ctor_set(v_reuseFailAlloc_255_, 2, v_rate_235_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb(lean_object* v_F_257_, lean_object* v_cfg_258_, lean_object* v_sponge_259_, lean_object* v_value_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg(v_cfg_258_, v_sponge_259_, v_value_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(lean_object* v_cfg_262_, lean_object* v_x_263_, lean_object* v_x_264_){
_start:
{
if (lean_obj_tag(v_x_264_) == 0)
{
lean_dec_ref(v_cfg_262_);
return v_x_263_;
}
else
{
lean_object* v_head_265_; lean_object* v_tail_266_; lean_object* v___x_267_; 
v_head_265_ = lean_ctor_get(v_x_264_, 0);
lean_inc(v_head_265_);
v_tail_266_ = lean_ctor_get(v_x_264_, 1);
lean_inc(v_tail_266_);
lean_dec_ref_known(v_x_264_, 2);
lean_inc_ref(v_cfg_262_);
v___x_267_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg(v_cfg_262_, v_x_263_, v_head_265_);
v_x_263_ = v___x_267_;
v_x_264_ = v_tail_266_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorbWords___redArg(lean_object* v_cfg_269_, lean_object* v_sponge_270_, lean_object* v_values_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(v_cfg_269_, v_sponge_270_, v_values_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorbWords(lean_object* v_F_273_, lean_object* v_cfg_274_, lean_object* v_sponge_275_, lean_object* v_values_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(v_cfg_274_, v_sponge_275_, v_values_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0(lean_object* v_F_278_, lean_object* v_cfg_279_, lean_object* v_x_280_, lean_object* v_x_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(v_cfg_279_, v_x_280_, v_x_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvent___redArg(lean_object* v_cfg_283_, lean_object* v_inst_284_, lean_object* v_sponge_285_, lean_object* v_event_286_){
_start:
{
if (lean_obj_tag(v_event_286_) == 0)
{
lean_object* v_value_287_; lean_object* v___x_288_; 
lean_dec_ref(v_inst_284_);
v_value_287_ = lean_ctor_get(v_event_286_, 0);
lean_inc(v_value_287_);
lean_dec_ref_known(v_event_286_, 1);
v___x_288_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg(v_cfg_283_, v_sponge_285_, v_value_287_);
return v___x_288_;
}
else
{
lean_object* v_digest_289_; lean_object* v_digestWords_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v_digest_289_ = lean_ctor_get(v_event_286_, 0);
lean_inc(v_digest_289_);
lean_dec_ref_known(v_event_286_, 1);
v_digestWords_290_ = lean_ctor_get(v_inst_284_, 2);
lean_inc_ref(v_digestWords_290_);
lean_dec_ref(v_inst_284_);
v___x_291_ = lean_apply_1(v_digestWords_290_, v_digest_289_);
v___x_292_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(v_cfg_283_, v_sponge_285_, v___x_291_);
return v___x_292_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvent(lean_object* v_F_293_, lean_object* v_Digest_294_, lean_object* v_cfg_295_, lean_object* v_inst_296_, lean_object* v_sponge_297_, lean_object* v_event_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvent___redArg(v_cfg_295_, v_inst_296_, v_sponge_297_, v_event_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents___redArg___lam__0(lean_object* v_cfg_300_, lean_object* v_inst_301_, lean_object* v_current_302_, lean_object* v_event_303_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvent___redArg(v_cfg_300_, v_inst_301_, v_current_302_, v_event_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents___redArg(lean_object* v_cfg_305_, lean_object* v_inst_306_, lean_object* v_sponge_307_, lean_object* v_events_308_){
_start:
{
lean_object* v___f_309_; lean_object* v___x_310_; 
v___f_309_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents___redArg___lam__0), 4, 2);
lean_closure_set(v___f_309_, 0, v_cfg_305_);
lean_closure_set(v___f_309_, 1, v_inst_306_);
v___x_310_ = l_List_foldl___redArg(v___f_309_, v_sponge_307_, v_events_308_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents(lean_object* v_F_311_, lean_object* v_Digest_312_, lean_object* v_cfg_313_, lean_object* v_inst_314_, lean_object* v_sponge_315_, lean_object* v_events_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeEvents___redArg(v_cfg_313_, v_inst_314_, v_sponge_315_, v_events_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeExt___redArg(lean_object* v_cfg_318_, lean_object* v_inst_319_, lean_object* v_sponge_320_, lean_object* v_value_321_){
_start:
{
lean_object* v_extWordCount_322_; lean_object* v_extToWord_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v_extWordCount_322_ = lean_ctor_get(v_inst_319_, 0);
lean_inc(v_extWordCount_322_);
v_extToWord_323_ = lean_ctor_get(v_inst_319_, 1);
lean_inc(v_extToWord_323_);
lean_dec_ref(v_inst_319_);
v___x_324_ = lean_apply_1(v_extToWord_323_, v_value_321_);
v___x_325_ = l_List_ofFn___redArg(v_extWordCount_322_, v___x_324_);
v___x_326_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(v_cfg_318_, v_sponge_320_, v___x_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeExt(lean_object* v_F_327_, lean_object* v_EF_328_, lean_object* v_cfg_329_, lean_object* v_inst_330_, lean_object* v_sponge_331_, lean_object* v_value_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeExt___redArg(v_cfg_329_, v_inst_330_, v_sponge_331_, v_value_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeCommit___redArg(lean_object* v_cfg_334_, lean_object* v_inst_335_, lean_object* v_sponge_336_, lean_object* v_digest_337_){
_start:
{
lean_object* v_digestWords_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v_digestWords_338_ = lean_ctor_get(v_inst_335_, 2);
lean_inc_ref(v_digestWords_338_);
lean_dec_ref(v_inst_335_);
v___x_339_ = lean_apply_1(v_digestWords_338_, v_digest_337_);
v___x_340_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Fundamentals_Runtime_DuplexSponge_absorbWords_spec__0___redArg(v_cfg_334_, v_sponge_336_, v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeCommit(lean_object* v_F_341_, lean_object* v_Digest_342_, lean_object* v_cfg_343_, lean_object* v_inst_344_, lean_object* v_sponge_345_, lean_object* v_digest_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeCommit___redArg(v_cfg_343_, v_inst_344_, v_sponge_345_, v_digest_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_prepareForSqueeze___redArg(lean_object* v_cfg_348_, lean_object* v_sponge_349_){
_start:
{
lean_object* v_state_350_; lean_object* v_absorbIdx_351_; lean_object* v_sampleIdx_352_; lean_object* v___x_353_; uint8_t v___x_359_; 
v_state_350_ = lean_ctor_get(v_sponge_349_, 0);
v_absorbIdx_351_ = lean_ctor_get(v_sponge_349_, 1);
v_sampleIdx_352_ = lean_ctor_get(v_sponge_349_, 2);
v___x_353_ = lean_unsigned_to_nat(0u);
v___x_359_ = lean_nat_dec_eq(v_absorbIdx_351_, v___x_353_);
if (v___x_359_ == 0)
{
lean_inc_ref(v_state_350_);
lean_dec_ref(v_sponge_349_);
goto v___jp_354_;
}
else
{
uint8_t v___x_360_; 
v___x_360_ = lean_nat_dec_eq(v_sampleIdx_352_, v___x_353_);
if (v___x_360_ == 0)
{
lean_dec_ref(v_cfg_348_);
return v_sponge_349_;
}
else
{
lean_inc_ref(v_state_350_);
lean_dec_ref(v_sponge_349_);
goto v___jp_354_;
}
}
v___jp_354_:
{
lean_object* v_rate_355_; lean_object* v_permute_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v_rate_355_ = lean_ctor_get(v_cfg_348_, 1);
lean_inc(v_rate_355_);
v_permute_356_ = lean_ctor_get(v_cfg_348_, 3);
lean_inc_ref(v_permute_356_);
lean_dec_ref(v_cfg_348_);
v___x_357_ = lean_apply_1(v_permute_356_, v_state_350_);
v___x_358_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v___x_353_);
lean_ctor_set(v___x_358_, 2, v_rate_355_);
return v___x_358_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_prepareForSqueeze(lean_object* v_F_361_, lean_object* v_cfg_362_, lean_object* v_sponge_363_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_prepareForSqueeze___redArg(v_cfg_362_, v_sponge_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample___redArg(lean_object* v_cfg_365_, lean_object* v_sponge_366_){
_start:
{
lean_object* v_prepared_367_; lean_object* v_state_368_; lean_object* v_absorbIdx_369_; lean_object* v_sampleIdx_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_381_; 
v_prepared_367_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_prepareForSqueeze___redArg(v_cfg_365_, v_sponge_366_);
v_state_368_ = lean_ctor_get(v_prepared_367_, 0);
v_absorbIdx_369_ = lean_ctor_get(v_prepared_367_, 1);
v_sampleIdx_370_ = lean_ctor_get(v_prepared_367_, 2);
v_isSharedCheck_381_ = !lean_is_exclusive(v_prepared_367_);
if (v_isSharedCheck_381_ == 0)
{
v___x_372_ = v_prepared_367_;
v_isShared_373_ = v_isSharedCheck_381_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_sampleIdx_370_);
lean_inc(v_absorbIdx_369_);
lean_inc(v_state_368_);
lean_dec(v_prepared_367_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_381_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_374_; lean_object* v_nextSampleIdx_375_; lean_object* v_value_376_; lean_object* v_nextSponge_378_; 
v___x_374_ = lean_unsigned_to_nat(1u);
v_nextSampleIdx_375_ = lean_nat_sub(v_sampleIdx_370_, v___x_374_);
lean_dec(v_sampleIdx_370_);
v_value_376_ = lean_array_fget(v_state_368_, v_nextSampleIdx_375_);
if (v_isShared_373_ == 0)
{
lean_ctor_set(v___x_372_, 2, v_nextSampleIdx_375_);
v_nextSponge_378_ = v___x_372_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_state_368_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v_absorbIdx_369_);
lean_ctor_set(v_reuseFailAlloc_380_, 2, v_nextSampleIdx_375_);
v_nextSponge_378_ = v_reuseFailAlloc_380_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
lean_object* v___x_379_; 
v___x_379_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_379_, 0, v_value_376_);
lean_ctor_set(v___x_379_, 1, v_nextSponge_378_);
return v___x_379_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample(lean_object* v_F_382_, lean_object* v_cfg_383_, lean_object* v_sponge_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample___redArg(v_cfg_383_, v_sponge_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg(lean_object* v_cfg_386_, lean_object* v_count_387_, lean_object* v_sponge_388_){
_start:
{
lean_object* v_zero_389_; uint8_t v_isZero_390_; 
v_zero_389_ = lean_unsigned_to_nat(0u);
v_isZero_390_ = lean_nat_dec_eq(v_count_387_, v_zero_389_);
if (v_isZero_390_ == 1)
{
lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec_ref(v_cfg_386_);
v___x_391_ = lean_box(0);
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
lean_ctor_set(v___x_392_, 1, v_sponge_388_);
return v___x_392_;
}
else
{
lean_object* v___x_393_; lean_object* v_fst_394_; lean_object* v_snd_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_414_; 
lean_inc_ref(v_cfg_386_);
v___x_393_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample___redArg(v_cfg_386_, v_sponge_388_);
v_fst_394_ = lean_ctor_get(v___x_393_, 0);
v_snd_395_ = lean_ctor_get(v___x_393_, 1);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_414_ == 0)
{
v___x_397_ = v___x_393_;
v_isShared_398_ = v_isSharedCheck_414_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_snd_395_);
lean_inc(v_fst_394_);
lean_dec(v___x_393_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_414_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v_one_399_; lean_object* v_n_400_; lean_object* v___x_401_; lean_object* v_fst_402_; lean_object* v_snd_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_413_; 
v_one_399_ = lean_unsigned_to_nat(1u);
v_n_400_ = lean_nat_sub(v_count_387_, v_one_399_);
v___x_401_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg(v_cfg_386_, v_n_400_, v_snd_395_);
lean_dec(v_n_400_);
v_fst_402_ = lean_ctor_get(v___x_401_, 0);
v_snd_403_ = lean_ctor_get(v___x_401_, 1);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_413_ == 0)
{
v___x_405_ = v___x_401_;
v_isShared_406_ = v_isSharedCheck_413_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_snd_403_);
lean_inc(v_fst_402_);
lean_dec(v___x_401_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_413_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_408_; 
if (v_isShared_398_ == 0)
{
lean_ctor_set_tag(v___x_397_, 1);
lean_ctor_set(v___x_397_, 1, v_fst_402_);
v___x_408_ = v___x_397_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_fst_394_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v_fst_402_);
v___x_408_ = v_reuseFailAlloc_412_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
lean_object* v___x_410_; 
if (v_isShared_406_ == 0)
{
lean_ctor_set(v___x_405_, 0, v___x_408_);
v___x_410_ = v___x_405_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v___x_408_);
lean_ctor_set(v_reuseFailAlloc_411_, 1, v_snd_403_);
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
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg___boxed(lean_object* v_cfg_415_, lean_object* v_count_416_, lean_object* v_sponge_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg(v_cfg_415_, v_count_416_, v_sponge_417_);
lean_dec(v_count_416_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords(lean_object* v_F_419_, lean_object* v_cfg_420_, lean_object* v_count_421_, lean_object* v_sponge_422_){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg(v_cfg_420_, v_count_421_, v_sponge_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___boxed(lean_object* v_F_424_, lean_object* v_cfg_425_, lean_object* v_count_426_, lean_object* v_sponge_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords(v_F_424_, v_cfg_425_, v_count_426_, v_sponge_427_);
lean_dec(v_count_426_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg___lam__0(lean_object* v_fst_429_, lean_object* v_i_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = l_List_get___redArg(v_fst_429_, v_i_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg___lam__0___boxed(lean_object* v_fst_432_, lean_object* v_i_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg___lam__0(v_fst_432_, v_i_433_);
lean_dec(v_fst_432_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg(lean_object* v_cfg_435_, lean_object* v_inst_436_, lean_object* v_sponge_437_){
_start:
{
lean_object* v_extWordCount_438_; lean_object* v_extOfWords_439_; lean_object* v___x_440_; lean_object* v_fst_441_; lean_object* v_snd_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_451_; 
v_extWordCount_438_ = lean_ctor_get(v_inst_436_, 0);
lean_inc(v_extWordCount_438_);
v_extOfWords_439_ = lean_ctor_get(v_inst_436_, 2);
lean_inc(v_extOfWords_439_);
lean_dec_ref(v_inst_436_);
v___x_440_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleWords___redArg(v_cfg_435_, v_extWordCount_438_, v_sponge_437_);
lean_dec(v_extWordCount_438_);
v_fst_441_ = lean_ctor_get(v___x_440_, 0);
v_snd_442_ = lean_ctor_get(v___x_440_, 1);
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_440_);
if (v_isSharedCheck_451_ == 0)
{
v___x_444_ = v___x_440_;
v_isShared_445_ = v_isSharedCheck_451_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_snd_442_);
lean_inc(v_fst_441_);
lean_dec(v___x_440_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_451_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___f_446_; lean_object* v_sampledExt_447_; lean_object* v___x_449_; 
v___f_446_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_446_, 0, v_fst_441_);
v_sampledExt_447_ = lean_apply_1(v_extOfWords_439_, v___f_446_);
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 0, v_sampledExt_447_);
v___x_449_ = v___x_444_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_sampledExt_447_);
lean_ctor_set(v_reuseFailAlloc_450_, 1, v_snd_442_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt(lean_object* v_F_452_, lean_object* v_EF_453_, lean_object* v_cfg_454_, lean_object* v_inst_455_, lean_object* v_sponge_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg(v_cfg_454_, v_inst_455_, v_sponge_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx(uint8_t v_x_458_){
_start:
{
switch(v_x_458_)
{
case 0:
{
lean_object* v___x_459_; 
v___x_459_ = lean_unsigned_to_nat(0u);
return v___x_459_;
}
case 1:
{
lean_object* v___x_460_; 
v___x_460_ = lean_unsigned_to_nat(1u);
return v___x_460_;
}
case 2:
{
lean_object* v___x_461_; 
v___x_461_ = lean_unsigned_to_nat(2u);
return v___x_461_;
}
case 3:
{
lean_object* v___x_462_; 
v___x_462_ = lean_unsigned_to_nat(3u);
return v___x_462_;
}
case 4:
{
lean_object* v___x_463_; 
v___x_463_ = lean_unsigned_to_nat(4u);
return v___x_463_;
}
case 5:
{
lean_object* v___x_464_; 
v___x_464_ = lean_unsigned_to_nat(5u);
return v___x_464_;
}
case 6:
{
lean_object* v___x_465_; 
v___x_465_ = lean_unsigned_to_nat(6u);
return v___x_465_;
}
case 7:
{
lean_object* v___x_466_; 
v___x_466_ = lean_unsigned_to_nat(7u);
return v___x_466_;
}
case 8:
{
lean_object* v___x_467_; 
v___x_467_ = lean_unsigned_to_nat(8u);
return v___x_467_;
}
default: 
{
lean_object* v___x_468_; 
v___x_468_ = lean_unsigned_to_nat(9u);
return v___x_468_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx___boxed(lean_object* v_x_469_){
_start:
{
uint8_t v_x_boxed_470_; lean_object* v_res_471_; 
v_x_boxed_470_ = lean_unbox(v_x_469_);
v_res_471_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx(v_x_boxed_470_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___redArg(lean_object* v_k_472_){
_start:
{
lean_inc(v_k_472_);
return v_k_472_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___redArg___boxed(lean_object* v_k_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___redArg(v_k_473_);
lean_dec(v_k_473_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim(lean_object* v_motive_475_, lean_object* v_ctorIdx_476_, uint8_t v_t_477_, lean_object* v_h_478_, lean_object* v_k_479_){
_start:
{
lean_inc(v_k_479_);
return v_k_479_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim___boxed(lean_object* v_motive_480_, lean_object* v_ctorIdx_481_, lean_object* v_t_482_, lean_object* v_h_483_, lean_object* v_k_484_){
_start:
{
uint8_t v_t_boxed_485_; lean_object* v_res_486_; 
v_t_boxed_485_ = lean_unbox(v_t_482_);
v_res_486_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorElim(v_motive_480_, v_ctorIdx_481_, v_t_boxed_485_, v_h_483_, v_k_484_);
lean_dec(v_k_484_);
lean_dec(v_ctorIdx_481_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___redArg(lean_object* v_traceHeightsTooLarge_487_){
_start:
{
lean_inc(v_traceHeightsTooLarge_487_);
return v_traceHeightsTooLarge_487_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___redArg___boxed(lean_object* v_traceHeightsTooLarge_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___redArg(v_traceHeightsTooLarge_488_);
lean_dec(v_traceHeightsTooLarge_488_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim(lean_object* v_motive_490_, uint8_t v_t_491_, lean_object* v_h_492_, lean_object* v_traceHeightsTooLarge_493_){
_start:
{
lean_inc(v_traceHeightsTooLarge_493_);
return v_traceHeightsTooLarge_493_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim___boxed(lean_object* v_motive_494_, lean_object* v_t_495_, lean_object* v_h_496_, lean_object* v_traceHeightsTooLarge_497_){
_start:
{
uint8_t v_t_boxed_498_; lean_object* v_res_499_; 
v_t_boxed_498_ = lean_unbox(v_t_495_);
v_res_499_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_traceHeightsTooLarge_elim(v_motive_494_, v_t_boxed_498_, v_h_496_, v_traceHeightsTooLarge_497_);
lean_dec(v_traceHeightsTooLarge_497_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___redArg(lean_object* v_preprocessedTraceHeightMismatch_500_){
_start:
{
lean_inc(v_preprocessedTraceHeightMismatch_500_);
return v_preprocessedTraceHeightMismatch_500_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___redArg___boxed(lean_object* v_preprocessedTraceHeightMismatch_501_){
_start:
{
lean_object* v_res_502_; 
v_res_502_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___redArg(v_preprocessedTraceHeightMismatch_501_);
lean_dec(v_preprocessedTraceHeightMismatch_501_);
return v_res_502_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim(lean_object* v_motive_503_, uint8_t v_t_504_, lean_object* v_h_505_, lean_object* v_preprocessedTraceHeightMismatch_506_){
_start:
{
lean_inc(v_preprocessedTraceHeightMismatch_506_);
return v_preprocessedTraceHeightMismatch_506_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim___boxed(lean_object* v_motive_507_, lean_object* v_t_508_, lean_object* v_h_509_, lean_object* v_preprocessedTraceHeightMismatch_510_){
_start:
{
uint8_t v_t_boxed_511_; lean_object* v_res_512_; 
v_t_boxed_511_ = lean_unbox(v_t_508_);
v_res_512_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_preprocessedTraceHeightMismatch_elim(v_motive_507_, v_t_boxed_511_, v_h_509_, v_preprocessedTraceHeightMismatch_510_);
lean_dec(v_preprocessedTraceHeightMismatch_510_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___redArg(lean_object* v_emptyTraces_513_){
_start:
{
lean_inc(v_emptyTraces_513_);
return v_emptyTraces_513_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___redArg___boxed(lean_object* v_emptyTraces_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___redArg(v_emptyTraces_514_);
lean_dec(v_emptyTraces_514_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim(lean_object* v_motive_516_, uint8_t v_t_517_, lean_object* v_h_518_, lean_object* v_emptyTraces_519_){
_start:
{
lean_inc(v_emptyTraces_519_);
return v_emptyTraces_519_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim___boxed(lean_object* v_motive_520_, lean_object* v_t_521_, lean_object* v_h_522_, lean_object* v_emptyTraces_523_){
_start:
{
uint8_t v_t_boxed_524_; lean_object* v_res_525_; 
v_t_boxed_524_ = lean_unbox(v_t_521_);
v_res_525_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_emptyTraces_elim(v_motive_520_, v_t_boxed_524_, v_h_522_, v_emptyTraces_523_);
lean_dec(v_emptyTraces_523_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___redArg(lean_object* v_proofShapeError_526_){
_start:
{
lean_inc(v_proofShapeError_526_);
return v_proofShapeError_526_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___redArg___boxed(lean_object* v_proofShapeError_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___redArg(v_proofShapeError_527_);
lean_dec(v_proofShapeError_527_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim(lean_object* v_motive_529_, uint8_t v_t_530_, lean_object* v_h_531_, lean_object* v_proofShapeError_532_){
_start:
{
lean_inc(v_proofShapeError_532_);
return v_proofShapeError_532_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim___boxed(lean_object* v_motive_533_, lean_object* v_t_534_, lean_object* v_h_535_, lean_object* v_proofShapeError_536_){
_start:
{
uint8_t v_t_boxed_537_; lean_object* v_res_538_; 
v_t_boxed_537_ = lean_unbox(v_t_534_);
v_res_538_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_proofShapeError_elim(v_motive_533_, v_t_boxed_537_, v_h_535_, v_proofShapeError_536_);
lean_dec(v_proofShapeError_536_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___redArg(lean_object* v_challengeDerivationError_539_){
_start:
{
lean_inc(v_challengeDerivationError_539_);
return v_challengeDerivationError_539_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___redArg___boxed(lean_object* v_challengeDerivationError_540_){
_start:
{
lean_object* v_res_541_; 
v_res_541_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___redArg(v_challengeDerivationError_540_);
lean_dec(v_challengeDerivationError_540_);
return v_res_541_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim(lean_object* v_motive_542_, uint8_t v_t_543_, lean_object* v_h_544_, lean_object* v_challengeDerivationError_545_){
_start:
{
lean_inc(v_challengeDerivationError_545_);
return v_challengeDerivationError_545_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim___boxed(lean_object* v_motive_546_, lean_object* v_t_547_, lean_object* v_h_548_, lean_object* v_challengeDerivationError_549_){
_start:
{
uint8_t v_t_boxed_550_; lean_object* v_res_551_; 
v_t_boxed_550_ = lean_unbox(v_t_547_);
v_res_551_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_challengeDerivationError_elim(v_motive_546_, v_t_boxed_550_, v_h_548_, v_challengeDerivationError_549_);
lean_dec(v_challengeDerivationError_549_);
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___redArg(lean_object* v_batchConstraintError_552_){
_start:
{
lean_inc(v_batchConstraintError_552_);
return v_batchConstraintError_552_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___redArg___boxed(lean_object* v_batchConstraintError_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___redArg(v_batchConstraintError_553_);
lean_dec(v_batchConstraintError_553_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim(lean_object* v_motive_555_, uint8_t v_t_556_, lean_object* v_h_557_, lean_object* v_batchConstraintError_558_){
_start:
{
lean_inc(v_batchConstraintError_558_);
return v_batchConstraintError_558_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim___boxed(lean_object* v_motive_559_, lean_object* v_t_560_, lean_object* v_h_561_, lean_object* v_batchConstraintError_562_){
_start:
{
uint8_t v_t_boxed_563_; lean_object* v_res_564_; 
v_t_boxed_563_ = lean_unbox(v_t_560_);
v_res_564_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_batchConstraintError_elim(v_motive_559_, v_t_boxed_563_, v_h_561_, v_batchConstraintError_562_);
lean_dec(v_batchConstraintError_562_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___redArg(lean_object* v_stackedReductionError_565_){
_start:
{
lean_inc(v_stackedReductionError_565_);
return v_stackedReductionError_565_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___redArg___boxed(lean_object* v_stackedReductionError_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___redArg(v_stackedReductionError_566_);
lean_dec(v_stackedReductionError_566_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim(lean_object* v_motive_568_, uint8_t v_t_569_, lean_object* v_h_570_, lean_object* v_stackedReductionError_571_){
_start:
{
lean_inc(v_stackedReductionError_571_);
return v_stackedReductionError_571_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim___boxed(lean_object* v_motive_572_, lean_object* v_t_573_, lean_object* v_h_574_, lean_object* v_stackedReductionError_575_){
_start:
{
uint8_t v_t_boxed_576_; lean_object* v_res_577_; 
v_t_boxed_576_ = lean_unbox(v_t_573_);
v_res_577_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_stackedReductionError_elim(v_motive_572_, v_t_boxed_576_, v_h_574_, v_stackedReductionError_575_);
lean_dec(v_stackedReductionError_575_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___redArg(lean_object* v_invalidPrismPoint_578_){
_start:
{
lean_inc(v_invalidPrismPoint_578_);
return v_invalidPrismPoint_578_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___redArg___boxed(lean_object* v_invalidPrismPoint_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___redArg(v_invalidPrismPoint_579_);
lean_dec(v_invalidPrismPoint_579_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim(lean_object* v_motive_581_, uint8_t v_t_582_, lean_object* v_h_583_, lean_object* v_invalidPrismPoint_584_){
_start:
{
lean_inc(v_invalidPrismPoint_584_);
return v_invalidPrismPoint_584_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim___boxed(lean_object* v_motive_585_, lean_object* v_t_586_, lean_object* v_h_587_, lean_object* v_invalidPrismPoint_588_){
_start:
{
uint8_t v_t_boxed_589_; lean_object* v_res_590_; 
v_t_boxed_589_ = lean_unbox(v_t_586_);
v_res_590_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_invalidPrismPoint_elim(v_motive_585_, v_t_boxed_589_, v_h_587_, v_invalidPrismPoint_588_);
lean_dec(v_invalidPrismPoint_588_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___redArg(lean_object* v_whirError_591_){
_start:
{
lean_inc(v_whirError_591_);
return v_whirError_591_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___redArg___boxed(lean_object* v_whirError_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___redArg(v_whirError_592_);
lean_dec(v_whirError_592_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim(lean_object* v_motive_594_, uint8_t v_t_595_, lean_object* v_h_596_, lean_object* v_whirError_597_){
_start:
{
lean_inc(v_whirError_597_);
return v_whirError_597_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim___boxed(lean_object* v_motive_598_, lean_object* v_t_599_, lean_object* v_h_600_, lean_object* v_whirError_601_){
_start:
{
uint8_t v_t_boxed_602_; lean_object* v_res_603_; 
v_t_boxed_602_ = lean_unbox(v_t_599_);
v_res_603_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_whirError_elim(v_motive_598_, v_t_boxed_602_, v_h_600_, v_whirError_601_);
lean_dec(v_whirError_601_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___redArg(lean_object* v_systemParamsMismatch_604_){
_start:
{
lean_inc(v_systemParamsMismatch_604_);
return v_systemParamsMismatch_604_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___redArg___boxed(lean_object* v_systemParamsMismatch_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___redArg(v_systemParamsMismatch_605_);
lean_dec(v_systemParamsMismatch_605_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim(lean_object* v_motive_607_, uint8_t v_t_608_, lean_object* v_h_609_, lean_object* v_systemParamsMismatch_610_){
_start:
{
lean_inc(v_systemParamsMismatch_610_);
return v_systemParamsMismatch_610_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim___boxed(lean_object* v_motive_611_, lean_object* v_t_612_, lean_object* v_h_613_, lean_object* v_systemParamsMismatch_614_){
_start:
{
uint8_t v_t_boxed_615_; lean_object* v_res_616_; 
v_t_boxed_615_ = lean_unbox(v_t_612_);
v_res_616_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_systemParamsMismatch_elim(v_motive_611_, v_t_boxed_615_, v_h_613_, v_systemParamsMismatch_614_);
lean_dec(v_systemParamsMismatch_614_);
return v_res_616_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ofNat(lean_object* v_n_617_){
_start:
{
lean_object* v___x_618_; uint8_t v___x_619_; 
v___x_618_ = lean_unsigned_to_nat(4u);
v___x_619_ = lean_nat_dec_le(v_n_617_, v___x_618_);
if (v___x_619_ == 0)
{
lean_object* v___x_620_; uint8_t v___x_621_; 
v___x_620_ = lean_unsigned_to_nat(6u);
v___x_621_ = lean_nat_dec_le(v_n_617_, v___x_620_);
if (v___x_621_ == 0)
{
lean_object* v___x_622_; uint8_t v___x_623_; 
v___x_622_ = lean_unsigned_to_nat(7u);
v___x_623_ = lean_nat_dec_le(v_n_617_, v___x_622_);
if (v___x_623_ == 0)
{
lean_object* v___x_624_; uint8_t v___x_625_; 
v___x_624_ = lean_unsigned_to_nat(8u);
v___x_625_ = lean_nat_dec_le(v_n_617_, v___x_624_);
if (v___x_625_ == 0)
{
uint8_t v___x_626_; 
v___x_626_ = 9;
return v___x_626_;
}
else
{
uint8_t v___x_627_; 
v___x_627_ = 8;
return v___x_627_;
}
}
else
{
uint8_t v___x_628_; 
v___x_628_ = 7;
return v___x_628_;
}
}
else
{
lean_object* v___x_629_; uint8_t v___x_630_; 
v___x_629_ = lean_unsigned_to_nat(5u);
v___x_630_ = lean_nat_dec_le(v_n_617_, v___x_629_);
if (v___x_630_ == 0)
{
uint8_t v___x_631_; 
v___x_631_ = 6;
return v___x_631_;
}
else
{
uint8_t v___x_632_; 
v___x_632_ = 5;
return v___x_632_;
}
}
}
else
{
lean_object* v___x_633_; uint8_t v___x_634_; 
v___x_633_ = lean_unsigned_to_nat(1u);
v___x_634_ = lean_nat_dec_le(v_n_617_, v___x_633_);
if (v___x_634_ == 0)
{
lean_object* v___x_635_; uint8_t v___x_636_; 
v___x_635_ = lean_unsigned_to_nat(2u);
v___x_636_ = lean_nat_dec_le(v_n_617_, v___x_635_);
if (v___x_636_ == 0)
{
lean_object* v___x_637_; uint8_t v___x_638_; 
v___x_637_ = lean_unsigned_to_nat(3u);
v___x_638_ = lean_nat_dec_le(v_n_617_, v___x_637_);
if (v___x_638_ == 0)
{
uint8_t v___x_639_; 
v___x_639_ = 4;
return v___x_639_;
}
else
{
uint8_t v___x_640_; 
v___x_640_ = 3;
return v___x_640_;
}
}
else
{
uint8_t v___x_641_; 
v___x_641_ = 2;
return v___x_641_;
}
}
else
{
lean_object* v___x_642_; uint8_t v___x_643_; 
v___x_642_ = lean_unsigned_to_nat(0u);
v___x_643_ = lean_nat_dec_le(v_n_617_, v___x_642_);
if (v___x_643_ == 0)
{
uint8_t v___x_644_; 
v___x_644_ = 1;
return v___x_644_;
}
else
{
uint8_t v___x_645_; 
v___x_645_ = 0;
return v___x_645_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ofNat___boxed(lean_object* v_n_646_){
_start:
{
uint8_t v_res_647_; lean_object* v_r_648_; 
v_res_647_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ofNat(v_n_646_);
lean_dec(v_n_646_);
v_r_648_ = lean_box(v_res_647_);
return v_r_648_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqVerifierError(uint8_t v_x_649_, uint8_t v_y_650_){
_start:
{
lean_object* v___x_651_; lean_object* v___x_652_; uint8_t v___x_653_; 
v___x_651_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx(v_x_649_);
v___x_652_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_VerifierError_ctorIdx(v_y_650_);
v___x_653_ = lean_nat_dec_eq(v___x_651_, v___x_652_);
lean_dec(v___x_652_);
lean_dec(v___x_651_);
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqVerifierError___boxed(lean_object* v_x_654_, lean_object* v_y_655_){
_start:
{
uint8_t v_x_13__boxed_656_; uint8_t v_y_14__boxed_657_; uint8_t v_res_658_; lean_object* v_r_659_; 
v_x_13__boxed_656_ = lean_unbox(v_x_654_);
v_y_14__boxed_657_ = lean_unbox(v_y_655_);
v_res_658_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instDecidableEqVerifierError(v_x_13__boxed_656_, v_y_14__boxed_657_);
v_r_659_ = lean_box(v_res_658_);
return v_r_659_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr(uint8_t v_x_690_, lean_object* v_prec_691_){
_start:
{
lean_object* v___y_693_; lean_object* v___y_700_; lean_object* v___y_707_; lean_object* v___y_714_; lean_object* v___y_721_; lean_object* v___y_728_; lean_object* v___y_735_; lean_object* v___y_742_; lean_object* v___y_749_; lean_object* v___y_756_; 
switch(v_x_690_)
{
case 0:
{
lean_object* v___x_762_; uint8_t v___x_763_; 
v___x_762_ = lean_unsigned_to_nat(1024u);
v___x_763_ = lean_nat_dec_le(v___x_762_, v_prec_691_);
if (v___x_763_ == 0)
{
lean_object* v___x_764_; 
v___x_764_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_693_ = v___x_764_;
goto v___jp_692_;
}
else
{
lean_object* v___x_765_; 
v___x_765_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_693_ = v___x_765_;
goto v___jp_692_;
}
}
case 1:
{
lean_object* v___x_766_; uint8_t v___x_767_; 
v___x_766_ = lean_unsigned_to_nat(1024u);
v___x_767_ = lean_nat_dec_le(v___x_766_, v_prec_691_);
if (v___x_767_ == 0)
{
lean_object* v___x_768_; 
v___x_768_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_700_ = v___x_768_;
goto v___jp_699_;
}
else
{
lean_object* v___x_769_; 
v___x_769_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_700_ = v___x_769_;
goto v___jp_699_;
}
}
case 2:
{
lean_object* v___x_770_; uint8_t v___x_771_; 
v___x_770_ = lean_unsigned_to_nat(1024u);
v___x_771_ = lean_nat_dec_le(v___x_770_, v_prec_691_);
if (v___x_771_ == 0)
{
lean_object* v___x_772_; 
v___x_772_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_707_ = v___x_772_;
goto v___jp_706_;
}
else
{
lean_object* v___x_773_; 
v___x_773_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_707_ = v___x_773_;
goto v___jp_706_;
}
}
case 3:
{
lean_object* v___x_774_; uint8_t v___x_775_; 
v___x_774_ = lean_unsigned_to_nat(1024u);
v___x_775_ = lean_nat_dec_le(v___x_774_, v_prec_691_);
if (v___x_775_ == 0)
{
lean_object* v___x_776_; 
v___x_776_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_714_ = v___x_776_;
goto v___jp_713_;
}
else
{
lean_object* v___x_777_; 
v___x_777_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_714_ = v___x_777_;
goto v___jp_713_;
}
}
case 4:
{
lean_object* v___x_778_; uint8_t v___x_779_; 
v___x_778_ = lean_unsigned_to_nat(1024u);
v___x_779_ = lean_nat_dec_le(v___x_778_, v_prec_691_);
if (v___x_779_ == 0)
{
lean_object* v___x_780_; 
v___x_780_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_721_ = v___x_780_;
goto v___jp_720_;
}
else
{
lean_object* v___x_781_; 
v___x_781_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_721_ = v___x_781_;
goto v___jp_720_;
}
}
case 5:
{
lean_object* v___x_782_; uint8_t v___x_783_; 
v___x_782_ = lean_unsigned_to_nat(1024u);
v___x_783_ = lean_nat_dec_le(v___x_782_, v_prec_691_);
if (v___x_783_ == 0)
{
lean_object* v___x_784_; 
v___x_784_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_728_ = v___x_784_;
goto v___jp_727_;
}
else
{
lean_object* v___x_785_; 
v___x_785_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_728_ = v___x_785_;
goto v___jp_727_;
}
}
case 6:
{
lean_object* v___x_786_; uint8_t v___x_787_; 
v___x_786_ = lean_unsigned_to_nat(1024u);
v___x_787_ = lean_nat_dec_le(v___x_786_, v_prec_691_);
if (v___x_787_ == 0)
{
lean_object* v___x_788_; 
v___x_788_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_735_ = v___x_788_;
goto v___jp_734_;
}
else
{
lean_object* v___x_789_; 
v___x_789_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_735_ = v___x_789_;
goto v___jp_734_;
}
}
case 7:
{
lean_object* v___x_790_; uint8_t v___x_791_; 
v___x_790_ = lean_unsigned_to_nat(1024u);
v___x_791_ = lean_nat_dec_le(v___x_790_, v_prec_691_);
if (v___x_791_ == 0)
{
lean_object* v___x_792_; 
v___x_792_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_742_ = v___x_792_;
goto v___jp_741_;
}
else
{
lean_object* v___x_793_; 
v___x_793_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_742_ = v___x_793_;
goto v___jp_741_;
}
}
case 8:
{
lean_object* v___x_794_; uint8_t v___x_795_; 
v___x_794_ = lean_unsigned_to_nat(1024u);
v___x_795_ = lean_nat_dec_le(v___x_794_, v_prec_691_);
if (v___x_795_ == 0)
{
lean_object* v___x_796_; 
v___x_796_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_749_ = v___x_796_;
goto v___jp_748_;
}
else
{
lean_object* v___x_797_; 
v___x_797_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_749_ = v___x_797_;
goto v___jp_748_;
}
}
default: 
{
lean_object* v___x_798_; uint8_t v___x_799_; 
v___x_798_ = lean_unsigned_to_nat(1024u);
v___x_799_ = lean_nat_dec_le(v___x_798_, v_prec_691_);
if (v___x_799_ == 0)
{
lean_object* v___x_800_; 
v___x_800_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__3);
v___y_756_ = v___x_800_;
goto v___jp_755_;
}
else
{
lean_object* v___x_801_; 
v___x_801_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprTranscriptEvent_repr___redArg___closed__4);
v___y_756_ = v___x_801_;
goto v___jp_755_;
}
}
}
v___jp_692_:
{
lean_object* v___x_694_; lean_object* v___x_695_; uint8_t v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v___x_694_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__1));
lean_inc(v___y_693_);
v___x_695_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_695_, 0, v___y_693_);
lean_ctor_set(v___x_695_, 1, v___x_694_);
v___x_696_ = 0;
v___x_697_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_697_, 0, v___x_695_);
lean_ctor_set_uint8(v___x_697_, sizeof(void*)*1, v___x_696_);
v___x_698_ = l_Repr_addAppParen(v___x_697_, v_prec_691_);
return v___x_698_;
}
v___jp_699_:
{
lean_object* v___x_701_; lean_object* v___x_702_; uint8_t v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_701_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__3));
lean_inc(v___y_700_);
v___x_702_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_702_, 0, v___y_700_);
lean_ctor_set(v___x_702_, 1, v___x_701_);
v___x_703_ = 0;
v___x_704_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_704_, 0, v___x_702_);
lean_ctor_set_uint8(v___x_704_, sizeof(void*)*1, v___x_703_);
v___x_705_ = l_Repr_addAppParen(v___x_704_, v_prec_691_);
return v___x_705_;
}
v___jp_706_:
{
lean_object* v___x_708_; lean_object* v___x_709_; uint8_t v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_708_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__5));
lean_inc(v___y_707_);
v___x_709_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_709_, 0, v___y_707_);
lean_ctor_set(v___x_709_, 1, v___x_708_);
v___x_710_ = 0;
v___x_711_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_711_, 0, v___x_709_);
lean_ctor_set_uint8(v___x_711_, sizeof(void*)*1, v___x_710_);
v___x_712_ = l_Repr_addAppParen(v___x_711_, v_prec_691_);
return v___x_712_;
}
v___jp_713_:
{
lean_object* v___x_715_; lean_object* v___x_716_; uint8_t v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_715_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__7));
lean_inc(v___y_714_);
v___x_716_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_716_, 0, v___y_714_);
lean_ctor_set(v___x_716_, 1, v___x_715_);
v___x_717_ = 0;
v___x_718_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_718_, 0, v___x_716_);
lean_ctor_set_uint8(v___x_718_, sizeof(void*)*1, v___x_717_);
v___x_719_ = l_Repr_addAppParen(v___x_718_, v_prec_691_);
return v___x_719_;
}
v___jp_720_:
{
lean_object* v___x_722_; lean_object* v___x_723_; uint8_t v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; 
v___x_722_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__9));
lean_inc(v___y_721_);
v___x_723_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_723_, 0, v___y_721_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
v___x_724_ = 0;
v___x_725_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_725_, 0, v___x_723_);
lean_ctor_set_uint8(v___x_725_, sizeof(void*)*1, v___x_724_);
v___x_726_ = l_Repr_addAppParen(v___x_725_, v_prec_691_);
return v___x_726_;
}
v___jp_727_:
{
lean_object* v___x_729_; lean_object* v___x_730_; uint8_t v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_729_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__11));
lean_inc(v___y_728_);
v___x_730_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_730_, 0, v___y_728_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = 0;
v___x_732_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_732_, 0, v___x_730_);
lean_ctor_set_uint8(v___x_732_, sizeof(void*)*1, v___x_731_);
v___x_733_ = l_Repr_addAppParen(v___x_732_, v_prec_691_);
return v___x_733_;
}
v___jp_734_:
{
lean_object* v___x_736_; lean_object* v___x_737_; uint8_t v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; 
v___x_736_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__13));
lean_inc(v___y_735_);
v___x_737_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_737_, 0, v___y_735_);
lean_ctor_set(v___x_737_, 1, v___x_736_);
v___x_738_ = 0;
v___x_739_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_739_, 0, v___x_737_);
lean_ctor_set_uint8(v___x_739_, sizeof(void*)*1, v___x_738_);
v___x_740_ = l_Repr_addAppParen(v___x_739_, v_prec_691_);
return v___x_740_;
}
v___jp_741_:
{
lean_object* v___x_743_; lean_object* v___x_744_; uint8_t v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; 
v___x_743_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__15));
lean_inc(v___y_742_);
v___x_744_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_744_, 0, v___y_742_);
lean_ctor_set(v___x_744_, 1, v___x_743_);
v___x_745_ = 0;
v___x_746_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_746_, 0, v___x_744_);
lean_ctor_set_uint8(v___x_746_, sizeof(void*)*1, v___x_745_);
v___x_747_ = l_Repr_addAppParen(v___x_746_, v_prec_691_);
return v___x_747_;
}
v___jp_748_:
{
lean_object* v___x_750_; lean_object* v___x_751_; uint8_t v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; 
v___x_750_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__17));
lean_inc(v___y_749_);
v___x_751_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_751_, 0, v___y_749_);
lean_ctor_set(v___x_751_, 1, v___x_750_);
v___x_752_ = 0;
v___x_753_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_753_, 0, v___x_751_);
lean_ctor_set_uint8(v___x_753_, sizeof(void*)*1, v___x_752_);
v___x_754_ = l_Repr_addAppParen(v___x_753_, v_prec_691_);
return v___x_754_;
}
v___jp_755_:
{
lean_object* v___x_757_; lean_object* v___x_758_; uint8_t v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; 
v___x_757_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___closed__19));
lean_inc(v___y_756_);
v___x_758_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_758_, 0, v___y_756_);
lean_ctor_set(v___x_758_, 1, v___x_757_);
v___x_759_ = 0;
v___x_760_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_760_, 0, v___x_758_);
lean_ctor_set_uint8(v___x_760_, sizeof(void*)*1, v___x_759_);
v___x_761_ = l_Repr_addAppParen(v___x_760_, v_prec_691_);
return v___x_761_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr___boxed(lean_object* v_x_802_, lean_object* v_prec_803_){
_start:
{
uint8_t v_x_565__boxed_804_; lean_object* v_res_805_; 
v_x_565__boxed_804_ = lean_unbox(v_x_802_);
v_res_805_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_instReprVerifierError_repr(v_x_565__boxed_804_, v_prec_803_);
lean_dec(v_prec_803_);
return v_res_805_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_init___redArg(lean_object* v_inst_808_){
_start:
{
lean_object* v___x_809_; 
v___x_809_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg(v_inst_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_init(lean_object* v_F_810_, lean_object* v_inst_811_){
_start:
{
lean_object* v___x_812_; 
v___x_812_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_init___redArg(v_inst_811_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(lean_object* v_inst_813_, lean_object* v_value_814_, lean_object* v_a_815_){
_start:
{
lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_816_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_absorb___redArg(v_inst_813_, v_a_815_, v_value_814_);
v___x_817_ = lean_box(0);
v___x_818_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_818_, 0, v___x_817_);
lean_ctor_set(v___x_818_, 1, v___x_816_);
v___x_819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase(lean_object* v_F_820_, lean_object* v_inst_821_, lean_object* v_value_822_, lean_object* v_a_823_){
_start:
{
lean_object* v___x_824_; 
v___x_824_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(v_inst_821_, v_value_822_, v_a_823_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(lean_object* v_inst_825_, lean_object* v_inst_826_, lean_object* v_value_827_, lean_object* v_a_828_){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; 
v___x_829_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeExt___redArg(v_inst_825_, v_inst_826_, v_a_828_, v_value_827_);
v___x_830_ = lean_box(0);
v___x_831_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_831_, 0, v___x_830_);
lean_ctor_set(v___x_831_, 1, v___x_829_);
v___x_832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_832_, 0, v___x_831_);
return v___x_832_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt(lean_object* v_F_833_, lean_object* v_EF_834_, lean_object* v_inst_835_, lean_object* v_inst_836_, lean_object* v_value_837_, lean_object* v_a_838_){
_start:
{
lean_object* v___x_839_; 
v___x_839_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_835_, v_inst_836_, v_value_837_, v_a_838_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___lam__0(lean_object* v_inst_840_, lean_object* v_inst_841_, lean_object* v_x_842_, lean_object* v_value_843_, lean_object* v___y_844_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_840_, v_inst_841_, v_value_843_, v___y_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg(lean_object* v_inst_891_, lean_object* v_inst_892_, lean_object* v_values_893_, lean_object* v_a_894_){
_start:
{
lean_object* v___f_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_284__overap_898_; lean_object* v___x_899_; 
v___f_895_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___lam__0), 5, 2);
lean_closure_set(v___f_895_, 0, v_inst_891_);
lean_closure_set(v___f_895_, 1, v_inst_892_);
v___x_896_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg___closed__19));
v___x_897_ = lean_box(0);
v___x_284__overap_898_ = l_List_foldlM___redArg(v___x_896_, v___f_895_, v___x_897_, v_values_893_);
v___x_899_ = lean_apply_1(v___x_284__overap_898_, v_a_894_);
if (lean_obj_tag(v___x_899_) == 0)
{
return v___x_899_;
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_916_; 
v_a_900_ = lean_ctor_get(v___x_899_, 0);
v_isSharedCheck_916_ = !lean_is_exclusive(v___x_899_);
if (v_isSharedCheck_916_ == 0)
{
v___x_902_ = v___x_899_;
v_isShared_903_ = v_isSharedCheck_916_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_899_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_916_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v_snd_904_; lean_object* v___x_906_; uint8_t v_isShared_907_; uint8_t v_isSharedCheck_914_; 
v_snd_904_ = lean_ctor_get(v_a_900_, 1);
v_isSharedCheck_914_ = !lean_is_exclusive(v_a_900_);
if (v_isSharedCheck_914_ == 0)
{
lean_object* v_unused_915_; 
v_unused_915_ = lean_ctor_get(v_a_900_, 0);
lean_dec(v_unused_915_);
v___x_906_ = v_a_900_;
v_isShared_907_ = v_isSharedCheck_914_;
goto v_resetjp_905_;
}
else
{
lean_inc(v_snd_904_);
lean_dec(v_a_900_);
v___x_906_ = lean_box(0);
v_isShared_907_ = v_isSharedCheck_914_;
goto v_resetjp_905_;
}
v_resetjp_905_:
{
lean_object* v___x_909_; 
if (v_isShared_907_ == 0)
{
lean_ctor_set(v___x_906_, 0, v___x_897_);
v___x_909_ = v___x_906_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v___x_897_);
lean_ctor_set(v_reuseFailAlloc_913_, 1, v_snd_904_);
v___x_909_ = v_reuseFailAlloc_913_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
lean_object* v___x_911_; 
if (v_isShared_903_ == 0)
{
lean_ctor_set(v___x_902_, 0, v___x_909_);
v___x_911_ = v___x_902_;
goto v_reusejp_910_;
}
else
{
lean_object* v_reuseFailAlloc_912_; 
v_reuseFailAlloc_912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_912_, 0, v___x_909_);
v___x_911_ = v_reuseFailAlloc_912_;
goto v_reusejp_910_;
}
v_reusejp_910_:
{
return v___x_911_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList(lean_object* v_F_917_, lean_object* v_EF_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_values_921_, lean_object* v_a_922_){
_start:
{
lean_object* v___x_923_; 
v___x_923_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg(v_inst_919_, v_inst_920_, v_values_921_, v_a_922_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(lean_object* v_inst_924_, lean_object* v_inst_925_, lean_object* v_digest_926_, lean_object* v_a_927_){
_start:
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; 
v___x_928_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_observeCommit___redArg(v_inst_924_, v_inst_925_, v_a_927_, v_digest_926_);
v___x_929_ = lean_box(0);
v___x_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_929_);
lean_ctor_set(v___x_930_, 1, v___x_928_);
v___x_931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_931_, 0, v___x_930_);
return v___x_931_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit(lean_object* v_F_932_, lean_object* v_Digest_933_, lean_object* v_inst_934_, lean_object* v_inst_935_, lean_object* v_digest_936_, lean_object* v_a_937_){
_start:
{
lean_object* v___x_938_; 
v___x_938_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeCommit___redArg(v_inst_934_, v_inst_935_, v_digest_936_, v_a_937_);
return v___x_938_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBase___redArg(lean_object* v_inst_939_, lean_object* v_a_940_){
_start:
{
lean_object* v___x_941_; lean_object* v___x_942_; 
v___x_941_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sample___redArg(v_inst_939_, v_a_940_);
v___x_942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_942_, 0, v___x_941_);
return v___x_942_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBase(lean_object* v_F_943_, lean_object* v_inst_944_, lean_object* v_a_945_){
_start:
{
lean_object* v___x_946_; 
v___x_946_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBase___redArg(v_inst_944_, v_a_945_);
return v___x_946_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(lean_object* v_inst_947_, lean_object* v_inst_948_, lean_object* v_a_949_){
_start:
{
lean_object* v___x_950_; lean_object* v___x_951_; 
v___x_950_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_DuplexSponge_sampleExt___redArg(v_inst_947_, v_inst_948_, v_a_949_);
v___x_951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_951_, 0, v___x_950_);
return v___x_951_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt(lean_object* v_F_952_, lean_object* v_EF_953_, lean_object* v_inst_954_, lean_object* v_inst_955_, lean_object* v_a_956_){
_start:
{
lean_object* v___x_957_; 
v___x_957_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_954_, v_inst_955_, v_a_956_);
return v___x_957_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg(lean_object* v_inst_958_, lean_object* v_inst_959_, lean_object* v_bits_960_, lean_object* v_a_961_){
_start:
{
lean_object* v___x_962_; lean_object* v_a_963_; lean_object* v___x_965_; uint8_t v_isShared_966_; uint8_t v_isSharedCheck_984_; 
v___x_962_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBase___redArg(v_inst_958_, v_a_961_);
v_a_963_ = lean_ctor_get(v___x_962_, 0);
v_isSharedCheck_984_ = !lean_is_exclusive(v___x_962_);
if (v_isSharedCheck_984_ == 0)
{
v___x_965_ = v___x_962_;
v_isShared_966_ = v_isSharedCheck_984_;
goto v_resetjp_964_;
}
else
{
lean_inc(v_a_963_);
lean_dec(v___x_962_);
v___x_965_ = lean_box(0);
v_isShared_966_ = v_isSharedCheck_984_;
goto v_resetjp_964_;
}
v_resetjp_964_:
{
lean_object* v_fst_967_; lean_object* v_snd_968_; lean_object* v___x_970_; uint8_t v_isShared_971_; uint8_t v_isSharedCheck_983_; 
v_fst_967_ = lean_ctor_get(v_a_963_, 0);
v_snd_968_ = lean_ctor_get(v_a_963_, 1);
v_isSharedCheck_983_ = !lean_is_exclusive(v_a_963_);
if (v_isSharedCheck_983_ == 0)
{
v___x_970_ = v_a_963_;
v_isShared_971_ = v_isSharedCheck_983_;
goto v_resetjp_969_;
}
else
{
lean_inc(v_snd_968_);
lean_inc(v_fst_967_);
lean_dec(v_a_963_);
v___x_970_ = lean_box(0);
v_isShared_971_ = v_isSharedCheck_983_;
goto v_resetjp_969_;
}
v_resetjp_969_:
{
lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_978_; 
v___x_972_ = lean_apply_1(v_inst_959_, v_fst_967_);
v___x_973_ = lean_unsigned_to_nat(1u);
v___x_974_ = lean_nat_shiftl(v___x_973_, v_bits_960_);
v___x_975_ = lean_nat_sub(v___x_974_, v___x_973_);
lean_dec(v___x_974_);
v___x_976_ = lean_nat_land(v___x_972_, v___x_975_);
lean_dec(v___x_975_);
lean_dec(v___x_972_);
if (v_isShared_971_ == 0)
{
lean_ctor_set(v___x_970_, 0, v___x_976_);
v___x_978_ = v___x_970_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_982_; 
v_reuseFailAlloc_982_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_982_, 0, v___x_976_);
lean_ctor_set(v_reuseFailAlloc_982_, 1, v_snd_968_);
v___x_978_ = v_reuseFailAlloc_982_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
lean_object* v___x_980_; 
if (v_isShared_966_ == 0)
{
lean_ctor_set(v___x_965_, 0, v___x_978_);
v___x_980_ = v___x_965_;
goto v_reusejp_979_;
}
else
{
lean_object* v_reuseFailAlloc_981_; 
v_reuseFailAlloc_981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_981_, 0, v___x_978_);
v___x_980_ = v_reuseFailAlloc_981_;
goto v_reusejp_979_;
}
v_reusejp_979_:
{
return v___x_980_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg___boxed(lean_object* v_inst_985_, lean_object* v_inst_986_, lean_object* v_bits_987_, lean_object* v_a_988_){
_start:
{
lean_object* v_res_989_; 
v_res_989_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg(v_inst_985_, v_inst_986_, v_bits_987_, v_a_988_);
lean_dec(v_bits_987_);
return v_res_989_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits(lean_object* v_F_990_, lean_object* v_inst_991_, lean_object* v_inst_992_, lean_object* v_bits_993_, lean_object* v_a_994_){
_start:
{
lean_object* v___x_995_; 
v___x_995_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg(v_inst_991_, v_inst_992_, v_bits_993_, v_a_994_);
return v___x_995_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___boxed(lean_object* v_F_996_, lean_object* v_inst_997_, lean_object* v_inst_998_, lean_object* v_bits_999_, lean_object* v_a_1000_){
_start:
{
lean_object* v_res_1001_; 
v_res_1001_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits(v_F_996_, v_inst_997_, v_inst_998_, v_bits_999_, v_a_1000_);
lean_dec(v_bits_999_);
return v_res_1001_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg(lean_object* v_inst_1005_, lean_object* v_inst_1006_, lean_object* v_powBits_1007_, lean_object* v_witness_1008_, lean_object* v_a_1009_){
_start:
{
lean_object* v___x_1010_; uint8_t v___x_1011_; 
v___x_1010_ = lean_unsigned_to_nat(0u);
v___x_1011_ = lean_nat_dec_eq(v_powBits_1007_, v___x_1010_);
if (v___x_1011_ == 0)
{
lean_object* v___x_1012_; lean_object* v_a_1013_; lean_object* v_snd_1014_; lean_object* v___x_1015_; 
lean_inc_ref(v_inst_1005_);
v___x_1012_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observeBase___redArg(v_inst_1005_, v_witness_1008_, v_a_1009_);
v_a_1013_ = lean_ctor_get(v___x_1012_, 0);
lean_inc(v_a_1013_);
lean_dec_ref(v___x_1012_);
v_snd_1014_ = lean_ctor_get(v_a_1013_, 1);
lean_inc(v_snd_1014_);
lean_dec(v_a_1013_);
v___x_1015_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_sampleBits___redArg(v_inst_1005_, v_inst_1006_, v_powBits_1007_, v_snd_1014_);
if (lean_obj_tag(v___x_1015_) == 0)
{
lean_object* v_a_1016_; lean_object* v___x_1018_; uint8_t v_isShared_1019_; uint8_t v_isSharedCheck_1023_; 
v_a_1016_ = lean_ctor_get(v___x_1015_, 0);
v_isSharedCheck_1023_ = !lean_is_exclusive(v___x_1015_);
if (v_isSharedCheck_1023_ == 0)
{
v___x_1018_ = v___x_1015_;
v_isShared_1019_ = v_isSharedCheck_1023_;
goto v_resetjp_1017_;
}
else
{
lean_inc(v_a_1016_);
lean_dec(v___x_1015_);
v___x_1018_ = lean_box(0);
v_isShared_1019_ = v_isSharedCheck_1023_;
goto v_resetjp_1017_;
}
v_resetjp_1017_:
{
lean_object* v___x_1021_; 
if (v_isShared_1019_ == 0)
{
v___x_1021_ = v___x_1018_;
goto v_reusejp_1020_;
}
else
{
lean_object* v_reuseFailAlloc_1022_; 
v_reuseFailAlloc_1022_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1022_, 0, v_a_1016_);
v___x_1021_ = v_reuseFailAlloc_1022_;
goto v_reusejp_1020_;
}
v_reusejp_1020_:
{
return v___x_1021_;
}
}
}
else
{
lean_object* v_a_1024_; lean_object* v___x_1026_; uint8_t v_isShared_1027_; uint8_t v_isSharedCheck_1043_; 
v_a_1024_ = lean_ctor_get(v___x_1015_, 0);
v_isSharedCheck_1043_ = !lean_is_exclusive(v___x_1015_);
if (v_isSharedCheck_1043_ == 0)
{
v___x_1026_ = v___x_1015_;
v_isShared_1027_ = v_isSharedCheck_1043_;
goto v_resetjp_1025_;
}
else
{
lean_inc(v_a_1024_);
lean_dec(v___x_1015_);
v___x_1026_ = lean_box(0);
v_isShared_1027_ = v_isSharedCheck_1043_;
goto v_resetjp_1025_;
}
v_resetjp_1025_:
{
lean_object* v_fst_1028_; lean_object* v_snd_1029_; lean_object* v___x_1031_; uint8_t v_isShared_1032_; uint8_t v_isSharedCheck_1042_; 
v_fst_1028_ = lean_ctor_get(v_a_1024_, 0);
v_snd_1029_ = lean_ctor_get(v_a_1024_, 1);
v_isSharedCheck_1042_ = !lean_is_exclusive(v_a_1024_);
if (v_isSharedCheck_1042_ == 0)
{
v___x_1031_ = v_a_1024_;
v_isShared_1032_ = v_isSharedCheck_1042_;
goto v_resetjp_1030_;
}
else
{
lean_inc(v_snd_1029_);
lean_inc(v_fst_1028_);
lean_dec(v_a_1024_);
v___x_1031_ = lean_box(0);
v_isShared_1032_ = v_isSharedCheck_1042_;
goto v_resetjp_1030_;
}
v_resetjp_1030_:
{
uint8_t v___x_1033_; 
v___x_1033_ = lean_nat_dec_eq(v_fst_1028_, v___x_1010_);
lean_dec(v_fst_1028_);
if (v___x_1033_ == 0)
{
lean_object* v___x_1034_; 
lean_del_object(v___x_1031_);
lean_dec(v_snd_1029_);
lean_del_object(v___x_1026_);
v___x_1034_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg___closed__0));
return v___x_1034_;
}
else
{
lean_object* v___x_1035_; lean_object* v___x_1037_; 
v___x_1035_ = lean_box(0);
if (v_isShared_1032_ == 0)
{
lean_ctor_set(v___x_1031_, 0, v___x_1035_);
v___x_1037_ = v___x_1031_;
goto v_reusejp_1036_;
}
else
{
lean_object* v_reuseFailAlloc_1041_; 
v_reuseFailAlloc_1041_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1041_, 0, v___x_1035_);
lean_ctor_set(v_reuseFailAlloc_1041_, 1, v_snd_1029_);
v___x_1037_ = v_reuseFailAlloc_1041_;
goto v_reusejp_1036_;
}
v_reusejp_1036_:
{
lean_object* v___x_1039_; 
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 0, v___x_1037_);
v___x_1039_ = v___x_1026_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v___x_1037_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
return v___x_1039_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; 
lean_dec(v_witness_1008_);
lean_dec_ref(v_inst_1006_);
lean_dec_ref(v_inst_1005_);
v___x_1044_ = lean_box(0);
v___x_1045_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1045_, 0, v___x_1044_);
lean_ctor_set(v___x_1045_, 1, v_a_1009_);
v___x_1046_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1045_);
return v___x_1046_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg___boxed(lean_object* v_inst_1047_, lean_object* v_inst_1048_, lean_object* v_powBits_1049_, lean_object* v_witness_1050_, lean_object* v_a_1051_){
_start:
{
lean_object* v_res_1052_; 
v_res_1052_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg(v_inst_1047_, v_inst_1048_, v_powBits_1049_, v_witness_1050_, v_a_1051_);
lean_dec(v_powBits_1049_);
return v_res_1052_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness(lean_object* v_F_1053_, lean_object* v_inst_1054_, lean_object* v_inst_1055_, lean_object* v_powBits_1056_, lean_object* v_witness_1057_, lean_object* v_a_1058_){
_start:
{
lean_object* v___x_1059_; 
v___x_1059_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg(v_inst_1054_, v_inst_1055_, v_powBits_1056_, v_witness_1057_, v_a_1058_);
return v___x_1059_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___boxed(lean_object* v_F_1060_, lean_object* v_inst_1061_, lean_object* v_inst_1062_, lean_object* v_powBits_1063_, lean_object* v_witness_1064_, lean_object* v_a_1065_){
_start:
{
lean_object* v_res_1066_; 
v_res_1066_ = lp_swirl_x2drbr_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness(v_F_1060_, v_inst_1061_, v_inst_1062_, v_powBits_1063_, v_witness_1064_, v_a_1065_);
lean_dec(v_powBits_1063_);
return v_res_1066_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_Runtime_Core(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
