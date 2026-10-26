// Lean compiler output
// Module: Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Batch
// Imports: public import Init public meta import Init public import Fundamentals.Spec.Runtime.Core public import Swirl.Spec.ReferenceVerifier.Ops public import Swirl.Spec.ReferenceVerifier.Verifier.Runtime.Shape
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
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
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
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_SystemParams_logupPowBits(lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_totalInteractions___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_calculateNLogup(lean_object*, lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "values"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "numeratorClaim"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "denominatorClaim"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "xi"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "claim"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "point"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "eqAtPoint"};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput(lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__4, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__7, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__9, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_map, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_pure, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17_value;
static const lean_closure_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_bind, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17_value),((lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18_value)}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0_value;
static lean_once_cell_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___boxed(lean_object**);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___boxed(lean_object**);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_instDecidableEqList___redArg(v_inst_1_, v_x_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg___boxed(lean_object* v_inst_5_, lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg(v_inst_5_, v_x_6_, v_x_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq(lean_object* v_EF_10_, lean_object* v_inst_11_, lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
uint8_t v___x_14_; 
v___x_14_ = l_instDecidableEqList___redArg(v_inst_11_, v_x_12_, v_x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___boxed(lean_object* v_EF_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq(v_EF_15_, v_inst_16_, v_x_17_, v_x_18_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg(lean_object* v_inst_21_, lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
uint8_t v___x_24_; 
v___x_24_ = l_instDecidableEqList___redArg(v_inst_21_, v_x_22_, v_x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg___boxed(lean_object* v_inst_25_, lean_object* v_x_26_, lean_object* v_x_27_){
_start:
{
uint8_t v_res_28_; lean_object* v_r_29_; 
v_res_28_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg(v_inst_25_, v_x_26_, v_x_27_);
v_r_29_ = lean_box(v_res_28_);
return v_r_29_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput(lean_object* v_EF_30_, lean_object* v_inst_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v___x_34_; 
v___x_34_ = l_instDecidableEqList___redArg(v_inst_31_, v_x_32_, v_x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___boxed(lean_object* v_EF_35_, lean_object* v_inst_36_, lean_object* v_x_37_, lean_object* v_x_38_){
_start:
{
uint8_t v_res_39_; lean_object* v_r_40_; 
v_res_39_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput(v_EF_35_, v_inst_36_, v_x_37_, v_x_38_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = lean_unsigned_to_nat(10u);
v___x_55_ = lean_nat_to_int(v___x_54_);
return v___x_55_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0));
v___x_58_ = lean_string_length(v___x_57_);
return v___x_58_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9);
v___x_60_ = lean_nat_to_int(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg(lean_object* v_inst_65_, lean_object* v_x_66_){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; uint8_t v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_67_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6));
v___x_68_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7);
v___x_69_ = l_List_repr___redArg(v_inst_65_, v_x_66_);
v___x_70_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_68_);
lean_ctor_set(v___x_70_, 1, v___x_69_);
v___x_71_ = 0;
v___x_72_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_72_, 0, v___x_70_);
lean_ctor_set_uint8(v___x_72_, sizeof(void*)*1, v___x_71_);
v___x_73_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_67_);
lean_ctor_set(v___x_73_, 1, v___x_72_);
v___x_74_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10);
v___x_75_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11));
v___x_76_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_73_);
v___x_77_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12));
v___x_78_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_76_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
v___x_79_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_74_);
lean_ctor_set(v___x_79_, 1, v___x_78_);
v___x_80_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set_uint8(v___x_80_, sizeof(void*)*1, v___x_71_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr(lean_object* v_EF_81_, lean_object* v_inst_82_, lean_object* v_x_83_, lean_object* v_prec_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg(v_inst_82_, v_x_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed(lean_object* v_EF_86_, lean_object* v_inst_87_, lean_object* v_x_88_, lean_object* v_prec_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr(v_EF_86_, v_inst_87_, v_x_88_, v_prec_89_);
lean_dec(v_prec_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput___redArg(lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_92_, 0, lean_box(0));
lean_closure_set(v___x_92_, 1, v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput(lean_object* v_EF_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_95_, 0, lean_box(0));
lean_closure_set(v___x_95_, 1, v_inst_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(lean_object* v_traceVdata_96_, lean_object* v_lSkip_97_, lean_object* v_x_98_, lean_object* v_x_99_){
_start:
{
if (lean_obj_tag(v_x_99_) == 0)
{
return v_x_98_;
}
else
{
lean_object* v_head_100_; lean_object* v_tail_101_; lean_object* v___x_102_; 
v_head_100_ = lean_ctor_get(v_x_99_, 0);
lean_inc(v_head_100_);
v_tail_101_ = lean_ctor_get(v_x_99_, 1);
lean_inc(v_tail_101_);
lean_dec_ref_known(v_x_99_, 2);
v___x_102_ = l_List_get_x3fInternal___redArg(v_traceVdata_96_, v_head_100_);
if (lean_obj_tag(v___x_102_) == 0)
{
v_x_99_ = v_tail_101_;
goto _start;
}
else
{
lean_object* v_val_104_; 
v_val_104_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_val_104_);
lean_dec_ref_known(v___x_102_, 1);
if (lean_obj_tag(v_val_104_) == 0)
{
v_x_99_ = v_tail_101_;
goto _start;
}
else
{
lean_object* v_val_106_; lean_object* v_logHeight_107_; lean_object* v___x_108_; uint8_t v___x_109_; 
v_val_106_ = lean_ctor_get(v_val_104_, 0);
lean_inc(v_val_106_);
lean_dec_ref_known(v_val_104_, 1);
v_logHeight_107_ = lean_ctor_get(v_val_106_, 0);
lean_inc(v_logHeight_107_);
lean_dec(v_val_106_);
v___x_108_ = lean_nat_sub(v_logHeight_107_, v_lSkip_97_);
lean_dec(v_logHeight_107_);
v___x_109_ = lean_nat_dec_le(v_x_98_, v___x_108_);
if (v___x_109_ == 0)
{
lean_dec(v___x_108_);
v_x_99_ = v_tail_101_;
goto _start;
}
else
{
lean_dec(v_x_98_);
v_x_98_ = v___x_108_;
v_x_99_ = v_tail_101_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg___boxed(lean_object* v_traceVdata_112_, lean_object* v_lSkip_113_, lean_object* v_x_114_, lean_object* v_x_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(v_traceVdata_112_, v_lSkip_113_, v_x_114_, v_x_115_);
lean_dec(v_lSkip_113_);
lean_dec(v_traceVdata_112_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(lean_object* v_lSkip_117_, lean_object* v_traceVdata_118_, lean_object* v_traceIdToAirId_119_){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lean_unsigned_to_nat(0u);
v___x_121_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(v_traceVdata_118_, v_lSkip_117_, v___x_120_, v_traceIdToAirId_119_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg___boxed(lean_object* v_lSkip_122_, lean_object* v_traceVdata_123_, lean_object* v_traceIdToAirId_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(v_lSkip_122_, v_traceVdata_123_, v_traceIdToAirId_124_);
lean_dec(v_traceVdata_123_);
lean_dec(v_lSkip_122_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax(lean_object* v_Digest_126_, lean_object* v_lSkip_127_, lean_object* v_traceVdata_128_, lean_object* v_traceIdToAirId_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(v_lSkip_127_, v_traceVdata_128_, v_traceIdToAirId_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___boxed(lean_object* v_Digest_131_, lean_object* v_lSkip_132_, lean_object* v_traceVdata_133_, lean_object* v_traceIdToAirId_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax(v_Digest_131_, v_lSkip_132_, v_traceVdata_133_, v_traceIdToAirId_134_);
lean_dec(v_traceVdata_133_);
lean_dec(v_lSkip_132_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0(lean_object* v_Digest_136_, lean_object* v_traceVdata_137_, lean_object* v_lSkip_138_, lean_object* v_x_139_, lean_object* v_x_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(v_traceVdata_137_, v_lSkip_138_, v_x_139_, v_x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___boxed(lean_object* v_Digest_142_, lean_object* v_traceVdata_143_, lean_object* v_lSkip_144_, lean_object* v_x_145_, lean_object* v_x_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0(v_Digest_142_, v_traceVdata_143_, v_lSkip_144_, v_x_145_, v_x_146_);
lean_dec(v_lSkip_144_);
lean_dec(v_traceVdata_143_);
return v_res_147_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(lean_object* v_inst_148_, lean_object* v_x_149_, lean_object* v_x_150_){
_start:
{
lean_object* v_numeratorClaim_151_; lean_object* v_denominatorClaim_152_; lean_object* v_xi_153_; lean_object* v_numeratorClaim_154_; lean_object* v_denominatorClaim_155_; lean_object* v_xi_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v_numeratorClaim_151_ = lean_ctor_get(v_x_149_, 0);
lean_inc(v_numeratorClaim_151_);
v_denominatorClaim_152_ = lean_ctor_get(v_x_149_, 1);
lean_inc(v_denominatorClaim_152_);
v_xi_153_ = lean_ctor_get(v_x_149_, 2);
lean_inc(v_xi_153_);
lean_dec_ref(v_x_149_);
v_numeratorClaim_154_ = lean_ctor_get(v_x_150_, 0);
lean_inc(v_numeratorClaim_154_);
v_denominatorClaim_155_ = lean_ctor_get(v_x_150_, 1);
lean_inc(v_denominatorClaim_155_);
v_xi_156_ = lean_ctor_get(v_x_150_, 2);
lean_inc(v_xi_156_);
lean_dec_ref(v_x_150_);
lean_inc_ref(v_inst_148_);
v___x_157_ = lean_apply_2(v_inst_148_, v_numeratorClaim_151_, v_numeratorClaim_154_);
v___x_158_ = lean_unbox(v___x_157_);
if (v___x_158_ == 0)
{
uint8_t v___x_159_; 
lean_dec(v_xi_156_);
lean_dec(v_denominatorClaim_155_);
lean_dec(v_xi_153_);
lean_dec(v_denominatorClaim_152_);
lean_dec_ref(v_inst_148_);
v___x_159_ = lean_unbox(v___x_157_);
return v___x_159_;
}
else
{
lean_object* v___x_160_; uint8_t v___x_161_; 
lean_inc_ref(v_inst_148_);
v___x_160_ = lean_apply_2(v_inst_148_, v_denominatorClaim_152_, v_denominatorClaim_155_);
v___x_161_ = lean_unbox(v___x_160_);
if (v___x_161_ == 0)
{
uint8_t v___x_162_; 
lean_dec(v_xi_156_);
lean_dec(v_xi_153_);
lean_dec_ref(v_inst_148_);
v___x_162_ = lean_unbox(v___x_160_);
return v___x_162_;
}
else
{
uint8_t v___x_163_; 
v___x_163_ = l_instDecidableEqList___redArg(v_inst_148_, v_xi_153_, v_xi_156_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg___boxed(lean_object* v_inst_164_, lean_object* v_x_165_, lean_object* v_x_166_){
_start:
{
uint8_t v_res_167_; lean_object* v_r_168_; 
v_res_167_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_164_, v_x_165_, v_x_166_);
v_r_168_ = lean_box(v_res_167_);
return v_r_168_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq(lean_object* v_EF_169_, lean_object* v_inst_170_, lean_object* v_x_171_, lean_object* v_x_172_){
_start:
{
uint8_t v___x_173_; 
v___x_173_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_170_, v_x_171_, v_x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___boxed(lean_object* v_EF_174_, lean_object* v_inst_175_, lean_object* v_x_176_, lean_object* v_x_177_){
_start:
{
uint8_t v_res_178_; lean_object* v_r_179_; 
v_res_178_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq(v_EF_174_, v_inst_175_, v_x_176_, v_x_177_);
v_r_179_ = lean_box(v_res_178_);
return v_r_179_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg(lean_object* v_inst_180_, lean_object* v_x_181_, lean_object* v_x_182_){
_start:
{
uint8_t v___x_183_; 
v___x_183_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_180_, v_x_181_, v_x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg___boxed(lean_object* v_inst_184_, lean_object* v_x_185_, lean_object* v_x_186_){
_start:
{
uint8_t v_res_187_; lean_object* v_r_188_; 
v_res_187_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg(v_inst_184_, v_x_185_, v_x_186_);
v_r_188_ = lean_box(v_res_187_);
return v_r_188_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput(lean_object* v_EF_189_, lean_object* v_inst_190_, lean_object* v_x_191_, lean_object* v_x_192_){
_start:
{
uint8_t v___x_193_; 
v___x_193_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_190_, v_x_191_, v_x_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___boxed(lean_object* v_EF_194_, lean_object* v_inst_195_, lean_object* v_x_196_, lean_object* v_x_197_){
_start:
{
uint8_t v_res_198_; lean_object* v_r_199_; 
v_res_198_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput(v_EF_194_, v_inst_195_, v_x_196_, v_x_197_);
v_r_199_ = lean_box(v_res_198_);
return v_r_199_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = lean_unsigned_to_nat(18u);
v___x_210_ = lean_nat_to_int(v___x_209_);
return v___x_210_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_unsigned_to_nat(20u);
v___x_218_ = lean_nat_to_int(v___x_217_);
return v___x_218_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_222_ = lean_unsigned_to_nat(6u);
v___x_223_ = lean_nat_to_int(v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg(lean_object* v_inst_224_, lean_object* v_x_225_){
_start:
{
lean_object* v_numeratorClaim_226_; lean_object* v_denominatorClaim_227_; lean_object* v_xi_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; uint8_t v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v_numeratorClaim_226_ = lean_ctor_get(v_x_225_, 0);
lean_inc(v_numeratorClaim_226_);
v_denominatorClaim_227_ = lean_ctor_get(v_x_225_, 1);
lean_inc(v_denominatorClaim_227_);
v_xi_228_ = lean_ctor_get(v_x_225_, 2);
lean_inc(v_xi_228_);
lean_dec_ref(v_x_225_);
v___x_229_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5));
v___x_230_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3));
v___x_231_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4);
v___x_232_ = lean_unsigned_to_nat(0u);
lean_inc_ref_n(v_inst_224_, 2);
v___x_233_ = lean_apply_2(v_inst_224_, v_numeratorClaim_226_, v___x_232_);
v___x_234_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_231_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
v___x_235_ = 0;
v___x_236_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_236_, 0, v___x_234_);
lean_ctor_set_uint8(v___x_236_, sizeof(void*)*1, v___x_235_);
v___x_237_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_230_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6));
v___x_239_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_237_);
lean_ctor_set(v___x_239_, 1, v___x_238_);
v___x_240_ = lean_box(1);
v___x_241_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_239_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8));
v___x_243_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_241_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
v___x_244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v___x_229_);
v___x_245_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9);
v___x_246_ = lean_apply_2(v_inst_224_, v_denominatorClaim_227_, v___x_232_);
v___x_247_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_247_, 0, v___x_245_);
lean_ctor_set(v___x_247_, 1, v___x_246_);
v___x_248_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_248_, 0, v___x_247_);
lean_ctor_set_uint8(v___x_248_, sizeof(void*)*1, v___x_235_);
v___x_249_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_244_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
v___x_250_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v___x_238_);
v___x_251_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___x_240_);
v___x_252_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11));
v___x_253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_251_);
lean_ctor_set(v___x_253_, 1, v___x_252_);
v___x_254_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v___x_229_);
v___x_255_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12);
v___x_256_ = l_List_repr___redArg(v_inst_224_, v_xi_228_);
v___x_257_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_255_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
v___x_258_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set_uint8(v___x_258_, sizeof(void*)*1, v___x_235_);
v___x_259_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_254_);
lean_ctor_set(v___x_259_, 1, v___x_258_);
v___x_260_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10);
v___x_261_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11));
v___x_262_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_259_);
v___x_263_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12));
v___x_264_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_262_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_260_);
lean_ctor_set(v___x_265_, 1, v___x_264_);
v___x_266_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_266_, 0, v___x_265_);
lean_ctor_set_uint8(v___x_266_, sizeof(void*)*1, v___x_235_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr(lean_object* v_EF_267_, lean_object* v_inst_268_, lean_object* v_x_269_, lean_object* v_prec_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg(v_inst_268_, v_x_269_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed(lean_object* v_EF_272_, lean_object* v_inst_273_, lean_object* v_x_274_, lean_object* v_prec_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr(v_EF_272_, v_inst_273_, v_x_274_, v_prec_275_);
lean_dec(v_prec_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput___redArg(lean_object* v_inst_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_278_, 0, lean_box(0));
lean_closure_set(v___x_278_, 1, v_inst_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput(lean_object* v_EF_279_, lean_object* v_inst_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_281_, 0, lean_box(0));
lean_closure_set(v___x_281_, 1, v_inst_280_);
return v___x_281_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(lean_object* v_inst_282_, lean_object* v_x_283_, lean_object* v_x_284_){
_start:
{
lean_object* v_claim_285_; lean_object* v_point_286_; lean_object* v_eqAtPoint_287_; lean_object* v_claim_288_; lean_object* v_point_289_; lean_object* v_eqAtPoint_290_; lean_object* v___x_291_; uint8_t v___x_292_; 
v_claim_285_ = lean_ctor_get(v_x_283_, 0);
lean_inc(v_claim_285_);
v_point_286_ = lean_ctor_get(v_x_283_, 1);
lean_inc(v_point_286_);
v_eqAtPoint_287_ = lean_ctor_get(v_x_283_, 2);
lean_inc(v_eqAtPoint_287_);
lean_dec_ref(v_x_283_);
v_claim_288_ = lean_ctor_get(v_x_284_, 0);
lean_inc(v_claim_288_);
v_point_289_ = lean_ctor_get(v_x_284_, 1);
lean_inc(v_point_289_);
v_eqAtPoint_290_ = lean_ctor_get(v_x_284_, 2);
lean_inc(v_eqAtPoint_290_);
lean_dec_ref(v_x_284_);
lean_inc_ref(v_inst_282_);
v___x_291_ = lean_apply_2(v_inst_282_, v_claim_285_, v_claim_288_);
v___x_292_ = lean_unbox(v___x_291_);
if (v___x_292_ == 0)
{
uint8_t v___x_293_; 
lean_dec(v_eqAtPoint_290_);
lean_dec(v_point_289_);
lean_dec(v_eqAtPoint_287_);
lean_dec(v_point_286_);
lean_dec_ref(v_inst_282_);
v___x_293_ = lean_unbox(v___x_291_);
return v___x_293_;
}
else
{
uint8_t v___x_294_; 
lean_inc_ref(v_inst_282_);
v___x_294_ = l_instDecidableEqList___redArg(v_inst_282_, v_point_286_, v_point_289_);
if (v___x_294_ == 0)
{
lean_dec(v_eqAtPoint_290_);
lean_dec(v_eqAtPoint_287_);
lean_dec_ref(v_inst_282_);
return v___x_294_;
}
else
{
lean_object* v___x_295_; uint8_t v___x_296_; 
v___x_295_ = lean_apply_2(v_inst_282_, v_eqAtPoint_287_, v_eqAtPoint_290_);
v___x_296_ = lean_unbox(v___x_295_);
return v___x_296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg___boxed(lean_object* v_inst_297_, lean_object* v_x_298_, lean_object* v_x_299_){
_start:
{
uint8_t v_res_300_; lean_object* v_r_301_; 
v_res_300_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_297_, v_x_298_, v_x_299_);
v_r_301_ = lean_box(v_res_300_);
return v_r_301_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq(lean_object* v_EF_302_, lean_object* v_inst_303_, lean_object* v_x_304_, lean_object* v_x_305_){
_start:
{
uint8_t v___x_306_; 
v___x_306_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_303_, v_x_304_, v_x_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___boxed(lean_object* v_EF_307_, lean_object* v_inst_308_, lean_object* v_x_309_, lean_object* v_x_310_){
_start:
{
uint8_t v_res_311_; lean_object* v_r_312_; 
v_res_311_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq(v_EF_307_, v_inst_308_, v_x_309_, v_x_310_);
v_r_312_ = lean_box(v_res_311_);
return v_r_312_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg(lean_object* v_inst_313_, lean_object* v_x_314_, lean_object* v_x_315_){
_start:
{
uint8_t v___x_316_; 
v___x_316_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_313_, v_x_314_, v_x_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg___boxed(lean_object* v_inst_317_, lean_object* v_x_318_, lean_object* v_x_319_){
_start:
{
uint8_t v_res_320_; lean_object* v_r_321_; 
v_res_320_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg(v_inst_317_, v_x_318_, v_x_319_);
v_r_321_ = lean_box(v_res_320_);
return v_r_321_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput(lean_object* v_EF_322_, lean_object* v_inst_323_, lean_object* v_x_324_, lean_object* v_x_325_){
_start:
{
uint8_t v___x_326_; 
v___x_326_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_323_, v_x_324_, v_x_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___boxed(lean_object* v_EF_327_, lean_object* v_inst_328_, lean_object* v_x_329_, lean_object* v_x_330_){
_start:
{
uint8_t v_res_331_; lean_object* v_r_332_; 
v_res_331_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput(v_EF_327_, v_inst_328_, v_x_329_, v_x_330_);
v_r_332_ = lean_box(v_res_331_);
return v_r_332_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_unsigned_to_nat(9u);
v___x_343_ = lean_nat_to_int(v___x_342_);
return v___x_343_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_350_ = lean_unsigned_to_nat(13u);
v___x_351_ = lean_nat_to_int(v___x_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg(lean_object* v_inst_352_, lean_object* v_x_353_){
_start:
{
lean_object* v_claim_354_; lean_object* v_point_355_; lean_object* v_eqAtPoint_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; uint8_t v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v_claim_354_ = lean_ctor_get(v_x_353_, 0);
lean_inc(v_claim_354_);
v_point_355_ = lean_ctor_get(v_x_353_, 1);
lean_inc(v_point_355_);
v_eqAtPoint_356_ = lean_ctor_get(v_x_353_, 2);
lean_inc(v_eqAtPoint_356_);
lean_dec_ref(v_x_353_);
v___x_357_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5));
v___x_358_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3));
v___x_359_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4);
v___x_360_ = lean_unsigned_to_nat(0u);
lean_inc_ref_n(v_inst_352_, 2);
v___x_361_ = lean_apply_2(v_inst_352_, v_claim_354_, v___x_360_);
v___x_362_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_359_);
lean_ctor_set(v___x_362_, 1, v___x_361_);
v___x_363_ = 0;
v___x_364_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_364_, 0, v___x_362_);
lean_ctor_set_uint8(v___x_364_, sizeof(void*)*1, v___x_363_);
v___x_365_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_358_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
v___x_366_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6));
v___x_367_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_367_, 0, v___x_365_);
lean_ctor_set(v___x_367_, 1, v___x_366_);
v___x_368_ = lean_box(1);
v___x_369_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_367_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6));
v___x_371_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_369_);
lean_ctor_set(v___x_371_, 1, v___x_370_);
v___x_372_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_357_);
v___x_373_ = l_List_repr___redArg(v_inst_352_, v_point_355_);
v___x_374_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_359_);
lean_ctor_set(v___x_374_, 1, v___x_373_);
v___x_375_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set_uint8(v___x_375_, sizeof(void*)*1, v___x_363_);
v___x_376_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_376_, 0, v___x_372_);
lean_ctor_set(v___x_376_, 1, v___x_375_);
v___x_377_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set(v___x_377_, 1, v___x_366_);
v___x_378_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set(v___x_378_, 1, v___x_368_);
v___x_379_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8));
v___x_380_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_380_, 0, v___x_378_);
lean_ctor_set(v___x_380_, 1, v___x_379_);
v___x_381_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v___x_357_);
v___x_382_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9);
v___x_383_ = lean_apply_2(v_inst_352_, v_eqAtPoint_356_, v___x_360_);
v___x_384_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_382_);
lean_ctor_set(v___x_384_, 1, v___x_383_);
v___x_385_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_385_, 0, v___x_384_);
lean_ctor_set_uint8(v___x_385_, sizeof(void*)*1, v___x_363_);
v___x_386_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_381_);
lean_ctor_set(v___x_386_, 1, v___x_385_);
v___x_387_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10);
v___x_388_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11));
v___x_389_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_388_);
lean_ctor_set(v___x_389_, 1, v___x_386_);
v___x_390_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12));
v___x_391_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_389_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_387_);
lean_ctor_set(v___x_392_, 1, v___x_391_);
v___x_393_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_393_, 0, v___x_392_);
lean_ctor_set_uint8(v___x_393_, sizeof(void*)*1, v___x_363_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr(lean_object* v_EF_394_, lean_object* v_inst_395_, lean_object* v_x_396_, lean_object* v_prec_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg(v_inst_395_, v_x_396_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed(lean_object* v_EF_399_, lean_object* v_inst_400_, lean_object* v_x_401_, lean_object* v_prec_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr(v_EF_399_, v_inst_400_, v_x_401_, v_prec_402_);
lean_dec(v_prec_402_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput___redArg(lean_object* v_inst_404_){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_405_, 0, lean_box(0));
lean_closure_set(v___x_405_, 1, v_inst_404_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput(lean_object* v_EF_406_, lean_object* v_inst_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_408_, 0, lean_box(0));
lean_closure_set(v___x_408_, 1, v_inst_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(lean_object* v_values_412_, lean_object* v_idx_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = l_List_get_x3fInternal___redArg(v_values_412_, v_idx_413_);
if (lean_obj_tag(v___x_414_) == 0)
{
lean_object* v___x_415_; 
v___x_415_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_415_;
}
else
{
lean_object* v_val_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_423_; 
v_val_416_ = lean_ctor_get(v___x_414_, 0);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_423_ == 0)
{
v___x_418_ = v___x_414_;
v_isShared_419_ = v_isSharedCheck_423_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_val_416_);
lean_dec(v___x_414_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_423_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
lean_object* v___x_421_; 
if (v_isShared_419_ == 0)
{
v___x_421_ = v___x_418_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v_val_416_);
v___x_421_ = v_reuseFailAlloc_422_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
return v___x_421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___boxed(lean_object* v_values_424_, lean_object* v_idx_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_424_, v_idx_425_);
lean_dec(v_values_424_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem(lean_object* v_00_u03b1_427_, lean_object* v_values_428_, lean_object* v_idx_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_428_, v_idx_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___boxed(lean_object* v_00_u03b1_431_, lean_object* v_values_432_, lean_object* v_idx_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem(v_00_u03b1_431_, v_values_432_, v_idx_433_);
lean_dec(v_values_432_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(lean_object* v_value_435_){
_start:
{
if (lean_obj_tag(v_value_435_) == 0)
{
lean_object* v___x_436_; 
v___x_436_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_436_;
}
else
{
lean_object* v_val_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_444_; 
v_val_437_ = lean_ctor_get(v_value_435_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v_value_435_);
if (v_isSharedCheck_444_ == 0)
{
v___x_439_ = v_value_435_;
v_isShared_440_ = v_isSharedCheck_444_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_val_437_);
lean_dec(v_value_435_);
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
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_val_437_);
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
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption(lean_object* v_00_u03b1_445_, lean_object* v_value_446_){
_start:
{
lean_object* v___x_447_; 
v___x_447_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v_value_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(lean_object* v_f_448_, lean_object* v_init_449_, lean_object* v_x_450_){
_start:
{
if (lean_obj_tag(v_x_450_) == 0)
{
lean_object* v___x_451_; lean_object* v___x_452_; 
lean_dec(v_f_448_);
v___x_451_ = lean_box(0);
v___x_452_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_452_, 0, v_init_449_);
lean_ctor_set(v___x_452_, 1, v___x_451_);
return v___x_452_;
}
else
{
lean_object* v_head_453_; lean_object* v_tail_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_463_; 
v_head_453_ = lean_ctor_get(v_x_450_, 0);
v_tail_454_ = lean_ctor_get(v_x_450_, 1);
v_isSharedCheck_463_ = !lean_is_exclusive(v_x_450_);
if (v_isSharedCheck_463_ == 0)
{
v___x_456_ = v_x_450_;
v_isShared_457_ = v_isSharedCheck_463_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_tail_454_);
lean_inc(v_head_453_);
lean_dec(v_x_450_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_463_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_461_; 
lean_inc(v_f_448_);
lean_inc(v_init_449_);
v___x_458_ = lean_apply_2(v_f_448_, v_init_449_, v_head_453_);
v___x_459_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v_f_448_, v___x_458_, v_tail_454_);
if (v_isShared_457_ == 0)
{
lean_ctor_set(v___x_456_, 1, v___x_459_);
lean_ctor_set(v___x_456_, 0, v_init_449_);
v___x_461_ = v___x_456_;
goto v_reusejp_460_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v_init_449_);
lean_ctor_set(v_reuseFailAlloc_462_, 1, v___x_459_);
v___x_461_ = v_reuseFailAlloc_462_;
goto v_reusejp_460_;
}
v_reusejp_460_:
{
return v___x_461_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl(lean_object* v_00_u03b1_464_, lean_object* v_00_u03b2_465_, lean_object* v_f_466_, lean_object* v_init_467_, lean_object* v_x_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v_f_466_, v_init_467_, v_x_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(lean_object* v_proof_470_){
_start:
{
lean_object* v_claimsPerLayer_471_; uint8_t v___x_472_; 
v_claimsPerLayer_471_ = lean_ctor_get(v_proof_470_, 2);
v___x_472_ = l_List_isEmpty___redArg(v_claimsPerLayer_471_);
if (v___x_472_ == 0)
{
lean_object* v___x_473_; 
v___x_473_ = l_List_lengthTR___redArg(v_claimsPerLayer_471_);
return v___x_473_;
}
else
{
lean_object* v___x_474_; 
v___x_474_ = lean_unsigned_to_nat(0u);
return v___x_474_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg___boxed(lean_object* v_proof_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(v_proof_475_);
lean_dec_ref(v_proof_475_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof(lean_object* v_F_477_, lean_object* v_EF_478_, lean_object* v_proof_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(v_proof_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___boxed(lean_object* v_F_481_, lean_object* v_EF_482_, lean_object* v_proof_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof(v_F_481_, v_EF_482_, v_proof_483_);
lean_dec_ref(v_proof_483_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(lean_object* v_lSkip_485_, lean_object* v_proof_486_){
_start:
{
lean_object* v_numGkrRounds_487_; lean_object* v___x_488_; uint8_t v___x_489_; 
v_numGkrRounds_487_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(v_proof_486_);
v___x_488_ = lean_unsigned_to_nat(0u);
v___x_489_ = lean_nat_dec_eq(v_numGkrRounds_487_, v___x_488_);
if (v___x_489_ == 0)
{
lean_object* v___x_490_; 
v___x_490_ = lean_nat_sub(v_numGkrRounds_487_, v_lSkip_485_);
lean_dec(v_numGkrRounds_487_);
return v___x_490_;
}
else
{
lean_dec(v_numGkrRounds_487_);
return v___x_488_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg___boxed(lean_object* v_lSkip_491_, lean_object* v_proof_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(v_lSkip_491_, v_proof_492_);
lean_dec_ref(v_proof_492_);
lean_dec(v_lSkip_491_);
return v_res_493_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof(lean_object* v_F_494_, lean_object* v_EF_495_, lean_object* v_lSkip_496_, lean_object* v_proof_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(v_lSkip_496_, v_proof_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___boxed(lean_object* v_F_499_, lean_object* v_EF_500_, lean_object* v_lSkip_501_, lean_object* v_proof_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof(v_F_499_, v_EF_500_, v_lSkip_501_, v_proof_502_);
lean_dec_ref(v_proof_502_);
lean_dec(v_lSkip_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(lean_object* v_fo_504_, lean_object* v_claims_505_){
_start:
{
lean_object* v_toRingOps_506_; lean_object* v_toSemiringOps_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_524_; 
v_toRingOps_506_ = lean_ctor_get(v_fo_504_, 0);
lean_inc_ref(v_toRingOps_506_);
lean_dec_ref(v_fo_504_);
v_toSemiringOps_507_ = lean_ctor_get(v_toRingOps_506_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v_toRingOps_506_);
if (v_isSharedCheck_524_ == 0)
{
lean_object* v_unused_525_; 
v_unused_525_ = lean_ctor_get(v_toRingOps_506_, 1);
lean_dec(v_unused_525_);
v___x_509_ = v_toRingOps_506_;
v_isShared_510_ = v_isSharedCheck_524_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_toSemiringOps_507_);
lean_dec(v_toRingOps_506_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_524_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v_add_511_; lean_object* v_mul_512_; lean_object* v_pXi0_513_; lean_object* v_pXi1_514_; lean_object* v_qXi0_515_; lean_object* v_qXi1_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v_numeratorCross_519_; lean_object* v_denominatorCross_520_; lean_object* v___x_522_; 
v_add_511_ = lean_ctor_get(v_toSemiringOps_507_, 3);
lean_inc(v_add_511_);
v_mul_512_ = lean_ctor_get(v_toSemiringOps_507_, 4);
lean_inc_n(v_mul_512_, 3);
lean_dec_ref(v_toSemiringOps_507_);
v_pXi0_513_ = lean_ctor_get(v_claims_505_, 0);
lean_inc(v_pXi0_513_);
v_pXi1_514_ = lean_ctor_get(v_claims_505_, 1);
lean_inc(v_pXi1_514_);
v_qXi0_515_ = lean_ctor_get(v_claims_505_, 2);
lean_inc_n(v_qXi0_515_, 2);
v_qXi1_516_ = lean_ctor_get(v_claims_505_, 3);
lean_inc_n(v_qXi1_516_, 2);
lean_dec_ref(v_claims_505_);
v___x_517_ = lean_apply_2(v_mul_512_, v_pXi0_513_, v_qXi1_516_);
v___x_518_ = lean_apply_2(v_mul_512_, v_pXi1_514_, v_qXi0_515_);
v_numeratorCross_519_ = lean_apply_2(v_add_511_, v___x_517_, v___x_518_);
v_denominatorCross_520_ = lean_apply_2(v_mul_512_, v_qXi0_515_, v_qXi1_516_);
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 1, v_denominatorCross_520_);
lean_ctor_set(v___x_509_, 0, v_numeratorCross_519_);
v___x_522_ = v___x_509_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v_numeratorCross_519_);
lean_ctor_set(v_reuseFailAlloc_523_, 1, v_denominatorCross_520_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
return v___x_522_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations(lean_object* v_EF_526_, lean_object* v_fo_527_, lean_object* v_claims_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(v_fo_527_, v_claims_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(lean_object* v_fo_530_, lean_object* v_claims_531_, lean_object* v_mu_532_){
_start:
{
lean_object* v_toRingOps_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_546_; 
v_toRingOps_533_ = lean_ctor_get(v_fo_530_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v_fo_530_);
if (v_isSharedCheck_546_ == 0)
{
lean_object* v_unused_547_; 
v_unused_547_ = lean_ctor_get(v_fo_530_, 1);
lean_dec(v_unused_547_);
v___x_535_ = v_fo_530_;
v_isShared_536_ = v_isSharedCheck_546_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_toRingOps_533_);
lean_dec(v_fo_530_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_546_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v_pXi0_537_; lean_object* v_pXi1_538_; lean_object* v_qXi0_539_; lean_object* v_qXi1_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_544_; 
v_pXi0_537_ = lean_ctor_get(v_claims_531_, 0);
lean_inc(v_pXi0_537_);
v_pXi1_538_ = lean_ctor_get(v_claims_531_, 1);
lean_inc(v_pXi1_538_);
v_qXi0_539_ = lean_ctor_get(v_claims_531_, 2);
lean_inc(v_qXi0_539_);
v_qXi1_540_ = lean_ctor_get(v_claims_531_, 3);
lean_inc(v_qXi1_540_);
lean_dec_ref(v_claims_531_);
lean_inc(v_mu_532_);
lean_inc_ref(v_toRingOps_533_);
v___x_541_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(v_toRingOps_533_, v_pXi0_537_, v_pXi1_538_, v_mu_532_);
v___x_542_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(v_toRingOps_533_, v_qXi0_539_, v_qXi1_540_, v_mu_532_);
if (v_isShared_536_ == 0)
{
lean_ctor_set(v___x_535_, 1, v___x_542_);
lean_ctor_set(v___x_535_, 0, v___x_541_);
v___x_544_ = v___x_535_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v___x_541_);
lean_ctor_set(v_reuseFailAlloc_545_, 1, v___x_542_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation(lean_object* v_EF_548_, lean_object* v_fo_549_, lean_object* v_claims_550_, lean_object* v_mu_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(v_fo_549_, v_claims_550_, v_mu_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_claims_555_, lean_object* v_a_556_){
_start:
{
lean_object* v_pXi0_557_; lean_object* v_pXi1_558_; lean_object* v_qXi0_559_; lean_object* v_qXi1_560_; lean_object* v___x_561_; lean_object* v_a_562_; lean_object* v_snd_563_; lean_object* v___x_564_; lean_object* v_a_565_; lean_object* v_snd_566_; lean_object* v___x_567_; lean_object* v_a_568_; lean_object* v_snd_569_; lean_object* v___x_570_; 
v_pXi0_557_ = lean_ctor_get(v_claims_555_, 0);
lean_inc(v_pXi0_557_);
v_pXi1_558_ = lean_ctor_get(v_claims_555_, 1);
lean_inc(v_pXi1_558_);
v_qXi0_559_ = lean_ctor_get(v_claims_555_, 2);
lean_inc(v_qXi0_559_);
v_qXi1_560_ = lean_ctor_get(v_claims_555_, 3);
lean_inc(v_qXi1_560_);
lean_dec_ref(v_claims_555_);
lean_inc_ref_n(v_inst_554_, 3);
lean_inc_ref_n(v_inst_553_, 3);
v___x_561_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_pXi0_557_, v_a_556_);
v_a_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc(v_a_562_);
lean_dec_ref(v___x_561_);
v_snd_563_ = lean_ctor_get(v_a_562_, 1);
lean_inc(v_snd_563_);
lean_dec(v_a_562_);
v___x_564_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_qXi0_559_, v_snd_563_);
v_a_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_a_565_);
lean_dec_ref(v___x_564_);
v_snd_566_ = lean_ctor_get(v_a_565_, 1);
lean_inc(v_snd_566_);
lean_dec(v_a_565_);
v___x_567_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_pXi1_558_, v_snd_566_);
v_a_568_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_a_568_);
lean_dec_ref(v___x_567_);
v_snd_569_ = lean_ctor_get(v_a_568_, 1);
lean_inc(v_snd_569_);
lean_dec(v_a_568_);
v___x_570_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_qXi1_560_, v_snd_569_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM(lean_object* v_F_571_, lean_object* v_EF_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_claims_575_, lean_object* v_a_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(v_inst_573_, v_inst_574_, v_claims_575_, v_a_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_fo_583_, lean_object* v_claim_584_, lean_object* v_eqAtPoint_585_, lean_object* v_pointRev_586_, lean_object* v_gkrPoint_587_, lean_object* v_polys_588_, lean_object* v_a_589_){
_start:
{
if (lean_obj_tag(v_gkrPoint_587_) == 0)
{
lean_dec_ref(v_fo_583_);
lean_dec_ref(v_inst_582_);
lean_dec_ref(v_inst_581_);
if (lean_obj_tag(v_polys_588_) == 0)
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; 
v___x_592_ = l_List_reverse___redArg(v_pointRev_586_);
v___x_593_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_593_, 0, v_claim_584_);
lean_ctor_set(v___x_593_, 1, v___x_592_);
lean_ctor_set(v___x_593_, 2, v_eqAtPoint_585_);
v___x_594_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_594_, 0, v___x_593_);
lean_ctor_set(v___x_594_, 1, v_a_589_);
v___x_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
return v___x_595_;
}
else
{
lean_dec_ref(v_a_589_);
lean_dec(v_polys_588_);
lean_dec(v_pointRev_586_);
lean_dec(v_eqAtPoint_585_);
lean_dec(v_claim_584_);
goto v___jp_590_;
}
}
else
{
if (lean_obj_tag(v_polys_588_) == 1)
{
lean_object* v_head_596_; lean_object* v_tail_597_; lean_object* v_head_598_; lean_object* v_tail_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_640_; 
v_head_596_ = lean_ctor_get(v_gkrPoint_587_, 0);
lean_inc(v_head_596_);
v_tail_597_ = lean_ctor_get(v_gkrPoint_587_, 1);
lean_inc(v_tail_597_);
lean_dec_ref_known(v_gkrPoint_587_, 2);
v_head_598_ = lean_ctor_get(v_polys_588_, 0);
v_tail_599_ = lean_ctor_get(v_polys_588_, 1);
v_isSharedCheck_640_ = !lean_is_exclusive(v_polys_588_);
if (v_isSharedCheck_640_ == 0)
{
v___x_601_ = v_polys_588_;
v_isShared_602_ = v_isSharedCheck_640_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_tail_599_);
lean_inc(v_head_598_);
lean_dec(v_polys_588_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_640_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_603_; lean_object* v_e1_604_; lean_object* v___x_605_; lean_object* v_a_606_; lean_object* v_snd_607_; lean_object* v___x_608_; lean_object* v_e2_609_; lean_object* v___x_610_; lean_object* v_a_611_; lean_object* v_snd_612_; lean_object* v___x_613_; lean_object* v_e3_614_; lean_object* v___x_615_; lean_object* v_a_616_; lean_object* v_snd_617_; lean_object* v___x_618_; lean_object* v_a_619_; lean_object* v_toRingOps_620_; lean_object* v_toSemiringOps_621_; lean_object* v_fst_622_; lean_object* v_snd_623_; lean_object* v_sub_624_; lean_object* v_one_625_; lean_object* v_add_626_; lean_object* v_mul_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_637_; 
v___x_603_ = lean_unsigned_to_nat(0u);
v_e1_604_ = lean_array_fget(v_head_598_, v___x_603_);
lean_inc_n(v_e1_604_, 2);
lean_inc_ref_n(v_inst_582_, 4);
lean_inc_ref_n(v_inst_581_, 4);
v___x_605_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_581_, v_inst_582_, v_e1_604_, v_a_589_);
v_a_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_a_606_);
lean_dec_ref(v___x_605_);
v_snd_607_ = lean_ctor_get(v_a_606_, 1);
lean_inc(v_snd_607_);
lean_dec(v_a_606_);
v___x_608_ = lean_unsigned_to_nat(1u);
v_e2_609_ = lean_array_fget(v_head_598_, v___x_608_);
lean_inc(v_e2_609_);
v___x_610_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_581_, v_inst_582_, v_e2_609_, v_snd_607_);
v_a_611_ = lean_ctor_get(v___x_610_, 0);
lean_inc(v_a_611_);
lean_dec_ref(v___x_610_);
v_snd_612_ = lean_ctor_get(v_a_611_, 1);
lean_inc(v_snd_612_);
lean_dec(v_a_611_);
v___x_613_ = lean_unsigned_to_nat(2u);
v_e3_614_ = lean_array_fget(v_head_598_, v___x_613_);
lean_dec(v_head_598_);
lean_inc(v_e3_614_);
v___x_615_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_581_, v_inst_582_, v_e3_614_, v_snd_612_);
v_a_616_ = lean_ctor_get(v___x_615_, 0);
lean_inc(v_a_616_);
lean_dec_ref(v___x_615_);
v_snd_617_ = lean_ctor_get(v_a_616_, 1);
lean_inc(v_snd_617_);
lean_dec(v_a_616_);
v___x_618_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_581_, v_inst_582_, v_snd_617_);
v_a_619_ = lean_ctor_get(v___x_618_, 0);
lean_inc(v_a_619_);
lean_dec_ref(v___x_618_);
v_toRingOps_620_ = lean_ctor_get(v_fo_583_, 0);
v_toSemiringOps_621_ = lean_ctor_get(v_toRingOps_620_, 0);
v_fst_622_ = lean_ctor_get(v_a_619_, 0);
lean_inc_n(v_fst_622_, 4);
v_snd_623_ = lean_ctor_get(v_a_619_, 1);
lean_inc(v_snd_623_);
lean_dec(v_a_619_);
v_sub_624_ = lean_ctor_get(v_toRingOps_620_, 1);
v_one_625_ = lean_ctor_get(v_toSemiringOps_621_, 1);
v_add_626_ = lean_ctor_get(v_toSemiringOps_621_, 3);
v_mul_627_ = lean_ctor_get(v_toSemiringOps_621_, 4);
lean_inc_n(v_sub_624_, 3);
v___x_628_ = lean_apply_2(v_sub_624_, v_claim_584_, v_e1_604_);
lean_inc_ref(v_fo_583_);
v___x_629_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(v_fo_583_, v___x_628_, v_e1_604_, v_e2_609_, v_e3_614_, v_fst_622_);
lean_inc_n(v_mul_627_, 3);
lean_inc(v_head_596_);
v___x_630_ = lean_apply_2(v_mul_627_, v_head_596_, v_fst_622_);
lean_inc_n(v_one_625_, 2);
v___x_631_ = lean_apply_2(v_sub_624_, v_one_625_, v_head_596_);
v___x_632_ = lean_apply_2(v_sub_624_, v_one_625_, v_fst_622_);
v___x_633_ = lean_apply_2(v_mul_627_, v___x_631_, v___x_632_);
lean_inc(v_add_626_);
v___x_634_ = lean_apply_2(v_add_626_, v___x_630_, v___x_633_);
v___x_635_ = lean_apply_2(v_mul_627_, v_eqAtPoint_585_, v___x_634_);
if (v_isShared_602_ == 0)
{
lean_ctor_set(v___x_601_, 1, v_pointRev_586_);
lean_ctor_set(v___x_601_, 0, v_fst_622_);
v___x_637_ = v___x_601_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_fst_622_);
lean_ctor_set(v_reuseFailAlloc_639_, 1, v_pointRev_586_);
v___x_637_ = v_reuseFailAlloc_639_;
goto v_reusejp_636_;
}
v_reusejp_636_:
{
v_claim_584_ = v___x_629_;
v_eqAtPoint_585_ = v___x_635_;
v_pointRev_586_ = v___x_637_;
v_gkrPoint_587_ = v_tail_597_;
v_polys_588_ = v_tail_599_;
v_a_589_ = v_snd_623_;
goto _start;
}
}
}
else
{
lean_dec_ref_known(v_gkrPoint_587_, 2);
lean_dec_ref(v_a_589_);
lean_dec(v_polys_588_);
lean_dec(v_pointRev_586_);
lean_dec(v_eqAtPoint_585_);
lean_dec(v_claim_584_);
lean_dec_ref(v_fo_583_);
lean_dec_ref(v_inst_582_);
lean_dec_ref(v_inst_581_);
goto v___jp_590_;
}
}
v___jp_590_:
{
lean_object* v___x_591_; 
v___x_591_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0));
return v___x_591_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM(lean_object* v_F_641_, lean_object* v_EF_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_fo_645_, lean_object* v_claim_646_, lean_object* v_eqAtPoint_647_, lean_object* v_pointRev_648_, lean_object* v_gkrPoint_649_, lean_object* v_polys_650_, lean_object* v_a_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(v_inst_643_, v_inst_644_, v_fo_645_, v_claim_646_, v_eqAtPoint_647_, v_pointRev_648_, v_gkrPoint_649_, v_polys_650_, v_a_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_fo_655_, lean_object* v_proof_656_, lean_object* v_round_657_, lean_object* v_claim_658_, lean_object* v_gkrPoint_659_, lean_object* v_a_660_){
_start:
{
lean_object* v_sumcheckPolys_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v_sumcheckPolys_661_ = lean_ctor_get(v_proof_656_, 3);
v___x_662_ = lean_unsigned_to_nat(1u);
v___x_663_ = lean_nat_sub(v_round_657_, v___x_662_);
v___x_664_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_sumcheckPolys_661_, v___x_663_);
if (lean_obj_tag(v___x_664_) == 0)
{
lean_object* v___x_665_; 
lean_dec_ref_known(v___x_664_, 1);
lean_dec_ref(v_a_660_);
lean_dec(v_gkrPoint_659_);
lean_dec(v_claim_658_);
lean_dec_ref(v_fo_655_);
lean_dec_ref(v_inst_654_);
lean_dec_ref(v_inst_653_);
v___x_665_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0));
return v___x_665_;
}
else
{
lean_object* v_toRingOps_666_; lean_object* v_toSemiringOps_667_; lean_object* v_a_668_; lean_object* v_one_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v_toRingOps_666_ = lean_ctor_get(v_fo_655_, 0);
v_toSemiringOps_667_ = lean_ctor_get(v_toRingOps_666_, 0);
v_a_668_ = lean_ctor_get(v___x_664_, 0);
lean_inc(v_a_668_);
lean_dec_ref_known(v___x_664_, 1);
v_one_669_ = lean_ctor_get(v_toSemiringOps_667_, 1);
lean_inc(v_one_669_);
v___x_670_ = lean_box(0);
v___x_671_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(v_inst_653_, v_inst_654_, v_fo_655_, v_claim_658_, v_one_669_, v___x_670_, v_gkrPoint_659_, v_a_668_, v_a_660_);
return v___x_671_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg___boxed(lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_fo_674_, lean_object* v_proof_675_, lean_object* v_round_676_, lean_object* v_claim_677_, lean_object* v_gkrPoint_678_, lean_object* v_a_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(v_inst_672_, v_inst_673_, v_fo_674_, v_proof_675_, v_round_676_, v_claim_677_, v_gkrPoint_678_, v_a_679_);
lean_dec(v_round_676_);
lean_dec_ref(v_proof_675_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM(lean_object* v_F_681_, lean_object* v_EF_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_fo_685_, lean_object* v_proof_686_, lean_object* v_round_687_, lean_object* v_claim_688_, lean_object* v_gkrPoint_689_, lean_object* v_a_690_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(v_inst_683_, v_inst_684_, v_fo_685_, v_proof_686_, v_round_687_, v_claim_688_, v_gkrPoint_689_, v_a_690_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___boxed(lean_object* v_F_692_, lean_object* v_EF_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_fo_696_, lean_object* v_proof_697_, lean_object* v_round_698_, lean_object* v_claim_699_, lean_object* v_gkrPoint_700_, lean_object* v_a_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM(v_F_692_, v_EF_693_, v_inst_694_, v_inst_695_, v_fo_696_, v_proof_697_, v_round_698_, v_claim_699_, v_gkrPoint_700_, v_a_701_);
lean_dec(v_round_698_);
lean_dec_ref(v_proof_697_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_inst_708_, lean_object* v_fo_709_, lean_object* v_proof_710_, lean_object* v_round_711_, lean_object* v_remainingRounds_712_, lean_object* v_numeratorClaim_713_, lean_object* v_denominatorClaim_714_, lean_object* v_gkrPoint_715_, lean_object* v_a_716_){
_start:
{
lean_object* v___x_717_; uint8_t v___x_718_; 
v___x_717_ = lean_unsigned_to_nat(0u);
v___x_718_ = lean_nat_dec_eq(v_remainingRounds_712_, v___x_717_);
if (v___x_718_ == 0)
{
lean_object* v___x_719_; lean_object* v_a_720_; lean_object* v_toRingOps_721_; lean_object* v_toSemiringOps_722_; lean_object* v_fst_723_; lean_object* v_snd_724_; lean_object* v_add_725_; lean_object* v_mul_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; 
lean_inc_ref_n(v_inst_708_, 2);
lean_inc_ref_n(v_inst_707_, 2);
v___x_719_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_707_, v_inst_708_, v_a_716_);
v_a_720_ = lean_ctor_get(v___x_719_, 0);
lean_inc(v_a_720_);
lean_dec_ref(v___x_719_);
v_toRingOps_721_ = lean_ctor_get(v_fo_709_, 0);
v_toSemiringOps_722_ = lean_ctor_get(v_toRingOps_721_, 0);
v_fst_723_ = lean_ctor_get(v_a_720_, 0);
lean_inc_n(v_fst_723_, 2);
v_snd_724_ = lean_ctor_get(v_a_720_, 1);
lean_inc(v_snd_724_);
lean_dec(v_a_720_);
v_add_725_ = lean_ctor_get(v_toSemiringOps_722_, 3);
v_mul_726_ = lean_ctor_get(v_toSemiringOps_722_, 4);
lean_inc(v_mul_726_);
v___x_727_ = lean_apply_2(v_mul_726_, v_fst_723_, v_denominatorClaim_714_);
lean_inc(v_add_725_);
v___x_728_ = lean_apply_2(v_add_725_, v_numeratorClaim_713_, v___x_727_);
lean_inc_ref(v_fo_709_);
v___x_729_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(v_inst_707_, v_inst_708_, v_fo_709_, v_proof_710_, v_round_711_, v___x_728_, v_gkrPoint_715_, v_snd_724_);
if (lean_obj_tag(v___x_729_) == 0)
{
lean_object* v_a_730_; lean_object* v___x_732_; uint8_t v_isShared_733_; uint8_t v_isSharedCheck_737_; 
lean_dec(v_fst_723_);
lean_dec(v_remainingRounds_712_);
lean_dec(v_round_711_);
lean_dec_ref(v_fo_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_inst_707_);
lean_dec_ref(v_inst_706_);
v_a_730_ = lean_ctor_get(v___x_729_, 0);
v_isSharedCheck_737_ = !lean_is_exclusive(v___x_729_);
if (v_isSharedCheck_737_ == 0)
{
v___x_732_ = v___x_729_;
v_isShared_733_ = v_isSharedCheck_737_;
goto v_resetjp_731_;
}
else
{
lean_inc(v_a_730_);
lean_dec(v___x_729_);
v___x_732_ = lean_box(0);
v_isShared_733_ = v_isSharedCheck_737_;
goto v_resetjp_731_;
}
v_resetjp_731_:
{
lean_object* v___x_735_; 
if (v_isShared_733_ == 0)
{
v___x_735_ = v___x_732_;
goto v_reusejp_734_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v_a_730_);
v___x_735_ = v_reuseFailAlloc_736_;
goto v_reusejp_734_;
}
v_reusejp_734_:
{
return v___x_735_;
}
}
}
else
{
lean_object* v_a_738_; lean_object* v_fst_739_; lean_object* v_snd_740_; lean_object* v_claimsPerLayer_741_; lean_object* v___x_742_; 
v_a_738_ = lean_ctor_get(v___x_729_, 0);
lean_inc(v_a_738_);
lean_dec_ref_known(v___x_729_, 1);
v_fst_739_ = lean_ctor_get(v_a_738_, 0);
lean_inc(v_fst_739_);
v_snd_740_ = lean_ctor_get(v_a_738_, 1);
lean_inc(v_snd_740_);
lean_dec(v_a_738_);
v_claimsPerLayer_741_ = lean_ctor_get(v_proof_710_, 2);
lean_inc(v_round_711_);
v___x_742_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_claimsPerLayer_741_, v_round_711_);
if (lean_obj_tag(v___x_742_) == 0)
{
lean_object* v___x_743_; 
lean_dec_ref_known(v___x_742_, 1);
lean_dec(v_snd_740_);
lean_dec(v_fst_739_);
lean_dec(v_fst_723_);
lean_dec(v_remainingRounds_712_);
lean_dec(v_round_711_);
lean_dec_ref(v_fo_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_inst_707_);
lean_dec_ref(v_inst_706_);
v___x_743_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_743_;
}
else
{
lean_object* v_a_744_; lean_object* v___x_745_; lean_object* v_a_746_; lean_object* v_snd_747_; lean_object* v___x_748_; lean_object* v_fst_749_; lean_object* v_snd_750_; lean_object* v_claim_751_; lean_object* v_point_752_; lean_object* v_eqAtPoint_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; uint8_t v___x_758_; 
v_a_744_ = lean_ctor_get(v___x_742_, 0);
lean_inc_n(v_a_744_, 3);
lean_dec_ref_known(v___x_742_, 1);
lean_inc_ref(v_inst_708_);
lean_inc_ref(v_inst_707_);
v___x_745_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(v_inst_707_, v_inst_708_, v_a_744_, v_snd_740_);
v_a_746_ = lean_ctor_get(v___x_745_, 0);
lean_inc(v_a_746_);
lean_dec_ref(v___x_745_);
v_snd_747_ = lean_ctor_get(v_a_746_, 1);
lean_inc(v_snd_747_);
lean_dec(v_a_746_);
lean_inc_ref(v_fo_709_);
v___x_748_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(v_fo_709_, v_a_744_);
v_fst_749_ = lean_ctor_get(v___x_748_, 0);
lean_inc(v_fst_749_);
v_snd_750_ = lean_ctor_get(v___x_748_, 1);
lean_inc(v_snd_750_);
lean_dec_ref(v___x_748_);
v_claim_751_ = lean_ctor_get(v_fst_739_, 0);
lean_inc(v_claim_751_);
v_point_752_ = lean_ctor_get(v_fst_739_, 1);
lean_inc(v_point_752_);
v_eqAtPoint_753_ = lean_ctor_get(v_fst_739_, 2);
lean_inc(v_eqAtPoint_753_);
lean_dec(v_fst_739_);
lean_inc_n(v_mul_726_, 2);
v___x_754_ = lean_apply_2(v_mul_726_, v_fst_723_, v_snd_750_);
lean_inc(v_add_725_);
v___x_755_ = lean_apply_2(v_add_725_, v_fst_749_, v___x_754_);
v___x_756_ = lean_apply_2(v_mul_726_, v___x_755_, v_eqAtPoint_753_);
lean_inc_ref(v_inst_706_);
v___x_757_ = lean_apply_2(v_inst_706_, v___x_756_, v_claim_751_);
v___x_758_ = lean_unbox(v___x_757_);
if (v___x_758_ == 0)
{
lean_object* v___x_759_; 
lean_dec(v_point_752_);
lean_dec(v_snd_747_);
lean_dec(v_a_744_);
lean_dec(v_remainingRounds_712_);
lean_dec(v_round_711_);
lean_dec_ref(v_fo_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_inst_707_);
lean_dec_ref(v_inst_706_);
v___x_759_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_759_;
}
else
{
lean_object* v___x_760_; lean_object* v_a_761_; lean_object* v_fst_762_; lean_object* v_snd_763_; lean_object* v___x_764_; lean_object* v_fst_765_; lean_object* v_snd_766_; lean_object* v___x_768_; uint8_t v_isShared_769_; uint8_t v_isSharedCheck_777_; 
lean_inc_ref(v_inst_708_);
lean_inc_ref(v_inst_707_);
v___x_760_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_707_, v_inst_708_, v_snd_747_);
v_a_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc(v_a_761_);
lean_dec_ref(v___x_760_);
v_fst_762_ = lean_ctor_get(v_a_761_, 0);
lean_inc_n(v_fst_762_, 2);
v_snd_763_ = lean_ctor_get(v_a_761_, 1);
lean_inc(v_snd_763_);
lean_dec(v_a_761_);
lean_inc_ref(v_fo_709_);
v___x_764_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(v_fo_709_, v_a_744_, v_fst_762_);
v_fst_765_ = lean_ctor_get(v___x_764_, 0);
v_snd_766_ = lean_ctor_get(v___x_764_, 1);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_764_);
if (v_isSharedCheck_777_ == 0)
{
v___x_768_ = v___x_764_;
v_isShared_769_ = v_isSharedCheck_777_;
goto v_resetjp_767_;
}
else
{
lean_inc(v_snd_766_);
lean_inc(v_fst_765_);
lean_dec(v___x_764_);
v___x_768_ = lean_box(0);
v_isShared_769_ = v_isSharedCheck_777_;
goto v_resetjp_767_;
}
v_resetjp_767_:
{
lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_774_; 
v___x_770_ = lean_unsigned_to_nat(1u);
v___x_771_ = lean_nat_add(v_round_711_, v___x_770_);
lean_dec(v_round_711_);
v___x_772_ = lean_nat_sub(v_remainingRounds_712_, v___x_770_);
lean_dec(v_remainingRounds_712_);
if (v_isShared_769_ == 0)
{
lean_ctor_set_tag(v___x_768_, 1);
lean_ctor_set(v___x_768_, 1, v_point_752_);
lean_ctor_set(v___x_768_, 0, v_fst_762_);
v___x_774_ = v___x_768_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_fst_762_);
lean_ctor_set(v_reuseFailAlloc_776_, 1, v_point_752_);
v___x_774_ = v_reuseFailAlloc_776_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
v_round_711_ = v___x_771_;
v_remainingRounds_712_ = v___x_772_;
v_numeratorClaim_713_ = v_fst_765_;
v_denominatorClaim_714_ = v_snd_766_;
v_gkrPoint_715_ = v___x_774_;
v_a_716_ = v_snd_763_;
goto _start;
}
}
}
}
}
}
else
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; 
lean_dec(v_remainingRounds_712_);
lean_dec(v_round_711_);
lean_dec_ref(v_fo_709_);
lean_dec_ref(v_inst_708_);
lean_dec_ref(v_inst_707_);
lean_dec_ref(v_inst_706_);
v___x_778_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_778_, 0, v_numeratorClaim_713_);
lean_ctor_set(v___x_778_, 1, v_denominatorClaim_714_);
lean_ctor_set(v___x_778_, 2, v_gkrPoint_715_);
v___x_779_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_779_, 0, v___x_778_);
lean_ctor_set(v___x_779_, 1, v_a_716_);
v___x_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_780_, 0, v___x_779_);
return v___x_780_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___boxed(lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_fo_784_, lean_object* v_proof_785_, lean_object* v_round_786_, lean_object* v_remainingRounds_787_, lean_object* v_numeratorClaim_788_, lean_object* v_denominatorClaim_789_, lean_object* v_gkrPoint_790_, lean_object* v_a_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(v_inst_781_, v_inst_782_, v_inst_783_, v_fo_784_, v_proof_785_, v_round_786_, v_remainingRounds_787_, v_numeratorClaim_788_, v_denominatorClaim_789_, v_gkrPoint_790_, v_a_791_);
lean_dec_ref(v_proof_785_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM(lean_object* v_F_793_, lean_object* v_EF_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_fo_798_, lean_object* v_proof_799_, lean_object* v_round_800_, lean_object* v_remainingRounds_801_, lean_object* v_numeratorClaim_802_, lean_object* v_denominatorClaim_803_, lean_object* v_gkrPoint_804_, lean_object* v_a_805_){
_start:
{
lean_object* v___x_806_; 
v___x_806_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(v_inst_795_, v_inst_796_, v_inst_797_, v_fo_798_, v_proof_799_, v_round_800_, v_remainingRounds_801_, v_numeratorClaim_802_, v_denominatorClaim_803_, v_gkrPoint_804_, v_a_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___boxed(lean_object* v_F_807_, lean_object* v_EF_808_, lean_object* v_inst_809_, lean_object* v_inst_810_, lean_object* v_inst_811_, lean_object* v_fo_812_, lean_object* v_proof_813_, lean_object* v_round_814_, lean_object* v_remainingRounds_815_, lean_object* v_numeratorClaim_816_, lean_object* v_denominatorClaim_817_, lean_object* v_gkrPoint_818_, lean_object* v_a_819_){
_start:
{
lean_object* v_res_820_; 
v_res_820_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM(v_F_807_, v_EF_808_, v_inst_809_, v_inst_810_, v_inst_811_, v_fo_812_, v_proof_813_, v_round_814_, v_remainingRounds_815_, v_numeratorClaim_816_, v_denominatorClaim_817_, v_gkrPoint_818_, v_a_819_);
lean_dec_ref(v_proof_813_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter___redArg(lean_object* v_x_821_, lean_object* v_h__1_822_){
_start:
{
lean_object* v_fst_823_; lean_object* v_snd_824_; lean_object* v___x_825_; 
v_fst_823_ = lean_ctor_get(v_x_821_, 0);
lean_inc(v_fst_823_);
v_snd_824_ = lean_ctor_get(v_x_821_, 1);
lean_inc(v_snd_824_);
lean_dec_ref(v_x_821_);
v___x_825_ = lean_apply_2(v_h__1_822_, v_fst_823_, v_snd_824_);
return v___x_825_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter(lean_object* v_EF_826_, lean_object* v_motive_827_, lean_object* v_x_828_, lean_object* v_h__1_829_){
_start:
{
lean_object* v_fst_830_; lean_object* v_snd_831_; lean_object* v___x_832_; 
v_fst_830_ = lean_ctor_get(v_x_828_, 0);
lean_inc(v_fst_830_);
v_snd_831_ = lean_ctor_get(v_x_828_, 1);
lean_inc(v_snd_831_);
lean_dec_ref(v_x_828_);
v___x_832_ = lean_apply_2(v_h__1_829_, v_fst_830_, v_snd_831_);
return v___x_832_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(lean_object* v_inst_833_, lean_object* v_inst_834_, lean_object* v_inst_835_, lean_object* v_fo_836_, lean_object* v_proof_837_, lean_object* v_totalRounds_838_, lean_object* v_a_839_){
_start:
{
lean_object* v___x_842_; uint8_t v___x_843_; 
v___x_842_ = lean_unsigned_to_nat(0u);
v___x_843_ = lean_nat_dec_eq(v_totalRounds_838_, v___x_842_);
if (v___x_843_ == 0)
{
lean_object* v_q0Claim_844_; lean_object* v_claimsPerLayer_845_; lean_object* v_sumcheckPolys_846_; uint8_t v___y_848_; lean_object* v___x_888_; uint8_t v___x_889_; 
v_q0Claim_844_ = lean_ctor_get(v_proof_837_, 1);
v_claimsPerLayer_845_ = lean_ctor_get(v_proof_837_, 2);
v_sumcheckPolys_846_ = lean_ctor_get(v_proof_837_, 3);
v___x_888_ = l_List_lengthTR___redArg(v_claimsPerLayer_845_);
v___x_889_ = lean_nat_dec_eq(v___x_888_, v_totalRounds_838_);
lean_dec(v___x_888_);
if (v___x_889_ == 0)
{
lean_dec_ref(v_a_839_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
goto v___jp_840_;
}
else
{
if (v___x_843_ == 0)
{
lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; uint8_t v___x_893_; 
v___x_890_ = l_List_lengthTR___redArg(v_sumcheckPolys_846_);
v___x_891_ = lean_unsigned_to_nat(1u);
v___x_892_ = lean_nat_sub(v_totalRounds_838_, v___x_891_);
v___x_893_ = lean_nat_dec_eq(v___x_890_, v___x_892_);
lean_dec(v___x_892_);
lean_dec(v___x_890_);
if (v___x_893_ == 0)
{
v___y_848_ = v___x_889_;
goto v___jp_847_;
}
else
{
v___y_848_ = v___x_843_;
goto v___jp_847_;
}
}
else
{
lean_dec_ref(v_a_839_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
goto v___jp_840_;
}
}
v___jp_847_:
{
if (v___y_848_ == 0)
{
lean_object* v___x_849_; lean_object* v_a_850_; lean_object* v_snd_851_; lean_object* v___x_852_; 
lean_inc(v_q0Claim_844_);
lean_inc_ref(v_inst_835_);
lean_inc_ref(v_inst_834_);
v___x_849_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_834_, v_inst_835_, v_q0Claim_844_, v_a_839_);
v_a_850_ = lean_ctor_get(v___x_849_, 0);
lean_inc(v_a_850_);
lean_dec_ref(v___x_849_);
v_snd_851_ = lean_ctor_get(v_a_850_, 1);
lean_inc(v_snd_851_);
lean_dec(v_a_850_);
v___x_852_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_claimsPerLayer_845_, v___x_842_);
if (lean_obj_tag(v___x_852_) == 0)
{
lean_object* v___x_853_; 
lean_dec_ref_known(v___x_852_, 1);
lean_dec(v_snd_851_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
v___x_853_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_853_;
}
else
{
lean_object* v_a_854_; lean_object* v___x_855_; lean_object* v_a_856_; lean_object* v_snd_857_; lean_object* v___x_858_; lean_object* v_toRingOps_859_; lean_object* v_toSemiringOps_860_; lean_object* v_fst_861_; lean_object* v_snd_862_; lean_object* v_zero_863_; lean_object* v___x_864_; uint8_t v___x_865_; 
v_a_854_ = lean_ctor_get(v___x_852_, 0);
lean_inc_n(v_a_854_, 3);
lean_dec_ref_known(v___x_852_, 1);
lean_inc_ref(v_inst_835_);
lean_inc_ref(v_inst_834_);
v___x_855_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(v_inst_834_, v_inst_835_, v_a_854_, v_snd_851_);
v_a_856_ = lean_ctor_get(v___x_855_, 0);
lean_inc(v_a_856_);
lean_dec_ref(v___x_855_);
v_snd_857_ = lean_ctor_get(v_a_856_, 1);
lean_inc(v_snd_857_);
lean_dec(v_a_856_);
lean_inc_ref(v_fo_836_);
v___x_858_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(v_fo_836_, v_a_854_);
v_toRingOps_859_ = lean_ctor_get(v_fo_836_, 0);
v_toSemiringOps_860_ = lean_ctor_get(v_toRingOps_859_, 0);
v_fst_861_ = lean_ctor_get(v___x_858_, 0);
lean_inc(v_fst_861_);
v_snd_862_ = lean_ctor_get(v___x_858_, 1);
lean_inc(v_snd_862_);
lean_dec_ref(v___x_858_);
v_zero_863_ = lean_ctor_get(v_toSemiringOps_860_, 0);
lean_inc_ref(v_inst_833_);
lean_inc(v_zero_863_);
v___x_864_ = lean_apply_2(v_inst_833_, v_fst_861_, v_zero_863_);
v___x_865_ = lean_unbox(v___x_864_);
if (v___x_865_ == 0)
{
lean_object* v___x_866_; 
lean_dec(v_snd_862_);
lean_dec(v_snd_857_);
lean_dec(v_a_854_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
v___x_866_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_866_;
}
else
{
lean_object* v___x_867_; uint8_t v___x_868_; 
lean_inc_ref(v_inst_833_);
lean_inc(v_q0Claim_844_);
v___x_867_ = lean_apply_2(v_inst_833_, v_snd_862_, v_q0Claim_844_);
v___x_868_ = lean_unbox(v___x_867_);
if (v___x_868_ == 0)
{
lean_object* v___x_869_; 
lean_dec(v_snd_857_);
lean_dec(v_a_854_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
v___x_869_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_869_;
}
else
{
lean_object* v___x_870_; lean_object* v_a_871_; lean_object* v_fst_872_; lean_object* v_snd_873_; lean_object* v___x_874_; lean_object* v_fst_875_; lean_object* v_snd_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_887_; 
lean_inc_ref(v_inst_835_);
lean_inc_ref(v_inst_834_);
v___x_870_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_834_, v_inst_835_, v_snd_857_);
v_a_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_a_871_);
lean_dec_ref(v___x_870_);
v_fst_872_ = lean_ctor_get(v_a_871_, 0);
lean_inc_n(v_fst_872_, 2);
v_snd_873_ = lean_ctor_get(v_a_871_, 1);
lean_inc(v_snd_873_);
lean_dec(v_a_871_);
lean_inc_ref(v_fo_836_);
v___x_874_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(v_fo_836_, v_a_854_, v_fst_872_);
v_fst_875_ = lean_ctor_get(v___x_874_, 0);
v_snd_876_ = lean_ctor_get(v___x_874_, 1);
v_isSharedCheck_887_ = !lean_is_exclusive(v___x_874_);
if (v_isSharedCheck_887_ == 0)
{
v___x_878_ = v___x_874_;
v_isShared_879_ = v_isSharedCheck_887_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_snd_876_);
lean_inc(v_fst_875_);
lean_dec(v___x_874_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_887_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_884_; 
v___x_880_ = lean_unsigned_to_nat(1u);
v___x_881_ = lean_nat_sub(v_totalRounds_838_, v___x_880_);
v___x_882_ = lean_box(0);
if (v_isShared_879_ == 0)
{
lean_ctor_set_tag(v___x_878_, 1);
lean_ctor_set(v___x_878_, 1, v___x_882_);
lean_ctor_set(v___x_878_, 0, v_fst_872_);
v___x_884_ = v___x_878_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_886_; 
v_reuseFailAlloc_886_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_886_, 0, v_fst_872_);
lean_ctor_set(v_reuseFailAlloc_886_, 1, v___x_882_);
v___x_884_ = v_reuseFailAlloc_886_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
lean_object* v___x_885_; 
v___x_885_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(v_inst_833_, v_inst_834_, v_inst_835_, v_fo_836_, v_proof_837_, v___x_880_, v___x_881_, v_fst_875_, v_snd_876_, v___x_884_, v_snd_873_);
lean_dec_ref(v_proof_837_);
return v___x_885_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_a_839_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
goto v___jp_840_;
}
}
}
else
{
lean_object* v___x_894_; 
lean_dec_ref(v_a_839_);
lean_dec_ref(v_proof_837_);
lean_dec_ref(v_fo_836_);
lean_dec_ref(v_inst_835_);
lean_dec_ref(v_inst_834_);
lean_dec_ref(v_inst_833_);
v___x_894_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_894_;
}
v___jp_840_:
{
lean_object* v___x_841_; 
v___x_841_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_841_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg___boxed(lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_fo_898_, lean_object* v_proof_899_, lean_object* v_totalRounds_900_, lean_object* v_a_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(v_inst_895_, v_inst_896_, v_inst_897_, v_fo_898_, v_proof_899_, v_totalRounds_900_, v_a_901_);
lean_dec(v_totalRounds_900_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM(lean_object* v_F_903_, lean_object* v_EF_904_, lean_object* v_inst_905_, lean_object* v_inst_906_, lean_object* v_inst_907_, lean_object* v_fo_908_, lean_object* v_proof_909_, lean_object* v_totalRounds_910_, lean_object* v_a_911_){
_start:
{
lean_object* v___x_912_; 
v___x_912_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(v_inst_905_, v_inst_906_, v_inst_907_, v_fo_908_, v_proof_909_, v_totalRounds_910_, v_a_911_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___boxed(lean_object* v_F_913_, lean_object* v_EF_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_fo_918_, lean_object* v_proof_919_, lean_object* v_totalRounds_920_, lean_object* v_a_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM(v_F_913_, v_EF_914_, v_inst_915_, v_inst_916_, v_inst_917_, v_fo_918_, v_proof_919_, v_totalRounds_920_, v_a_921_);
lean_dec(v_totalRounds_920_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__0(lean_object* v_toSemiringOps_923_, lean_object* v_acc_924_, lean_object* v_x_925_){
_start:
{
lean_object* v_mul_926_; lean_object* v___x_927_; 
v_mul_926_ = lean_ctor_get(v_toSemiringOps_923_, 4);
lean_inc(v_mul_926_);
lean_dec_ref(v_toSemiringOps_923_);
v___x_927_ = lean_apply_2(v_mul_926_, v_acc_924_, v_x_925_);
return v___x_927_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__1(lean_object* v_toSemiringOps_928_, lean_object* v_sub_929_, lean_object* v_acc_930_, lean_object* v_x_931_){
_start:
{
lean_object* v_one_932_; lean_object* v_mul_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v_one_932_ = lean_ctor_get(v_toSemiringOps_928_, 1);
lean_inc(v_one_932_);
v_mul_933_ = lean_ctor_get(v_toSemiringOps_928_, 4);
lean_inc(v_mul_933_);
lean_dec_ref(v_toSemiringOps_928_);
v___x_934_ = lean_apply_2(v_sub_929_, v_one_932_, v_x_931_);
v___x_935_ = lean_apply_2(v_mul_933_, v_acc_930_, v___x_934_);
return v___x_935_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(lean_object* v_fo_936_, lean_object* v_inst_937_, lean_object* v_preprocessed_938_, lean_object* v_partitionedMain_939_, lean_object* v_publicValues_940_, lean_object* v_rs_941_, lean_object* v_lSkip_942_){
_start:
{
lean_object* v_toRingOps_943_; lean_object* v_inv_944_; lean_object* v_toSemiringOps_945_; lean_object* v_sub_946_; lean_object* v___f_947_; lean_object* v___f_948_; lean_object* v___y_950_; lean_object* v___y_951_; lean_object* v___y_972_; 
v_toRingOps_943_ = lean_ctor_get(v_fo_936_, 0);
v_inv_944_ = lean_ctor_get(v_fo_936_, 1);
v_toSemiringOps_945_ = lean_ctor_get(v_toRingOps_943_, 0);
v_sub_946_ = lean_ctor_get(v_toRingOps_943_, 1);
lean_inc_ref_n(v_toSemiringOps_945_, 2);
v___f_947_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_947_, 0, v_toSemiringOps_945_);
lean_inc(v_sub_946_);
v___f_948_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__1), 4, 2);
lean_closure_set(v___f_948_, 0, v_toSemiringOps_945_);
lean_closure_set(v___f_948_, 1, v_sub_946_);
if (lean_obj_tag(v_rs_941_) == 0)
{
lean_object* v_zero_974_; 
v_zero_974_ = lean_ctor_get(v_toSemiringOps_945_, 0);
lean_inc(v_zero_974_);
v___y_972_ = v_zero_974_;
goto v___jp_971_;
}
else
{
lean_object* v_head_975_; 
v_head_975_ = lean_ctor_get(v_rs_941_, 0);
lean_inc(v_head_975_);
v___y_972_ = v_head_975_;
goto v___jp_971_;
}
v___jp_949_:
{
lean_object* v_one_952_; lean_object* v_natCast_953_; lean_object* v_mul_954_; lean_object* v_pow_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v_inv_959_; lean_object* v_omega_960_; lean_object* v_prodOneMinus_961_; lean_object* v_prodX_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; 
v_one_952_ = lean_ctor_get(v_toSemiringOps_945_, 1);
v_natCast_953_ = lean_ctor_get(v_toSemiringOps_945_, 2);
v_mul_954_ = lean_ctor_get(v_toSemiringOps_945_, 4);
lean_inc_n(v_mul_954_, 5);
v_pow_955_ = lean_ctor_get(v_toSemiringOps_945_, 5);
v___x_956_ = lean_unsigned_to_nat(2u);
lean_inc(v_natCast_953_);
v___x_957_ = lean_apply_1(v_natCast_953_, v___x_956_);
lean_inc(v_pow_955_);
lean_inc_n(v_lSkip_942_, 3);
v___x_958_ = lean_apply_2(v_pow_955_, v___x_957_, v_lSkip_942_);
lean_inc(v_inv_944_);
v_inv_959_ = lean_apply_1(v_inv_944_, v___x_958_);
v_omega_960_ = lean_apply_1(v_inst_937_, v_lSkip_942_);
lean_inc(v___y_951_);
lean_inc_n(v_one_952_, 2);
v_prodOneMinus_961_ = l_List_foldl___redArg(v___f_948_, v_one_952_, v___y_951_);
v_prodX_962_ = l_List_foldl___redArg(v___f_947_, v_one_952_, v___y_951_);
lean_inc(v___y_950_);
lean_inc_ref(v_fo_936_);
v___x_963_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(v_fo_936_, v___y_950_, v_lSkip_942_);
lean_inc(v_inv_959_);
v___x_964_ = lean_apply_2(v_mul_954_, v_inv_959_, v___x_963_);
v___x_965_ = lean_apply_2(v_mul_954_, v___x_964_, v_prodOneMinus_961_);
v___x_966_ = lean_apply_2(v_mul_954_, v___y_950_, v_omega_960_);
v___x_967_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(v_fo_936_, v___x_966_, v_lSkip_942_);
v___x_968_ = lean_apply_2(v_mul_954_, v_inv_959_, v___x_967_);
v___x_969_ = lean_apply_2(v_mul_954_, v___x_968_, v_prodX_962_);
v___x_970_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_970_, 0, v_preprocessed_938_);
lean_ctor_set(v___x_970_, 1, v_partitionedMain_939_);
lean_ctor_set(v___x_970_, 2, v_publicValues_940_);
lean_ctor_set(v___x_970_, 3, v___x_965_);
lean_ctor_set(v___x_970_, 4, v___x_969_);
return v___x_970_;
}
v___jp_971_:
{
if (lean_obj_tag(v_rs_941_) == 0)
{
v___y_950_ = v___y_972_;
v___y_951_ = v_rs_941_;
goto v___jp_949_;
}
else
{
lean_object* v_tail_973_; 
v_tail_973_ = lean_ctor_get(v_rs_941_, 1);
lean_inc(v_tail_973_);
lean_dec_ref_known(v_rs_941_, 2);
v___y_950_ = v___y_972_;
v___y_951_ = v_tail_973_;
goto v___jp_949_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27(lean_object* v_F_976_, lean_object* v_EF_977_, lean_object* v_fo_978_, lean_object* v_inst_979_, lean_object* v_preprocessed_980_, lean_object* v_partitionedMain_981_, lean_object* v_publicValues_982_, lean_object* v_rs_983_, lean_object* v_lSkip_984_){
_start:
{
lean_object* v___x_985_; 
v___x_985_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(v_fo_978_, v_inst_979_, v_preprocessed_980_, v_partitionedMain_981_, v_publicValues_982_, v_rs_983_, v_lSkip_984_);
return v___x_985_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst___redArg(lean_object* v_algMap_986_, lean_object* v_c_987_){
_start:
{
lean_object* v___x_988_; 
v___x_988_ = lean_apply_1(v_algMap_986_, v_c_987_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst(lean_object* v_F_989_, lean_object* v_EF_990_, lean_object* v_algMap_991_, lean_object* v_c_992_){
_start:
{
lean_object* v___x_993_; 
v___x_993_ = lean_apply_1(v_algMap_991_, v_c_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(lean_object* v_fo_994_, lean_object* v_algMap_995_, lean_object* v_e_996_, lean_object* v_v_997_){
_start:
{
lean_object* v_entry_998_; 
v_entry_998_ = lean_ctor_get(v_v_997_, 0);
switch(lean_obj_tag(v_entry_998_))
{
case 0:
{
lean_object* v_preprocessed_999_; 
lean_inc_ref(v_entry_998_);
lean_dec(v_algMap_995_);
v_preprocessed_999_ = lean_ctor_get(v_e_996_, 0);
if (lean_obj_tag(v_preprocessed_999_) == 0)
{
lean_object* v_toRingOps_1000_; lean_object* v_toSemiringOps_1001_; lean_object* v_zero_1002_; 
lean_dec_ref_known(v_entry_998_, 1);
lean_dec_ref(v_v_997_);
v_toRingOps_1000_ = lean_ctor_get(v_fo_994_, 0);
v_toSemiringOps_1001_ = lean_ctor_get(v_toRingOps_1000_, 0);
v_zero_1002_ = lean_ctor_get(v_toSemiringOps_1001_, 0);
lean_inc(v_zero_1002_);
return v_zero_1002_;
}
else
{
lean_object* v_index_1003_; lean_object* v_offset_1004_; lean_object* v_val_1005_; lean_object* v___x_1006_; 
v_index_1003_ = lean_ctor_get(v_v_997_, 1);
lean_inc(v_index_1003_);
lean_dec_ref(v_v_997_);
v_offset_1004_ = lean_ctor_get(v_entry_998_, 0);
lean_inc(v_offset_1004_);
lean_dec_ref_known(v_entry_998_, 1);
v_val_1005_ = lean_ctor_get(v_preprocessed_999_, 0);
v___x_1006_ = l_List_get_x3fInternal___redArg(v_val_1005_, v_index_1003_);
if (lean_obj_tag(v___x_1006_) == 0)
{
lean_object* v_toRingOps_1007_; lean_object* v_toSemiringOps_1008_; lean_object* v_zero_1009_; 
lean_dec(v_offset_1004_);
v_toRingOps_1007_ = lean_ctor_get(v_fo_994_, 0);
v_toSemiringOps_1008_ = lean_ctor_get(v_toRingOps_1007_, 0);
v_zero_1009_ = lean_ctor_get(v_toSemiringOps_1008_, 0);
lean_inc(v_zero_1009_);
return v_zero_1009_;
}
else
{
lean_object* v_val_1010_; lean_object* v_fst_1011_; lean_object* v_snd_1012_; lean_object* v___x_1013_; uint8_t v___x_1014_; 
v_val_1010_ = lean_ctor_get(v___x_1006_, 0);
lean_inc(v_val_1010_);
lean_dec_ref_known(v___x_1006_, 1);
v_fst_1011_ = lean_ctor_get(v_val_1010_, 0);
lean_inc(v_fst_1011_);
v_snd_1012_ = lean_ctor_get(v_val_1010_, 1);
lean_inc(v_snd_1012_);
lean_dec(v_val_1010_);
v___x_1013_ = lean_unsigned_to_nat(0u);
v___x_1014_ = lean_nat_dec_eq(v_offset_1004_, v___x_1013_);
lean_dec(v_offset_1004_);
if (v___x_1014_ == 0)
{
lean_dec(v_fst_1011_);
return v_snd_1012_;
}
else
{
lean_dec(v_snd_1012_);
return v_fst_1011_;
}
}
}
}
case 1:
{
lean_object* v_index_1015_; lean_object* v_partIndex_1016_; lean_object* v_offset_1017_; lean_object* v_partitionedMain_1018_; lean_object* v___x_1019_; 
lean_inc_ref(v_entry_998_);
lean_dec(v_algMap_995_);
v_index_1015_ = lean_ctor_get(v_v_997_, 1);
lean_inc(v_index_1015_);
lean_dec_ref(v_v_997_);
v_partIndex_1016_ = lean_ctor_get(v_entry_998_, 0);
lean_inc(v_partIndex_1016_);
v_offset_1017_ = lean_ctor_get(v_entry_998_, 1);
lean_inc(v_offset_1017_);
lean_dec_ref_known(v_entry_998_, 2);
v_partitionedMain_1018_ = lean_ctor_get(v_e_996_, 1);
v___x_1019_ = l_List_get_x3fInternal___redArg(v_partitionedMain_1018_, v_partIndex_1016_);
if (lean_obj_tag(v___x_1019_) == 0)
{
lean_object* v_toRingOps_1020_; lean_object* v_toSemiringOps_1021_; lean_object* v_zero_1022_; 
lean_dec(v_offset_1017_);
lean_dec(v_index_1015_);
v_toRingOps_1020_ = lean_ctor_get(v_fo_994_, 0);
v_toSemiringOps_1021_ = lean_ctor_get(v_toRingOps_1020_, 0);
v_zero_1022_ = lean_ctor_get(v_toSemiringOps_1021_, 0);
lean_inc(v_zero_1022_);
return v_zero_1022_;
}
else
{
lean_object* v_val_1023_; lean_object* v___x_1024_; 
v_val_1023_ = lean_ctor_get(v___x_1019_, 0);
lean_inc(v_val_1023_);
lean_dec_ref_known(v___x_1019_, 1);
v___x_1024_ = l_List_get_x3fInternal___redArg(v_val_1023_, v_index_1015_);
lean_dec(v_val_1023_);
if (lean_obj_tag(v___x_1024_) == 0)
{
lean_object* v_toRingOps_1025_; lean_object* v_toSemiringOps_1026_; lean_object* v_zero_1027_; 
lean_dec(v_offset_1017_);
v_toRingOps_1025_ = lean_ctor_get(v_fo_994_, 0);
v_toSemiringOps_1026_ = lean_ctor_get(v_toRingOps_1025_, 0);
v_zero_1027_ = lean_ctor_get(v_toSemiringOps_1026_, 0);
lean_inc(v_zero_1027_);
return v_zero_1027_;
}
else
{
lean_object* v_val_1028_; lean_object* v_fst_1029_; lean_object* v_snd_1030_; lean_object* v___x_1031_; uint8_t v___x_1032_; 
v_val_1028_ = lean_ctor_get(v___x_1024_, 0);
lean_inc(v_val_1028_);
lean_dec_ref_known(v___x_1024_, 1);
v_fst_1029_ = lean_ctor_get(v_val_1028_, 0);
lean_inc(v_fst_1029_);
v_snd_1030_ = lean_ctor_get(v_val_1028_, 1);
lean_inc(v_snd_1030_);
lean_dec(v_val_1028_);
v___x_1031_ = lean_unsigned_to_nat(0u);
v___x_1032_ = lean_nat_dec_eq(v_offset_1017_, v___x_1031_);
lean_dec(v_offset_1017_);
if (v___x_1032_ == 0)
{
lean_dec(v_fst_1029_);
return v_snd_1030_;
}
else
{
lean_dec(v_snd_1030_);
return v_fst_1029_;
}
}
}
}
case 2:
{
lean_object* v_index_1033_; lean_object* v_publicValues_1034_; lean_object* v___x_1035_; 
v_index_1033_ = lean_ctor_get(v_v_997_, 1);
lean_inc(v_index_1033_);
lean_dec_ref(v_v_997_);
v_publicValues_1034_ = lean_ctor_get(v_e_996_, 2);
v___x_1035_ = l_List_get_x3fInternal___redArg(v_publicValues_1034_, v_index_1033_);
if (lean_obj_tag(v___x_1035_) == 0)
{
lean_object* v_toRingOps_1036_; lean_object* v_toSemiringOps_1037_; lean_object* v_zero_1038_; 
lean_dec(v_algMap_995_);
v_toRingOps_1036_ = lean_ctor_get(v_fo_994_, 0);
v_toSemiringOps_1037_ = lean_ctor_get(v_toRingOps_1036_, 0);
v_zero_1038_ = lean_ctor_get(v_toSemiringOps_1037_, 0);
lean_inc(v_zero_1038_);
return v_zero_1038_;
}
else
{
lean_object* v_val_1039_; lean_object* v___x_1040_; 
v_val_1039_ = lean_ctor_get(v___x_1035_, 0);
lean_inc(v_val_1039_);
lean_dec_ref_known(v___x_1035_, 1);
v___x_1040_ = lean_apply_1(v_algMap_995_, v_val_1039_);
return v___x_1040_;
}
}
default: 
{
lean_object* v_toRingOps_1041_; lean_object* v_toSemiringOps_1042_; lean_object* v_zero_1043_; 
lean_dec_ref(v_v_997_);
lean_dec(v_algMap_995_);
v_toRingOps_1041_ = lean_ctor_get(v_fo_994_, 0);
v_toSemiringOps_1042_ = lean_ctor_get(v_toRingOps_1041_, 0);
v_zero_1043_ = lean_ctor_get(v_toSemiringOps_1042_, 0);
lean_inc(v_zero_1043_);
return v_zero_1043_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg___boxed(lean_object* v_fo_1044_, lean_object* v_algMap_1045_, lean_object* v_e_1046_, lean_object* v_v_1047_){
_start:
{
lean_object* v_res_1048_; 
v_res_1048_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(v_fo_1044_, v_algMap_1045_, v_e_1046_, v_v_1047_);
lean_dec_ref(v_e_1046_);
lean_dec_ref(v_fo_1044_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar(lean_object* v_F_1049_, lean_object* v_EF_1050_, lean_object* v_fo_1051_, lean_object* v_algMap_1052_, lean_object* v_e_1053_, lean_object* v_v_1054_){
_start:
{
lean_object* v___x_1055_; 
v___x_1055_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(v_fo_1051_, v_algMap_1052_, v_e_1053_, v_v_1054_);
return v___x_1055_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___boxed(lean_object* v_F_1056_, lean_object* v_EF_1057_, lean_object* v_fo_1058_, lean_object* v_algMap_1059_, lean_object* v_e_1060_, lean_object* v_v_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar(v_F_1056_, v_EF_1057_, v_fo_1058_, v_algMap_1059_, v_e_1060_, v_v_1061_);
lean_dec_ref(v_e_1060_);
lean_dec_ref(v_fo_1058_);
return v_res_1062_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(lean_object* v_algMap_1063_, lean_object* v_e_1064_, lean_object* v_v_1065_){
_start:
{
lean_object* v_entry_1066_; 
v_entry_1066_ = lean_ctor_get(v_v_1065_, 0);
switch(lean_obj_tag(v_entry_1066_))
{
case 0:
{
lean_object* v_index_1067_; lean_object* v_offset_1068_; lean_object* v_preprocessed_1069_; lean_object* v___x_1070_; 
lean_inc_ref(v_entry_1066_);
lean_dec(v_algMap_1063_);
v_index_1067_ = lean_ctor_get(v_v_1065_, 1);
lean_inc(v_index_1067_);
lean_dec_ref(v_v_1065_);
v_offset_1068_ = lean_ctor_get(v_entry_1066_, 0);
lean_inc(v_offset_1068_);
lean_dec_ref_known(v_entry_1066_, 1);
v_preprocessed_1069_ = lean_ctor_get(v_e_1064_, 0);
lean_inc(v_preprocessed_1069_);
lean_dec_ref(v_e_1064_);
v___x_1070_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v_preprocessed_1069_);
if (lean_obj_tag(v___x_1070_) == 0)
{
lean_object* v___x_1071_; 
lean_dec_ref_known(v___x_1070_, 1);
lean_dec(v_offset_1068_);
lean_dec(v_index_1067_);
v___x_1071_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1071_;
}
else
{
lean_object* v_a_1072_; lean_object* v___x_1073_; 
v_a_1072_ = lean_ctor_get(v___x_1070_, 0);
lean_inc(v_a_1072_);
lean_dec_ref_known(v___x_1070_, 1);
v___x_1073_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_1072_, v_index_1067_);
lean_dec(v_a_1072_);
if (lean_obj_tag(v___x_1073_) == 0)
{
lean_object* v___x_1074_; 
lean_dec_ref_known(v___x_1073_, 1);
lean_dec(v_offset_1068_);
v___x_1074_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1074_;
}
else
{
lean_object* v_a_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1089_; 
v_a_1075_ = lean_ctor_get(v___x_1073_, 0);
v_isSharedCheck_1089_ = !lean_is_exclusive(v___x_1073_);
if (v_isSharedCheck_1089_ == 0)
{
v___x_1077_ = v___x_1073_;
v_isShared_1078_ = v_isSharedCheck_1089_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_a_1075_);
lean_dec(v___x_1073_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1089_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
lean_object* v___x_1079_; uint8_t v___x_1080_; 
v___x_1079_ = lean_unsigned_to_nat(0u);
v___x_1080_ = lean_nat_dec_eq(v_offset_1068_, v___x_1079_);
lean_dec(v_offset_1068_);
if (v___x_1080_ == 0)
{
lean_object* v_snd_1081_; lean_object* v___x_1083_; 
v_snd_1081_ = lean_ctor_get(v_a_1075_, 1);
lean_inc(v_snd_1081_);
lean_dec(v_a_1075_);
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 0, v_snd_1081_);
v___x_1083_ = v___x_1077_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_snd_1081_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
return v___x_1083_;
}
}
else
{
lean_object* v_fst_1085_; lean_object* v___x_1087_; 
v_fst_1085_ = lean_ctor_get(v_a_1075_, 0);
lean_inc(v_fst_1085_);
lean_dec(v_a_1075_);
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 0, v_fst_1085_);
v___x_1087_ = v___x_1077_;
goto v_reusejp_1086_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v_fst_1085_);
v___x_1087_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1086_;
}
v_reusejp_1086_:
{
return v___x_1087_;
}
}
}
}
}
}
case 1:
{
lean_object* v_index_1090_; lean_object* v_partIndex_1091_; lean_object* v_offset_1092_; lean_object* v_partitionedMain_1093_; lean_object* v___x_1094_; 
lean_inc_ref(v_entry_1066_);
lean_dec(v_algMap_1063_);
v_index_1090_ = lean_ctor_get(v_v_1065_, 1);
lean_inc(v_index_1090_);
lean_dec_ref(v_v_1065_);
v_partIndex_1091_ = lean_ctor_get(v_entry_1066_, 0);
lean_inc(v_partIndex_1091_);
v_offset_1092_ = lean_ctor_get(v_entry_1066_, 1);
lean_inc(v_offset_1092_);
lean_dec_ref_known(v_entry_1066_, 2);
v_partitionedMain_1093_ = lean_ctor_get(v_e_1064_, 1);
lean_inc(v_partitionedMain_1093_);
lean_dec_ref(v_e_1064_);
v___x_1094_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_partitionedMain_1093_, v_partIndex_1091_);
lean_dec(v_partitionedMain_1093_);
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v___x_1095_; 
lean_dec_ref_known(v___x_1094_, 1);
lean_dec(v_offset_1092_);
lean_dec(v_index_1090_);
v___x_1095_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1095_;
}
else
{
lean_object* v_a_1096_; lean_object* v___x_1097_; 
v_a_1096_ = lean_ctor_get(v___x_1094_, 0);
lean_inc(v_a_1096_);
lean_dec_ref_known(v___x_1094_, 1);
v___x_1097_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_1096_, v_index_1090_);
lean_dec(v_a_1096_);
if (lean_obj_tag(v___x_1097_) == 0)
{
lean_object* v___x_1098_; 
lean_dec_ref_known(v___x_1097_, 1);
lean_dec(v_offset_1092_);
v___x_1098_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1098_;
}
else
{
lean_object* v_a_1099_; lean_object* v___x_1101_; uint8_t v_isShared_1102_; uint8_t v_isSharedCheck_1113_; 
v_a_1099_ = lean_ctor_get(v___x_1097_, 0);
v_isSharedCheck_1113_ = !lean_is_exclusive(v___x_1097_);
if (v_isSharedCheck_1113_ == 0)
{
v___x_1101_ = v___x_1097_;
v_isShared_1102_ = v_isSharedCheck_1113_;
goto v_resetjp_1100_;
}
else
{
lean_inc(v_a_1099_);
lean_dec(v___x_1097_);
v___x_1101_ = lean_box(0);
v_isShared_1102_ = v_isSharedCheck_1113_;
goto v_resetjp_1100_;
}
v_resetjp_1100_:
{
lean_object* v___x_1103_; uint8_t v___x_1104_; 
v___x_1103_ = lean_unsigned_to_nat(0u);
v___x_1104_ = lean_nat_dec_eq(v_offset_1092_, v___x_1103_);
lean_dec(v_offset_1092_);
if (v___x_1104_ == 0)
{
lean_object* v_snd_1105_; lean_object* v___x_1107_; 
v_snd_1105_ = lean_ctor_get(v_a_1099_, 1);
lean_inc(v_snd_1105_);
lean_dec(v_a_1099_);
if (v_isShared_1102_ == 0)
{
lean_ctor_set(v___x_1101_, 0, v_snd_1105_);
v___x_1107_ = v___x_1101_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v_snd_1105_);
v___x_1107_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
return v___x_1107_;
}
}
else
{
lean_object* v_fst_1109_; lean_object* v___x_1111_; 
v_fst_1109_ = lean_ctor_get(v_a_1099_, 0);
lean_inc(v_fst_1109_);
lean_dec(v_a_1099_);
if (v_isShared_1102_ == 0)
{
lean_ctor_set(v___x_1101_, 0, v_fst_1109_);
v___x_1111_ = v___x_1101_;
goto v_reusejp_1110_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v_fst_1109_);
v___x_1111_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1110_;
}
v_reusejp_1110_:
{
return v___x_1111_;
}
}
}
}
}
}
case 2:
{
lean_object* v_index_1114_; lean_object* v_publicValues_1115_; lean_object* v___x_1116_; 
v_index_1114_ = lean_ctor_get(v_v_1065_, 1);
lean_inc(v_index_1114_);
lean_dec_ref(v_v_1065_);
v_publicValues_1115_ = lean_ctor_get(v_e_1064_, 2);
lean_inc(v_publicValues_1115_);
lean_dec_ref(v_e_1064_);
v___x_1116_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_publicValues_1115_, v_index_1114_);
lean_dec(v_publicValues_1115_);
if (lean_obj_tag(v___x_1116_) == 0)
{
lean_object* v___x_1117_; 
lean_dec_ref_known(v___x_1116_, 1);
lean_dec(v_algMap_1063_);
v___x_1117_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1117_;
}
else
{
lean_object* v_a_1118_; lean_object* v___x_1120_; uint8_t v_isShared_1121_; uint8_t v_isSharedCheck_1126_; 
v_a_1118_ = lean_ctor_get(v___x_1116_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1116_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1120_ = v___x_1116_;
v_isShared_1121_ = v_isSharedCheck_1126_;
goto v_resetjp_1119_;
}
else
{
lean_inc(v_a_1118_);
lean_dec(v___x_1116_);
v___x_1120_ = lean_box(0);
v_isShared_1121_ = v_isSharedCheck_1126_;
goto v_resetjp_1119_;
}
v_resetjp_1119_:
{
lean_object* v___x_1122_; lean_object* v___x_1124_; 
v___x_1122_ = lean_apply_1(v_algMap_1063_, v_a_1118_);
if (v_isShared_1121_ == 0)
{
lean_ctor_set(v___x_1120_, 0, v___x_1122_);
v___x_1124_ = v___x_1120_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v___x_1122_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
}
}
default: 
{
lean_object* v___x_1127_; 
lean_dec_ref(v_v_1065_);
lean_dec_ref(v_e_1064_);
lean_dec(v_algMap_1063_);
v___x_1127_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM(lean_object* v_F_1128_, lean_object* v_EF_1129_, lean_object* v_algMap_1130_, lean_object* v_e_1131_, lean_object* v_v_1132_){
_start:
{
lean_object* v___x_1133_; 
v___x_1133_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(v_algMap_1130_, v_e_1131_, v_v_1132_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg(lean_object* v_e_1134_){
_start:
{
lean_object* v_isFirstRow_1135_; 
v_isFirstRow_1135_ = lean_ctor_get(v_e_1134_, 3);
lean_inc(v_isFirstRow_1135_);
return v_isFirstRow_1135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg___boxed(lean_object* v_e_1136_){
_start:
{
lean_object* v_res_1137_; 
v_res_1137_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg(v_e_1136_);
lean_dec_ref(v_e_1136_);
return v_res_1137_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow(lean_object* v_F_1138_, lean_object* v_EF_1139_, lean_object* v_e_1140_){
_start:
{
lean_object* v_isFirstRow_1141_; 
v_isFirstRow_1141_ = lean_ctor_get(v_e_1140_, 3);
lean_inc(v_isFirstRow_1141_);
return v_isFirstRow_1141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___boxed(lean_object* v_F_1142_, lean_object* v_EF_1143_, lean_object* v_e_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow(v_F_1142_, v_EF_1143_, v_e_1144_);
lean_dec_ref(v_e_1144_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg(lean_object* v_e_1146_){
_start:
{
lean_object* v_isLastRow_1147_; 
v_isLastRow_1147_ = lean_ctor_get(v_e_1146_, 4);
lean_inc(v_isLastRow_1147_);
return v_isLastRow_1147_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg___boxed(lean_object* v_e_1148_){
_start:
{
lean_object* v_res_1149_; 
v_res_1149_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg(v_e_1148_);
lean_dec_ref(v_e_1148_);
return v_res_1149_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow(lean_object* v_F_1150_, lean_object* v_EF_1151_, lean_object* v_e_1152_){
_start:
{
lean_object* v_isLastRow_1153_; 
v_isLastRow_1153_ = lean_ctor_get(v_e_1152_, 4);
lean_inc(v_isLastRow_1153_);
return v_isLastRow_1153_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___boxed(lean_object* v_F_1154_, lean_object* v_EF_1155_, lean_object* v_e_1156_){
_start:
{
lean_object* v_res_1157_; 
v_res_1157_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow(v_F_1154_, v_EF_1155_, v_e_1156_);
lean_dec_ref(v_e_1156_);
return v_res_1157_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(lean_object* v_fo_1158_, lean_object* v_e_1159_){
_start:
{
lean_object* v_toRingOps_1160_; lean_object* v_toSemiringOps_1161_; lean_object* v_sub_1162_; lean_object* v_one_1163_; lean_object* v_isLastRow_1164_; lean_object* v___x_1165_; 
v_toRingOps_1160_ = lean_ctor_get(v_fo_1158_, 0);
lean_inc_ref(v_toRingOps_1160_);
lean_dec_ref(v_fo_1158_);
v_toSemiringOps_1161_ = lean_ctor_get(v_toRingOps_1160_, 0);
lean_inc_ref(v_toSemiringOps_1161_);
v_sub_1162_ = lean_ctor_get(v_toRingOps_1160_, 1);
lean_inc(v_sub_1162_);
lean_dec_ref(v_toRingOps_1160_);
v_one_1163_ = lean_ctor_get(v_toSemiringOps_1161_, 1);
lean_inc(v_one_1163_);
lean_dec_ref(v_toSemiringOps_1161_);
v_isLastRow_1164_ = lean_ctor_get(v_e_1159_, 4);
lean_inc(v_isLastRow_1164_);
lean_dec_ref(v_e_1159_);
v___x_1165_ = lean_apply_2(v_sub_1162_, v_one_1163_, v_isLastRow_1164_);
return v___x_1165_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition(lean_object* v_F_1166_, lean_object* v_EF_1167_, lean_object* v_fo_1168_, lean_object* v_e_1169_){
_start:
{
lean_object* v___x_1170_; 
v___x_1170_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(v_fo_1168_, v_e_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(lean_object* v_fo_1171_, lean_object* v_algMap_1172_, lean_object* v_e_1173_, lean_object* v_values_1174_, lean_object* v_x_1175_){
_start:
{
switch(lean_obj_tag(v_x_1175_))
{
case 0:
{
lean_object* v_v_1176_; lean_object* v___x_1177_; 
lean_dec_ref(v_fo_1171_);
v_v_1176_ = lean_ctor_get(v_x_1175_, 0);
lean_inc_ref(v_v_1176_);
lean_dec_ref_known(v_x_1175_, 1);
v___x_1177_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(v_algMap_1172_, v_e_1173_, v_v_1176_);
return v___x_1177_;
}
case 1:
{
lean_object* v_isFirstRow_1178_; lean_object* v___x_1179_; 
lean_dec(v_algMap_1172_);
lean_dec_ref(v_fo_1171_);
v_isFirstRow_1178_ = lean_ctor_get(v_e_1173_, 3);
lean_inc(v_isFirstRow_1178_);
lean_dec_ref(v_e_1173_);
v___x_1179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1179_, 0, v_isFirstRow_1178_);
return v___x_1179_;
}
case 2:
{
lean_object* v_isLastRow_1180_; lean_object* v___x_1181_; 
lean_dec(v_algMap_1172_);
lean_dec_ref(v_fo_1171_);
v_isLastRow_1180_ = lean_ctor_get(v_e_1173_, 4);
lean_inc(v_isLastRow_1180_);
lean_dec_ref(v_e_1173_);
v___x_1181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1181_, 0, v_isLastRow_1180_);
return v___x_1181_;
}
case 3:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; 
lean_dec(v_algMap_1172_);
v___x_1182_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(v_fo_1171_, v_e_1173_);
v___x_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1182_);
return v___x_1183_;
}
case 4:
{
lean_object* v_c_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1192_; 
lean_dec_ref(v_e_1173_);
lean_dec_ref(v_fo_1171_);
v_c_1184_ = lean_ctor_get(v_x_1175_, 0);
v_isSharedCheck_1192_ = !lean_is_exclusive(v_x_1175_);
if (v_isSharedCheck_1192_ == 0)
{
v___x_1186_ = v_x_1175_;
v_isShared_1187_ = v_isSharedCheck_1192_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_c_1184_);
lean_dec(v_x_1175_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1192_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1188_; lean_object* v___x_1190_; 
v___x_1188_ = lean_apply_1(v_algMap_1172_, v_c_1184_);
if (v_isShared_1187_ == 0)
{
lean_ctor_set_tag(v___x_1186_, 1);
lean_ctor_set(v___x_1186_, 0, v___x_1188_);
v___x_1190_ = v___x_1186_;
goto v_reusejp_1189_;
}
else
{
lean_object* v_reuseFailAlloc_1191_; 
v_reuseFailAlloc_1191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1191_, 0, v___x_1188_);
v___x_1190_ = v_reuseFailAlloc_1191_;
goto v_reusejp_1189_;
}
v_reusejp_1189_:
{
return v___x_1190_;
}
}
}
case 5:
{
lean_object* v_leftIdx_1193_; lean_object* v_rightIdx_1194_; lean_object* v___x_1195_; 
lean_dec_ref(v_e_1173_);
lean_dec(v_algMap_1172_);
v_leftIdx_1193_ = lean_ctor_get(v_x_1175_, 0);
lean_inc(v_leftIdx_1193_);
v_rightIdx_1194_ = lean_ctor_get(v_x_1175_, 1);
lean_inc(v_rightIdx_1194_);
lean_dec_ref_known(v_x_1175_, 3);
v___x_1195_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_leftIdx_1193_);
if (lean_obj_tag(v___x_1195_) == 0)
{
lean_dec(v_rightIdx_1194_);
lean_dec_ref(v_fo_1171_);
return v___x_1195_;
}
else
{
lean_object* v_a_1196_; lean_object* v___x_1197_; 
v_a_1196_ = lean_ctor_get(v___x_1195_, 0);
lean_inc(v_a_1196_);
lean_dec_ref_known(v___x_1195_, 1);
v___x_1197_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_rightIdx_1194_);
if (lean_obj_tag(v___x_1197_) == 0)
{
lean_dec(v_a_1196_);
lean_dec_ref(v_fo_1171_);
return v___x_1197_;
}
else
{
lean_object* v_toRingOps_1198_; lean_object* v_toSemiringOps_1199_; lean_object* v_a_1200_; lean_object* v___x_1202_; uint8_t v_isShared_1203_; uint8_t v_isSharedCheck_1209_; 
v_toRingOps_1198_ = lean_ctor_get(v_fo_1171_, 0);
lean_inc_ref(v_toRingOps_1198_);
lean_dec_ref(v_fo_1171_);
v_toSemiringOps_1199_ = lean_ctor_get(v_toRingOps_1198_, 0);
lean_inc_ref(v_toSemiringOps_1199_);
lean_dec_ref(v_toRingOps_1198_);
v_a_1200_ = lean_ctor_get(v___x_1197_, 0);
v_isSharedCheck_1209_ = !lean_is_exclusive(v___x_1197_);
if (v_isSharedCheck_1209_ == 0)
{
v___x_1202_ = v___x_1197_;
v_isShared_1203_ = v_isSharedCheck_1209_;
goto v_resetjp_1201_;
}
else
{
lean_inc(v_a_1200_);
lean_dec(v___x_1197_);
v___x_1202_ = lean_box(0);
v_isShared_1203_ = v_isSharedCheck_1209_;
goto v_resetjp_1201_;
}
v_resetjp_1201_:
{
lean_object* v_add_1204_; lean_object* v___x_1205_; lean_object* v___x_1207_; 
v_add_1204_ = lean_ctor_get(v_toSemiringOps_1199_, 3);
lean_inc(v_add_1204_);
lean_dec_ref(v_toSemiringOps_1199_);
v___x_1205_ = lean_apply_2(v_add_1204_, v_a_1196_, v_a_1200_);
if (v_isShared_1203_ == 0)
{
lean_ctor_set(v___x_1202_, 0, v___x_1205_);
v___x_1207_ = v___x_1202_;
goto v_reusejp_1206_;
}
else
{
lean_object* v_reuseFailAlloc_1208_; 
v_reuseFailAlloc_1208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1208_, 0, v___x_1205_);
v___x_1207_ = v_reuseFailAlloc_1208_;
goto v_reusejp_1206_;
}
v_reusejp_1206_:
{
return v___x_1207_;
}
}
}
}
}
case 6:
{
lean_object* v_leftIdx_1210_; lean_object* v_rightIdx_1211_; lean_object* v___x_1212_; 
lean_dec_ref(v_e_1173_);
lean_dec(v_algMap_1172_);
v_leftIdx_1210_ = lean_ctor_get(v_x_1175_, 0);
lean_inc(v_leftIdx_1210_);
v_rightIdx_1211_ = lean_ctor_get(v_x_1175_, 1);
lean_inc(v_rightIdx_1211_);
lean_dec_ref_known(v_x_1175_, 3);
v___x_1212_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_leftIdx_1210_);
if (lean_obj_tag(v___x_1212_) == 0)
{
lean_dec(v_rightIdx_1211_);
lean_dec_ref(v_fo_1171_);
return v___x_1212_;
}
else
{
lean_object* v_a_1213_; lean_object* v___x_1214_; 
v_a_1213_ = lean_ctor_get(v___x_1212_, 0);
lean_inc(v_a_1213_);
lean_dec_ref_known(v___x_1212_, 1);
v___x_1214_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_rightIdx_1211_);
if (lean_obj_tag(v___x_1214_) == 0)
{
lean_dec(v_a_1213_);
lean_dec_ref(v_fo_1171_);
return v___x_1214_;
}
else
{
lean_object* v_toRingOps_1215_; lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1225_; 
v_toRingOps_1215_ = lean_ctor_get(v_fo_1171_, 0);
lean_inc_ref(v_toRingOps_1215_);
lean_dec_ref(v_fo_1171_);
v_a_1216_ = lean_ctor_get(v___x_1214_, 0);
v_isSharedCheck_1225_ = !lean_is_exclusive(v___x_1214_);
if (v_isSharedCheck_1225_ == 0)
{
v___x_1218_ = v___x_1214_;
v_isShared_1219_ = v_isSharedCheck_1225_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1214_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1225_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v_sub_1220_; lean_object* v___x_1221_; lean_object* v___x_1223_; 
v_sub_1220_ = lean_ctor_get(v_toRingOps_1215_, 1);
lean_inc(v_sub_1220_);
lean_dec_ref(v_toRingOps_1215_);
v___x_1221_ = lean_apply_2(v_sub_1220_, v_a_1213_, v_a_1216_);
if (v_isShared_1219_ == 0)
{
lean_ctor_set(v___x_1218_, 0, v___x_1221_);
v___x_1223_ = v___x_1218_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v___x_1221_);
v___x_1223_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
return v___x_1223_;
}
}
}
}
}
case 7:
{
lean_object* v_idx_1226_; lean_object* v___x_1227_; 
lean_dec_ref(v_e_1173_);
lean_dec(v_algMap_1172_);
v_idx_1226_ = lean_ctor_get(v_x_1175_, 0);
lean_inc(v_idx_1226_);
lean_dec_ref_known(v_x_1175_, 2);
v___x_1227_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_idx_1226_);
if (lean_obj_tag(v___x_1227_) == 0)
{
lean_dec_ref(v_fo_1171_);
return v___x_1227_;
}
else
{
lean_object* v_toRingOps_1228_; lean_object* v_toSemiringOps_1229_; lean_object* v_a_1230_; lean_object* v___x_1232_; uint8_t v_isShared_1233_; uint8_t v_isSharedCheck_1240_; 
v_toRingOps_1228_ = lean_ctor_get(v_fo_1171_, 0);
lean_inc_ref(v_toRingOps_1228_);
lean_dec_ref(v_fo_1171_);
v_toSemiringOps_1229_ = lean_ctor_get(v_toRingOps_1228_, 0);
lean_inc_ref(v_toSemiringOps_1229_);
v_a_1230_ = lean_ctor_get(v___x_1227_, 0);
v_isSharedCheck_1240_ = !lean_is_exclusive(v___x_1227_);
if (v_isSharedCheck_1240_ == 0)
{
v___x_1232_ = v___x_1227_;
v_isShared_1233_ = v_isSharedCheck_1240_;
goto v_resetjp_1231_;
}
else
{
lean_inc(v_a_1230_);
lean_dec(v___x_1227_);
v___x_1232_ = lean_box(0);
v_isShared_1233_ = v_isSharedCheck_1240_;
goto v_resetjp_1231_;
}
v_resetjp_1231_:
{
lean_object* v_sub_1234_; lean_object* v_zero_1235_; lean_object* v___x_1236_; lean_object* v___x_1238_; 
v_sub_1234_ = lean_ctor_get(v_toRingOps_1228_, 1);
lean_inc(v_sub_1234_);
lean_dec_ref(v_toRingOps_1228_);
v_zero_1235_ = lean_ctor_get(v_toSemiringOps_1229_, 0);
lean_inc(v_zero_1235_);
lean_dec_ref(v_toSemiringOps_1229_);
v___x_1236_ = lean_apply_2(v_sub_1234_, v_zero_1235_, v_a_1230_);
if (v_isShared_1233_ == 0)
{
lean_ctor_set(v___x_1232_, 0, v___x_1236_);
v___x_1238_ = v___x_1232_;
goto v_reusejp_1237_;
}
else
{
lean_object* v_reuseFailAlloc_1239_; 
v_reuseFailAlloc_1239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1239_, 0, v___x_1236_);
v___x_1238_ = v_reuseFailAlloc_1239_;
goto v_reusejp_1237_;
}
v_reusejp_1237_:
{
return v___x_1238_;
}
}
}
}
default: 
{
lean_object* v_leftIdx_1241_; lean_object* v_rightIdx_1242_; lean_object* v___x_1243_; 
lean_dec_ref(v_e_1173_);
lean_dec(v_algMap_1172_);
v_leftIdx_1241_ = lean_ctor_get(v_x_1175_, 0);
lean_inc(v_leftIdx_1241_);
v_rightIdx_1242_ = lean_ctor_get(v_x_1175_, 1);
lean_inc(v_rightIdx_1242_);
lean_dec_ref_known(v_x_1175_, 3);
v___x_1243_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_leftIdx_1241_);
if (lean_obj_tag(v___x_1243_) == 0)
{
lean_dec(v_rightIdx_1242_);
lean_dec_ref(v_fo_1171_);
return v___x_1243_;
}
else
{
lean_object* v_a_1244_; lean_object* v___x_1245_; 
v_a_1244_ = lean_ctor_get(v___x_1243_, 0);
lean_inc(v_a_1244_);
lean_dec_ref_known(v___x_1243_, 1);
v___x_1245_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1174_, v_rightIdx_1242_);
if (lean_obj_tag(v___x_1245_) == 0)
{
lean_dec(v_a_1244_);
lean_dec_ref(v_fo_1171_);
return v___x_1245_;
}
else
{
lean_object* v_toRingOps_1246_; lean_object* v_toSemiringOps_1247_; lean_object* v_a_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1257_; 
v_toRingOps_1246_ = lean_ctor_get(v_fo_1171_, 0);
lean_inc_ref(v_toRingOps_1246_);
lean_dec_ref(v_fo_1171_);
v_toSemiringOps_1247_ = lean_ctor_get(v_toRingOps_1246_, 0);
lean_inc_ref(v_toSemiringOps_1247_);
lean_dec_ref(v_toRingOps_1246_);
v_a_1248_ = lean_ctor_get(v___x_1245_, 0);
v_isSharedCheck_1257_ = !lean_is_exclusive(v___x_1245_);
if (v_isSharedCheck_1257_ == 0)
{
v___x_1250_ = v___x_1245_;
v_isShared_1251_ = v_isSharedCheck_1257_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_a_1248_);
lean_dec(v___x_1245_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1257_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v_mul_1252_; lean_object* v___x_1253_; lean_object* v___x_1255_; 
v_mul_1252_ = lean_ctor_get(v_toSemiringOps_1247_, 4);
lean_inc(v_mul_1252_);
lean_dec_ref(v_toSemiringOps_1247_);
v___x_1253_ = lean_apply_2(v_mul_1252_, v_a_1244_, v_a_1248_);
if (v_isShared_1251_ == 0)
{
lean_ctor_set(v___x_1250_, 0, v___x_1253_);
v___x_1255_ = v___x_1250_;
goto v_reusejp_1254_;
}
else
{
lean_object* v_reuseFailAlloc_1256_; 
v_reuseFailAlloc_1256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1256_, 0, v___x_1253_);
v___x_1255_ = v_reuseFailAlloc_1256_;
goto v_reusejp_1254_;
}
v_reusejp_1254_:
{
return v___x_1255_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg___boxed(lean_object* v_fo_1258_, lean_object* v_algMap_1259_, lean_object* v_e_1260_, lean_object* v_values_1261_, lean_object* v_x_1262_){
_start:
{
lean_object* v_res_1263_; 
v_res_1263_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(v_fo_1258_, v_algMap_1259_, v_e_1260_, v_values_1261_, v_x_1262_);
lean_dec(v_values_1261_);
return v_res_1263_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM(lean_object* v_F_1264_, lean_object* v_EF_1265_, lean_object* v_fo_1266_, lean_object* v_algMap_1267_, lean_object* v_e_1268_, lean_object* v_values_1269_, lean_object* v_x_1270_){
_start:
{
lean_object* v___x_1271_; 
v___x_1271_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(v_fo_1266_, v_algMap_1267_, v_e_1268_, v_values_1269_, v_x_1270_);
return v___x_1271_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___boxed(lean_object* v_F_1272_, lean_object* v_EF_1273_, lean_object* v_fo_1274_, lean_object* v_algMap_1275_, lean_object* v_e_1276_, lean_object* v_values_1277_, lean_object* v_x_1278_){
_start:
{
lean_object* v_res_1279_; 
v_res_1279_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM(v_F_1272_, v_EF_1273_, v_fo_1274_, v_algMap_1275_, v_e_1276_, v_values_1277_, v_x_1278_);
lean_dec(v_values_1277_);
return v_res_1279_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(lean_object* v_fo_1283_, lean_object* v_algMap_1284_, lean_object* v_e_1285_, lean_object* v_x_1286_, lean_object* v_x_1287_){
_start:
{
if (lean_obj_tag(v_x_1287_) == 0)
{
lean_object* v___x_1288_; 
lean_dec_ref(v_e_1285_);
lean_dec(v_algMap_1284_);
lean_dec_ref(v_fo_1283_);
v___x_1288_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1288_, 0, v_x_1286_);
return v___x_1288_;
}
else
{
lean_object* v_head_1289_; lean_object* v_tail_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1303_; 
v_head_1289_ = lean_ctor_get(v_x_1287_, 0);
v_tail_1290_ = lean_ctor_get(v_x_1287_, 1);
v_isSharedCheck_1303_ = !lean_is_exclusive(v_x_1287_);
if (v_isSharedCheck_1303_ == 0)
{
v___x_1292_ = v_x_1287_;
v_isShared_1293_ = v_isSharedCheck_1303_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_tail_1290_);
lean_inc(v_head_1289_);
lean_dec(v_x_1287_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1303_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
lean_object* v___x_1294_; 
lean_inc_ref(v_e_1285_);
lean_inc(v_algMap_1284_);
lean_inc_ref(v_fo_1283_);
v___x_1294_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(v_fo_1283_, v_algMap_1284_, v_e_1285_, v_x_1286_, v_head_1289_);
if (lean_obj_tag(v___x_1294_) == 0)
{
lean_object* v___x_1295_; 
lean_dec_ref_known(v___x_1294_, 1);
lean_del_object(v___x_1292_);
lean_dec(v_tail_1290_);
lean_dec(v_x_1286_);
lean_dec_ref(v_e_1285_);
lean_dec(v_algMap_1284_);
lean_dec_ref(v_fo_1283_);
v___x_1295_ = ((lean_object*)(lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0));
return v___x_1295_;
}
else
{
lean_object* v_a_1296_; lean_object* v___x_1297_; lean_object* v___x_1299_; 
v_a_1296_ = lean_ctor_get(v___x_1294_, 0);
lean_inc(v_a_1296_);
lean_dec_ref_known(v___x_1294_, 1);
v___x_1297_ = lean_box(0);
if (v_isShared_1293_ == 0)
{
lean_ctor_set(v___x_1292_, 1, v___x_1297_);
lean_ctor_set(v___x_1292_, 0, v_a_1296_);
v___x_1299_ = v___x_1292_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v_a_1296_);
lean_ctor_set(v_reuseFailAlloc_1302_, 1, v___x_1297_);
v___x_1299_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
lean_object* v___x_1300_; 
v___x_1300_ = l_List_appendTR___redArg(v_x_1286_, v___x_1299_);
v_x_1286_ = v___x_1300_;
v_x_1287_ = v_tail_1290_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(lean_object* v_fo_1304_, lean_object* v_algMap_1305_, lean_object* v_e_1306_, lean_object* v_nodes_1307_){
_start:
{
lean_object* v___x_1308_; lean_object* v___x_1309_; 
v___x_1308_ = lean_box(0);
v___x_1309_ = lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(v_fo_1304_, v_algMap_1305_, v_e_1306_, v___x_1308_, v_nodes_1307_);
return v___x_1309_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM(lean_object* v_F_1310_, lean_object* v_EF_1311_, lean_object* v_fo_1312_, lean_object* v_algMap_1313_, lean_object* v_e_1314_, lean_object* v_nodes_1315_){
_start:
{
lean_object* v___x_1316_; 
v___x_1316_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(v_fo_1312_, v_algMap_1313_, v_e_1314_, v_nodes_1315_);
return v___x_1316_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0(lean_object* v_EF_1317_, lean_object* v_F_1318_, lean_object* v_fo_1319_, lean_object* v_algMap_1320_, lean_object* v_e_1321_, lean_object* v_x_1322_, lean_object* v_x_1323_){
_start:
{
lean_object* v___x_1324_; 
v___x_1324_ = lp_swirl_x2dfv_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(v_fo_1319_, v_algMap_1320_, v_e_1321_, v_x_1322_, v_x_1323_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(lean_object* v_x_1330_){
_start:
{
if (lean_obj_tag(v_x_1330_) == 0)
{
lean_object* v___x_1331_; 
v___x_1331_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0));
return v___x_1331_;
}
else
{
lean_object* v_tail_1332_; 
v_tail_1332_ = lean_ctor_get(v_x_1330_, 1);
lean_inc(v_tail_1332_);
if (lean_obj_tag(v_tail_1332_) == 0)
{
lean_object* v___x_1333_; 
lean_dec_ref_known(v_x_1330_, 2);
v___x_1333_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1));
return v___x_1333_;
}
else
{
lean_object* v_head_1334_; lean_object* v___x_1336_; uint8_t v_isShared_1337_; uint8_t v_isSharedCheck_1359_; 
v_head_1334_ = lean_ctor_get(v_x_1330_, 0);
v_isSharedCheck_1359_ = !lean_is_exclusive(v_x_1330_);
if (v_isSharedCheck_1359_ == 0)
{
lean_object* v_unused_1360_; 
v_unused_1360_ = lean_ctor_get(v_x_1330_, 1);
lean_dec(v_unused_1360_);
v___x_1336_ = v_x_1330_;
v_isShared_1337_ = v_isSharedCheck_1359_;
goto v_resetjp_1335_;
}
else
{
lean_inc(v_head_1334_);
lean_dec(v_x_1330_);
v___x_1336_ = lean_box(0);
v_isShared_1337_ = v_isSharedCheck_1359_;
goto v_resetjp_1335_;
}
v_resetjp_1335_:
{
lean_object* v_head_1338_; lean_object* v_tail_1339_; lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1358_; 
v_head_1338_ = lean_ctor_get(v_tail_1332_, 0);
v_tail_1339_ = lean_ctor_get(v_tail_1332_, 1);
v_isSharedCheck_1358_ = !lean_is_exclusive(v_tail_1332_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1341_ = v_tail_1332_;
v_isShared_1342_ = v_isSharedCheck_1358_;
goto v_resetjp_1340_;
}
else
{
lean_inc(v_tail_1339_);
lean_inc(v_head_1338_);
lean_dec(v_tail_1332_);
v___x_1341_ = lean_box(0);
v_isShared_1342_ = v_isSharedCheck_1358_;
goto v_resetjp_1340_;
}
v_resetjp_1340_:
{
lean_object* v___x_1343_; 
v___x_1343_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(v_tail_1339_);
if (lean_obj_tag(v___x_1343_) == 0)
{
lean_del_object(v___x_1341_);
lean_dec(v_head_1338_);
lean_del_object(v___x_1336_);
lean_dec(v_head_1334_);
return v___x_1343_;
}
else
{
lean_object* v_a_1344_; lean_object* v___x_1346_; uint8_t v_isShared_1347_; uint8_t v_isSharedCheck_1357_; 
v_a_1344_ = lean_ctor_get(v___x_1343_, 0);
v_isSharedCheck_1357_ = !lean_is_exclusive(v___x_1343_);
if (v_isSharedCheck_1357_ == 0)
{
v___x_1346_ = v___x_1343_;
v_isShared_1347_ = v_isSharedCheck_1357_;
goto v_resetjp_1345_;
}
else
{
lean_inc(v_a_1344_);
lean_dec(v___x_1343_);
v___x_1346_ = lean_box(0);
v_isShared_1347_ = v_isSharedCheck_1357_;
goto v_resetjp_1345_;
}
v_resetjp_1345_:
{
lean_object* v___x_1349_; 
if (v_isShared_1337_ == 0)
{
lean_ctor_set_tag(v___x_1336_, 0);
lean_ctor_set(v___x_1336_, 1, v_head_1338_);
v___x_1349_ = v___x_1336_;
goto v_reusejp_1348_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v_head_1334_);
lean_ctor_set(v_reuseFailAlloc_1356_, 1, v_head_1338_);
v___x_1349_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1348_;
}
v_reusejp_1348_:
{
lean_object* v___x_1351_; 
if (v_isShared_1342_ == 0)
{
lean_ctor_set(v___x_1341_, 1, v_a_1344_);
lean_ctor_set(v___x_1341_, 0, v___x_1349_);
v___x_1351_ = v___x_1341_;
goto v_reusejp_1350_;
}
else
{
lean_object* v_reuseFailAlloc_1355_; 
v_reuseFailAlloc_1355_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1355_, 0, v___x_1349_);
lean_ctor_set(v_reuseFailAlloc_1355_, 1, v_a_1344_);
v___x_1351_ = v_reuseFailAlloc_1355_;
goto v_reusejp_1350_;
}
v_reusejp_1350_:
{
lean_object* v___x_1353_; 
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v___x_1351_);
v___x_1353_ = v___x_1346_;
goto v_reusejp_1352_;
}
else
{
lean_object* v_reuseFailAlloc_1354_; 
v_reuseFailAlloc_1354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1354_, 0, v___x_1351_);
v___x_1353_ = v_reuseFailAlloc_1354_;
goto v_reusejp_1352_;
}
v_reusejp_1352_:
{
return v___x_1353_;
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
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent(lean_object* v_EF_1361_, lean_object* v_x_1362_){
_start:
{
lean_object* v___x_1363_; 
v___x_1363_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(v_x_1362_);
return v___x_1363_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(lean_object* v_fo_1364_, lean_object* v_a_1365_, lean_object* v_a_1366_){
_start:
{
if (lean_obj_tag(v_a_1365_) == 0)
{
lean_object* v___x_1367_; 
lean_dec_ref(v_fo_1364_);
v___x_1367_ = l_List_reverse___redArg(v_a_1366_);
return v___x_1367_;
}
else
{
lean_object* v_toRingOps_1368_; lean_object* v_toSemiringOps_1369_; lean_object* v___x_1371_; uint8_t v_isShared_1372_; uint8_t v_isSharedCheck_1387_; 
v_toRingOps_1368_ = lean_ctor_get(v_fo_1364_, 0);
lean_inc_ref(v_toRingOps_1368_);
v_toSemiringOps_1369_ = lean_ctor_get(v_toRingOps_1368_, 0);
v_isSharedCheck_1387_ = !lean_is_exclusive(v_toRingOps_1368_);
if (v_isSharedCheck_1387_ == 0)
{
lean_object* v_unused_1388_; 
v_unused_1388_ = lean_ctor_get(v_toRingOps_1368_, 1);
lean_dec(v_unused_1388_);
v___x_1371_ = v_toRingOps_1368_;
v_isShared_1372_ = v_isSharedCheck_1387_;
goto v_resetjp_1370_;
}
else
{
lean_inc(v_toSemiringOps_1369_);
lean_dec(v_toRingOps_1368_);
v___x_1371_ = lean_box(0);
v_isShared_1372_ = v_isSharedCheck_1387_;
goto v_resetjp_1370_;
}
v_resetjp_1370_:
{
lean_object* v_head_1373_; lean_object* v_tail_1374_; lean_object* v___x_1376_; uint8_t v_isShared_1377_; uint8_t v_isSharedCheck_1386_; 
v_head_1373_ = lean_ctor_get(v_a_1365_, 0);
v_tail_1374_ = lean_ctor_get(v_a_1365_, 1);
v_isSharedCheck_1386_ = !lean_is_exclusive(v_a_1365_);
if (v_isSharedCheck_1386_ == 0)
{
v___x_1376_ = v_a_1365_;
v_isShared_1377_ = v_isSharedCheck_1386_;
goto v_resetjp_1375_;
}
else
{
lean_inc(v_tail_1374_);
lean_inc(v_head_1373_);
lean_dec(v_a_1365_);
v___x_1376_ = lean_box(0);
v_isShared_1377_ = v_isSharedCheck_1386_;
goto v_resetjp_1375_;
}
v_resetjp_1375_:
{
lean_object* v_zero_1378_; lean_object* v___x_1380_; 
v_zero_1378_ = lean_ctor_get(v_toSemiringOps_1369_, 0);
lean_inc(v_zero_1378_);
lean_dec_ref(v_toSemiringOps_1369_);
if (v_isShared_1372_ == 0)
{
lean_ctor_set(v___x_1371_, 1, v_zero_1378_);
lean_ctor_set(v___x_1371_, 0, v_head_1373_);
v___x_1380_ = v___x_1371_;
goto v_reusejp_1379_;
}
else
{
lean_object* v_reuseFailAlloc_1385_; 
v_reuseFailAlloc_1385_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1385_, 0, v_head_1373_);
lean_ctor_set(v_reuseFailAlloc_1385_, 1, v_zero_1378_);
v___x_1380_ = v_reuseFailAlloc_1385_;
goto v_reusejp_1379_;
}
v_reusejp_1379_:
{
lean_object* v___x_1382_; 
if (v_isShared_1377_ == 0)
{
lean_ctor_set(v___x_1376_, 1, v_a_1366_);
lean_ctor_set(v___x_1376_, 0, v___x_1380_);
v___x_1382_ = v___x_1376_;
goto v_reusejp_1381_;
}
else
{
lean_object* v_reuseFailAlloc_1384_; 
v_reuseFailAlloc_1384_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1384_, 0, v___x_1380_);
lean_ctor_set(v_reuseFailAlloc_1384_, 1, v_a_1366_);
v___x_1382_ = v_reuseFailAlloc_1384_;
goto v_reusejp_1381_;
}
v_reusejp_1381_:
{
v_a_1365_ = v_tail_1374_;
v_a_1366_ = v___x_1382_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(lean_object* v_fo_1389_, lean_object* v_openings_1390_, uint8_t v_needRot_1391_){
_start:
{
if (v_needRot_1391_ == 0)
{
lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; 
v___x_1392_ = lean_box(0);
v___x_1393_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(v_fo_1389_, v_openings_1390_, v___x_1392_);
v___x_1394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1394_, 0, v___x_1393_);
return v___x_1394_;
}
else
{
lean_object* v___x_1395_; 
lean_dec_ref(v_fo_1389_);
v___x_1395_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(v_openings_1390_);
return v___x_1395_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg___boxed(lean_object* v_fo_1396_, lean_object* v_openings_1397_, lean_object* v_needRot_1398_){
_start:
{
uint8_t v_needRot_boxed_1399_; lean_object* v_res_1400_; 
v_needRot_boxed_1399_ = lean_unbox(v_needRot_1398_);
v_res_1400_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_1396_, v_openings_1397_, v_needRot_boxed_1399_);
return v_res_1400_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot(lean_object* v_EF_1401_, lean_object* v_fo_1402_, lean_object* v_openings_1403_, uint8_t v_needRot_1404_){
_start:
{
lean_object* v___x_1405_; 
v___x_1405_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_1402_, v_openings_1403_, v_needRot_1404_);
return v___x_1405_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___boxed(lean_object* v_EF_1406_, lean_object* v_fo_1407_, lean_object* v_openings_1408_, lean_object* v_needRot_1409_){
_start:
{
uint8_t v_needRot_boxed_1410_; lean_object* v_res_1411_; 
v_needRot_boxed_1410_ = lean_unbox(v_needRot_1409_);
v_res_1411_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot(v_EF_1406_, v_fo_1407_, v_openings_1408_, v_needRot_boxed_1410_);
return v_res_1411_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0(lean_object* v_EF_1412_, lean_object* v_fo_1413_, lean_object* v_a_1414_, lean_object* v_a_1415_){
_start:
{
lean_object* v___x_1416_; 
v___x_1416_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(v_fo_1413_, v_a_1414_, v_a_1415_);
return v___x_1416_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___lam__0(lean_object* v_inst_1417_, lean_object* v_inst_1418_, lean_object* v_x_1419_, lean_object* v_pair_1420_, lean_object* v___y_1421_){
_start:
{
lean_object* v_fst_1422_; lean_object* v_snd_1423_; lean_object* v___x_1424_; lean_object* v_a_1425_; lean_object* v_snd_1426_; lean_object* v___x_1427_; 
v_fst_1422_ = lean_ctor_get(v_pair_1420_, 0);
lean_inc(v_fst_1422_);
v_snd_1423_ = lean_ctor_get(v_pair_1420_, 1);
lean_inc(v_snd_1423_);
lean_dec_ref(v_pair_1420_);
lean_inc_ref(v_inst_1418_);
lean_inc_ref(v_inst_1417_);
v___x_1424_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_1417_, v_inst_1418_, v_fst_1422_, v___y_1421_);
v_a_1425_ = lean_ctor_get(v___x_1424_, 0);
lean_inc(v_a_1425_);
lean_dec_ref(v___x_1424_);
v_snd_1426_ = lean_ctor_get(v_a_1425_, 1);
lean_inc(v_snd_1426_);
lean_dec(v_a_1425_);
v___x_1427_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_1417_, v_inst_1418_, v_snd_1423_, v_snd_1426_);
return v___x_1427_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(lean_object* v_inst_1473_, lean_object* v_inst_1474_, lean_object* v_pairs_1475_, lean_object* v_a_1476_){
_start:
{
lean_object* v___f_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_362__overap_1480_; lean_object* v___x_1481_; 
v___f_1477_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___lam__0), 5, 2);
lean_closure_set(v___f_1477_, 0, v_inst_1473_);
lean_closure_set(v___f_1477_, 1, v_inst_1474_);
v___x_1478_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19));
v___x_1479_ = lean_box(0);
v___x_362__overap_1480_ = l_List_foldlM___redArg(v___x_1478_, v___f_1477_, v___x_1479_, v_pairs_1475_);
v___x_1481_ = lean_apply_1(v___x_362__overap_1480_, v_a_1476_);
if (lean_obj_tag(v___x_1481_) == 0)
{
return v___x_1481_;
}
else
{
lean_object* v_a_1482_; lean_object* v___x_1484_; uint8_t v_isShared_1485_; uint8_t v_isSharedCheck_1498_; 
v_a_1482_ = lean_ctor_get(v___x_1481_, 0);
v_isSharedCheck_1498_ = !lean_is_exclusive(v___x_1481_);
if (v_isSharedCheck_1498_ == 0)
{
v___x_1484_ = v___x_1481_;
v_isShared_1485_ = v_isSharedCheck_1498_;
goto v_resetjp_1483_;
}
else
{
lean_inc(v_a_1482_);
lean_dec(v___x_1481_);
v___x_1484_ = lean_box(0);
v_isShared_1485_ = v_isSharedCheck_1498_;
goto v_resetjp_1483_;
}
v_resetjp_1483_:
{
lean_object* v_snd_1486_; lean_object* v___x_1488_; uint8_t v_isShared_1489_; uint8_t v_isSharedCheck_1496_; 
v_snd_1486_ = lean_ctor_get(v_a_1482_, 1);
v_isSharedCheck_1496_ = !lean_is_exclusive(v_a_1482_);
if (v_isSharedCheck_1496_ == 0)
{
lean_object* v_unused_1497_; 
v_unused_1497_ = lean_ctor_get(v_a_1482_, 0);
lean_dec(v_unused_1497_);
v___x_1488_ = v_a_1482_;
v_isShared_1489_ = v_isSharedCheck_1496_;
goto v_resetjp_1487_;
}
else
{
lean_inc(v_snd_1486_);
lean_dec(v_a_1482_);
v___x_1488_ = lean_box(0);
v_isShared_1489_ = v_isSharedCheck_1496_;
goto v_resetjp_1487_;
}
v_resetjp_1487_:
{
lean_object* v___x_1491_; 
if (v_isShared_1489_ == 0)
{
lean_ctor_set(v___x_1488_, 0, v___x_1479_);
v___x_1491_ = v___x_1488_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v___x_1479_);
lean_ctor_set(v_reuseFailAlloc_1495_, 1, v_snd_1486_);
v___x_1491_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
lean_object* v___x_1493_; 
if (v_isShared_1485_ == 0)
{
lean_ctor_set(v___x_1484_, 0, v___x_1491_);
v___x_1493_ = v___x_1484_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1494_; 
v_reuseFailAlloc_1494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1494_, 0, v___x_1491_);
v___x_1493_ = v_reuseFailAlloc_1494_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
return v___x_1493_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM(lean_object* v_F_1499_, lean_object* v_EF_1500_, lean_object* v_inst_1501_, lean_object* v_inst_1502_, lean_object* v_pairs_1503_, lean_object* v_a_1504_){
_start:
{
lean_object* v___x_1505_; 
v___x_1505_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(v_inst_1501_, v_inst_1502_, v_pairs_1503_, v_a_1504_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(lean_object* v_inst_1509_, lean_object* v_inst_1510_, lean_object* v_fo_1511_, lean_object* v_openings_1512_, uint8_t v_needRot_1513_, lean_object* v_a_1514_){
_start:
{
lean_object* v___x_1515_; 
v___x_1515_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_1511_, v_openings_1512_, v_needRot_1513_);
if (lean_obj_tag(v___x_1515_) == 0)
{
lean_object* v___x_1516_; 
lean_dec_ref_known(v___x_1515_, 1);
lean_dec_ref(v_a_1514_);
lean_dec_ref(v_inst_1510_);
lean_dec_ref(v_inst_1509_);
v___x_1516_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1516_;
}
else
{
lean_object* v_a_1517_; lean_object* v___x_1518_; 
v_a_1517_ = lean_ctor_get(v___x_1515_, 0);
lean_inc(v_a_1517_);
lean_dec_ref_known(v___x_1515_, 1);
v___x_1518_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(v_inst_1509_, v_inst_1510_, v_a_1517_, v_a_1514_);
return v___x_1518_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___boxed(lean_object* v_inst_1519_, lean_object* v_inst_1520_, lean_object* v_fo_1521_, lean_object* v_openings_1522_, lean_object* v_needRot_1523_, lean_object* v_a_1524_){
_start:
{
uint8_t v_needRot_boxed_1525_; lean_object* v_res_1526_; 
v_needRot_boxed_1525_ = lean_unbox(v_needRot_1523_);
v_res_1526_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1519_, v_inst_1520_, v_fo_1521_, v_openings_1522_, v_needRot_boxed_1525_, v_a_1524_);
return v_res_1526_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM(lean_object* v_F_1527_, lean_object* v_EF_1528_, lean_object* v_inst_1529_, lean_object* v_inst_1530_, lean_object* v_fo_1531_, lean_object* v_openings_1532_, uint8_t v_needRot_1533_, lean_object* v_a_1534_){
_start:
{
lean_object* v___x_1535_; 
v___x_1535_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1529_, v_inst_1530_, v_fo_1531_, v_openings_1532_, v_needRot_1533_, v_a_1534_);
return v___x_1535_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___boxed(lean_object* v_F_1536_, lean_object* v_EF_1537_, lean_object* v_inst_1538_, lean_object* v_inst_1539_, lean_object* v_fo_1540_, lean_object* v_openings_1541_, lean_object* v_needRot_1542_, lean_object* v_a_1543_){
_start:
{
uint8_t v_needRot_boxed_1544_; lean_object* v_res_1545_; 
v_needRot_boxed_1544_ = lean_unbox(v_needRot_1542_);
v_res_1545_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM(v_F_1536_, v_EF_1537_, v_inst_1538_, v_inst_1539_, v_fo_1540_, v_openings_1541_, v_needRot_boxed_1544_, v_a_1543_);
return v_res_1545_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0(lean_object* v_inst_1546_, lean_object* v_inst_1547_, lean_object* v_fo_1548_, lean_object* v_vk_1549_, lean_object* v_x_1550_, lean_object* v_entry_1551_, lean_object* v___y_1552_){
_start:
{
lean_object* v_fst_1553_; lean_object* v_snd_1554_; uint8_t v___y_1556_; lean_object* v___x_1562_; 
v_fst_1553_ = lean_ctor_get(v_entry_1551_, 0);
lean_inc(v_fst_1553_);
v_snd_1554_ = lean_ctor_get(v_entry_1551_, 1);
lean_inc(v_snd_1554_);
lean_dec_ref(v_entry_1551_);
v___x_1562_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_1549_, v_snd_1554_);
if (lean_obj_tag(v___x_1562_) == 0)
{
uint8_t v___x_1563_; 
v___x_1563_ = 0;
v___y_1556_ = v___x_1563_;
goto v___jp_1555_;
}
else
{
lean_object* v_val_1564_; lean_object* v_params_1565_; uint8_t v_needRot_1566_; 
v_val_1564_ = lean_ctor_get(v___x_1562_, 0);
lean_inc(v_val_1564_);
lean_dec_ref_known(v___x_1562_, 1);
v_params_1565_ = lean_ctor_get(v_val_1564_, 1);
lean_inc_ref(v_params_1565_);
lean_dec(v_val_1564_);
v_needRot_1566_ = lean_ctor_get_uint8(v_params_1565_, sizeof(void*)*2);
lean_dec_ref(v_params_1565_);
v___y_1556_ = v_needRot_1566_;
goto v___jp_1555_;
}
v___jp_1555_:
{
lean_object* v___x_1557_; lean_object* v___x_1558_; 
v___x_1557_ = lean_unsigned_to_nat(0u);
v___x_1558_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_fst_1553_, v___x_1557_);
lean_dec(v_fst_1553_);
if (lean_obj_tag(v___x_1558_) == 0)
{
lean_object* v___x_1559_; 
lean_dec_ref_known(v___x_1558_, 1);
lean_dec_ref(v___y_1552_);
lean_dec_ref(v_fo_1548_);
lean_dec_ref(v_inst_1547_);
lean_dec_ref(v_inst_1546_);
v___x_1559_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1559_;
}
else
{
lean_object* v_a_1560_; lean_object* v___x_1561_; 
v_a_1560_ = lean_ctor_get(v___x_1558_, 0);
lean_inc(v_a_1560_);
lean_dec_ref_known(v___x_1558_, 1);
v___x_1561_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1546_, v_inst_1547_, v_fo_1548_, v_a_1560_, v___y_1556_, v___y_1552_);
return v___x_1561_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0___boxed(lean_object* v_inst_1567_, lean_object* v_inst_1568_, lean_object* v_fo_1569_, lean_object* v_vk_1570_, lean_object* v_x_1571_, lean_object* v_entry_1572_, lean_object* v___y_1573_){
_start:
{
lean_object* v_res_1574_; 
v_res_1574_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0(v_inst_1567_, v_inst_1568_, v_fo_1569_, v_vk_1570_, v_x_1571_, v_entry_1572_, v___y_1573_);
lean_dec_ref(v_vk_1570_);
return v_res_1574_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1(lean_object* v_inst_1575_, lean_object* v_inst_1576_, lean_object* v_fo_1577_, uint8_t v_needRot_1578_, lean_object* v_x_1579_, lean_object* v_openings_1580_, lean_object* v___y_1581_){
_start:
{
lean_object* v___x_1582_; 
v___x_1582_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1575_, v_inst_1576_, v_fo_1577_, v_openings_1580_, v_needRot_1578_, v___y_1581_);
return v___x_1582_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1___boxed(lean_object* v_inst_1583_, lean_object* v_inst_1584_, lean_object* v_fo_1585_, lean_object* v_needRot_1586_, lean_object* v_x_1587_, lean_object* v_openings_1588_, lean_object* v___y_1589_){
_start:
{
uint8_t v_needRot_boxed_1590_; lean_object* v_res_1591_; 
v_needRot_boxed_1590_ = lean_unbox(v_needRot_1586_);
v_res_1591_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1(v_inst_1583_, v_inst_1584_, v_fo_1585_, v_needRot_boxed_1590_, v_x_1587_, v_openings_1588_, v___y_1589_);
return v_res_1591_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2(lean_object* v_traceIdToAirId_1592_, lean_object* v_vk_1593_, lean_object* v_inst_1594_, lean_object* v_inst_1595_, lean_object* v_fo_1596_, lean_object* v___x_1597_, lean_object* v___x_1598_, lean_object* v_x_1599_, lean_object* v_entry_1600_, lean_object* v___y_1601_){
_start:
{
lean_object* v_fst_1602_; lean_object* v_snd_1603_; lean_object* v___x_1604_; 
v_fst_1602_ = lean_ctor_get(v_entry_1600_, 0);
lean_inc(v_fst_1602_);
v_snd_1603_ = lean_ctor_get(v_entry_1600_, 1);
lean_inc(v_snd_1603_);
lean_dec_ref(v_entry_1600_);
v___x_1604_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_traceIdToAirId_1592_, v_fst_1602_);
if (lean_obj_tag(v___x_1604_) == 0)
{
lean_object* v___x_1605_; 
lean_dec_ref_known(v___x_1604_, 1);
lean_dec(v_snd_1603_);
lean_dec_ref(v___y_1601_);
lean_dec_ref(v___x_1597_);
lean_dec_ref(v_fo_1596_);
lean_dec_ref(v_inst_1595_);
lean_dec_ref(v_inst_1594_);
v___x_1605_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1605_;
}
else
{
lean_object* v_a_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; 
v_a_1606_ = lean_ctor_get(v___x_1604_, 0);
lean_inc(v_a_1606_);
lean_dec_ref_known(v___x_1604_, 1);
v___x_1607_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_1593_, v_a_1606_);
v___x_1608_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v___x_1607_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_object* v___x_1609_; 
lean_dec_ref_known(v___x_1608_, 1);
lean_dec(v_snd_1603_);
lean_dec_ref(v___y_1601_);
lean_dec_ref(v___x_1597_);
lean_dec_ref(v_fo_1596_);
lean_dec_ref(v_inst_1595_);
lean_dec_ref(v_inst_1594_);
v___x_1609_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1609_;
}
else
{
lean_object* v_a_1610_; lean_object* v_params_1611_; uint8_t v_needRot_1612_; lean_object* v___x_1613_; lean_object* v___f_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1893__overap_1617_; lean_object* v___x_1618_; 
v_a_1610_ = lean_ctor_get(v___x_1608_, 0);
lean_inc(v_a_1610_);
lean_dec_ref_known(v___x_1608_, 1);
v_params_1611_ = lean_ctor_get(v_a_1610_, 1);
lean_inc_ref(v_params_1611_);
lean_dec(v_a_1610_);
v_needRot_1612_ = lean_ctor_get_uint8(v_params_1611_, sizeof(void*)*2);
lean_dec_ref(v_params_1611_);
v___x_1613_ = lean_box(v_needRot_1612_);
v___f_1614_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1___boxed), 7, 4);
lean_closure_set(v___f_1614_, 0, v_inst_1594_);
lean_closure_set(v___f_1614_, 1, v_inst_1595_);
lean_closure_set(v___f_1614_, 2, v_fo_1596_);
lean_closure_set(v___f_1614_, 3, v___x_1613_);
v___x_1615_ = lean_unsigned_to_nat(1u);
v___x_1616_ = l_List_drop___redArg(v___x_1615_, v_snd_1603_);
lean_dec(v_snd_1603_);
v___x_1893__overap_1617_ = l_List_foldlM___redArg(v___x_1597_, v___f_1614_, v___x_1598_, v___x_1616_);
v___x_1618_ = lean_apply_1(v___x_1893__overap_1617_, v___y_1601_);
if (lean_obj_tag(v___x_1618_) == 0)
{
return v___x_1618_;
}
else
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1635_; 
v_a_1619_ = lean_ctor_get(v___x_1618_, 0);
v_isSharedCheck_1635_ = !lean_is_exclusive(v___x_1618_);
if (v_isSharedCheck_1635_ == 0)
{
v___x_1621_ = v___x_1618_;
v_isShared_1622_ = v_isSharedCheck_1635_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1618_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1635_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v_snd_1623_; lean_object* v___x_1625_; uint8_t v_isShared_1626_; uint8_t v_isSharedCheck_1633_; 
v_snd_1623_ = lean_ctor_get(v_a_1619_, 1);
v_isSharedCheck_1633_ = !lean_is_exclusive(v_a_1619_);
if (v_isSharedCheck_1633_ == 0)
{
lean_object* v_unused_1634_; 
v_unused_1634_ = lean_ctor_get(v_a_1619_, 0);
lean_dec(v_unused_1634_);
v___x_1625_ = v_a_1619_;
v_isShared_1626_ = v_isSharedCheck_1633_;
goto v_resetjp_1624_;
}
else
{
lean_inc(v_snd_1623_);
lean_dec(v_a_1619_);
v___x_1625_ = lean_box(0);
v_isShared_1626_ = v_isSharedCheck_1633_;
goto v_resetjp_1624_;
}
v_resetjp_1624_:
{
lean_object* v___x_1628_; 
if (v_isShared_1626_ == 0)
{
lean_ctor_set(v___x_1625_, 0, v___x_1598_);
v___x_1628_ = v___x_1625_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1632_; 
v_reuseFailAlloc_1632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1632_, 0, v___x_1598_);
lean_ctor_set(v_reuseFailAlloc_1632_, 1, v_snd_1623_);
v___x_1628_ = v_reuseFailAlloc_1632_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
lean_object* v___x_1630_; 
if (v_isShared_1622_ == 0)
{
lean_ctor_set(v___x_1621_, 0, v___x_1628_);
v___x_1630_ = v___x_1621_;
goto v_reusejp_1629_;
}
else
{
lean_object* v_reuseFailAlloc_1631_; 
v_reuseFailAlloc_1631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1631_, 0, v___x_1628_);
v___x_1630_ = v_reuseFailAlloc_1631_;
goto v_reusejp_1629_;
}
v_reusejp_1629_:
{
return v___x_1630_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2___boxed(lean_object* v_traceIdToAirId_1636_, lean_object* v_vk_1637_, lean_object* v_inst_1638_, lean_object* v_inst_1639_, lean_object* v_fo_1640_, lean_object* v___x_1641_, lean_object* v___x_1642_, lean_object* v_x_1643_, lean_object* v_entry_1644_, lean_object* v___y_1645_){
_start:
{
lean_object* v_res_1646_; 
v_res_1646_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2(v_traceIdToAirId_1636_, v_vk_1637_, v_inst_1638_, v_inst_1639_, v_fo_1640_, v___x_1641_, v___x_1642_, v_x_1643_, v_entry_1644_, v___y_1645_);
lean_dec_ref(v_vk_1637_);
lean_dec(v_traceIdToAirId_1636_);
return v_res_1646_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(lean_object* v_inst_1647_, lean_object* v_inst_1648_, lean_object* v_fo_1649_, lean_object* v_vk_1650_, lean_object* v_columnOpenings_1651_, lean_object* v_traceIdToAirId_1652_, lean_object* v_a_1653_){
_start:
{
lean_object* v___f_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1396__overap_1658_; lean_object* v___x_1659_; 
lean_inc_ref(v_vk_1650_);
lean_inc_ref(v_fo_1649_);
lean_inc_ref(v_inst_1648_);
lean_inc_ref(v_inst_1647_);
v___f_1654_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0___boxed), 7, 4);
lean_closure_set(v___f_1654_, 0, v_inst_1647_);
lean_closure_set(v___f_1654_, 1, v_inst_1648_);
lean_closure_set(v___f_1654_, 2, v_fo_1649_);
lean_closure_set(v___f_1654_, 3, v_vk_1650_);
v___x_1655_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19));
v___x_1656_ = lean_box(0);
lean_inc(v_traceIdToAirId_1652_);
lean_inc(v_columnOpenings_1651_);
v___x_1657_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_columnOpenings_1651_, v_traceIdToAirId_1652_);
v___x_1396__overap_1658_ = l_List_foldlM___redArg(v___x_1655_, v___f_1654_, v___x_1656_, v___x_1657_);
v___x_1659_ = lean_apply_1(v___x_1396__overap_1658_, v_a_1653_);
if (lean_obj_tag(v___x_1659_) == 0)
{
lean_dec(v_traceIdToAirId_1652_);
lean_dec(v_columnOpenings_1651_);
lean_dec_ref(v_vk_1650_);
lean_dec_ref(v_fo_1649_);
lean_dec_ref(v_inst_1648_);
lean_dec_ref(v_inst_1647_);
return v___x_1659_;
}
else
{
lean_object* v_a_1660_; lean_object* v_snd_1661_; lean_object* v___f_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1609__overap_1666_; lean_object* v___x_1667_; 
v_a_1660_ = lean_ctor_get(v___x_1659_, 0);
lean_inc(v_a_1660_);
lean_dec_ref_known(v___x_1659_, 1);
v_snd_1661_ = lean_ctor_get(v_a_1660_, 1);
lean_inc(v_snd_1661_);
lean_dec(v_a_1660_);
v___f_1662_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2___boxed), 10, 7);
lean_closure_set(v___f_1662_, 0, v_traceIdToAirId_1652_);
lean_closure_set(v___f_1662_, 1, v_vk_1650_);
lean_closure_set(v___f_1662_, 2, v_inst_1647_);
lean_closure_set(v___f_1662_, 3, v_inst_1648_);
lean_closure_set(v___f_1662_, 4, v_fo_1649_);
lean_closure_set(v___f_1662_, 5, v___x_1655_);
lean_closure_set(v___f_1662_, 6, v___x_1656_);
v___x_1663_ = l_List_lengthTR___redArg(v_columnOpenings_1651_);
v___x_1664_ = l_List_range(v___x_1663_);
v___x_1665_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1664_, v_columnOpenings_1651_);
v___x_1609__overap_1666_ = l_List_foldlM___redArg(v___x_1655_, v___f_1662_, v___x_1656_, v___x_1665_);
v___x_1667_ = lean_apply_1(v___x_1609__overap_1666_, v_snd_1661_);
if (lean_obj_tag(v___x_1667_) == 0)
{
return v___x_1667_;
}
else
{
lean_object* v_a_1668_; lean_object* v___x_1670_; uint8_t v_isShared_1671_; uint8_t v_isSharedCheck_1684_; 
v_a_1668_ = lean_ctor_get(v___x_1667_, 0);
v_isSharedCheck_1684_ = !lean_is_exclusive(v___x_1667_);
if (v_isSharedCheck_1684_ == 0)
{
v___x_1670_ = v___x_1667_;
v_isShared_1671_ = v_isSharedCheck_1684_;
goto v_resetjp_1669_;
}
else
{
lean_inc(v_a_1668_);
lean_dec(v___x_1667_);
v___x_1670_ = lean_box(0);
v_isShared_1671_ = v_isSharedCheck_1684_;
goto v_resetjp_1669_;
}
v_resetjp_1669_:
{
lean_object* v_snd_1672_; lean_object* v___x_1674_; uint8_t v_isShared_1675_; uint8_t v_isSharedCheck_1682_; 
v_snd_1672_ = lean_ctor_get(v_a_1668_, 1);
v_isSharedCheck_1682_ = !lean_is_exclusive(v_a_1668_);
if (v_isSharedCheck_1682_ == 0)
{
lean_object* v_unused_1683_; 
v_unused_1683_ = lean_ctor_get(v_a_1668_, 0);
lean_dec(v_unused_1683_);
v___x_1674_ = v_a_1668_;
v_isShared_1675_ = v_isSharedCheck_1682_;
goto v_resetjp_1673_;
}
else
{
lean_inc(v_snd_1672_);
lean_dec(v_a_1668_);
v___x_1674_ = lean_box(0);
v_isShared_1675_ = v_isSharedCheck_1682_;
goto v_resetjp_1673_;
}
v_resetjp_1673_:
{
lean_object* v___x_1677_; 
if (v_isShared_1675_ == 0)
{
lean_ctor_set(v___x_1674_, 0, v___x_1656_);
v___x_1677_ = v___x_1674_;
goto v_reusejp_1676_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v___x_1656_);
lean_ctor_set(v_reuseFailAlloc_1681_, 1, v_snd_1672_);
v___x_1677_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1676_;
}
v_reusejp_1676_:
{
lean_object* v___x_1679_; 
if (v_isShared_1671_ == 0)
{
lean_ctor_set(v___x_1670_, 0, v___x_1677_);
v___x_1679_ = v___x_1670_;
goto v_reusejp_1678_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v___x_1677_);
v___x_1679_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1678_;
}
v_reusejp_1678_:
{
return v___x_1679_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM(lean_object* v_F_1685_, lean_object* v_EF_1686_, lean_object* v_Digest_1687_, lean_object* v_inst_1688_, lean_object* v_inst_1689_, lean_object* v_fo_1690_, lean_object* v_vk_1691_, lean_object* v_columnOpenings_1692_, lean_object* v_traceIdToAirId_1693_, lean_object* v_a_1694_){
_start:
{
lean_object* v___x_1695_; 
v___x_1695_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(v_inst_1688_, v_inst_1689_, v_fo_1690_, v_vk_1691_, v_columnOpenings_1692_, v_traceIdToAirId_1693_, v_a_1694_);
return v___x_1695_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___lam__0(lean_object* v_fo_1696_, lean_object* v_acc_1697_, lean_object* v_x_1698_){
_start:
{
lean_object* v_toRingOps_1699_; lean_object* v_sub_1700_; lean_object* v___x_1701_; 
v_toRingOps_1699_ = lean_ctor_get(v_fo_1696_, 0);
lean_inc_ref(v_toRingOps_1699_);
lean_dec_ref(v_fo_1696_);
v_sub_1700_ = lean_ctor_get(v_toRingOps_1699_, 1);
lean_inc(v_sub_1700_);
lean_dec_ref(v_toRingOps_1699_);
v___x_1701_ = lean_apply_2(v_sub_1700_, v_acc_1697_, v_x_1698_);
return v___x_1701_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(lean_object* v_inst_1707_, lean_object* v_fo_1708_, lean_object* v_numeratorTermPerAir_1709_, lean_object* v_denominatorTermPerAir_1710_, lean_object* v_alphaLogup_1711_, lean_object* v_pXiClaim_1712_, lean_object* v_qXiClaim_1713_){
_start:
{
lean_object* v_toRingOps_1716_; lean_object* v_toSemiringOps_1717_; lean_object* v_zero_1718_; lean_object* v___f_1719_; lean_object* v_pFinal_1720_; lean_object* v_qFinal_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; uint8_t v___x_1724_; 
v_toRingOps_1716_ = lean_ctor_get(v_fo_1708_, 0);
v_toSemiringOps_1717_ = lean_ctor_get(v_toRingOps_1716_, 0);
v_zero_1718_ = lean_ctor_get(v_toSemiringOps_1717_, 0);
lean_inc(v_zero_1718_);
v___f_1719_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1719_, 0, v_fo_1708_);
lean_inc_ref(v___f_1719_);
v_pFinal_1720_ = l_List_foldl___redArg(v___f_1719_, v_pXiClaim_1712_, v_numeratorTermPerAir_1709_);
v_qFinal_1721_ = l_List_foldl___redArg(v___f_1719_, v_qXiClaim_1713_, v_denominatorTermPerAir_1710_);
lean_inc_ref(v_inst_1707_);
v___x_1722_ = lean_apply_2(v_inst_1707_, v_qFinal_1721_, v_alphaLogup_1711_);
v___x_1723_ = lean_apply_2(v_inst_1707_, v_pFinal_1720_, v_zero_1718_);
v___x_1724_ = lean_unbox(v___x_1723_);
if (v___x_1724_ == 0)
{
goto v___jp_1714_;
}
else
{
uint8_t v___x_1725_; 
v___x_1725_ = lean_unbox(v___x_1722_);
if (v___x_1725_ == 0)
{
goto v___jp_1714_;
}
else
{
lean_object* v___x_1726_; 
v___x_1726_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1));
return v___x_1726_;
}
}
v___jp_1714_:
{
lean_object* v___x_1715_; 
v___x_1715_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0));
return v___x_1715_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi(lean_object* v_EF_1727_, lean_object* v_inst_1728_, lean_object* v_fo_1729_, lean_object* v_numeratorTermPerAir_1730_, lean_object* v_denominatorTermPerAir_1731_, lean_object* v_alphaLogup_1732_, lean_object* v_pXiClaim_1733_, lean_object* v_qXiClaim_1734_){
_start:
{
lean_object* v___x_1735_; 
v___x_1735_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(v_inst_1728_, v_fo_1729_, v_numeratorTermPerAir_1730_, v_denominatorTermPerAir_1731_, v_alphaLogup_1732_, v_pXiClaim_1733_, v_qXiClaim_1734_);
return v___x_1735_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(lean_object* v_fo_1736_, lean_object* v_mu_1737_, lean_object* v_x_1738_, lean_object* v_x_1739_){
_start:
{
if (lean_obj_tag(v_x_1739_) == 0)
{
lean_dec(v_mu_1737_);
lean_dec_ref(v_fo_1736_);
return v_x_1738_;
}
else
{
lean_object* v_head_1740_; lean_object* v_toRingOps_1741_; lean_object* v_toSemiringOps_1742_; lean_object* v_tail_1743_; lean_object* v_fst_1744_; lean_object* v_snd_1745_; lean_object* v_fst_1746_; lean_object* v_snd_1747_; lean_object* v___x_1749_; uint8_t v_isShared_1750_; uint8_t v_isSharedCheck_1763_; 
v_head_1740_ = lean_ctor_get(v_x_1739_, 0);
lean_inc(v_head_1740_);
v_toRingOps_1741_ = lean_ctor_get(v_fo_1736_, 0);
v_toSemiringOps_1742_ = lean_ctor_get(v_toRingOps_1741_, 0);
v_tail_1743_ = lean_ctor_get(v_x_1739_, 1);
lean_inc(v_tail_1743_);
lean_dec_ref_known(v_x_1739_, 2);
v_fst_1744_ = lean_ctor_get(v_x_1738_, 0);
lean_inc(v_fst_1744_);
v_snd_1745_ = lean_ctor_get(v_x_1738_, 1);
lean_inc(v_snd_1745_);
lean_dec_ref(v_x_1738_);
v_fst_1746_ = lean_ctor_get(v_head_1740_, 0);
v_snd_1747_ = lean_ctor_get(v_head_1740_, 1);
v_isSharedCheck_1763_ = !lean_is_exclusive(v_head_1740_);
if (v_isSharedCheck_1763_ == 0)
{
v___x_1749_ = v_head_1740_;
v_isShared_1750_ = v_isSharedCheck_1763_;
goto v_resetjp_1748_;
}
else
{
lean_inc(v_snd_1747_);
lean_inc(v_fst_1746_);
lean_dec(v_head_1740_);
v___x_1749_ = lean_box(0);
v_isShared_1750_ = v_isSharedCheck_1763_;
goto v_resetjp_1748_;
}
v_resetjp_1748_:
{
lean_object* v_add_1751_; lean_object* v_mul_1752_; lean_object* v___x_1753_; lean_object* v_sumClaim_1754_; lean_object* v_curMuPow_1755_; lean_object* v___x_1756_; lean_object* v_sumClaim_1757_; lean_object* v___x_1758_; lean_object* v___x_1760_; 
v_add_1751_ = lean_ctor_get(v_toSemiringOps_1742_, 3);
v_mul_1752_ = lean_ctor_get(v_toSemiringOps_1742_, 4);
lean_inc_n(v_mul_1752_, 4);
lean_inc(v_fst_1744_);
v___x_1753_ = lean_apply_2(v_mul_1752_, v_fst_1746_, v_fst_1744_);
lean_inc_n(v_add_1751_, 2);
v_sumClaim_1754_ = lean_apply_2(v_add_1751_, v_snd_1745_, v___x_1753_);
lean_inc_n(v_mu_1737_, 2);
v_curMuPow_1755_ = lean_apply_2(v_mul_1752_, v_fst_1744_, v_mu_1737_);
lean_inc(v_curMuPow_1755_);
v___x_1756_ = lean_apply_2(v_mul_1752_, v_snd_1747_, v_curMuPow_1755_);
v_sumClaim_1757_ = lean_apply_2(v_add_1751_, v_sumClaim_1754_, v___x_1756_);
v___x_1758_ = lean_apply_2(v_mul_1752_, v_curMuPow_1755_, v_mu_1737_);
if (v_isShared_1750_ == 0)
{
lean_ctor_set(v___x_1749_, 1, v_sumClaim_1757_);
lean_ctor_set(v___x_1749_, 0, v___x_1758_);
v___x_1760_ = v___x_1749_;
goto v_reusejp_1759_;
}
else
{
lean_object* v_reuseFailAlloc_1762_; 
v_reuseFailAlloc_1762_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1762_, 0, v___x_1758_);
lean_ctor_set(v_reuseFailAlloc_1762_, 1, v_sumClaim_1757_);
v___x_1760_ = v_reuseFailAlloc_1762_;
goto v_reusejp_1759_;
}
v_reusejp_1759_:
{
v_x_1738_ = v___x_1760_;
v_x_1739_ = v_tail_1743_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(lean_object* v_fo_1764_, lean_object* v_numeratorTermPerAir_1765_, lean_object* v_denominatorTermPerAir_1766_, lean_object* v_mu_1767_){
_start:
{
lean_object* v_toRingOps_1768_; lean_object* v_toSemiringOps_1769_; lean_object* v___x_1771_; uint8_t v_isShared_1772_; uint8_t v_isSharedCheck_1781_; 
v_toRingOps_1768_ = lean_ctor_get(v_fo_1764_, 0);
lean_inc_ref(v_toRingOps_1768_);
v_toSemiringOps_1769_ = lean_ctor_get(v_toRingOps_1768_, 0);
v_isSharedCheck_1781_ = !lean_is_exclusive(v_toRingOps_1768_);
if (v_isSharedCheck_1781_ == 0)
{
lean_object* v_unused_1782_; 
v_unused_1782_ = lean_ctor_get(v_toRingOps_1768_, 1);
lean_dec(v_unused_1782_);
v___x_1771_ = v_toRingOps_1768_;
v_isShared_1772_ = v_isSharedCheck_1781_;
goto v_resetjp_1770_;
}
else
{
lean_inc(v_toSemiringOps_1769_);
lean_dec(v_toRingOps_1768_);
v___x_1771_ = lean_box(0);
v_isShared_1772_ = v_isSharedCheck_1781_;
goto v_resetjp_1770_;
}
v_resetjp_1770_:
{
lean_object* v_zero_1773_; lean_object* v_one_1774_; lean_object* v___x_1776_; 
v_zero_1773_ = lean_ctor_get(v_toSemiringOps_1769_, 0);
lean_inc(v_zero_1773_);
v_one_1774_ = lean_ctor_get(v_toSemiringOps_1769_, 1);
lean_inc(v_one_1774_);
lean_dec_ref(v_toSemiringOps_1769_);
if (v_isShared_1772_ == 0)
{
lean_ctor_set(v___x_1771_, 1, v_zero_1773_);
lean_ctor_set(v___x_1771_, 0, v_one_1774_);
v___x_1776_ = v___x_1771_;
goto v_reusejp_1775_;
}
else
{
lean_object* v_reuseFailAlloc_1780_; 
v_reuseFailAlloc_1780_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1780_, 0, v_one_1774_);
lean_ctor_set(v_reuseFailAlloc_1780_, 1, v_zero_1773_);
v___x_1776_ = v_reuseFailAlloc_1780_;
goto v_reusejp_1775_;
}
v_reusejp_1775_:
{
lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v_snd_1779_; 
v___x_1777_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_numeratorTermPerAir_1765_, v_denominatorTermPerAir_1766_);
v___x_1778_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(v_fo_1764_, v_mu_1767_, v___x_1776_, v___x_1777_);
v_snd_1779_ = lean_ctor_get(v___x_1778_, 1);
lean_inc(v_snd_1779_);
lean_dec_ref(v___x_1778_);
return v_snd_1779_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim(lean_object* v_EF_1783_, lean_object* v_fo_1784_, lean_object* v_numeratorTermPerAir_1785_, lean_object* v_denominatorTermPerAir_1786_, lean_object* v_mu_1787_){
_start:
{
lean_object* v___x_1788_; 
v___x_1788_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(v_fo_1784_, v_numeratorTermPerAir_1785_, v_denominatorTermPerAir_1786_, v_mu_1787_);
return v___x_1788_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0(lean_object* v_EF_1789_, lean_object* v_fo_1790_, lean_object* v_mu_1791_, lean_object* v_x_1792_, lean_object* v_x_1793_){
_start:
{
lean_object* v___x_1794_; 
v___x_1794_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(v_fo_1790_, v_mu_1791_, v_x_1792_, v_x_1793_);
return v___x_1794_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(lean_object* v_inst_1795_, lean_object* v_fo_1796_, lean_object* v_univariateRoundCoeffs_1797_, lean_object* v_sumClaim_1798_, lean_object* v_lSkip_1799_, lean_object* v_r0_1800_){
_start:
{
lean_object* v_sumUnivDomainS0_1801_; lean_object* v___x_1802_; uint8_t v___x_1803_; 
lean_inc(v_univariateRoundCoeffs_1797_);
lean_inc_ref(v_fo_1796_);
v_sumUnivDomainS0_1801_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(v_fo_1796_, v_univariateRoundCoeffs_1797_, v_lSkip_1799_);
v___x_1802_ = lean_apply_2(v_inst_1795_, v_sumClaim_1798_, v_sumUnivDomainS0_1801_);
v___x_1803_ = lean_unbox(v___x_1802_);
if (v___x_1803_ == 0)
{
lean_object* v___x_1804_; 
lean_dec(v_r0_1800_);
lean_dec(v_univariateRoundCoeffs_1797_);
lean_dec_ref(v_fo_1796_);
v___x_1804_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1804_;
}
else
{
lean_object* v_toRingOps_1805_; lean_object* v_toSemiringOps_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; 
v_toRingOps_1805_ = lean_ctor_get(v_fo_1796_, 0);
lean_inc_ref(v_toRingOps_1805_);
lean_dec_ref(v_fo_1796_);
v_toSemiringOps_1806_ = lean_ctor_get(v_toRingOps_1805_, 0);
lean_inc_ref(v_toSemiringOps_1806_);
lean_dec_ref(v_toRingOps_1805_);
v___x_1807_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_toSemiringOps_1806_, v_univariateRoundCoeffs_1797_, v_r0_1800_);
v___x_1808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1808_, 0, v___x_1807_);
return v___x_1808_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg___boxed(lean_object* v_inst_1809_, lean_object* v_fo_1810_, lean_object* v_univariateRoundCoeffs_1811_, lean_object* v_sumClaim_1812_, lean_object* v_lSkip_1813_, lean_object* v_r0_1814_){
_start:
{
lean_object* v_res_1815_; 
v_res_1815_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(v_inst_1809_, v_fo_1810_, v_univariateRoundCoeffs_1811_, v_sumClaim_1812_, v_lSkip_1813_, v_r0_1814_);
lean_dec(v_lSkip_1813_);
return v_res_1815_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound(lean_object* v_EF_1816_, lean_object* v_inst_1817_, lean_object* v_fo_1818_, lean_object* v_univariateRoundCoeffs_1819_, lean_object* v_sumClaim_1820_, lean_object* v_lSkip_1821_, lean_object* v_r0_1822_){
_start:
{
lean_object* v___x_1823_; 
v___x_1823_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(v_inst_1817_, v_fo_1818_, v_univariateRoundCoeffs_1819_, v_sumClaim_1820_, v_lSkip_1821_, v_r0_1822_);
return v___x_1823_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___boxed(lean_object* v_EF_1824_, lean_object* v_inst_1825_, lean_object* v_fo_1826_, lean_object* v_univariateRoundCoeffs_1827_, lean_object* v_sumClaim_1828_, lean_object* v_lSkip_1829_, lean_object* v_r0_1830_){
_start:
{
lean_object* v_res_1831_; 
v_res_1831_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound(v_EF_1824_, v_inst_1825_, v_fo_1826_, v_univariateRoundCoeffs_1827_, v_sumClaim_1828_, v_lSkip_1829_, v_r0_1830_);
lean_dec(v_lSkip_1829_);
return v_res_1831_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0(lean_object* v_mul_1832_, lean_object* v_acc_1833_, lean_object* v_x_1834_){
_start:
{
lean_object* v___x_1835_; 
v___x_1835_ = lean_apply_2(v_mul_1832_, v_acc_1833_, v_x_1834_);
return v___x_1835_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(lean_object* v___x_1836_, lean_object* v_a_1837_, lean_object* v_a_1838_){
_start:
{
if (lean_obj_tag(v_a_1837_) == 0)
{
lean_object* v___x_1839_; 
lean_dec_ref(v___x_1836_);
v___x_1839_ = l_List_reverse___redArg(v_a_1838_);
return v___x_1839_;
}
else
{
lean_object* v_head_1840_; lean_object* v_tail_1841_; lean_object* v___x_1843_; uint8_t v_isShared_1844_; uint8_t v_isSharedCheck_1853_; 
v_head_1840_ = lean_ctor_get(v_a_1837_, 0);
v_tail_1841_ = lean_ctor_get(v_a_1837_, 1);
v_isSharedCheck_1853_ = !lean_is_exclusive(v_a_1837_);
if (v_isSharedCheck_1853_ == 0)
{
v___x_1843_ = v_a_1837_;
v_isShared_1844_ = v_isSharedCheck_1853_;
goto v_resetjp_1842_;
}
else
{
lean_inc(v_tail_1841_);
lean_inc(v_head_1840_);
lean_dec(v_a_1837_);
v___x_1843_ = lean_box(0);
v_isShared_1844_ = v_isSharedCheck_1853_;
goto v_resetjp_1842_;
}
v_resetjp_1842_:
{
lean_object* v_natCast_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1850_; 
v_natCast_1845_ = lean_ctor_get(v___x_1836_, 2);
v___x_1846_ = lean_unsigned_to_nat(1u);
v___x_1847_ = lean_nat_add(v_head_1840_, v___x_1846_);
lean_dec(v_head_1840_);
lean_inc(v_natCast_1845_);
v___x_1848_ = lean_apply_1(v_natCast_1845_, v___x_1847_);
if (v_isShared_1844_ == 0)
{
lean_ctor_set(v___x_1843_, 1, v_a_1838_);
lean_ctor_set(v___x_1843_, 0, v___x_1848_);
v___x_1850_ = v___x_1843_;
goto v_reusejp_1849_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v___x_1848_);
lean_ctor_set(v_reuseFailAlloc_1852_, 1, v_a_1838_);
v___x_1850_ = v_reuseFailAlloc_1852_;
goto v_reusejp_1849_;
}
v_reusejp_1849_:
{
v_a_1837_ = v_tail_1841_;
v_a_1838_ = v___x_1850_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(lean_object* v___x_1854_, lean_object* v___x_1855_, lean_object* v_r_1856_, lean_object* v_a_1857_, lean_object* v_a_1858_){
_start:
{
if (lean_obj_tag(v_a_1857_) == 0)
{
lean_object* v___x_1859_; 
lean_dec(v_r_1856_);
lean_dec_ref(v___x_1855_);
lean_dec_ref(v___x_1854_);
v___x_1859_ = l_List_reverse___redArg(v_a_1858_);
return v___x_1859_;
}
else
{
lean_object* v_head_1860_; lean_object* v_tail_1861_; lean_object* v___x_1863_; uint8_t v_isShared_1864_; uint8_t v_isSharedCheck_1873_; 
v_head_1860_ = lean_ctor_get(v_a_1857_, 0);
v_tail_1861_ = lean_ctor_get(v_a_1857_, 1);
v_isSharedCheck_1873_ = !lean_is_exclusive(v_a_1857_);
if (v_isSharedCheck_1873_ == 0)
{
v___x_1863_ = v_a_1857_;
v_isShared_1864_ = v_isSharedCheck_1873_;
goto v_resetjp_1862_;
}
else
{
lean_inc(v_tail_1861_);
lean_inc(v_head_1860_);
lean_dec(v_a_1857_);
v___x_1863_ = lean_box(0);
v_isShared_1864_ = v_isSharedCheck_1873_;
goto v_resetjp_1862_;
}
v_resetjp_1862_:
{
lean_object* v_sub_1865_; lean_object* v_natCast_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1870_; 
v_sub_1865_ = lean_ctor_get(v___x_1854_, 1);
v_natCast_1866_ = lean_ctor_get(v___x_1855_, 2);
lean_inc(v_natCast_1866_);
v___x_1867_ = lean_apply_1(v_natCast_1866_, v_head_1860_);
lean_inc(v_sub_1865_);
lean_inc(v_r_1856_);
v___x_1868_ = lean_apply_2(v_sub_1865_, v_r_1856_, v___x_1867_);
if (v_isShared_1864_ == 0)
{
lean_ctor_set(v___x_1863_, 1, v_a_1858_);
lean_ctor_set(v___x_1863_, 0, v___x_1868_);
v___x_1870_ = v___x_1863_;
goto v_reusejp_1869_;
}
else
{
lean_object* v_reuseFailAlloc_1872_; 
v_reuseFailAlloc_1872_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1872_, 0, v___x_1868_);
lean_ctor_set(v_reuseFailAlloc_1872_, 1, v_a_1858_);
v___x_1870_ = v_reuseFailAlloc_1872_;
goto v_reusejp_1869_;
}
v_reusejp_1869_:
{
v_a_1857_ = v_tail_1861_;
v_a_1858_ = v___x_1870_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(lean_object* v___x_1874_, lean_object* v_sDeg_1875_, lean_object* v_invfact_1876_, lean_object* v_sufProduct_1877_, lean_object* v_prefProduct_1878_, lean_object* v_evals_1879_, lean_object* v_x_1880_, lean_object* v_x_1881_){
_start:
{
if (lean_obj_tag(v_x_1881_) == 0)
{
lean_dec_ref(v___x_1874_);
return v_x_1880_;
}
else
{
lean_object* v_head_1882_; lean_object* v_tail_1883_; lean_object* v_zero_1884_; lean_object* v_add_1885_; lean_object* v_mul_1886_; lean_object* v___y_1888_; lean_object* v___y_1889_; lean_object* v___y_1894_; lean_object* v___y_1895_; lean_object* v___y_1896_; lean_object* v___y_1901_; lean_object* v___y_1902_; lean_object* v___y_1903_; lean_object* v___y_1908_; lean_object* v___y_1909_; lean_object* v___y_1915_; lean_object* v___x_1918_; 
v_head_1882_ = lean_ctor_get(v_x_1881_, 0);
lean_inc_n(v_head_1882_, 2);
v_tail_1883_ = lean_ctor_get(v_x_1881_, 1);
lean_inc(v_tail_1883_);
lean_dec_ref_known(v_x_1881_, 2);
v_zero_1884_ = lean_ctor_get(v___x_1874_, 0);
v_add_1885_ = lean_ctor_get(v___x_1874_, 3);
v_mul_1886_ = lean_ctor_get(v___x_1874_, 4);
v___x_1918_ = l_List_get_x3fInternal___redArg(v_evals_1879_, v_head_1882_);
if (lean_obj_tag(v___x_1918_) == 0)
{
lean_inc(v_zero_1884_);
v___y_1915_ = v_zero_1884_;
goto v___jp_1914_;
}
else
{
lean_object* v_val_1919_; 
v_val_1919_ = lean_ctor_get(v___x_1918_, 0);
lean_inc(v_val_1919_);
lean_dec_ref_known(v___x_1918_, 1);
v___y_1915_ = v_val_1919_;
goto v___jp_1914_;
}
v___jp_1887_:
{
lean_object* v___x_1890_; lean_object* v___x_1891_; 
lean_inc(v_mul_1886_);
v___x_1890_ = lean_apply_2(v_mul_1886_, v___y_1888_, v___y_1889_);
lean_inc(v_add_1885_);
v___x_1891_ = lean_apply_2(v_add_1885_, v_x_1880_, v___x_1890_);
v_x_1880_ = v___x_1891_;
v_x_1881_ = v_tail_1883_;
goto _start;
}
v___jp_1893_:
{
lean_object* v___x_1897_; lean_object* v___x_1898_; 
lean_inc(v_mul_1886_);
v___x_1897_ = lean_apply_2(v_mul_1886_, v___y_1894_, v___y_1896_);
v___x_1898_ = l_List_get_x3fInternal___redArg(v_invfact_1876_, v___y_1895_);
if (lean_obj_tag(v___x_1898_) == 0)
{
lean_inc(v_zero_1884_);
v___y_1888_ = v___x_1897_;
v___y_1889_ = v_zero_1884_;
goto v___jp_1887_;
}
else
{
lean_object* v_val_1899_; 
v_val_1899_ = lean_ctor_get(v___x_1898_, 0);
lean_inc(v_val_1899_);
lean_dec_ref_known(v___x_1898_, 1);
v___y_1888_ = v___x_1897_;
v___y_1889_ = v_val_1899_;
goto v___jp_1887_;
}
}
v___jp_1900_:
{
lean_object* v___x_1904_; lean_object* v___x_1905_; 
lean_inc(v_mul_1886_);
v___x_1904_ = lean_apply_2(v_mul_1886_, v___y_1902_, v___y_1903_);
v___x_1905_ = l_List_get_x3fInternal___redArg(v_invfact_1876_, v_head_1882_);
if (lean_obj_tag(v___x_1905_) == 0)
{
lean_inc(v_zero_1884_);
v___y_1894_ = v___x_1904_;
v___y_1895_ = v___y_1901_;
v___y_1896_ = v_zero_1884_;
goto v___jp_1893_;
}
else
{
lean_object* v_val_1906_; 
v_val_1906_ = lean_ctor_get(v___x_1905_, 0);
lean_inc(v_val_1906_);
lean_dec_ref_known(v___x_1905_, 1);
v___y_1894_ = v___x_1904_;
v___y_1895_ = v___y_1901_;
v___y_1896_ = v_val_1906_;
goto v___jp_1893_;
}
}
v___jp_1907_:
{
lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; 
lean_inc(v_mul_1886_);
v___x_1910_ = lean_apply_2(v_mul_1886_, v___y_1908_, v___y_1909_);
v___x_1911_ = lean_nat_sub(v_sDeg_1875_, v_head_1882_);
lean_inc(v___x_1911_);
v___x_1912_ = l_List_get_x3fInternal___redArg(v_sufProduct_1877_, v___x_1911_);
if (lean_obj_tag(v___x_1912_) == 0)
{
lean_inc(v_zero_1884_);
v___y_1901_ = v___x_1911_;
v___y_1902_ = v___x_1910_;
v___y_1903_ = v_zero_1884_;
goto v___jp_1900_;
}
else
{
lean_object* v_val_1913_; 
v_val_1913_ = lean_ctor_get(v___x_1912_, 0);
lean_inc(v_val_1913_);
lean_dec_ref_known(v___x_1912_, 1);
v___y_1901_ = v___x_1911_;
v___y_1902_ = v___x_1910_;
v___y_1903_ = v_val_1913_;
goto v___jp_1900_;
}
}
v___jp_1914_:
{
lean_object* v___x_1916_; 
lean_inc(v_head_1882_);
v___x_1916_ = l_List_get_x3fInternal___redArg(v_prefProduct_1878_, v_head_1882_);
if (lean_obj_tag(v___x_1916_) == 0)
{
lean_inc(v_zero_1884_);
v___y_1908_ = v___y_1915_;
v___y_1909_ = v_zero_1884_;
goto v___jp_1907_;
}
else
{
lean_object* v_val_1917_; 
v_val_1917_ = lean_ctor_get(v___x_1916_, 0);
lean_inc(v_val_1917_);
lean_dec_ref_known(v___x_1916_, 1);
v___y_1908_ = v___y_1915_;
v___y_1909_ = v_val_1917_;
goto v___jp_1907_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg___boxed(lean_object* v___x_1920_, lean_object* v_sDeg_1921_, lean_object* v_invfact_1922_, lean_object* v_sufProduct_1923_, lean_object* v_prefProduct_1924_, lean_object* v_evals_1925_, lean_object* v_x_1926_, lean_object* v_x_1927_){
_start:
{
lean_object* v_res_1928_; 
v_res_1928_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(v___x_1920_, v_sDeg_1921_, v_invfact_1922_, v_sufProduct_1923_, v_prefProduct_1924_, v_evals_1925_, v_x_1926_, v_x_1927_);
lean_dec(v_evals_1925_);
lean_dec(v_prefProduct_1924_);
lean_dec(v_sufProduct_1923_);
lean_dec(v_invfact_1922_);
lean_dec(v_sDeg_1921_);
return v_res_1928_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(lean_object* v___x_1929_, lean_object* v___x_1930_, lean_object* v_sDeg_1931_, lean_object* v_r_1932_, lean_object* v_a_1933_, lean_object* v_a_1934_){
_start:
{
if (lean_obj_tag(v_a_1933_) == 0)
{
lean_object* v___x_1935_; 
lean_dec(v_r_1932_);
lean_dec_ref(v___x_1930_);
lean_dec_ref(v___x_1929_);
v___x_1935_ = l_List_reverse___redArg(v_a_1934_);
return v___x_1935_;
}
else
{
lean_object* v_head_1936_; lean_object* v_tail_1937_; lean_object* v___x_1939_; uint8_t v_isShared_1940_; uint8_t v_isSharedCheck_1950_; 
v_head_1936_ = lean_ctor_get(v_a_1933_, 0);
v_tail_1937_ = lean_ctor_get(v_a_1933_, 1);
v_isSharedCheck_1950_ = !lean_is_exclusive(v_a_1933_);
if (v_isSharedCheck_1950_ == 0)
{
v___x_1939_ = v_a_1933_;
v_isShared_1940_ = v_isSharedCheck_1950_;
goto v_resetjp_1938_;
}
else
{
lean_inc(v_tail_1937_);
lean_inc(v_head_1936_);
lean_dec(v_a_1933_);
v___x_1939_ = lean_box(0);
v_isShared_1940_ = v_isSharedCheck_1950_;
goto v_resetjp_1938_;
}
v_resetjp_1938_:
{
lean_object* v_sub_1941_; lean_object* v_natCast_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1947_; 
v_sub_1941_ = lean_ctor_get(v___x_1929_, 1);
v_natCast_1942_ = lean_ctor_get(v___x_1930_, 2);
v___x_1943_ = lean_nat_sub(v_sDeg_1931_, v_head_1936_);
lean_dec(v_head_1936_);
lean_inc(v_natCast_1942_);
v___x_1944_ = lean_apply_1(v_natCast_1942_, v___x_1943_);
lean_inc(v_sub_1941_);
lean_inc(v_r_1932_);
v___x_1945_ = lean_apply_2(v_sub_1941_, v___x_1944_, v_r_1932_);
if (v_isShared_1940_ == 0)
{
lean_ctor_set(v___x_1939_, 1, v_a_1934_);
lean_ctor_set(v___x_1939_, 0, v___x_1945_);
v___x_1947_ = v___x_1939_;
goto v_reusejp_1946_;
}
else
{
lean_object* v_reuseFailAlloc_1949_; 
v_reuseFailAlloc_1949_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1949_, 0, v___x_1945_);
lean_ctor_set(v_reuseFailAlloc_1949_, 1, v_a_1934_);
v___x_1947_ = v_reuseFailAlloc_1949_;
goto v_reusejp_1946_;
}
v_reusejp_1946_:
{
v_a_1933_ = v_tail_1937_;
v_a_1934_ = v___x_1947_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg___boxed(lean_object* v___x_1951_, lean_object* v___x_1952_, lean_object* v_sDeg_1953_, lean_object* v_r_1954_, lean_object* v_a_1955_, lean_object* v_a_1956_){
_start:
{
lean_object* v_res_1957_; 
v_res_1957_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(v___x_1951_, v___x_1952_, v_sDeg_1953_, v_r_1954_, v_a_1955_, v_a_1956_);
lean_dec(v_sDeg_1953_);
return v_res_1957_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(lean_object* v_fo_1958_, lean_object* v_evals_1959_, lean_object* v_r_1960_){
_start:
{
lean_object* v_toRingOps_1961_; lean_object* v_toSemiringOps_1962_; lean_object* v_zero_1963_; lean_object* v_one_1964_; lean_object* v_mul_1965_; lean_object* v___f_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v_sDeg_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v_factorials_1973_; lean_object* v_invfact_1974_; lean_object* v___x_1975_; lean_object* v_prefProduct_1976_; lean_object* v___x_1977_; lean_object* v_sufProduct_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; 
v_toRingOps_1961_ = lean_ctor_get(v_fo_1958_, 0);
lean_inc_ref_n(v_toRingOps_1961_, 2);
v_toSemiringOps_1962_ = lean_ctor_get(v_toRingOps_1961_, 0);
lean_inc_ref_n(v_toSemiringOps_1962_, 4);
v_zero_1963_ = lean_ctor_get(v_toSemiringOps_1962_, 0);
lean_inc(v_zero_1963_);
v_one_1964_ = lean_ctor_get(v_toSemiringOps_1962_, 1);
v_mul_1965_ = lean_ctor_get(v_toSemiringOps_1962_, 4);
lean_inc(v_mul_1965_);
v___f_1966_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1966_, 0, v_mul_1965_);
v___x_1967_ = l_List_lengthTR___redArg(v_evals_1959_);
v___x_1968_ = lean_unsigned_to_nat(1u);
v_sDeg_1969_ = lean_nat_sub(v___x_1967_, v___x_1968_);
lean_dec(v___x_1967_);
lean_inc(v_sDeg_1969_);
v___x_1970_ = l_List_range(v_sDeg_1969_);
v___x_1971_ = lean_box(0);
lean_inc_n(v___x_1970_, 2);
v___x_1972_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(v_toSemiringOps_1962_, v___x_1970_, v___x_1971_);
lean_inc_n(v_one_1964_, 3);
lean_inc_ref_n(v___f_1966_, 2);
v_factorials_1973_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_1966_, v_one_1964_, v___x_1972_);
v_invfact_1974_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(v_fo_1958_, v_factorials_1973_);
lean_inc(v_r_1960_);
v___x_1975_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(v_toRingOps_1961_, v_toSemiringOps_1962_, v_r_1960_, v___x_1970_, v___x_1971_);
v_prefProduct_1976_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_1966_, v_one_1964_, v___x_1975_);
v___x_1977_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(v_toRingOps_1961_, v_toSemiringOps_1962_, v_sDeg_1969_, v_r_1960_, v___x_1970_, v___x_1971_);
v_sufProduct_1978_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_1966_, v_one_1964_, v___x_1977_);
v___x_1979_ = lean_nat_add(v_sDeg_1969_, v___x_1968_);
v___x_1980_ = l_List_range(v___x_1979_);
v___x_1981_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(v_toSemiringOps_1962_, v_sDeg_1969_, v_invfact_1974_, v_sufProduct_1978_, v_prefProduct_1976_, v_evals_1959_, v_zero_1963_, v___x_1980_);
lean_dec(v_prefProduct_1976_);
lean_dec(v_sufProduct_1978_);
lean_dec(v_invfact_1974_);
lean_dec(v_sDeg_1969_);
return v___x_1981_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___boxed(lean_object* v_fo_1982_, lean_object* v_evals_1983_, lean_object* v_r_1984_){
_start:
{
lean_object* v_res_1985_; 
v_res_1985_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(v_fo_1982_, v_evals_1983_, v_r_1984_);
lean_dec(v_evals_1983_);
return v_res_1985_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints(lean_object* v_EF_1986_, lean_object* v_fo_1987_, lean_object* v_evals_1988_, lean_object* v_r_1989_){
_start:
{
lean_object* v___x_1990_; 
v___x_1990_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(v_fo_1987_, v_evals_1988_, v_r_1989_);
return v___x_1990_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___boxed(lean_object* v_EF_1991_, lean_object* v_fo_1992_, lean_object* v_evals_1993_, lean_object* v_r_1994_){
_start:
{
lean_object* v_res_1995_; 
v_res_1995_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints(v_EF_1991_, v_fo_1992_, v_evals_1993_, v_r_1994_);
lean_dec(v_evals_1993_);
return v_res_1995_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0(lean_object* v_EF_1996_, lean_object* v___x_1997_, lean_object* v_a_1998_, lean_object* v_a_1999_){
_start:
{
lean_object* v___x_2000_; 
v___x_2000_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(v___x_1997_, v_a_1998_, v_a_1999_);
return v___x_2000_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1(lean_object* v_EF_2001_, lean_object* v___x_2002_, lean_object* v___x_2003_, lean_object* v_r_2004_, lean_object* v_a_2005_, lean_object* v_a_2006_){
_start:
{
lean_object* v___x_2007_; 
v___x_2007_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(v___x_2002_, v___x_2003_, v_r_2004_, v_a_2005_, v_a_2006_);
return v___x_2007_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2(lean_object* v_EF_2008_, lean_object* v___x_2009_, lean_object* v___x_2010_, lean_object* v_sDeg_2011_, lean_object* v_r_2012_, lean_object* v_a_2013_, lean_object* v_a_2014_){
_start:
{
lean_object* v___x_2015_; 
v___x_2015_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(v___x_2009_, v___x_2010_, v_sDeg_2011_, v_r_2012_, v_a_2013_, v_a_2014_);
return v___x_2015_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___boxed(lean_object* v_EF_2016_, lean_object* v___x_2017_, lean_object* v___x_2018_, lean_object* v_sDeg_2019_, lean_object* v_r_2020_, lean_object* v_a_2021_, lean_object* v_a_2022_){
_start:
{
lean_object* v_res_2023_; 
v_res_2023_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2(v_EF_2016_, v___x_2017_, v___x_2018_, v_sDeg_2019_, v_r_2020_, v_a_2021_, v_a_2022_);
lean_dec(v_sDeg_2019_);
return v_res_2023_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3(lean_object* v_EF_2024_, lean_object* v___x_2025_, lean_object* v_sDeg_2026_, lean_object* v_invfact_2027_, lean_object* v_sufProduct_2028_, lean_object* v_prefProduct_2029_, lean_object* v_evals_2030_, lean_object* v_x_2031_, lean_object* v_x_2032_){
_start:
{
lean_object* v___x_2033_; 
v___x_2033_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(v___x_2025_, v_sDeg_2026_, v_invfact_2027_, v_sufProduct_2028_, v_prefProduct_2029_, v_evals_2030_, v_x_2031_, v_x_2032_);
return v___x_2033_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___boxed(lean_object* v_EF_2034_, lean_object* v___x_2035_, lean_object* v_sDeg_2036_, lean_object* v_invfact_2037_, lean_object* v_sufProduct_2038_, lean_object* v_prefProduct_2039_, lean_object* v_evals_2040_, lean_object* v_x_2041_, lean_object* v_x_2042_){
_start:
{
lean_object* v_res_2043_; 
v_res_2043_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3(v_EF_2034_, v___x_2035_, v_sDeg_2036_, v_invfact_2037_, v_sufProduct_2038_, v_prefProduct_2039_, v_evals_2040_, v_x_2041_, v_x_2042_);
lean_dec(v_evals_2040_);
lean_dec(v_prefProduct_2039_);
lean_dec(v_sufProduct_2038_);
lean_dec(v_invfact_2037_);
lean_dec(v_sDeg_2036_);
return v_res_2043_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(lean_object* v_fo_2044_, lean_object* v_curSum_2045_, lean_object* v_batchSEvals_2046_, lean_object* v_r_2047_){
_start:
{
lean_object* v_toRingOps_2048_; lean_object* v___y_2050_; 
v_toRingOps_2048_ = lean_ctor_get(v_fo_2044_, 0);
lean_inc_ref(v_toRingOps_2048_);
if (lean_obj_tag(v_batchSEvals_2046_) == 0)
{
lean_object* v_toSemiringOps_2062_; lean_object* v_zero_2063_; 
v_toSemiringOps_2062_ = lean_ctor_get(v_toRingOps_2048_, 0);
v_zero_2063_ = lean_ctor_get(v_toSemiringOps_2062_, 0);
lean_inc(v_zero_2063_);
v___y_2050_ = v_zero_2063_;
goto v___jp_2049_;
}
else
{
lean_object* v_head_2064_; 
v_head_2064_ = lean_ctor_get(v_batchSEvals_2046_, 0);
lean_inc(v_head_2064_);
v___y_2050_ = v_head_2064_;
goto v___jp_2049_;
}
v___jp_2049_:
{
lean_object* v_sub_2051_; lean_object* v___x_2053_; uint8_t v_isShared_2054_; uint8_t v_isSharedCheck_2060_; 
v_sub_2051_ = lean_ctor_get(v_toRingOps_2048_, 1);
v_isSharedCheck_2060_ = !lean_is_exclusive(v_toRingOps_2048_);
if (v_isSharedCheck_2060_ == 0)
{
lean_object* v_unused_2061_; 
v_unused_2061_ = lean_ctor_get(v_toRingOps_2048_, 0);
lean_dec(v_unused_2061_);
v___x_2053_ = v_toRingOps_2048_;
v_isShared_2054_ = v_isSharedCheck_2060_;
goto v_resetjp_2052_;
}
else
{
lean_inc(v_sub_2051_);
lean_dec(v_toRingOps_2048_);
v___x_2053_ = lean_box(0);
v_isShared_2054_ = v_isSharedCheck_2060_;
goto v_resetjp_2052_;
}
v_resetjp_2052_:
{
lean_object* v_s0_2055_; lean_object* v___x_2057_; 
v_s0_2055_ = lean_apply_2(v_sub_2051_, v_curSum_2045_, v___y_2050_);
if (v_isShared_2054_ == 0)
{
lean_ctor_set_tag(v___x_2053_, 1);
lean_ctor_set(v___x_2053_, 1, v_batchSEvals_2046_);
lean_ctor_set(v___x_2053_, 0, v_s0_2055_);
v___x_2057_ = v___x_2053_;
goto v_reusejp_2056_;
}
else
{
lean_object* v_reuseFailAlloc_2059_; 
v_reuseFailAlloc_2059_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2059_, 0, v_s0_2055_);
lean_ctor_set(v_reuseFailAlloc_2059_, 1, v_batchSEvals_2046_);
v___x_2057_ = v_reuseFailAlloc_2059_;
goto v_reusejp_2056_;
}
v_reusejp_2056_:
{
lean_object* v___x_2058_; 
v___x_2058_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(v_fo_2044_, v___x_2057_, v_r_2047_);
lean_dec_ref(v___x_2057_);
return v___x_2058_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound(lean_object* v_EF_2065_, lean_object* v_fo_2066_, lean_object* v_curSum_2067_, lean_object* v_batchSEvals_2068_, lean_object* v_r_2069_){
_start:
{
lean_object* v___x_2070_; 
v___x_2070_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(v_fo_2066_, v_curSum_2067_, v_batchSEvals_2068_, v_r_2069_);
return v___x_2070_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(lean_object* v_bInt_2071_, lean_object* v_fo_2072_, lean_object* v_a_2073_, lean_object* v_a_2074_){
_start:
{
if (lean_obj_tag(v_a_2073_) == 0)
{
lean_object* v___x_2075_; 
lean_dec_ref(v_fo_2072_);
v___x_2075_ = l_List_reverse___redArg(v_a_2074_);
return v___x_2075_;
}
else
{
lean_object* v_head_2076_; lean_object* v_tail_2077_; lean_object* v___x_2079_; uint8_t v_isShared_2080_; uint8_t v_isSharedCheck_2097_; 
v_head_2076_ = lean_ctor_get(v_a_2073_, 0);
v_tail_2077_ = lean_ctor_get(v_a_2073_, 1);
v_isSharedCheck_2097_ = !lean_is_exclusive(v_a_2073_);
if (v_isSharedCheck_2097_ == 0)
{
v___x_2079_ = v_a_2073_;
v_isShared_2080_ = v_isSharedCheck_2097_;
goto v_resetjp_2078_;
}
else
{
lean_inc(v_tail_2077_);
lean_inc(v_head_2076_);
lean_dec(v_a_2073_);
v___x_2079_ = lean_box(0);
v_isShared_2080_ = v_isSharedCheck_2097_;
goto v_resetjp_2078_;
}
v_resetjp_2078_:
{
lean_object* v___y_2082_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; uint8_t v___x_2090_; 
v___x_2087_ = lean_nat_shiftr(v_bInt_2071_, v_head_2076_);
lean_dec(v_head_2076_);
v___x_2088_ = lean_unsigned_to_nat(1u);
v___x_2089_ = lean_nat_land(v___x_2087_, v___x_2088_);
lean_dec(v___x_2087_);
v___x_2090_ = lean_nat_dec_eq(v___x_2089_, v___x_2088_);
lean_dec(v___x_2089_);
if (v___x_2090_ == 0)
{
lean_object* v_toRingOps_2091_; lean_object* v_toSemiringOps_2092_; lean_object* v_zero_2093_; 
v_toRingOps_2091_ = lean_ctor_get(v_fo_2072_, 0);
v_toSemiringOps_2092_ = lean_ctor_get(v_toRingOps_2091_, 0);
v_zero_2093_ = lean_ctor_get(v_toSemiringOps_2092_, 0);
lean_inc(v_zero_2093_);
v___y_2082_ = v_zero_2093_;
goto v___jp_2081_;
}
else
{
lean_object* v_toRingOps_2094_; lean_object* v_toSemiringOps_2095_; lean_object* v_one_2096_; 
v_toRingOps_2094_ = lean_ctor_get(v_fo_2072_, 0);
v_toSemiringOps_2095_ = lean_ctor_get(v_toRingOps_2094_, 0);
v_one_2096_ = lean_ctor_get(v_toSemiringOps_2095_, 1);
lean_inc(v_one_2096_);
v___y_2082_ = v_one_2096_;
goto v___jp_2081_;
}
v___jp_2081_:
{
lean_object* v___x_2084_; 
if (v_isShared_2080_ == 0)
{
lean_ctor_set(v___x_2079_, 1, v_a_2074_);
lean_ctor_set(v___x_2079_, 0, v___y_2082_);
v___x_2084_ = v___x_2079_;
goto v_reusejp_2083_;
}
else
{
lean_object* v_reuseFailAlloc_2086_; 
v_reuseFailAlloc_2086_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2086_, 0, v___y_2082_);
lean_ctor_set(v_reuseFailAlloc_2086_, 1, v_a_2074_);
v___x_2084_ = v_reuseFailAlloc_2086_;
goto v_reusejp_2083_;
}
v_reusejp_2083_:
{
v_a_2073_ = v_tail_2077_;
v_a_2074_ = v___x_2084_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg___boxed(lean_object* v_bInt_2098_, lean_object* v_fo_2099_, lean_object* v_a_2100_, lean_object* v_a_2101_){
_start:
{
lean_object* v_res_2102_; 
v_res_2102_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2098_, v_fo_2099_, v_a_2100_, v_a_2101_);
lean_dec(v_bInt_2098_);
return v_res_2102_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(lean_object* v___x_2103_, lean_object* v_bitWidth_2104_, lean_object* v_fo_2105_, lean_object* v_xiSlice_2106_, lean_object* v_blockSize_2107_, lean_object* v_x_2108_, lean_object* v_x_2109_){
_start:
{
if (lean_obj_tag(v_x_2109_) == 0)
{
lean_dec(v_xiSlice_2106_);
lean_dec_ref(v_fo_2105_);
lean_dec(v_bitWidth_2104_);
return v_x_2108_;
}
else
{
lean_object* v_tail_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2134_; 
v_tail_2110_ = lean_ctor_get(v_x_2109_, 1);
v_isSharedCheck_2134_ = !lean_is_exclusive(v_x_2109_);
if (v_isSharedCheck_2134_ == 0)
{
lean_object* v_unused_2135_; 
v_unused_2135_ = lean_ctor_get(v_x_2109_, 0);
lean_dec(v_unused_2135_);
v___x_2112_ = v_x_2109_;
v_isShared_2113_ = v_isSharedCheck_2134_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_tail_2110_);
lean_dec(v_x_2109_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2134_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v_fst_2114_; lean_object* v_snd_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2133_; 
v_fst_2114_ = lean_ctor_get(v_x_2108_, 0);
v_snd_2115_ = lean_ctor_get(v_x_2108_, 1);
v_isSharedCheck_2133_ = !lean_is_exclusive(v_x_2108_);
if (v_isSharedCheck_2133_ == 0)
{
v___x_2117_ = v_x_2108_;
v_isShared_2118_ = v_isSharedCheck_2133_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_snd_2115_);
lean_inc(v_fst_2114_);
lean_dec(v_x_2108_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2133_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
lean_object* v_toRingOps_2119_; lean_object* v_bInt_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v_bVec_2123_; lean_object* v_eq3b_2124_; lean_object* v___x_2126_; 
v_toRingOps_2119_ = lean_ctor_get(v_fo_2105_, 0);
v_bInt_2120_ = lean_nat_shiftr(v_snd_2115_, v___x_2103_);
lean_inc(v_bitWidth_2104_);
v___x_2121_ = l_List_range(v_bitWidth_2104_);
v___x_2122_ = lean_box(0);
lean_inc_ref(v_fo_2105_);
v_bVec_2123_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2120_, v_fo_2105_, v___x_2121_, v___x_2122_);
lean_dec(v_bInt_2120_);
lean_inc(v_xiSlice_2106_);
lean_inc_ref(v_toRingOps_2119_);
v_eq3b_2124_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_2119_, v_xiSlice_2106_, v_bVec_2123_);
if (v_isShared_2113_ == 0)
{
lean_ctor_set(v___x_2112_, 1, v_fst_2114_);
lean_ctor_set(v___x_2112_, 0, v_eq3b_2124_);
v___x_2126_ = v___x_2112_;
goto v_reusejp_2125_;
}
else
{
lean_object* v_reuseFailAlloc_2132_; 
v_reuseFailAlloc_2132_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2132_, 0, v_eq3b_2124_);
lean_ctor_set(v_reuseFailAlloc_2132_, 1, v_fst_2114_);
v___x_2126_ = v_reuseFailAlloc_2132_;
goto v_reusejp_2125_;
}
v_reusejp_2125_:
{
lean_object* v___x_2127_; lean_object* v___x_2129_; 
v___x_2127_ = lean_nat_add(v_snd_2115_, v_blockSize_2107_);
lean_dec(v_snd_2115_);
if (v_isShared_2118_ == 0)
{
lean_ctor_set(v___x_2117_, 1, v___x_2127_);
lean_ctor_set(v___x_2117_, 0, v___x_2126_);
v___x_2129_ = v___x_2117_;
goto v_reusejp_2128_;
}
else
{
lean_object* v_reuseFailAlloc_2131_; 
v_reuseFailAlloc_2131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2131_, 0, v___x_2126_);
lean_ctor_set(v_reuseFailAlloc_2131_, 1, v___x_2127_);
v___x_2129_ = v_reuseFailAlloc_2131_;
goto v_reusejp_2128_;
}
v_reusejp_2128_:
{
v_x_2108_ = v___x_2129_;
v_x_2109_ = v_tail_2110_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg___boxed(lean_object* v___x_2136_, lean_object* v_bitWidth_2137_, lean_object* v_fo_2138_, lean_object* v_xiSlice_2139_, lean_object* v_blockSize_2140_, lean_object* v_x_2141_, lean_object* v_x_2142_){
_start:
{
lean_object* v_res_2143_; 
v_res_2143_ = lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(v___x_2136_, v_bitWidth_2137_, v_fo_2138_, v_xiSlice_2139_, v_blockSize_2140_, v_x_2141_, v_x_2142_);
lean_dec(v_blockSize_2140_);
lean_dec(v___x_2136_);
return v_res_2143_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(lean_object* v___x_2144_, lean_object* v_fo_2145_, lean_object* v_bitWidth_2146_, lean_object* v_xiSlice_2147_, lean_object* v_blockSize_2148_, lean_object* v_x_2149_, lean_object* v_x_2150_){
_start:
{
if (lean_obj_tag(v_x_2150_) == 0)
{
lean_dec(v_xiSlice_2147_);
lean_dec(v_bitWidth_2146_);
lean_dec_ref(v_fo_2145_);
return v_x_2149_;
}
else
{
lean_object* v_tail_2151_; lean_object* v___x_2153_; uint8_t v_isShared_2154_; uint8_t v_isSharedCheck_2175_; 
v_tail_2151_ = lean_ctor_get(v_x_2150_, 1);
v_isSharedCheck_2175_ = !lean_is_exclusive(v_x_2150_);
if (v_isSharedCheck_2175_ == 0)
{
lean_object* v_unused_2176_; 
v_unused_2176_ = lean_ctor_get(v_x_2150_, 0);
lean_dec(v_unused_2176_);
v___x_2153_ = v_x_2150_;
v_isShared_2154_ = v_isSharedCheck_2175_;
goto v_resetjp_2152_;
}
else
{
lean_inc(v_tail_2151_);
lean_dec(v_x_2150_);
v___x_2153_ = lean_box(0);
v_isShared_2154_ = v_isSharedCheck_2175_;
goto v_resetjp_2152_;
}
v_resetjp_2152_:
{
lean_object* v_fst_2155_; lean_object* v_snd_2156_; lean_object* v___x_2158_; uint8_t v_isShared_2159_; uint8_t v_isSharedCheck_2174_; 
v_fst_2155_ = lean_ctor_get(v_x_2149_, 0);
v_snd_2156_ = lean_ctor_get(v_x_2149_, 1);
v_isSharedCheck_2174_ = !lean_is_exclusive(v_x_2149_);
if (v_isSharedCheck_2174_ == 0)
{
v___x_2158_ = v_x_2149_;
v_isShared_2159_ = v_isSharedCheck_2174_;
goto v_resetjp_2157_;
}
else
{
lean_inc(v_snd_2156_);
lean_inc(v_fst_2155_);
lean_dec(v_x_2149_);
v___x_2158_ = lean_box(0);
v_isShared_2159_ = v_isSharedCheck_2174_;
goto v_resetjp_2157_;
}
v_resetjp_2157_:
{
lean_object* v_toRingOps_2160_; lean_object* v_bInt_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v_bVec_2164_; lean_object* v_eq3b_2165_; lean_object* v___x_2167_; 
v_toRingOps_2160_ = lean_ctor_get(v_fo_2145_, 0);
v_bInt_2161_ = lean_nat_shiftr(v_snd_2156_, v___x_2144_);
lean_inc(v_bitWidth_2146_);
v___x_2162_ = l_List_range(v_bitWidth_2146_);
v___x_2163_ = lean_box(0);
lean_inc_ref(v_fo_2145_);
v_bVec_2164_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2161_, v_fo_2145_, v___x_2162_, v___x_2163_);
lean_dec(v_bInt_2161_);
lean_inc(v_xiSlice_2147_);
lean_inc_ref(v_toRingOps_2160_);
v_eq3b_2165_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_2160_, v_xiSlice_2147_, v_bVec_2164_);
if (v_isShared_2154_ == 0)
{
lean_ctor_set(v___x_2153_, 1, v_fst_2155_);
lean_ctor_set(v___x_2153_, 0, v_eq3b_2165_);
v___x_2167_ = v___x_2153_;
goto v_reusejp_2166_;
}
else
{
lean_object* v_reuseFailAlloc_2173_; 
v_reuseFailAlloc_2173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2173_, 0, v_eq3b_2165_);
lean_ctor_set(v_reuseFailAlloc_2173_, 1, v_fst_2155_);
v___x_2167_ = v_reuseFailAlloc_2173_;
goto v_reusejp_2166_;
}
v_reusejp_2166_:
{
lean_object* v___x_2168_; lean_object* v___x_2170_; 
v___x_2168_ = lean_nat_add(v_snd_2156_, v_blockSize_2148_);
lean_dec(v_snd_2156_);
if (v_isShared_2159_ == 0)
{
lean_ctor_set(v___x_2158_, 1, v___x_2168_);
lean_ctor_set(v___x_2158_, 0, v___x_2167_);
v___x_2170_ = v___x_2158_;
goto v_reusejp_2169_;
}
else
{
lean_object* v_reuseFailAlloc_2172_; 
v_reuseFailAlloc_2172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2172_, 0, v___x_2167_);
lean_ctor_set(v_reuseFailAlloc_2172_, 1, v___x_2168_);
v___x_2170_ = v_reuseFailAlloc_2172_;
goto v_reusejp_2169_;
}
v_reusejp_2169_:
{
lean_object* v___x_2171_; 
v___x_2171_ = lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(v___x_2144_, v_bitWidth_2146_, v_fo_2145_, v_xiSlice_2147_, v_blockSize_2148_, v___x_2170_, v_tail_2151_);
return v___x_2171_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg___boxed(lean_object* v___x_2177_, lean_object* v_fo_2178_, lean_object* v_bitWidth_2179_, lean_object* v_xiSlice_2180_, lean_object* v_blockSize_2181_, lean_object* v_x_2182_, lean_object* v_x_2183_){
_start:
{
lean_object* v_res_2184_; 
v_res_2184_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(v___x_2177_, v_fo_2178_, v_bitWidth_2179_, v_xiSlice_2180_, v_blockSize_2181_, v_x_2182_, v_x_2183_);
lean_dec(v_blockSize_2181_);
lean_dec(v___x_2177_);
return v_res_2184_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(lean_object* v_nLogup_2187_, lean_object* v_lSkip_2188_, lean_object* v_xi_2189_, lean_object* v_fo_2190_, lean_object* v_x_2191_, lean_object* v_x_2192_){
_start:
{
if (lean_obj_tag(v_x_2192_) == 0)
{
lean_dec_ref(v_fo_2190_);
return v_x_2191_;
}
else
{
lean_object* v_head_2193_; lean_object* v_tail_2194_; lean_object* v___x_2196_; uint8_t v_isShared_2197_; uint8_t v_isSharedCheck_2244_; 
v_head_2193_ = lean_ctor_get(v_x_2192_, 0);
v_tail_2194_ = lean_ctor_get(v_x_2192_, 1);
v_isSharedCheck_2244_ = !lean_is_exclusive(v_x_2192_);
if (v_isSharedCheck_2244_ == 0)
{
v___x_2196_ = v_x_2192_;
v_isShared_2197_ = v_isSharedCheck_2244_;
goto v_resetjp_2195_;
}
else
{
lean_inc(v_tail_2194_);
lean_inc(v_head_2193_);
lean_dec(v_x_2192_);
v___x_2196_ = lean_box(0);
v_isShared_2197_ = v_isSharedCheck_2244_;
goto v_resetjp_2195_;
}
v_resetjp_2195_:
{
lean_object* v_fst_2198_; lean_object* v_snd_2199_; lean_object* v_fst_2200_; lean_object* v_snd_2201_; lean_object* v___x_2203_; uint8_t v_isShared_2204_; uint8_t v_isSharedCheck_2243_; 
v_fst_2198_ = lean_ctor_get(v_x_2191_, 0);
lean_inc(v_fst_2198_);
v_snd_2199_ = lean_ctor_get(v_x_2191_, 1);
lean_inc(v_snd_2199_);
lean_dec_ref(v_x_2191_);
v_fst_2200_ = lean_ctor_get(v_head_2193_, 0);
v_snd_2201_ = lean_ctor_get(v_head_2193_, 1);
v_isSharedCheck_2243_ = !lean_is_exclusive(v_head_2193_);
if (v_isSharedCheck_2243_ == 0)
{
v___x_2203_ = v_head_2193_;
v_isShared_2204_ = v_isSharedCheck_2243_;
goto v_resetjp_2202_;
}
else
{
lean_inc(v_snd_2201_);
lean_inc(v_fst_2200_);
lean_dec(v_head_2193_);
v___x_2203_ = lean_box(0);
v_isShared_2204_ = v_isSharedCheck_2243_;
goto v_resetjp_2202_;
}
v_resetjp_2202_:
{
lean_object* v___x_2205_; uint8_t v___x_2206_; 
v___x_2205_ = lean_unsigned_to_nat(0u);
v___x_2206_ = lean_nat_dec_eq(v_fst_2200_, v___x_2205_);
if (v___x_2206_ == 0)
{
lean_object* v_nLift_2207_; lean_object* v_bitWidth_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v_xiSlice_2212_; lean_object* v___x_2213_; lean_object* v_blockSize_2214_; lean_object* v___x_2215_; lean_object* v___x_2217_; 
v_nLift_2207_ = l_Int_toNat(v_snd_2201_);
lean_dec(v_snd_2201_);
v_bitWidth_2208_ = lean_nat_sub(v_nLogup_2187_, v_nLift_2207_);
v___x_2209_ = lean_nat_add(v_lSkip_2188_, v_nLift_2207_);
lean_dec(v_nLift_2207_);
lean_inc(v___x_2209_);
v___x_2210_ = l_List_drop___redArg(v___x_2209_, v_xi_2189_);
v___x_2211_ = ((lean_object*)(lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0));
lean_inc(v_bitWidth_2208_);
lean_inc(v___x_2210_);
v_xiSlice_2212_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v___x_2210_, v___x_2210_, v_bitWidth_2208_, v___x_2211_);
lean_dec(v___x_2210_);
v___x_2213_ = lean_unsigned_to_nat(2u);
v_blockSize_2214_ = lean_nat_pow(v___x_2213_, v___x_2209_);
v___x_2215_ = lean_box(0);
if (v_isShared_2204_ == 0)
{
lean_ctor_set(v___x_2203_, 1, v_snd_2199_);
lean_ctor_set(v___x_2203_, 0, v___x_2215_);
v___x_2217_ = v___x_2203_;
goto v_reusejp_2216_;
}
else
{
lean_object* v_reuseFailAlloc_2234_; 
v_reuseFailAlloc_2234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2234_, 0, v___x_2215_);
lean_ctor_set(v_reuseFailAlloc_2234_, 1, v_snd_2199_);
v___x_2217_ = v_reuseFailAlloc_2234_;
goto v_reusejp_2216_;
}
v_reusejp_2216_:
{
lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v_fst_2220_; lean_object* v_snd_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2233_; 
v___x_2218_ = l_List_range(v_fst_2200_);
lean_inc_ref(v_fo_2190_);
v___x_2219_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(v___x_2209_, v_fo_2190_, v_bitWidth_2208_, v_xiSlice_2212_, v_blockSize_2214_, v___x_2217_, v___x_2218_);
lean_dec(v_blockSize_2214_);
lean_dec(v___x_2209_);
v_fst_2220_ = lean_ctor_get(v___x_2219_, 0);
v_snd_2221_ = lean_ctor_get(v___x_2219_, 1);
v_isSharedCheck_2233_ = !lean_is_exclusive(v___x_2219_);
if (v_isSharedCheck_2233_ == 0)
{
v___x_2223_ = v___x_2219_;
v_isShared_2224_ = v_isSharedCheck_2233_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_snd_2221_);
lean_inc(v_fst_2220_);
lean_dec(v___x_2219_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2233_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v___x_2225_; lean_object* v___x_2227_; 
v___x_2225_ = l_List_reverse___redArg(v_fst_2220_);
if (v_isShared_2197_ == 0)
{
lean_ctor_set(v___x_2196_, 1, v_fst_2198_);
lean_ctor_set(v___x_2196_, 0, v___x_2225_);
v___x_2227_ = v___x_2196_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2232_; 
v_reuseFailAlloc_2232_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2232_, 0, v___x_2225_);
lean_ctor_set(v_reuseFailAlloc_2232_, 1, v_fst_2198_);
v___x_2227_ = v_reuseFailAlloc_2232_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
lean_object* v___x_2229_; 
if (v_isShared_2224_ == 0)
{
lean_ctor_set(v___x_2223_, 0, v___x_2227_);
v___x_2229_ = v___x_2223_;
goto v_reusejp_2228_;
}
else
{
lean_object* v_reuseFailAlloc_2231_; 
v_reuseFailAlloc_2231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2231_, 0, v___x_2227_);
lean_ctor_set(v_reuseFailAlloc_2231_, 1, v_snd_2221_);
v___x_2229_ = v_reuseFailAlloc_2231_;
goto v_reusejp_2228_;
}
v_reusejp_2228_:
{
v_x_2191_ = v___x_2229_;
v_x_2192_ = v_tail_2194_;
goto _start;
}
}
}
}
}
else
{
lean_object* v___x_2235_; lean_object* v___x_2237_; 
lean_dec(v_snd_2201_);
lean_dec(v_fst_2200_);
v___x_2235_ = lean_box(0);
if (v_isShared_2197_ == 0)
{
lean_ctor_set(v___x_2196_, 1, v_fst_2198_);
lean_ctor_set(v___x_2196_, 0, v___x_2235_);
v___x_2237_ = v___x_2196_;
goto v_reusejp_2236_;
}
else
{
lean_object* v_reuseFailAlloc_2242_; 
v_reuseFailAlloc_2242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2242_, 0, v___x_2235_);
lean_ctor_set(v_reuseFailAlloc_2242_, 1, v_fst_2198_);
v___x_2237_ = v_reuseFailAlloc_2242_;
goto v_reusejp_2236_;
}
v_reusejp_2236_:
{
lean_object* v___x_2239_; 
if (v_isShared_2204_ == 0)
{
lean_ctor_set(v___x_2203_, 1, v_snd_2199_);
lean_ctor_set(v___x_2203_, 0, v___x_2237_);
v___x_2239_ = v___x_2203_;
goto v_reusejp_2238_;
}
else
{
lean_object* v_reuseFailAlloc_2241_; 
v_reuseFailAlloc_2241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2241_, 0, v___x_2237_);
lean_ctor_set(v_reuseFailAlloc_2241_, 1, v_snd_2199_);
v___x_2239_ = v_reuseFailAlloc_2241_;
goto v_reusejp_2238_;
}
v_reusejp_2238_:
{
v_x_2191_ = v___x_2239_;
v_x_2192_ = v_tail_2194_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___boxed(lean_object* v_nLogup_2245_, lean_object* v_lSkip_2246_, lean_object* v_xi_2247_, lean_object* v_fo_2248_, lean_object* v_x_2249_, lean_object* v_x_2250_){
_start:
{
lean_object* v_res_2251_; 
v_res_2251_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(v_nLogup_2245_, v_lSkip_2246_, v_xi_2247_, v_fo_2248_, v_x_2249_, v_x_2250_);
lean_dec(v_xi_2247_);
lean_dec(v_lSkip_2246_);
lean_dec(v_nLogup_2245_);
return v_res_2251_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(lean_object* v_fo_2255_, lean_object* v_lSkip_2256_, lean_object* v_nLogup_2257_, lean_object* v_xi_2258_, lean_object* v_interactionCountPerTrace_2259_, lean_object* v_nPerTrace_2260_){
_start:
{
lean_object* v_entries_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v_fst_2264_; lean_object* v___x_2265_; 
v_entries_2261_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_interactionCountPerTrace_2259_, v_nPerTrace_2260_);
v___x_2262_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0));
v___x_2263_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(v_nLogup_2257_, v_lSkip_2256_, v_xi_2258_, v_fo_2255_, v___x_2262_, v_entries_2261_);
v_fst_2264_ = lean_ctor_get(v___x_2263_, 0);
lean_inc(v_fst_2264_);
lean_dec_ref(v___x_2263_);
v___x_2265_ = l_List_reverse___redArg(v_fst_2264_);
return v___x_2265_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___boxed(lean_object* v_fo_2266_, lean_object* v_lSkip_2267_, lean_object* v_nLogup_2268_, lean_object* v_xi_2269_, lean_object* v_interactionCountPerTrace_2270_, lean_object* v_nPerTrace_2271_){
_start:
{
lean_object* v_res_2272_; 
v_res_2272_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(v_fo_2266_, v_lSkip_2267_, v_nLogup_2268_, v_xi_2269_, v_interactionCountPerTrace_2270_, v_nPerTrace_2271_);
lean_dec(v_xi_2269_);
lean_dec(v_nLogup_2268_);
lean_dec(v_lSkip_2267_);
return v_res_2272_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace(lean_object* v_EF_2273_, lean_object* v_fo_2274_, lean_object* v_lSkip_2275_, lean_object* v_nLogup_2276_, lean_object* v_xi_2277_, lean_object* v_interactionCountPerTrace_2278_, lean_object* v_nPerTrace_2279_){
_start:
{
lean_object* v___x_2280_; 
v___x_2280_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(v_fo_2274_, v_lSkip_2275_, v_nLogup_2276_, v_xi_2277_, v_interactionCountPerTrace_2278_, v_nPerTrace_2279_);
return v___x_2280_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___boxed(lean_object* v_EF_2281_, lean_object* v_fo_2282_, lean_object* v_lSkip_2283_, lean_object* v_nLogup_2284_, lean_object* v_xi_2285_, lean_object* v_interactionCountPerTrace_2286_, lean_object* v_nPerTrace_2287_){
_start:
{
lean_object* v_res_2288_; 
v_res_2288_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace(v_EF_2281_, v_fo_2282_, v_lSkip_2283_, v_nLogup_2284_, v_xi_2285_, v_interactionCountPerTrace_2286_, v_nPerTrace_2287_);
lean_dec(v_xi_2285_);
lean_dec(v_nLogup_2284_);
lean_dec(v_lSkip_2283_);
return v_res_2288_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0(lean_object* v_EF_2289_, lean_object* v_bInt_2290_, lean_object* v_fo_2291_, lean_object* v_a_2292_, lean_object* v_a_2293_){
_start:
{
lean_object* v___x_2294_; 
v___x_2294_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2290_, v_fo_2291_, v_a_2292_, v_a_2293_);
return v___x_2294_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___boxed(lean_object* v_EF_2295_, lean_object* v_bInt_2296_, lean_object* v_fo_2297_, lean_object* v_a_2298_, lean_object* v_a_2299_){
_start:
{
lean_object* v_res_2300_; 
v_res_2300_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0(v_EF_2295_, v_bInt_2296_, v_fo_2297_, v_a_2298_, v_a_2299_);
lean_dec(v_bInt_2296_);
return v_res_2300_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1(lean_object* v_EF_2301_, lean_object* v___x_2302_, lean_object* v_fo_2303_, lean_object* v_bitWidth_2304_, lean_object* v_xiSlice_2305_, lean_object* v_blockSize_2306_, lean_object* v_x_2307_, lean_object* v_x_2308_){
_start:
{
lean_object* v___x_2309_; 
v___x_2309_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(v___x_2302_, v_fo_2303_, v_bitWidth_2304_, v_xiSlice_2305_, v_blockSize_2306_, v_x_2307_, v_x_2308_);
return v___x_2309_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___boxed(lean_object* v_EF_2310_, lean_object* v___x_2311_, lean_object* v_fo_2312_, lean_object* v_bitWidth_2313_, lean_object* v_xiSlice_2314_, lean_object* v_blockSize_2315_, lean_object* v_x_2316_, lean_object* v_x_2317_){
_start:
{
lean_object* v_res_2318_; 
v_res_2318_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1(v_EF_2310_, v___x_2311_, v_fo_2312_, v_bitWidth_2313_, v_xiSlice_2314_, v_blockSize_2315_, v_x_2316_, v_x_2317_);
lean_dec(v_blockSize_2315_);
lean_dec(v___x_2311_);
return v_res_2318_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2(lean_object* v_EF_2319_, lean_object* v_nLogup_2320_, lean_object* v_lSkip_2321_, lean_object* v_xi_2322_, lean_object* v_fo_2323_, lean_object* v_x_2324_, lean_object* v_x_2325_){
_start:
{
lean_object* v___x_2326_; 
v___x_2326_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(v_nLogup_2320_, v_lSkip_2321_, v_xi_2322_, v_fo_2323_, v_x_2324_, v_x_2325_);
return v___x_2326_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___boxed(lean_object* v_EF_2327_, lean_object* v_nLogup_2328_, lean_object* v_lSkip_2329_, lean_object* v_xi_2330_, lean_object* v_fo_2331_, lean_object* v_x_2332_, lean_object* v_x_2333_){
_start:
{
lean_object* v_res_2334_; 
v_res_2334_ = lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2(v_EF_2327_, v_nLogup_2328_, v_lSkip_2329_, v_xi_2330_, v_fo_2331_, v_x_2332_, v_x_2333_);
lean_dec(v_xi_2330_);
lean_dec(v_lSkip_2329_);
lean_dec(v_nLogup_2328_);
return v_res_2334_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1(lean_object* v_EF_2335_, lean_object* v___x_2336_, lean_object* v_bitWidth_2337_, lean_object* v_fo_2338_, lean_object* v_xiSlice_2339_, lean_object* v_blockSize_2340_, lean_object* v_x_2341_, lean_object* v_x_2342_){
_start:
{
lean_object* v___x_2343_; 
v___x_2343_ = lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(v___x_2336_, v_bitWidth_2337_, v_fo_2338_, v_xiSlice_2339_, v_blockSize_2340_, v_x_2341_, v_x_2342_);
return v___x_2343_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___boxed(lean_object* v_EF_2344_, lean_object* v___x_2345_, lean_object* v_bitWidth_2346_, lean_object* v_fo_2347_, lean_object* v_xiSlice_2348_, lean_object* v_blockSize_2349_, lean_object* v_x_2350_, lean_object* v_x_2351_){
_start:
{
lean_object* v_res_2352_; 
v_res_2352_ = lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1(v_EF_2344_, v___x_2345_, v_bitWidth_2346_, v_fo_2347_, v_xiSlice_2348_, v_blockSize_2349_, v_x_2350_, v_x_2351_);
lean_dec(v_blockSize_2349_);
lean_dec(v___x_2345_);
return v_res_2352_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(lean_object* v___x_2353_, lean_object* v_a_2354_, lean_object* v_a_2355_){
_start:
{
if (lean_obj_tag(v_a_2354_) == 0)
{
lean_object* v___x_2356_; 
lean_dec_ref(v___x_2353_);
v___x_2356_ = l_List_reverse___redArg(v_a_2355_);
return v___x_2356_;
}
else
{
lean_object* v_head_2357_; lean_object* v_tail_2358_; lean_object* v___x_2360_; uint8_t v_isShared_2361_; uint8_t v_isSharedCheck_2370_; 
v_head_2357_ = lean_ctor_get(v_a_2354_, 0);
v_tail_2358_ = lean_ctor_get(v_a_2354_, 1);
v_isSharedCheck_2370_ = !lean_is_exclusive(v_a_2354_);
if (v_isSharedCheck_2370_ == 0)
{
v___x_2360_ = v_a_2354_;
v_isShared_2361_ = v_isSharedCheck_2370_;
goto v_resetjp_2359_;
}
else
{
lean_inc(v_tail_2358_);
lean_inc(v_head_2357_);
lean_dec(v_a_2354_);
v___x_2360_ = lean_box(0);
v_isShared_2361_ = v_isSharedCheck_2370_;
goto v_resetjp_2359_;
}
v_resetjp_2359_:
{
lean_object* v_fst_2362_; lean_object* v_snd_2363_; lean_object* v_mul_2364_; lean_object* v___x_2365_; lean_object* v___x_2367_; 
v_fst_2362_ = lean_ctor_get(v_head_2357_, 0);
lean_inc(v_fst_2362_);
v_snd_2363_ = lean_ctor_get(v_head_2357_, 1);
lean_inc(v_snd_2363_);
lean_dec(v_head_2357_);
v_mul_2364_ = lean_ctor_get(v___x_2353_, 4);
lean_inc(v_mul_2364_);
v___x_2365_ = lean_apply_2(v_mul_2364_, v_fst_2362_, v_snd_2363_);
if (v_isShared_2361_ == 0)
{
lean_ctor_set(v___x_2360_, 1, v_a_2355_);
lean_ctor_set(v___x_2360_, 0, v___x_2365_);
v___x_2367_ = v___x_2360_;
goto v_reusejp_2366_;
}
else
{
lean_object* v_reuseFailAlloc_2369_; 
v_reuseFailAlloc_2369_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2369_, 0, v___x_2365_);
lean_ctor_set(v_reuseFailAlloc_2369_, 1, v_a_2355_);
v___x_2367_ = v_reuseFailAlloc_2369_;
goto v_reusejp_2366_;
}
v_reusejp_2366_:
{
v_a_2354_ = v_tail_2358_;
v_a_2355_ = v___x_2367_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(lean_object* v_factors_2371_, lean_object* v_toSemiringOps_2372_, lean_object* v___x_2373_, lean_object* v_eqs_2374_){
_start:
{
lean_object* v___x_2375_; lean_object* v___x_2376_; 
v___x_2375_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_eqs_2374_, v_factors_2371_);
v___x_2376_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(v_toSemiringOps_2372_, v___x_2375_, v___x_2373_);
return v___x_2376_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(lean_object* v_nMax_2377_, lean_object* v_rs_2378_, lean_object* v___x_2379_, lean_object* v_a_2380_, lean_object* v_a_2381_){
_start:
{
if (lean_obj_tag(v_a_2380_) == 0)
{
lean_object* v___x_2382_; 
lean_dec(v___x_2379_);
v___x_2382_ = l_List_reverse___redArg(v_a_2381_);
return v___x_2382_;
}
else
{
lean_object* v_head_2383_; lean_object* v_tail_2384_; lean_object* v___x_2386_; uint8_t v_isShared_2387_; uint8_t v_isSharedCheck_2397_; 
v_head_2383_ = lean_ctor_get(v_a_2380_, 0);
v_tail_2384_ = lean_ctor_get(v_a_2380_, 1);
v_isSharedCheck_2397_ = !lean_is_exclusive(v_a_2380_);
if (v_isSharedCheck_2397_ == 0)
{
v___x_2386_ = v_a_2380_;
v_isShared_2387_ = v_isSharedCheck_2397_;
goto v_resetjp_2385_;
}
else
{
lean_inc(v_tail_2384_);
lean_inc(v_head_2383_);
lean_dec(v_a_2380_);
v___x_2386_ = lean_box(0);
v_isShared_2387_ = v_isSharedCheck_2397_;
goto v_resetjp_2385_;
}
v_resetjp_2385_:
{
lean_object* v___y_2389_; lean_object* v___x_2394_; lean_object* v___x_2395_; 
v___x_2394_ = lean_nat_sub(v_nMax_2377_, v_head_2383_);
lean_dec(v_head_2383_);
v___x_2395_ = l_List_get_x3fInternal___redArg(v_rs_2378_, v___x_2394_);
if (lean_obj_tag(v___x_2395_) == 0)
{
lean_inc(v___x_2379_);
v___y_2389_ = v___x_2379_;
goto v___jp_2388_;
}
else
{
lean_object* v_val_2396_; 
v_val_2396_ = lean_ctor_get(v___x_2395_, 0);
lean_inc(v_val_2396_);
lean_dec_ref_known(v___x_2395_, 1);
v___y_2389_ = v_val_2396_;
goto v___jp_2388_;
}
v___jp_2388_:
{
lean_object* v___x_2391_; 
if (v_isShared_2387_ == 0)
{
lean_ctor_set(v___x_2386_, 1, v_a_2381_);
lean_ctor_set(v___x_2386_, 0, v___y_2389_);
v___x_2391_ = v___x_2386_;
goto v_reusejp_2390_;
}
else
{
lean_object* v_reuseFailAlloc_2393_; 
v_reuseFailAlloc_2393_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2393_, 0, v___y_2389_);
lean_ctor_set(v_reuseFailAlloc_2393_, 1, v_a_2381_);
v___x_2391_ = v_reuseFailAlloc_2393_;
goto v_reusejp_2390_;
}
v_reusejp_2390_:
{
v_a_2380_ = v_tail_2384_;
v_a_2381_ = v___x_2391_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg___boxed(lean_object* v_nMax_2398_, lean_object* v_rs_2399_, lean_object* v___x_2400_, lean_object* v_a_2401_, lean_object* v_a_2402_){
_start:
{
lean_object* v_res_2403_; 
v_res_2403_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(v_nMax_2398_, v_rs_2399_, v___x_2400_, v_a_2401_, v_a_2402_);
lean_dec(v_rs_2399_);
lean_dec(v_nMax_2398_);
return v_res_2403_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(lean_object* v___x_2404_, lean_object* v_rs_2405_, lean_object* v___x_2406_, lean_object* v_lSkip_2407_, lean_object* v_xi_2408_, lean_object* v_a_2409_, lean_object* v_a_2410_){
_start:
{
if (lean_obj_tag(v_a_2409_) == 0)
{
lean_object* v___x_2411_; 
lean_dec(v___x_2406_);
lean_dec_ref(v___x_2404_);
v___x_2411_ = l_List_reverse___redArg(v_a_2410_);
return v___x_2411_;
}
else
{
lean_object* v_head_2412_; lean_object* v_tail_2413_; lean_object* v___x_2415_; uint8_t v_isShared_2416_; uint8_t v_isSharedCheck_2438_; 
v_head_2412_ = lean_ctor_get(v_a_2409_, 0);
v_tail_2413_ = lean_ctor_get(v_a_2409_, 1);
v_isSharedCheck_2438_ = !lean_is_exclusive(v_a_2409_);
if (v_isSharedCheck_2438_ == 0)
{
v___x_2415_ = v_a_2409_;
v_isShared_2416_ = v_isSharedCheck_2438_;
goto v_resetjp_2414_;
}
else
{
lean_inc(v_tail_2413_);
lean_inc(v_head_2412_);
lean_dec(v_a_2409_);
v___x_2415_ = lean_box(0);
v_isShared_2416_ = v_isSharedCheck_2438_;
goto v_resetjp_2414_;
}
v_resetjp_2414_:
{
lean_object* v___y_2418_; lean_object* v___y_2419_; lean_object* v___y_2420_; lean_object* v___y_2428_; lean_object* v___x_2435_; lean_object* v___x_2436_; 
v___x_2435_ = lean_nat_add(v_lSkip_2407_, v_head_2412_);
v___x_2436_ = l_List_get_x3fInternal___redArg(v_xi_2408_, v___x_2435_);
if (lean_obj_tag(v___x_2436_) == 0)
{
lean_inc(v___x_2406_);
v___y_2428_ = v___x_2406_;
goto v___jp_2427_;
}
else
{
lean_object* v_val_2437_; 
v_val_2437_ = lean_ctor_get(v___x_2436_, 0);
lean_inc(v_val_2437_);
lean_dec_ref_known(v___x_2436_, 1);
v___y_2428_ = v_val_2437_;
goto v___jp_2427_;
}
v___jp_2417_:
{
lean_object* v___x_2422_; 
lean_inc(v___y_2418_);
if (v_isShared_2416_ == 0)
{
lean_ctor_set(v___x_2415_, 1, v___y_2418_);
lean_ctor_set(v___x_2415_, 0, v___y_2420_);
v___x_2422_ = v___x_2415_;
goto v_reusejp_2421_;
}
else
{
lean_object* v_reuseFailAlloc_2426_; 
v_reuseFailAlloc_2426_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2426_, 0, v___y_2420_);
lean_ctor_set(v_reuseFailAlloc_2426_, 1, v___y_2418_);
v___x_2422_ = v_reuseFailAlloc_2426_;
goto v_reusejp_2421_;
}
v_reusejp_2421_:
{
lean_object* v___x_2423_; lean_object* v___x_2424_; 
lean_inc_ref(v___x_2404_);
v___x_2423_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v___x_2404_, v___y_2419_, v___x_2422_);
v___x_2424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2424_, 0, v___x_2423_);
lean_ctor_set(v___x_2424_, 1, v_a_2410_);
v_a_2409_ = v_tail_2413_;
v_a_2410_ = v___x_2424_;
goto _start;
}
}
v___jp_2427_:
{
lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; 
v___x_2429_ = lean_box(0);
v___x_2430_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2430_, 0, v___y_2428_);
lean_ctor_set(v___x_2430_, 1, v___x_2429_);
v___x_2431_ = lean_unsigned_to_nat(1u);
v___x_2432_ = lean_nat_add(v_head_2412_, v___x_2431_);
lean_dec(v_head_2412_);
v___x_2433_ = l_List_get_x3fInternal___redArg(v_rs_2405_, v___x_2432_);
if (lean_obj_tag(v___x_2433_) == 0)
{
lean_inc(v___x_2406_);
v___y_2418_ = v___x_2429_;
v___y_2419_ = v___x_2430_;
v___y_2420_ = v___x_2406_;
goto v___jp_2417_;
}
else
{
lean_object* v_val_2434_; 
v_val_2434_ = lean_ctor_get(v___x_2433_, 0);
lean_inc(v_val_2434_);
lean_dec_ref_known(v___x_2433_, 1);
v___y_2418_ = v___x_2429_;
v___y_2419_ = v___x_2430_;
v___y_2420_ = v_val_2434_;
goto v___jp_2417_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg___boxed(lean_object* v___x_2439_, lean_object* v_rs_2440_, lean_object* v___x_2441_, lean_object* v_lSkip_2442_, lean_object* v_xi_2443_, lean_object* v_a_2444_, lean_object* v_a_2445_){
_start:
{
lean_object* v_res_2446_; 
v_res_2446_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(v___x_2439_, v_rs_2440_, v___x_2441_, v_lSkip_2442_, v_xi_2443_, v_a_2444_, v_a_2445_);
lean_dec(v_xi_2443_);
lean_dec(v_lSkip_2442_);
lean_dec(v_rs_2440_);
return v_res_2446_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(lean_object* v_fo_2447_, lean_object* v_lSkip_2448_, lean_object* v_nMax_2449_, lean_object* v_xi_2450_, lean_object* v_rs_2451_, lean_object* v_omegaSkipPows_2452_){
_start:
{
lean_object* v_toRingOps_2453_; lean_object* v_toSemiringOps_2454_; lean_object* v_zero_2455_; lean_object* v_one_2456_; lean_object* v_mul_2457_; lean_object* v___f_2458_; lean_object* v___y_2460_; lean_object* v___y_2461_; lean_object* v___y_2478_; 
v_toRingOps_2453_ = lean_ctor_get(v_fo_2447_, 0);
lean_inc_ref(v_toRingOps_2453_);
v_toSemiringOps_2454_ = lean_ctor_get(v_toRingOps_2453_, 0);
lean_inc_ref(v_toSemiringOps_2454_);
v_zero_2455_ = lean_ctor_get(v_toSemiringOps_2454_, 0);
v_one_2456_ = lean_ctor_get(v_toSemiringOps_2454_, 1);
v_mul_2457_ = lean_ctor_get(v_toSemiringOps_2454_, 4);
lean_inc(v_mul_2457_);
v___f_2458_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2458_, 0, v_mul_2457_);
if (lean_obj_tag(v_rs_2451_) == 0)
{
lean_inc(v_zero_2455_);
v___y_2478_ = v_zero_2455_;
goto v___jp_2477_;
}
else
{
lean_object* v_head_2480_; 
v_head_2480_ = lean_ctor_get(v_rs_2451_, 0);
lean_inc(v_head_2480_);
v___y_2478_ = v_head_2480_;
goto v___jp_2477_;
}
v___jp_2459_:
{
lean_object* v___x_2462_; lean_object* v_xiPrefix_2463_; lean_object* v_eqN0_2464_; lean_object* v_eqSharpN0_2465_; lean_object* v___x_2466_; lean_object* v___x_2467_; lean_object* v_multipliers_2468_; lean_object* v_eqNsUnadj_2469_; lean_object* v_eqSharpNsUnadj_2470_; lean_object* v___x_2471_; lean_object* v_factorsRev_2472_; lean_object* v_factors_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; 
v___x_2462_ = ((lean_object*)(lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0));
lean_inc_n(v_lSkip_2448_, 3);
lean_inc(v_xi_2450_);
v_xiPrefix_2463_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_xi_2450_, v_xi_2450_, v_lSkip_2448_, v___x_2462_);
lean_inc_ref(v_fo_2447_);
v_eqN0_2464_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_2447_, v_lSkip_2448_, v___y_2461_, v___y_2460_);
lean_dec(v___y_2461_);
v_eqSharpN0_2465_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(v_fo_2447_, v_lSkip_2448_, v_omegaSkipPows_2452_, v_xiPrefix_2463_, v___y_2460_);
lean_dec(v___y_2460_);
lean_inc(v_nMax_2449_);
v___x_2466_ = l_List_range(v_nMax_2449_);
v___x_2467_ = lean_box(0);
lean_inc(v___x_2466_);
lean_inc_n(v_zero_2455_, 2);
v_multipliers_2468_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(v_toRingOps_2453_, v_rs_2451_, v_zero_2455_, v_lSkip_2448_, v_xi_2450_, v___x_2466_, v___x_2467_);
lean_dec(v_xi_2450_);
lean_dec(v_lSkip_2448_);
lean_inc(v_multipliers_2468_);
lean_inc_ref_n(v___f_2458_, 2);
v_eqNsUnadj_2469_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2458_, v_eqN0_2464_, v_multipliers_2468_);
v_eqSharpNsUnadj_2470_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2458_, v_eqSharpN0_2465_, v_multipliers_2468_);
v___x_2471_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(v_nMax_2449_, v_rs_2451_, v_zero_2455_, v___x_2466_, v___x_2467_);
lean_dec(v_rs_2451_);
lean_dec(v_nMax_2449_);
lean_inc(v_one_2456_);
v_factorsRev_2472_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2458_, v_one_2456_, v___x_2471_);
v_factors_2473_ = l_List_reverse___redArg(v_factorsRev_2472_);
lean_inc_ref(v_toSemiringOps_2454_);
lean_inc(v_factors_2473_);
v___x_2474_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(v_factors_2473_, v_toSemiringOps_2454_, v___x_2467_, v_eqNsUnadj_2469_);
v___x_2475_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(v_factors_2473_, v_toSemiringOps_2454_, v___x_2467_, v_eqSharpNsUnadj_2470_);
v___x_2476_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2476_, 0, v___x_2474_);
lean_ctor_set(v___x_2476_, 1, v___x_2475_);
return v___x_2476_;
}
v___jp_2477_:
{
if (lean_obj_tag(v_xi_2450_) == 0)
{
lean_inc(v_zero_2455_);
v___y_2460_ = v___y_2478_;
v___y_2461_ = v_zero_2455_;
goto v___jp_2459_;
}
else
{
lean_object* v_head_2479_; 
v_head_2479_ = lean_ctor_get(v_xi_2450_, 0);
lean_inc(v_head_2479_);
v___y_2460_ = v___y_2478_;
v___y_2461_ = v_head_2479_;
goto v___jp_2459_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs(lean_object* v_EF_2481_, lean_object* v_fo_2482_, lean_object* v_lSkip_2483_, lean_object* v_nMax_2484_, lean_object* v_xi_2485_, lean_object* v_rs_2486_, lean_object* v_omegaSkipPows_2487_){
_start:
{
lean_object* v___x_2488_; 
v___x_2488_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(v_fo_2482_, v_lSkip_2483_, v_nMax_2484_, v_xi_2485_, v_rs_2486_, v_omegaSkipPows_2487_);
return v___x_2488_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0(lean_object* v_EF_2489_, lean_object* v___x_2490_, lean_object* v_rs_2491_, lean_object* v___x_2492_, lean_object* v_lSkip_2493_, lean_object* v_xi_2494_, lean_object* v_a_2495_, lean_object* v_a_2496_){
_start:
{
lean_object* v___x_2497_; 
v___x_2497_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(v___x_2490_, v_rs_2491_, v___x_2492_, v_lSkip_2493_, v_xi_2494_, v_a_2495_, v_a_2496_);
return v___x_2497_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___boxed(lean_object* v_EF_2498_, lean_object* v___x_2499_, lean_object* v_rs_2500_, lean_object* v___x_2501_, lean_object* v_lSkip_2502_, lean_object* v_xi_2503_, lean_object* v_a_2504_, lean_object* v_a_2505_){
_start:
{
lean_object* v_res_2506_; 
v_res_2506_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0(v_EF_2498_, v___x_2499_, v_rs_2500_, v___x_2501_, v_lSkip_2502_, v_xi_2503_, v_a_2504_, v_a_2505_);
lean_dec(v_xi_2503_);
lean_dec(v_lSkip_2502_);
lean_dec(v_rs_2500_);
return v_res_2506_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1(lean_object* v_EF_2507_, lean_object* v_nMax_2508_, lean_object* v_rs_2509_, lean_object* v___x_2510_, lean_object* v_a_2511_, lean_object* v_a_2512_){
_start:
{
lean_object* v___x_2513_; 
v___x_2513_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(v_nMax_2508_, v_rs_2509_, v___x_2510_, v_a_2511_, v_a_2512_);
return v___x_2513_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___boxed(lean_object* v_EF_2514_, lean_object* v_nMax_2515_, lean_object* v_rs_2516_, lean_object* v___x_2517_, lean_object* v_a_2518_, lean_object* v_a_2519_){
_start:
{
lean_object* v_res_2520_; 
v_res_2520_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1(v_EF_2514_, v_nMax_2515_, v_rs_2516_, v___x_2517_, v_a_2518_, v_a_2519_);
lean_dec(v_rs_2516_);
lean_dec(v_nMax_2515_);
return v_res_2520_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2(lean_object* v_EF_2521_, lean_object* v___x_2522_, lean_object* v_a_2523_, lean_object* v_a_2524_){
_start:
{
lean_object* v___x_2525_; 
v___x_2525_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(v___x_2522_, v_a_2523_, v_a_2524_);
return v___x_2525_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0(lean_object* v_a_2526_, lean_object* v_mul_2527_, lean_object* v_add_2528_, lean_object* v_acc_2529_, lean_object* v_entry_2530_){
_start:
{
lean_object* v_fst_2531_; lean_object* v_snd_2532_; lean_object* v___x_2533_; 
v_fst_2531_ = lean_ctor_get(v_entry_2530_, 0);
lean_inc(v_fst_2531_);
v_snd_2532_ = lean_ctor_get(v_entry_2530_, 1);
lean_inc(v_snd_2532_);
lean_dec_ref(v_entry_2530_);
v___x_2533_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2526_, v_snd_2532_);
if (lean_obj_tag(v___x_2533_) == 0)
{
lean_dec(v_fst_2531_);
lean_dec(v_acc_2529_);
lean_dec(v_add_2528_);
lean_dec(v_mul_2527_);
return v___x_2533_;
}
else
{
lean_object* v_a_2534_; lean_object* v___x_2536_; uint8_t v_isShared_2537_; uint8_t v_isSharedCheck_2543_; 
v_a_2534_ = lean_ctor_get(v___x_2533_, 0);
v_isSharedCheck_2543_ = !lean_is_exclusive(v___x_2533_);
if (v_isSharedCheck_2543_ == 0)
{
v___x_2536_ = v___x_2533_;
v_isShared_2537_ = v_isSharedCheck_2543_;
goto v_resetjp_2535_;
}
else
{
lean_inc(v_a_2534_);
lean_dec(v___x_2533_);
v___x_2536_ = lean_box(0);
v_isShared_2537_ = v_isSharedCheck_2543_;
goto v_resetjp_2535_;
}
v_resetjp_2535_:
{
lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2541_; 
v___x_2538_ = lean_apply_2(v_mul_2527_, v_a_2534_, v_fst_2531_);
v___x_2539_ = lean_apply_2(v_add_2528_, v_acc_2529_, v___x_2538_);
if (v_isShared_2537_ == 0)
{
lean_ctor_set(v___x_2536_, 0, v___x_2539_);
v___x_2541_ = v___x_2536_;
goto v_reusejp_2540_;
}
else
{
lean_object* v_reuseFailAlloc_2542_; 
v_reuseFailAlloc_2542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2542_, 0, v___x_2539_);
v___x_2541_ = v_reuseFailAlloc_2542_;
goto v_reusejp_2540_;
}
v_reusejp_2540_:
{
return v___x_2541_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0___boxed(lean_object* v_a_2544_, lean_object* v_mul_2545_, lean_object* v_add_2546_, lean_object* v_acc_2547_, lean_object* v_entry_2548_){
_start:
{
lean_object* v_res_2549_; 
v_res_2549_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0(v_a_2544_, v_mul_2545_, v_add_2546_, v_acc_2547_, v_entry_2548_);
lean_dec(v_a_2544_);
return v_res_2549_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1(lean_object* v_a_2550_, lean_object* v_idx_2551_){
_start:
{
lean_object* v___x_2552_; 
v___x_2552_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2550_, v_idx_2551_);
return v___x_2552_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1___boxed(lean_object* v_a_2553_, lean_object* v_idx_2554_){
_start:
{
lean_object* v_res_2555_; 
v_res_2555_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1(v_a_2553_, v_idx_2554_);
lean_dec(v_a_2553_);
return v_res_2555_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2(lean_object* v_mul_2556_, lean_object* v_add_2557_, lean_object* v_acc_2558_, lean_object* v_entry_2559_){
_start:
{
lean_object* v_fst_2560_; lean_object* v_snd_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; 
v_fst_2560_ = lean_ctor_get(v_entry_2559_, 0);
lean_inc(v_fst_2560_);
v_snd_2561_ = lean_ctor_get(v_entry_2559_, 1);
lean_inc(v_snd_2561_);
lean_dec_ref(v_entry_2559_);
v___x_2562_ = lean_apply_2(v_mul_2556_, v_fst_2560_, v_snd_2561_);
v___x_2563_ = lean_apply_2(v_add_2557_, v_acc_2558_, v___x_2562_);
return v___x_2563_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3(lean_object* v_a_2567_, lean_object* v___x_2568_, lean_object* v___f_2569_, lean_object* v_natCast_2570_, lean_object* v_add_2571_, lean_object* v_one_2572_, lean_object* v_fo_2573_, lean_object* v_betaLogup_2574_, lean_object* v___f_2575_, lean_object* v_zero_2576_, lean_object* v_interaction_2577_){
_start:
{
lean_object* v_message_2578_; lean_object* v_count_2579_; lean_object* v_busIndex_2580_; lean_object* v___x_2581_; 
v_message_2578_ = lean_ctor_get(v_interaction_2577_, 0);
lean_inc(v_message_2578_);
v_count_2579_ = lean_ctor_get(v_interaction_2577_, 1);
lean_inc(v_count_2579_);
v_busIndex_2580_ = lean_ctor_get(v_interaction_2577_, 2);
lean_inc(v_busIndex_2580_);
lean_dec_ref(v_interaction_2577_);
v___x_2581_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2567_, v_count_2579_);
if (lean_obj_tag(v___x_2581_) == 0)
{
lean_object* v___x_2582_; 
lean_dec_ref_known(v___x_2581_, 1);
lean_dec(v_busIndex_2580_);
lean_dec(v_message_2578_);
lean_dec(v_zero_2576_);
lean_dec(v___f_2575_);
lean_dec(v_betaLogup_2574_);
lean_dec_ref(v_fo_2573_);
lean_dec(v_one_2572_);
lean_dec(v_add_2571_);
lean_dec(v_natCast_2570_);
lean_dec_ref(v___f_2569_);
lean_dec_ref(v___x_2568_);
v___x_2582_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0));
return v___x_2582_;
}
else
{
lean_object* v_a_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; 
v_a_2583_ = lean_ctor_get(v___x_2581_, 0);
lean_inc(v_a_2583_);
lean_dec_ref_known(v___x_2581_, 1);
v___x_2584_ = lean_box(0);
v___x_2585_ = l_List_mapM_loop___redArg(v___x_2568_, v___f_2569_, v_message_2578_, v___x_2584_);
if (lean_obj_tag(v___x_2585_) == 0)
{
lean_object* v_a_2586_; lean_object* v___x_2588_; uint8_t v_isShared_2589_; uint8_t v_isSharedCheck_2593_; 
lean_dec(v_a_2583_);
lean_dec(v_busIndex_2580_);
lean_dec(v_zero_2576_);
lean_dec(v___f_2575_);
lean_dec(v_betaLogup_2574_);
lean_dec_ref(v_fo_2573_);
lean_dec(v_one_2572_);
lean_dec(v_add_2571_);
lean_dec(v_natCast_2570_);
v_a_2586_ = lean_ctor_get(v___x_2585_, 0);
v_isSharedCheck_2593_ = !lean_is_exclusive(v___x_2585_);
if (v_isSharedCheck_2593_ == 0)
{
v___x_2588_ = v___x_2585_;
v_isShared_2589_ = v_isSharedCheck_2593_;
goto v_resetjp_2587_;
}
else
{
lean_inc(v_a_2586_);
lean_dec(v___x_2585_);
v___x_2588_ = lean_box(0);
v_isShared_2589_ = v_isSharedCheck_2593_;
goto v_resetjp_2587_;
}
v_resetjp_2587_:
{
lean_object* v___x_2591_; 
if (v_isShared_2589_ == 0)
{
v___x_2591_ = v___x_2588_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2592_; 
v_reuseFailAlloc_2592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2592_, 0, v_a_2586_);
v___x_2591_ = v_reuseFailAlloc_2592_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
return v___x_2591_;
}
}
}
else
{
lean_object* v_a_2594_; lean_object* v___x_2596_; uint8_t v_isShared_2597_; uint8_t v_isSharedCheck_2610_; 
v_a_2594_ = lean_ctor_get(v___x_2585_, 0);
v_isSharedCheck_2610_ = !lean_is_exclusive(v___x_2585_);
if (v_isSharedCheck_2610_ == 0)
{
v___x_2596_ = v___x_2585_;
v_isShared_2597_ = v_isSharedCheck_2610_;
goto v_resetjp_2595_;
}
else
{
lean_inc(v_a_2594_);
lean_dec(v___x_2585_);
v___x_2596_ = lean_box(0);
v_isShared_2597_ = v_isSharedCheck_2610_;
goto v_resetjp_2595_;
}
v_resetjp_2595_:
{
lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2608_; 
v___x_2598_ = lean_apply_1(v_natCast_2570_, v_busIndex_2580_);
v___x_2599_ = lean_apply_2(v_add_2571_, v___x_2598_, v_one_2572_);
v___x_2600_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2600_, 0, v___x_2599_);
lean_ctor_set(v___x_2600_, 1, v___x_2584_);
v___x_2601_ = l_List_appendTR___redArg(v_a_2594_, v___x_2600_);
v___x_2602_ = l_List_lengthTR___redArg(v___x_2601_);
v___x_2603_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_2573_, v_betaLogup_2574_, v___x_2602_);
v___x_2604_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_2601_, v___x_2603_);
v___x_2605_ = l_List_foldl___redArg(v___f_2575_, v_zero_2576_, v___x_2604_);
v___x_2606_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2606_, 0, v_a_2583_);
lean_ctor_set(v___x_2606_, 1, v___x_2605_);
if (v_isShared_2597_ == 0)
{
lean_ctor_set(v___x_2596_, 0, v___x_2606_);
v___x_2608_ = v___x_2596_;
goto v_reusejp_2607_;
}
else
{
lean_object* v_reuseFailAlloc_2609_; 
v_reuseFailAlloc_2609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2609_, 0, v___x_2606_);
v___x_2608_ = v_reuseFailAlloc_2609_;
goto v_reusejp_2607_;
}
v_reusejp_2607_:
{
return v___x_2608_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___boxed(lean_object* v_a_2611_, lean_object* v___x_2612_, lean_object* v___f_2613_, lean_object* v_natCast_2614_, lean_object* v_add_2615_, lean_object* v_one_2616_, lean_object* v_fo_2617_, lean_object* v_betaLogup_2618_, lean_object* v___f_2619_, lean_object* v_zero_2620_, lean_object* v_interaction_2621_){
_start:
{
lean_object* v_res_2622_; 
v_res_2622_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3(v_a_2611_, v___x_2612_, v___f_2613_, v_natCast_2614_, v_add_2615_, v_one_2616_, v_fo_2617_, v_betaLogup_2618_, v___f_2619_, v_zero_2620_, v_interaction_2621_);
lean_dec(v_a_2611_);
return v_res_2622_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__4(lean_object* v_mul_2623_, lean_object* v_add_2624_, lean_object* v_state_2625_, lean_object* v_entry_2626_){
_start:
{
lean_object* v_snd_2627_; lean_object* v_fst_2628_; lean_object* v_snd_2629_; lean_object* v_fst_2630_; lean_object* v_fst_2631_; lean_object* v_snd_2632_; lean_object* v___x_2634_; uint8_t v_isShared_2635_; uint8_t v_isSharedCheck_2643_; 
v_snd_2627_ = lean_ctor_get(v_entry_2626_, 1);
lean_inc(v_snd_2627_);
v_fst_2628_ = lean_ctor_get(v_state_2625_, 0);
lean_inc(v_fst_2628_);
v_snd_2629_ = lean_ctor_get(v_state_2625_, 1);
lean_inc(v_snd_2629_);
lean_dec_ref(v_state_2625_);
v_fst_2630_ = lean_ctor_get(v_entry_2626_, 0);
lean_inc(v_fst_2630_);
lean_dec_ref(v_entry_2626_);
v_fst_2631_ = lean_ctor_get(v_snd_2627_, 0);
v_snd_2632_ = lean_ctor_get(v_snd_2627_, 1);
v_isSharedCheck_2643_ = !lean_is_exclusive(v_snd_2627_);
if (v_isSharedCheck_2643_ == 0)
{
v___x_2634_ = v_snd_2627_;
v_isShared_2635_ = v_isSharedCheck_2643_;
goto v_resetjp_2633_;
}
else
{
lean_inc(v_snd_2632_);
lean_inc(v_fst_2631_);
lean_dec(v_snd_2627_);
v___x_2634_ = lean_box(0);
v_isShared_2635_ = v_isSharedCheck_2643_;
goto v_resetjp_2633_;
}
v_resetjp_2633_:
{
lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2641_; 
lean_inc(v_mul_2623_);
lean_inc(v_fst_2630_);
v___x_2636_ = lean_apply_2(v_mul_2623_, v_fst_2630_, v_fst_2631_);
lean_inc(v_add_2624_);
v___x_2637_ = lean_apply_2(v_add_2624_, v_fst_2628_, v___x_2636_);
v___x_2638_ = lean_apply_2(v_mul_2623_, v_fst_2630_, v_snd_2632_);
v___x_2639_ = lean_apply_2(v_add_2624_, v_snd_2629_, v___x_2638_);
if (v_isShared_2635_ == 0)
{
lean_ctor_set(v___x_2634_, 1, v___x_2639_);
lean_ctor_set(v___x_2634_, 0, v___x_2637_);
v___x_2641_ = v___x_2634_;
goto v_reusejp_2640_;
}
else
{
lean_object* v_reuseFailAlloc_2642_; 
v_reuseFailAlloc_2642_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2642_, 0, v___x_2637_);
lean_ctor_set(v_reuseFailAlloc_2642_, 1, v___x_2639_);
v___x_2641_ = v_reuseFailAlloc_2642_;
goto v_reusejp_2640_;
}
v_reusejp_2640_:
{
return v___x_2641_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5(lean_object* v_fo_2647_, uint8_t v_needRot_2648_, lean_object* v_acc_2649_, lean_object* v_opening_2650_){
_start:
{
lean_object* v___x_2651_; 
v___x_2651_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_2647_, v_opening_2650_, v_needRot_2648_);
if (lean_obj_tag(v___x_2651_) == 0)
{
lean_object* v___x_2652_; 
lean_dec_ref_known(v___x_2651_, 1);
lean_dec(v_acc_2649_);
v___x_2652_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0));
return v___x_2652_;
}
else
{
lean_object* v_a_2653_; lean_object* v___x_2655_; uint8_t v_isShared_2656_; uint8_t v_isSharedCheck_2663_; 
v_a_2653_ = lean_ctor_get(v___x_2651_, 0);
v_isSharedCheck_2663_ = !lean_is_exclusive(v___x_2651_);
if (v_isSharedCheck_2663_ == 0)
{
v___x_2655_ = v___x_2651_;
v_isShared_2656_ = v_isSharedCheck_2663_;
goto v_resetjp_2654_;
}
else
{
lean_inc(v_a_2653_);
lean_dec(v___x_2651_);
v___x_2655_ = lean_box(0);
v_isShared_2656_ = v_isSharedCheck_2663_;
goto v_resetjp_2654_;
}
v_resetjp_2654_:
{
lean_object* v___x_2657_; lean_object* v___x_2658_; lean_object* v___x_2659_; lean_object* v___x_2661_; 
v___x_2657_ = lean_box(0);
v___x_2658_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2658_, 0, v_a_2653_);
lean_ctor_set(v___x_2658_, 1, v___x_2657_);
v___x_2659_ = l_List_appendTR___redArg(v_acc_2649_, v___x_2658_);
if (v_isShared_2656_ == 0)
{
lean_ctor_set(v___x_2655_, 0, v___x_2659_);
v___x_2661_ = v___x_2655_;
goto v_reusejp_2660_;
}
else
{
lean_object* v_reuseFailAlloc_2662_; 
v_reuseFailAlloc_2662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2662_, 0, v___x_2659_);
v___x_2661_ = v_reuseFailAlloc_2662_;
goto v_reusejp_2660_;
}
v_reusejp_2660_:
{
return v___x_2661_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___boxed(lean_object* v_fo_2664_, lean_object* v_needRot_2665_, lean_object* v_acc_2666_, lean_object* v_opening_2667_){
_start:
{
uint8_t v_needRot_boxed_2668_; lean_object* v_res_2669_; 
v_needRot_boxed_2668_ = lean_unbox(v_needRot_2665_);
v_res_2669_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5(v_fo_2664_, v_needRot_boxed_2668_, v_acc_2666_, v_opening_2667_);
return v_res_2669_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1(void){
_start:
{
lean_object* v___x_2673_; lean_object* v___x_2674_; 
v___x_2673_ = lean_unsigned_to_nat(0u);
v___x_2674_ = lean_nat_to_int(v___x_2673_);
return v___x_2674_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(lean_object* v_fo_2675_, lean_object* v_algMap_2676_, lean_object* v_inst_2677_, lean_object* v_vk_2678_, lean_object* v_columnOpenings_2679_, lean_object* v_publicValues_2680_, lean_object* v_lSkip_2681_, lean_object* v_rs_2682_, lean_object* v_lambda_2683_, lean_object* v_betaLogup_2684_, lean_object* v_traceIdx_2685_, lean_object* v_airIdx_2686_, lean_object* v_n_2687_, lean_object* v_eq3bs_2688_, lean_object* v_eqXiR_2689_, lean_object* v_eqSharpXiR_2690_){
_start:
{
lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; 
v___x_2691_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9));
lean_inc(v_airIdx_2686_);
v___x_2692_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_2678_, v_airIdx_2686_);
v___x_2693_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v___x_2692_);
if (lean_obj_tag(v___x_2693_) == 0)
{
lean_object* v___x_2694_; 
lean_dec_ref_known(v___x_2693_, 1);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_traceIdx_2685_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2694_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2694_;
}
else
{
lean_object* v_a_2695_; lean_object* v___x_2696_; 
v_a_2695_ = lean_ctor_get(v___x_2693_, 0);
lean_inc(v_a_2695_);
lean_dec_ref_known(v___x_2693_, 1);
v___x_2696_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_columnOpenings_2679_, v_traceIdx_2685_);
if (lean_obj_tag(v___x_2696_) == 0)
{
lean_object* v___x_2697_; 
lean_dec_ref_known(v___x_2696_, 1);
lean_dec(v_a_2695_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2697_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2697_;
}
else
{
lean_object* v_a_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; 
v_a_2698_ = lean_ctor_get(v___x_2696_, 0);
lean_inc(v_a_2698_);
lean_dec_ref_known(v___x_2696_, 1);
v___x_2699_ = lean_unsigned_to_nat(0u);
v___x_2700_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2698_, v___x_2699_);
if (lean_obj_tag(v___x_2700_) == 0)
{
lean_object* v___x_2701_; 
lean_dec_ref_known(v___x_2700_, 1);
lean_dec(v_a_2698_);
lean_dec(v_a_2695_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2701_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2701_;
}
else
{
lean_object* v_a_2702_; lean_object* v_preprocessedData_2703_; lean_object* v_params_2704_; lean_object* v_symbolicConstraints_2705_; lean_object* v___y_2707_; lean_object* v___y_2708_; lean_object* v_fst_2709_; lean_object* v_fst_2710_; lean_object* v_snd_2711_; lean_object* v___y_2797_; lean_object* v___y_2798_; lean_object* v___y_2799_; lean_object* v___y_2800_; lean_object* v_natCast_2801_; lean_object* v_pow_2802_; lean_object* v___y_2803_; lean_object* v___y_2804_; lean_object* v___y_2805_; uint8_t v_needRot_2813_; lean_object* v___x_2814_; 
v_a_2702_ = lean_ctor_get(v___x_2700_, 0);
lean_inc(v_a_2702_);
lean_dec_ref_known(v___x_2700_, 1);
v_preprocessedData_2703_ = lean_ctor_get(v_a_2695_, 0);
lean_inc(v_preprocessedData_2703_);
v_params_2704_ = lean_ctor_get(v_a_2695_, 1);
lean_inc_ref(v_params_2704_);
v_symbolicConstraints_2705_ = lean_ctor_get(v_a_2695_, 2);
lean_inc_ref(v_symbolicConstraints_2705_);
lean_dec(v_a_2695_);
v_needRot_2813_ = lean_ctor_get_uint8(v_params_2704_, sizeof(void*)*2);
lean_dec_ref(v_params_2704_);
lean_inc_ref(v_fo_2675_);
v___x_2814_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_2675_, v_a_2702_, v_needRot_2813_);
if (lean_obj_tag(v___x_2814_) == 0)
{
lean_object* v___x_2815_; 
lean_dec_ref_known(v___x_2814_, 1);
lean_dec_ref(v_symbolicConstraints_2705_);
lean_dec(v_preprocessedData_2703_);
lean_dec(v_a_2698_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2815_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2815_;
}
else
{
lean_object* v_a_2816_; lean_object* v___x_2817_; lean_object* v___f_2818_; lean_object* v___y_2820_; lean_object* v___y_2821_; lean_object* v___y_2822_; 
v_a_2816_ = lean_ctor_get(v___x_2814_, 0);
lean_inc(v_a_2816_);
lean_dec_ref_known(v___x_2814_, 1);
v___x_2817_ = lean_box(v_needRot_2813_);
lean_inc_ref(v_fo_2675_);
v___f_2818_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___boxed), 4, 2);
lean_closure_set(v___f_2818_, 0, v_fo_2675_);
lean_closure_set(v___f_2818_, 1, v___x_2817_);
if (lean_obj_tag(v_preprocessedData_2703_) == 0)
{
lean_object* v___x_2858_; lean_object* v___x_2859_; 
v___x_2858_ = lean_box(0);
v___x_2859_ = lean_unsigned_to_nat(1u);
v___y_2820_ = v___x_2858_;
v___y_2821_ = v___x_2859_;
v___y_2822_ = v___x_2699_;
goto v___jp_2819_;
}
else
{
lean_object* v___x_2861_; uint8_t v_isShared_2862_; uint8_t v_isSharedCheck_2873_; 
v_isSharedCheck_2873_ = !lean_is_exclusive(v_preprocessedData_2703_);
if (v_isSharedCheck_2873_ == 0)
{
lean_object* v_unused_2874_; 
v_unused_2874_ = lean_ctor_get(v_preprocessedData_2703_, 0);
lean_dec(v_unused_2874_);
v___x_2861_ = v_preprocessedData_2703_;
v_isShared_2862_ = v_isSharedCheck_2873_;
goto v_resetjp_2860_;
}
else
{
lean_dec(v_preprocessedData_2703_);
v___x_2861_ = lean_box(0);
v_isShared_2862_ = v_isSharedCheck_2873_;
goto v_resetjp_2860_;
}
v_resetjp_2860_:
{
lean_object* v___x_2863_; lean_object* v___x_2864_; 
v___x_2863_ = lean_unsigned_to_nat(1u);
v___x_2864_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2698_, v___x_2863_);
if (lean_obj_tag(v___x_2864_) == 0)
{
lean_object* v___x_2865_; 
lean_dec_ref_known(v___x_2864_, 1);
lean_del_object(v___x_2861_);
lean_dec_ref(v___f_2818_);
lean_dec(v_a_2816_);
lean_dec_ref(v_symbolicConstraints_2705_);
lean_dec(v_a_2698_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2865_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2865_;
}
else
{
lean_object* v_a_2866_; lean_object* v___x_2867_; 
v_a_2866_ = lean_ctor_get(v___x_2864_, 0);
lean_inc(v_a_2866_);
lean_dec_ref_known(v___x_2864_, 1);
lean_inc_ref(v_fo_2675_);
v___x_2867_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_2675_, v_a_2866_, v_needRot_2813_);
if (lean_obj_tag(v___x_2867_) == 0)
{
lean_object* v___x_2868_; 
lean_dec_ref_known(v___x_2867_, 1);
lean_del_object(v___x_2861_);
lean_dec_ref(v___f_2818_);
lean_dec(v_a_2816_);
lean_dec_ref(v_symbolicConstraints_2705_);
lean_dec(v_a_2698_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2868_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2868_;
}
else
{
lean_object* v_a_2869_; lean_object* v___x_2871_; 
v_a_2869_ = lean_ctor_get(v___x_2867_, 0);
lean_inc(v_a_2869_);
lean_dec_ref_known(v___x_2867_, 1);
if (v_isShared_2862_ == 0)
{
lean_ctor_set(v___x_2861_, 0, v_a_2869_);
v___x_2871_ = v___x_2861_;
goto v_reusejp_2870_;
}
else
{
lean_object* v_reuseFailAlloc_2872_; 
v_reuseFailAlloc_2872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2872_, 0, v_a_2869_);
v___x_2871_ = v_reuseFailAlloc_2872_;
goto v_reusejp_2870_;
}
v_reusejp_2870_:
{
v___y_2820_ = v___x_2871_;
v___y_2821_ = v___x_2863_;
v___y_2822_ = v___x_2863_;
goto v___jp_2819_;
}
}
}
}
}
v___jp_2819_:
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2826_; 
v___x_2823_ = lean_nat_add(v___y_2821_, v___y_2822_);
lean_dec(v___y_2822_);
v___x_2824_ = l_List_drop___redArg(v___x_2823_, v_a_2698_);
lean_dec(v_a_2698_);
v___x_2825_ = lean_box(0);
v___x_2826_ = l_List_foldlM___redArg(v___x_2691_, v___f_2818_, v___x_2825_, v___x_2824_);
if (lean_obj_tag(v___x_2826_) == 0)
{
lean_object* v_a_2827_; lean_object* v___x_2829_; uint8_t v_isShared_2830_; uint8_t v_isSharedCheck_2834_; 
lean_dec(v___y_2821_);
lean_dec(v___y_2820_);
lean_dec(v_a_2816_);
lean_dec_ref(v_symbolicConstraints_2705_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_airIdx_2686_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_rs_2682_);
lean_dec(v_lSkip_2681_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v_a_2827_ = lean_ctor_get(v___x_2826_, 0);
v_isSharedCheck_2834_ = !lean_is_exclusive(v___x_2826_);
if (v_isSharedCheck_2834_ == 0)
{
v___x_2829_ = v___x_2826_;
v_isShared_2830_ = v_isSharedCheck_2834_;
goto v_resetjp_2828_;
}
else
{
lean_inc(v_a_2827_);
lean_dec(v___x_2826_);
v___x_2829_ = lean_box(0);
v_isShared_2830_ = v_isSharedCheck_2834_;
goto v_resetjp_2828_;
}
v_resetjp_2828_:
{
lean_object* v___x_2832_; 
if (v_isShared_2830_ == 0)
{
v___x_2832_ = v___x_2829_;
goto v_reusejp_2831_;
}
else
{
lean_object* v_reuseFailAlloc_2833_; 
v_reuseFailAlloc_2833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2833_, 0, v_a_2827_);
v___x_2832_ = v_reuseFailAlloc_2833_;
goto v_reusejp_2831_;
}
v_reusejp_2831_:
{
return v___x_2832_;
}
}
}
else
{
lean_object* v_a_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; lean_object* v___x_2838_; uint8_t v___x_2839_; 
v_a_2835_ = lean_ctor_get(v___x_2826_, 0);
lean_inc(v_a_2835_);
lean_dec_ref_known(v___x_2826_, 1);
v___x_2836_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2836_, 0, v_a_2816_);
lean_ctor_set(v___x_2836_, 1, v___x_2825_);
v___x_2837_ = l_List_appendTR___redArg(v_a_2835_, v___x_2836_);
v___x_2838_ = lean_obj_once(&lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1, &lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1_once, _init_lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1);
v___x_2839_ = lean_int_dec_lt(v_n_2687_, v___x_2838_);
if (v___x_2839_ == 0)
{
lean_object* v_toRingOps_2840_; lean_object* v_toSemiringOps_2841_; lean_object* v_one_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; 
v_toRingOps_2840_ = lean_ctor_get(v_fo_2675_, 0);
v_toSemiringOps_2841_ = lean_ctor_get(v_toRingOps_2840_, 0);
v_one_2842_ = lean_ctor_get(v_toSemiringOps_2841_, 1);
v___x_2843_ = l_Int_toNat(v_n_2687_);
v___x_2844_ = lean_nat_add(v___x_2843_, v___y_2821_);
lean_dec(v___y_2821_);
lean_dec(v___x_2843_);
v___x_2845_ = ((lean_object*)(lp_swirl_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0));
lean_inc(v_rs_2682_);
v___x_2846_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_rs_2682_, v_rs_2682_, v___x_2844_, v___x_2845_);
lean_dec(v_rs_2682_);
lean_inc(v_one_2842_);
v___y_2707_ = v___y_2820_;
v___y_2708_ = v___x_2837_;
v_fst_2709_ = v_lSkip_2681_;
v_fst_2710_ = v___x_2846_;
v_snd_2711_ = v_one_2842_;
goto v___jp_2706_;
}
else
{
lean_object* v_toRingOps_2847_; lean_object* v_inv_2848_; lean_object* v_toSemiringOps_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; 
lean_dec(v___y_2821_);
v_toRingOps_2847_ = lean_ctor_get(v_fo_2675_, 0);
v_inv_2848_ = lean_ctor_get(v_fo_2675_, 1);
v_toSemiringOps_2849_ = lean_ctor_get(v_toRingOps_2847_, 0);
v___x_2850_ = lean_nat_abs(v_n_2687_);
v___x_2851_ = lean_nat_sub(v_lSkip_2681_, v___x_2850_);
lean_dec(v_lSkip_2681_);
if (lean_obj_tag(v_rs_2682_) == 0)
{
lean_object* v_zero_2852_; lean_object* v_natCast_2853_; lean_object* v_pow_2854_; 
v_zero_2852_ = lean_ctor_get(v_toSemiringOps_2849_, 0);
v_natCast_2853_ = lean_ctor_get(v_toSemiringOps_2849_, 2);
v_pow_2854_ = lean_ctor_get(v_toSemiringOps_2849_, 5);
lean_inc(v_zero_2852_);
lean_inc(v_pow_2854_);
lean_inc(v_natCast_2853_);
lean_inc_ref(v_toSemiringOps_2849_);
lean_inc(v_inv_2848_);
v___y_2797_ = v___y_2820_;
v___y_2798_ = v_inv_2848_;
v___y_2799_ = v___x_2850_;
v___y_2800_ = v_toSemiringOps_2849_;
v_natCast_2801_ = v_natCast_2853_;
v_pow_2802_ = v_pow_2854_;
v___y_2803_ = v___x_2837_;
v___y_2804_ = v___x_2851_;
v___y_2805_ = v_zero_2852_;
goto v___jp_2796_;
}
else
{
lean_object* v_head_2855_; lean_object* v_natCast_2856_; lean_object* v_pow_2857_; 
v_head_2855_ = lean_ctor_get(v_rs_2682_, 0);
lean_inc(v_head_2855_);
lean_dec_ref_known(v_rs_2682_, 2);
v_natCast_2856_ = lean_ctor_get(v_toSemiringOps_2849_, 2);
v_pow_2857_ = lean_ctor_get(v_toSemiringOps_2849_, 5);
lean_inc(v_pow_2857_);
lean_inc(v_natCast_2856_);
lean_inc_ref(v_toSemiringOps_2849_);
lean_inc(v_inv_2848_);
v___y_2797_ = v___y_2820_;
v___y_2798_ = v_inv_2848_;
v___y_2799_ = v___x_2850_;
v___y_2800_ = v_toSemiringOps_2849_;
v_natCast_2801_ = v_natCast_2856_;
v_pow_2802_ = v_pow_2857_;
v___y_2803_ = v___x_2837_;
v___y_2804_ = v___x_2851_;
v___y_2805_ = v_head_2855_;
goto v___jp_2796_;
}
}
}
}
}
v___jp_2706_:
{
lean_object* v___x_2712_; 
v___x_2712_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_publicValues_2680_, v_airIdx_2686_);
if (lean_obj_tag(v___x_2712_) == 0)
{
lean_object* v___x_2713_; 
lean_dec_ref_known(v___x_2712_, 1);
lean_dec(v_snd_2711_);
lean_dec(v_fst_2710_);
lean_dec(v_fst_2709_);
lean_dec(v___y_2708_);
lean_dec(v___y_2707_);
lean_dec_ref(v_symbolicConstraints_2705_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec(v_inst_2677_);
lean_dec(v_algMap_2676_);
lean_dec_ref(v_fo_2675_);
v___x_2713_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2713_;
}
else
{
lean_object* v_constraints_2714_; lean_object* v_a_2715_; lean_object* v_interactions_2716_; lean_object* v_nodes_2717_; lean_object* v_constraintIdx_2718_; lean_object* v___x_2720_; uint8_t v_isShared_2721_; uint8_t v_isSharedCheck_2795_; 
v_constraints_2714_ = lean_ctor_get(v_symbolicConstraints_2705_, 0);
lean_inc_ref(v_constraints_2714_);
v_a_2715_ = lean_ctor_get(v___x_2712_, 0);
lean_inc(v_a_2715_);
lean_dec_ref_known(v___x_2712_, 1);
v_interactions_2716_ = lean_ctor_get(v_symbolicConstraints_2705_, 1);
lean_inc(v_interactions_2716_);
lean_dec_ref(v_symbolicConstraints_2705_);
v_nodes_2717_ = lean_ctor_get(v_constraints_2714_, 0);
v_constraintIdx_2718_ = lean_ctor_get(v_constraints_2714_, 1);
v_isSharedCheck_2795_ = !lean_is_exclusive(v_constraints_2714_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2720_ = v_constraints_2714_;
v_isShared_2721_ = v_isSharedCheck_2795_;
goto v_resetjp_2719_;
}
else
{
lean_inc(v_constraintIdx_2718_);
lean_inc(v_nodes_2717_);
lean_dec(v_constraints_2714_);
v___x_2720_ = lean_box(0);
v_isShared_2721_ = v_isSharedCheck_2795_;
goto v_resetjp_2719_;
}
v_resetjp_2719_:
{
lean_object* v___x_2722_; lean_object* v___x_2723_; 
lean_inc_ref_n(v_fo_2675_, 2);
v___x_2722_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(v_fo_2675_, v_inst_2677_, v___y_2707_, v___y_2708_, v_a_2715_, v_fst_2710_, v_fst_2709_);
v___x_2723_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(v_fo_2675_, v_algMap_2676_, v___x_2722_, v_nodes_2717_);
if (lean_obj_tag(v___x_2723_) == 0)
{
lean_object* v___x_2724_; 
lean_dec_ref_known(v___x_2723_, 1);
lean_del_object(v___x_2720_);
lean_dec(v_constraintIdx_2718_);
lean_dec(v_interactions_2716_);
lean_dec(v_snd_2711_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_betaLogup_2684_);
lean_dec(v_lambda_2683_);
lean_dec_ref(v_fo_2675_);
v___x_2724_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2724_;
}
else
{
lean_object* v_toRingOps_2725_; lean_object* v_toSemiringOps_2726_; lean_object* v___x_2728_; uint8_t v_isShared_2729_; uint8_t v_isSharedCheck_2793_; 
v_toRingOps_2725_ = lean_ctor_get(v_fo_2675_, 0);
lean_inc_ref(v_toRingOps_2725_);
v_toSemiringOps_2726_ = lean_ctor_get(v_toRingOps_2725_, 0);
v_isSharedCheck_2793_ = !lean_is_exclusive(v_toRingOps_2725_);
if (v_isSharedCheck_2793_ == 0)
{
lean_object* v_unused_2794_; 
v_unused_2794_ = lean_ctor_get(v_toRingOps_2725_, 1);
lean_dec(v_unused_2794_);
v___x_2728_ = v_toRingOps_2725_;
v_isShared_2729_ = v_isSharedCheck_2793_;
goto v_resetjp_2727_;
}
else
{
lean_inc(v_toSemiringOps_2726_);
lean_dec(v_toRingOps_2725_);
v___x_2728_ = lean_box(0);
v_isShared_2729_ = v_isSharedCheck_2793_;
goto v_resetjp_2727_;
}
v_resetjp_2727_:
{
lean_object* v_a_2730_; lean_object* v_zero_2731_; lean_object* v_one_2732_; lean_object* v_natCast_2733_; lean_object* v_add_2734_; lean_object* v_mul_2735_; lean_object* v___f_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; 
v_a_2730_ = lean_ctor_get(v___x_2723_, 0);
lean_inc_n(v_a_2730_, 2);
lean_dec_ref_known(v___x_2723_, 1);
v_zero_2731_ = lean_ctor_get(v_toSemiringOps_2726_, 0);
lean_inc_n(v_zero_2731_, 2);
v_one_2732_ = lean_ctor_get(v_toSemiringOps_2726_, 1);
lean_inc(v_one_2732_);
v_natCast_2733_ = lean_ctor_get(v_toSemiringOps_2726_, 2);
lean_inc(v_natCast_2733_);
v_add_2734_ = lean_ctor_get(v_toSemiringOps_2726_, 3);
lean_inc_n(v_add_2734_, 2);
v_mul_2735_ = lean_ctor_get(v_toSemiringOps_2726_, 4);
lean_inc_n(v_mul_2735_, 2);
lean_dec_ref(v_toSemiringOps_2726_);
v___f_2736_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_2736_, 0, v_a_2730_);
lean_closure_set(v___f_2736_, 1, v_mul_2735_);
lean_closure_set(v___f_2736_, 2, v_add_2734_);
v___x_2737_ = l_List_lengthTR___redArg(v_constraintIdx_2718_);
lean_inc_ref(v_fo_2675_);
v___x_2738_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_2675_, v_lambda_2683_, v___x_2737_);
v___x_2739_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_2738_, v_constraintIdx_2718_);
v___x_2740_ = l_List_foldlM___redArg(v___x_2691_, v___f_2736_, v_zero_2731_, v___x_2739_);
if (lean_obj_tag(v___x_2740_) == 0)
{
lean_object* v_a_2741_; lean_object* v___x_2743_; uint8_t v_isShared_2744_; uint8_t v_isSharedCheck_2748_; 
lean_dec(v_mul_2735_);
lean_dec(v_add_2734_);
lean_dec(v_natCast_2733_);
lean_dec(v_one_2732_);
lean_dec(v_zero_2731_);
lean_dec(v_a_2730_);
lean_del_object(v___x_2728_);
lean_del_object(v___x_2720_);
lean_dec(v_interactions_2716_);
lean_dec(v_snd_2711_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
lean_dec(v_betaLogup_2684_);
lean_dec_ref(v_fo_2675_);
v_a_2741_ = lean_ctor_get(v___x_2740_, 0);
v_isSharedCheck_2748_ = !lean_is_exclusive(v___x_2740_);
if (v_isSharedCheck_2748_ == 0)
{
v___x_2743_ = v___x_2740_;
v_isShared_2744_ = v_isSharedCheck_2748_;
goto v_resetjp_2742_;
}
else
{
lean_inc(v_a_2741_);
lean_dec(v___x_2740_);
v___x_2743_ = lean_box(0);
v_isShared_2744_ = v_isSharedCheck_2748_;
goto v_resetjp_2742_;
}
v_resetjp_2742_:
{
lean_object* v___x_2746_; 
if (v_isShared_2744_ == 0)
{
v___x_2746_ = v___x_2743_;
goto v_reusejp_2745_;
}
else
{
lean_object* v_reuseFailAlloc_2747_; 
v_reuseFailAlloc_2747_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2747_, 0, v_a_2741_);
v___x_2746_ = v_reuseFailAlloc_2747_;
goto v_reusejp_2745_;
}
v_reusejp_2745_:
{
return v___x_2746_;
}
}
}
else
{
lean_object* v_a_2749_; lean_object* v___f_2750_; lean_object* v___f_2751_; lean_object* v___f_2752_; lean_object* v___x_2753_; lean_object* v___x_2754_; 
v_a_2749_ = lean_ctor_get(v___x_2740_, 0);
lean_inc(v_a_2749_);
lean_dec_ref_known(v___x_2740_, 1);
lean_inc(v_a_2730_);
v___f_2750_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_2750_, 0, v_a_2730_);
lean_inc_n(v_add_2734_, 2);
lean_inc(v_mul_2735_);
v___f_2751_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2), 4, 2);
lean_closure_set(v___f_2751_, 0, v_mul_2735_);
lean_closure_set(v___f_2751_, 1, v_add_2734_);
lean_inc(v_zero_2731_);
v___f_2752_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___boxed), 11, 10);
lean_closure_set(v___f_2752_, 0, v_a_2730_);
lean_closure_set(v___f_2752_, 1, v___x_2691_);
lean_closure_set(v___f_2752_, 2, v___f_2750_);
lean_closure_set(v___f_2752_, 3, v_natCast_2733_);
lean_closure_set(v___f_2752_, 4, v_add_2734_);
lean_closure_set(v___f_2752_, 5, v_one_2732_);
lean_closure_set(v___f_2752_, 6, v_fo_2675_);
lean_closure_set(v___f_2752_, 7, v_betaLogup_2684_);
lean_closure_set(v___f_2752_, 8, v___f_2751_);
lean_closure_set(v___f_2752_, 9, v_zero_2731_);
v___x_2753_ = lean_box(0);
v___x_2754_ = l_List_mapM_loop___redArg(v___x_2691_, v___f_2752_, v_interactions_2716_, v___x_2753_);
if (lean_obj_tag(v___x_2754_) == 0)
{
lean_object* v_a_2755_; lean_object* v___x_2757_; uint8_t v_isShared_2758_; uint8_t v_isSharedCheck_2762_; 
lean_dec(v_a_2749_);
lean_dec(v_mul_2735_);
lean_dec(v_add_2734_);
lean_dec(v_zero_2731_);
lean_del_object(v___x_2728_);
lean_del_object(v___x_2720_);
lean_dec(v_snd_2711_);
lean_dec(v_eqSharpXiR_2690_);
lean_dec(v_eqXiR_2689_);
lean_dec(v_eq3bs_2688_);
v_a_2755_ = lean_ctor_get(v___x_2754_, 0);
v_isSharedCheck_2762_ = !lean_is_exclusive(v___x_2754_);
if (v_isSharedCheck_2762_ == 0)
{
v___x_2757_ = v___x_2754_;
v_isShared_2758_ = v_isSharedCheck_2762_;
goto v_resetjp_2756_;
}
else
{
lean_inc(v_a_2755_);
lean_dec(v___x_2754_);
v___x_2757_ = lean_box(0);
v_isShared_2758_ = v_isSharedCheck_2762_;
goto v_resetjp_2756_;
}
v_resetjp_2756_:
{
lean_object* v___x_2760_; 
if (v_isShared_2758_ == 0)
{
v___x_2760_ = v___x_2757_;
goto v_reusejp_2759_;
}
else
{
lean_object* v_reuseFailAlloc_2761_; 
v_reuseFailAlloc_2761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2761_, 0, v_a_2755_);
v___x_2760_ = v_reuseFailAlloc_2761_;
goto v_reusejp_2759_;
}
v_reusejp_2759_:
{
return v___x_2760_;
}
}
}
else
{
lean_object* v_a_2763_; lean_object* v___x_2765_; uint8_t v_isShared_2766_; uint8_t v_isSharedCheck_2792_; 
v_a_2763_ = lean_ctor_get(v___x_2754_, 0);
v_isSharedCheck_2792_ = !lean_is_exclusive(v___x_2754_);
if (v_isSharedCheck_2792_ == 0)
{
v___x_2765_ = v___x_2754_;
v_isShared_2766_ = v_isSharedCheck_2792_;
goto v_resetjp_2764_;
}
else
{
lean_inc(v_a_2763_);
lean_dec(v___x_2754_);
v___x_2765_ = lean_box(0);
v_isShared_2766_ = v_isSharedCheck_2792_;
goto v_resetjp_2764_;
}
v_resetjp_2764_:
{
lean_object* v___f_2767_; lean_object* v___x_2769_; 
lean_inc(v_mul_2735_);
v___f_2767_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2767_, 0, v_mul_2735_);
lean_closure_set(v___f_2767_, 1, v_add_2734_);
lean_inc(v_zero_2731_);
if (v_isShared_2729_ == 0)
{
lean_ctor_set(v___x_2728_, 1, v_zero_2731_);
lean_ctor_set(v___x_2728_, 0, v_zero_2731_);
v___x_2769_ = v___x_2728_;
goto v_reusejp_2768_;
}
else
{
lean_object* v_reuseFailAlloc_2791_; 
v_reuseFailAlloc_2791_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2791_, 0, v_zero_2731_);
lean_ctor_set(v_reuseFailAlloc_2791_, 1, v_zero_2731_);
v___x_2769_ = v_reuseFailAlloc_2791_;
goto v_reusejp_2768_;
}
v_reusejp_2768_:
{
lean_object* v___x_2770_; lean_object* v___x_2771_; lean_object* v_fst_2772_; lean_object* v_snd_2773_; lean_object* v___x_2775_; uint8_t v_isShared_2776_; uint8_t v_isSharedCheck_2790_; 
v___x_2770_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_eq3bs_2688_, v_a_2763_);
v___x_2771_ = l_List_foldl___redArg(v___f_2767_, v___x_2769_, v___x_2770_);
v_fst_2772_ = lean_ctor_get(v___x_2771_, 0);
v_snd_2773_ = lean_ctor_get(v___x_2771_, 1);
v_isSharedCheck_2790_ = !lean_is_exclusive(v___x_2771_);
if (v_isSharedCheck_2790_ == 0)
{
v___x_2775_ = v___x_2771_;
v_isShared_2776_ = v_isSharedCheck_2790_;
goto v_resetjp_2774_;
}
else
{
lean_inc(v_snd_2773_);
lean_inc(v_fst_2772_);
lean_dec(v___x_2771_);
v___x_2775_ = lean_box(0);
v_isShared_2776_ = v_isSharedCheck_2790_;
goto v_resetjp_2774_;
}
v_resetjp_2774_:
{
lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2782_; 
lean_inc_n(v_mul_2735_, 3);
v___x_2777_ = lean_apply_2(v_mul_2735_, v_eqXiR_2689_, v_a_2749_);
v___x_2778_ = lean_apply_2(v_mul_2735_, v_fst_2772_, v_snd_2711_);
lean_inc(v_eqSharpXiR_2690_);
v___x_2779_ = lean_apply_2(v_mul_2735_, v___x_2778_, v_eqSharpXiR_2690_);
v___x_2780_ = lean_apply_2(v_mul_2735_, v_snd_2773_, v_eqSharpXiR_2690_);
if (v_isShared_2776_ == 0)
{
lean_ctor_set(v___x_2775_, 1, v___x_2777_);
lean_ctor_set(v___x_2775_, 0, v___x_2780_);
v___x_2782_ = v___x_2775_;
goto v_reusejp_2781_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v___x_2780_);
lean_ctor_set(v_reuseFailAlloc_2789_, 1, v___x_2777_);
v___x_2782_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2781_;
}
v_reusejp_2781_:
{
lean_object* v___x_2784_; 
if (v_isShared_2721_ == 0)
{
lean_ctor_set(v___x_2720_, 1, v___x_2782_);
lean_ctor_set(v___x_2720_, 0, v___x_2779_);
v___x_2784_ = v___x_2720_;
goto v_reusejp_2783_;
}
else
{
lean_object* v_reuseFailAlloc_2788_; 
v_reuseFailAlloc_2788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2788_, 0, v___x_2779_);
lean_ctor_set(v_reuseFailAlloc_2788_, 1, v___x_2782_);
v___x_2784_ = v_reuseFailAlloc_2788_;
goto v_reusejp_2783_;
}
v_reusejp_2783_:
{
lean_object* v___x_2786_; 
if (v_isShared_2766_ == 0)
{
lean_ctor_set(v___x_2765_, 0, v___x_2784_);
v___x_2786_ = v___x_2765_;
goto v_reusejp_2785_;
}
else
{
lean_object* v_reuseFailAlloc_2787_; 
v_reuseFailAlloc_2787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2787_, 0, v___x_2784_);
v___x_2786_ = v_reuseFailAlloc_2787_;
goto v_reusejp_2785_;
}
v_reusejp_2785_:
{
return v___x_2786_;
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
v___jp_2796_:
{
lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; 
v___x_2806_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v___y_2800_, v___y_2805_, v___y_2799_);
lean_dec(v___y_2805_);
v___x_2807_ = lean_box(0);
v___x_2808_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2808_, 0, v___x_2806_);
lean_ctor_set(v___x_2808_, 1, v___x_2807_);
v___x_2809_ = lean_unsigned_to_nat(2u);
v___x_2810_ = lean_apply_1(v_natCast_2801_, v___x_2809_);
v___x_2811_ = lean_apply_2(v_pow_2802_, v___x_2810_, v___y_2799_);
v___x_2812_ = lean_apply_1(v___y_2798_, v___x_2811_);
v___y_2707_ = v___y_2797_;
v___y_2708_ = v___y_2803_;
v_fst_2709_ = v___y_2804_;
v_fst_2710_ = v___x_2808_;
v_snd_2711_ = v___x_2812_;
goto v___jp_2706_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___boxed(lean_object* v_fo_2875_, lean_object* v_algMap_2876_, lean_object* v_inst_2877_, lean_object* v_vk_2878_, lean_object* v_columnOpenings_2879_, lean_object* v_publicValues_2880_, lean_object* v_lSkip_2881_, lean_object* v_rs_2882_, lean_object* v_lambda_2883_, lean_object* v_betaLogup_2884_, lean_object* v_traceIdx_2885_, lean_object* v_airIdx_2886_, lean_object* v_n_2887_, lean_object* v_eq3bs_2888_, lean_object* v_eqXiR_2889_, lean_object* v_eqSharpXiR_2890_){
_start:
{
lean_object* v_res_2891_; 
v_res_2891_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(v_fo_2875_, v_algMap_2876_, v_inst_2877_, v_vk_2878_, v_columnOpenings_2879_, v_publicValues_2880_, v_lSkip_2881_, v_rs_2882_, v_lambda_2883_, v_betaLogup_2884_, v_traceIdx_2885_, v_airIdx_2886_, v_n_2887_, v_eq3bs_2888_, v_eqXiR_2889_, v_eqSharpXiR_2890_);
lean_dec(v_n_2887_);
lean_dec(v_publicValues_2880_);
lean_dec(v_columnOpenings_2879_);
lean_dec_ref(v_vk_2878_);
return v_res_2891_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace(lean_object* v_F_2892_, lean_object* v_EF_2893_, lean_object* v_Digest_2894_, lean_object* v_fo_2895_, lean_object* v_algMap_2896_, lean_object* v_inst_2897_, lean_object* v_vk_2898_, lean_object* v_columnOpenings_2899_, lean_object* v_publicValues_2900_, lean_object* v_lSkip_2901_, lean_object* v_rs_2902_, lean_object* v_lambda_2903_, lean_object* v_betaLogup_2904_, lean_object* v_traceIdx_2905_, lean_object* v_airIdx_2906_, lean_object* v_n_2907_, lean_object* v_eq3bs_2908_, lean_object* v_eqXiR_2909_, lean_object* v_eqSharpXiR_2910_){
_start:
{
lean_object* v___x_2911_; 
v___x_2911_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(v_fo_2895_, v_algMap_2896_, v_inst_2897_, v_vk_2898_, v_columnOpenings_2899_, v_publicValues_2900_, v_lSkip_2901_, v_rs_2902_, v_lambda_2903_, v_betaLogup_2904_, v_traceIdx_2905_, v_airIdx_2906_, v_n_2907_, v_eq3bs_2908_, v_eqXiR_2909_, v_eqSharpXiR_2910_);
return v___x_2911_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___boxed(lean_object** _args){
lean_object* v_F_2912_ = _args[0];
lean_object* v_EF_2913_ = _args[1];
lean_object* v_Digest_2914_ = _args[2];
lean_object* v_fo_2915_ = _args[3];
lean_object* v_algMap_2916_ = _args[4];
lean_object* v_inst_2917_ = _args[5];
lean_object* v_vk_2918_ = _args[6];
lean_object* v_columnOpenings_2919_ = _args[7];
lean_object* v_publicValues_2920_ = _args[8];
lean_object* v_lSkip_2921_ = _args[9];
lean_object* v_rs_2922_ = _args[10];
lean_object* v_lambda_2923_ = _args[11];
lean_object* v_betaLogup_2924_ = _args[12];
lean_object* v_traceIdx_2925_ = _args[13];
lean_object* v_airIdx_2926_ = _args[14];
lean_object* v_n_2927_ = _args[15];
lean_object* v_eq3bs_2928_ = _args[16];
lean_object* v_eqXiR_2929_ = _args[17];
lean_object* v_eqSharpXiR_2930_ = _args[18];
_start:
{
lean_object* v_res_2931_; 
v_res_2931_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace(v_F_2912_, v_EF_2913_, v_Digest_2914_, v_fo_2915_, v_algMap_2916_, v_inst_2917_, v_vk_2918_, v_columnOpenings_2919_, v_publicValues_2920_, v_lSkip_2921_, v_rs_2922_, v_lambda_2923_, v_betaLogup_2924_, v_traceIdx_2925_, v_airIdx_2926_, v_n_2927_, v_eq3bs_2928_, v_eqXiR_2929_, v_eqSharpXiR_2930_);
lean_dec(v_n_2927_);
lean_dec(v_publicValues_2920_);
lean_dec(v_columnOpenings_2919_);
lean_dec_ref(v_vk_2918_);
return v_res_2931_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0(lean_object* v_fo_2932_, lean_object* v_00___2933_){
_start:
{
lean_object* v_toRingOps_2934_; lean_object* v_toSemiringOps_2935_; lean_object* v_zero_2936_; 
v_toRingOps_2934_ = lean_ctor_get(v_fo_2932_, 0);
v_toSemiringOps_2935_ = lean_ctor_get(v_toRingOps_2934_, 0);
v_zero_2936_ = lean_ctor_get(v_toSemiringOps_2935_, 0);
lean_inc(v_zero_2936_);
return v_zero_2936_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0___boxed(lean_object* v_fo_2937_, lean_object* v_00___2938_){
_start:
{
lean_object* v_res_2939_; 
v_res_2939_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0(v_fo_2937_, v_00___2938_);
lean_dec_ref(v_fo_2937_);
return v_res_2939_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1(lean_object* v_nPerTrace_2943_, lean_object* v_fo_2944_, lean_object* v_algMap_2945_, lean_object* v_inst_2946_, lean_object* v_vk_2947_, lean_object* v_columnOpenings_2948_, lean_object* v_publicValues_2949_, lean_object* v_lSkip_2950_, lean_object* v_rs_2951_, lean_object* v_lambda_2952_, lean_object* v_betaLogup_2953_, lean_object* v_eqSharpNs_2954_, lean_object* v___f_2955_, lean_object* v_eqNs_2956_, lean_object* v_eq3bPerTrace_2957_, lean_object* v_state_2958_, lean_object* v_entry_2959_){
_start:
{
lean_object* v_fst_2960_; lean_object* v_snd_2961_; lean_object* v___x_2963_; uint8_t v_isShared_2964_; uint8_t v_isSharedCheck_3037_; 
v_fst_2960_ = lean_ctor_get(v_state_2958_, 0);
v_snd_2961_ = lean_ctor_get(v_state_2958_, 1);
v_isSharedCheck_3037_ = !lean_is_exclusive(v_state_2958_);
if (v_isSharedCheck_3037_ == 0)
{
v___x_2963_ = v_state_2958_;
v_isShared_2964_ = v_isSharedCheck_3037_;
goto v_resetjp_2962_;
}
else
{
lean_inc(v_snd_2961_);
lean_inc(v_fst_2960_);
lean_dec(v_state_2958_);
v___x_2963_ = lean_box(0);
v_isShared_2964_ = v_isSharedCheck_3037_;
goto v_resetjp_2962_;
}
v_resetjp_2962_:
{
lean_object* v_fst_2965_; lean_object* v_snd_2966_; lean_object* v___x_2968_; uint8_t v_isShared_2969_; uint8_t v_isSharedCheck_3036_; 
v_fst_2965_ = lean_ctor_get(v_entry_2959_, 0);
v_snd_2966_ = lean_ctor_get(v_entry_2959_, 1);
v_isSharedCheck_3036_ = !lean_is_exclusive(v_entry_2959_);
if (v_isSharedCheck_3036_ == 0)
{
v___x_2968_ = v_entry_2959_;
v_isShared_2969_ = v_isSharedCheck_3036_;
goto v_resetjp_2967_;
}
else
{
lean_inc(v_snd_2966_);
lean_inc(v_fst_2965_);
lean_dec(v_entry_2959_);
v___x_2968_ = lean_box(0);
v_isShared_2969_ = v_isSharedCheck_3036_;
goto v_resetjp_2967_;
}
v_resetjp_2967_:
{
lean_object* v___x_2970_; 
lean_inc(v_fst_2965_);
v___x_2970_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_nPerTrace_2943_, v_fst_2965_);
if (lean_obj_tag(v___x_2970_) == 0)
{
lean_object* v___x_2971_; 
lean_dec_ref_known(v___x_2970_, 1);
lean_del_object(v___x_2968_);
lean_dec(v_snd_2966_);
lean_dec(v_fst_2965_);
lean_del_object(v___x_2963_);
lean_dec(v_snd_2961_);
lean_dec(v_fst_2960_);
lean_dec(v___f_2955_);
lean_dec(v_betaLogup_2953_);
lean_dec(v_lambda_2952_);
lean_dec(v_rs_2951_);
lean_dec(v_lSkip_2950_);
lean_dec(v_inst_2946_);
lean_dec(v_algMap_2945_);
lean_dec_ref(v_fo_2944_);
v___x_2971_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0));
return v___x_2971_;
}
else
{
lean_object* v_a_2972_; lean_object* v___y_2974_; lean_object* v___y_2975_; lean_object* v___y_2976_; lean_object* v___y_3019_; lean_object* v___y_3020_; lean_object* v___y_3021_; lean_object* v___y_3027_; lean_object* v___x_3033_; 
v_a_2972_ = lean_ctor_get(v___x_2970_, 0);
lean_inc(v_a_2972_);
lean_dec_ref_known(v___x_2970_, 1);
lean_inc(v_fst_2965_);
v___x_3033_ = l_List_get_x3fInternal___redArg(v_eq3bPerTrace_2957_, v_fst_2965_);
if (lean_obj_tag(v___x_3033_) == 0)
{
lean_object* v___x_3034_; 
v___x_3034_ = lean_box(0);
v___y_3027_ = v___x_3034_;
goto v___jp_3026_;
}
else
{
lean_object* v_val_3035_; 
v_val_3035_ = lean_ctor_get(v___x_3033_, 0);
lean_inc(v_val_3035_);
lean_dec_ref_known(v___x_3033_, 1);
v___y_3027_ = v_val_3035_;
goto v___jp_3026_;
}
v___jp_2973_:
{
lean_object* v___x_2977_; 
v___x_2977_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(v_fo_2944_, v_algMap_2945_, v_inst_2946_, v_vk_2947_, v_columnOpenings_2948_, v_publicValues_2949_, v_lSkip_2950_, v_rs_2951_, v_lambda_2952_, v_betaLogup_2953_, v_fst_2965_, v_snd_2966_, v_a_2972_, v___y_2975_, v___y_2974_, v___y_2976_);
lean_dec(v_a_2972_);
if (lean_obj_tag(v___x_2977_) == 0)
{
lean_object* v_a_2978_; lean_object* v___x_2980_; uint8_t v_isShared_2981_; uint8_t v_isSharedCheck_2985_; 
lean_del_object(v___x_2968_);
lean_del_object(v___x_2963_);
lean_dec(v_snd_2961_);
lean_dec(v_fst_2960_);
v_a_2978_ = lean_ctor_get(v___x_2977_, 0);
v_isSharedCheck_2985_ = !lean_is_exclusive(v___x_2977_);
if (v_isSharedCheck_2985_ == 0)
{
v___x_2980_ = v___x_2977_;
v_isShared_2981_ = v_isSharedCheck_2985_;
goto v_resetjp_2979_;
}
else
{
lean_inc(v_a_2978_);
lean_dec(v___x_2977_);
v___x_2980_ = lean_box(0);
v_isShared_2981_ = v_isSharedCheck_2985_;
goto v_resetjp_2979_;
}
v_resetjp_2979_:
{
lean_object* v___x_2983_; 
if (v_isShared_2981_ == 0)
{
v___x_2983_ = v___x_2980_;
goto v_reusejp_2982_;
}
else
{
lean_object* v_reuseFailAlloc_2984_; 
v_reuseFailAlloc_2984_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2984_, 0, v_a_2978_);
v___x_2983_ = v_reuseFailAlloc_2984_;
goto v_reusejp_2982_;
}
v_reusejp_2982_:
{
return v___x_2983_;
}
}
}
else
{
lean_object* v_a_2986_; lean_object* v___x_2988_; uint8_t v_isShared_2989_; uint8_t v_isSharedCheck_3017_; 
v_a_2986_ = lean_ctor_get(v___x_2977_, 0);
v_isSharedCheck_3017_ = !lean_is_exclusive(v___x_2977_);
if (v_isSharedCheck_3017_ == 0)
{
v___x_2988_ = v___x_2977_;
v_isShared_2989_ = v_isSharedCheck_3017_;
goto v_resetjp_2987_;
}
else
{
lean_inc(v_a_2986_);
lean_dec(v___x_2977_);
v___x_2988_ = lean_box(0);
v_isShared_2989_ = v_isSharedCheck_3017_;
goto v_resetjp_2987_;
}
v_resetjp_2987_:
{
lean_object* v_snd_2990_; lean_object* v_fst_2991_; lean_object* v___x_2993_; uint8_t v_isShared_2994_; uint8_t v_isSharedCheck_3016_; 
v_snd_2990_ = lean_ctor_get(v_a_2986_, 1);
v_fst_2991_ = lean_ctor_get(v_a_2986_, 0);
v_isSharedCheck_3016_ = !lean_is_exclusive(v_a_2986_);
if (v_isSharedCheck_3016_ == 0)
{
v___x_2993_ = v_a_2986_;
v_isShared_2994_ = v_isSharedCheck_3016_;
goto v_resetjp_2992_;
}
else
{
lean_inc(v_snd_2990_);
lean_inc(v_fst_2991_);
lean_dec(v_a_2986_);
v___x_2993_ = lean_box(0);
v_isShared_2994_ = v_isSharedCheck_3016_;
goto v_resetjp_2992_;
}
v_resetjp_2992_:
{
lean_object* v_fst_2995_; lean_object* v_snd_2996_; lean_object* v___x_2998_; uint8_t v_isShared_2999_; uint8_t v_isSharedCheck_3015_; 
v_fst_2995_ = lean_ctor_get(v_snd_2990_, 0);
v_snd_2996_ = lean_ctor_get(v_snd_2990_, 1);
v_isSharedCheck_3015_ = !lean_is_exclusive(v_snd_2990_);
if (v_isSharedCheck_3015_ == 0)
{
v___x_2998_ = v_snd_2990_;
v_isShared_2999_ = v_isSharedCheck_3015_;
goto v_resetjp_2997_;
}
else
{
lean_inc(v_snd_2996_);
lean_inc(v_fst_2995_);
lean_dec(v_snd_2990_);
v___x_2998_ = lean_box(0);
v_isShared_2999_ = v_isSharedCheck_3015_;
goto v_resetjp_2997_;
}
v_resetjp_2997_:
{
lean_object* v___x_3001_; 
if (v_isShared_2994_ == 0)
{
lean_ctor_set_tag(v___x_2993_, 1);
lean_ctor_set(v___x_2993_, 1, v_fst_2960_);
v___x_3001_ = v___x_2993_;
goto v_reusejp_3000_;
}
else
{
lean_object* v_reuseFailAlloc_3014_; 
v_reuseFailAlloc_3014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3014_, 0, v_fst_2991_);
lean_ctor_set(v_reuseFailAlloc_3014_, 1, v_fst_2960_);
v___x_3001_ = v_reuseFailAlloc_3014_;
goto v_reusejp_3000_;
}
v_reusejp_3000_:
{
lean_object* v___x_3003_; 
if (v_isShared_2969_ == 0)
{
lean_ctor_set_tag(v___x_2968_, 1);
lean_ctor_set(v___x_2968_, 1, v___x_3001_);
lean_ctor_set(v___x_2968_, 0, v_fst_2995_);
v___x_3003_ = v___x_2968_;
goto v_reusejp_3002_;
}
else
{
lean_object* v_reuseFailAlloc_3013_; 
v_reuseFailAlloc_3013_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3013_, 0, v_fst_2995_);
lean_ctor_set(v_reuseFailAlloc_3013_, 1, v___x_3001_);
v___x_3003_ = v_reuseFailAlloc_3013_;
goto v_reusejp_3002_;
}
v_reusejp_3002_:
{
lean_object* v___x_3005_; 
if (v_isShared_2964_ == 0)
{
lean_ctor_set_tag(v___x_2963_, 1);
lean_ctor_set(v___x_2963_, 0, v_snd_2996_);
v___x_3005_ = v___x_2963_;
goto v_reusejp_3004_;
}
else
{
lean_object* v_reuseFailAlloc_3012_; 
v_reuseFailAlloc_3012_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3012_, 0, v_snd_2996_);
lean_ctor_set(v_reuseFailAlloc_3012_, 1, v_snd_2961_);
v___x_3005_ = v_reuseFailAlloc_3012_;
goto v_reusejp_3004_;
}
v_reusejp_3004_:
{
lean_object* v___x_3007_; 
if (v_isShared_2999_ == 0)
{
lean_ctor_set(v___x_2998_, 1, v___x_3005_);
lean_ctor_set(v___x_2998_, 0, v___x_3003_);
v___x_3007_ = v___x_2998_;
goto v_reusejp_3006_;
}
else
{
lean_object* v_reuseFailAlloc_3011_; 
v_reuseFailAlloc_3011_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3011_, 0, v___x_3003_);
lean_ctor_set(v_reuseFailAlloc_3011_, 1, v___x_3005_);
v___x_3007_ = v_reuseFailAlloc_3011_;
goto v_reusejp_3006_;
}
v_reusejp_3006_:
{
lean_object* v___x_3009_; 
if (v_isShared_2989_ == 0)
{
lean_ctor_set(v___x_2988_, 0, v___x_3007_);
v___x_3009_ = v___x_2988_;
goto v_reusejp_3008_;
}
else
{
lean_object* v_reuseFailAlloc_3010_; 
v_reuseFailAlloc_3010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3010_, 0, v___x_3007_);
v___x_3009_ = v_reuseFailAlloc_3010_;
goto v_reusejp_3008_;
}
v_reusejp_3008_:
{
return v___x_3009_;
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
v___jp_3018_:
{
lean_object* v___x_3022_; 
v___x_3022_ = l_List_get_x3fInternal___redArg(v_eqSharpNs_2954_, v___y_3019_);
if (lean_obj_tag(v___x_3022_) == 0)
{
lean_object* v___x_3023_; lean_object* v___x_3024_; 
v___x_3023_ = lean_box(0);
v___x_3024_ = lean_apply_1(v___f_2955_, v___x_3023_);
v___y_2974_ = v___y_3021_;
v___y_2975_ = v___y_3020_;
v___y_2976_ = v___x_3024_;
goto v___jp_2973_;
}
else
{
lean_object* v_val_3025_; 
lean_dec(v___f_2955_);
v_val_3025_ = lean_ctor_get(v___x_3022_, 0);
lean_inc(v_val_3025_);
lean_dec_ref_known(v___x_3022_, 1);
v___y_2974_ = v___y_3021_;
v___y_2975_ = v___y_3020_;
v___y_2976_ = v_val_3025_;
goto v___jp_2973_;
}
}
v___jp_3026_:
{
lean_object* v___x_3028_; lean_object* v___x_3029_; 
v___x_3028_ = l_Int_toNat(v_a_2972_);
lean_inc(v___x_3028_);
v___x_3029_ = l_List_get_x3fInternal___redArg(v_eqNs_2956_, v___x_3028_);
if (lean_obj_tag(v___x_3029_) == 0)
{
lean_object* v___x_3030_; lean_object* v___x_3031_; 
v___x_3030_ = lean_box(0);
lean_inc(v___f_2955_);
v___x_3031_ = lean_apply_1(v___f_2955_, v___x_3030_);
v___y_3019_ = v___x_3028_;
v___y_3020_ = v___y_3027_;
v___y_3021_ = v___x_3031_;
goto v___jp_3018_;
}
else
{
lean_object* v_val_3032_; 
v_val_3032_ = lean_ctor_get(v___x_3029_, 0);
lean_inc(v_val_3032_);
lean_dec_ref_known(v___x_3029_, 1);
v___y_3019_ = v___x_3028_;
v___y_3020_ = v___y_3027_;
v___y_3021_ = v_val_3032_;
goto v___jp_3018_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___boxed(lean_object** _args){
lean_object* v_nPerTrace_3038_ = _args[0];
lean_object* v_fo_3039_ = _args[1];
lean_object* v_algMap_3040_ = _args[2];
lean_object* v_inst_3041_ = _args[3];
lean_object* v_vk_3042_ = _args[4];
lean_object* v_columnOpenings_3043_ = _args[5];
lean_object* v_publicValues_3044_ = _args[6];
lean_object* v_lSkip_3045_ = _args[7];
lean_object* v_rs_3046_ = _args[8];
lean_object* v_lambda_3047_ = _args[9];
lean_object* v_betaLogup_3048_ = _args[10];
lean_object* v_eqSharpNs_3049_ = _args[11];
lean_object* v___f_3050_ = _args[12];
lean_object* v_eqNs_3051_ = _args[13];
lean_object* v_eq3bPerTrace_3052_ = _args[14];
lean_object* v_state_3053_ = _args[15];
lean_object* v_entry_3054_ = _args[16];
_start:
{
lean_object* v_res_3055_; 
v_res_3055_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1(v_nPerTrace_3038_, v_fo_3039_, v_algMap_3040_, v_inst_3041_, v_vk_3042_, v_columnOpenings_3043_, v_publicValues_3044_, v_lSkip_3045_, v_rs_3046_, v_lambda_3047_, v_betaLogup_3048_, v_eqSharpNs_3049_, v___f_3050_, v_eqNs_3051_, v_eq3bPerTrace_3052_, v_state_3053_, v_entry_3054_);
lean_dec(v_eq3bPerTrace_3052_);
lean_dec(v_eqNs_3051_);
lean_dec(v_eqSharpNs_3049_);
lean_dec(v_publicValues_3044_);
lean_dec(v_columnOpenings_3043_);
lean_dec_ref(v_vk_3042_);
lean_dec(v_nPerTrace_3038_);
return v_res_3055_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(lean_object* v_fo_3058_, lean_object* v_algMap_3059_, lean_object* v_inst_3060_, lean_object* v_vk_3061_, lean_object* v_columnOpenings_3062_, lean_object* v_publicValues_3063_, lean_object* v_traceIdToAirId_3064_, lean_object* v_nPerTrace_3065_, lean_object* v_lSkip_3066_, lean_object* v_rs_3067_, lean_object* v_lambda_3068_, lean_object* v_mu_3069_, lean_object* v_betaLogup_3070_, lean_object* v_eq3bPerTrace_3071_, lean_object* v_eqNs_3072_, lean_object* v_eqSharpNs_3073_){
_start:
{
lean_object* v___f_3074_; lean_object* v___f_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; lean_object* v_traces_3079_; lean_object* v___x_3080_; lean_object* v___x_3081_; 
lean_inc_ref_n(v_fo_3058_, 2);
v___f_3074_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3074_, 0, v_fo_3058_);
v___f_3075_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___boxed), 17, 15);
lean_closure_set(v___f_3075_, 0, v_nPerTrace_3065_);
lean_closure_set(v___f_3075_, 1, v_fo_3058_);
lean_closure_set(v___f_3075_, 2, v_algMap_3059_);
lean_closure_set(v___f_3075_, 3, v_inst_3060_);
lean_closure_set(v___f_3075_, 4, v_vk_3061_);
lean_closure_set(v___f_3075_, 5, v_columnOpenings_3062_);
lean_closure_set(v___f_3075_, 6, v_publicValues_3063_);
lean_closure_set(v___f_3075_, 7, v_lSkip_3066_);
lean_closure_set(v___f_3075_, 8, v_rs_3067_);
lean_closure_set(v___f_3075_, 9, v_lambda_3068_);
lean_closure_set(v___f_3075_, 10, v_betaLogup_3070_);
lean_closure_set(v___f_3075_, 11, v_eqSharpNs_3073_);
lean_closure_set(v___f_3075_, 12, v___f_3074_);
lean_closure_set(v___f_3075_, 13, v_eqNs_3072_);
lean_closure_set(v___f_3075_, 14, v_eq3bPerTrace_3071_);
v___x_3076_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9));
v___x_3077_ = l_List_lengthTR___redArg(v_traceIdToAirId_3064_);
v___x_3078_ = l_List_range(v___x_3077_);
v_traces_3079_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_3078_, v_traceIdToAirId_3064_);
v___x_3080_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0));
v___x_3081_ = l_List_foldlM___redArg(v___x_3076_, v___f_3075_, v___x_3080_, v_traces_3079_);
if (lean_obj_tag(v___x_3081_) == 0)
{
lean_object* v_a_3082_; lean_object* v___x_3084_; uint8_t v_isShared_3085_; uint8_t v_isSharedCheck_3089_; 
lean_dec(v_mu_3069_);
lean_dec_ref(v_fo_3058_);
v_a_3082_ = lean_ctor_get(v___x_3081_, 0);
v_isSharedCheck_3089_ = !lean_is_exclusive(v___x_3081_);
if (v_isSharedCheck_3089_ == 0)
{
v___x_3084_ = v___x_3081_;
v_isShared_3085_ = v_isSharedCheck_3089_;
goto v_resetjp_3083_;
}
else
{
lean_inc(v_a_3082_);
lean_dec(v___x_3081_);
v___x_3084_ = lean_box(0);
v_isShared_3085_ = v_isSharedCheck_3089_;
goto v_resetjp_3083_;
}
v_resetjp_3083_:
{
lean_object* v___x_3087_; 
if (v_isShared_3085_ == 0)
{
v___x_3087_ = v___x_3084_;
goto v_reusejp_3086_;
}
else
{
lean_object* v_reuseFailAlloc_3088_; 
v_reuseFailAlloc_3088_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3088_, 0, v_a_3082_);
v___x_3087_ = v_reuseFailAlloc_3088_;
goto v_reusejp_3086_;
}
v_reusejp_3086_:
{
return v___x_3087_;
}
}
}
else
{
lean_object* v_a_3090_; lean_object* v___x_3092_; uint8_t v_isShared_3093_; uint8_t v_isSharedCheck_3112_; 
v_a_3090_ = lean_ctor_get(v___x_3081_, 0);
v_isSharedCheck_3112_ = !lean_is_exclusive(v___x_3081_);
if (v_isSharedCheck_3112_ == 0)
{
v___x_3092_ = v___x_3081_;
v_isShared_3093_ = v_isSharedCheck_3112_;
goto v_resetjp_3091_;
}
else
{
lean_inc(v_a_3090_);
lean_dec(v___x_3081_);
v___x_3092_ = lean_box(0);
v_isShared_3093_ = v_isSharedCheck_3112_;
goto v_resetjp_3091_;
}
v_resetjp_3091_:
{
lean_object* v_toRingOps_3094_; lean_object* v_toSemiringOps_3095_; lean_object* v_fst_3096_; lean_object* v_snd_3097_; lean_object* v_zero_3098_; lean_object* v_add_3099_; lean_object* v_mul_3100_; lean_object* v___f_3101_; lean_object* v___x_3102_; lean_object* v___x_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; lean_object* v___x_3110_; 
v_toRingOps_3094_ = lean_ctor_get(v_fo_3058_, 0);
v_toSemiringOps_3095_ = lean_ctor_get(v_toRingOps_3094_, 0);
v_fst_3096_ = lean_ctor_get(v_a_3090_, 0);
lean_inc(v_fst_3096_);
v_snd_3097_ = lean_ctor_get(v_a_3090_, 1);
lean_inc(v_snd_3097_);
lean_dec(v_a_3090_);
v_zero_3098_ = lean_ctor_get(v_toSemiringOps_3095_, 0);
lean_inc(v_zero_3098_);
v_add_3099_ = lean_ctor_get(v_toSemiringOps_3095_, 3);
v_mul_3100_ = lean_ctor_get(v_toSemiringOps_3095_, 4);
lean_inc(v_add_3099_);
lean_inc(v_mul_3100_);
v___f_3101_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2), 4, 2);
lean_closure_set(v___f_3101_, 0, v_mul_3100_);
lean_closure_set(v___f_3101_, 1, v_add_3099_);
v___x_3102_ = l_List_reverse___redArg(v_fst_3096_);
v___x_3103_ = l_List_reverse___redArg(v_snd_3097_);
v___x_3104_ = l_List_appendTR___redArg(v___x_3102_, v___x_3103_);
v___x_3105_ = l_List_lengthTR___redArg(v___x_3104_);
v___x_3106_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_3058_, v_mu_3069_, v___x_3105_);
v___x_3107_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_3104_, v___x_3106_);
v___x_3108_ = l_List_foldl___redArg(v___f_3101_, v_zero_3098_, v___x_3107_);
if (v_isShared_3093_ == 0)
{
lean_ctor_set(v___x_3092_, 0, v___x_3108_);
v___x_3110_ = v___x_3092_;
goto v_reusejp_3109_;
}
else
{
lean_object* v_reuseFailAlloc_3111_; 
v_reuseFailAlloc_3111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3111_, 0, v___x_3108_);
v___x_3110_ = v_reuseFailAlloc_3111_;
goto v_reusejp_3109_;
}
v_reusejp_3109_:
{
return v___x_3110_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings(lean_object* v_F_3113_, lean_object* v_EF_3114_, lean_object* v_Digest_3115_, lean_object* v_fo_3116_, lean_object* v_algMap_3117_, lean_object* v_inst_3118_, lean_object* v_vk_3119_, lean_object* v_columnOpenings_3120_, lean_object* v_publicValues_3121_, lean_object* v_traceIdToAirId_3122_, lean_object* v_nPerTrace_3123_, lean_object* v_lSkip_3124_, lean_object* v_rs_3125_, lean_object* v_lambda_3126_, lean_object* v_mu_3127_, lean_object* v_betaLogup_3128_, lean_object* v_eq3bPerTrace_3129_, lean_object* v_eqNs_3130_, lean_object* v_eqSharpNs_3131_){
_start:
{
lean_object* v___x_3132_; 
v___x_3132_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(v_fo_3116_, v_algMap_3117_, v_inst_3118_, v_vk_3119_, v_columnOpenings_3120_, v_publicValues_3121_, v_traceIdToAirId_3122_, v_nPerTrace_3123_, v_lSkip_3124_, v_rs_3125_, v_lambda_3126_, v_mu_3127_, v_betaLogup_3128_, v_eq3bPerTrace_3129_, v_eqNs_3130_, v_eqSharpNs_3131_);
return v___x_3132_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___boxed(lean_object** _args){
lean_object* v_F_3133_ = _args[0];
lean_object* v_EF_3134_ = _args[1];
lean_object* v_Digest_3135_ = _args[2];
lean_object* v_fo_3136_ = _args[3];
lean_object* v_algMap_3137_ = _args[4];
lean_object* v_inst_3138_ = _args[5];
lean_object* v_vk_3139_ = _args[6];
lean_object* v_columnOpenings_3140_ = _args[7];
lean_object* v_publicValues_3141_ = _args[8];
lean_object* v_traceIdToAirId_3142_ = _args[9];
lean_object* v_nPerTrace_3143_ = _args[10];
lean_object* v_lSkip_3144_ = _args[11];
lean_object* v_rs_3145_ = _args[12];
lean_object* v_lambda_3146_ = _args[13];
lean_object* v_mu_3147_ = _args[14];
lean_object* v_betaLogup_3148_ = _args[15];
lean_object* v_eq3bPerTrace_3149_ = _args[16];
lean_object* v_eqNs_3150_ = _args[17];
lean_object* v_eqSharpNs_3151_ = _args[18];
_start:
{
lean_object* v_res_3152_; 
v_res_3152_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings(v_F_3133_, v_EF_3134_, v_Digest_3135_, v_fo_3136_, v_algMap_3137_, v_inst_3138_, v_vk_3139_, v_columnOpenings_3140_, v_publicValues_3141_, v_traceIdToAirId_3142_, v_nPerTrace_3143_, v_lSkip_3144_, v_rs_3145_, v_lambda_3146_, v_mu_3147_, v_betaLogup_3148_, v_eq3bPerTrace_3149_, v_eqNs_3150_, v_eqSharpNs_3151_);
return v_res_3152_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0(lean_object* v_vk_3153_, lean_object* v_airId_3154_){
_start:
{
lean_object* v___x_3155_; 
v___x_3155_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_3153_, v_airId_3154_);
if (lean_obj_tag(v___x_3155_) == 0)
{
lean_object* v___x_3156_; 
v___x_3156_ = lean_unsigned_to_nat(0u);
return v___x_3156_;
}
else
{
lean_object* v_val_3157_; lean_object* v_symbolicConstraints_3158_; lean_object* v_interactions_3159_; lean_object* v___x_3160_; 
v_val_3157_ = lean_ctor_get(v___x_3155_, 0);
lean_inc(v_val_3157_);
lean_dec_ref_known(v___x_3155_, 1);
v_symbolicConstraints_3158_ = lean_ctor_get(v_val_3157_, 2);
lean_inc_ref(v_symbolicConstraints_3158_);
lean_dec(v_val_3157_);
v_interactions_3159_ = lean_ctor_get(v_symbolicConstraints_3158_, 1);
lean_inc(v_interactions_3159_);
lean_dec_ref(v_symbolicConstraints_3158_);
v___x_3160_ = l_List_lengthTR___redArg(v_interactions_3159_);
lean_dec(v_interactions_3159_);
return v___x_3160_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0___boxed(lean_object* v_vk_3161_, lean_object* v_airId_3162_){
_start:
{
lean_object* v_res_3163_; 
v_res_3163_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0(v_vk_3161_, v_airId_3162_);
lean_dec_ref(v_vk_3161_);
return v_res_3163_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__1(lean_object* v_inst_3164_, lean_object* v_inst_3165_, lean_object* v_fo_3166_, lean_object* v_acc_3167_, lean_object* v_roundPoly_3168_, lean_object* v___y_3169_){
_start:
{
lean_object* v_fst_3170_; lean_object* v_snd_3171_; lean_object* v___x_3173_; uint8_t v_isShared_3174_; uint8_t v_isSharedCheck_3216_; 
v_fst_3170_ = lean_ctor_get(v_acc_3167_, 0);
v_snd_3171_ = lean_ctor_get(v_acc_3167_, 1);
v_isSharedCheck_3216_ = !lean_is_exclusive(v_acc_3167_);
if (v_isSharedCheck_3216_ == 0)
{
v___x_3173_ = v_acc_3167_;
v_isShared_3174_ = v_isSharedCheck_3216_;
goto v_resetjp_3172_;
}
else
{
lean_inc(v_snd_3171_);
lean_inc(v_fst_3170_);
lean_dec(v_acc_3167_);
v___x_3173_ = lean_box(0);
v_isShared_3174_ = v_isSharedCheck_3216_;
goto v_resetjp_3172_;
}
v_resetjp_3172_:
{
lean_object* v___x_3175_; 
lean_inc(v_roundPoly_3168_);
lean_inc_ref(v_inst_3165_);
lean_inc_ref(v_inst_3164_);
v___x_3175_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg(v_inst_3164_, v_inst_3165_, v_roundPoly_3168_, v___y_3169_);
if (lean_obj_tag(v___x_3175_) == 0)
{
lean_object* v_a_3176_; lean_object* v___x_3178_; uint8_t v_isShared_3179_; uint8_t v_isSharedCheck_3183_; 
lean_del_object(v___x_3173_);
lean_dec(v_snd_3171_);
lean_dec(v_fst_3170_);
lean_dec(v_roundPoly_3168_);
lean_dec_ref(v_fo_3166_);
lean_dec_ref(v_inst_3165_);
lean_dec_ref(v_inst_3164_);
v_a_3176_ = lean_ctor_get(v___x_3175_, 0);
v_isSharedCheck_3183_ = !lean_is_exclusive(v___x_3175_);
if (v_isSharedCheck_3183_ == 0)
{
v___x_3178_ = v___x_3175_;
v_isShared_3179_ = v_isSharedCheck_3183_;
goto v_resetjp_3177_;
}
else
{
lean_inc(v_a_3176_);
lean_dec(v___x_3175_);
v___x_3178_ = lean_box(0);
v_isShared_3179_ = v_isSharedCheck_3183_;
goto v_resetjp_3177_;
}
v_resetjp_3177_:
{
lean_object* v___x_3181_; 
if (v_isShared_3179_ == 0)
{
v___x_3181_ = v___x_3178_;
goto v_reusejp_3180_;
}
else
{
lean_object* v_reuseFailAlloc_3182_; 
v_reuseFailAlloc_3182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3182_, 0, v_a_3176_);
v___x_3181_ = v_reuseFailAlloc_3182_;
goto v_reusejp_3180_;
}
v_reusejp_3180_:
{
return v___x_3181_;
}
}
}
else
{
lean_object* v_a_3184_; lean_object* v_snd_3185_; lean_object* v___x_3187_; uint8_t v_isShared_3188_; uint8_t v_isSharedCheck_3214_; 
v_a_3184_ = lean_ctor_get(v___x_3175_, 0);
lean_inc(v_a_3184_);
lean_dec_ref_known(v___x_3175_, 1);
v_snd_3185_ = lean_ctor_get(v_a_3184_, 1);
v_isSharedCheck_3214_ = !lean_is_exclusive(v_a_3184_);
if (v_isSharedCheck_3214_ == 0)
{
lean_object* v_unused_3215_; 
v_unused_3215_ = lean_ctor_get(v_a_3184_, 0);
lean_dec(v_unused_3215_);
v___x_3187_ = v_a_3184_;
v_isShared_3188_ = v_isSharedCheck_3214_;
goto v_resetjp_3186_;
}
else
{
lean_inc(v_snd_3185_);
lean_dec(v_a_3184_);
v___x_3187_ = lean_box(0);
v_isShared_3188_ = v_isSharedCheck_3214_;
goto v_resetjp_3186_;
}
v_resetjp_3186_:
{
lean_object* v___x_3189_; lean_object* v_a_3190_; lean_object* v___x_3192_; uint8_t v_isShared_3193_; uint8_t v_isSharedCheck_3213_; 
v___x_3189_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3164_, v_inst_3165_, v_snd_3185_);
v_a_3190_ = lean_ctor_get(v___x_3189_, 0);
v_isSharedCheck_3213_ = !lean_is_exclusive(v___x_3189_);
if (v_isSharedCheck_3213_ == 0)
{
v___x_3192_ = v___x_3189_;
v_isShared_3193_ = v_isSharedCheck_3213_;
goto v_resetjp_3191_;
}
else
{
lean_inc(v_a_3190_);
lean_dec(v___x_3189_);
v___x_3192_ = lean_box(0);
v_isShared_3193_ = v_isSharedCheck_3213_;
goto v_resetjp_3191_;
}
v_resetjp_3191_:
{
lean_object* v_fst_3194_; lean_object* v_snd_3195_; lean_object* v___x_3197_; uint8_t v_isShared_3198_; uint8_t v_isSharedCheck_3212_; 
v_fst_3194_ = lean_ctor_get(v_a_3190_, 0);
v_snd_3195_ = lean_ctor_get(v_a_3190_, 1);
v_isSharedCheck_3212_ = !lean_is_exclusive(v_a_3190_);
if (v_isSharedCheck_3212_ == 0)
{
v___x_3197_ = v_a_3190_;
v_isShared_3198_ = v_isSharedCheck_3212_;
goto v_resetjp_3196_;
}
else
{
lean_inc(v_snd_3195_);
lean_inc(v_fst_3194_);
lean_dec(v_a_3190_);
v___x_3197_ = lean_box(0);
v_isShared_3198_ = v_isSharedCheck_3212_;
goto v_resetjp_3196_;
}
v_resetjp_3196_:
{
lean_object* v___x_3199_; lean_object* v___x_3201_; 
lean_inc(v_fst_3194_);
v___x_3199_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(v_fo_3166_, v_fst_3170_, v_roundPoly_3168_, v_fst_3194_);
if (v_isShared_3174_ == 0)
{
lean_ctor_set_tag(v___x_3173_, 1);
lean_ctor_set(v___x_3173_, 0, v_fst_3194_);
v___x_3201_ = v___x_3173_;
goto v_reusejp_3200_;
}
else
{
lean_object* v_reuseFailAlloc_3211_; 
v_reuseFailAlloc_3211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3211_, 0, v_fst_3194_);
lean_ctor_set(v_reuseFailAlloc_3211_, 1, v_snd_3171_);
v___x_3201_ = v_reuseFailAlloc_3211_;
goto v_reusejp_3200_;
}
v_reusejp_3200_:
{
lean_object* v___x_3203_; 
if (v_isShared_3198_ == 0)
{
lean_ctor_set(v___x_3197_, 1, v___x_3201_);
lean_ctor_set(v___x_3197_, 0, v___x_3199_);
v___x_3203_ = v___x_3197_;
goto v_reusejp_3202_;
}
else
{
lean_object* v_reuseFailAlloc_3210_; 
v_reuseFailAlloc_3210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3210_, 0, v___x_3199_);
lean_ctor_set(v_reuseFailAlloc_3210_, 1, v___x_3201_);
v___x_3203_ = v_reuseFailAlloc_3210_;
goto v_reusejp_3202_;
}
v_reusejp_3202_:
{
lean_object* v___x_3205_; 
if (v_isShared_3188_ == 0)
{
lean_ctor_set(v___x_3187_, 1, v_snd_3195_);
lean_ctor_set(v___x_3187_, 0, v___x_3203_);
v___x_3205_ = v___x_3187_;
goto v_reusejp_3204_;
}
else
{
lean_object* v_reuseFailAlloc_3209_; 
v_reuseFailAlloc_3209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3209_, 0, v___x_3203_);
lean_ctor_set(v_reuseFailAlloc_3209_, 1, v_snd_3195_);
v___x_3205_ = v_reuseFailAlloc_3209_;
goto v_reusejp_3204_;
}
v_reusejp_3204_:
{
lean_object* v___x_3207_; 
if (v_isShared_3193_ == 0)
{
lean_ctor_set(v___x_3192_, 0, v___x_3205_);
v___x_3207_ = v___x_3192_;
goto v_reusejp_3206_;
}
else
{
lean_object* v_reuseFailAlloc_3208_; 
v_reuseFailAlloc_3208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3208_, 0, v___x_3205_);
v___x_3207_ = v_reuseFailAlloc_3208_;
goto v_reusejp_3206_;
}
v_reusejp_3206_:
{
return v___x_3207_;
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
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__2(lean_object* v_inst_3217_, lean_object* v_inst_3218_, lean_object* v_x_3219_, lean_object* v_entry_3220_, lean_object* v___y_3221_){
_start:
{
lean_object* v_fst_3222_; lean_object* v_snd_3223_; lean_object* v___x_3224_; lean_object* v_a_3225_; lean_object* v_snd_3226_; lean_object* v___x_3227_; 
v_fst_3222_ = lean_ctor_get(v_entry_3220_, 0);
lean_inc(v_fst_3222_);
v_snd_3223_ = lean_ctor_get(v_entry_3220_, 1);
lean_inc(v_snd_3223_);
lean_dec_ref(v_entry_3220_);
lean_inc_ref(v_inst_3218_);
lean_inc_ref(v_inst_3217_);
v___x_3224_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_3217_, v_inst_3218_, v_fst_3222_, v___y_3221_);
v_a_3225_ = lean_ctor_get(v___x_3224_, 0);
lean_inc(v_a_3225_);
lean_dec_ref(v___x_3224_);
v_snd_3226_ = lean_ctor_get(v_a_3225_, 1);
lean_inc(v_snd_3226_);
lean_dec(v_a_3225_);
v___x_3227_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExt___redArg(v_inst_3217_, v_inst_3218_, v_snd_3223_, v_snd_3226_);
return v___x_3227_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3(lean_object* v_inst_3228_, lean_object* v_inst_3229_, lean_object* v_acc_3230_, lean_object* v_x_3231_, lean_object* v___y_3232_){
_start:
{
lean_object* v_fst_3233_; lean_object* v___x_3235_; uint8_t v_isShared_3236_; uint8_t v_isSharedCheck_3260_; 
v_fst_3233_ = lean_ctor_get(v_acc_3230_, 0);
v_isSharedCheck_3260_ = !lean_is_exclusive(v_acc_3230_);
if (v_isSharedCheck_3260_ == 0)
{
lean_object* v_unused_3261_; 
v_unused_3261_ = lean_ctor_get(v_acc_3230_, 1);
lean_dec(v_unused_3261_);
v___x_3235_ = v_acc_3230_;
v_isShared_3236_ = v_isSharedCheck_3260_;
goto v_resetjp_3234_;
}
else
{
lean_inc(v_fst_3233_);
lean_dec(v_acc_3230_);
v___x_3235_ = lean_box(0);
v_isShared_3236_ = v_isSharedCheck_3260_;
goto v_resetjp_3234_;
}
v_resetjp_3234_:
{
lean_object* v___x_3237_; lean_object* v_a_3238_; lean_object* v___x_3240_; uint8_t v_isShared_3241_; uint8_t v_isSharedCheck_3259_; 
v___x_3237_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3228_, v_inst_3229_, v___y_3232_);
v_a_3238_ = lean_ctor_get(v___x_3237_, 0);
v_isSharedCheck_3259_ = !lean_is_exclusive(v___x_3237_);
if (v_isSharedCheck_3259_ == 0)
{
v___x_3240_ = v___x_3237_;
v_isShared_3241_ = v_isSharedCheck_3259_;
goto v_resetjp_3239_;
}
else
{
lean_inc(v_a_3238_);
lean_dec(v___x_3237_);
v___x_3240_ = lean_box(0);
v_isShared_3241_ = v_isSharedCheck_3259_;
goto v_resetjp_3239_;
}
v_resetjp_3239_:
{
lean_object* v_fst_3242_; lean_object* v_snd_3243_; lean_object* v___x_3245_; uint8_t v_isShared_3246_; uint8_t v_isSharedCheck_3258_; 
v_fst_3242_ = lean_ctor_get(v_a_3238_, 0);
v_snd_3243_ = lean_ctor_get(v_a_3238_, 1);
v_isSharedCheck_3258_ = !lean_is_exclusive(v_a_3238_);
if (v_isSharedCheck_3258_ == 0)
{
v___x_3245_ = v_a_3238_;
v_isShared_3246_ = v_isSharedCheck_3258_;
goto v_resetjp_3244_;
}
else
{
lean_inc(v_snd_3243_);
lean_inc(v_fst_3242_);
lean_dec(v_a_3238_);
v___x_3245_ = lean_box(0);
v_isShared_3246_ = v_isSharedCheck_3258_;
goto v_resetjp_3244_;
}
v_resetjp_3244_:
{
lean_object* v___x_3247_; lean_object* v___x_3248_; lean_object* v___x_3250_; 
v___x_3247_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3247_, 0, v_fst_3242_);
lean_ctor_set(v___x_3247_, 1, v_fst_3233_);
v___x_3248_ = lean_box(0);
if (v_isShared_3246_ == 0)
{
lean_ctor_set(v___x_3245_, 1, v___x_3248_);
lean_ctor_set(v___x_3245_, 0, v___x_3247_);
v___x_3250_ = v___x_3245_;
goto v_reusejp_3249_;
}
else
{
lean_object* v_reuseFailAlloc_3257_; 
v_reuseFailAlloc_3257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3257_, 0, v___x_3247_);
lean_ctor_set(v_reuseFailAlloc_3257_, 1, v___x_3248_);
v___x_3250_ = v_reuseFailAlloc_3257_;
goto v_reusejp_3249_;
}
v_reusejp_3249_:
{
lean_object* v___x_3252_; 
if (v_isShared_3236_ == 0)
{
lean_ctor_set(v___x_3235_, 1, v_snd_3243_);
lean_ctor_set(v___x_3235_, 0, v___x_3250_);
v___x_3252_ = v___x_3235_;
goto v_reusejp_3251_;
}
else
{
lean_object* v_reuseFailAlloc_3256_; 
v_reuseFailAlloc_3256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3256_, 0, v___x_3250_);
lean_ctor_set(v_reuseFailAlloc_3256_, 1, v_snd_3243_);
v___x_3252_ = v_reuseFailAlloc_3256_;
goto v_reusejp_3251_;
}
v_reusejp_3251_:
{
lean_object* v___x_3254_; 
if (v_isShared_3241_ == 0)
{
lean_ctor_set(v___x_3240_, 0, v___x_3252_);
v___x_3254_ = v___x_3240_;
goto v_reusejp_3253_;
}
else
{
lean_object* v_reuseFailAlloc_3255_; 
v_reuseFailAlloc_3255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3255_, 0, v___x_3252_);
v___x_3254_ = v_reuseFailAlloc_3255_;
goto v_reusejp_3253_;
}
v_reusejp_3253_:
{
return v___x_3254_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3___boxed(lean_object* v_inst_3262_, lean_object* v_inst_3263_, lean_object* v_acc_3264_, lean_object* v_x_3265_, lean_object* v___y_3266_){
_start:
{
lean_object* v_res_3267_; 
v_res_3267_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3(v_inst_3262_, v_inst_3263_, v_acc_3264_, v_x_3265_, v___y_3266_);
lean_dec(v_x_3265_);
return v_res_3267_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(lean_object* v_inst_3274_, lean_object* v_inst_3275_, lean_object* v_inst_3276_, lean_object* v_inst_3277_, lean_object* v_inst_3278_, lean_object* v_fo_3279_, lean_object* v_algMap_3280_, lean_object* v_vk_3281_, lean_object* v_gkrProof_3282_, lean_object* v_batchProof_3283_, lean_object* v_traceVdata_3284_, lean_object* v_publicValues_3285_, lean_object* v_traceIdToAirId_3286_, lean_object* v_nPerTrace_3287_, lean_object* v_a_3288_){
_start:
{
lean_object* v_inner_3289_; lean_object* v_params_3290_; lean_object* v___x_3291_; lean_object* v_logupPowWitness_3292_; lean_object* v_q0Claim_3293_; lean_object* v___x_3294_; lean_object* v___x_3295_; 
v_inner_3289_ = lean_ctor_get(v_vk_3281_, 0);
v_params_3290_ = lean_ctor_get(v_inner_3289_, 0);
v___x_3291_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19));
v_logupPowWitness_3292_ = lean_ctor_get(v_gkrProof_3282_, 0);
v_q0Claim_3293_ = lean_ctor_get(v_gkrProof_3282_, 1);
v___x_3294_ = lp_swirl_x2dfv_Fundamentals_Runtime_SystemParams_logupPowBits(v_params_3290_);
lean_inc(v_logupPowWitness_3292_);
lean_inc_ref(v_inst_3276_);
v___x_3295_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observePowWitness___redArg(v_inst_3276_, v_inst_3277_, v___x_3294_, v_logupPowWitness_3292_, v_a_3288_);
lean_dec(v___x_3294_);
if (lean_obj_tag(v___x_3295_) == 0)
{
lean_object* v_a_3296_; lean_object* v___x_3298_; uint8_t v_isShared_3299_; uint8_t v_isSharedCheck_3303_; 
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_batchProof_3283_);
lean_dec_ref(v_gkrProof_3282_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3296_ = lean_ctor_get(v___x_3295_, 0);
v_isSharedCheck_3303_ = !lean_is_exclusive(v___x_3295_);
if (v_isSharedCheck_3303_ == 0)
{
v___x_3298_ = v___x_3295_;
v_isShared_3299_ = v_isSharedCheck_3303_;
goto v_resetjp_3297_;
}
else
{
lean_inc(v_a_3296_);
lean_dec(v___x_3295_);
v___x_3298_ = lean_box(0);
v_isShared_3299_ = v_isSharedCheck_3303_;
goto v_resetjp_3297_;
}
v_resetjp_3297_:
{
lean_object* v___x_3301_; 
if (v_isShared_3299_ == 0)
{
v___x_3301_ = v___x_3298_;
goto v_reusejp_3300_;
}
else
{
lean_object* v_reuseFailAlloc_3302_; 
v_reuseFailAlloc_3302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3302_, 0, v_a_3296_);
v___x_3301_ = v_reuseFailAlloc_3302_;
goto v_reusejp_3300_;
}
v_reusejp_3300_:
{
return v___x_3301_;
}
}
}
else
{
lean_object* v_a_3304_; lean_object* v_snd_3305_; lean_object* v___x_3306_; lean_object* v_a_3307_; lean_object* v_fst_3308_; lean_object* v_snd_3309_; lean_object* v___x_3310_; lean_object* v_a_3311_; lean_object* v_fst_3312_; lean_object* v_snd_3313_; lean_object* v_lSkip_3314_; lean_object* v___f_3315_; lean_object* v___f_3316_; lean_object* v___f_3317_; lean_object* v___f_3318_; lean_object* v_nMax_3319_; lean_object* v___x_3320_; lean_object* v___x_3321_; lean_object* v___y_3323_; lean_object* v___y_3324_; lean_object* v___y_3325_; lean_object* v___y_3326_; lean_object* v___y_3327_; lean_object* v_fst_3485_; lean_object* v_fst_3486_; lean_object* v_snd_3487_; lean_object* v___y_3488_; lean_object* v___x_3490_; uint8_t v___x_3491_; 
v_a_3304_ = lean_ctor_get(v___x_3295_, 0);
lean_inc(v_a_3304_);
lean_dec_ref_known(v___x_3295_, 1);
v_snd_3305_ = lean_ctor_get(v_a_3304_, 1);
lean_inc(v_snd_3305_);
lean_dec(v_a_3304_);
lean_inc_ref_n(v_inst_3278_, 5);
lean_inc_ref_n(v_inst_3276_, 5);
v___x_3306_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3276_, v_inst_3278_, v_snd_3305_);
v_a_3307_ = lean_ctor_get(v___x_3306_, 0);
lean_inc(v_a_3307_);
lean_dec_ref(v___x_3306_);
v_fst_3308_ = lean_ctor_get(v_a_3307_, 0);
lean_inc(v_fst_3308_);
v_snd_3309_ = lean_ctor_get(v_a_3307_, 1);
lean_inc(v_snd_3309_);
lean_dec(v_a_3307_);
v___x_3310_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3276_, v_inst_3278_, v_snd_3309_);
v_a_3311_ = lean_ctor_get(v___x_3310_, 0);
lean_inc(v_a_3311_);
lean_dec_ref(v___x_3310_);
v_fst_3312_ = lean_ctor_get(v_a_3311_, 0);
lean_inc(v_fst_3312_);
v_snd_3313_ = lean_ctor_get(v_a_3311_, 1);
lean_inc(v_snd_3313_);
lean_dec(v_a_3311_);
v_lSkip_3314_ = lean_ctor_get(v_params_3290_, 0);
lean_inc(v_lSkip_3314_);
lean_inc_ref(v_vk_3281_);
v___f_3315_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3315_, 0, v_vk_3281_);
lean_inc_ref(v_fo_3279_);
v___f_3316_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__1), 6, 3);
lean_closure_set(v___f_3316_, 0, v_inst_3276_);
lean_closure_set(v___f_3316_, 1, v_inst_3278_);
lean_closure_set(v___f_3316_, 2, v_fo_3279_);
v___f_3317_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__2), 5, 2);
lean_closure_set(v___f_3317_, 0, v_inst_3276_);
lean_closure_set(v___f_3317_, 1, v_inst_3278_);
v___f_3318_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3___boxed), 5, 2);
lean_closure_set(v___f_3318_, 0, v_inst_3276_);
lean_closure_set(v___f_3318_, 1, v_inst_3278_);
lean_inc_n(v_traceIdToAirId_3286_, 2);
v_nMax_3319_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(v_lSkip_3314_, v_traceVdata_3284_, v_traceIdToAirId_3286_);
lean_inc_ref(v_params_3290_);
v___x_3320_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_totalInteractions___redArg(v_params_3290_, v_vk_3281_, v_traceVdata_3284_, v_traceIdToAirId_3286_);
v___x_3321_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_calculateNLogup(v_lSkip_3314_, v___x_3320_);
v___x_3490_ = lean_unsigned_to_nat(0u);
v___x_3491_ = lean_nat_dec_lt(v___x_3490_, v___x_3320_);
lean_dec(v___x_3320_);
if (v___x_3491_ == 0)
{
lean_object* v_toRingOps_3492_; lean_object* v_toSemiringOps_3493_; lean_object* v_zero_3494_; lean_object* v_one_3495_; lean_object* v___x_3496_; uint8_t v___x_3497_; 
lean_inc(v_q0Claim_3293_);
lean_dec_ref(v_gkrProof_3282_);
v_toRingOps_3492_ = lean_ctor_get(v_fo_3279_, 0);
v_toSemiringOps_3493_ = lean_ctor_get(v_toRingOps_3492_, 0);
v_zero_3494_ = lean_ctor_get(v_toSemiringOps_3493_, 0);
v_one_3495_ = lean_ctor_get(v_toSemiringOps_3493_, 1);
lean_inc_ref(v_inst_3274_);
lean_inc(v_one_3495_);
v___x_3496_ = lean_apply_2(v_inst_3274_, v_q0Claim_3293_, v_one_3495_);
v___x_3497_ = lean_unbox(v___x_3496_);
if (v___x_3497_ == 0)
{
lean_object* v___x_3498_; 
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3318_);
lean_dec_ref(v___f_3317_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_snd_3313_);
lean_dec(v_fst_3312_);
lean_dec(v_fst_3308_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_batchProof_3283_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v___x_3498_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3498_;
}
else
{
lean_object* v___x_3499_; 
v___x_3499_ = lean_box(0);
lean_inc(v_fst_3308_);
lean_inc(v_zero_3494_);
v_fst_3485_ = v_zero_3494_;
v_fst_3486_ = v_fst_3308_;
v_snd_3487_ = v___x_3499_;
v___y_3488_ = v_snd_3313_;
goto v___jp_3484_;
}
}
else
{
lean_object* v___x_3500_; lean_object* v___x_3501_; 
v___x_3500_ = lean_nat_add(v_lSkip_3314_, v___x_3321_);
lean_inc_ref(v_fo_3279_);
lean_inc_ref(v_inst_3278_);
lean_inc_ref(v_inst_3276_);
lean_inc_ref(v_inst_3274_);
v___x_3501_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(v_inst_3274_, v_inst_3276_, v_inst_3278_, v_fo_3279_, v_gkrProof_3282_, v___x_3500_, v_snd_3313_);
lean_dec(v___x_3500_);
if (lean_obj_tag(v___x_3501_) == 0)
{
lean_object* v_a_3502_; lean_object* v___x_3504_; uint8_t v_isShared_3505_; uint8_t v_isSharedCheck_3509_; 
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3318_);
lean_dec_ref(v___f_3317_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_fst_3308_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_batchProof_3283_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3502_ = lean_ctor_get(v___x_3501_, 0);
v_isSharedCheck_3509_ = !lean_is_exclusive(v___x_3501_);
if (v_isSharedCheck_3509_ == 0)
{
v___x_3504_ = v___x_3501_;
v_isShared_3505_ = v_isSharedCheck_3509_;
goto v_resetjp_3503_;
}
else
{
lean_inc(v_a_3502_);
lean_dec(v___x_3501_);
v___x_3504_ = lean_box(0);
v_isShared_3505_ = v_isSharedCheck_3509_;
goto v_resetjp_3503_;
}
v_resetjp_3503_:
{
lean_object* v___x_3507_; 
if (v_isShared_3505_ == 0)
{
v___x_3507_ = v___x_3504_;
goto v_reusejp_3506_;
}
else
{
lean_object* v_reuseFailAlloc_3508_; 
v_reuseFailAlloc_3508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3508_, 0, v_a_3502_);
v___x_3507_ = v_reuseFailAlloc_3508_;
goto v_reusejp_3506_;
}
v_reusejp_3506_:
{
return v___x_3507_;
}
}
}
else
{
lean_object* v_a_3510_; lean_object* v_fst_3511_; lean_object* v_snd_3512_; lean_object* v_numeratorClaim_3513_; lean_object* v_denominatorClaim_3514_; lean_object* v_xi_3515_; 
v_a_3510_ = lean_ctor_get(v___x_3501_, 0);
lean_inc(v_a_3510_);
lean_dec_ref_known(v___x_3501_, 1);
v_fst_3511_ = lean_ctor_get(v_a_3510_, 0);
lean_inc(v_fst_3511_);
v_snd_3512_ = lean_ctor_get(v_a_3510_, 1);
lean_inc(v_snd_3512_);
lean_dec(v_a_3510_);
v_numeratorClaim_3513_ = lean_ctor_get(v_fst_3511_, 0);
lean_inc(v_numeratorClaim_3513_);
v_denominatorClaim_3514_ = lean_ctor_get(v_fst_3511_, 1);
lean_inc(v_denominatorClaim_3514_);
v_xi_3515_ = lean_ctor_get(v_fst_3511_, 2);
lean_inc(v_xi_3515_);
lean_dec(v_fst_3511_);
v_fst_3485_ = v_numeratorClaim_3513_;
v_fst_3486_ = v_denominatorClaim_3514_;
v_snd_3487_ = v_xi_3515_;
v___y_3488_ = v_snd_3512_;
goto v___jp_3484_;
}
}
v___jp_3322_:
{
lean_object* v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_9425__overap_3335_; lean_object* v___x_3336_; 
v___x_3328_ = lean_nat_add(v_lSkip_3314_, v___y_3327_);
lean_dec(v___y_3327_);
v___x_3329_ = l_List_lengthTR___redArg(v___y_3325_);
v___x_3330_ = lean_nat_sub(v___x_3328_, v___x_3329_);
lean_dec(v___x_3329_);
lean_dec(v___x_3328_);
v___x_3331_ = lean_box(0);
v___x_3332_ = lean_box(0);
v___x_3333_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0));
v___x_3334_ = l_List_range(v___x_3330_);
v___x_9425__overap_3335_ = l_List_foldlM___redArg(v___x_3291_, v___f_3318_, v___x_3333_, v___x_3334_);
v___x_3336_ = lean_apply_1(v___x_9425__overap_3335_, v___y_3323_);
if (lean_obj_tag(v___x_3336_) == 0)
{
lean_object* v_a_3337_; lean_object* v___x_3339_; uint8_t v_isShared_3340_; uint8_t v_isSharedCheck_3344_; 
lean_dec(v___y_3326_);
lean_dec(v___y_3325_);
lean_dec(v___y_3324_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3317_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_fst_3308_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_batchProof_3283_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3337_ = lean_ctor_get(v___x_3336_, 0);
v_isSharedCheck_3344_ = !lean_is_exclusive(v___x_3336_);
if (v_isSharedCheck_3344_ == 0)
{
v___x_3339_ = v___x_3336_;
v_isShared_3340_ = v_isSharedCheck_3344_;
goto v_resetjp_3338_;
}
else
{
lean_inc(v_a_3337_);
lean_dec(v___x_3336_);
v___x_3339_ = lean_box(0);
v_isShared_3340_ = v_isSharedCheck_3344_;
goto v_resetjp_3338_;
}
v_resetjp_3338_:
{
lean_object* v___x_3342_; 
if (v_isShared_3340_ == 0)
{
v___x_3342_ = v___x_3339_;
goto v_reusejp_3341_;
}
else
{
lean_object* v_reuseFailAlloc_3343_; 
v_reuseFailAlloc_3343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3343_, 0, v_a_3337_);
v___x_3342_ = v_reuseFailAlloc_3343_;
goto v_reusejp_3341_;
}
v_reusejp_3341_:
{
return v___x_3342_;
}
}
}
else
{
lean_object* v_a_3345_; lean_object* v_fst_3346_; lean_object* v_snd_3347_; lean_object* v_fst_3348_; lean_object* v___x_3349_; lean_object* v_a_3350_; lean_object* v_fst_3351_; lean_object* v_snd_3352_; lean_object* v_numeratorTermPerAir_3353_; lean_object* v_denominatorTermPerAir_3354_; lean_object* v_univariateRoundCoeffs_3355_; lean_object* v_sumcheckRoundPolys_3356_; lean_object* v_columnOpenings_3357_; lean_object* v___x_3358_; lean_object* v___x_3359_; uint8_t v___x_3360_; 
v_a_3345_ = lean_ctor_get(v___x_3336_, 0);
lean_inc(v_a_3345_);
lean_dec_ref_known(v___x_3336_, 1);
v_fst_3346_ = lean_ctor_get(v_a_3345_, 0);
lean_inc(v_fst_3346_);
v_snd_3347_ = lean_ctor_get(v_a_3345_, 1);
lean_inc(v_snd_3347_);
lean_dec(v_a_3345_);
v_fst_3348_ = lean_ctor_get(v_fst_3346_, 0);
lean_inc(v_fst_3348_);
lean_dec(v_fst_3346_);
lean_inc_ref(v_inst_3278_);
lean_inc_ref(v_inst_3276_);
v___x_3349_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3276_, v_inst_3278_, v_snd_3347_);
v_a_3350_ = lean_ctor_get(v___x_3349_, 0);
lean_inc(v_a_3350_);
lean_dec_ref(v___x_3349_);
v_fst_3351_ = lean_ctor_get(v_a_3350_, 0);
lean_inc(v_fst_3351_);
v_snd_3352_ = lean_ctor_get(v_a_3350_, 1);
lean_inc(v_snd_3352_);
lean_dec(v_a_3350_);
v_numeratorTermPerAir_3353_ = lean_ctor_get(v_batchProof_3283_, 0);
lean_inc(v_numeratorTermPerAir_3353_);
v_denominatorTermPerAir_3354_ = lean_ctor_get(v_batchProof_3283_, 1);
lean_inc(v_denominatorTermPerAir_3354_);
v_univariateRoundCoeffs_3355_ = lean_ctor_get(v_batchProof_3283_, 2);
lean_inc(v_univariateRoundCoeffs_3355_);
v_sumcheckRoundPolys_3356_ = lean_ctor_get(v_batchProof_3283_, 3);
lean_inc(v_sumcheckRoundPolys_3356_);
v_columnOpenings_3357_ = lean_ctor_get(v_batchProof_3283_, 4);
lean_inc(v_columnOpenings_3357_);
lean_dec_ref(v_batchProof_3283_);
v___x_3358_ = l_List_lengthTR___redArg(v_numeratorTermPerAir_3353_);
v___x_3359_ = l_List_lengthTR___redArg(v_denominatorTermPerAir_3354_);
v___x_3360_ = lean_nat_dec_eq(v___x_3358_, v___x_3359_);
lean_dec(v___x_3359_);
lean_dec(v___x_3358_);
if (v___x_3360_ == 0)
{
lean_object* v___x_3361_; 
lean_dec(v_columnOpenings_3357_);
lean_dec(v_sumcheckRoundPolys_3356_);
lean_dec(v_univariateRoundCoeffs_3355_);
lean_dec(v_denominatorTermPerAir_3354_);
lean_dec(v_numeratorTermPerAir_3353_);
lean_dec(v_snd_3352_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3326_);
lean_dec(v___y_3325_);
lean_dec(v___y_3324_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3317_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_fst_3308_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v___x_3361_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3361_;
}
else
{
lean_object* v___x_3362_; lean_object* v___x_9704__overap_3363_; lean_object* v___x_3364_; 
lean_inc(v_denominatorTermPerAir_3354_);
lean_inc(v_numeratorTermPerAir_3353_);
v___x_3362_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_numeratorTermPerAir_3353_, v_denominatorTermPerAir_3354_);
v___x_9704__overap_3363_ = l_List_foldlM___redArg(v___x_3291_, v___f_3317_, v___x_3332_, v___x_3362_);
v___x_3364_ = lean_apply_1(v___x_9704__overap_3363_, v_snd_3352_);
if (lean_obj_tag(v___x_3364_) == 0)
{
lean_object* v_a_3365_; lean_object* v___x_3367_; uint8_t v_isShared_3368_; uint8_t v_isSharedCheck_3372_; 
lean_dec(v_columnOpenings_3357_);
lean_dec(v_sumcheckRoundPolys_3356_);
lean_dec(v_univariateRoundCoeffs_3355_);
lean_dec(v_denominatorTermPerAir_3354_);
lean_dec(v_numeratorTermPerAir_3353_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3326_);
lean_dec(v___y_3325_);
lean_dec(v___y_3324_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_fst_3308_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3365_ = lean_ctor_get(v___x_3364_, 0);
v_isSharedCheck_3372_ = !lean_is_exclusive(v___x_3364_);
if (v_isSharedCheck_3372_ == 0)
{
v___x_3367_ = v___x_3364_;
v_isShared_3368_ = v_isSharedCheck_3372_;
goto v_resetjp_3366_;
}
else
{
lean_inc(v_a_3365_);
lean_dec(v___x_3364_);
v___x_3367_ = lean_box(0);
v_isShared_3368_ = v_isSharedCheck_3372_;
goto v_resetjp_3366_;
}
v_resetjp_3366_:
{
lean_object* v___x_3370_; 
if (v_isShared_3368_ == 0)
{
v___x_3370_ = v___x_3367_;
goto v_reusejp_3369_;
}
else
{
lean_object* v_reuseFailAlloc_3371_; 
v_reuseFailAlloc_3371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3371_, 0, v_a_3365_);
v___x_3370_ = v_reuseFailAlloc_3371_;
goto v_reusejp_3369_;
}
v_reusejp_3369_:
{
return v___x_3370_;
}
}
}
else
{
lean_object* v_a_3373_; lean_object* v_snd_3374_; lean_object* v___x_3375_; 
v_a_3373_ = lean_ctor_get(v___x_3364_, 0);
lean_inc(v_a_3373_);
lean_dec_ref_known(v___x_3364_, 1);
v_snd_3374_ = lean_ctor_get(v_a_3373_, 1);
lean_inc(v_snd_3374_);
lean_dec(v_a_3373_);
lean_inc(v_denominatorTermPerAir_3354_);
lean_inc(v_numeratorTermPerAir_3353_);
lean_inc_ref(v_fo_3279_);
lean_inc_ref(v_inst_3274_);
v___x_3375_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(v_inst_3274_, v_fo_3279_, v_numeratorTermPerAir_3353_, v_denominatorTermPerAir_3354_, v_fst_3308_, v___y_3326_, v___y_3324_);
if (lean_obj_tag(v___x_3375_) == 0)
{
lean_object* v___x_3376_; 
lean_dec_ref_known(v___x_3375_, 1);
lean_dec(v_snd_3374_);
lean_dec(v_columnOpenings_3357_);
lean_dec(v_sumcheckRoundPolys_3356_);
lean_dec(v_univariateRoundCoeffs_3355_);
lean_dec(v_denominatorTermPerAir_3354_);
lean_dec(v_numeratorTermPerAir_3353_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3325_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v___x_3376_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3376_;
}
else
{
lean_object* v___x_3377_; lean_object* v_a_3378_; lean_object* v_fst_3379_; lean_object* v_snd_3380_; lean_object* v___x_3381_; 
lean_dec_ref_known(v___x_3375_, 1);
lean_inc_ref_n(v_inst_3278_, 2);
lean_inc_ref_n(v_inst_3276_, 2);
v___x_3377_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3276_, v_inst_3278_, v_snd_3374_);
v_a_3378_ = lean_ctor_get(v___x_3377_, 0);
lean_inc(v_a_3378_);
lean_dec_ref(v___x_3377_);
v_fst_3379_ = lean_ctor_get(v_a_3378_, 0);
lean_inc(v_fst_3379_);
v_snd_3380_ = lean_ctor_get(v_a_3378_, 1);
lean_inc(v_snd_3380_);
lean_dec(v_a_3378_);
lean_inc(v_univariateRoundCoeffs_3355_);
v___x_3381_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_observeExtList___redArg(v_inst_3276_, v_inst_3278_, v_univariateRoundCoeffs_3355_, v_snd_3380_);
if (lean_obj_tag(v___x_3381_) == 0)
{
lean_object* v_a_3382_; lean_object* v___x_3384_; uint8_t v_isShared_3385_; uint8_t v_isSharedCheck_3389_; 
lean_dec(v_fst_3379_);
lean_dec(v_columnOpenings_3357_);
lean_dec(v_sumcheckRoundPolys_3356_);
lean_dec(v_univariateRoundCoeffs_3355_);
lean_dec(v_denominatorTermPerAir_3354_);
lean_dec(v_numeratorTermPerAir_3353_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3325_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3382_ = lean_ctor_get(v___x_3381_, 0);
v_isSharedCheck_3389_ = !lean_is_exclusive(v___x_3381_);
if (v_isSharedCheck_3389_ == 0)
{
v___x_3384_ = v___x_3381_;
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
else
{
lean_inc(v_a_3382_);
lean_dec(v___x_3381_);
v___x_3384_ = lean_box(0);
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
v_resetjp_3383_:
{
lean_object* v___x_3387_; 
if (v_isShared_3385_ == 0)
{
v___x_3387_ = v___x_3384_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3388_; 
v_reuseFailAlloc_3388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3388_, 0, v_a_3382_);
v___x_3387_ = v_reuseFailAlloc_3388_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
return v___x_3387_;
}
}
}
else
{
lean_object* v_a_3390_; lean_object* v_snd_3391_; lean_object* v___x_3393_; uint8_t v_isShared_3394_; uint8_t v_isSharedCheck_3482_; 
v_a_3390_ = lean_ctor_get(v___x_3381_, 0);
lean_inc(v_a_3390_);
lean_dec_ref_known(v___x_3381_, 1);
v_snd_3391_ = lean_ctor_get(v_a_3390_, 1);
v_isSharedCheck_3482_ = !lean_is_exclusive(v_a_3390_);
if (v_isSharedCheck_3482_ == 0)
{
lean_object* v_unused_3483_; 
v_unused_3483_ = lean_ctor_get(v_a_3390_, 0);
lean_dec(v_unused_3483_);
v___x_3393_ = v_a_3390_;
v_isShared_3394_ = v_isSharedCheck_3482_;
goto v_resetjp_3392_;
}
else
{
lean_inc(v_snd_3391_);
lean_dec(v_a_3390_);
v___x_3393_ = lean_box(0);
v_isShared_3394_ = v_isSharedCheck_3482_;
goto v_resetjp_3392_;
}
v_resetjp_3392_:
{
lean_object* v___x_3395_; lean_object* v_a_3396_; lean_object* v_fst_3397_; lean_object* v_snd_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3481_; 
lean_inc_ref(v_inst_3278_);
lean_inc_ref(v_inst_3276_);
v___x_3395_ = lp_swirl_x2dfv_Fundamentals_Runtime_TranscriptM_sampleExt___redArg(v_inst_3276_, v_inst_3278_, v_snd_3391_);
v_a_3396_ = lean_ctor_get(v___x_3395_, 0);
lean_inc(v_a_3396_);
lean_dec_ref(v___x_3395_);
v_fst_3397_ = lean_ctor_get(v_a_3396_, 0);
v_snd_3398_ = lean_ctor_get(v_a_3396_, 1);
v_isSharedCheck_3481_ = !lean_is_exclusive(v_a_3396_);
if (v_isSharedCheck_3481_ == 0)
{
v___x_3400_ = v_a_3396_;
v_isShared_3401_ = v_isSharedCheck_3481_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_snd_3398_);
lean_inc(v_fst_3397_);
lean_dec(v_a_3396_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3481_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3402_; lean_object* v___x_3403_; 
lean_inc(v_fst_3379_);
lean_inc_ref_n(v_fo_3279_, 2);
v___x_3402_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(v_fo_3279_, v_numeratorTermPerAir_3353_, v_denominatorTermPerAir_3354_, v_fst_3379_);
lean_inc(v_fst_3397_);
lean_inc_ref(v_inst_3274_);
v___x_3403_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(v_inst_3274_, v_fo_3279_, v_univariateRoundCoeffs_3355_, v___x_3402_, v_lSkip_3314_, v_fst_3397_);
if (lean_obj_tag(v___x_3403_) == 0)
{
lean_object* v___x_3404_; 
lean_dec_ref_known(v___x_3403_, 1);
lean_del_object(v___x_3400_);
lean_dec(v_snd_3398_);
lean_dec(v_fst_3397_);
lean_del_object(v___x_3393_);
lean_dec(v_fst_3379_);
lean_dec(v_columnOpenings_3357_);
lean_dec(v_sumcheckRoundPolys_3356_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3325_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v___x_3404_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3404_;
}
else
{
lean_object* v_a_3405_; lean_object* v___x_3406_; uint8_t v___x_3407_; 
v_a_3405_ = lean_ctor_get(v___x_3403_, 0);
lean_inc(v_a_3405_);
lean_dec_ref_known(v___x_3403_, 1);
v___x_3406_ = l_List_lengthTR___redArg(v_sumcheckRoundPolys_3356_);
v___x_3407_ = lean_nat_dec_eq(v___x_3406_, v_nMax_3319_);
lean_dec(v___x_3406_);
if (v___x_3407_ == 0)
{
lean_object* v___x_3408_; 
lean_dec(v_a_3405_);
lean_del_object(v___x_3400_);
lean_dec(v_snd_3398_);
lean_dec(v_fst_3397_);
lean_del_object(v___x_3393_);
lean_dec(v_fst_3379_);
lean_dec(v_columnOpenings_3357_);
lean_dec(v_sumcheckRoundPolys_3356_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3325_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3316_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v___x_3408_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3408_;
}
else
{
lean_object* v___x_3410_; 
if (v_isShared_3394_ == 0)
{
lean_ctor_set_tag(v___x_3393_, 1);
lean_ctor_set(v___x_3393_, 1, v___x_3331_);
lean_ctor_set(v___x_3393_, 0, v_fst_3397_);
v___x_3410_ = v___x_3393_;
goto v_reusejp_3409_;
}
else
{
lean_object* v_reuseFailAlloc_3480_; 
v_reuseFailAlloc_3480_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3480_, 0, v_fst_3397_);
lean_ctor_set(v_reuseFailAlloc_3480_, 1, v___x_3331_);
v___x_3410_ = v_reuseFailAlloc_3480_;
goto v_reusejp_3409_;
}
v_reusejp_3409_:
{
lean_object* v___x_3412_; 
if (v_isShared_3401_ == 0)
{
lean_ctor_set(v___x_3400_, 1, v___x_3410_);
lean_ctor_set(v___x_3400_, 0, v_a_3405_);
v___x_3412_ = v___x_3400_;
goto v_reusejp_3411_;
}
else
{
lean_object* v_reuseFailAlloc_3479_; 
v_reuseFailAlloc_3479_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3479_, 0, v_a_3405_);
lean_ctor_set(v_reuseFailAlloc_3479_, 1, v___x_3410_);
v___x_3412_ = v_reuseFailAlloc_3479_;
goto v_reusejp_3411_;
}
v_reusejp_3411_:
{
lean_object* v___x_9788__overap_3413_; lean_object* v___x_3414_; 
v___x_9788__overap_3413_ = l_List_foldlM___redArg(v___x_3291_, v___f_3316_, v___x_3412_, v_sumcheckRoundPolys_3356_);
v___x_3414_ = lean_apply_1(v___x_9788__overap_3413_, v_snd_3398_);
if (lean_obj_tag(v___x_3414_) == 0)
{
lean_object* v_a_3415_; lean_object* v___x_3417_; uint8_t v_isShared_3418_; uint8_t v_isSharedCheck_3422_; 
lean_dec(v_fst_3379_);
lean_dec(v_columnOpenings_3357_);
lean_dec(v_fst_3351_);
lean_dec(v_fst_3348_);
lean_dec(v___y_3325_);
lean_dec(v___x_3321_);
lean_dec(v_nMax_3319_);
lean_dec_ref(v___f_3315_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec_ref(v_inst_3278_);
lean_dec_ref(v_inst_3276_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3415_ = lean_ctor_get(v___x_3414_, 0);
v_isSharedCheck_3422_ = !lean_is_exclusive(v___x_3414_);
if (v_isSharedCheck_3422_ == 0)
{
v___x_3417_ = v___x_3414_;
v_isShared_3418_ = v_isSharedCheck_3422_;
goto v_resetjp_3416_;
}
else
{
lean_inc(v_a_3415_);
lean_dec(v___x_3414_);
v___x_3417_ = lean_box(0);
v_isShared_3418_ = v_isSharedCheck_3422_;
goto v_resetjp_3416_;
}
v_resetjp_3416_:
{
lean_object* v___x_3420_; 
if (v_isShared_3418_ == 0)
{
v___x_3420_ = v___x_3417_;
goto v_reusejp_3419_;
}
else
{
lean_object* v_reuseFailAlloc_3421_; 
v_reuseFailAlloc_3421_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3421_, 0, v_a_3415_);
v___x_3420_ = v_reuseFailAlloc_3421_;
goto v_reusejp_3419_;
}
v_reusejp_3419_:
{
return v___x_3420_;
}
}
}
else
{
lean_object* v_a_3423_; lean_object* v_fst_3424_; lean_object* v_snd_3425_; lean_object* v_fst_3426_; lean_object* v_snd_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v_fst_3438_; lean_object* v_snd_3439_; lean_object* v___x_3440_; 
v_a_3423_ = lean_ctor_get(v___x_3414_, 0);
lean_inc(v_a_3423_);
lean_dec_ref_known(v___x_3414_, 1);
v_fst_3424_ = lean_ctor_get(v_a_3423_, 0);
lean_inc(v_fst_3424_);
v_snd_3425_ = lean_ctor_get(v_a_3423_, 1);
lean_inc(v_snd_3425_);
lean_dec(v_a_3423_);
v_fst_3426_ = lean_ctor_get(v_fst_3424_, 0);
lean_inc(v_fst_3426_);
v_snd_3427_ = lean_ctor_get(v_fst_3424_, 1);
lean_inc(v_snd_3427_);
lean_dec(v_fst_3424_);
v___x_3428_ = l_List_reverse___redArg(v_fst_3348_);
v___x_3429_ = l_List_appendTR___redArg(v___y_3325_, v___x_3428_);
v___x_3430_ = l_List_reverse___redArg(v_snd_3427_);
lean_inc_n(v_traceIdToAirId_3286_, 2);
v___x_3431_ = l_List_mapTR_loop___redArg(v___f_3315_, v_traceIdToAirId_3286_, v___x_3331_);
lean_inc(v_nPerTrace_3287_);
lean_inc_ref_n(v_fo_3279_, 4);
v___x_3432_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(v_fo_3279_, v_lSkip_3314_, v___x_3321_, v___x_3429_, v___x_3431_, v_nPerTrace_3287_);
lean_dec(v___x_3321_);
lean_inc(v_inst_3275_);
lean_inc_n(v_lSkip_3314_, 2);
v___x_3433_ = lean_apply_1(v_inst_3275_, v_lSkip_3314_);
v___x_3434_ = lean_unsigned_to_nat(2u);
v___x_3435_ = lean_nat_pow(v___x_3434_, v_lSkip_3314_);
v___x_3436_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_3279_, v___x_3433_, v___x_3435_);
lean_inc(v___x_3430_);
v___x_3437_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(v_fo_3279_, v_lSkip_3314_, v_nMax_3319_, v___x_3429_, v___x_3430_, v___x_3436_);
v_fst_3438_ = lean_ctor_get(v___x_3437_, 0);
lean_inc(v_fst_3438_);
v_snd_3439_ = lean_ctor_get(v___x_3437_, 1);
lean_inc(v_snd_3439_);
lean_dec_ref(v___x_3437_);
lean_inc(v_columnOpenings_3357_);
lean_inc_ref(v_vk_3281_);
v___x_3440_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(v_inst_3276_, v_inst_3278_, v_fo_3279_, v_vk_3281_, v_columnOpenings_3357_, v_traceIdToAirId_3286_, v_snd_3425_);
if (lean_obj_tag(v___x_3440_) == 0)
{
lean_object* v_a_3441_; lean_object* v___x_3443_; uint8_t v_isShared_3444_; uint8_t v_isSharedCheck_3448_; 
lean_dec(v_snd_3439_);
lean_dec(v_fst_3438_);
lean_dec(v___x_3432_);
lean_dec(v___x_3430_);
lean_dec(v_fst_3426_);
lean_dec(v_fst_3379_);
lean_dec(v_columnOpenings_3357_);
lean_dec(v_fst_3351_);
lean_dec(v_lSkip_3314_);
lean_dec(v_fst_3312_);
lean_dec(v_nPerTrace_3287_);
lean_dec(v_traceIdToAirId_3286_);
lean_dec(v_publicValues_3285_);
lean_dec_ref(v_vk_3281_);
lean_dec(v_algMap_3280_);
lean_dec_ref(v_fo_3279_);
lean_dec(v_inst_3275_);
lean_dec_ref(v_inst_3274_);
v_a_3441_ = lean_ctor_get(v___x_3440_, 0);
v_isSharedCheck_3448_ = !lean_is_exclusive(v___x_3440_);
if (v_isSharedCheck_3448_ == 0)
{
v___x_3443_ = v___x_3440_;
v_isShared_3444_ = v_isSharedCheck_3448_;
goto v_resetjp_3442_;
}
else
{
lean_inc(v_a_3441_);
lean_dec(v___x_3440_);
v___x_3443_ = lean_box(0);
v_isShared_3444_ = v_isSharedCheck_3448_;
goto v_resetjp_3442_;
}
v_resetjp_3442_:
{
lean_object* v___x_3446_; 
if (v_isShared_3444_ == 0)
{
v___x_3446_ = v___x_3443_;
goto v_reusejp_3445_;
}
else
{
lean_object* v_reuseFailAlloc_3447_; 
v_reuseFailAlloc_3447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3447_, 0, v_a_3441_);
v___x_3446_ = v_reuseFailAlloc_3447_;
goto v_reusejp_3445_;
}
v_reusejp_3445_:
{
return v___x_3446_;
}
}
}
else
{
lean_object* v_a_3449_; lean_object* v_snd_3450_; lean_object* v___x_3452_; uint8_t v_isShared_3453_; uint8_t v_isSharedCheck_3477_; 
v_a_3449_ = lean_ctor_get(v___x_3440_, 0);
lean_inc(v_a_3449_);
lean_dec_ref_known(v___x_3440_, 1);
v_snd_3450_ = lean_ctor_get(v_a_3449_, 1);
v_isSharedCheck_3477_ = !lean_is_exclusive(v_a_3449_);
if (v_isSharedCheck_3477_ == 0)
{
lean_object* v_unused_3478_; 
v_unused_3478_ = lean_ctor_get(v_a_3449_, 0);
lean_dec(v_unused_3478_);
v___x_3452_ = v_a_3449_;
v_isShared_3453_ = v_isSharedCheck_3477_;
goto v_resetjp_3451_;
}
else
{
lean_inc(v_snd_3450_);
lean_dec(v_a_3449_);
v___x_3452_ = lean_box(0);
v_isShared_3453_ = v_isSharedCheck_3477_;
goto v_resetjp_3451_;
}
v_resetjp_3451_:
{
lean_object* v___x_3454_; 
lean_inc(v___x_3430_);
v___x_3454_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(v_fo_3279_, v_algMap_3280_, v_inst_3275_, v_vk_3281_, v_columnOpenings_3357_, v_publicValues_3285_, v_traceIdToAirId_3286_, v_nPerTrace_3287_, v_lSkip_3314_, v___x_3430_, v_fst_3351_, v_fst_3379_, v_fst_3312_, v___x_3432_, v_fst_3438_, v_snd_3439_);
if (lean_obj_tag(v___x_3454_) == 0)
{
lean_object* v_a_3455_; lean_object* v___x_3457_; uint8_t v_isShared_3458_; uint8_t v_isSharedCheck_3462_; 
lean_del_object(v___x_3452_);
lean_dec(v_snd_3450_);
lean_dec(v___x_3430_);
lean_dec(v_fst_3426_);
lean_dec_ref(v_inst_3274_);
v_a_3455_ = lean_ctor_get(v___x_3454_, 0);
v_isSharedCheck_3462_ = !lean_is_exclusive(v___x_3454_);
if (v_isSharedCheck_3462_ == 0)
{
v___x_3457_ = v___x_3454_;
v_isShared_3458_ = v_isSharedCheck_3462_;
goto v_resetjp_3456_;
}
else
{
lean_inc(v_a_3455_);
lean_dec(v___x_3454_);
v___x_3457_ = lean_box(0);
v_isShared_3458_ = v_isSharedCheck_3462_;
goto v_resetjp_3456_;
}
v_resetjp_3456_:
{
lean_object* v___x_3460_; 
if (v_isShared_3458_ == 0)
{
v___x_3460_ = v___x_3457_;
goto v_reusejp_3459_;
}
else
{
lean_object* v_reuseFailAlloc_3461_; 
v_reuseFailAlloc_3461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3461_, 0, v_a_3455_);
v___x_3460_ = v_reuseFailAlloc_3461_;
goto v_reusejp_3459_;
}
v_reusejp_3459_:
{
return v___x_3460_;
}
}
}
else
{
lean_object* v_a_3463_; lean_object* v___x_3465_; uint8_t v_isShared_3466_; uint8_t v_isSharedCheck_3476_; 
v_a_3463_ = lean_ctor_get(v___x_3454_, 0);
v_isSharedCheck_3476_ = !lean_is_exclusive(v___x_3454_);
if (v_isSharedCheck_3476_ == 0)
{
v___x_3465_ = v___x_3454_;
v_isShared_3466_ = v_isSharedCheck_3476_;
goto v_resetjp_3464_;
}
else
{
lean_inc(v_a_3463_);
lean_dec(v___x_3454_);
v___x_3465_ = lean_box(0);
v_isShared_3466_ = v_isSharedCheck_3476_;
goto v_resetjp_3464_;
}
v_resetjp_3464_:
{
lean_object* v___x_3467_; uint8_t v___x_3468_; 
v___x_3467_ = lean_apply_2(v_inst_3274_, v_fst_3426_, v_a_3463_);
v___x_3468_ = lean_unbox(v___x_3467_);
if (v___x_3468_ == 0)
{
lean_object* v___x_3469_; 
lean_del_object(v___x_3465_);
lean_del_object(v___x_3452_);
lean_dec(v_snd_3450_);
lean_dec(v___x_3430_);
v___x_3469_ = ((lean_object*)(lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3469_;
}
else
{
lean_object* v___x_3471_; 
if (v_isShared_3453_ == 0)
{
lean_ctor_set(v___x_3452_, 0, v___x_3430_);
v___x_3471_ = v___x_3452_;
goto v_reusejp_3470_;
}
else
{
lean_object* v_reuseFailAlloc_3475_; 
v_reuseFailAlloc_3475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3475_, 0, v___x_3430_);
lean_ctor_set(v_reuseFailAlloc_3475_, 1, v_snd_3450_);
v___x_3471_ = v_reuseFailAlloc_3475_;
goto v_reusejp_3470_;
}
v_reusejp_3470_:
{
lean_object* v___x_3473_; 
if (v_isShared_3466_ == 0)
{
lean_ctor_set(v___x_3465_, 0, v___x_3471_);
v___x_3473_ = v___x_3465_;
goto v_reusejp_3472_;
}
else
{
lean_object* v_reuseFailAlloc_3474_; 
v_reuseFailAlloc_3474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3474_, 0, v___x_3471_);
v___x_3473_ = v_reuseFailAlloc_3474_;
goto v_reusejp_3472_;
}
v_reusejp_3472_:
{
return v___x_3473_;
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
v___jp_3484_:
{
uint8_t v___x_3489_; 
v___x_3489_ = lean_nat_dec_le(v_nMax_3319_, v___x_3321_);
if (v___x_3489_ == 0)
{
lean_inc(v_nMax_3319_);
v___y_3323_ = v___y_3488_;
v___y_3324_ = v_fst_3486_;
v___y_3325_ = v_snd_3487_;
v___y_3326_ = v_fst_3485_;
v___y_3327_ = v_nMax_3319_;
goto v___jp_3322_;
}
else
{
lean_inc(v___x_3321_);
v___y_3323_ = v___y_3488_;
v___y_3324_ = v_fst_3486_;
v___y_3325_ = v_snd_3487_;
v___y_3326_ = v_fst_3485_;
v___y_3327_ = v___x_3321_;
goto v___jp_3322_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___boxed(lean_object* v_inst_3516_, lean_object* v_inst_3517_, lean_object* v_inst_3518_, lean_object* v_inst_3519_, lean_object* v_inst_3520_, lean_object* v_fo_3521_, lean_object* v_algMap_3522_, lean_object* v_vk_3523_, lean_object* v_gkrProof_3524_, lean_object* v_batchProof_3525_, lean_object* v_traceVdata_3526_, lean_object* v_publicValues_3527_, lean_object* v_traceIdToAirId_3528_, lean_object* v_nPerTrace_3529_, lean_object* v_a_3530_){
_start:
{
lean_object* v_res_3531_; 
v_res_3531_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(v_inst_3516_, v_inst_3517_, v_inst_3518_, v_inst_3519_, v_inst_3520_, v_fo_3521_, v_algMap_3522_, v_vk_3523_, v_gkrProof_3524_, v_batchProof_3525_, v_traceVdata_3526_, v_publicValues_3527_, v_traceIdToAirId_3528_, v_nPerTrace_3529_, v_a_3530_);
lean_dec(v_traceVdata_3526_);
return v_res_3531_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM(lean_object* v_F_3532_, lean_object* v_EF_3533_, lean_object* v_Digest_3534_, lean_object* v_inst_3535_, lean_object* v_inst_3536_, lean_object* v_inst_3537_, lean_object* v_inst_3538_, lean_object* v_inst_3539_, lean_object* v_fo_3540_, lean_object* v_algMap_3541_, lean_object* v_vk_3542_, lean_object* v_gkrProof_3543_, lean_object* v_batchProof_3544_, lean_object* v_traceVdata_3545_, lean_object* v_publicValues_3546_, lean_object* v_traceIdToAirId_3547_, lean_object* v_nPerTrace_3548_, lean_object* v_a_3549_){
_start:
{
lean_object* v___x_3550_; 
v___x_3550_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(v_inst_3535_, v_inst_3536_, v_inst_3537_, v_inst_3538_, v_inst_3539_, v_fo_3540_, v_algMap_3541_, v_vk_3542_, v_gkrProof_3543_, v_batchProof_3544_, v_traceVdata_3545_, v_publicValues_3546_, v_traceIdToAirId_3547_, v_nPerTrace_3548_, v_a_3549_);
return v___x_3550_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___boxed(lean_object** _args){
lean_object* v_F_3551_ = _args[0];
lean_object* v_EF_3552_ = _args[1];
lean_object* v_Digest_3553_ = _args[2];
lean_object* v_inst_3554_ = _args[3];
lean_object* v_inst_3555_ = _args[4];
lean_object* v_inst_3556_ = _args[5];
lean_object* v_inst_3557_ = _args[6];
lean_object* v_inst_3558_ = _args[7];
lean_object* v_fo_3559_ = _args[8];
lean_object* v_algMap_3560_ = _args[9];
lean_object* v_vk_3561_ = _args[10];
lean_object* v_gkrProof_3562_ = _args[11];
lean_object* v_batchProof_3563_ = _args[12];
lean_object* v_traceVdata_3564_ = _args[13];
lean_object* v_publicValues_3565_ = _args[14];
lean_object* v_traceIdToAirId_3566_ = _args[15];
lean_object* v_nPerTrace_3567_ = _args[16];
lean_object* v_a_3568_ = _args[17];
_start:
{
lean_object* v_res_3569_; 
v_res_3569_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM(v_F_3551_, v_EF_3552_, v_Digest_3553_, v_inst_3554_, v_inst_3555_, v_inst_3556_, v_inst_3557_, v_inst_3558_, v_fo_3559_, v_algMap_3560_, v_vk_3561_, v_gkrProof_3562_, v_batchProof_3563_, v_traceVdata_3564_, v_publicValues_3565_, v_traceIdToAirId_3566_, v_nPerTrace_3567_, v_a_3568_);
lean_dec(v_traceVdata_3564_);
return v_res_3569_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(lean_object* v_inst_3570_, lean_object* v_inst_3571_, lean_object* v_inst_3572_, lean_object* v_inst_3573_, lean_object* v_inst_3574_, lean_object* v_fo_3575_, lean_object* v_algMap_3576_, lean_object* v_transcript_3577_, lean_object* v_vk_3578_, lean_object* v_publicValues_3579_, lean_object* v_gkrProof_3580_, lean_object* v_batchProof_3581_, lean_object* v_traceVdata_3582_, lean_object* v_traceIdToAirId_3583_, lean_object* v_nPerTrace_3584_){
_start:
{
lean_object* v___x_3585_; 
v___x_3585_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(v_inst_3570_, v_inst_3571_, v_inst_3572_, v_inst_3573_, v_inst_3574_, v_fo_3575_, v_algMap_3576_, v_vk_3578_, v_gkrProof_3580_, v_batchProof_3581_, v_traceVdata_3582_, v_publicValues_3579_, v_traceIdToAirId_3583_, v_nPerTrace_3584_, v_transcript_3577_);
if (lean_obj_tag(v___x_3585_) == 0)
{
lean_object* v_a_3586_; lean_object* v___x_3588_; uint8_t v_isShared_3589_; uint8_t v_isSharedCheck_3593_; 
v_a_3586_ = lean_ctor_get(v___x_3585_, 0);
v_isSharedCheck_3593_ = !lean_is_exclusive(v___x_3585_);
if (v_isSharedCheck_3593_ == 0)
{
v___x_3588_ = v___x_3585_;
v_isShared_3589_ = v_isSharedCheck_3593_;
goto v_resetjp_3587_;
}
else
{
lean_inc(v_a_3586_);
lean_dec(v___x_3585_);
v___x_3588_ = lean_box(0);
v_isShared_3589_ = v_isSharedCheck_3593_;
goto v_resetjp_3587_;
}
v_resetjp_3587_:
{
lean_object* v___x_3591_; 
if (v_isShared_3589_ == 0)
{
v___x_3591_ = v___x_3588_;
goto v_reusejp_3590_;
}
else
{
lean_object* v_reuseFailAlloc_3592_; 
v_reuseFailAlloc_3592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3592_, 0, v_a_3586_);
v___x_3591_ = v_reuseFailAlloc_3592_;
goto v_reusejp_3590_;
}
v_reusejp_3590_:
{
return v___x_3591_;
}
}
}
else
{
lean_object* v_a_3594_; lean_object* v___x_3596_; uint8_t v_isShared_3597_; uint8_t v_isSharedCheck_3610_; 
v_a_3594_ = lean_ctor_get(v___x_3585_, 0);
v_isSharedCheck_3610_ = !lean_is_exclusive(v___x_3585_);
if (v_isSharedCheck_3610_ == 0)
{
v___x_3596_ = v___x_3585_;
v_isShared_3597_ = v_isSharedCheck_3610_;
goto v_resetjp_3595_;
}
else
{
lean_inc(v_a_3594_);
lean_dec(v___x_3585_);
v___x_3596_ = lean_box(0);
v_isShared_3597_ = v_isSharedCheck_3610_;
goto v_resetjp_3595_;
}
v_resetjp_3595_:
{
lean_object* v_fst_3598_; lean_object* v_snd_3599_; lean_object* v___x_3601_; uint8_t v_isShared_3602_; uint8_t v_isSharedCheck_3609_; 
v_fst_3598_ = lean_ctor_get(v_a_3594_, 0);
v_snd_3599_ = lean_ctor_get(v_a_3594_, 1);
v_isSharedCheck_3609_ = !lean_is_exclusive(v_a_3594_);
if (v_isSharedCheck_3609_ == 0)
{
v___x_3601_ = v_a_3594_;
v_isShared_3602_ = v_isSharedCheck_3609_;
goto v_resetjp_3600_;
}
else
{
lean_inc(v_snd_3599_);
lean_inc(v_fst_3598_);
lean_dec(v_a_3594_);
v___x_3601_ = lean_box(0);
v_isShared_3602_ = v_isSharedCheck_3609_;
goto v_resetjp_3600_;
}
v_resetjp_3600_:
{
lean_object* v___x_3604_; 
if (v_isShared_3602_ == 0)
{
v___x_3604_ = v___x_3601_;
goto v_reusejp_3603_;
}
else
{
lean_object* v_reuseFailAlloc_3608_; 
v_reuseFailAlloc_3608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3608_, 0, v_fst_3598_);
lean_ctor_set(v_reuseFailAlloc_3608_, 1, v_snd_3599_);
v___x_3604_ = v_reuseFailAlloc_3608_;
goto v_reusejp_3603_;
}
v_reusejp_3603_:
{
lean_object* v___x_3606_; 
if (v_isShared_3597_ == 0)
{
lean_ctor_set(v___x_3596_, 0, v___x_3604_);
v___x_3606_ = v___x_3596_;
goto v_reusejp_3605_;
}
else
{
lean_object* v_reuseFailAlloc_3607_; 
v_reuseFailAlloc_3607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3607_, 0, v___x_3604_);
v___x_3606_ = v_reuseFailAlloc_3607_;
goto v_reusejp_3605_;
}
v_reusejp_3605_:
{
return v___x_3606_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg___boxed(lean_object* v_inst_3611_, lean_object* v_inst_3612_, lean_object* v_inst_3613_, lean_object* v_inst_3614_, lean_object* v_inst_3615_, lean_object* v_fo_3616_, lean_object* v_algMap_3617_, lean_object* v_transcript_3618_, lean_object* v_vk_3619_, lean_object* v_publicValues_3620_, lean_object* v_gkrProof_3621_, lean_object* v_batchProof_3622_, lean_object* v_traceVdata_3623_, lean_object* v_traceIdToAirId_3624_, lean_object* v_nPerTrace_3625_){
_start:
{
lean_object* v_res_3626_; 
v_res_3626_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(v_inst_3611_, v_inst_3612_, v_inst_3613_, v_inst_3614_, v_inst_3615_, v_fo_3616_, v_algMap_3617_, v_transcript_3618_, v_vk_3619_, v_publicValues_3620_, v_gkrProof_3621_, v_batchProof_3622_, v_traceVdata_3623_, v_traceIdToAirId_3624_, v_nPerTrace_3625_);
lean_dec(v_traceVdata_3623_);
return v_res_3626_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify(lean_object* v_F_3627_, lean_object* v_EF_3628_, lean_object* v_Digest_3629_, lean_object* v_inst_3630_, lean_object* v_inst_3631_, lean_object* v_inst_3632_, lean_object* v_inst_3633_, lean_object* v_inst_3634_, lean_object* v_fo_3635_, lean_object* v_algMap_3636_, lean_object* v_transcript_3637_, lean_object* v_vk_3638_, lean_object* v_publicValues_3639_, lean_object* v_gkrProof_3640_, lean_object* v_batchProof_3641_, lean_object* v_traceVdata_3642_, lean_object* v_traceIdToAirId_3643_, lean_object* v_nPerTrace_3644_){
_start:
{
lean_object* v___x_3645_; 
v___x_3645_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(v_inst_3630_, v_inst_3631_, v_inst_3632_, v_inst_3633_, v_inst_3634_, v_fo_3635_, v_algMap_3636_, v_transcript_3637_, v_vk_3638_, v_publicValues_3639_, v_gkrProof_3640_, v_batchProof_3641_, v_traceVdata_3642_, v_traceIdToAirId_3643_, v_nPerTrace_3644_);
return v___x_3645_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___boxed(lean_object** _args){
lean_object* v_F_3646_ = _args[0];
lean_object* v_EF_3647_ = _args[1];
lean_object* v_Digest_3648_ = _args[2];
lean_object* v_inst_3649_ = _args[3];
lean_object* v_inst_3650_ = _args[4];
lean_object* v_inst_3651_ = _args[5];
lean_object* v_inst_3652_ = _args[6];
lean_object* v_inst_3653_ = _args[7];
lean_object* v_fo_3654_ = _args[8];
lean_object* v_algMap_3655_ = _args[9];
lean_object* v_transcript_3656_ = _args[10];
lean_object* v_vk_3657_ = _args[11];
lean_object* v_publicValues_3658_ = _args[12];
lean_object* v_gkrProof_3659_ = _args[13];
lean_object* v_batchProof_3660_ = _args[14];
lean_object* v_traceVdata_3661_ = _args[15];
lean_object* v_traceIdToAirId_3662_ = _args[16];
lean_object* v_nPerTrace_3663_ = _args[17];
_start:
{
lean_object* v_res_3664_; 
v_res_3664_ = lp_swirl_x2dfv_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify(v_F_3646_, v_EF_3647_, v_Digest_3648_, v_inst_3649_, v_inst_3650_, v_inst_3651_, v_inst_3652_, v_inst_3653_, v_fo_3654_, v_algMap_3655_, v_transcript_3656_, v_vk_3657_, v_publicValues_3658_, v_gkrProof_3659_, v_batchProof_3660_, v_traceVdata_3661_, v_traceIdToAirId_3662_, v_nPerTrace_3663_);
lean_dec(v_traceVdata_3661_);
return v_res_3664_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Core(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Ops(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Shape(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Batch(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Swirl_Spec_ReferenceVerifier_Verifier_Runtime_Shape(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
