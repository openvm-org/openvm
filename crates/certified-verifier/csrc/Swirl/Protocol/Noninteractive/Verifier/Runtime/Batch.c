// Lean compiler output
// Module: Swirl.Protocol.Noninteractive.Verifier.Runtime.Batch
// Imports: public import Init public meta import Init public import Swirl.Protocol.Noninteractive.Runtime.Core public import Swirl.Protocol.Noninteractive.Ops public import Swirl.Protocol.Noninteractive.Verifier.Runtime.Shape
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
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_logupPowBits(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observePowWitness___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_totalInteractions___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_calculateNLogup(lean_object*, lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_SymbolicEvaluator_evalExpr___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_SymbolicEvaluator_evalNodes___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "values"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "numeratorClaim"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "denominatorClaim"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "xi"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "claim"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "point"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "eqAtPoint"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput(lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodes___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__4_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__5_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__6_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__7_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__4, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__7, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__9, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_map, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__14_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_pure, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__15_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__16_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__11_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__12_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_bind, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__17_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__18_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___boxed(lean_object**);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___boxed(lean_object**);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_instDecidableEqList___redArg(v_inst_1_, v_x_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg___boxed(lean_object* v_inst_5_, lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___redArg(v_inst_5_, v_x_6_, v_x_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq(lean_object* v_EF_10_, lean_object* v_inst_11_, lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
uint8_t v___x_14_; 
v___x_14_ = l_instDecidableEqList___redArg(v_inst_11_, v_x_12_, v_x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq___boxed(lean_object* v_EF_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput_decEq(v_EF_15_, v_inst_16_, v_x_17_, v_x_18_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg(lean_object* v_inst_21_, lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
uint8_t v___x_24_; 
v___x_24_ = l_instDecidableEqList___redArg(v_inst_21_, v_x_22_, v_x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg___boxed(lean_object* v_inst_25_, lean_object* v_x_26_, lean_object* v_x_27_){
_start:
{
uint8_t v_res_28_; lean_object* v_r_29_; 
v_res_28_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___redArg(v_inst_25_, v_x_26_, v_x_27_);
v_r_29_ = lean_box(v_res_28_);
return v_r_29_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput(lean_object* v_EF_30_, lean_object* v_inst_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v___x_34_; 
v___x_34_ = l_instDecidableEqList___redArg(v_inst_31_, v_x_32_, v_x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput___boxed(lean_object* v_EF_35_, lean_object* v_inst_36_, lean_object* v_x_37_, lean_object* v_x_38_){
_start:
{
uint8_t v_res_39_; lean_object* v_r_40_; 
v_res_39_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqOutput(v_EF_35_, v_inst_36_, v_x_37_, v_x_38_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = lean_unsigned_to_nat(10u);
v___x_55_ = lean_nat_to_int(v___x_54_);
return v___x_55_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__0));
v___x_58_ = lean_string_length(v___x_57_);
return v___x_58_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__9);
v___x_60_ = lean_nat_to_int(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg(lean_object* v_inst_65_, lean_object* v_x_66_){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; uint8_t v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_67_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__6));
v___x_68_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__7);
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
v___x_74_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10);
v___x_75_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11));
v___x_76_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_73_);
v___x_77_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12));
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr(lean_object* v_EF_81_, lean_object* v_inst_82_, lean_object* v_x_83_, lean_object* v_prec_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg(v_inst_82_, v_x_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed(lean_object* v_EF_86_, lean_object* v_inst_87_, lean_object* v_x_88_, lean_object* v_prec_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr(v_EF_86_, v_inst_87_, v_x_88_, v_prec_89_);
lean_dec(v_prec_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput___redArg(lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_92_, 0, lean_box(0));
lean_closure_set(v___x_92_, 1, v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput(lean_object* v_EF_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_95_, 0, lean_box(0));
lean_closure_set(v___x_95_, 1, v_inst_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(lean_object* v_traceVdata_96_, lean_object* v_lSkip_97_, lean_object* v_x_98_, lean_object* v_x_99_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg___boxed(lean_object* v_traceVdata_112_, lean_object* v_lSkip_113_, lean_object* v_x_114_, lean_object* v_x_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(v_traceVdata_112_, v_lSkip_113_, v_x_114_, v_x_115_);
lean_dec(v_lSkip_113_);
lean_dec(v_traceVdata_112_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(lean_object* v_lSkip_117_, lean_object* v_traceVdata_118_, lean_object* v_traceIdToAirId_119_){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lean_unsigned_to_nat(0u);
v___x_121_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(v_traceVdata_118_, v_lSkip_117_, v___x_120_, v_traceIdToAirId_119_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg___boxed(lean_object* v_lSkip_122_, lean_object* v_traceVdata_123_, lean_object* v_traceIdToAirId_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(v_lSkip_122_, v_traceVdata_123_, v_traceIdToAirId_124_);
lean_dec(v_traceVdata_123_);
lean_dec(v_lSkip_122_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax(lean_object* v_Digest_126_, lean_object* v_lSkip_127_, lean_object* v_traceVdata_128_, lean_object* v_traceIdToAirId_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(v_lSkip_127_, v_traceVdata_128_, v_traceIdToAirId_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___boxed(lean_object* v_Digest_131_, lean_object* v_lSkip_132_, lean_object* v_traceVdata_133_, lean_object* v_traceIdToAirId_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax(v_Digest_131_, v_lSkip_132_, v_traceVdata_133_, v_traceIdToAirId_134_);
lean_dec(v_traceVdata_133_);
lean_dec(v_lSkip_132_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0(lean_object* v_Digest_136_, lean_object* v_traceVdata_137_, lean_object* v_lSkip_138_, lean_object* v_x_139_, lean_object* v_x_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___redArg(v_traceVdata_137_, v_lSkip_138_, v_x_139_, v_x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0___boxed(lean_object* v_Digest_142_, lean_object* v_traceVdata_143_, lean_object* v_lSkip_144_, lean_object* v_x_145_, lean_object* v_x_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax_spec__0(v_Digest_142_, v_traceVdata_143_, v_lSkip_144_, v_x_145_, v_x_146_);
lean_dec(v_lSkip_144_);
lean_dec(v_traceVdata_143_);
return v_res_147_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(lean_object* v_inst_148_, lean_object* v_x_149_, lean_object* v_x_150_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg___boxed(lean_object* v_inst_164_, lean_object* v_x_165_, lean_object* v_x_166_){
_start:
{
uint8_t v_res_167_; lean_object* v_r_168_; 
v_res_167_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_164_, v_x_165_, v_x_166_);
v_r_168_ = lean_box(v_res_167_);
return v_r_168_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq(lean_object* v_EF_169_, lean_object* v_inst_170_, lean_object* v_x_171_, lean_object* v_x_172_){
_start:
{
uint8_t v___x_173_; 
v___x_173_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_170_, v_x_171_, v_x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___boxed(lean_object* v_EF_174_, lean_object* v_inst_175_, lean_object* v_x_176_, lean_object* v_x_177_){
_start:
{
uint8_t v_res_178_; lean_object* v_r_179_; 
v_res_178_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq(v_EF_174_, v_inst_175_, v_x_176_, v_x_177_);
v_r_179_ = lean_box(v_res_178_);
return v_r_179_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg(lean_object* v_inst_180_, lean_object* v_x_181_, lean_object* v_x_182_){
_start:
{
uint8_t v___x_183_; 
v___x_183_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_180_, v_x_181_, v_x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg___boxed(lean_object* v_inst_184_, lean_object* v_x_185_, lean_object* v_x_186_){
_start:
{
uint8_t v_res_187_; lean_object* v_r_188_; 
v_res_187_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___redArg(v_inst_184_, v_x_185_, v_x_186_);
v_r_188_ = lean_box(v_res_187_);
return v_r_188_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput(lean_object* v_EF_189_, lean_object* v_inst_190_, lean_object* v_x_191_, lean_object* v_x_192_){
_start:
{
uint8_t v___x_193_; 
v___x_193_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput_decEq___redArg(v_inst_190_, v_x_191_, v_x_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput___boxed(lean_object* v_EF_194_, lean_object* v_inst_195_, lean_object* v_x_196_, lean_object* v_x_197_){
_start:
{
uint8_t v_res_198_; lean_object* v_r_199_; 
v_res_198_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrOutput(v_EF_194_, v_inst_195_, v_x_196_, v_x_197_);
v_r_199_ = lean_box(v_res_198_);
return v_r_199_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = lean_unsigned_to_nat(18u);
v___x_210_ = lean_nat_to_int(v___x_209_);
return v___x_210_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_unsigned_to_nat(20u);
v___x_218_ = lean_nat_to_int(v___x_217_);
return v___x_218_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_222_ = lean_unsigned_to_nat(6u);
v___x_223_ = lean_nat_to_int(v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg(lean_object* v_inst_224_, lean_object* v_x_225_){
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
v___x_229_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5));
v___x_230_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__3));
v___x_231_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__4);
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
v___x_238_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6));
v___x_239_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_237_);
lean_ctor_set(v___x_239_, 1, v___x_238_);
v___x_240_ = lean_box(1);
v___x_241_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_239_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__8));
v___x_243_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_241_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
v___x_244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v___x_229_);
v___x_245_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__9);
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
v___x_252_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__11));
v___x_253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_251_);
lean_ctor_set(v___x_253_, 1, v___x_252_);
v___x_254_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v___x_229_);
v___x_255_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__12);
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
v___x_260_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10);
v___x_261_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11));
v___x_262_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_259_);
v___x_263_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12));
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr(lean_object* v_EF_267_, lean_object* v_inst_268_, lean_object* v_x_269_, lean_object* v_prec_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg(v_inst_268_, v_x_269_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed(lean_object* v_EF_272_, lean_object* v_inst_273_, lean_object* v_x_274_, lean_object* v_prec_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr(v_EF_272_, v_inst_273_, v_x_274_, v_prec_275_);
lean_dec(v_prec_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput___redArg(lean_object* v_inst_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_278_, 0, lean_box(0));
lean_closure_set(v___x_278_, 1, v_inst_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput(lean_object* v_EF_279_, lean_object* v_inst_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_281_, 0, lean_box(0));
lean_closure_set(v___x_281_, 1, v_inst_280_);
return v___x_281_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(lean_object* v_inst_282_, lean_object* v_x_283_, lean_object* v_x_284_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg___boxed(lean_object* v_inst_297_, lean_object* v_x_298_, lean_object* v_x_299_){
_start:
{
uint8_t v_res_300_; lean_object* v_r_301_; 
v_res_300_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_297_, v_x_298_, v_x_299_);
v_r_301_ = lean_box(v_res_300_);
return v_r_301_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq(lean_object* v_EF_302_, lean_object* v_inst_303_, lean_object* v_x_304_, lean_object* v_x_305_){
_start:
{
uint8_t v___x_306_; 
v___x_306_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_303_, v_x_304_, v_x_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___boxed(lean_object* v_EF_307_, lean_object* v_inst_308_, lean_object* v_x_309_, lean_object* v_x_310_){
_start:
{
uint8_t v_res_311_; lean_object* v_r_312_; 
v_res_311_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq(v_EF_307_, v_inst_308_, v_x_309_, v_x_310_);
v_r_312_ = lean_box(v_res_311_);
return v_r_312_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg(lean_object* v_inst_313_, lean_object* v_x_314_, lean_object* v_x_315_){
_start:
{
uint8_t v___x_316_; 
v___x_316_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_313_, v_x_314_, v_x_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg___boxed(lean_object* v_inst_317_, lean_object* v_x_318_, lean_object* v_x_319_){
_start:
{
uint8_t v_res_320_; lean_object* v_r_321_; 
v_res_320_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___redArg(v_inst_317_, v_x_318_, v_x_319_);
v_r_321_ = lean_box(v_res_320_);
return v_r_321_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput(lean_object* v_EF_322_, lean_object* v_inst_323_, lean_object* v_x_324_, lean_object* v_x_325_){
_start:
{
uint8_t v___x_326_; 
v___x_326_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput_decEq___redArg(v_inst_323_, v_x_324_, v_x_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput___boxed(lean_object* v_EF_327_, lean_object* v_inst_328_, lean_object* v_x_329_, lean_object* v_x_330_){
_start:
{
uint8_t v_res_331_; lean_object* v_r_332_; 
v_res_331_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instDecidableEqGkrSumcheckOutput(v_EF_327_, v_inst_328_, v_x_329_, v_x_330_);
v_r_332_ = lean_box(v_res_331_);
return v_r_332_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_unsigned_to_nat(9u);
v___x_343_ = lean_nat_to_int(v___x_342_);
return v___x_343_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_350_ = lean_unsigned_to_nat(13u);
v___x_351_ = lean_nat_to_int(v___x_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg(lean_object* v_inst_352_, lean_object* v_x_353_){
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
v___x_357_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__5));
v___x_358_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__3));
v___x_359_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__4);
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
v___x_366_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrOutput_repr___redArg___closed__6));
v___x_367_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_367_, 0, v___x_365_);
lean_ctor_set(v___x_367_, 1, v___x_366_);
v___x_368_ = lean_box(1);
v___x_369_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_367_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__6));
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
v___x_379_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__8));
v___x_380_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_380_, 0, v___x_378_);
lean_ctor_set(v___x_380_, 1, v___x_379_);
v___x_381_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v___x_357_);
v___x_382_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg___closed__9);
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
v___x_387_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__10);
v___x_388_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__11));
v___x_389_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_388_);
lean_ctor_set(v___x_389_, 1, v___x_386_);
v___x_390_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprOutput_repr___redArg___closed__12));
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr(lean_object* v_EF_394_, lean_object* v_inst_395_, lean_object* v_x_396_, lean_object* v_prec_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___redArg(v_inst_395_, v_x_396_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed(lean_object* v_EF_399_, lean_object* v_inst_400_, lean_object* v_x_401_, lean_object* v_prec_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr(v_EF_399_, v_inst_400_, v_x_401_, v_prec_402_);
lean_dec(v_prec_402_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput___redArg(lean_object* v_inst_404_){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_405_, 0, lean_box(0));
lean_closure_set(v___x_405_, 1, v_inst_404_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput(lean_object* v_EF_406_, lean_object* v_inst_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_instReprGkrSumcheckOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_408_, 0, lean_box(0));
lean_closure_set(v___x_408_, 1, v_inst_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(lean_object* v_values_412_, lean_object* v_idx_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = l_List_get_x3fInternal___redArg(v_values_412_, v_idx_413_);
if (lean_obj_tag(v___x_414_) == 0)
{
lean_object* v___x_415_; 
v___x_415_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___boxed(lean_object* v_values_424_, lean_object* v_idx_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_424_, v_idx_425_);
lean_dec(v_values_424_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem(lean_object* v_00_u03b1_427_, lean_object* v_values_428_, lean_object* v_idx_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_428_, v_idx_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___boxed(lean_object* v_00_u03b1_431_, lean_object* v_values_432_, lean_object* v_idx_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem(v_00_u03b1_431_, v_values_432_, v_idx_433_);
lean_dec(v_values_432_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(lean_object* v_value_435_){
_start:
{
if (lean_obj_tag(v_value_435_) == 0)
{
lean_object* v___x_436_; 
v___x_436_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption(lean_object* v_00_u03b1_445_, lean_object* v_value_446_){
_start:
{
lean_object* v___x_447_; 
v___x_447_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v_value_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(lean_object* v_f_448_, lean_object* v_init_449_, lean_object* v_x_450_){
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
v___x_459_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v_f_448_, v___x_458_, v_tail_454_);
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl(lean_object* v_00_u03b1_464_, lean_object* v_00_u03b2_465_, lean_object* v_f_466_, lean_object* v_init_467_, lean_object* v_x_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v_f_466_, v_init_467_, v_x_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(lean_object* v_proof_470_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg___boxed(lean_object* v_proof_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(v_proof_475_);
lean_dec_ref(v_proof_475_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof(lean_object* v_F_477_, lean_object* v_EF_478_, lean_object* v_proof_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(v_proof_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___boxed(lean_object* v_F_481_, lean_object* v_EF_482_, lean_object* v_proof_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof(v_F_481_, v_EF_482_, v_proof_483_);
lean_dec_ref(v_proof_483_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(lean_object* v_lSkip_485_, lean_object* v_proof_486_){
_start:
{
lean_object* v_numGkrRounds_487_; lean_object* v___x_488_; uint8_t v___x_489_; 
v_numGkrRounds_487_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_numGkrRoundsFromGkrProof___redArg(v_proof_486_);
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg___boxed(lean_object* v_lSkip_491_, lean_object* v_proof_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(v_lSkip_491_, v_proof_492_);
lean_dec_ref(v_proof_492_);
lean_dec(v_lSkip_491_);
return v_res_493_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof(lean_object* v_F_494_, lean_object* v_EF_495_, lean_object* v_lSkip_496_, lean_object* v_proof_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___redArg(v_lSkip_496_, v_proof_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof___boxed(lean_object* v_F_499_, lean_object* v_EF_500_, lean_object* v_lSkip_501_, lean_object* v_proof_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_nLogupFromGkrProof(v_F_499_, v_EF_500_, v_lSkip_501_, v_proof_502_);
lean_dec_ref(v_proof_502_);
lean_dec(v_lSkip_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(lean_object* v_fo_504_, lean_object* v_claims_505_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations(lean_object* v_EF_526_, lean_object* v_fo_527_, lean_object* v_claims_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(v_fo_527_, v_claims_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(lean_object* v_fo_530_, lean_object* v_claims_531_, lean_object* v_mu_532_){
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
v___x_541_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(v_toRingOps_533_, v_pXi0_537_, v_pXi1_538_, v_mu_532_);
v___x_542_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(v_toRingOps_533_, v_qXi0_539_, v_qXi1_540_, v_mu_532_);
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation(lean_object* v_EF_548_, lean_object* v_fo_549_, lean_object* v_claims_550_, lean_object* v_mu_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(v_fo_549_, v_claims_550_, v_mu_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_claims_555_, lean_object* v_a_556_){
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
v___x_561_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_pXi0_557_, v_a_556_);
v_a_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc(v_a_562_);
lean_dec_ref(v___x_561_);
v_snd_563_ = lean_ctor_get(v_a_562_, 1);
lean_inc(v_snd_563_);
lean_dec(v_a_562_);
v___x_564_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_qXi0_559_, v_snd_563_);
v_a_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_a_565_);
lean_dec_ref(v___x_564_);
v_snd_566_ = lean_ctor_get(v_a_565_, 1);
lean_inc(v_snd_566_);
lean_dec(v_a_565_);
v___x_567_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_pXi1_558_, v_snd_566_);
v_a_568_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_a_568_);
lean_dec_ref(v___x_567_);
v_snd_569_ = lean_ctor_get(v_a_568_, 1);
lean_inc(v_snd_569_);
lean_dec(v_a_568_);
v___x_570_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_553_, v_inst_554_, v_qXi1_560_, v_snd_569_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM(lean_object* v_F_571_, lean_object* v_EF_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_claims_575_, lean_object* v_a_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(v_inst_573_, v_inst_574_, v_claims_575_, v_a_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_fo_583_, lean_object* v_claim_584_, lean_object* v_eqAtPoint_585_, lean_object* v_pointRev_586_, lean_object* v_gkrPoint_587_, lean_object* v_polys_588_, lean_object* v_a_589_){
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
v___x_605_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_581_, v_inst_582_, v_e1_604_, v_a_589_);
v_a_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_a_606_);
lean_dec_ref(v___x_605_);
v_snd_607_ = lean_ctor_get(v_a_606_, 1);
lean_inc(v_snd_607_);
lean_dec(v_a_606_);
v___x_608_ = lean_unsigned_to_nat(1u);
v_e2_609_ = lean_array_fget(v_head_598_, v___x_608_);
lean_inc(v_e2_609_);
v___x_610_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_581_, v_inst_582_, v_e2_609_, v_snd_607_);
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
v___x_615_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_581_, v_inst_582_, v_e3_614_, v_snd_612_);
v_a_616_ = lean_ctor_get(v___x_615_, 0);
lean_inc(v_a_616_);
lean_dec_ref(v___x_615_);
v_snd_617_ = lean_ctor_get(v_a_616_, 1);
lean_inc(v_snd_617_);
lean_dec(v_a_616_);
v___x_618_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_581_, v_inst_582_, v_snd_617_);
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
v___x_629_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(v_fo_583_, v___x_628_, v_e1_604_, v_e2_609_, v_e3_614_, v_fst_622_);
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
v___x_591_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0));
return v___x_591_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM(lean_object* v_F_641_, lean_object* v_EF_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_fo_645_, lean_object* v_claim_646_, lean_object* v_eqAtPoint_647_, lean_object* v_pointRev_648_, lean_object* v_gkrPoint_649_, lean_object* v_polys_650_, lean_object* v_a_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(v_inst_643_, v_inst_644_, v_fo_645_, v_claim_646_, v_eqAtPoint_647_, v_pointRev_648_, v_gkrPoint_649_, v_polys_650_, v_a_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_fo_655_, lean_object* v_proof_656_, lean_object* v_round_657_, lean_object* v_claim_658_, lean_object* v_gkrPoint_659_, lean_object* v_a_660_){
_start:
{
lean_object* v_sumcheckPolys_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v_sumcheckPolys_661_ = lean_ctor_get(v_proof_656_, 3);
v___x_662_ = lean_unsigned_to_nat(1u);
v___x_663_ = lean_nat_sub(v_round_657_, v___x_662_);
v___x_664_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_sumcheckPolys_661_, v___x_663_);
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
v___x_665_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg___closed__0));
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
v___x_671_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckRoundsM___redArg(v_inst_653_, v_inst_654_, v_fo_655_, v_claim_658_, v_one_669_, v___x_670_, v_gkrPoint_659_, v_a_668_, v_a_660_);
return v___x_671_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg___boxed(lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_fo_674_, lean_object* v_proof_675_, lean_object* v_round_676_, lean_object* v_claim_677_, lean_object* v_gkrPoint_678_, lean_object* v_a_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(v_inst_672_, v_inst_673_, v_fo_674_, v_proof_675_, v_round_676_, v_claim_677_, v_gkrPoint_678_, v_a_679_);
lean_dec(v_round_676_);
lean_dec_ref(v_proof_675_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM(lean_object* v_F_681_, lean_object* v_EF_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_fo_685_, lean_object* v_proof_686_, lean_object* v_round_687_, lean_object* v_claim_688_, lean_object* v_gkrPoint_689_, lean_object* v_a_690_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(v_inst_683_, v_inst_684_, v_fo_685_, v_proof_686_, v_round_687_, v_claim_688_, v_gkrPoint_689_, v_a_690_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___boxed(lean_object* v_F_692_, lean_object* v_EF_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_fo_696_, lean_object* v_proof_697_, lean_object* v_round_698_, lean_object* v_claim_699_, lean_object* v_gkrPoint_700_, lean_object* v_a_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM(v_F_692_, v_EF_693_, v_inst_694_, v_inst_695_, v_fo_696_, v_proof_697_, v_round_698_, v_claim_699_, v_gkrPoint_700_, v_a_701_);
lean_dec(v_round_698_);
lean_dec_ref(v_proof_697_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_inst_708_, lean_object* v_fo_709_, lean_object* v_proof_710_, lean_object* v_round_711_, lean_object* v_remainingRounds_712_, lean_object* v_numeratorClaim_713_, lean_object* v_denominatorClaim_714_, lean_object* v_gkrPoint_715_, lean_object* v_a_716_){
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
v___x_719_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_707_, v_inst_708_, v_a_716_);
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
v___x_729_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrSumcheckM___redArg(v_inst_707_, v_inst_708_, v_fo_709_, v_proof_710_, v_round_711_, v___x_728_, v_gkrPoint_715_, v_snd_724_);
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
v___x_742_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_claimsPerLayer_741_, v_round_711_);
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
v___x_743_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
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
v___x_745_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(v_inst_707_, v_inst_708_, v_a_744_, v_snd_740_);
v_a_746_ = lean_ctor_get(v___x_745_, 0);
lean_inc(v_a_746_);
lean_dec_ref(v___x_745_);
v_snd_747_ = lean_ctor_get(v_a_746_, 1);
lean_inc(v_snd_747_);
lean_dec(v_a_746_);
lean_inc_ref(v_fo_709_);
v___x_748_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(v_fo_709_, v_a_744_);
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
v___x_759_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_759_;
}
else
{
lean_object* v___x_760_; lean_object* v_a_761_; lean_object* v_fst_762_; lean_object* v_snd_763_; lean_object* v___x_764_; lean_object* v_fst_765_; lean_object* v_snd_766_; lean_object* v___x_768_; uint8_t v_isShared_769_; uint8_t v_isSharedCheck_777_; 
lean_inc_ref(v_inst_708_);
lean_inc_ref(v_inst_707_);
v___x_760_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_707_, v_inst_708_, v_snd_747_);
v_a_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc(v_a_761_);
lean_dec_ref(v___x_760_);
v_fst_762_ = lean_ctor_get(v_a_761_, 0);
lean_inc_n(v_fst_762_, 2);
v_snd_763_ = lean_ctor_get(v_a_761_, 1);
lean_inc(v_snd_763_);
lean_dec(v_a_761_);
lean_inc_ref(v_fo_709_);
v___x_764_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(v_fo_709_, v_a_744_, v_fst_762_);
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___boxed(lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_fo_784_, lean_object* v_proof_785_, lean_object* v_round_786_, lean_object* v_remainingRounds_787_, lean_object* v_numeratorClaim_788_, lean_object* v_denominatorClaim_789_, lean_object* v_gkrPoint_790_, lean_object* v_a_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(v_inst_781_, v_inst_782_, v_inst_783_, v_fo_784_, v_proof_785_, v_round_786_, v_remainingRounds_787_, v_numeratorClaim_788_, v_denominatorClaim_789_, v_gkrPoint_790_, v_a_791_);
lean_dec_ref(v_proof_785_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM(lean_object* v_F_793_, lean_object* v_EF_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_fo_798_, lean_object* v_proof_799_, lean_object* v_round_800_, lean_object* v_remainingRounds_801_, lean_object* v_numeratorClaim_802_, lean_object* v_denominatorClaim_803_, lean_object* v_gkrPoint_804_, lean_object* v_a_805_){
_start:
{
lean_object* v___x_806_; 
v___x_806_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(v_inst_795_, v_inst_796_, v_inst_797_, v_fo_798_, v_proof_799_, v_round_800_, v_remainingRounds_801_, v_numeratorClaim_802_, v_denominatorClaim_803_, v_gkrPoint_804_, v_a_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___boxed(lean_object* v_F_807_, lean_object* v_EF_808_, lean_object* v_inst_809_, lean_object* v_inst_810_, lean_object* v_inst_811_, lean_object* v_fo_812_, lean_object* v_proof_813_, lean_object* v_round_814_, lean_object* v_remainingRounds_815_, lean_object* v_numeratorClaim_816_, lean_object* v_denominatorClaim_817_, lean_object* v_gkrPoint_818_, lean_object* v_a_819_){
_start:
{
lean_object* v_res_820_; 
v_res_820_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM(v_F_807_, v_EF_808_, v_inst_809_, v_inst_810_, v_inst_811_, v_fo_812_, v_proof_813_, v_round_814_, v_remainingRounds_815_, v_numeratorClaim_816_, v_denominatorClaim_817_, v_gkrPoint_818_, v_a_819_);
lean_dec_ref(v_proof_813_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter___redArg(lean_object* v_x_821_, lean_object* v_h__1_822_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_0__Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM_match__1_splitter(lean_object* v_EF_826_, lean_object* v_motive_827_, lean_object* v_x_828_, lean_object* v_h__1_829_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(lean_object* v_inst_833_, lean_object* v_inst_834_, lean_object* v_inst_835_, lean_object* v_fo_836_, lean_object* v_proof_837_, lean_object* v_totalRounds_838_, lean_object* v_a_839_){
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
v___x_849_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_834_, v_inst_835_, v_q0Claim_844_, v_a_839_);
v_a_850_ = lean_ctor_get(v___x_849_, 0);
lean_inc(v_a_850_);
lean_dec_ref(v___x_849_);
v_snd_851_ = lean_ctor_get(v_a_850_, 1);
lean_inc(v_snd_851_);
lean_dec(v_a_850_);
v___x_852_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_claimsPerLayer_845_, v___x_842_);
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
v___x_853_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
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
v___x_855_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeGkrLayerClaimsM___redArg(v_inst_834_, v_inst_835_, v_a_854_, v_snd_851_);
v_a_856_ = lean_ctor_get(v___x_855_, 0);
lean_inc(v_a_856_);
lean_dec_ref(v___x_855_);
v_snd_857_ = lean_ctor_get(v_a_856_, 1);
lean_inc(v_snd_857_);
lean_dec(v_a_856_);
lean_inc_ref(v_fo_836_);
v___x_858_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeRecursiveRelations___redArg(v_fo_836_, v_a_854_);
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
v___x_866_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
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
v___x_869_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_869_;
}
else
{
lean_object* v___x_870_; lean_object* v_a_871_; lean_object* v_fst_872_; lean_object* v_snd_873_; lean_object* v___x_874_; lean_object* v_fst_875_; lean_object* v_snd_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_887_; 
lean_inc_ref(v_inst_835_);
lean_inc_ref(v_inst_834_);
v___x_870_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_834_, v_inst_835_, v_snd_857_);
v_a_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_a_871_);
lean_dec_ref(v___x_870_);
v_fst_872_ = lean_ctor_get(v_a_871_, 0);
lean_inc_n(v_fst_872_, 2);
v_snd_873_ = lean_ctor_get(v_a_871_, 1);
lean_inc(v_snd_873_);
lean_dec(v_a_871_);
lean_inc_ref(v_fo_836_);
v___x_874_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_reduceToSingleEvaluation___redArg(v_fo_836_, v_a_854_, v_fst_872_);
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
v___x_885_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg(v_inst_833_, v_inst_834_, v_inst_835_, v_fo_836_, v_proof_837_, v___x_880_, v___x_881_, v_fst_875_, v_snd_876_, v___x_884_, v_snd_873_);
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
v___x_894_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_894_;
}
v___jp_840_:
{
lean_object* v___x_841_; 
v___x_841_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrLaterRoundsM___redArg___closed__0));
return v___x_841_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg___boxed(lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_fo_898_, lean_object* v_proof_899_, lean_object* v_totalRounds_900_, lean_object* v_a_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(v_inst_895_, v_inst_896_, v_inst_897_, v_fo_898_, v_proof_899_, v_totalRounds_900_, v_a_901_);
lean_dec(v_totalRounds_900_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM(lean_object* v_F_903_, lean_object* v_EF_904_, lean_object* v_inst_905_, lean_object* v_inst_906_, lean_object* v_inst_907_, lean_object* v_fo_908_, lean_object* v_proof_909_, lean_object* v_totalRounds_910_, lean_object* v_a_911_){
_start:
{
lean_object* v___x_912_; 
v___x_912_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(v_inst_905_, v_inst_906_, v_inst_907_, v_fo_908_, v_proof_909_, v_totalRounds_910_, v_a_911_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___boxed(lean_object* v_F_913_, lean_object* v_EF_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_fo_918_, lean_object* v_proof_919_, lean_object* v_totalRounds_920_, lean_object* v_a_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM(v_F_913_, v_EF_914_, v_inst_915_, v_inst_916_, v_inst_917_, v_fo_918_, v_proof_919_, v_totalRounds_920_, v_a_921_);
lean_dec(v_totalRounds_920_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__0(lean_object* v_toSemiringOps_923_, lean_object* v_acc_924_, lean_object* v_x_925_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__1(lean_object* v_toSemiringOps_928_, lean_object* v_sub_929_, lean_object* v_acc_930_, lean_object* v_x_931_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(lean_object* v_fo_936_, lean_object* v_inst_937_, lean_object* v_preprocessed_938_, lean_object* v_partitionedMain_939_, lean_object* v_publicValues_940_, lean_object* v_rs_941_, lean_object* v_lSkip_942_){
_start:
{
lean_object* v_toRingOps_943_; lean_object* v_inv_944_; lean_object* v_toSemiringOps_945_; lean_object* v_sub_946_; lean_object* v___f_947_; lean_object* v___f_948_; lean_object* v___y_950_; lean_object* v___y_951_; lean_object* v___y_972_; 
v_toRingOps_943_ = lean_ctor_get(v_fo_936_, 0);
v_inv_944_ = lean_ctor_get(v_fo_936_, 1);
v_toSemiringOps_945_ = lean_ctor_get(v_toRingOps_943_, 0);
v_sub_946_ = lean_ctor_get(v_toRingOps_943_, 1);
lean_inc_ref_n(v_toSemiringOps_945_, 2);
v___f_947_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_947_, 0, v_toSemiringOps_945_);
lean_inc(v_sub_946_);
v___f_948_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg___lam__1), 4, 2);
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
v___x_963_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(v_fo_936_, v___y_950_, v_lSkip_942_);
lean_inc(v_inv_959_);
v___x_964_ = lean_apply_2(v_mul_954_, v_inv_959_, v___x_963_);
v___x_965_ = lean_apply_2(v_mul_954_, v___x_964_, v_prodOneMinus_961_);
v___x_966_ = lean_apply_2(v_mul_954_, v___y_950_, v_omega_960_);
v___x_967_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_progressionExp2___redArg(v_fo_936_, v___x_966_, v_lSkip_942_);
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27(lean_object* v_F_976_, lean_object* v_EF_977_, lean_object* v_fo_978_, lean_object* v_inst_979_, lean_object* v_preprocessed_980_, lean_object* v_partitionedMain_981_, lean_object* v_publicValues_982_, lean_object* v_rs_983_, lean_object* v_lSkip_984_){
_start:
{
lean_object* v___x_985_; 
v___x_985_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(v_fo_978_, v_inst_979_, v_preprocessed_980_, v_partitionedMain_981_, v_publicValues_982_, v_rs_983_, v_lSkip_984_);
return v___x_985_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst___redArg(lean_object* v_algMap_986_, lean_object* v_c_987_){
_start:
{
lean_object* v___x_988_; 
v___x_988_ = lean_apply_1(v_algMap_986_, v_c_987_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalConst(lean_object* v_F_989_, lean_object* v_EF_990_, lean_object* v_algMap_991_, lean_object* v_c_992_){
_start:
{
lean_object* v___x_993_; 
v___x_993_ = lean_apply_1(v_algMap_991_, v_c_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(lean_object* v_fo_994_, lean_object* v_algMap_995_, lean_object* v_e_996_, lean_object* v_v_997_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg___boxed(lean_object* v_fo_1044_, lean_object* v_algMap_1045_, lean_object* v_e_1046_, lean_object* v_v_1047_){
_start:
{
lean_object* v_res_1048_; 
v_res_1048_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(v_fo_1044_, v_algMap_1045_, v_e_1046_, v_v_1047_);
lean_dec_ref(v_e_1046_);
lean_dec_ref(v_fo_1044_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar(lean_object* v_F_1049_, lean_object* v_EF_1050_, lean_object* v_fo_1051_, lean_object* v_algMap_1052_, lean_object* v_e_1053_, lean_object* v_v_1054_){
_start:
{
lean_object* v___x_1055_; 
v___x_1055_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___redArg(v_fo_1051_, v_algMap_1052_, v_e_1053_, v_v_1054_);
return v___x_1055_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___boxed(lean_object* v_F_1056_, lean_object* v_EF_1057_, lean_object* v_fo_1058_, lean_object* v_algMap_1059_, lean_object* v_e_1060_, lean_object* v_v_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar(v_F_1056_, v_EF_1057_, v_fo_1058_, v_algMap_1059_, v_e_1060_, v_v_1061_);
lean_dec_ref(v_e_1060_);
lean_dec_ref(v_fo_1058_);
return v_res_1062_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(lean_object* v_algMap_1063_, lean_object* v_e_1064_, lean_object* v_v_1065_){
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
v___x_1070_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v_preprocessed_1069_);
if (lean_obj_tag(v___x_1070_) == 0)
{
lean_object* v___x_1071_; 
lean_dec_ref_known(v___x_1070_, 1);
lean_dec(v_offset_1068_);
lean_dec(v_index_1067_);
v___x_1071_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1071_;
}
else
{
lean_object* v_a_1072_; lean_object* v___x_1073_; 
v_a_1072_ = lean_ctor_get(v___x_1070_, 0);
lean_inc(v_a_1072_);
lean_dec_ref_known(v___x_1070_, 1);
v___x_1073_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_1072_, v_index_1067_);
lean_dec(v_a_1072_);
if (lean_obj_tag(v___x_1073_) == 0)
{
lean_object* v___x_1074_; 
lean_dec_ref_known(v___x_1073_, 1);
lean_dec(v_offset_1068_);
v___x_1074_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
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
v___x_1094_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_partitionedMain_1093_, v_partIndex_1091_);
lean_dec(v_partitionedMain_1093_);
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v___x_1095_; 
lean_dec_ref_known(v___x_1094_, 1);
lean_dec(v_offset_1092_);
lean_dec(v_index_1090_);
v___x_1095_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1095_;
}
else
{
lean_object* v_a_1096_; lean_object* v___x_1097_; 
v_a_1096_ = lean_ctor_get(v___x_1094_, 0);
lean_inc(v_a_1096_);
lean_dec_ref_known(v___x_1094_, 1);
v___x_1097_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_1096_, v_index_1090_);
lean_dec(v_a_1096_);
if (lean_obj_tag(v___x_1097_) == 0)
{
lean_object* v___x_1098_; 
lean_dec_ref_known(v___x_1097_, 1);
lean_dec(v_offset_1092_);
v___x_1098_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
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
v___x_1116_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_publicValues_1115_, v_index_1114_);
lean_dec(v_publicValues_1115_);
if (lean_obj_tag(v___x_1116_) == 0)
{
lean_object* v___x_1117_; 
lean_dec_ref_known(v___x_1116_, 1);
lean_dec(v_algMap_1063_);
v___x_1117_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
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
v___x_1127_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM(lean_object* v_F_1128_, lean_object* v_EF_1129_, lean_object* v_algMap_1130_, lean_object* v_e_1131_, lean_object* v_v_1132_){
_start:
{
lean_object* v___x_1133_; 
v___x_1133_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(v_algMap_1130_, v_e_1131_, v_v_1132_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg(lean_object* v_e_1134_){
_start:
{
lean_object* v_isFirstRow_1135_; 
v_isFirstRow_1135_ = lean_ctor_get(v_e_1134_, 3);
lean_inc(v_isFirstRow_1135_);
return v_isFirstRow_1135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg___boxed(lean_object* v_e_1136_){
_start:
{
lean_object* v_res_1137_; 
v_res_1137_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___redArg(v_e_1136_);
lean_dec_ref(v_e_1136_);
return v_res_1137_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow(lean_object* v_F_1138_, lean_object* v_EF_1139_, lean_object* v_e_1140_){
_start:
{
lean_object* v_isFirstRow_1141_; 
v_isFirstRow_1141_ = lean_ctor_get(v_e_1140_, 3);
lean_inc(v_isFirstRow_1141_);
return v_isFirstRow_1141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow___boxed(lean_object* v_F_1142_, lean_object* v_EF_1143_, lean_object* v_e_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsFirstRow(v_F_1142_, v_EF_1143_, v_e_1144_);
lean_dec_ref(v_e_1144_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg(lean_object* v_e_1146_){
_start:
{
lean_object* v_isLastRow_1147_; 
v_isLastRow_1147_ = lean_ctor_get(v_e_1146_, 4);
lean_inc(v_isLastRow_1147_);
return v_isLastRow_1147_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg___boxed(lean_object* v_e_1148_){
_start:
{
lean_object* v_res_1149_; 
v_res_1149_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___redArg(v_e_1148_);
lean_dec_ref(v_e_1148_);
return v_res_1149_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow(lean_object* v_F_1150_, lean_object* v_EF_1151_, lean_object* v_e_1152_){
_start:
{
lean_object* v_isLastRow_1153_; 
v_isLastRow_1153_ = lean_ctor_get(v_e_1152_, 4);
lean_inc(v_isLastRow_1153_);
return v_isLastRow_1153_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow___boxed(lean_object* v_F_1154_, lean_object* v_EF_1155_, lean_object* v_e_1156_){
_start:
{
lean_object* v_res_1157_; 
v_res_1157_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsLastRow(v_F_1154_, v_EF_1155_, v_e_1156_);
lean_dec_ref(v_e_1156_);
return v_res_1157_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(lean_object* v_fo_1158_, lean_object* v_e_1159_){
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition(lean_object* v_F_1166_, lean_object* v_EF_1167_, lean_object* v_fo_1168_, lean_object* v_e_1169_){
_start:
{
lean_object* v___x_1170_; 
v___x_1170_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(v_fo_1168_, v_e_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg___lam__0(lean_object* v_algMap_1171_, lean_object* v___y_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lean_apply_1(v_algMap_1171_, v___y_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg(lean_object* v_fo_1174_, lean_object* v_algMap_1175_, lean_object* v_e_1176_){
_start:
{
lean_object* v_isFirstRow_1177_; lean_object* v_isLastRow_1178_; lean_object* v___f_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; 
v_isFirstRow_1177_ = lean_ctor_get(v_e_1176_, 3);
lean_inc(v_isFirstRow_1177_);
v_isLastRow_1178_ = lean_ctor_get(v_e_1176_, 4);
lean_inc(v_isLastRow_1178_);
lean_inc(v_algMap_1175_);
v___f_1179_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1179_, 0, v_algMap_1175_);
lean_inc_ref(v_e_1176_);
lean_inc_ref(v_fo_1174_);
v___x_1180_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVar___boxed), 6, 5);
lean_closure_set(v___x_1180_, 0, lean_box(0));
lean_closure_set(v___x_1180_, 1, lean_box(0));
lean_closure_set(v___x_1180_, 2, v_fo_1174_);
lean_closure_set(v___x_1180_, 3, v_algMap_1175_);
lean_closure_set(v___x_1180_, 4, v_e_1176_);
v___x_1181_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(v_fo_1174_, v_e_1176_);
v___x_1182_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1182_, 0, v___f_1179_);
lean_ctor_set(v___x_1182_, 1, v___x_1180_);
lean_ctor_set(v___x_1182_, 2, v_isFirstRow_1177_);
lean_ctor_set(v___x_1182_, 3, v_isLastRow_1178_);
lean_ctor_set(v___x_1182_, 4, v___x_1181_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator(lean_object* v_F_1183_, lean_object* v_EF_1184_, lean_object* v_fo_1185_, lean_object* v_algMap_1186_, lean_object* v_e_1187_){
_start:
{
lean_object* v___x_1188_; 
v___x_1188_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg(v_fo_1185_, v_algMap_1186_, v_e_1187_);
return v___x_1188_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalExpr___redArg(lean_object* v_fo_1189_, lean_object* v_algMap_1190_, lean_object* v_e_1191_, lean_object* v_expr_1192_){
_start:
{
lean_object* v___x_1193_; lean_object* v___x_1194_; 
lean_inc_ref(v_fo_1189_);
v___x_1193_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg(v_fo_1189_, v_algMap_1190_, v_e_1191_);
v___x_1194_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_SymbolicEvaluator_evalExpr___redArg(v_fo_1189_, v___x_1193_, v_expr_1192_);
return v___x_1194_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalExpr(lean_object* v_F_1195_, lean_object* v_EF_1196_, lean_object* v_fo_1197_, lean_object* v_algMap_1198_, lean_object* v_e_1199_, lean_object* v_expr_1200_){
_start:
{
lean_object* v___x_1201_; 
v___x_1201_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalExpr___redArg(v_fo_1197_, v_algMap_1198_, v_e_1199_, v_expr_1200_);
return v___x_1201_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodes___redArg(lean_object* v_fo_1202_, lean_object* v_algMap_1203_, lean_object* v_e_1204_, lean_object* v_nodes_1205_){
_start:
{
lean_object* v___x_1206_; lean_object* v___x_1207_; 
lean_inc_ref(v_fo_1202_);
v___x_1206_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_toSymbolicEvaluator___redArg(v_fo_1202_, v_algMap_1203_, v_e_1204_);
v___x_1207_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_SymbolicEvaluator_evalNodes___redArg(v_fo_1202_, v___x_1206_, v_nodes_1205_);
return v___x_1207_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodes(lean_object* v_F_1208_, lean_object* v_EF_1209_, lean_object* v_fo_1210_, lean_object* v_algMap_1211_, lean_object* v_e_1212_, lean_object* v_nodes_1213_){
_start:
{
lean_object* v___x_1214_; 
v___x_1214_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodes___redArg(v_fo_1210_, v_algMap_1211_, v_e_1212_, v_nodes_1213_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(lean_object* v_fo_1215_, lean_object* v_algMap_1216_, lean_object* v_e_1217_, lean_object* v_values_1218_, lean_object* v_x_1219_){
_start:
{
switch(lean_obj_tag(v_x_1219_))
{
case 0:
{
lean_object* v_v_1220_; lean_object* v___x_1221_; 
lean_dec_ref(v_fo_1215_);
v_v_1220_ = lean_ctor_get(v_x_1219_, 0);
lean_inc_ref(v_v_1220_);
lean_dec_ref_known(v_x_1219_, 1);
v___x_1221_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalVarM___redArg(v_algMap_1216_, v_e_1217_, v_v_1220_);
return v___x_1221_;
}
case 1:
{
lean_object* v_isFirstRow_1222_; lean_object* v___x_1223_; 
lean_dec(v_algMap_1216_);
lean_dec_ref(v_fo_1215_);
v_isFirstRow_1222_ = lean_ctor_get(v_e_1217_, 3);
lean_inc(v_isFirstRow_1222_);
lean_dec_ref(v_e_1217_);
v___x_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1223_, 0, v_isFirstRow_1222_);
return v___x_1223_;
}
case 2:
{
lean_object* v_isLastRow_1224_; lean_object* v___x_1225_; 
lean_dec(v_algMap_1216_);
lean_dec_ref(v_fo_1215_);
v_isLastRow_1224_ = lean_ctor_get(v_e_1217_, 4);
lean_inc(v_isLastRow_1224_);
lean_dec_ref(v_e_1217_);
v___x_1225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1225_, 0, v_isLastRow_1224_);
return v___x_1225_;
}
case 3:
{
lean_object* v___x_1226_; lean_object* v___x_1227_; 
lean_dec(v_algMap_1216_);
v___x_1226_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalIsTransition___redArg(v_fo_1215_, v_e_1217_);
v___x_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1227_, 0, v___x_1226_);
return v___x_1227_;
}
case 4:
{
lean_object* v_c_1228_; lean_object* v___x_1230_; uint8_t v_isShared_1231_; uint8_t v_isSharedCheck_1236_; 
lean_dec_ref(v_e_1217_);
lean_dec_ref(v_fo_1215_);
v_c_1228_ = lean_ctor_get(v_x_1219_, 0);
v_isSharedCheck_1236_ = !lean_is_exclusive(v_x_1219_);
if (v_isSharedCheck_1236_ == 0)
{
v___x_1230_ = v_x_1219_;
v_isShared_1231_ = v_isSharedCheck_1236_;
goto v_resetjp_1229_;
}
else
{
lean_inc(v_c_1228_);
lean_dec(v_x_1219_);
v___x_1230_ = lean_box(0);
v_isShared_1231_ = v_isSharedCheck_1236_;
goto v_resetjp_1229_;
}
v_resetjp_1229_:
{
lean_object* v___x_1232_; lean_object* v___x_1234_; 
v___x_1232_ = lean_apply_1(v_algMap_1216_, v_c_1228_);
if (v_isShared_1231_ == 0)
{
lean_ctor_set_tag(v___x_1230_, 1);
lean_ctor_set(v___x_1230_, 0, v___x_1232_);
v___x_1234_ = v___x_1230_;
goto v_reusejp_1233_;
}
else
{
lean_object* v_reuseFailAlloc_1235_; 
v_reuseFailAlloc_1235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1235_, 0, v___x_1232_);
v___x_1234_ = v_reuseFailAlloc_1235_;
goto v_reusejp_1233_;
}
v_reusejp_1233_:
{
return v___x_1234_;
}
}
}
case 5:
{
lean_object* v_leftIdx_1237_; lean_object* v_rightIdx_1238_; lean_object* v___x_1239_; 
lean_dec_ref(v_e_1217_);
lean_dec(v_algMap_1216_);
v_leftIdx_1237_ = lean_ctor_get(v_x_1219_, 0);
lean_inc(v_leftIdx_1237_);
v_rightIdx_1238_ = lean_ctor_get(v_x_1219_, 1);
lean_inc(v_rightIdx_1238_);
lean_dec_ref_known(v_x_1219_, 3);
v___x_1239_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_leftIdx_1237_);
if (lean_obj_tag(v___x_1239_) == 0)
{
lean_dec(v_rightIdx_1238_);
lean_dec_ref(v_fo_1215_);
return v___x_1239_;
}
else
{
lean_object* v_a_1240_; lean_object* v___x_1241_; 
v_a_1240_ = lean_ctor_get(v___x_1239_, 0);
lean_inc(v_a_1240_);
lean_dec_ref_known(v___x_1239_, 1);
v___x_1241_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_rightIdx_1238_);
if (lean_obj_tag(v___x_1241_) == 0)
{
lean_dec(v_a_1240_);
lean_dec_ref(v_fo_1215_);
return v___x_1241_;
}
else
{
lean_object* v_toRingOps_1242_; lean_object* v_toSemiringOps_1243_; lean_object* v_a_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1253_; 
v_toRingOps_1242_ = lean_ctor_get(v_fo_1215_, 0);
lean_inc_ref(v_toRingOps_1242_);
lean_dec_ref(v_fo_1215_);
v_toSemiringOps_1243_ = lean_ctor_get(v_toRingOps_1242_, 0);
lean_inc_ref(v_toSemiringOps_1243_);
lean_dec_ref(v_toRingOps_1242_);
v_a_1244_ = lean_ctor_get(v___x_1241_, 0);
v_isSharedCheck_1253_ = !lean_is_exclusive(v___x_1241_);
if (v_isSharedCheck_1253_ == 0)
{
v___x_1246_ = v___x_1241_;
v_isShared_1247_ = v_isSharedCheck_1253_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_a_1244_);
lean_dec(v___x_1241_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1253_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
lean_object* v_add_1248_; lean_object* v___x_1249_; lean_object* v___x_1251_; 
v_add_1248_ = lean_ctor_get(v_toSemiringOps_1243_, 3);
lean_inc(v_add_1248_);
lean_dec_ref(v_toSemiringOps_1243_);
v___x_1249_ = lean_apply_2(v_add_1248_, v_a_1240_, v_a_1244_);
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 0, v___x_1249_);
v___x_1251_ = v___x_1246_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1252_; 
v_reuseFailAlloc_1252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1252_, 0, v___x_1249_);
v___x_1251_ = v_reuseFailAlloc_1252_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
return v___x_1251_;
}
}
}
}
}
case 6:
{
lean_object* v_leftIdx_1254_; lean_object* v_rightIdx_1255_; lean_object* v___x_1256_; 
lean_dec_ref(v_e_1217_);
lean_dec(v_algMap_1216_);
v_leftIdx_1254_ = lean_ctor_get(v_x_1219_, 0);
lean_inc(v_leftIdx_1254_);
v_rightIdx_1255_ = lean_ctor_get(v_x_1219_, 1);
lean_inc(v_rightIdx_1255_);
lean_dec_ref_known(v_x_1219_, 3);
v___x_1256_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_leftIdx_1254_);
if (lean_obj_tag(v___x_1256_) == 0)
{
lean_dec(v_rightIdx_1255_);
lean_dec_ref(v_fo_1215_);
return v___x_1256_;
}
else
{
lean_object* v_a_1257_; lean_object* v___x_1258_; 
v_a_1257_ = lean_ctor_get(v___x_1256_, 0);
lean_inc(v_a_1257_);
lean_dec_ref_known(v___x_1256_, 1);
v___x_1258_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_rightIdx_1255_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_dec(v_a_1257_);
lean_dec_ref(v_fo_1215_);
return v___x_1258_;
}
else
{
lean_object* v_toRingOps_1259_; lean_object* v_a_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1269_; 
v_toRingOps_1259_ = lean_ctor_get(v_fo_1215_, 0);
lean_inc_ref(v_toRingOps_1259_);
lean_dec_ref(v_fo_1215_);
v_a_1260_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1269_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1269_ == 0)
{
v___x_1262_ = v___x_1258_;
v_isShared_1263_ = v_isSharedCheck_1269_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_a_1260_);
lean_dec(v___x_1258_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1269_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v_sub_1264_; lean_object* v___x_1265_; lean_object* v___x_1267_; 
v_sub_1264_ = lean_ctor_get(v_toRingOps_1259_, 1);
lean_inc(v_sub_1264_);
lean_dec_ref(v_toRingOps_1259_);
v___x_1265_ = lean_apply_2(v_sub_1264_, v_a_1257_, v_a_1260_);
if (v_isShared_1263_ == 0)
{
lean_ctor_set(v___x_1262_, 0, v___x_1265_);
v___x_1267_ = v___x_1262_;
goto v_reusejp_1266_;
}
else
{
lean_object* v_reuseFailAlloc_1268_; 
v_reuseFailAlloc_1268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1268_, 0, v___x_1265_);
v___x_1267_ = v_reuseFailAlloc_1268_;
goto v_reusejp_1266_;
}
v_reusejp_1266_:
{
return v___x_1267_;
}
}
}
}
}
case 7:
{
lean_object* v_idx_1270_; lean_object* v___x_1271_; 
lean_dec_ref(v_e_1217_);
lean_dec(v_algMap_1216_);
v_idx_1270_ = lean_ctor_get(v_x_1219_, 0);
lean_inc(v_idx_1270_);
lean_dec_ref_known(v_x_1219_, 2);
v___x_1271_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_idx_1270_);
if (lean_obj_tag(v___x_1271_) == 0)
{
lean_dec_ref(v_fo_1215_);
return v___x_1271_;
}
else
{
lean_object* v_toRingOps_1272_; lean_object* v_toSemiringOps_1273_; lean_object* v_a_1274_; lean_object* v___x_1276_; uint8_t v_isShared_1277_; uint8_t v_isSharedCheck_1284_; 
v_toRingOps_1272_ = lean_ctor_get(v_fo_1215_, 0);
lean_inc_ref(v_toRingOps_1272_);
lean_dec_ref(v_fo_1215_);
v_toSemiringOps_1273_ = lean_ctor_get(v_toRingOps_1272_, 0);
lean_inc_ref(v_toSemiringOps_1273_);
v_a_1274_ = lean_ctor_get(v___x_1271_, 0);
v_isSharedCheck_1284_ = !lean_is_exclusive(v___x_1271_);
if (v_isSharedCheck_1284_ == 0)
{
v___x_1276_ = v___x_1271_;
v_isShared_1277_ = v_isSharedCheck_1284_;
goto v_resetjp_1275_;
}
else
{
lean_inc(v_a_1274_);
lean_dec(v___x_1271_);
v___x_1276_ = lean_box(0);
v_isShared_1277_ = v_isSharedCheck_1284_;
goto v_resetjp_1275_;
}
v_resetjp_1275_:
{
lean_object* v_sub_1278_; lean_object* v_zero_1279_; lean_object* v___x_1280_; lean_object* v___x_1282_; 
v_sub_1278_ = lean_ctor_get(v_toRingOps_1272_, 1);
lean_inc(v_sub_1278_);
lean_dec_ref(v_toRingOps_1272_);
v_zero_1279_ = lean_ctor_get(v_toSemiringOps_1273_, 0);
lean_inc(v_zero_1279_);
lean_dec_ref(v_toSemiringOps_1273_);
v___x_1280_ = lean_apply_2(v_sub_1278_, v_zero_1279_, v_a_1274_);
if (v_isShared_1277_ == 0)
{
lean_ctor_set(v___x_1276_, 0, v___x_1280_);
v___x_1282_ = v___x_1276_;
goto v_reusejp_1281_;
}
else
{
lean_object* v_reuseFailAlloc_1283_; 
v_reuseFailAlloc_1283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1283_, 0, v___x_1280_);
v___x_1282_ = v_reuseFailAlloc_1283_;
goto v_reusejp_1281_;
}
v_reusejp_1281_:
{
return v___x_1282_;
}
}
}
}
default: 
{
lean_object* v_leftIdx_1285_; lean_object* v_rightIdx_1286_; lean_object* v___x_1287_; 
lean_dec_ref(v_e_1217_);
lean_dec(v_algMap_1216_);
v_leftIdx_1285_ = lean_ctor_get(v_x_1219_, 0);
lean_inc(v_leftIdx_1285_);
v_rightIdx_1286_ = lean_ctor_get(v_x_1219_, 1);
lean_inc(v_rightIdx_1286_);
lean_dec_ref_known(v_x_1219_, 3);
v___x_1287_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_leftIdx_1285_);
if (lean_obj_tag(v___x_1287_) == 0)
{
lean_dec(v_rightIdx_1286_);
lean_dec_ref(v_fo_1215_);
return v___x_1287_;
}
else
{
lean_object* v_a_1288_; lean_object* v___x_1289_; 
v_a_1288_ = lean_ctor_get(v___x_1287_, 0);
lean_inc(v_a_1288_);
lean_dec_ref_known(v___x_1287_, 1);
v___x_1289_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_values_1218_, v_rightIdx_1286_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_dec(v_a_1288_);
lean_dec_ref(v_fo_1215_);
return v___x_1289_;
}
else
{
lean_object* v_toRingOps_1290_; lean_object* v_toSemiringOps_1291_; lean_object* v_a_1292_; lean_object* v___x_1294_; uint8_t v_isShared_1295_; uint8_t v_isSharedCheck_1301_; 
v_toRingOps_1290_ = lean_ctor_get(v_fo_1215_, 0);
lean_inc_ref(v_toRingOps_1290_);
lean_dec_ref(v_fo_1215_);
v_toSemiringOps_1291_ = lean_ctor_get(v_toRingOps_1290_, 0);
lean_inc_ref(v_toSemiringOps_1291_);
lean_dec_ref(v_toRingOps_1290_);
v_a_1292_ = lean_ctor_get(v___x_1289_, 0);
v_isSharedCheck_1301_ = !lean_is_exclusive(v___x_1289_);
if (v_isSharedCheck_1301_ == 0)
{
v___x_1294_ = v___x_1289_;
v_isShared_1295_ = v_isSharedCheck_1301_;
goto v_resetjp_1293_;
}
else
{
lean_inc(v_a_1292_);
lean_dec(v___x_1289_);
v___x_1294_ = lean_box(0);
v_isShared_1295_ = v_isSharedCheck_1301_;
goto v_resetjp_1293_;
}
v_resetjp_1293_:
{
lean_object* v_mul_1296_; lean_object* v___x_1297_; lean_object* v___x_1299_; 
v_mul_1296_ = lean_ctor_get(v_toSemiringOps_1291_, 4);
lean_inc(v_mul_1296_);
lean_dec_ref(v_toSemiringOps_1291_);
v___x_1297_ = lean_apply_2(v_mul_1296_, v_a_1288_, v_a_1292_);
if (v_isShared_1295_ == 0)
{
lean_ctor_set(v___x_1294_, 0, v___x_1297_);
v___x_1299_ = v___x_1294_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v___x_1297_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg___boxed(lean_object* v_fo_1302_, lean_object* v_algMap_1303_, lean_object* v_e_1304_, lean_object* v_values_1305_, lean_object* v_x_1306_){
_start:
{
lean_object* v_res_1307_; 
v_res_1307_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(v_fo_1302_, v_algMap_1303_, v_e_1304_, v_values_1305_, v_x_1306_);
lean_dec(v_values_1305_);
return v_res_1307_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM(lean_object* v_F_1308_, lean_object* v_EF_1309_, lean_object* v_fo_1310_, lean_object* v_algMap_1311_, lean_object* v_e_1312_, lean_object* v_values_1313_, lean_object* v_x_1314_){
_start:
{
lean_object* v___x_1315_; 
v___x_1315_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(v_fo_1310_, v_algMap_1311_, v_e_1312_, v_values_1313_, v_x_1314_);
return v___x_1315_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___boxed(lean_object* v_F_1316_, lean_object* v_EF_1317_, lean_object* v_fo_1318_, lean_object* v_algMap_1319_, lean_object* v_e_1320_, lean_object* v_values_1321_, lean_object* v_x_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM(v_F_1316_, v_EF_1317_, v_fo_1318_, v_algMap_1319_, v_e_1320_, v_values_1321_, v_x_1322_);
lean_dec(v_values_1321_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(lean_object* v_fo_1327_, lean_object* v_algMap_1328_, lean_object* v_e_1329_, lean_object* v_x_1330_, lean_object* v_x_1331_){
_start:
{
if (lean_obj_tag(v_x_1331_) == 0)
{
lean_object* v___x_1332_; 
lean_dec_ref(v_e_1329_);
lean_dec(v_algMap_1328_);
lean_dec_ref(v_fo_1327_);
v___x_1332_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1332_, 0, v_x_1330_);
return v___x_1332_;
}
else
{
lean_object* v_head_1333_; lean_object* v_tail_1334_; lean_object* v___x_1336_; uint8_t v_isShared_1337_; uint8_t v_isSharedCheck_1347_; 
v_head_1333_ = lean_ctor_get(v_x_1331_, 0);
v_tail_1334_ = lean_ctor_get(v_x_1331_, 1);
v_isSharedCheck_1347_ = !lean_is_exclusive(v_x_1331_);
if (v_isSharedCheck_1347_ == 0)
{
v___x_1336_ = v_x_1331_;
v_isShared_1337_ = v_isSharedCheck_1347_;
goto v_resetjp_1335_;
}
else
{
lean_inc(v_tail_1334_);
lean_inc(v_head_1333_);
lean_dec(v_x_1331_);
v___x_1336_ = lean_box(0);
v_isShared_1337_ = v_isSharedCheck_1347_;
goto v_resetjp_1335_;
}
v_resetjp_1335_:
{
lean_object* v___x_1338_; 
lean_inc_ref(v_e_1329_);
lean_inc(v_algMap_1328_);
lean_inc_ref(v_fo_1327_);
v___x_1338_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodeM___redArg(v_fo_1327_, v_algMap_1328_, v_e_1329_, v_x_1330_, v_head_1333_);
if (lean_obj_tag(v___x_1338_) == 0)
{
lean_object* v___x_1339_; 
lean_dec_ref_known(v___x_1338_, 1);
lean_del_object(v___x_1336_);
lean_dec(v_tail_1334_);
lean_dec(v_x_1330_);
lean_dec_ref(v_e_1329_);
lean_dec(v_algMap_1328_);
lean_dec_ref(v_fo_1327_);
v___x_1339_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg___closed__0));
return v___x_1339_;
}
else
{
lean_object* v_a_1340_; lean_object* v___x_1341_; lean_object* v___x_1343_; 
v_a_1340_ = lean_ctor_get(v___x_1338_, 0);
lean_inc(v_a_1340_);
lean_dec_ref_known(v___x_1338_, 1);
v___x_1341_ = lean_box(0);
if (v_isShared_1337_ == 0)
{
lean_ctor_set(v___x_1336_, 1, v___x_1341_);
lean_ctor_set(v___x_1336_, 0, v_a_1340_);
v___x_1343_ = v___x_1336_;
goto v_reusejp_1342_;
}
else
{
lean_object* v_reuseFailAlloc_1346_; 
v_reuseFailAlloc_1346_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1346_, 0, v_a_1340_);
lean_ctor_set(v_reuseFailAlloc_1346_, 1, v___x_1341_);
v___x_1343_ = v_reuseFailAlloc_1346_;
goto v_reusejp_1342_;
}
v_reusejp_1342_:
{
lean_object* v___x_1344_; 
v___x_1344_ = l_List_appendTR___redArg(v_x_1330_, v___x_1343_);
v_x_1330_ = v___x_1344_;
v_x_1331_ = v_tail_1334_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(lean_object* v_fo_1348_, lean_object* v_algMap_1349_, lean_object* v_e_1350_, lean_object* v_nodes_1351_){
_start:
{
lean_object* v___x_1352_; lean_object* v___x_1353_; 
v___x_1352_ = lean_box(0);
v___x_1353_ = lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(v_fo_1348_, v_algMap_1349_, v_e_1350_, v___x_1352_, v_nodes_1351_);
return v___x_1353_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM(lean_object* v_F_1354_, lean_object* v_EF_1355_, lean_object* v_fo_1356_, lean_object* v_algMap_1357_, lean_object* v_e_1358_, lean_object* v_nodes_1359_){
_start:
{
lean_object* v___x_1360_; 
v___x_1360_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(v_fo_1356_, v_algMap_1357_, v_e_1358_, v_nodes_1359_);
return v___x_1360_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0(lean_object* v_EF_1361_, lean_object* v_F_1362_, lean_object* v_fo_1363_, lean_object* v_algMap_1364_, lean_object* v_e_1365_, lean_object* v_x_1366_, lean_object* v_x_1367_){
_start:
{
lean_object* v___x_1368_; 
v___x_1368_ = lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM_spec__0___redArg(v_fo_1363_, v_algMap_1364_, v_e_1365_, v_x_1366_, v_x_1367_);
return v___x_1368_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(lean_object* v_x_1374_){
_start:
{
if (lean_obj_tag(v_x_1374_) == 0)
{
lean_object* v___x_1375_; 
v___x_1375_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__0));
return v___x_1375_;
}
else
{
lean_object* v_tail_1376_; 
v_tail_1376_ = lean_ctor_get(v_x_1374_, 1);
lean_inc(v_tail_1376_);
if (lean_obj_tag(v_tail_1376_) == 0)
{
lean_object* v___x_1377_; 
lean_dec_ref_known(v_x_1374_, 2);
v___x_1377_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg___closed__1));
return v___x_1377_;
}
else
{
lean_object* v_head_1378_; lean_object* v___x_1380_; uint8_t v_isShared_1381_; uint8_t v_isSharedCheck_1403_; 
v_head_1378_ = lean_ctor_get(v_x_1374_, 0);
v_isSharedCheck_1403_ = !lean_is_exclusive(v_x_1374_);
if (v_isSharedCheck_1403_ == 0)
{
lean_object* v_unused_1404_; 
v_unused_1404_ = lean_ctor_get(v_x_1374_, 1);
lean_dec(v_unused_1404_);
v___x_1380_ = v_x_1374_;
v_isShared_1381_ = v_isSharedCheck_1403_;
goto v_resetjp_1379_;
}
else
{
lean_inc(v_head_1378_);
lean_dec(v_x_1374_);
v___x_1380_ = lean_box(0);
v_isShared_1381_ = v_isSharedCheck_1403_;
goto v_resetjp_1379_;
}
v_resetjp_1379_:
{
lean_object* v_head_1382_; lean_object* v_tail_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1402_; 
v_head_1382_ = lean_ctor_get(v_tail_1376_, 0);
v_tail_1383_ = lean_ctor_get(v_tail_1376_, 1);
v_isSharedCheck_1402_ = !lean_is_exclusive(v_tail_1376_);
if (v_isSharedCheck_1402_ == 0)
{
v___x_1385_ = v_tail_1376_;
v_isShared_1386_ = v_isSharedCheck_1402_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_tail_1383_);
lean_inc(v_head_1382_);
lean_dec(v_tail_1376_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1402_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
lean_object* v___x_1387_; 
v___x_1387_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(v_tail_1383_);
if (lean_obj_tag(v___x_1387_) == 0)
{
lean_del_object(v___x_1385_);
lean_dec(v_head_1382_);
lean_del_object(v___x_1380_);
lean_dec(v_head_1378_);
return v___x_1387_;
}
else
{
lean_object* v_a_1388_; lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1401_; 
v_a_1388_ = lean_ctor_get(v___x_1387_, 0);
v_isSharedCheck_1401_ = !lean_is_exclusive(v___x_1387_);
if (v_isSharedCheck_1401_ == 0)
{
v___x_1390_ = v___x_1387_;
v_isShared_1391_ = v_isSharedCheck_1401_;
goto v_resetjp_1389_;
}
else
{
lean_inc(v_a_1388_);
lean_dec(v___x_1387_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1401_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
lean_object* v___x_1393_; 
if (v_isShared_1381_ == 0)
{
lean_ctor_set_tag(v___x_1380_, 0);
lean_ctor_set(v___x_1380_, 1, v_head_1382_);
v___x_1393_ = v___x_1380_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v_head_1378_);
lean_ctor_set(v_reuseFailAlloc_1400_, 1, v_head_1382_);
v___x_1393_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
lean_object* v___x_1395_; 
if (v_isShared_1386_ == 0)
{
lean_ctor_set(v___x_1385_, 1, v_a_1388_);
lean_ctor_set(v___x_1385_, 0, v___x_1393_);
v___x_1395_ = v___x_1385_;
goto v_reusejp_1394_;
}
else
{
lean_object* v_reuseFailAlloc_1399_; 
v_reuseFailAlloc_1399_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1399_, 0, v___x_1393_);
lean_ctor_set(v_reuseFailAlloc_1399_, 1, v_a_1388_);
v___x_1395_ = v_reuseFailAlloc_1399_;
goto v_reusejp_1394_;
}
v_reusejp_1394_:
{
lean_object* v___x_1397_; 
if (v_isShared_1391_ == 0)
{
lean_ctor_set(v___x_1390_, 0, v___x_1395_);
v___x_1397_ = v___x_1390_;
goto v_reusejp_1396_;
}
else
{
lean_object* v_reuseFailAlloc_1398_; 
v_reuseFailAlloc_1398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1398_, 0, v___x_1395_);
v___x_1397_ = v_reuseFailAlloc_1398_;
goto v_reusejp_1396_;
}
v_reusejp_1396_:
{
return v___x_1397_;
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent(lean_object* v_EF_1405_, lean_object* v_x_1406_){
_start:
{
lean_object* v___x_1407_; 
v___x_1407_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(v_x_1406_);
return v___x_1407_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(lean_object* v_fo_1408_, lean_object* v_a_1409_, lean_object* v_a_1410_){
_start:
{
if (lean_obj_tag(v_a_1409_) == 0)
{
lean_object* v___x_1411_; 
lean_dec_ref(v_fo_1408_);
v___x_1411_ = l_List_reverse___redArg(v_a_1410_);
return v___x_1411_;
}
else
{
lean_object* v_toRingOps_1412_; lean_object* v_toSemiringOps_1413_; lean_object* v___x_1415_; uint8_t v_isShared_1416_; uint8_t v_isSharedCheck_1431_; 
v_toRingOps_1412_ = lean_ctor_get(v_fo_1408_, 0);
lean_inc_ref(v_toRingOps_1412_);
v_toSemiringOps_1413_ = lean_ctor_get(v_toRingOps_1412_, 0);
v_isSharedCheck_1431_ = !lean_is_exclusive(v_toRingOps_1412_);
if (v_isSharedCheck_1431_ == 0)
{
lean_object* v_unused_1432_; 
v_unused_1432_ = lean_ctor_get(v_toRingOps_1412_, 1);
lean_dec(v_unused_1432_);
v___x_1415_ = v_toRingOps_1412_;
v_isShared_1416_ = v_isSharedCheck_1431_;
goto v_resetjp_1414_;
}
else
{
lean_inc(v_toSemiringOps_1413_);
lean_dec(v_toRingOps_1412_);
v___x_1415_ = lean_box(0);
v_isShared_1416_ = v_isSharedCheck_1431_;
goto v_resetjp_1414_;
}
v_resetjp_1414_:
{
lean_object* v_head_1417_; lean_object* v_tail_1418_; lean_object* v___x_1420_; uint8_t v_isShared_1421_; uint8_t v_isSharedCheck_1430_; 
v_head_1417_ = lean_ctor_get(v_a_1409_, 0);
v_tail_1418_ = lean_ctor_get(v_a_1409_, 1);
v_isSharedCheck_1430_ = !lean_is_exclusive(v_a_1409_);
if (v_isSharedCheck_1430_ == 0)
{
v___x_1420_ = v_a_1409_;
v_isShared_1421_ = v_isSharedCheck_1430_;
goto v_resetjp_1419_;
}
else
{
lean_inc(v_tail_1418_);
lean_inc(v_head_1417_);
lean_dec(v_a_1409_);
v___x_1420_ = lean_box(0);
v_isShared_1421_ = v_isSharedCheck_1430_;
goto v_resetjp_1419_;
}
v_resetjp_1419_:
{
lean_object* v_zero_1422_; lean_object* v___x_1424_; 
v_zero_1422_ = lean_ctor_get(v_toSemiringOps_1413_, 0);
lean_inc(v_zero_1422_);
lean_dec_ref(v_toSemiringOps_1413_);
if (v_isShared_1416_ == 0)
{
lean_ctor_set(v___x_1415_, 1, v_zero_1422_);
lean_ctor_set(v___x_1415_, 0, v_head_1417_);
v___x_1424_ = v___x_1415_;
goto v_reusejp_1423_;
}
else
{
lean_object* v_reuseFailAlloc_1429_; 
v_reuseFailAlloc_1429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1429_, 0, v_head_1417_);
lean_ctor_set(v_reuseFailAlloc_1429_, 1, v_zero_1422_);
v___x_1424_ = v_reuseFailAlloc_1429_;
goto v_reusejp_1423_;
}
v_reusejp_1423_:
{
lean_object* v___x_1426_; 
if (v_isShared_1421_ == 0)
{
lean_ctor_set(v___x_1420_, 1, v_a_1410_);
lean_ctor_set(v___x_1420_, 0, v___x_1424_);
v___x_1426_ = v___x_1420_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v___x_1424_);
lean_ctor_set(v_reuseFailAlloc_1428_, 1, v_a_1410_);
v___x_1426_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
v_a_1409_ = v_tail_1418_;
v_a_1410_ = v___x_1426_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(lean_object* v_fo_1433_, lean_object* v_openings_1434_, uint8_t v_needRot_1435_){
_start:
{
if (v_needRot_1435_ == 0)
{
lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; 
v___x_1436_ = lean_box(0);
v___x_1437_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(v_fo_1433_, v_openings_1434_, v___x_1436_);
v___x_1438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1438_, 0, v___x_1437_);
return v___x_1438_;
}
else
{
lean_object* v___x_1439_; 
lean_dec_ref(v_fo_1433_);
v___x_1439_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_pairAdjacent___redArg(v_openings_1434_);
return v___x_1439_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg___boxed(lean_object* v_fo_1440_, lean_object* v_openings_1441_, lean_object* v_needRot_1442_){
_start:
{
uint8_t v_needRot_boxed_1443_; lean_object* v_res_1444_; 
v_needRot_boxed_1443_ = lean_unbox(v_needRot_1442_);
v_res_1444_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_1440_, v_openings_1441_, v_needRot_boxed_1443_);
return v_res_1444_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot(lean_object* v_EF_1445_, lean_object* v_fo_1446_, lean_object* v_openings_1447_, uint8_t v_needRot_1448_){
_start:
{
lean_object* v___x_1449_; 
v___x_1449_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_1446_, v_openings_1447_, v_needRot_1448_);
return v___x_1449_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___boxed(lean_object* v_EF_1450_, lean_object* v_fo_1451_, lean_object* v_openings_1452_, lean_object* v_needRot_1453_){
_start:
{
uint8_t v_needRot_boxed_1454_; lean_object* v_res_1455_; 
v_needRot_boxed_1454_ = lean_unbox(v_needRot_1453_);
v_res_1455_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot(v_EF_1450_, v_fo_1451_, v_openings_1452_, v_needRot_boxed_1454_);
return v_res_1455_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0(lean_object* v_EF_1456_, lean_object* v_fo_1457_, lean_object* v_a_1458_, lean_object* v_a_1459_){
_start:
{
lean_object* v___x_1460_; 
v___x_1460_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(v_fo_1457_, v_a_1458_, v_a_1459_);
return v___x_1460_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___lam__0(lean_object* v_inst_1461_, lean_object* v_inst_1462_, lean_object* v_x_1463_, lean_object* v_pair_1464_, lean_object* v___y_1465_){
_start:
{
lean_object* v_fst_1466_; lean_object* v_snd_1467_; lean_object* v___x_1468_; lean_object* v_a_1469_; lean_object* v_snd_1470_; lean_object* v___x_1471_; 
v_fst_1466_ = lean_ctor_get(v_pair_1464_, 0);
lean_inc(v_fst_1466_);
v_snd_1467_ = lean_ctor_get(v_pair_1464_, 1);
lean_inc(v_snd_1467_);
lean_dec_ref(v_pair_1464_);
lean_inc_ref(v_inst_1462_);
lean_inc_ref(v_inst_1461_);
v___x_1468_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_1461_, v_inst_1462_, v_fst_1466_, v___y_1465_);
v_a_1469_ = lean_ctor_get(v___x_1468_, 0);
lean_inc(v_a_1469_);
lean_dec_ref(v___x_1468_);
v_snd_1470_ = lean_ctor_get(v_a_1469_, 1);
lean_inc(v_snd_1470_);
lean_dec(v_a_1469_);
v___x_1471_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_1461_, v_inst_1462_, v_snd_1467_, v_snd_1470_);
return v___x_1471_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(lean_object* v_inst_1517_, lean_object* v_inst_1518_, lean_object* v_pairs_1519_, lean_object* v_a_1520_){
_start:
{
lean_object* v___f_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_362__overap_1524_; lean_object* v___x_1525_; 
v___f_1521_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___lam__0), 5, 2);
lean_closure_set(v___f_1521_, 0, v_inst_1517_);
lean_closure_set(v___f_1521_, 1, v_inst_1518_);
v___x_1522_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19));
v___x_1523_ = lean_box(0);
v___x_362__overap_1524_ = l_List_foldlM___redArg(v___x_1522_, v___f_1521_, v___x_1523_, v_pairs_1519_);
v___x_1525_ = lean_apply_1(v___x_362__overap_1524_, v_a_1520_);
if (lean_obj_tag(v___x_1525_) == 0)
{
return v___x_1525_;
}
else
{
lean_object* v_a_1526_; lean_object* v___x_1528_; uint8_t v_isShared_1529_; uint8_t v_isSharedCheck_1542_; 
v_a_1526_ = lean_ctor_get(v___x_1525_, 0);
v_isSharedCheck_1542_ = !lean_is_exclusive(v___x_1525_);
if (v_isSharedCheck_1542_ == 0)
{
v___x_1528_ = v___x_1525_;
v_isShared_1529_ = v_isSharedCheck_1542_;
goto v_resetjp_1527_;
}
else
{
lean_inc(v_a_1526_);
lean_dec(v___x_1525_);
v___x_1528_ = lean_box(0);
v_isShared_1529_ = v_isSharedCheck_1542_;
goto v_resetjp_1527_;
}
v_resetjp_1527_:
{
lean_object* v_snd_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1540_; 
v_snd_1530_ = lean_ctor_get(v_a_1526_, 1);
v_isSharedCheck_1540_ = !lean_is_exclusive(v_a_1526_);
if (v_isSharedCheck_1540_ == 0)
{
lean_object* v_unused_1541_; 
v_unused_1541_ = lean_ctor_get(v_a_1526_, 0);
lean_dec(v_unused_1541_);
v___x_1532_ = v_a_1526_;
v_isShared_1533_ = v_isSharedCheck_1540_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_snd_1530_);
lean_dec(v_a_1526_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1540_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v___x_1535_; 
if (v_isShared_1533_ == 0)
{
lean_ctor_set(v___x_1532_, 0, v___x_1523_);
v___x_1535_ = v___x_1532_;
goto v_reusejp_1534_;
}
else
{
lean_object* v_reuseFailAlloc_1539_; 
v_reuseFailAlloc_1539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1539_, 0, v___x_1523_);
lean_ctor_set(v_reuseFailAlloc_1539_, 1, v_snd_1530_);
v___x_1535_ = v_reuseFailAlloc_1539_;
goto v_reusejp_1534_;
}
v_reusejp_1534_:
{
lean_object* v___x_1537_; 
if (v_isShared_1529_ == 0)
{
lean_ctor_set(v___x_1528_, 0, v___x_1535_);
v___x_1537_ = v___x_1528_;
goto v_reusejp_1536_;
}
else
{
lean_object* v_reuseFailAlloc_1538_; 
v_reuseFailAlloc_1538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1538_, 0, v___x_1535_);
v___x_1537_ = v_reuseFailAlloc_1538_;
goto v_reusejp_1536_;
}
v_reusejp_1536_:
{
return v___x_1537_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM(lean_object* v_F_1543_, lean_object* v_EF_1544_, lean_object* v_inst_1545_, lean_object* v_inst_1546_, lean_object* v_pairs_1547_, lean_object* v_a_1548_){
_start:
{
lean_object* v___x_1549_; 
v___x_1549_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(v_inst_1545_, v_inst_1546_, v_pairs_1547_, v_a_1548_);
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(lean_object* v_inst_1553_, lean_object* v_inst_1554_, lean_object* v_fo_1555_, lean_object* v_openings_1556_, uint8_t v_needRot_1557_, lean_object* v_a_1558_){
_start:
{
lean_object* v___x_1559_; 
v___x_1559_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_1555_, v_openings_1556_, v_needRot_1557_);
if (lean_obj_tag(v___x_1559_) == 0)
{
lean_object* v___x_1560_; 
lean_dec_ref_known(v___x_1559_, 1);
lean_dec_ref(v_a_1558_);
lean_dec_ref(v_inst_1554_);
lean_dec_ref(v_inst_1553_);
v___x_1560_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1560_;
}
else
{
lean_object* v_a_1561_; lean_object* v___x_1562_; 
v_a_1561_ = lean_ctor_get(v___x_1559_, 0);
lean_inc(v_a_1561_);
lean_dec_ref_known(v___x_1559_, 1);
v___x_1562_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg(v_inst_1553_, v_inst_1554_, v_a_1561_, v_a_1558_);
return v___x_1562_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___boxed(lean_object* v_inst_1563_, lean_object* v_inst_1564_, lean_object* v_fo_1565_, lean_object* v_openings_1566_, lean_object* v_needRot_1567_, lean_object* v_a_1568_){
_start:
{
uint8_t v_needRot_boxed_1569_; lean_object* v_res_1570_; 
v_needRot_boxed_1569_ = lean_unbox(v_needRot_1567_);
v_res_1570_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1563_, v_inst_1564_, v_fo_1565_, v_openings_1566_, v_needRot_boxed_1569_, v_a_1568_);
return v_res_1570_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM(lean_object* v_F_1571_, lean_object* v_EF_1572_, lean_object* v_inst_1573_, lean_object* v_inst_1574_, lean_object* v_fo_1575_, lean_object* v_openings_1576_, uint8_t v_needRot_1577_, lean_object* v_a_1578_){
_start:
{
lean_object* v___x_1579_; 
v___x_1579_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1573_, v_inst_1574_, v_fo_1575_, v_openings_1576_, v_needRot_1577_, v_a_1578_);
return v___x_1579_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___boxed(lean_object* v_F_1580_, lean_object* v_EF_1581_, lean_object* v_inst_1582_, lean_object* v_inst_1583_, lean_object* v_fo_1584_, lean_object* v_openings_1585_, lean_object* v_needRot_1586_, lean_object* v_a_1587_){
_start:
{
uint8_t v_needRot_boxed_1588_; lean_object* v_res_1589_; 
v_needRot_boxed_1588_ = lean_unbox(v_needRot_1586_);
v_res_1589_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM(v_F_1580_, v_EF_1581_, v_inst_1582_, v_inst_1583_, v_fo_1584_, v_openings_1585_, v_needRot_boxed_1588_, v_a_1587_);
return v_res_1589_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0(lean_object* v_inst_1590_, lean_object* v_inst_1591_, lean_object* v_fo_1592_, lean_object* v_vk_1593_, lean_object* v_x_1594_, lean_object* v_entry_1595_, lean_object* v___y_1596_){
_start:
{
lean_object* v_fst_1597_; lean_object* v_snd_1598_; uint8_t v___y_1600_; lean_object* v___x_1606_; 
v_fst_1597_ = lean_ctor_get(v_entry_1595_, 0);
lean_inc(v_fst_1597_);
v_snd_1598_ = lean_ctor_get(v_entry_1595_, 1);
lean_inc(v_snd_1598_);
lean_dec_ref(v_entry_1595_);
v___x_1606_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_1593_, v_snd_1598_);
if (lean_obj_tag(v___x_1606_) == 0)
{
uint8_t v___x_1607_; 
v___x_1607_ = 0;
v___y_1600_ = v___x_1607_;
goto v___jp_1599_;
}
else
{
lean_object* v_val_1608_; lean_object* v_params_1609_; uint8_t v_needRot_1610_; 
v_val_1608_ = lean_ctor_get(v___x_1606_, 0);
lean_inc(v_val_1608_);
lean_dec_ref_known(v___x_1606_, 1);
v_params_1609_ = lean_ctor_get(v_val_1608_, 1);
lean_inc_ref(v_params_1609_);
lean_dec(v_val_1608_);
v_needRot_1610_ = lean_ctor_get_uint8(v_params_1609_, sizeof(void*)*2);
lean_dec_ref(v_params_1609_);
v___y_1600_ = v_needRot_1610_;
goto v___jp_1599_;
}
v___jp_1599_:
{
lean_object* v___x_1601_; lean_object* v___x_1602_; 
v___x_1601_ = lean_unsigned_to_nat(0u);
v___x_1602_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_fst_1597_, v___x_1601_);
lean_dec(v_fst_1597_);
if (lean_obj_tag(v___x_1602_) == 0)
{
lean_object* v___x_1603_; 
lean_dec_ref_known(v___x_1602_, 1);
lean_dec_ref(v___y_1596_);
lean_dec_ref(v_fo_1592_);
lean_dec_ref(v_inst_1591_);
lean_dec_ref(v_inst_1590_);
v___x_1603_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1603_;
}
else
{
lean_object* v_a_1604_; lean_object* v___x_1605_; 
v_a_1604_ = lean_ctor_get(v___x_1602_, 0);
lean_inc(v_a_1604_);
lean_dec_ref_known(v___x_1602_, 1);
v___x_1605_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1590_, v_inst_1591_, v_fo_1592_, v_a_1604_, v___y_1600_, v___y_1596_);
return v___x_1605_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0___boxed(lean_object* v_inst_1611_, lean_object* v_inst_1612_, lean_object* v_fo_1613_, lean_object* v_vk_1614_, lean_object* v_x_1615_, lean_object* v_entry_1616_, lean_object* v___y_1617_){
_start:
{
lean_object* v_res_1618_; 
v_res_1618_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0(v_inst_1611_, v_inst_1612_, v_fo_1613_, v_vk_1614_, v_x_1615_, v_entry_1616_, v___y_1617_);
lean_dec_ref(v_vk_1614_);
return v_res_1618_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1(lean_object* v_inst_1619_, lean_object* v_inst_1620_, lean_object* v_fo_1621_, uint8_t v_needRot_1622_, lean_object* v_x_1623_, lean_object* v_openings_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v___x_1626_; 
v___x_1626_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg(v_inst_1619_, v_inst_1620_, v_fo_1621_, v_openings_1624_, v_needRot_1622_, v___y_1625_);
return v___x_1626_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1___boxed(lean_object* v_inst_1627_, lean_object* v_inst_1628_, lean_object* v_fo_1629_, lean_object* v_needRot_1630_, lean_object* v_x_1631_, lean_object* v_openings_1632_, lean_object* v___y_1633_){
_start:
{
uint8_t v_needRot_boxed_1634_; lean_object* v_res_1635_; 
v_needRot_boxed_1634_ = lean_unbox(v_needRot_1630_);
v_res_1635_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1(v_inst_1627_, v_inst_1628_, v_fo_1629_, v_needRot_boxed_1634_, v_x_1631_, v_openings_1632_, v___y_1633_);
return v_res_1635_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2(lean_object* v_traceIdToAirId_1636_, lean_object* v_vk_1637_, lean_object* v_inst_1638_, lean_object* v_inst_1639_, lean_object* v_fo_1640_, lean_object* v___x_1641_, lean_object* v___x_1642_, lean_object* v_x_1643_, lean_object* v_entry_1644_, lean_object* v___y_1645_){
_start:
{
lean_object* v_fst_1646_; lean_object* v_snd_1647_; lean_object* v___x_1648_; 
v_fst_1646_ = lean_ctor_get(v_entry_1644_, 0);
lean_inc(v_fst_1646_);
v_snd_1647_ = lean_ctor_get(v_entry_1644_, 1);
lean_inc(v_snd_1647_);
lean_dec_ref(v_entry_1644_);
v___x_1648_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_traceIdToAirId_1636_, v_fst_1646_);
if (lean_obj_tag(v___x_1648_) == 0)
{
lean_object* v___x_1649_; 
lean_dec_ref_known(v___x_1648_, 1);
lean_dec(v_snd_1647_);
lean_dec_ref(v___y_1645_);
lean_dec_ref(v___x_1641_);
lean_dec_ref(v_fo_1640_);
lean_dec_ref(v_inst_1639_);
lean_dec_ref(v_inst_1638_);
v___x_1649_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1649_;
}
else
{
lean_object* v_a_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; 
v_a_1650_ = lean_ctor_get(v___x_1648_, 0);
lean_inc(v_a_1650_);
lean_dec_ref_known(v___x_1648_, 1);
v___x_1651_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_1637_, v_a_1650_);
v___x_1652_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v___x_1651_);
if (lean_obj_tag(v___x_1652_) == 0)
{
lean_object* v___x_1653_; 
lean_dec_ref_known(v___x_1652_, 1);
lean_dec(v_snd_1647_);
lean_dec_ref(v___y_1645_);
lean_dec_ref(v___x_1641_);
lean_dec_ref(v_fo_1640_);
lean_dec_ref(v_inst_1639_);
lean_dec_ref(v_inst_1638_);
v___x_1653_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningPartM___redArg___closed__0));
return v___x_1653_;
}
else
{
lean_object* v_a_1654_; lean_object* v_params_1655_; uint8_t v_needRot_1656_; lean_object* v___x_1657_; lean_object* v___f_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1893__overap_1661_; lean_object* v___x_1662_; 
v_a_1654_ = lean_ctor_get(v___x_1652_, 0);
lean_inc(v_a_1654_);
lean_dec_ref_known(v___x_1652_, 1);
v_params_1655_ = lean_ctor_get(v_a_1654_, 1);
lean_inc_ref(v_params_1655_);
lean_dec(v_a_1654_);
v_needRot_1656_ = lean_ctor_get_uint8(v_params_1655_, sizeof(void*)*2);
lean_dec_ref(v_params_1655_);
v___x_1657_ = lean_box(v_needRot_1656_);
v___f_1658_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__1___boxed), 7, 4);
lean_closure_set(v___f_1658_, 0, v_inst_1638_);
lean_closure_set(v___f_1658_, 1, v_inst_1639_);
lean_closure_set(v___f_1658_, 2, v_fo_1640_);
lean_closure_set(v___f_1658_, 3, v___x_1657_);
v___x_1659_ = lean_unsigned_to_nat(1u);
v___x_1660_ = l_List_drop___redArg(v___x_1659_, v_snd_1647_);
lean_dec(v_snd_1647_);
v___x_1893__overap_1661_ = l_List_foldlM___redArg(v___x_1641_, v___f_1658_, v___x_1642_, v___x_1660_);
v___x_1662_ = lean_apply_1(v___x_1893__overap_1661_, v___y_1645_);
if (lean_obj_tag(v___x_1662_) == 0)
{
return v___x_1662_;
}
else
{
lean_object* v_a_1663_; lean_object* v___x_1665_; uint8_t v_isShared_1666_; uint8_t v_isSharedCheck_1679_; 
v_a_1663_ = lean_ctor_get(v___x_1662_, 0);
v_isSharedCheck_1679_ = !lean_is_exclusive(v___x_1662_);
if (v_isSharedCheck_1679_ == 0)
{
v___x_1665_ = v___x_1662_;
v_isShared_1666_ = v_isSharedCheck_1679_;
goto v_resetjp_1664_;
}
else
{
lean_inc(v_a_1663_);
lean_dec(v___x_1662_);
v___x_1665_ = lean_box(0);
v_isShared_1666_ = v_isSharedCheck_1679_;
goto v_resetjp_1664_;
}
v_resetjp_1664_:
{
lean_object* v_snd_1667_; lean_object* v___x_1669_; uint8_t v_isShared_1670_; uint8_t v_isSharedCheck_1677_; 
v_snd_1667_ = lean_ctor_get(v_a_1663_, 1);
v_isSharedCheck_1677_ = !lean_is_exclusive(v_a_1663_);
if (v_isSharedCheck_1677_ == 0)
{
lean_object* v_unused_1678_; 
v_unused_1678_ = lean_ctor_get(v_a_1663_, 0);
lean_dec(v_unused_1678_);
v___x_1669_ = v_a_1663_;
v_isShared_1670_ = v_isSharedCheck_1677_;
goto v_resetjp_1668_;
}
else
{
lean_inc(v_snd_1667_);
lean_dec(v_a_1663_);
v___x_1669_ = lean_box(0);
v_isShared_1670_ = v_isSharedCheck_1677_;
goto v_resetjp_1668_;
}
v_resetjp_1668_:
{
lean_object* v___x_1672_; 
if (v_isShared_1670_ == 0)
{
lean_ctor_set(v___x_1669_, 0, v___x_1642_);
v___x_1672_ = v___x_1669_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1676_; 
v_reuseFailAlloc_1676_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1676_, 0, v___x_1642_);
lean_ctor_set(v_reuseFailAlloc_1676_, 1, v_snd_1667_);
v___x_1672_ = v_reuseFailAlloc_1676_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
lean_object* v___x_1674_; 
if (v_isShared_1666_ == 0)
{
lean_ctor_set(v___x_1665_, 0, v___x_1672_);
v___x_1674_ = v___x_1665_;
goto v_reusejp_1673_;
}
else
{
lean_object* v_reuseFailAlloc_1675_; 
v_reuseFailAlloc_1675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1675_, 0, v___x_1672_);
v___x_1674_ = v_reuseFailAlloc_1675_;
goto v_reusejp_1673_;
}
v_reusejp_1673_:
{
return v___x_1674_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2___boxed(lean_object* v_traceIdToAirId_1680_, lean_object* v_vk_1681_, lean_object* v_inst_1682_, lean_object* v_inst_1683_, lean_object* v_fo_1684_, lean_object* v___x_1685_, lean_object* v___x_1686_, lean_object* v_x_1687_, lean_object* v_entry_1688_, lean_object* v___y_1689_){
_start:
{
lean_object* v_res_1690_; 
v_res_1690_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2(v_traceIdToAirId_1680_, v_vk_1681_, v_inst_1682_, v_inst_1683_, v_fo_1684_, v___x_1685_, v___x_1686_, v_x_1687_, v_entry_1688_, v___y_1689_);
lean_dec_ref(v_vk_1681_);
lean_dec(v_traceIdToAirId_1680_);
return v_res_1690_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(lean_object* v_inst_1691_, lean_object* v_inst_1692_, lean_object* v_fo_1693_, lean_object* v_vk_1694_, lean_object* v_columnOpenings_1695_, lean_object* v_traceIdToAirId_1696_, lean_object* v_a_1697_){
_start:
{
lean_object* v___f_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1396__overap_1702_; lean_object* v___x_1703_; 
lean_inc_ref(v_vk_1694_);
lean_inc_ref(v_fo_1693_);
lean_inc_ref(v_inst_1692_);
lean_inc_ref(v_inst_1691_);
v___f_1698_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__0___boxed), 7, 4);
lean_closure_set(v___f_1698_, 0, v_inst_1691_);
lean_closure_set(v___f_1698_, 1, v_inst_1692_);
lean_closure_set(v___f_1698_, 2, v_fo_1693_);
lean_closure_set(v___f_1698_, 3, v_vk_1694_);
v___x_1699_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19));
v___x_1700_ = lean_box(0);
lean_inc(v_traceIdToAirId_1696_);
lean_inc(v_columnOpenings_1695_);
v___x_1701_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_columnOpenings_1695_, v_traceIdToAirId_1696_);
v___x_1396__overap_1702_ = l_List_foldlM___redArg(v___x_1699_, v___f_1698_, v___x_1700_, v___x_1701_);
v___x_1703_ = lean_apply_1(v___x_1396__overap_1702_, v_a_1697_);
if (lean_obj_tag(v___x_1703_) == 0)
{
lean_dec(v_traceIdToAirId_1696_);
lean_dec(v_columnOpenings_1695_);
lean_dec_ref(v_vk_1694_);
lean_dec_ref(v_fo_1693_);
lean_dec_ref(v_inst_1692_);
lean_dec_ref(v_inst_1691_);
return v___x_1703_;
}
else
{
lean_object* v_a_1704_; lean_object* v_snd_1705_; lean_object* v___f_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1609__overap_1710_; lean_object* v___x_1711_; 
v_a_1704_ = lean_ctor_get(v___x_1703_, 0);
lean_inc(v_a_1704_);
lean_dec_ref_known(v___x_1703_, 1);
v_snd_1705_ = lean_ctor_get(v_a_1704_, 1);
lean_inc(v_snd_1705_);
lean_dec(v_a_1704_);
v___f_1706_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg___lam__2___boxed), 10, 7);
lean_closure_set(v___f_1706_, 0, v_traceIdToAirId_1696_);
lean_closure_set(v___f_1706_, 1, v_vk_1694_);
lean_closure_set(v___f_1706_, 2, v_inst_1691_);
lean_closure_set(v___f_1706_, 3, v_inst_1692_);
lean_closure_set(v___f_1706_, 4, v_fo_1693_);
lean_closure_set(v___f_1706_, 5, v___x_1699_);
lean_closure_set(v___f_1706_, 6, v___x_1700_);
v___x_1707_ = l_List_lengthTR___redArg(v_columnOpenings_1695_);
v___x_1708_ = l_List_range(v___x_1707_);
v___x_1709_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1708_, v_columnOpenings_1695_);
v___x_1609__overap_1710_ = l_List_foldlM___redArg(v___x_1699_, v___f_1706_, v___x_1700_, v___x_1709_);
v___x_1711_ = lean_apply_1(v___x_1609__overap_1710_, v_snd_1705_);
if (lean_obj_tag(v___x_1711_) == 0)
{
return v___x_1711_;
}
else
{
lean_object* v_a_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1728_; 
v_a_1712_ = lean_ctor_get(v___x_1711_, 0);
v_isSharedCheck_1728_ = !lean_is_exclusive(v___x_1711_);
if (v_isSharedCheck_1728_ == 0)
{
v___x_1714_ = v___x_1711_;
v_isShared_1715_ = v_isSharedCheck_1728_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_a_1712_);
lean_dec(v___x_1711_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1728_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v_snd_1716_; lean_object* v___x_1718_; uint8_t v_isShared_1719_; uint8_t v_isSharedCheck_1726_; 
v_snd_1716_ = lean_ctor_get(v_a_1712_, 1);
v_isSharedCheck_1726_ = !lean_is_exclusive(v_a_1712_);
if (v_isSharedCheck_1726_ == 0)
{
lean_object* v_unused_1727_; 
v_unused_1727_ = lean_ctor_get(v_a_1712_, 0);
lean_dec(v_unused_1727_);
v___x_1718_ = v_a_1712_;
v_isShared_1719_ = v_isSharedCheck_1726_;
goto v_resetjp_1717_;
}
else
{
lean_inc(v_snd_1716_);
lean_dec(v_a_1712_);
v___x_1718_ = lean_box(0);
v_isShared_1719_ = v_isSharedCheck_1726_;
goto v_resetjp_1717_;
}
v_resetjp_1717_:
{
lean_object* v___x_1721_; 
if (v_isShared_1719_ == 0)
{
lean_ctor_set(v___x_1718_, 0, v___x_1700_);
v___x_1721_ = v___x_1718_;
goto v_reusejp_1720_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v___x_1700_);
lean_ctor_set(v_reuseFailAlloc_1725_, 1, v_snd_1716_);
v___x_1721_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1720_;
}
v_reusejp_1720_:
{
lean_object* v___x_1723_; 
if (v_isShared_1715_ == 0)
{
lean_ctor_set(v___x_1714_, 0, v___x_1721_);
v___x_1723_ = v___x_1714_;
goto v_reusejp_1722_;
}
else
{
lean_object* v_reuseFailAlloc_1724_; 
v_reuseFailAlloc_1724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1724_, 0, v___x_1721_);
v___x_1723_ = v_reuseFailAlloc_1724_;
goto v_reusejp_1722_;
}
v_reusejp_1722_:
{
return v___x_1723_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM(lean_object* v_F_1729_, lean_object* v_EF_1730_, lean_object* v_Digest_1731_, lean_object* v_inst_1732_, lean_object* v_inst_1733_, lean_object* v_fo_1734_, lean_object* v_vk_1735_, lean_object* v_columnOpenings_1736_, lean_object* v_traceIdToAirId_1737_, lean_object* v_a_1738_){
_start:
{
lean_object* v___x_1739_; 
v___x_1739_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(v_inst_1732_, v_inst_1733_, v_fo_1734_, v_vk_1735_, v_columnOpenings_1736_, v_traceIdToAirId_1737_, v_a_1738_);
return v___x_1739_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___lam__0(lean_object* v_fo_1740_, lean_object* v_acc_1741_, lean_object* v_x_1742_){
_start:
{
lean_object* v_toRingOps_1743_; lean_object* v_sub_1744_; lean_object* v___x_1745_; 
v_toRingOps_1743_ = lean_ctor_get(v_fo_1740_, 0);
lean_inc_ref(v_toRingOps_1743_);
lean_dec_ref(v_fo_1740_);
v_sub_1744_ = lean_ctor_get(v_toRingOps_1743_, 1);
lean_inc(v_sub_1744_);
lean_dec_ref(v_toRingOps_1743_);
v___x_1745_ = lean_apply_2(v_sub_1744_, v_acc_1741_, v_x_1742_);
return v___x_1745_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(lean_object* v_inst_1751_, lean_object* v_fo_1752_, lean_object* v_numeratorTermPerAir_1753_, lean_object* v_denominatorTermPerAir_1754_, lean_object* v_alphaLogup_1755_, lean_object* v_pXiClaim_1756_, lean_object* v_qXiClaim_1757_){
_start:
{
lean_object* v_toRingOps_1760_; lean_object* v_toSemiringOps_1761_; lean_object* v_zero_1762_; lean_object* v___f_1763_; lean_object* v_pFinal_1764_; lean_object* v_qFinal_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; uint8_t v___x_1768_; 
v_toRingOps_1760_ = lean_ctor_get(v_fo_1752_, 0);
v_toSemiringOps_1761_ = lean_ctor_get(v_toRingOps_1760_, 0);
v_zero_1762_ = lean_ctor_get(v_toSemiringOps_1761_, 0);
lean_inc(v_zero_1762_);
v___f_1763_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1763_, 0, v_fo_1752_);
lean_inc_ref(v___f_1763_);
v_pFinal_1764_ = l_List_foldl___redArg(v___f_1763_, v_pXiClaim_1756_, v_numeratorTermPerAir_1753_);
v_qFinal_1765_ = l_List_foldl___redArg(v___f_1763_, v_qXiClaim_1757_, v_denominatorTermPerAir_1754_);
lean_inc_ref(v_inst_1751_);
v___x_1766_ = lean_apply_2(v_inst_1751_, v_qFinal_1765_, v_alphaLogup_1755_);
v___x_1767_ = lean_apply_2(v_inst_1751_, v_pFinal_1764_, v_zero_1762_);
v___x_1768_ = lean_unbox(v___x_1767_);
if (v___x_1768_ == 0)
{
goto v___jp_1758_;
}
else
{
uint8_t v___x_1769_; 
v___x_1769_ = lean_unbox(v___x_1766_);
if (v___x_1769_ == 0)
{
goto v___jp_1758_;
}
else
{
lean_object* v___x_1770_; 
v___x_1770_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__1));
return v___x_1770_;
}
}
v___jp_1758_:
{
lean_object* v___x_1759_; 
v___x_1759_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg___closed__0));
return v___x_1759_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi(lean_object* v_EF_1771_, lean_object* v_inst_1772_, lean_object* v_fo_1773_, lean_object* v_numeratorTermPerAir_1774_, lean_object* v_denominatorTermPerAir_1775_, lean_object* v_alphaLogup_1776_, lean_object* v_pXiClaim_1777_, lean_object* v_qXiClaim_1778_){
_start:
{
lean_object* v___x_1779_; 
v___x_1779_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(v_inst_1772_, v_fo_1773_, v_numeratorTermPerAir_1774_, v_denominatorTermPerAir_1775_, v_alphaLogup_1776_, v_pXiClaim_1777_, v_qXiClaim_1778_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(lean_object* v_fo_1780_, lean_object* v_mu_1781_, lean_object* v_x_1782_, lean_object* v_x_1783_){
_start:
{
if (lean_obj_tag(v_x_1783_) == 0)
{
lean_dec(v_mu_1781_);
lean_dec_ref(v_fo_1780_);
return v_x_1782_;
}
else
{
lean_object* v_head_1784_; lean_object* v_toRingOps_1785_; lean_object* v_toSemiringOps_1786_; lean_object* v_tail_1787_; lean_object* v_fst_1788_; lean_object* v_snd_1789_; lean_object* v_fst_1790_; lean_object* v_snd_1791_; lean_object* v___x_1793_; uint8_t v_isShared_1794_; uint8_t v_isSharedCheck_1807_; 
v_head_1784_ = lean_ctor_get(v_x_1783_, 0);
lean_inc(v_head_1784_);
v_toRingOps_1785_ = lean_ctor_get(v_fo_1780_, 0);
v_toSemiringOps_1786_ = lean_ctor_get(v_toRingOps_1785_, 0);
v_tail_1787_ = lean_ctor_get(v_x_1783_, 1);
lean_inc(v_tail_1787_);
lean_dec_ref_known(v_x_1783_, 2);
v_fst_1788_ = lean_ctor_get(v_x_1782_, 0);
lean_inc(v_fst_1788_);
v_snd_1789_ = lean_ctor_get(v_x_1782_, 1);
lean_inc(v_snd_1789_);
lean_dec_ref(v_x_1782_);
v_fst_1790_ = lean_ctor_get(v_head_1784_, 0);
v_snd_1791_ = lean_ctor_get(v_head_1784_, 1);
v_isSharedCheck_1807_ = !lean_is_exclusive(v_head_1784_);
if (v_isSharedCheck_1807_ == 0)
{
v___x_1793_ = v_head_1784_;
v_isShared_1794_ = v_isSharedCheck_1807_;
goto v_resetjp_1792_;
}
else
{
lean_inc(v_snd_1791_);
lean_inc(v_fst_1790_);
lean_dec(v_head_1784_);
v___x_1793_ = lean_box(0);
v_isShared_1794_ = v_isSharedCheck_1807_;
goto v_resetjp_1792_;
}
v_resetjp_1792_:
{
lean_object* v_add_1795_; lean_object* v_mul_1796_; lean_object* v___x_1797_; lean_object* v_sumClaim_1798_; lean_object* v_curMuPow_1799_; lean_object* v___x_1800_; lean_object* v_sumClaim_1801_; lean_object* v___x_1802_; lean_object* v___x_1804_; 
v_add_1795_ = lean_ctor_get(v_toSemiringOps_1786_, 3);
v_mul_1796_ = lean_ctor_get(v_toSemiringOps_1786_, 4);
lean_inc_n(v_mul_1796_, 4);
lean_inc(v_fst_1788_);
v___x_1797_ = lean_apply_2(v_mul_1796_, v_fst_1790_, v_fst_1788_);
lean_inc_n(v_add_1795_, 2);
v_sumClaim_1798_ = lean_apply_2(v_add_1795_, v_snd_1789_, v___x_1797_);
lean_inc_n(v_mu_1781_, 2);
v_curMuPow_1799_ = lean_apply_2(v_mul_1796_, v_fst_1788_, v_mu_1781_);
lean_inc(v_curMuPow_1799_);
v___x_1800_ = lean_apply_2(v_mul_1796_, v_snd_1791_, v_curMuPow_1799_);
v_sumClaim_1801_ = lean_apply_2(v_add_1795_, v_sumClaim_1798_, v___x_1800_);
v___x_1802_ = lean_apply_2(v_mul_1796_, v_curMuPow_1799_, v_mu_1781_);
if (v_isShared_1794_ == 0)
{
lean_ctor_set(v___x_1793_, 1, v_sumClaim_1801_);
lean_ctor_set(v___x_1793_, 0, v___x_1802_);
v___x_1804_ = v___x_1793_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1806_; 
v_reuseFailAlloc_1806_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1806_, 0, v___x_1802_);
lean_ctor_set(v_reuseFailAlloc_1806_, 1, v_sumClaim_1801_);
v___x_1804_ = v_reuseFailAlloc_1806_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
v_x_1782_ = v___x_1804_;
v_x_1783_ = v_tail_1787_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(lean_object* v_fo_1808_, lean_object* v_numeratorTermPerAir_1809_, lean_object* v_denominatorTermPerAir_1810_, lean_object* v_mu_1811_){
_start:
{
lean_object* v_toRingOps_1812_; lean_object* v_toSemiringOps_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1825_; 
v_toRingOps_1812_ = lean_ctor_get(v_fo_1808_, 0);
lean_inc_ref(v_toRingOps_1812_);
v_toSemiringOps_1813_ = lean_ctor_get(v_toRingOps_1812_, 0);
v_isSharedCheck_1825_ = !lean_is_exclusive(v_toRingOps_1812_);
if (v_isSharedCheck_1825_ == 0)
{
lean_object* v_unused_1826_; 
v_unused_1826_ = lean_ctor_get(v_toRingOps_1812_, 1);
lean_dec(v_unused_1826_);
v___x_1815_ = v_toRingOps_1812_;
v_isShared_1816_ = v_isSharedCheck_1825_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_toSemiringOps_1813_);
lean_dec(v_toRingOps_1812_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1825_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v_zero_1817_; lean_object* v_one_1818_; lean_object* v___x_1820_; 
v_zero_1817_ = lean_ctor_get(v_toSemiringOps_1813_, 0);
lean_inc(v_zero_1817_);
v_one_1818_ = lean_ctor_get(v_toSemiringOps_1813_, 1);
lean_inc(v_one_1818_);
lean_dec_ref(v_toSemiringOps_1813_);
if (v_isShared_1816_ == 0)
{
lean_ctor_set(v___x_1815_, 1, v_zero_1817_);
lean_ctor_set(v___x_1815_, 0, v_one_1818_);
v___x_1820_ = v___x_1815_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1824_; 
v_reuseFailAlloc_1824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1824_, 0, v_one_1818_);
lean_ctor_set(v_reuseFailAlloc_1824_, 1, v_zero_1817_);
v___x_1820_ = v_reuseFailAlloc_1824_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v_snd_1823_; 
v___x_1821_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_numeratorTermPerAir_1809_, v_denominatorTermPerAir_1810_);
v___x_1822_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(v_fo_1808_, v_mu_1811_, v___x_1820_, v___x_1821_);
v_snd_1823_ = lean_ctor_get(v___x_1822_, 1);
lean_inc(v_snd_1823_);
lean_dec_ref(v___x_1822_);
return v_snd_1823_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim(lean_object* v_EF_1827_, lean_object* v_fo_1828_, lean_object* v_numeratorTermPerAir_1829_, lean_object* v_denominatorTermPerAir_1830_, lean_object* v_mu_1831_){
_start:
{
lean_object* v___x_1832_; 
v___x_1832_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(v_fo_1828_, v_numeratorTermPerAir_1829_, v_denominatorTermPerAir_1830_, v_mu_1831_);
return v___x_1832_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0(lean_object* v_EF_1833_, lean_object* v_fo_1834_, lean_object* v_mu_1835_, lean_object* v_x_1836_, lean_object* v_x_1837_){
_start:
{
lean_object* v___x_1838_; 
v___x_1838_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim_spec__0___redArg(v_fo_1834_, v_mu_1835_, v_x_1836_, v_x_1837_);
return v___x_1838_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(lean_object* v_inst_1839_, lean_object* v_fo_1840_, lean_object* v_univariateRoundCoeffs_1841_, lean_object* v_sumClaim_1842_, lean_object* v_lSkip_1843_, lean_object* v_r0_1844_){
_start:
{
lean_object* v_sumUnivDomainS0_1845_; lean_object* v___x_1846_; uint8_t v___x_1847_; 
lean_inc(v_univariateRoundCoeffs_1841_);
lean_inc_ref(v_fo_1840_);
v_sumUnivDomainS0_1845_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumOverSkipDomain___redArg(v_fo_1840_, v_univariateRoundCoeffs_1841_, v_lSkip_1843_);
v___x_1846_ = lean_apply_2(v_inst_1839_, v_sumClaim_1842_, v_sumUnivDomainS0_1845_);
v___x_1847_ = lean_unbox(v___x_1846_);
if (v___x_1847_ == 0)
{
lean_object* v___x_1848_; 
lean_dec(v_r0_1844_);
lean_dec(v_univariateRoundCoeffs_1841_);
lean_dec_ref(v_fo_1840_);
v___x_1848_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg___closed__0));
return v___x_1848_;
}
else
{
lean_object* v_toRingOps_1849_; lean_object* v_toSemiringOps_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; 
v_toRingOps_1849_ = lean_ctor_get(v_fo_1840_, 0);
lean_inc_ref(v_toRingOps_1849_);
lean_dec_ref(v_fo_1840_);
v_toSemiringOps_1850_ = lean_ctor_get(v_toRingOps_1849_, 0);
lean_inc_ref(v_toSemiringOps_1850_);
lean_dec_ref(v_toRingOps_1849_);
v___x_1851_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_toSemiringOps_1850_, v_univariateRoundCoeffs_1841_, v_r0_1844_);
v___x_1852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1852_, 0, v___x_1851_);
return v___x_1852_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg___boxed(lean_object* v_inst_1853_, lean_object* v_fo_1854_, lean_object* v_univariateRoundCoeffs_1855_, lean_object* v_sumClaim_1856_, lean_object* v_lSkip_1857_, lean_object* v_r0_1858_){
_start:
{
lean_object* v_res_1859_; 
v_res_1859_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(v_inst_1853_, v_fo_1854_, v_univariateRoundCoeffs_1855_, v_sumClaim_1856_, v_lSkip_1857_, v_r0_1858_);
lean_dec(v_lSkip_1857_);
return v_res_1859_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound(lean_object* v_EF_1860_, lean_object* v_inst_1861_, lean_object* v_fo_1862_, lean_object* v_univariateRoundCoeffs_1863_, lean_object* v_sumClaim_1864_, lean_object* v_lSkip_1865_, lean_object* v_r0_1866_){
_start:
{
lean_object* v___x_1867_; 
v___x_1867_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(v_inst_1861_, v_fo_1862_, v_univariateRoundCoeffs_1863_, v_sumClaim_1864_, v_lSkip_1865_, v_r0_1866_);
return v___x_1867_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___boxed(lean_object* v_EF_1868_, lean_object* v_inst_1869_, lean_object* v_fo_1870_, lean_object* v_univariateRoundCoeffs_1871_, lean_object* v_sumClaim_1872_, lean_object* v_lSkip_1873_, lean_object* v_r0_1874_){
_start:
{
lean_object* v_res_1875_; 
v_res_1875_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound(v_EF_1868_, v_inst_1869_, v_fo_1870_, v_univariateRoundCoeffs_1871_, v_sumClaim_1872_, v_lSkip_1873_, v_r0_1874_);
lean_dec(v_lSkip_1873_);
return v_res_1875_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0(lean_object* v_mul_1876_, lean_object* v_acc_1877_, lean_object* v_x_1878_){
_start:
{
lean_object* v___x_1879_; 
v___x_1879_ = lean_apply_2(v_mul_1876_, v_acc_1877_, v_x_1878_);
return v___x_1879_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(lean_object* v___x_1880_, lean_object* v_a_1881_, lean_object* v_a_1882_){
_start:
{
if (lean_obj_tag(v_a_1881_) == 0)
{
lean_object* v___x_1883_; 
lean_dec_ref(v___x_1880_);
v___x_1883_ = l_List_reverse___redArg(v_a_1882_);
return v___x_1883_;
}
else
{
lean_object* v_head_1884_; lean_object* v_tail_1885_; lean_object* v___x_1887_; uint8_t v_isShared_1888_; uint8_t v_isSharedCheck_1897_; 
v_head_1884_ = lean_ctor_get(v_a_1881_, 0);
v_tail_1885_ = lean_ctor_get(v_a_1881_, 1);
v_isSharedCheck_1897_ = !lean_is_exclusive(v_a_1881_);
if (v_isSharedCheck_1897_ == 0)
{
v___x_1887_ = v_a_1881_;
v_isShared_1888_ = v_isSharedCheck_1897_;
goto v_resetjp_1886_;
}
else
{
lean_inc(v_tail_1885_);
lean_inc(v_head_1884_);
lean_dec(v_a_1881_);
v___x_1887_ = lean_box(0);
v_isShared_1888_ = v_isSharedCheck_1897_;
goto v_resetjp_1886_;
}
v_resetjp_1886_:
{
lean_object* v_natCast_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1894_; 
v_natCast_1889_ = lean_ctor_get(v___x_1880_, 2);
v___x_1890_ = lean_unsigned_to_nat(1u);
v___x_1891_ = lean_nat_add(v_head_1884_, v___x_1890_);
lean_dec(v_head_1884_);
lean_inc(v_natCast_1889_);
v___x_1892_ = lean_apply_1(v_natCast_1889_, v___x_1891_);
if (v_isShared_1888_ == 0)
{
lean_ctor_set(v___x_1887_, 1, v_a_1882_);
lean_ctor_set(v___x_1887_, 0, v___x_1892_);
v___x_1894_ = v___x_1887_;
goto v_reusejp_1893_;
}
else
{
lean_object* v_reuseFailAlloc_1896_; 
v_reuseFailAlloc_1896_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1896_, 0, v___x_1892_);
lean_ctor_set(v_reuseFailAlloc_1896_, 1, v_a_1882_);
v___x_1894_ = v_reuseFailAlloc_1896_;
goto v_reusejp_1893_;
}
v_reusejp_1893_:
{
v_a_1881_ = v_tail_1885_;
v_a_1882_ = v___x_1894_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(lean_object* v___x_1898_, lean_object* v___x_1899_, lean_object* v_r_1900_, lean_object* v_a_1901_, lean_object* v_a_1902_){
_start:
{
if (lean_obj_tag(v_a_1901_) == 0)
{
lean_object* v___x_1903_; 
lean_dec(v_r_1900_);
lean_dec_ref(v___x_1899_);
lean_dec_ref(v___x_1898_);
v___x_1903_ = l_List_reverse___redArg(v_a_1902_);
return v___x_1903_;
}
else
{
lean_object* v_head_1904_; lean_object* v_tail_1905_; lean_object* v___x_1907_; uint8_t v_isShared_1908_; uint8_t v_isSharedCheck_1917_; 
v_head_1904_ = lean_ctor_get(v_a_1901_, 0);
v_tail_1905_ = lean_ctor_get(v_a_1901_, 1);
v_isSharedCheck_1917_ = !lean_is_exclusive(v_a_1901_);
if (v_isSharedCheck_1917_ == 0)
{
v___x_1907_ = v_a_1901_;
v_isShared_1908_ = v_isSharedCheck_1917_;
goto v_resetjp_1906_;
}
else
{
lean_inc(v_tail_1905_);
lean_inc(v_head_1904_);
lean_dec(v_a_1901_);
v___x_1907_ = lean_box(0);
v_isShared_1908_ = v_isSharedCheck_1917_;
goto v_resetjp_1906_;
}
v_resetjp_1906_:
{
lean_object* v_sub_1909_; lean_object* v_natCast_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1914_; 
v_sub_1909_ = lean_ctor_get(v___x_1898_, 1);
v_natCast_1910_ = lean_ctor_get(v___x_1899_, 2);
lean_inc(v_natCast_1910_);
v___x_1911_ = lean_apply_1(v_natCast_1910_, v_head_1904_);
lean_inc(v_sub_1909_);
lean_inc(v_r_1900_);
v___x_1912_ = lean_apply_2(v_sub_1909_, v_r_1900_, v___x_1911_);
if (v_isShared_1908_ == 0)
{
lean_ctor_set(v___x_1907_, 1, v_a_1902_);
lean_ctor_set(v___x_1907_, 0, v___x_1912_);
v___x_1914_ = v___x_1907_;
goto v_reusejp_1913_;
}
else
{
lean_object* v_reuseFailAlloc_1916_; 
v_reuseFailAlloc_1916_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1916_, 0, v___x_1912_);
lean_ctor_set(v_reuseFailAlloc_1916_, 1, v_a_1902_);
v___x_1914_ = v_reuseFailAlloc_1916_;
goto v_reusejp_1913_;
}
v_reusejp_1913_:
{
v_a_1901_ = v_tail_1905_;
v_a_1902_ = v___x_1914_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(lean_object* v___x_1918_, lean_object* v_sDeg_1919_, lean_object* v_invfact_1920_, lean_object* v_sufProduct_1921_, lean_object* v_prefProduct_1922_, lean_object* v_evals_1923_, lean_object* v_x_1924_, lean_object* v_x_1925_){
_start:
{
if (lean_obj_tag(v_x_1925_) == 0)
{
lean_dec_ref(v___x_1918_);
return v_x_1924_;
}
else
{
lean_object* v_head_1926_; lean_object* v_tail_1927_; lean_object* v_zero_1928_; lean_object* v_add_1929_; lean_object* v_mul_1930_; lean_object* v___y_1932_; lean_object* v___y_1933_; lean_object* v___y_1938_; lean_object* v___y_1939_; lean_object* v___y_1940_; lean_object* v___y_1945_; lean_object* v___y_1946_; lean_object* v___y_1947_; lean_object* v___y_1952_; lean_object* v___y_1953_; lean_object* v___y_1959_; lean_object* v___x_1962_; 
v_head_1926_ = lean_ctor_get(v_x_1925_, 0);
lean_inc_n(v_head_1926_, 2);
v_tail_1927_ = lean_ctor_get(v_x_1925_, 1);
lean_inc(v_tail_1927_);
lean_dec_ref_known(v_x_1925_, 2);
v_zero_1928_ = lean_ctor_get(v___x_1918_, 0);
v_add_1929_ = lean_ctor_get(v___x_1918_, 3);
v_mul_1930_ = lean_ctor_get(v___x_1918_, 4);
v___x_1962_ = l_List_get_x3fInternal___redArg(v_evals_1923_, v_head_1926_);
if (lean_obj_tag(v___x_1962_) == 0)
{
lean_inc(v_zero_1928_);
v___y_1959_ = v_zero_1928_;
goto v___jp_1958_;
}
else
{
lean_object* v_val_1963_; 
v_val_1963_ = lean_ctor_get(v___x_1962_, 0);
lean_inc(v_val_1963_);
lean_dec_ref_known(v___x_1962_, 1);
v___y_1959_ = v_val_1963_;
goto v___jp_1958_;
}
v___jp_1931_:
{
lean_object* v___x_1934_; lean_object* v___x_1935_; 
lean_inc(v_mul_1930_);
v___x_1934_ = lean_apply_2(v_mul_1930_, v___y_1932_, v___y_1933_);
lean_inc(v_add_1929_);
v___x_1935_ = lean_apply_2(v_add_1929_, v_x_1924_, v___x_1934_);
v_x_1924_ = v___x_1935_;
v_x_1925_ = v_tail_1927_;
goto _start;
}
v___jp_1937_:
{
lean_object* v___x_1941_; lean_object* v___x_1942_; 
lean_inc(v_mul_1930_);
v___x_1941_ = lean_apply_2(v_mul_1930_, v___y_1939_, v___y_1940_);
v___x_1942_ = l_List_get_x3fInternal___redArg(v_invfact_1920_, v___y_1938_);
if (lean_obj_tag(v___x_1942_) == 0)
{
lean_inc(v_zero_1928_);
v___y_1932_ = v___x_1941_;
v___y_1933_ = v_zero_1928_;
goto v___jp_1931_;
}
else
{
lean_object* v_val_1943_; 
v_val_1943_ = lean_ctor_get(v___x_1942_, 0);
lean_inc(v_val_1943_);
lean_dec_ref_known(v___x_1942_, 1);
v___y_1932_ = v___x_1941_;
v___y_1933_ = v_val_1943_;
goto v___jp_1931_;
}
}
v___jp_1944_:
{
lean_object* v___x_1948_; lean_object* v___x_1949_; 
lean_inc(v_mul_1930_);
v___x_1948_ = lean_apply_2(v_mul_1930_, v___y_1945_, v___y_1947_);
v___x_1949_ = l_List_get_x3fInternal___redArg(v_invfact_1920_, v_head_1926_);
if (lean_obj_tag(v___x_1949_) == 0)
{
lean_inc(v_zero_1928_);
v___y_1938_ = v___y_1946_;
v___y_1939_ = v___x_1948_;
v___y_1940_ = v_zero_1928_;
goto v___jp_1937_;
}
else
{
lean_object* v_val_1950_; 
v_val_1950_ = lean_ctor_get(v___x_1949_, 0);
lean_inc(v_val_1950_);
lean_dec_ref_known(v___x_1949_, 1);
v___y_1938_ = v___y_1946_;
v___y_1939_ = v___x_1948_;
v___y_1940_ = v_val_1950_;
goto v___jp_1937_;
}
}
v___jp_1951_:
{
lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; 
lean_inc(v_mul_1930_);
v___x_1954_ = lean_apply_2(v_mul_1930_, v___y_1952_, v___y_1953_);
v___x_1955_ = lean_nat_sub(v_sDeg_1919_, v_head_1926_);
lean_inc(v___x_1955_);
v___x_1956_ = l_List_get_x3fInternal___redArg(v_sufProduct_1921_, v___x_1955_);
if (lean_obj_tag(v___x_1956_) == 0)
{
lean_inc(v_zero_1928_);
v___y_1945_ = v___x_1954_;
v___y_1946_ = v___x_1955_;
v___y_1947_ = v_zero_1928_;
goto v___jp_1944_;
}
else
{
lean_object* v_val_1957_; 
v_val_1957_ = lean_ctor_get(v___x_1956_, 0);
lean_inc(v_val_1957_);
lean_dec_ref_known(v___x_1956_, 1);
v___y_1945_ = v___x_1954_;
v___y_1946_ = v___x_1955_;
v___y_1947_ = v_val_1957_;
goto v___jp_1944_;
}
}
v___jp_1958_:
{
lean_object* v___x_1960_; 
lean_inc(v_head_1926_);
v___x_1960_ = l_List_get_x3fInternal___redArg(v_prefProduct_1922_, v_head_1926_);
if (lean_obj_tag(v___x_1960_) == 0)
{
lean_inc(v_zero_1928_);
v___y_1952_ = v___y_1959_;
v___y_1953_ = v_zero_1928_;
goto v___jp_1951_;
}
else
{
lean_object* v_val_1961_; 
v_val_1961_ = lean_ctor_get(v___x_1960_, 0);
lean_inc(v_val_1961_);
lean_dec_ref_known(v___x_1960_, 1);
v___y_1952_ = v___y_1959_;
v___y_1953_ = v_val_1961_;
goto v___jp_1951_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg___boxed(lean_object* v___x_1964_, lean_object* v_sDeg_1965_, lean_object* v_invfact_1966_, lean_object* v_sufProduct_1967_, lean_object* v_prefProduct_1968_, lean_object* v_evals_1969_, lean_object* v_x_1970_, lean_object* v_x_1971_){
_start:
{
lean_object* v_res_1972_; 
v_res_1972_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(v___x_1964_, v_sDeg_1965_, v_invfact_1966_, v_sufProduct_1967_, v_prefProduct_1968_, v_evals_1969_, v_x_1970_, v_x_1971_);
lean_dec(v_evals_1969_);
lean_dec(v_prefProduct_1968_);
lean_dec(v_sufProduct_1967_);
lean_dec(v_invfact_1966_);
lean_dec(v_sDeg_1965_);
return v_res_1972_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(lean_object* v___x_1973_, lean_object* v___x_1974_, lean_object* v_sDeg_1975_, lean_object* v_r_1976_, lean_object* v_a_1977_, lean_object* v_a_1978_){
_start:
{
if (lean_obj_tag(v_a_1977_) == 0)
{
lean_object* v___x_1979_; 
lean_dec(v_r_1976_);
lean_dec_ref(v___x_1974_);
lean_dec_ref(v___x_1973_);
v___x_1979_ = l_List_reverse___redArg(v_a_1978_);
return v___x_1979_;
}
else
{
lean_object* v_head_1980_; lean_object* v_tail_1981_; lean_object* v___x_1983_; uint8_t v_isShared_1984_; uint8_t v_isSharedCheck_1994_; 
v_head_1980_ = lean_ctor_get(v_a_1977_, 0);
v_tail_1981_ = lean_ctor_get(v_a_1977_, 1);
v_isSharedCheck_1994_ = !lean_is_exclusive(v_a_1977_);
if (v_isSharedCheck_1994_ == 0)
{
v___x_1983_ = v_a_1977_;
v_isShared_1984_ = v_isSharedCheck_1994_;
goto v_resetjp_1982_;
}
else
{
lean_inc(v_tail_1981_);
lean_inc(v_head_1980_);
lean_dec(v_a_1977_);
v___x_1983_ = lean_box(0);
v_isShared_1984_ = v_isSharedCheck_1994_;
goto v_resetjp_1982_;
}
v_resetjp_1982_:
{
lean_object* v_sub_1985_; lean_object* v_natCast_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1991_; 
v_sub_1985_ = lean_ctor_get(v___x_1973_, 1);
v_natCast_1986_ = lean_ctor_get(v___x_1974_, 2);
v___x_1987_ = lean_nat_sub(v_sDeg_1975_, v_head_1980_);
lean_dec(v_head_1980_);
lean_inc(v_natCast_1986_);
v___x_1988_ = lean_apply_1(v_natCast_1986_, v___x_1987_);
lean_inc(v_sub_1985_);
lean_inc(v_r_1976_);
v___x_1989_ = lean_apply_2(v_sub_1985_, v___x_1988_, v_r_1976_);
if (v_isShared_1984_ == 0)
{
lean_ctor_set(v___x_1983_, 1, v_a_1978_);
lean_ctor_set(v___x_1983_, 0, v___x_1989_);
v___x_1991_ = v___x_1983_;
goto v_reusejp_1990_;
}
else
{
lean_object* v_reuseFailAlloc_1993_; 
v_reuseFailAlloc_1993_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1993_, 0, v___x_1989_);
lean_ctor_set(v_reuseFailAlloc_1993_, 1, v_a_1978_);
v___x_1991_ = v_reuseFailAlloc_1993_;
goto v_reusejp_1990_;
}
v_reusejp_1990_:
{
v_a_1977_ = v_tail_1981_;
v_a_1978_ = v___x_1991_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg___boxed(lean_object* v___x_1995_, lean_object* v___x_1996_, lean_object* v_sDeg_1997_, lean_object* v_r_1998_, lean_object* v_a_1999_, lean_object* v_a_2000_){
_start:
{
lean_object* v_res_2001_; 
v_res_2001_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(v___x_1995_, v___x_1996_, v_sDeg_1997_, v_r_1998_, v_a_1999_, v_a_2000_);
lean_dec(v_sDeg_1997_);
return v_res_2001_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(lean_object* v_fo_2002_, lean_object* v_evals_2003_, lean_object* v_r_2004_){
_start:
{
lean_object* v_toRingOps_2005_; lean_object* v_toSemiringOps_2006_; lean_object* v_zero_2007_; lean_object* v_one_2008_; lean_object* v_mul_2009_; lean_object* v___f_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; lean_object* v_sDeg_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v_factorials_2017_; lean_object* v_invfact_2018_; lean_object* v___x_2019_; lean_object* v_prefProduct_2020_; lean_object* v___x_2021_; lean_object* v_sufProduct_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; 
v_toRingOps_2005_ = lean_ctor_get(v_fo_2002_, 0);
lean_inc_ref_n(v_toRingOps_2005_, 2);
v_toSemiringOps_2006_ = lean_ctor_get(v_toRingOps_2005_, 0);
lean_inc_ref_n(v_toSemiringOps_2006_, 4);
v_zero_2007_ = lean_ctor_get(v_toSemiringOps_2006_, 0);
lean_inc(v_zero_2007_);
v_one_2008_ = lean_ctor_get(v_toSemiringOps_2006_, 1);
v_mul_2009_ = lean_ctor_get(v_toSemiringOps_2006_, 4);
lean_inc(v_mul_2009_);
v___f_2010_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2010_, 0, v_mul_2009_);
v___x_2011_ = l_List_lengthTR___redArg(v_evals_2003_);
v___x_2012_ = lean_unsigned_to_nat(1u);
v_sDeg_2013_ = lean_nat_sub(v___x_2011_, v___x_2012_);
lean_dec(v___x_2011_);
lean_inc(v_sDeg_2013_);
v___x_2014_ = l_List_range(v_sDeg_2013_);
v___x_2015_ = lean_box(0);
lean_inc_n(v___x_2014_, 2);
v___x_2016_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(v_toSemiringOps_2006_, v___x_2014_, v___x_2015_);
lean_inc_n(v_one_2008_, 3);
lean_inc_ref_n(v___f_2010_, 2);
v_factorials_2017_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2010_, v_one_2008_, v___x_2016_);
v_invfact_2018_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_batchInverse___redArg(v_fo_2002_, v_factorials_2017_);
lean_inc(v_r_2004_);
v___x_2019_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(v_toRingOps_2005_, v_toSemiringOps_2006_, v_r_2004_, v___x_2014_, v___x_2015_);
v_prefProduct_2020_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2010_, v_one_2008_, v___x_2019_);
v___x_2021_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(v_toRingOps_2005_, v_toSemiringOps_2006_, v_sDeg_2013_, v_r_2004_, v___x_2014_, v___x_2015_);
v_sufProduct_2022_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2010_, v_one_2008_, v___x_2021_);
v___x_2023_ = lean_nat_add(v_sDeg_2013_, v___x_2012_);
v___x_2024_ = l_List_range(v___x_2023_);
v___x_2025_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(v_toSemiringOps_2006_, v_sDeg_2013_, v_invfact_2018_, v_sufProduct_2022_, v_prefProduct_2020_, v_evals_2003_, v_zero_2007_, v___x_2024_);
lean_dec(v_prefProduct_2020_);
lean_dec(v_sufProduct_2022_);
lean_dec(v_invfact_2018_);
lean_dec(v_sDeg_2013_);
return v___x_2025_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___boxed(lean_object* v_fo_2026_, lean_object* v_evals_2027_, lean_object* v_r_2028_){
_start:
{
lean_object* v_res_2029_; 
v_res_2029_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(v_fo_2026_, v_evals_2027_, v_r_2028_);
lean_dec(v_evals_2027_);
return v_res_2029_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints(lean_object* v_EF_2030_, lean_object* v_fo_2031_, lean_object* v_evals_2032_, lean_object* v_r_2033_){
_start:
{
lean_object* v___x_2034_; 
v___x_2034_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(v_fo_2031_, v_evals_2032_, v_r_2033_);
return v___x_2034_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___boxed(lean_object* v_EF_2035_, lean_object* v_fo_2036_, lean_object* v_evals_2037_, lean_object* v_r_2038_){
_start:
{
lean_object* v_res_2039_; 
v_res_2039_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints(v_EF_2035_, v_fo_2036_, v_evals_2037_, v_r_2038_);
lean_dec(v_evals_2037_);
return v_res_2039_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0(lean_object* v_EF_2040_, lean_object* v___x_2041_, lean_object* v_a_2042_, lean_object* v_a_2043_){
_start:
{
lean_object* v___x_2044_; 
v___x_2044_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__0___redArg(v___x_2041_, v_a_2042_, v_a_2043_);
return v___x_2044_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1(lean_object* v_EF_2045_, lean_object* v___x_2046_, lean_object* v___x_2047_, lean_object* v_r_2048_, lean_object* v_a_2049_, lean_object* v_a_2050_){
_start:
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__1___redArg(v___x_2046_, v___x_2047_, v_r_2048_, v_a_2049_, v_a_2050_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2(lean_object* v_EF_2052_, lean_object* v___x_2053_, lean_object* v___x_2054_, lean_object* v_sDeg_2055_, lean_object* v_r_2056_, lean_object* v_a_2057_, lean_object* v_a_2058_){
_start:
{
lean_object* v___x_2059_; 
v___x_2059_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___redArg(v___x_2053_, v___x_2054_, v_sDeg_2055_, v_r_2056_, v_a_2057_, v_a_2058_);
return v___x_2059_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2___boxed(lean_object* v_EF_2060_, lean_object* v___x_2061_, lean_object* v___x_2062_, lean_object* v_sDeg_2063_, lean_object* v_r_2064_, lean_object* v_a_2065_, lean_object* v_a_2066_){
_start:
{
lean_object* v_res_2067_; 
v_res_2067_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__2(v_EF_2060_, v___x_2061_, v___x_2062_, v_sDeg_2063_, v_r_2064_, v_a_2065_, v_a_2066_);
lean_dec(v_sDeg_2063_);
return v_res_2067_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3(lean_object* v_EF_2068_, lean_object* v___x_2069_, lean_object* v_sDeg_2070_, lean_object* v_invfact_2071_, lean_object* v_sufProduct_2072_, lean_object* v_prefProduct_2073_, lean_object* v_evals_2074_, lean_object* v_x_2075_, lean_object* v_x_2076_){
_start:
{
lean_object* v___x_2077_; 
v___x_2077_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___redArg(v___x_2069_, v_sDeg_2070_, v_invfact_2071_, v_sufProduct_2072_, v_prefProduct_2073_, v_evals_2074_, v_x_2075_, v_x_2076_);
return v___x_2077_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3___boxed(lean_object* v_EF_2078_, lean_object* v___x_2079_, lean_object* v_sDeg_2080_, lean_object* v_invfact_2081_, lean_object* v_sufProduct_2082_, lean_object* v_prefProduct_2083_, lean_object* v_evals_2084_, lean_object* v_x_2085_, lean_object* v_x_2086_){
_start:
{
lean_object* v_res_2087_; 
v_res_2087_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints_spec__3(v_EF_2078_, v___x_2079_, v_sDeg_2080_, v_invfact_2081_, v_sufProduct_2082_, v_prefProduct_2083_, v_evals_2084_, v_x_2085_, v_x_2086_);
lean_dec(v_evals_2084_);
lean_dec(v_prefProduct_2083_);
lean_dec(v_sufProduct_2082_);
lean_dec(v_invfact_2081_);
lean_dec(v_sDeg_2080_);
return v_res_2087_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(lean_object* v_fo_2088_, lean_object* v_curSum_2089_, lean_object* v_batchSEvals_2090_, lean_object* v_r_2091_){
_start:
{
lean_object* v_toRingOps_2092_; lean_object* v___y_2094_; 
v_toRingOps_2092_ = lean_ctor_get(v_fo_2088_, 0);
lean_inc_ref(v_toRingOps_2092_);
if (lean_obj_tag(v_batchSEvals_2090_) == 0)
{
lean_object* v_toSemiringOps_2106_; lean_object* v_zero_2107_; 
v_toSemiringOps_2106_ = lean_ctor_get(v_toRingOps_2092_, 0);
v_zero_2107_ = lean_ctor_get(v_toSemiringOps_2106_, 0);
lean_inc(v_zero_2107_);
v___y_2094_ = v_zero_2107_;
goto v___jp_2093_;
}
else
{
lean_object* v_head_2108_; 
v_head_2108_ = lean_ctor_get(v_batchSEvals_2090_, 0);
lean_inc(v_head_2108_);
v___y_2094_ = v_head_2108_;
goto v___jp_2093_;
}
v___jp_2093_:
{
lean_object* v_sub_2095_; lean_object* v___x_2097_; uint8_t v_isShared_2098_; uint8_t v_isSharedCheck_2104_; 
v_sub_2095_ = lean_ctor_get(v_toRingOps_2092_, 1);
v_isSharedCheck_2104_ = !lean_is_exclusive(v_toRingOps_2092_);
if (v_isSharedCheck_2104_ == 0)
{
lean_object* v_unused_2105_; 
v_unused_2105_ = lean_ctor_get(v_toRingOps_2092_, 0);
lean_dec(v_unused_2105_);
v___x_2097_ = v_toRingOps_2092_;
v_isShared_2098_ = v_isSharedCheck_2104_;
goto v_resetjp_2096_;
}
else
{
lean_inc(v_sub_2095_);
lean_dec(v_toRingOps_2092_);
v___x_2097_ = lean_box(0);
v_isShared_2098_ = v_isSharedCheck_2104_;
goto v_resetjp_2096_;
}
v_resetjp_2096_:
{
lean_object* v_s0_2099_; lean_object* v___x_2101_; 
v_s0_2099_ = lean_apply_2(v_sub_2095_, v_curSum_2089_, v___y_2094_);
if (v_isShared_2098_ == 0)
{
lean_ctor_set_tag(v___x_2097_, 1);
lean_ctor_set(v___x_2097_, 1, v_batchSEvals_2090_);
lean_ctor_set(v___x_2097_, 0, v_s0_2099_);
v___x_2101_ = v___x_2097_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v_s0_2099_);
lean_ctor_set(v_reuseFailAlloc_2103_, 1, v_batchSEvals_2090_);
v___x_2101_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
lean_object* v___x_2102_; 
v___x_2102_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg(v_fo_2088_, v___x_2101_, v_r_2091_);
lean_dec_ref(v___x_2101_);
return v___x_2102_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound(lean_object* v_EF_2109_, lean_object* v_fo_2110_, lean_object* v_curSum_2111_, lean_object* v_batchSEvals_2112_, lean_object* v_r_2113_){
_start:
{
lean_object* v___x_2114_; 
v___x_2114_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(v_fo_2110_, v_curSum_2111_, v_batchSEvals_2112_, v_r_2113_);
return v___x_2114_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(lean_object* v_bInt_2115_, lean_object* v_fo_2116_, lean_object* v_a_2117_, lean_object* v_a_2118_){
_start:
{
if (lean_obj_tag(v_a_2117_) == 0)
{
lean_object* v___x_2119_; 
lean_dec_ref(v_fo_2116_);
v___x_2119_ = l_List_reverse___redArg(v_a_2118_);
return v___x_2119_;
}
else
{
lean_object* v_head_2120_; lean_object* v_tail_2121_; lean_object* v___x_2123_; uint8_t v_isShared_2124_; uint8_t v_isSharedCheck_2141_; 
v_head_2120_ = lean_ctor_get(v_a_2117_, 0);
v_tail_2121_ = lean_ctor_get(v_a_2117_, 1);
v_isSharedCheck_2141_ = !lean_is_exclusive(v_a_2117_);
if (v_isSharedCheck_2141_ == 0)
{
v___x_2123_ = v_a_2117_;
v_isShared_2124_ = v_isSharedCheck_2141_;
goto v_resetjp_2122_;
}
else
{
lean_inc(v_tail_2121_);
lean_inc(v_head_2120_);
lean_dec(v_a_2117_);
v___x_2123_ = lean_box(0);
v_isShared_2124_ = v_isSharedCheck_2141_;
goto v_resetjp_2122_;
}
v_resetjp_2122_:
{
lean_object* v___y_2126_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; uint8_t v___x_2134_; 
v___x_2131_ = lean_nat_shiftr(v_bInt_2115_, v_head_2120_);
lean_dec(v_head_2120_);
v___x_2132_ = lean_unsigned_to_nat(1u);
v___x_2133_ = lean_nat_land(v___x_2131_, v___x_2132_);
lean_dec(v___x_2131_);
v___x_2134_ = lean_nat_dec_eq(v___x_2133_, v___x_2132_);
lean_dec(v___x_2133_);
if (v___x_2134_ == 0)
{
lean_object* v_toRingOps_2135_; lean_object* v_toSemiringOps_2136_; lean_object* v_zero_2137_; 
v_toRingOps_2135_ = lean_ctor_get(v_fo_2116_, 0);
v_toSemiringOps_2136_ = lean_ctor_get(v_toRingOps_2135_, 0);
v_zero_2137_ = lean_ctor_get(v_toSemiringOps_2136_, 0);
lean_inc(v_zero_2137_);
v___y_2126_ = v_zero_2137_;
goto v___jp_2125_;
}
else
{
lean_object* v_toRingOps_2138_; lean_object* v_toSemiringOps_2139_; lean_object* v_one_2140_; 
v_toRingOps_2138_ = lean_ctor_get(v_fo_2116_, 0);
v_toSemiringOps_2139_ = lean_ctor_get(v_toRingOps_2138_, 0);
v_one_2140_ = lean_ctor_get(v_toSemiringOps_2139_, 1);
lean_inc(v_one_2140_);
v___y_2126_ = v_one_2140_;
goto v___jp_2125_;
}
v___jp_2125_:
{
lean_object* v___x_2128_; 
if (v_isShared_2124_ == 0)
{
lean_ctor_set(v___x_2123_, 1, v_a_2118_);
lean_ctor_set(v___x_2123_, 0, v___y_2126_);
v___x_2128_ = v___x_2123_;
goto v_reusejp_2127_;
}
else
{
lean_object* v_reuseFailAlloc_2130_; 
v_reuseFailAlloc_2130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2130_, 0, v___y_2126_);
lean_ctor_set(v_reuseFailAlloc_2130_, 1, v_a_2118_);
v___x_2128_ = v_reuseFailAlloc_2130_;
goto v_reusejp_2127_;
}
v_reusejp_2127_:
{
v_a_2117_ = v_tail_2121_;
v_a_2118_ = v___x_2128_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg___boxed(lean_object* v_bInt_2142_, lean_object* v_fo_2143_, lean_object* v_a_2144_, lean_object* v_a_2145_){
_start:
{
lean_object* v_res_2146_; 
v_res_2146_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2142_, v_fo_2143_, v_a_2144_, v_a_2145_);
lean_dec(v_bInt_2142_);
return v_res_2146_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(lean_object* v___x_2147_, lean_object* v_bitWidth_2148_, lean_object* v_fo_2149_, lean_object* v_xiSlice_2150_, lean_object* v_blockSize_2151_, lean_object* v_x_2152_, lean_object* v_x_2153_){
_start:
{
if (lean_obj_tag(v_x_2153_) == 0)
{
lean_dec(v_xiSlice_2150_);
lean_dec_ref(v_fo_2149_);
lean_dec(v_bitWidth_2148_);
return v_x_2152_;
}
else
{
lean_object* v_tail_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2178_; 
v_tail_2154_ = lean_ctor_get(v_x_2153_, 1);
v_isSharedCheck_2178_ = !lean_is_exclusive(v_x_2153_);
if (v_isSharedCheck_2178_ == 0)
{
lean_object* v_unused_2179_; 
v_unused_2179_ = lean_ctor_get(v_x_2153_, 0);
lean_dec(v_unused_2179_);
v___x_2156_ = v_x_2153_;
v_isShared_2157_ = v_isSharedCheck_2178_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_tail_2154_);
lean_dec(v_x_2153_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2178_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
lean_object* v_fst_2158_; lean_object* v_snd_2159_; lean_object* v___x_2161_; uint8_t v_isShared_2162_; uint8_t v_isSharedCheck_2177_; 
v_fst_2158_ = lean_ctor_get(v_x_2152_, 0);
v_snd_2159_ = lean_ctor_get(v_x_2152_, 1);
v_isSharedCheck_2177_ = !lean_is_exclusive(v_x_2152_);
if (v_isSharedCheck_2177_ == 0)
{
v___x_2161_ = v_x_2152_;
v_isShared_2162_ = v_isSharedCheck_2177_;
goto v_resetjp_2160_;
}
else
{
lean_inc(v_snd_2159_);
lean_inc(v_fst_2158_);
lean_dec(v_x_2152_);
v___x_2161_ = lean_box(0);
v_isShared_2162_ = v_isSharedCheck_2177_;
goto v_resetjp_2160_;
}
v_resetjp_2160_:
{
lean_object* v_toRingOps_2163_; lean_object* v_bInt_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v_bVec_2167_; lean_object* v_eq3b_2168_; lean_object* v___x_2170_; 
v_toRingOps_2163_ = lean_ctor_get(v_fo_2149_, 0);
v_bInt_2164_ = lean_nat_shiftr(v_snd_2159_, v___x_2147_);
lean_inc(v_bitWidth_2148_);
v___x_2165_ = l_List_range(v_bitWidth_2148_);
v___x_2166_ = lean_box(0);
lean_inc_ref(v_fo_2149_);
v_bVec_2167_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2164_, v_fo_2149_, v___x_2165_, v___x_2166_);
lean_dec(v_bInt_2164_);
lean_inc(v_xiSlice_2150_);
lean_inc_ref(v_toRingOps_2163_);
v_eq3b_2168_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_2163_, v_xiSlice_2150_, v_bVec_2167_);
if (v_isShared_2157_ == 0)
{
lean_ctor_set(v___x_2156_, 1, v_fst_2158_);
lean_ctor_set(v___x_2156_, 0, v_eq3b_2168_);
v___x_2170_ = v___x_2156_;
goto v_reusejp_2169_;
}
else
{
lean_object* v_reuseFailAlloc_2176_; 
v_reuseFailAlloc_2176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2176_, 0, v_eq3b_2168_);
lean_ctor_set(v_reuseFailAlloc_2176_, 1, v_fst_2158_);
v___x_2170_ = v_reuseFailAlloc_2176_;
goto v_reusejp_2169_;
}
v_reusejp_2169_:
{
lean_object* v___x_2171_; lean_object* v___x_2173_; 
v___x_2171_ = lean_nat_add(v_snd_2159_, v_blockSize_2151_);
lean_dec(v_snd_2159_);
if (v_isShared_2162_ == 0)
{
lean_ctor_set(v___x_2161_, 1, v___x_2171_);
lean_ctor_set(v___x_2161_, 0, v___x_2170_);
v___x_2173_ = v___x_2161_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2175_; 
v_reuseFailAlloc_2175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2175_, 0, v___x_2170_);
lean_ctor_set(v_reuseFailAlloc_2175_, 1, v___x_2171_);
v___x_2173_ = v_reuseFailAlloc_2175_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
v_x_2152_ = v___x_2173_;
v_x_2153_ = v_tail_2154_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg___boxed(lean_object* v___x_2180_, lean_object* v_bitWidth_2181_, lean_object* v_fo_2182_, lean_object* v_xiSlice_2183_, lean_object* v_blockSize_2184_, lean_object* v_x_2185_, lean_object* v_x_2186_){
_start:
{
lean_object* v_res_2187_; 
v_res_2187_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(v___x_2180_, v_bitWidth_2181_, v_fo_2182_, v_xiSlice_2183_, v_blockSize_2184_, v_x_2185_, v_x_2186_);
lean_dec(v_blockSize_2184_);
lean_dec(v___x_2180_);
return v_res_2187_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(lean_object* v___x_2188_, lean_object* v_fo_2189_, lean_object* v_bitWidth_2190_, lean_object* v_xiSlice_2191_, lean_object* v_blockSize_2192_, lean_object* v_x_2193_, lean_object* v_x_2194_){
_start:
{
if (lean_obj_tag(v_x_2194_) == 0)
{
lean_dec(v_xiSlice_2191_);
lean_dec(v_bitWidth_2190_);
lean_dec_ref(v_fo_2189_);
return v_x_2193_;
}
else
{
lean_object* v_tail_2195_; lean_object* v___x_2197_; uint8_t v_isShared_2198_; uint8_t v_isSharedCheck_2219_; 
v_tail_2195_ = lean_ctor_get(v_x_2194_, 1);
v_isSharedCheck_2219_ = !lean_is_exclusive(v_x_2194_);
if (v_isSharedCheck_2219_ == 0)
{
lean_object* v_unused_2220_; 
v_unused_2220_ = lean_ctor_get(v_x_2194_, 0);
lean_dec(v_unused_2220_);
v___x_2197_ = v_x_2194_;
v_isShared_2198_ = v_isSharedCheck_2219_;
goto v_resetjp_2196_;
}
else
{
lean_inc(v_tail_2195_);
lean_dec(v_x_2194_);
v___x_2197_ = lean_box(0);
v_isShared_2198_ = v_isSharedCheck_2219_;
goto v_resetjp_2196_;
}
v_resetjp_2196_:
{
lean_object* v_fst_2199_; lean_object* v_snd_2200_; lean_object* v___x_2202_; uint8_t v_isShared_2203_; uint8_t v_isSharedCheck_2218_; 
v_fst_2199_ = lean_ctor_get(v_x_2193_, 0);
v_snd_2200_ = lean_ctor_get(v_x_2193_, 1);
v_isSharedCheck_2218_ = !lean_is_exclusive(v_x_2193_);
if (v_isSharedCheck_2218_ == 0)
{
v___x_2202_ = v_x_2193_;
v_isShared_2203_ = v_isSharedCheck_2218_;
goto v_resetjp_2201_;
}
else
{
lean_inc(v_snd_2200_);
lean_inc(v_fst_2199_);
lean_dec(v_x_2193_);
v___x_2202_ = lean_box(0);
v_isShared_2203_ = v_isSharedCheck_2218_;
goto v_resetjp_2201_;
}
v_resetjp_2201_:
{
lean_object* v_toRingOps_2204_; lean_object* v_bInt_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v_bVec_2208_; lean_object* v_eq3b_2209_; lean_object* v___x_2211_; 
v_toRingOps_2204_ = lean_ctor_get(v_fo_2189_, 0);
v_bInt_2205_ = lean_nat_shiftr(v_snd_2200_, v___x_2188_);
lean_inc(v_bitWidth_2190_);
v___x_2206_ = l_List_range(v_bitWidth_2190_);
v___x_2207_ = lean_box(0);
lean_inc_ref(v_fo_2189_);
v_bVec_2208_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2205_, v_fo_2189_, v___x_2206_, v___x_2207_);
lean_dec(v_bInt_2205_);
lean_inc(v_xiSlice_2191_);
lean_inc_ref(v_toRingOps_2204_);
v_eq3b_2209_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_2204_, v_xiSlice_2191_, v_bVec_2208_);
if (v_isShared_2198_ == 0)
{
lean_ctor_set(v___x_2197_, 1, v_fst_2199_);
lean_ctor_set(v___x_2197_, 0, v_eq3b_2209_);
v___x_2211_ = v___x_2197_;
goto v_reusejp_2210_;
}
else
{
lean_object* v_reuseFailAlloc_2217_; 
v_reuseFailAlloc_2217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2217_, 0, v_eq3b_2209_);
lean_ctor_set(v_reuseFailAlloc_2217_, 1, v_fst_2199_);
v___x_2211_ = v_reuseFailAlloc_2217_;
goto v_reusejp_2210_;
}
v_reusejp_2210_:
{
lean_object* v___x_2212_; lean_object* v___x_2214_; 
v___x_2212_ = lean_nat_add(v_snd_2200_, v_blockSize_2192_);
lean_dec(v_snd_2200_);
if (v_isShared_2203_ == 0)
{
lean_ctor_set(v___x_2202_, 1, v___x_2212_);
lean_ctor_set(v___x_2202_, 0, v___x_2211_);
v___x_2214_ = v___x_2202_;
goto v_reusejp_2213_;
}
else
{
lean_object* v_reuseFailAlloc_2216_; 
v_reuseFailAlloc_2216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2216_, 0, v___x_2211_);
lean_ctor_set(v_reuseFailAlloc_2216_, 1, v___x_2212_);
v___x_2214_ = v_reuseFailAlloc_2216_;
goto v_reusejp_2213_;
}
v_reusejp_2213_:
{
lean_object* v___x_2215_; 
v___x_2215_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(v___x_2188_, v_bitWidth_2190_, v_fo_2189_, v_xiSlice_2191_, v_blockSize_2192_, v___x_2214_, v_tail_2195_);
return v___x_2215_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg___boxed(lean_object* v___x_2221_, lean_object* v_fo_2222_, lean_object* v_bitWidth_2223_, lean_object* v_xiSlice_2224_, lean_object* v_blockSize_2225_, lean_object* v_x_2226_, lean_object* v_x_2227_){
_start:
{
lean_object* v_res_2228_; 
v_res_2228_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(v___x_2221_, v_fo_2222_, v_bitWidth_2223_, v_xiSlice_2224_, v_blockSize_2225_, v_x_2226_, v_x_2227_);
lean_dec(v_blockSize_2225_);
lean_dec(v___x_2221_);
return v_res_2228_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(lean_object* v_nLogup_2231_, lean_object* v_lSkip_2232_, lean_object* v_xi_2233_, lean_object* v_fo_2234_, lean_object* v_x_2235_, lean_object* v_x_2236_){
_start:
{
if (lean_obj_tag(v_x_2236_) == 0)
{
lean_dec_ref(v_fo_2234_);
return v_x_2235_;
}
else
{
lean_object* v_head_2237_; lean_object* v_tail_2238_; lean_object* v___x_2240_; uint8_t v_isShared_2241_; uint8_t v_isSharedCheck_2288_; 
v_head_2237_ = lean_ctor_get(v_x_2236_, 0);
v_tail_2238_ = lean_ctor_get(v_x_2236_, 1);
v_isSharedCheck_2288_ = !lean_is_exclusive(v_x_2236_);
if (v_isSharedCheck_2288_ == 0)
{
v___x_2240_ = v_x_2236_;
v_isShared_2241_ = v_isSharedCheck_2288_;
goto v_resetjp_2239_;
}
else
{
lean_inc(v_tail_2238_);
lean_inc(v_head_2237_);
lean_dec(v_x_2236_);
v___x_2240_ = lean_box(0);
v_isShared_2241_ = v_isSharedCheck_2288_;
goto v_resetjp_2239_;
}
v_resetjp_2239_:
{
lean_object* v_fst_2242_; lean_object* v_snd_2243_; lean_object* v_fst_2244_; lean_object* v_snd_2245_; lean_object* v___x_2247_; uint8_t v_isShared_2248_; uint8_t v_isSharedCheck_2287_; 
v_fst_2242_ = lean_ctor_get(v_x_2235_, 0);
lean_inc(v_fst_2242_);
v_snd_2243_ = lean_ctor_get(v_x_2235_, 1);
lean_inc(v_snd_2243_);
lean_dec_ref(v_x_2235_);
v_fst_2244_ = lean_ctor_get(v_head_2237_, 0);
v_snd_2245_ = lean_ctor_get(v_head_2237_, 1);
v_isSharedCheck_2287_ = !lean_is_exclusive(v_head_2237_);
if (v_isSharedCheck_2287_ == 0)
{
v___x_2247_ = v_head_2237_;
v_isShared_2248_ = v_isSharedCheck_2287_;
goto v_resetjp_2246_;
}
else
{
lean_inc(v_snd_2245_);
lean_inc(v_fst_2244_);
lean_dec(v_head_2237_);
v___x_2247_ = lean_box(0);
v_isShared_2248_ = v_isSharedCheck_2287_;
goto v_resetjp_2246_;
}
v_resetjp_2246_:
{
lean_object* v___x_2249_; uint8_t v___x_2250_; 
v___x_2249_ = lean_unsigned_to_nat(0u);
v___x_2250_ = lean_nat_dec_eq(v_fst_2244_, v___x_2249_);
if (v___x_2250_ == 0)
{
lean_object* v_nLift_2251_; lean_object* v_bitWidth_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v_xiSlice_2256_; lean_object* v___x_2257_; lean_object* v_blockSize_2258_; lean_object* v___x_2259_; lean_object* v___x_2261_; 
v_nLift_2251_ = l_Int_toNat(v_snd_2245_);
lean_dec(v_snd_2245_);
v_bitWidth_2252_ = lean_nat_sub(v_nLogup_2231_, v_nLift_2251_);
v___x_2253_ = lean_nat_add(v_lSkip_2232_, v_nLift_2251_);
lean_dec(v_nLift_2251_);
lean_inc(v___x_2253_);
v___x_2254_ = l_List_drop___redArg(v___x_2253_, v_xi_2233_);
v___x_2255_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0));
lean_inc(v_bitWidth_2252_);
lean_inc(v___x_2254_);
v_xiSlice_2256_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v___x_2254_, v___x_2254_, v_bitWidth_2252_, v___x_2255_);
lean_dec(v___x_2254_);
v___x_2257_ = lean_unsigned_to_nat(2u);
v_blockSize_2258_ = lean_nat_pow(v___x_2257_, v___x_2253_);
v___x_2259_ = lean_box(0);
if (v_isShared_2248_ == 0)
{
lean_ctor_set(v___x_2247_, 1, v_snd_2243_);
lean_ctor_set(v___x_2247_, 0, v___x_2259_);
v___x_2261_ = v___x_2247_;
goto v_reusejp_2260_;
}
else
{
lean_object* v_reuseFailAlloc_2278_; 
v_reuseFailAlloc_2278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2278_, 0, v___x_2259_);
lean_ctor_set(v_reuseFailAlloc_2278_, 1, v_snd_2243_);
v___x_2261_ = v_reuseFailAlloc_2278_;
goto v_reusejp_2260_;
}
v_reusejp_2260_:
{
lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v_fst_2264_; lean_object* v_snd_2265_; lean_object* v___x_2267_; uint8_t v_isShared_2268_; uint8_t v_isSharedCheck_2277_; 
v___x_2262_ = l_List_range(v_fst_2244_);
lean_inc_ref(v_fo_2234_);
v___x_2263_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(v___x_2253_, v_fo_2234_, v_bitWidth_2252_, v_xiSlice_2256_, v_blockSize_2258_, v___x_2261_, v___x_2262_);
lean_dec(v_blockSize_2258_);
lean_dec(v___x_2253_);
v_fst_2264_ = lean_ctor_get(v___x_2263_, 0);
v_snd_2265_ = lean_ctor_get(v___x_2263_, 1);
v_isSharedCheck_2277_ = !lean_is_exclusive(v___x_2263_);
if (v_isSharedCheck_2277_ == 0)
{
v___x_2267_ = v___x_2263_;
v_isShared_2268_ = v_isSharedCheck_2277_;
goto v_resetjp_2266_;
}
else
{
lean_inc(v_snd_2265_);
lean_inc(v_fst_2264_);
lean_dec(v___x_2263_);
v___x_2267_ = lean_box(0);
v_isShared_2268_ = v_isSharedCheck_2277_;
goto v_resetjp_2266_;
}
v_resetjp_2266_:
{
lean_object* v___x_2269_; lean_object* v___x_2271_; 
v___x_2269_ = l_List_reverse___redArg(v_fst_2264_);
if (v_isShared_2241_ == 0)
{
lean_ctor_set(v___x_2240_, 1, v_fst_2242_);
lean_ctor_set(v___x_2240_, 0, v___x_2269_);
v___x_2271_ = v___x_2240_;
goto v_reusejp_2270_;
}
else
{
lean_object* v_reuseFailAlloc_2276_; 
v_reuseFailAlloc_2276_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2276_, 0, v___x_2269_);
lean_ctor_set(v_reuseFailAlloc_2276_, 1, v_fst_2242_);
v___x_2271_ = v_reuseFailAlloc_2276_;
goto v_reusejp_2270_;
}
v_reusejp_2270_:
{
lean_object* v___x_2273_; 
if (v_isShared_2268_ == 0)
{
lean_ctor_set(v___x_2267_, 0, v___x_2271_);
v___x_2273_ = v___x_2267_;
goto v_reusejp_2272_;
}
else
{
lean_object* v_reuseFailAlloc_2275_; 
v_reuseFailAlloc_2275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2275_, 0, v___x_2271_);
lean_ctor_set(v_reuseFailAlloc_2275_, 1, v_snd_2265_);
v___x_2273_ = v_reuseFailAlloc_2275_;
goto v_reusejp_2272_;
}
v_reusejp_2272_:
{
v_x_2235_ = v___x_2273_;
v_x_2236_ = v_tail_2238_;
goto _start;
}
}
}
}
}
else
{
lean_object* v___x_2279_; lean_object* v___x_2281_; 
lean_dec(v_snd_2245_);
lean_dec(v_fst_2244_);
v___x_2279_ = lean_box(0);
if (v_isShared_2241_ == 0)
{
lean_ctor_set(v___x_2240_, 1, v_fst_2242_);
lean_ctor_set(v___x_2240_, 0, v___x_2279_);
v___x_2281_ = v___x_2240_;
goto v_reusejp_2280_;
}
else
{
lean_object* v_reuseFailAlloc_2286_; 
v_reuseFailAlloc_2286_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2286_, 0, v___x_2279_);
lean_ctor_set(v_reuseFailAlloc_2286_, 1, v_fst_2242_);
v___x_2281_ = v_reuseFailAlloc_2286_;
goto v_reusejp_2280_;
}
v_reusejp_2280_:
{
lean_object* v___x_2283_; 
if (v_isShared_2248_ == 0)
{
lean_ctor_set(v___x_2247_, 1, v_snd_2243_);
lean_ctor_set(v___x_2247_, 0, v___x_2281_);
v___x_2283_ = v___x_2247_;
goto v_reusejp_2282_;
}
else
{
lean_object* v_reuseFailAlloc_2285_; 
v_reuseFailAlloc_2285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2285_, 0, v___x_2281_);
lean_ctor_set(v_reuseFailAlloc_2285_, 1, v_snd_2243_);
v___x_2283_ = v_reuseFailAlloc_2285_;
goto v_reusejp_2282_;
}
v_reusejp_2282_:
{
v_x_2235_ = v___x_2283_;
v_x_2236_ = v_tail_2238_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___boxed(lean_object* v_nLogup_2289_, lean_object* v_lSkip_2290_, lean_object* v_xi_2291_, lean_object* v_fo_2292_, lean_object* v_x_2293_, lean_object* v_x_2294_){
_start:
{
lean_object* v_res_2295_; 
v_res_2295_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(v_nLogup_2289_, v_lSkip_2290_, v_xi_2291_, v_fo_2292_, v_x_2293_, v_x_2294_);
lean_dec(v_xi_2291_);
lean_dec(v_lSkip_2290_);
lean_dec(v_nLogup_2289_);
return v_res_2295_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(lean_object* v_fo_2299_, lean_object* v_lSkip_2300_, lean_object* v_nLogup_2301_, lean_object* v_xi_2302_, lean_object* v_interactionCountPerTrace_2303_, lean_object* v_nPerTrace_2304_){
_start:
{
lean_object* v_entries_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v_fst_2308_; lean_object* v___x_2309_; 
v_entries_2305_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_interactionCountPerTrace_2303_, v_nPerTrace_2304_);
v___x_2306_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___closed__0));
v___x_2307_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(v_nLogup_2301_, v_lSkip_2300_, v_xi_2302_, v_fo_2299_, v___x_2306_, v_entries_2305_);
v_fst_2308_ = lean_ctor_get(v___x_2307_, 0);
lean_inc(v_fst_2308_);
lean_dec_ref(v___x_2307_);
v___x_2309_ = l_List_reverse___redArg(v_fst_2308_);
return v___x_2309_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg___boxed(lean_object* v_fo_2310_, lean_object* v_lSkip_2311_, lean_object* v_nLogup_2312_, lean_object* v_xi_2313_, lean_object* v_interactionCountPerTrace_2314_, lean_object* v_nPerTrace_2315_){
_start:
{
lean_object* v_res_2316_; 
v_res_2316_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(v_fo_2310_, v_lSkip_2311_, v_nLogup_2312_, v_xi_2313_, v_interactionCountPerTrace_2314_, v_nPerTrace_2315_);
lean_dec(v_xi_2313_);
lean_dec(v_nLogup_2312_);
lean_dec(v_lSkip_2311_);
return v_res_2316_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace(lean_object* v_EF_2317_, lean_object* v_fo_2318_, lean_object* v_lSkip_2319_, lean_object* v_nLogup_2320_, lean_object* v_xi_2321_, lean_object* v_interactionCountPerTrace_2322_, lean_object* v_nPerTrace_2323_){
_start:
{
lean_object* v___x_2324_; 
v___x_2324_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(v_fo_2318_, v_lSkip_2319_, v_nLogup_2320_, v_xi_2321_, v_interactionCountPerTrace_2322_, v_nPerTrace_2323_);
return v___x_2324_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___boxed(lean_object* v_EF_2325_, lean_object* v_fo_2326_, lean_object* v_lSkip_2327_, lean_object* v_nLogup_2328_, lean_object* v_xi_2329_, lean_object* v_interactionCountPerTrace_2330_, lean_object* v_nPerTrace_2331_){
_start:
{
lean_object* v_res_2332_; 
v_res_2332_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace(v_EF_2325_, v_fo_2326_, v_lSkip_2327_, v_nLogup_2328_, v_xi_2329_, v_interactionCountPerTrace_2330_, v_nPerTrace_2331_);
lean_dec(v_xi_2329_);
lean_dec(v_nLogup_2328_);
lean_dec(v_lSkip_2327_);
return v_res_2332_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0(lean_object* v_EF_2333_, lean_object* v_bInt_2334_, lean_object* v_fo_2335_, lean_object* v_a_2336_, lean_object* v_a_2337_){
_start:
{
lean_object* v___x_2338_; 
v___x_2338_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___redArg(v_bInt_2334_, v_fo_2335_, v_a_2336_, v_a_2337_);
return v___x_2338_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0___boxed(lean_object* v_EF_2339_, lean_object* v_bInt_2340_, lean_object* v_fo_2341_, lean_object* v_a_2342_, lean_object* v_a_2343_){
_start:
{
lean_object* v_res_2344_; 
v_res_2344_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__0(v_EF_2339_, v_bInt_2340_, v_fo_2341_, v_a_2342_, v_a_2343_);
lean_dec(v_bInt_2340_);
return v_res_2344_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1(lean_object* v_EF_2345_, lean_object* v___x_2346_, lean_object* v_fo_2347_, lean_object* v_bitWidth_2348_, lean_object* v_xiSlice_2349_, lean_object* v_blockSize_2350_, lean_object* v_x_2351_, lean_object* v_x_2352_){
_start:
{
lean_object* v___x_2353_; 
v___x_2353_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___redArg(v___x_2346_, v_fo_2347_, v_bitWidth_2348_, v_xiSlice_2349_, v_blockSize_2350_, v_x_2351_, v_x_2352_);
return v___x_2353_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1___boxed(lean_object* v_EF_2354_, lean_object* v___x_2355_, lean_object* v_fo_2356_, lean_object* v_bitWidth_2357_, lean_object* v_xiSlice_2358_, lean_object* v_blockSize_2359_, lean_object* v_x_2360_, lean_object* v_x_2361_){
_start:
{
lean_object* v_res_2362_; 
v_res_2362_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1(v_EF_2354_, v___x_2355_, v_fo_2356_, v_bitWidth_2357_, v_xiSlice_2358_, v_blockSize_2359_, v_x_2360_, v_x_2361_);
lean_dec(v_blockSize_2359_);
lean_dec(v___x_2355_);
return v_res_2362_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2(lean_object* v_EF_2363_, lean_object* v_nLogup_2364_, lean_object* v_lSkip_2365_, lean_object* v_xi_2366_, lean_object* v_fo_2367_, lean_object* v_x_2368_, lean_object* v_x_2369_){
_start:
{
lean_object* v___x_2370_; 
v___x_2370_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg(v_nLogup_2364_, v_lSkip_2365_, v_xi_2366_, v_fo_2367_, v_x_2368_, v_x_2369_);
return v___x_2370_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___boxed(lean_object* v_EF_2371_, lean_object* v_nLogup_2372_, lean_object* v_lSkip_2373_, lean_object* v_xi_2374_, lean_object* v_fo_2375_, lean_object* v_x_2376_, lean_object* v_x_2377_){
_start:
{
lean_object* v_res_2378_; 
v_res_2378_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2(v_EF_2371_, v_nLogup_2372_, v_lSkip_2373_, v_xi_2374_, v_fo_2375_, v_x_2376_, v_x_2377_);
lean_dec(v_xi_2374_);
lean_dec(v_lSkip_2373_);
lean_dec(v_nLogup_2372_);
return v_res_2378_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1(lean_object* v_EF_2379_, lean_object* v___x_2380_, lean_object* v_bitWidth_2381_, lean_object* v_fo_2382_, lean_object* v_xiSlice_2383_, lean_object* v_blockSize_2384_, lean_object* v_x_2385_, lean_object* v_x_2386_){
_start:
{
lean_object* v___x_2387_; 
v___x_2387_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___redArg(v___x_2380_, v_bitWidth_2381_, v_fo_2382_, v_xiSlice_2383_, v_blockSize_2384_, v_x_2385_, v_x_2386_);
return v___x_2387_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1___boxed(lean_object* v_EF_2388_, lean_object* v___x_2389_, lean_object* v_bitWidth_2390_, lean_object* v_fo_2391_, lean_object* v_xiSlice_2392_, lean_object* v_blockSize_2393_, lean_object* v_x_2394_, lean_object* v_x_2395_){
_start:
{
lean_object* v_res_2396_; 
v_res_2396_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__1_spec__1(v_EF_2388_, v___x_2389_, v_bitWidth_2390_, v_fo_2391_, v_xiSlice_2392_, v_blockSize_2393_, v_x_2394_, v_x_2395_);
lean_dec(v_blockSize_2393_);
lean_dec(v___x_2389_);
return v_res_2396_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(lean_object* v___x_2397_, lean_object* v_a_2398_, lean_object* v_a_2399_){
_start:
{
if (lean_obj_tag(v_a_2398_) == 0)
{
lean_object* v___x_2400_; 
lean_dec_ref(v___x_2397_);
v___x_2400_ = l_List_reverse___redArg(v_a_2399_);
return v___x_2400_;
}
else
{
lean_object* v_head_2401_; lean_object* v_tail_2402_; lean_object* v___x_2404_; uint8_t v_isShared_2405_; uint8_t v_isSharedCheck_2414_; 
v_head_2401_ = lean_ctor_get(v_a_2398_, 0);
v_tail_2402_ = lean_ctor_get(v_a_2398_, 1);
v_isSharedCheck_2414_ = !lean_is_exclusive(v_a_2398_);
if (v_isSharedCheck_2414_ == 0)
{
v___x_2404_ = v_a_2398_;
v_isShared_2405_ = v_isSharedCheck_2414_;
goto v_resetjp_2403_;
}
else
{
lean_inc(v_tail_2402_);
lean_inc(v_head_2401_);
lean_dec(v_a_2398_);
v___x_2404_ = lean_box(0);
v_isShared_2405_ = v_isSharedCheck_2414_;
goto v_resetjp_2403_;
}
v_resetjp_2403_:
{
lean_object* v_fst_2406_; lean_object* v_snd_2407_; lean_object* v_mul_2408_; lean_object* v___x_2409_; lean_object* v___x_2411_; 
v_fst_2406_ = lean_ctor_get(v_head_2401_, 0);
lean_inc(v_fst_2406_);
v_snd_2407_ = lean_ctor_get(v_head_2401_, 1);
lean_inc(v_snd_2407_);
lean_dec(v_head_2401_);
v_mul_2408_ = lean_ctor_get(v___x_2397_, 4);
lean_inc(v_mul_2408_);
v___x_2409_ = lean_apply_2(v_mul_2408_, v_fst_2406_, v_snd_2407_);
if (v_isShared_2405_ == 0)
{
lean_ctor_set(v___x_2404_, 1, v_a_2399_);
lean_ctor_set(v___x_2404_, 0, v___x_2409_);
v___x_2411_ = v___x_2404_;
goto v_reusejp_2410_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v___x_2409_);
lean_ctor_set(v_reuseFailAlloc_2413_, 1, v_a_2399_);
v___x_2411_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2410_;
}
v_reusejp_2410_:
{
v_a_2398_ = v_tail_2402_;
v_a_2399_ = v___x_2411_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(lean_object* v_factors_2415_, lean_object* v_toSemiringOps_2416_, lean_object* v___x_2417_, lean_object* v_eqs_2418_){
_start:
{
lean_object* v___x_2419_; lean_object* v___x_2420_; 
v___x_2419_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_eqs_2418_, v_factors_2415_);
v___x_2420_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(v_toSemiringOps_2416_, v___x_2419_, v___x_2417_);
return v___x_2420_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(lean_object* v_nMax_2421_, lean_object* v_rs_2422_, lean_object* v___x_2423_, lean_object* v_a_2424_, lean_object* v_a_2425_){
_start:
{
if (lean_obj_tag(v_a_2424_) == 0)
{
lean_object* v___x_2426_; 
lean_dec(v___x_2423_);
v___x_2426_ = l_List_reverse___redArg(v_a_2425_);
return v___x_2426_;
}
else
{
lean_object* v_head_2427_; lean_object* v_tail_2428_; lean_object* v___x_2430_; uint8_t v_isShared_2431_; uint8_t v_isSharedCheck_2441_; 
v_head_2427_ = lean_ctor_get(v_a_2424_, 0);
v_tail_2428_ = lean_ctor_get(v_a_2424_, 1);
v_isSharedCheck_2441_ = !lean_is_exclusive(v_a_2424_);
if (v_isSharedCheck_2441_ == 0)
{
v___x_2430_ = v_a_2424_;
v_isShared_2431_ = v_isSharedCheck_2441_;
goto v_resetjp_2429_;
}
else
{
lean_inc(v_tail_2428_);
lean_inc(v_head_2427_);
lean_dec(v_a_2424_);
v___x_2430_ = lean_box(0);
v_isShared_2431_ = v_isSharedCheck_2441_;
goto v_resetjp_2429_;
}
v_resetjp_2429_:
{
lean_object* v___y_2433_; lean_object* v___x_2438_; lean_object* v___x_2439_; 
v___x_2438_ = lean_nat_sub(v_nMax_2421_, v_head_2427_);
lean_dec(v_head_2427_);
v___x_2439_ = l_List_get_x3fInternal___redArg(v_rs_2422_, v___x_2438_);
if (lean_obj_tag(v___x_2439_) == 0)
{
lean_inc(v___x_2423_);
v___y_2433_ = v___x_2423_;
goto v___jp_2432_;
}
else
{
lean_object* v_val_2440_; 
v_val_2440_ = lean_ctor_get(v___x_2439_, 0);
lean_inc(v_val_2440_);
lean_dec_ref_known(v___x_2439_, 1);
v___y_2433_ = v_val_2440_;
goto v___jp_2432_;
}
v___jp_2432_:
{
lean_object* v___x_2435_; 
if (v_isShared_2431_ == 0)
{
lean_ctor_set(v___x_2430_, 1, v_a_2425_);
lean_ctor_set(v___x_2430_, 0, v___y_2433_);
v___x_2435_ = v___x_2430_;
goto v_reusejp_2434_;
}
else
{
lean_object* v_reuseFailAlloc_2437_; 
v_reuseFailAlloc_2437_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2437_, 0, v___y_2433_);
lean_ctor_set(v_reuseFailAlloc_2437_, 1, v_a_2425_);
v___x_2435_ = v_reuseFailAlloc_2437_;
goto v_reusejp_2434_;
}
v_reusejp_2434_:
{
v_a_2424_ = v_tail_2428_;
v_a_2425_ = v___x_2435_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg___boxed(lean_object* v_nMax_2442_, lean_object* v_rs_2443_, lean_object* v___x_2444_, lean_object* v_a_2445_, lean_object* v_a_2446_){
_start:
{
lean_object* v_res_2447_; 
v_res_2447_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(v_nMax_2442_, v_rs_2443_, v___x_2444_, v_a_2445_, v_a_2446_);
lean_dec(v_rs_2443_);
lean_dec(v_nMax_2442_);
return v_res_2447_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(lean_object* v___x_2448_, lean_object* v_rs_2449_, lean_object* v___x_2450_, lean_object* v_lSkip_2451_, lean_object* v_xi_2452_, lean_object* v_a_2453_, lean_object* v_a_2454_){
_start:
{
if (lean_obj_tag(v_a_2453_) == 0)
{
lean_object* v___x_2455_; 
lean_dec(v___x_2450_);
lean_dec_ref(v___x_2448_);
v___x_2455_ = l_List_reverse___redArg(v_a_2454_);
return v___x_2455_;
}
else
{
lean_object* v_head_2456_; lean_object* v_tail_2457_; lean_object* v___x_2459_; uint8_t v_isShared_2460_; uint8_t v_isSharedCheck_2482_; 
v_head_2456_ = lean_ctor_get(v_a_2453_, 0);
v_tail_2457_ = lean_ctor_get(v_a_2453_, 1);
v_isSharedCheck_2482_ = !lean_is_exclusive(v_a_2453_);
if (v_isSharedCheck_2482_ == 0)
{
v___x_2459_ = v_a_2453_;
v_isShared_2460_ = v_isSharedCheck_2482_;
goto v_resetjp_2458_;
}
else
{
lean_inc(v_tail_2457_);
lean_inc(v_head_2456_);
lean_dec(v_a_2453_);
v___x_2459_ = lean_box(0);
v_isShared_2460_ = v_isSharedCheck_2482_;
goto v_resetjp_2458_;
}
v_resetjp_2458_:
{
lean_object* v___y_2462_; lean_object* v___y_2463_; lean_object* v___y_2464_; lean_object* v___y_2472_; lean_object* v___x_2479_; lean_object* v___x_2480_; 
v___x_2479_ = lean_nat_add(v_lSkip_2451_, v_head_2456_);
v___x_2480_ = l_List_get_x3fInternal___redArg(v_xi_2452_, v___x_2479_);
if (lean_obj_tag(v___x_2480_) == 0)
{
lean_inc(v___x_2450_);
v___y_2472_ = v___x_2450_;
goto v___jp_2471_;
}
else
{
lean_object* v_val_2481_; 
v_val_2481_ = lean_ctor_get(v___x_2480_, 0);
lean_inc(v_val_2481_);
lean_dec_ref_known(v___x_2480_, 1);
v___y_2472_ = v_val_2481_;
goto v___jp_2471_;
}
v___jp_2461_:
{
lean_object* v___x_2466_; 
lean_inc(v___y_2463_);
if (v_isShared_2460_ == 0)
{
lean_ctor_set(v___x_2459_, 1, v___y_2463_);
lean_ctor_set(v___x_2459_, 0, v___y_2464_);
v___x_2466_ = v___x_2459_;
goto v_reusejp_2465_;
}
else
{
lean_object* v_reuseFailAlloc_2470_; 
v_reuseFailAlloc_2470_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2470_, 0, v___y_2464_);
lean_ctor_set(v_reuseFailAlloc_2470_, 1, v___y_2463_);
v___x_2466_ = v_reuseFailAlloc_2470_;
goto v_reusejp_2465_;
}
v_reusejp_2465_:
{
lean_object* v___x_2467_; lean_object* v___x_2468_; 
lean_inc_ref(v___x_2448_);
v___x_2467_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v___x_2448_, v___y_2462_, v___x_2466_);
v___x_2468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2468_, 0, v___x_2467_);
lean_ctor_set(v___x_2468_, 1, v_a_2454_);
v_a_2453_ = v_tail_2457_;
v_a_2454_ = v___x_2468_;
goto _start;
}
}
v___jp_2471_:
{
lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; 
v___x_2473_ = lean_box(0);
v___x_2474_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2474_, 0, v___y_2472_);
lean_ctor_set(v___x_2474_, 1, v___x_2473_);
v___x_2475_ = lean_unsigned_to_nat(1u);
v___x_2476_ = lean_nat_add(v_head_2456_, v___x_2475_);
lean_dec(v_head_2456_);
v___x_2477_ = l_List_get_x3fInternal___redArg(v_rs_2449_, v___x_2476_);
if (lean_obj_tag(v___x_2477_) == 0)
{
lean_inc(v___x_2450_);
v___y_2462_ = v___x_2474_;
v___y_2463_ = v___x_2473_;
v___y_2464_ = v___x_2450_;
goto v___jp_2461_;
}
else
{
lean_object* v_val_2478_; 
v_val_2478_ = lean_ctor_get(v___x_2477_, 0);
lean_inc(v_val_2478_);
lean_dec_ref_known(v___x_2477_, 1);
v___y_2462_ = v___x_2474_;
v___y_2463_ = v___x_2473_;
v___y_2464_ = v_val_2478_;
goto v___jp_2461_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg___boxed(lean_object* v___x_2483_, lean_object* v_rs_2484_, lean_object* v___x_2485_, lean_object* v_lSkip_2486_, lean_object* v_xi_2487_, lean_object* v_a_2488_, lean_object* v_a_2489_){
_start:
{
lean_object* v_res_2490_; 
v_res_2490_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(v___x_2483_, v_rs_2484_, v___x_2485_, v_lSkip_2486_, v_xi_2487_, v_a_2488_, v_a_2489_);
lean_dec(v_xi_2487_);
lean_dec(v_lSkip_2486_);
lean_dec(v_rs_2484_);
return v_res_2490_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(lean_object* v_fo_2491_, lean_object* v_lSkip_2492_, lean_object* v_nMax_2493_, lean_object* v_xi_2494_, lean_object* v_rs_2495_, lean_object* v_omegaSkipPows_2496_){
_start:
{
lean_object* v_toRingOps_2497_; lean_object* v_toSemiringOps_2498_; lean_object* v_zero_2499_; lean_object* v_one_2500_; lean_object* v_mul_2501_; lean_object* v___f_2502_; lean_object* v___y_2504_; lean_object* v___y_2505_; lean_object* v___y_2522_; 
v_toRingOps_2497_ = lean_ctor_get(v_fo_2491_, 0);
lean_inc_ref(v_toRingOps_2497_);
v_toSemiringOps_2498_ = lean_ctor_get(v_toRingOps_2497_, 0);
lean_inc_ref(v_toSemiringOps_2498_);
v_zero_2499_ = lean_ctor_get(v_toSemiringOps_2498_, 0);
v_one_2500_ = lean_ctor_get(v_toSemiringOps_2498_, 1);
v_mul_2501_ = lean_ctor_get(v_toSemiringOps_2498_, 4);
lean_inc(v_mul_2501_);
v___f_2502_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_lagrangeAtIntegerPoints___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2502_, 0, v_mul_2501_);
if (lean_obj_tag(v_rs_2495_) == 0)
{
lean_inc(v_zero_2499_);
v___y_2522_ = v_zero_2499_;
goto v___jp_2521_;
}
else
{
lean_object* v_head_2524_; 
v_head_2524_ = lean_ctor_get(v_rs_2495_, 0);
lean_inc(v_head_2524_);
v___y_2522_ = v_head_2524_;
goto v___jp_2521_;
}
v___jp_2503_:
{
lean_object* v___x_2506_; lean_object* v_xiPrefix_2507_; lean_object* v_eqN0_2508_; lean_object* v_eqSharpN0_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v_multipliers_2512_; lean_object* v_eqNsUnadj_2513_; lean_object* v_eqSharpNsUnadj_2514_; lean_object* v___x_2515_; lean_object* v_factorsRev_2516_; lean_object* v_factors_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; 
v___x_2506_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0));
lean_inc_n(v_lSkip_2492_, 3);
lean_inc(v_xi_2494_);
v_xiPrefix_2507_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_xi_2494_, v_xi_2494_, v_lSkip_2492_, v___x_2506_);
lean_inc_ref(v_fo_2491_);
v_eqN0_2508_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_2491_, v_lSkip_2492_, v___y_2505_, v___y_2504_);
lean_dec(v___y_2505_);
v_eqSharpN0_2509_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(v_fo_2491_, v_lSkip_2492_, v_omegaSkipPows_2496_, v_xiPrefix_2507_, v___y_2504_);
lean_dec(v___y_2504_);
lean_inc(v_nMax_2493_);
v___x_2510_ = l_List_range(v_nMax_2493_);
v___x_2511_ = lean_box(0);
lean_inc(v___x_2510_);
lean_inc_n(v_zero_2499_, 2);
v_multipliers_2512_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(v_toRingOps_2497_, v_rs_2495_, v_zero_2499_, v_lSkip_2492_, v_xi_2494_, v___x_2510_, v___x_2511_);
lean_dec(v_xi_2494_);
lean_dec(v_lSkip_2492_);
lean_inc(v_multipliers_2512_);
lean_inc_ref_n(v___f_2502_, 2);
v_eqNsUnadj_2513_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2502_, v_eqN0_2508_, v_multipliers_2512_);
v_eqSharpNsUnadj_2514_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2502_, v_eqSharpN0_2509_, v_multipliers_2512_);
v___x_2515_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(v_nMax_2493_, v_rs_2495_, v_zero_2499_, v___x_2510_, v___x_2511_);
lean_dec(v_rs_2495_);
lean_dec(v_nMax_2493_);
lean_inc(v_one_2500_);
v_factorsRev_2516_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_scanl___redArg(v___f_2502_, v_one_2500_, v___x_2515_);
v_factors_2517_ = l_List_reverse___redArg(v_factorsRev_2516_);
lean_inc_ref(v_toSemiringOps_2498_);
lean_inc(v_factors_2517_);
v___x_2518_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(v_factors_2517_, v_toSemiringOps_2498_, v___x_2511_, v_eqNsUnadj_2513_);
v___x_2519_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg___lam__2(v_factors_2517_, v_toSemiringOps_2498_, v___x_2511_, v_eqSharpNsUnadj_2514_);
v___x_2520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2520_, 0, v___x_2518_);
lean_ctor_set(v___x_2520_, 1, v___x_2519_);
return v___x_2520_;
}
v___jp_2521_:
{
if (lean_obj_tag(v_xi_2494_) == 0)
{
lean_inc(v_zero_2499_);
v___y_2504_ = v___y_2522_;
v___y_2505_ = v_zero_2499_;
goto v___jp_2503_;
}
else
{
lean_object* v_head_2523_; 
v_head_2523_ = lean_ctor_get(v_xi_2494_, 0);
lean_inc(v_head_2523_);
v___y_2504_ = v___y_2522_;
v___y_2505_ = v_head_2523_;
goto v___jp_2503_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs(lean_object* v_EF_2525_, lean_object* v_fo_2526_, lean_object* v_lSkip_2527_, lean_object* v_nMax_2528_, lean_object* v_xi_2529_, lean_object* v_rs_2530_, lean_object* v_omegaSkipPows_2531_){
_start:
{
lean_object* v___x_2532_; 
v___x_2532_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(v_fo_2526_, v_lSkip_2527_, v_nMax_2528_, v_xi_2529_, v_rs_2530_, v_omegaSkipPows_2531_);
return v___x_2532_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0(lean_object* v_EF_2533_, lean_object* v___x_2534_, lean_object* v_rs_2535_, lean_object* v___x_2536_, lean_object* v_lSkip_2537_, lean_object* v_xi_2538_, lean_object* v_a_2539_, lean_object* v_a_2540_){
_start:
{
lean_object* v___x_2541_; 
v___x_2541_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___redArg(v___x_2534_, v_rs_2535_, v___x_2536_, v_lSkip_2537_, v_xi_2538_, v_a_2539_, v_a_2540_);
return v___x_2541_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0___boxed(lean_object* v_EF_2542_, lean_object* v___x_2543_, lean_object* v_rs_2544_, lean_object* v___x_2545_, lean_object* v_lSkip_2546_, lean_object* v_xi_2547_, lean_object* v_a_2548_, lean_object* v_a_2549_){
_start:
{
lean_object* v_res_2550_; 
v_res_2550_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__0(v_EF_2542_, v___x_2543_, v_rs_2544_, v___x_2545_, v_lSkip_2546_, v_xi_2547_, v_a_2548_, v_a_2549_);
lean_dec(v_xi_2547_);
lean_dec(v_lSkip_2546_);
lean_dec(v_rs_2544_);
return v_res_2550_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1(lean_object* v_EF_2551_, lean_object* v_nMax_2552_, lean_object* v_rs_2553_, lean_object* v___x_2554_, lean_object* v_a_2555_, lean_object* v_a_2556_){
_start:
{
lean_object* v___x_2557_; 
v___x_2557_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___redArg(v_nMax_2552_, v_rs_2553_, v___x_2554_, v_a_2555_, v_a_2556_);
return v___x_2557_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1___boxed(lean_object* v_EF_2558_, lean_object* v_nMax_2559_, lean_object* v_rs_2560_, lean_object* v___x_2561_, lean_object* v_a_2562_, lean_object* v_a_2563_){
_start:
{
lean_object* v_res_2564_; 
v_res_2564_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__1(v_EF_2558_, v_nMax_2559_, v_rs_2560_, v___x_2561_, v_a_2562_, v_a_2563_);
lean_dec(v_rs_2560_);
lean_dec(v_nMax_2559_);
return v_res_2564_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2(lean_object* v_EF_2565_, lean_object* v___x_2566_, lean_object* v_a_2567_, lean_object* v_a_2568_){
_start:
{
lean_object* v___x_2569_; 
v___x_2569_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs_spec__2___redArg(v___x_2566_, v_a_2567_, v_a_2568_);
return v___x_2569_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0(lean_object* v_a_2570_, lean_object* v_mul_2571_, lean_object* v_add_2572_, lean_object* v_acc_2573_, lean_object* v_entry_2574_){
_start:
{
lean_object* v_fst_2575_; lean_object* v_snd_2576_; lean_object* v___x_2577_; 
v_fst_2575_ = lean_ctor_get(v_entry_2574_, 0);
lean_inc(v_fst_2575_);
v_snd_2576_ = lean_ctor_get(v_entry_2574_, 1);
lean_inc(v_snd_2576_);
lean_dec_ref(v_entry_2574_);
v___x_2577_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2570_, v_snd_2576_);
if (lean_obj_tag(v___x_2577_) == 0)
{
lean_dec(v_fst_2575_);
lean_dec(v_acc_2573_);
lean_dec(v_add_2572_);
lean_dec(v_mul_2571_);
return v___x_2577_;
}
else
{
lean_object* v_a_2578_; lean_object* v___x_2580_; uint8_t v_isShared_2581_; uint8_t v_isSharedCheck_2587_; 
v_a_2578_ = lean_ctor_get(v___x_2577_, 0);
v_isSharedCheck_2587_ = !lean_is_exclusive(v___x_2577_);
if (v_isSharedCheck_2587_ == 0)
{
v___x_2580_ = v___x_2577_;
v_isShared_2581_ = v_isSharedCheck_2587_;
goto v_resetjp_2579_;
}
else
{
lean_inc(v_a_2578_);
lean_dec(v___x_2577_);
v___x_2580_ = lean_box(0);
v_isShared_2581_ = v_isSharedCheck_2587_;
goto v_resetjp_2579_;
}
v_resetjp_2579_:
{
lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2585_; 
v___x_2582_ = lean_apply_2(v_mul_2571_, v_a_2578_, v_fst_2575_);
v___x_2583_ = lean_apply_2(v_add_2572_, v_acc_2573_, v___x_2582_);
if (v_isShared_2581_ == 0)
{
lean_ctor_set(v___x_2580_, 0, v___x_2583_);
v___x_2585_ = v___x_2580_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2586_; 
v_reuseFailAlloc_2586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2586_, 0, v___x_2583_);
v___x_2585_ = v_reuseFailAlloc_2586_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
return v___x_2585_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0___boxed(lean_object* v_a_2588_, lean_object* v_mul_2589_, lean_object* v_add_2590_, lean_object* v_acc_2591_, lean_object* v_entry_2592_){
_start:
{
lean_object* v_res_2593_; 
v_res_2593_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0(v_a_2588_, v_mul_2589_, v_add_2590_, v_acc_2591_, v_entry_2592_);
lean_dec(v_a_2588_);
return v_res_2593_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1(lean_object* v_a_2594_, lean_object* v_idx_2595_){
_start:
{
lean_object* v___x_2596_; 
v___x_2596_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2594_, v_idx_2595_);
return v___x_2596_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1___boxed(lean_object* v_a_2597_, lean_object* v_idx_2598_){
_start:
{
lean_object* v_res_2599_; 
v_res_2599_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1(v_a_2597_, v_idx_2598_);
lean_dec(v_a_2597_);
return v_res_2599_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2(lean_object* v_mul_2600_, lean_object* v_add_2601_, lean_object* v_acc_2602_, lean_object* v_entry_2603_){
_start:
{
lean_object* v_fst_2604_; lean_object* v_snd_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; 
v_fst_2604_ = lean_ctor_get(v_entry_2603_, 0);
lean_inc(v_fst_2604_);
v_snd_2605_ = lean_ctor_get(v_entry_2603_, 1);
lean_inc(v_snd_2605_);
lean_dec_ref(v_entry_2603_);
v___x_2606_ = lean_apply_2(v_mul_2600_, v_fst_2604_, v_snd_2605_);
v___x_2607_ = lean_apply_2(v_add_2601_, v_acc_2602_, v___x_2606_);
return v___x_2607_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3(lean_object* v_a_2611_, lean_object* v___x_2612_, lean_object* v___f_2613_, lean_object* v_natCast_2614_, lean_object* v_add_2615_, lean_object* v_one_2616_, lean_object* v_fo_2617_, lean_object* v_betaLogup_2618_, lean_object* v___f_2619_, lean_object* v_zero_2620_, lean_object* v_interaction_2621_){
_start:
{
lean_object* v_message_2622_; lean_object* v_count_2623_; lean_object* v_busIndex_2624_; lean_object* v___x_2625_; 
v_message_2622_ = lean_ctor_get(v_interaction_2621_, 0);
lean_inc(v_message_2622_);
v_count_2623_ = lean_ctor_get(v_interaction_2621_, 1);
lean_inc(v_count_2623_);
v_busIndex_2624_ = lean_ctor_get(v_interaction_2621_, 2);
lean_inc(v_busIndex_2624_);
lean_dec_ref(v_interaction_2621_);
v___x_2625_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2611_, v_count_2623_);
if (lean_obj_tag(v___x_2625_) == 0)
{
lean_object* v___x_2626_; 
lean_dec_ref_known(v___x_2625_, 1);
lean_dec(v_busIndex_2624_);
lean_dec(v_message_2622_);
lean_dec(v_zero_2620_);
lean_dec(v___f_2619_);
lean_dec(v_betaLogup_2618_);
lean_dec_ref(v_fo_2617_);
lean_dec(v_one_2616_);
lean_dec(v_add_2615_);
lean_dec(v_natCast_2614_);
lean_dec_ref(v___f_2613_);
lean_dec_ref(v___x_2612_);
v___x_2626_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___closed__0));
return v___x_2626_;
}
else
{
lean_object* v_a_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; 
v_a_2627_ = lean_ctor_get(v___x_2625_, 0);
lean_inc(v_a_2627_);
lean_dec_ref_known(v___x_2625_, 1);
v___x_2628_ = lean_box(0);
v___x_2629_ = l_List_mapM_loop___redArg(v___x_2612_, v___f_2613_, v_message_2622_, v___x_2628_);
if (lean_obj_tag(v___x_2629_) == 0)
{
lean_object* v_a_2630_; lean_object* v___x_2632_; uint8_t v_isShared_2633_; uint8_t v_isSharedCheck_2637_; 
lean_dec(v_a_2627_);
lean_dec(v_busIndex_2624_);
lean_dec(v_zero_2620_);
lean_dec(v___f_2619_);
lean_dec(v_betaLogup_2618_);
lean_dec_ref(v_fo_2617_);
lean_dec(v_one_2616_);
lean_dec(v_add_2615_);
lean_dec(v_natCast_2614_);
v_a_2630_ = lean_ctor_get(v___x_2629_, 0);
v_isSharedCheck_2637_ = !lean_is_exclusive(v___x_2629_);
if (v_isSharedCheck_2637_ == 0)
{
v___x_2632_ = v___x_2629_;
v_isShared_2633_ = v_isSharedCheck_2637_;
goto v_resetjp_2631_;
}
else
{
lean_inc(v_a_2630_);
lean_dec(v___x_2629_);
v___x_2632_ = lean_box(0);
v_isShared_2633_ = v_isSharedCheck_2637_;
goto v_resetjp_2631_;
}
v_resetjp_2631_:
{
lean_object* v___x_2635_; 
if (v_isShared_2633_ == 0)
{
v___x_2635_ = v___x_2632_;
goto v_reusejp_2634_;
}
else
{
lean_object* v_reuseFailAlloc_2636_; 
v_reuseFailAlloc_2636_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2636_, 0, v_a_2630_);
v___x_2635_ = v_reuseFailAlloc_2636_;
goto v_reusejp_2634_;
}
v_reusejp_2634_:
{
return v___x_2635_;
}
}
}
else
{
lean_object* v_a_2638_; lean_object* v___x_2640_; uint8_t v_isShared_2641_; uint8_t v_isSharedCheck_2654_; 
v_a_2638_ = lean_ctor_get(v___x_2629_, 0);
v_isSharedCheck_2654_ = !lean_is_exclusive(v___x_2629_);
if (v_isSharedCheck_2654_ == 0)
{
v___x_2640_ = v___x_2629_;
v_isShared_2641_ = v_isSharedCheck_2654_;
goto v_resetjp_2639_;
}
else
{
lean_inc(v_a_2638_);
lean_dec(v___x_2629_);
v___x_2640_ = lean_box(0);
v_isShared_2641_ = v_isSharedCheck_2654_;
goto v_resetjp_2639_;
}
v_resetjp_2639_:
{
lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2652_; 
v___x_2642_ = lean_apply_1(v_natCast_2614_, v_busIndex_2624_);
v___x_2643_ = lean_apply_2(v_add_2615_, v___x_2642_, v_one_2616_);
v___x_2644_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2644_, 0, v___x_2643_);
lean_ctor_set(v___x_2644_, 1, v___x_2628_);
v___x_2645_ = l_List_appendTR___redArg(v_a_2638_, v___x_2644_);
v___x_2646_ = l_List_lengthTR___redArg(v___x_2645_);
v___x_2647_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_2617_, v_betaLogup_2618_, v___x_2646_);
v___x_2648_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_2645_, v___x_2647_);
v___x_2649_ = l_List_foldl___redArg(v___f_2619_, v_zero_2620_, v___x_2648_);
v___x_2650_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2650_, 0, v_a_2627_);
lean_ctor_set(v___x_2650_, 1, v___x_2649_);
if (v_isShared_2641_ == 0)
{
lean_ctor_set(v___x_2640_, 0, v___x_2650_);
v___x_2652_ = v___x_2640_;
goto v_reusejp_2651_;
}
else
{
lean_object* v_reuseFailAlloc_2653_; 
v_reuseFailAlloc_2653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2653_, 0, v___x_2650_);
v___x_2652_ = v_reuseFailAlloc_2653_;
goto v_reusejp_2651_;
}
v_reusejp_2651_:
{
return v___x_2652_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___boxed(lean_object* v_a_2655_, lean_object* v___x_2656_, lean_object* v___f_2657_, lean_object* v_natCast_2658_, lean_object* v_add_2659_, lean_object* v_one_2660_, lean_object* v_fo_2661_, lean_object* v_betaLogup_2662_, lean_object* v___f_2663_, lean_object* v_zero_2664_, lean_object* v_interaction_2665_){
_start:
{
lean_object* v_res_2666_; 
v_res_2666_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3(v_a_2655_, v___x_2656_, v___f_2657_, v_natCast_2658_, v_add_2659_, v_one_2660_, v_fo_2661_, v_betaLogup_2662_, v___f_2663_, v_zero_2664_, v_interaction_2665_);
lean_dec(v_a_2655_);
return v_res_2666_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__4(lean_object* v_mul_2667_, lean_object* v_add_2668_, lean_object* v_state_2669_, lean_object* v_entry_2670_){
_start:
{
lean_object* v_snd_2671_; lean_object* v_fst_2672_; lean_object* v_snd_2673_; lean_object* v_fst_2674_; lean_object* v_fst_2675_; lean_object* v_snd_2676_; lean_object* v___x_2678_; uint8_t v_isShared_2679_; uint8_t v_isSharedCheck_2687_; 
v_snd_2671_ = lean_ctor_get(v_entry_2670_, 1);
lean_inc(v_snd_2671_);
v_fst_2672_ = lean_ctor_get(v_state_2669_, 0);
lean_inc(v_fst_2672_);
v_snd_2673_ = lean_ctor_get(v_state_2669_, 1);
lean_inc(v_snd_2673_);
lean_dec_ref(v_state_2669_);
v_fst_2674_ = lean_ctor_get(v_entry_2670_, 0);
lean_inc(v_fst_2674_);
lean_dec_ref(v_entry_2670_);
v_fst_2675_ = lean_ctor_get(v_snd_2671_, 0);
v_snd_2676_ = lean_ctor_get(v_snd_2671_, 1);
v_isSharedCheck_2687_ = !lean_is_exclusive(v_snd_2671_);
if (v_isSharedCheck_2687_ == 0)
{
v___x_2678_ = v_snd_2671_;
v_isShared_2679_ = v_isSharedCheck_2687_;
goto v_resetjp_2677_;
}
else
{
lean_inc(v_snd_2676_);
lean_inc(v_fst_2675_);
lean_dec(v_snd_2671_);
v___x_2678_ = lean_box(0);
v_isShared_2679_ = v_isSharedCheck_2687_;
goto v_resetjp_2677_;
}
v_resetjp_2677_:
{
lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2685_; 
lean_inc(v_mul_2667_);
lean_inc(v_fst_2674_);
v___x_2680_ = lean_apply_2(v_mul_2667_, v_fst_2674_, v_fst_2675_);
lean_inc(v_add_2668_);
v___x_2681_ = lean_apply_2(v_add_2668_, v_fst_2672_, v___x_2680_);
v___x_2682_ = lean_apply_2(v_mul_2667_, v_fst_2674_, v_snd_2676_);
v___x_2683_ = lean_apply_2(v_add_2668_, v_snd_2673_, v___x_2682_);
if (v_isShared_2679_ == 0)
{
lean_ctor_set(v___x_2678_, 1, v___x_2683_);
lean_ctor_set(v___x_2678_, 0, v___x_2681_);
v___x_2685_ = v___x_2678_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2686_; 
v_reuseFailAlloc_2686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2686_, 0, v___x_2681_);
lean_ctor_set(v_reuseFailAlloc_2686_, 1, v___x_2683_);
v___x_2685_ = v_reuseFailAlloc_2686_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
return v___x_2685_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5(lean_object* v_fo_2691_, uint8_t v_needRot_2692_, lean_object* v_acc_2693_, lean_object* v_opening_2694_){
_start:
{
lean_object* v___x_2695_; 
v___x_2695_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_2691_, v_opening_2694_, v_needRot_2692_);
if (lean_obj_tag(v___x_2695_) == 0)
{
lean_object* v___x_2696_; 
lean_dec_ref_known(v___x_2695_, 1);
lean_dec(v_acc_2693_);
v___x_2696_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___closed__0));
return v___x_2696_;
}
else
{
lean_object* v_a_2697_; lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2707_; 
v_a_2697_ = lean_ctor_get(v___x_2695_, 0);
v_isSharedCheck_2707_ = !lean_is_exclusive(v___x_2695_);
if (v_isSharedCheck_2707_ == 0)
{
v___x_2699_ = v___x_2695_;
v_isShared_2700_ = v_isSharedCheck_2707_;
goto v_resetjp_2698_;
}
else
{
lean_inc(v_a_2697_);
lean_dec(v___x_2695_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2707_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2705_; 
v___x_2701_ = lean_box(0);
v___x_2702_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2702_, 0, v_a_2697_);
lean_ctor_set(v___x_2702_, 1, v___x_2701_);
v___x_2703_ = l_List_appendTR___redArg(v_acc_2693_, v___x_2702_);
if (v_isShared_2700_ == 0)
{
lean_ctor_set(v___x_2699_, 0, v___x_2703_);
v___x_2705_ = v___x_2699_;
goto v_reusejp_2704_;
}
else
{
lean_object* v_reuseFailAlloc_2706_; 
v_reuseFailAlloc_2706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2706_, 0, v___x_2703_);
v___x_2705_ = v_reuseFailAlloc_2706_;
goto v_reusejp_2704_;
}
v_reusejp_2704_:
{
return v___x_2705_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___boxed(lean_object* v_fo_2708_, lean_object* v_needRot_2709_, lean_object* v_acc_2710_, lean_object* v_opening_2711_){
_start:
{
uint8_t v_needRot_boxed_2712_; lean_object* v_res_2713_; 
v_needRot_boxed_2712_ = lean_unbox(v_needRot_2709_);
v_res_2713_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5(v_fo_2708_, v_needRot_boxed_2712_, v_acc_2710_, v_opening_2711_);
return v_res_2713_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1(void){
_start:
{
lean_object* v___x_2717_; lean_object* v___x_2718_; 
v___x_2717_ = lean_unsigned_to_nat(0u);
v___x_2718_ = lean_nat_to_int(v___x_2717_);
return v___x_2718_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(lean_object* v_fo_2719_, lean_object* v_algMap_2720_, lean_object* v_inst_2721_, lean_object* v_vk_2722_, lean_object* v_columnOpenings_2723_, lean_object* v_publicValues_2724_, lean_object* v_lSkip_2725_, lean_object* v_rs_2726_, lean_object* v_lambda_2727_, lean_object* v_betaLogup_2728_, lean_object* v_traceIdx_2729_, lean_object* v_airIdx_2730_, lean_object* v_n_2731_, lean_object* v_eq3bs_2732_, lean_object* v_eqXiR_2733_, lean_object* v_eqSharpXiR_2734_){
_start:
{
lean_object* v___x_2735_; lean_object* v___x_2736_; lean_object* v___x_2737_; 
v___x_2735_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9));
lean_inc(v_airIdx_2730_);
v___x_2736_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_2722_, v_airIdx_2730_);
v___x_2737_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchOption___redArg(v___x_2736_);
if (lean_obj_tag(v___x_2737_) == 0)
{
lean_object* v___x_2738_; 
lean_dec_ref_known(v___x_2737_, 1);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_traceIdx_2729_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2738_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2738_;
}
else
{
lean_object* v_a_2739_; lean_object* v___x_2740_; 
v_a_2739_ = lean_ctor_get(v___x_2737_, 0);
lean_inc(v_a_2739_);
lean_dec_ref_known(v___x_2737_, 1);
v___x_2740_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_columnOpenings_2723_, v_traceIdx_2729_);
if (lean_obj_tag(v___x_2740_) == 0)
{
lean_object* v___x_2741_; 
lean_dec_ref_known(v___x_2740_, 1);
lean_dec(v_a_2739_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2741_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2741_;
}
else
{
lean_object* v_a_2742_; lean_object* v___x_2743_; lean_object* v___x_2744_; 
v_a_2742_ = lean_ctor_get(v___x_2740_, 0);
lean_inc(v_a_2742_);
lean_dec_ref_known(v___x_2740_, 1);
v___x_2743_ = lean_unsigned_to_nat(0u);
v___x_2744_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2742_, v___x_2743_);
if (lean_obj_tag(v___x_2744_) == 0)
{
lean_object* v___x_2745_; 
lean_dec_ref_known(v___x_2744_, 1);
lean_dec(v_a_2742_);
lean_dec(v_a_2739_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2745_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2745_;
}
else
{
lean_object* v_a_2746_; lean_object* v_preprocessedData_2747_; lean_object* v_params_2748_; lean_object* v_symbolicConstraints_2749_; lean_object* v___y_2751_; lean_object* v___y_2752_; lean_object* v_fst_2753_; lean_object* v_fst_2754_; lean_object* v_snd_2755_; lean_object* v___y_2841_; lean_object* v_natCast_2842_; lean_object* v_pow_2843_; lean_object* v___y_2844_; lean_object* v___y_2845_; lean_object* v___y_2846_; lean_object* v___y_2847_; lean_object* v___y_2848_; lean_object* v___y_2849_; uint8_t v_needRot_2857_; lean_object* v___x_2858_; 
v_a_2746_ = lean_ctor_get(v___x_2744_, 0);
lean_inc(v_a_2746_);
lean_dec_ref_known(v___x_2744_, 1);
v_preprocessedData_2747_ = lean_ctor_get(v_a_2739_, 0);
lean_inc(v_preprocessedData_2747_);
v_params_2748_ = lean_ctor_get(v_a_2739_, 1);
lean_inc_ref(v_params_2748_);
v_symbolicConstraints_2749_ = lean_ctor_get(v_a_2739_, 2);
lean_inc_ref(v_symbolicConstraints_2749_);
lean_dec(v_a_2739_);
v_needRot_2857_ = lean_ctor_get_uint8(v_params_2748_, sizeof(void*)*2);
lean_dec_ref(v_params_2748_);
lean_inc_ref(v_fo_2719_);
v___x_2858_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_2719_, v_a_2746_, v_needRot_2857_);
if (lean_obj_tag(v___x_2858_) == 0)
{
lean_object* v___x_2859_; 
lean_dec_ref_known(v___x_2858_, 1);
lean_dec_ref(v_symbolicConstraints_2749_);
lean_dec(v_preprocessedData_2747_);
lean_dec(v_a_2742_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2859_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2859_;
}
else
{
lean_object* v_a_2860_; lean_object* v___x_2861_; lean_object* v___f_2862_; lean_object* v___y_2864_; lean_object* v___y_2865_; lean_object* v___y_2866_; 
v_a_2860_ = lean_ctor_get(v___x_2858_, 0);
lean_inc(v_a_2860_);
lean_dec_ref_known(v___x_2858_, 1);
v___x_2861_ = lean_box(v_needRot_2857_);
lean_inc_ref(v_fo_2719_);
v___f_2862_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__5___boxed), 4, 2);
lean_closure_set(v___f_2862_, 0, v_fo_2719_);
lean_closure_set(v___f_2862_, 1, v___x_2861_);
if (lean_obj_tag(v_preprocessedData_2747_) == 0)
{
lean_object* v___x_2902_; lean_object* v___x_2903_; 
v___x_2902_ = lean_box(0);
v___x_2903_ = lean_unsigned_to_nat(1u);
v___y_2864_ = v___x_2902_;
v___y_2865_ = v___x_2903_;
v___y_2866_ = v___x_2743_;
goto v___jp_2863_;
}
else
{
lean_object* v___x_2905_; uint8_t v_isShared_2906_; uint8_t v_isSharedCheck_2917_; 
v_isSharedCheck_2917_ = !lean_is_exclusive(v_preprocessedData_2747_);
if (v_isSharedCheck_2917_ == 0)
{
lean_object* v_unused_2918_; 
v_unused_2918_ = lean_ctor_get(v_preprocessedData_2747_, 0);
lean_dec(v_unused_2918_);
v___x_2905_ = v_preprocessedData_2747_;
v_isShared_2906_ = v_isSharedCheck_2917_;
goto v_resetjp_2904_;
}
else
{
lean_dec(v_preprocessedData_2747_);
v___x_2905_ = lean_box(0);
v_isShared_2906_ = v_isSharedCheck_2917_;
goto v_resetjp_2904_;
}
v_resetjp_2904_:
{
lean_object* v___x_2907_; lean_object* v___x_2908_; 
v___x_2907_ = lean_unsigned_to_nat(1u);
v___x_2908_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_a_2742_, v___x_2907_);
if (lean_obj_tag(v___x_2908_) == 0)
{
lean_object* v___x_2909_; 
lean_dec_ref_known(v___x_2908_, 1);
lean_del_object(v___x_2905_);
lean_dec_ref(v___f_2862_);
lean_dec(v_a_2860_);
lean_dec_ref(v_symbolicConstraints_2749_);
lean_dec(v_a_2742_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2909_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2909_;
}
else
{
lean_object* v_a_2910_; lean_object* v___x_2911_; 
v_a_2910_ = lean_ctor_get(v___x_2908_, 0);
lean_inc(v_a_2910_);
lean_dec_ref_known(v___x_2908_, 1);
lean_inc_ref(v_fo_2719_);
v___x_2911_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot___redArg(v_fo_2719_, v_a_2910_, v_needRot_2857_);
if (lean_obj_tag(v___x_2911_) == 0)
{
lean_object* v___x_2912_; 
lean_dec_ref_known(v___x_2911_, 1);
lean_del_object(v___x_2905_);
lean_dec_ref(v___f_2862_);
lean_dec(v_a_2860_);
lean_dec_ref(v_symbolicConstraints_2749_);
lean_dec(v_a_2742_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2912_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2912_;
}
else
{
lean_object* v_a_2913_; lean_object* v___x_2915_; 
v_a_2913_ = lean_ctor_get(v___x_2911_, 0);
lean_inc(v_a_2913_);
lean_dec_ref_known(v___x_2911_, 1);
if (v_isShared_2906_ == 0)
{
lean_ctor_set(v___x_2905_, 0, v_a_2913_);
v___x_2915_ = v___x_2905_;
goto v_reusejp_2914_;
}
else
{
lean_object* v_reuseFailAlloc_2916_; 
v_reuseFailAlloc_2916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2916_, 0, v_a_2913_);
v___x_2915_ = v_reuseFailAlloc_2916_;
goto v_reusejp_2914_;
}
v_reusejp_2914_:
{
v___y_2864_ = v___x_2915_;
v___y_2865_ = v___x_2907_;
v___y_2866_ = v___x_2907_;
goto v___jp_2863_;
}
}
}
}
}
v___jp_2863_:
{
lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; 
v___x_2867_ = lean_nat_add(v___y_2865_, v___y_2866_);
lean_dec(v___y_2866_);
v___x_2868_ = l_List_drop___redArg(v___x_2867_, v_a_2742_);
lean_dec(v_a_2742_);
v___x_2869_ = lean_box(0);
v___x_2870_ = l_List_foldlM___redArg(v___x_2735_, v___f_2862_, v___x_2869_, v___x_2868_);
if (lean_obj_tag(v___x_2870_) == 0)
{
lean_object* v_a_2871_; lean_object* v___x_2873_; uint8_t v_isShared_2874_; uint8_t v_isSharedCheck_2878_; 
lean_dec(v___y_2865_);
lean_dec(v___y_2864_);
lean_dec(v_a_2860_);
lean_dec_ref(v_symbolicConstraints_2749_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_airIdx_2730_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_rs_2726_);
lean_dec(v_lSkip_2725_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v_a_2871_ = lean_ctor_get(v___x_2870_, 0);
v_isSharedCheck_2878_ = !lean_is_exclusive(v___x_2870_);
if (v_isSharedCheck_2878_ == 0)
{
v___x_2873_ = v___x_2870_;
v_isShared_2874_ = v_isSharedCheck_2878_;
goto v_resetjp_2872_;
}
else
{
lean_inc(v_a_2871_);
lean_dec(v___x_2870_);
v___x_2873_ = lean_box(0);
v_isShared_2874_ = v_isSharedCheck_2878_;
goto v_resetjp_2872_;
}
v_resetjp_2872_:
{
lean_object* v___x_2876_; 
if (v_isShared_2874_ == 0)
{
v___x_2876_ = v___x_2873_;
goto v_reusejp_2875_;
}
else
{
lean_object* v_reuseFailAlloc_2877_; 
v_reuseFailAlloc_2877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2877_, 0, v_a_2871_);
v___x_2876_ = v_reuseFailAlloc_2877_;
goto v_reusejp_2875_;
}
v_reusejp_2875_:
{
return v___x_2876_;
}
}
}
else
{
lean_object* v_a_2879_; lean_object* v___x_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; uint8_t v___x_2883_; 
v_a_2879_ = lean_ctor_get(v___x_2870_, 0);
lean_inc(v_a_2879_);
lean_dec_ref_known(v___x_2870_, 1);
v___x_2880_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2880_, 0, v_a_2860_);
lean_ctor_set(v___x_2880_, 1, v___x_2869_);
v___x_2881_ = l_List_appendTR___redArg(v_a_2879_, v___x_2880_);
v___x_2882_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__1);
v___x_2883_ = lean_int_dec_lt(v_n_2731_, v___x_2882_);
if (v___x_2883_ == 0)
{
lean_object* v_toRingOps_2884_; lean_object* v_toSemiringOps_2885_; lean_object* v_one_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; 
v_toRingOps_2884_ = lean_ctor_get(v_fo_2719_, 0);
v_toSemiringOps_2885_ = lean_ctor_get(v_toRingOps_2884_, 0);
v_one_2886_ = lean_ctor_get(v_toSemiringOps_2885_, 1);
v___x_2887_ = l_Int_toNat(v_n_2731_);
v___x_2888_ = lean_nat_add(v___x_2887_, v___y_2865_);
lean_dec(v___y_2865_);
lean_dec(v___x_2887_);
v___x_2889_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace_spec__2___redArg___closed__0));
lean_inc(v_rs_2726_);
v___x_2890_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_rs_2726_, v_rs_2726_, v___x_2888_, v___x_2889_);
lean_dec(v_rs_2726_);
lean_inc(v_one_2886_);
v___y_2751_ = v___x_2881_;
v___y_2752_ = v___y_2864_;
v_fst_2753_ = v_lSkip_2725_;
v_fst_2754_ = v___x_2890_;
v_snd_2755_ = v_one_2886_;
goto v___jp_2750_;
}
else
{
lean_object* v_toRingOps_2891_; lean_object* v_inv_2892_; lean_object* v_toSemiringOps_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; 
lean_dec(v___y_2865_);
v_toRingOps_2891_ = lean_ctor_get(v_fo_2719_, 0);
v_inv_2892_ = lean_ctor_get(v_fo_2719_, 1);
v_toSemiringOps_2893_ = lean_ctor_get(v_toRingOps_2891_, 0);
v___x_2894_ = lean_nat_abs(v_n_2731_);
v___x_2895_ = lean_nat_sub(v_lSkip_2725_, v___x_2894_);
lean_dec(v_lSkip_2725_);
if (lean_obj_tag(v_rs_2726_) == 0)
{
lean_object* v_zero_2896_; lean_object* v_natCast_2897_; lean_object* v_pow_2898_; 
v_zero_2896_ = lean_ctor_get(v_toSemiringOps_2893_, 0);
v_natCast_2897_ = lean_ctor_get(v_toSemiringOps_2893_, 2);
v_pow_2898_ = lean_ctor_get(v_toSemiringOps_2893_, 5);
lean_inc(v_zero_2896_);
lean_inc(v_inv_2892_);
lean_inc(v_pow_2898_);
lean_inc(v_natCast_2897_);
lean_inc_ref(v_toSemiringOps_2893_);
v___y_2841_ = v_toSemiringOps_2893_;
v_natCast_2842_ = v_natCast_2897_;
v_pow_2843_ = v_pow_2898_;
v___y_2844_ = v___x_2881_;
v___y_2845_ = v___y_2864_;
v___y_2846_ = v___x_2895_;
v___y_2847_ = v___x_2894_;
v___y_2848_ = v_inv_2892_;
v___y_2849_ = v_zero_2896_;
goto v___jp_2840_;
}
else
{
lean_object* v_head_2899_; lean_object* v_natCast_2900_; lean_object* v_pow_2901_; 
v_head_2899_ = lean_ctor_get(v_rs_2726_, 0);
lean_inc(v_head_2899_);
lean_dec_ref_known(v_rs_2726_, 2);
v_natCast_2900_ = lean_ctor_get(v_toSemiringOps_2893_, 2);
v_pow_2901_ = lean_ctor_get(v_toSemiringOps_2893_, 5);
lean_inc(v_inv_2892_);
lean_inc(v_pow_2901_);
lean_inc(v_natCast_2900_);
lean_inc_ref(v_toSemiringOps_2893_);
v___y_2841_ = v_toSemiringOps_2893_;
v_natCast_2842_ = v_natCast_2900_;
v_pow_2843_ = v_pow_2901_;
v___y_2844_ = v___x_2881_;
v___y_2845_ = v___y_2864_;
v___y_2846_ = v___x_2895_;
v___y_2847_ = v___x_2894_;
v___y_2848_ = v_inv_2892_;
v___y_2849_ = v_head_2899_;
goto v___jp_2840_;
}
}
}
}
}
v___jp_2750_:
{
lean_object* v___x_2756_; 
v___x_2756_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_publicValues_2724_, v_airIdx_2730_);
if (lean_obj_tag(v___x_2756_) == 0)
{
lean_object* v___x_2757_; 
lean_dec_ref_known(v___x_2756_, 1);
lean_dec(v_snd_2755_);
lean_dec(v_fst_2754_);
lean_dec(v_fst_2753_);
lean_dec(v___y_2752_);
lean_dec(v___y_2751_);
lean_dec_ref(v_symbolicConstraints_2749_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec(v_inst_2721_);
lean_dec(v_algMap_2720_);
lean_dec_ref(v_fo_2719_);
v___x_2757_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2757_;
}
else
{
lean_object* v_constraints_2758_; lean_object* v_a_2759_; lean_object* v_interactions_2760_; lean_object* v_nodes_2761_; lean_object* v_constraintIdx_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2839_; 
v_constraints_2758_ = lean_ctor_get(v_symbolicConstraints_2749_, 0);
lean_inc_ref(v_constraints_2758_);
v_a_2759_ = lean_ctor_get(v___x_2756_, 0);
lean_inc(v_a_2759_);
lean_dec_ref_known(v___x_2756_, 1);
v_interactions_2760_ = lean_ctor_get(v_symbolicConstraints_2749_, 1);
lean_inc(v_interactions_2760_);
lean_dec_ref(v_symbolicConstraints_2749_);
v_nodes_2761_ = lean_ctor_get(v_constraints_2758_, 0);
v_constraintIdx_2762_ = lean_ctor_get(v_constraints_2758_, 1);
v_isSharedCheck_2839_ = !lean_is_exclusive(v_constraints_2758_);
if (v_isSharedCheck_2839_ == 0)
{
v___x_2764_ = v_constraints_2758_;
v_isShared_2765_ = v_isSharedCheck_2839_;
goto v_resetjp_2763_;
}
else
{
lean_inc(v_constraintIdx_2762_);
lean_inc(v_nodes_2761_);
lean_dec(v_constraints_2758_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2839_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v___x_2766_; lean_object* v___x_2767_; 
lean_inc_ref_n(v_fo_2719_, 2);
v___x_2766_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_mk_x27___redArg(v_fo_2719_, v_inst_2721_, v___y_2752_, v___y_2751_, v_a_2759_, v_fst_2754_, v_fst_2753_);
v___x_2767_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_VerifierConstraintEvaluator_evalNodesM___redArg(v_fo_2719_, v_algMap_2720_, v___x_2766_, v_nodes_2761_);
if (lean_obj_tag(v___x_2767_) == 0)
{
lean_object* v___x_2768_; 
lean_dec_ref_known(v___x_2767_, 1);
lean_del_object(v___x_2764_);
lean_dec(v_constraintIdx_2762_);
lean_dec(v_interactions_2760_);
lean_dec(v_snd_2755_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_betaLogup_2728_);
lean_dec(v_lambda_2727_);
lean_dec_ref(v_fo_2719_);
v___x_2768_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___closed__0));
return v___x_2768_;
}
else
{
lean_object* v_toRingOps_2769_; lean_object* v_toSemiringOps_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2837_; 
v_toRingOps_2769_ = lean_ctor_get(v_fo_2719_, 0);
lean_inc_ref(v_toRingOps_2769_);
v_toSemiringOps_2770_ = lean_ctor_get(v_toRingOps_2769_, 0);
v_isSharedCheck_2837_ = !lean_is_exclusive(v_toRingOps_2769_);
if (v_isSharedCheck_2837_ == 0)
{
lean_object* v_unused_2838_; 
v_unused_2838_ = lean_ctor_get(v_toRingOps_2769_, 1);
lean_dec(v_unused_2838_);
v___x_2772_ = v_toRingOps_2769_;
v_isShared_2773_ = v_isSharedCheck_2837_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_toSemiringOps_2770_);
lean_dec(v_toRingOps_2769_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2837_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
lean_object* v_a_2774_; lean_object* v_zero_2775_; lean_object* v_one_2776_; lean_object* v_natCast_2777_; lean_object* v_add_2778_; lean_object* v_mul_2779_; lean_object* v___f_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; 
v_a_2774_ = lean_ctor_get(v___x_2767_, 0);
lean_inc_n(v_a_2774_, 2);
lean_dec_ref_known(v___x_2767_, 1);
v_zero_2775_ = lean_ctor_get(v_toSemiringOps_2770_, 0);
lean_inc_n(v_zero_2775_, 2);
v_one_2776_ = lean_ctor_get(v_toSemiringOps_2770_, 1);
lean_inc(v_one_2776_);
v_natCast_2777_ = lean_ctor_get(v_toSemiringOps_2770_, 2);
lean_inc(v_natCast_2777_);
v_add_2778_ = lean_ctor_get(v_toSemiringOps_2770_, 3);
lean_inc_n(v_add_2778_, 2);
v_mul_2779_ = lean_ctor_get(v_toSemiringOps_2770_, 4);
lean_inc_n(v_mul_2779_, 2);
lean_dec_ref(v_toSemiringOps_2770_);
v___f_2780_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_2780_, 0, v_a_2774_);
lean_closure_set(v___f_2780_, 1, v_mul_2779_);
lean_closure_set(v___f_2780_, 2, v_add_2778_);
v___x_2781_ = l_List_lengthTR___redArg(v_constraintIdx_2762_);
lean_inc_ref(v_fo_2719_);
v___x_2782_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_2719_, v_lambda_2727_, v___x_2781_);
v___x_2783_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_2782_, v_constraintIdx_2762_);
v___x_2784_ = l_List_foldlM___redArg(v___x_2735_, v___f_2780_, v_zero_2775_, v___x_2783_);
if (lean_obj_tag(v___x_2784_) == 0)
{
lean_object* v_a_2785_; lean_object* v___x_2787_; uint8_t v_isShared_2788_; uint8_t v_isSharedCheck_2792_; 
lean_dec(v_mul_2779_);
lean_dec(v_add_2778_);
lean_dec(v_natCast_2777_);
lean_dec(v_one_2776_);
lean_dec(v_zero_2775_);
lean_dec(v_a_2774_);
lean_del_object(v___x_2772_);
lean_del_object(v___x_2764_);
lean_dec(v_interactions_2760_);
lean_dec(v_snd_2755_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
lean_dec(v_betaLogup_2728_);
lean_dec_ref(v_fo_2719_);
v_a_2785_ = lean_ctor_get(v___x_2784_, 0);
v_isSharedCheck_2792_ = !lean_is_exclusive(v___x_2784_);
if (v_isSharedCheck_2792_ == 0)
{
v___x_2787_ = v___x_2784_;
v_isShared_2788_ = v_isSharedCheck_2792_;
goto v_resetjp_2786_;
}
else
{
lean_inc(v_a_2785_);
lean_dec(v___x_2784_);
v___x_2787_ = lean_box(0);
v_isShared_2788_ = v_isSharedCheck_2792_;
goto v_resetjp_2786_;
}
v_resetjp_2786_:
{
lean_object* v___x_2790_; 
if (v_isShared_2788_ == 0)
{
v___x_2790_ = v___x_2787_;
goto v_reusejp_2789_;
}
else
{
lean_object* v_reuseFailAlloc_2791_; 
v_reuseFailAlloc_2791_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2791_, 0, v_a_2785_);
v___x_2790_ = v_reuseFailAlloc_2791_;
goto v_reusejp_2789_;
}
v_reusejp_2789_:
{
return v___x_2790_;
}
}
}
else
{
lean_object* v_a_2793_; lean_object* v___f_2794_; lean_object* v___f_2795_; lean_object* v___f_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; 
v_a_2793_ = lean_ctor_get(v___x_2784_, 0);
lean_inc(v_a_2793_);
lean_dec_ref_known(v___x_2784_, 1);
lean_inc(v_a_2774_);
v___f_2794_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_2794_, 0, v_a_2774_);
lean_inc_n(v_add_2778_, 2);
lean_inc(v_mul_2779_);
v___f_2795_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2), 4, 2);
lean_closure_set(v___f_2795_, 0, v_mul_2779_);
lean_closure_set(v___f_2795_, 1, v_add_2778_);
lean_inc(v_zero_2775_);
v___f_2796_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__3___boxed), 11, 10);
lean_closure_set(v___f_2796_, 0, v_a_2774_);
lean_closure_set(v___f_2796_, 1, v___x_2735_);
lean_closure_set(v___f_2796_, 2, v___f_2794_);
lean_closure_set(v___f_2796_, 3, v_natCast_2777_);
lean_closure_set(v___f_2796_, 4, v_add_2778_);
lean_closure_set(v___f_2796_, 5, v_one_2776_);
lean_closure_set(v___f_2796_, 6, v_fo_2719_);
lean_closure_set(v___f_2796_, 7, v_betaLogup_2728_);
lean_closure_set(v___f_2796_, 8, v___f_2795_);
lean_closure_set(v___f_2796_, 9, v_zero_2775_);
v___x_2797_ = lean_box(0);
v___x_2798_ = l_List_mapM_loop___redArg(v___x_2735_, v___f_2796_, v_interactions_2760_, v___x_2797_);
if (lean_obj_tag(v___x_2798_) == 0)
{
lean_object* v_a_2799_; lean_object* v___x_2801_; uint8_t v_isShared_2802_; uint8_t v_isSharedCheck_2806_; 
lean_dec(v_a_2793_);
lean_dec(v_mul_2779_);
lean_dec(v_add_2778_);
lean_dec(v_zero_2775_);
lean_del_object(v___x_2772_);
lean_del_object(v___x_2764_);
lean_dec(v_snd_2755_);
lean_dec(v_eqSharpXiR_2734_);
lean_dec(v_eqXiR_2733_);
lean_dec(v_eq3bs_2732_);
v_a_2799_ = lean_ctor_get(v___x_2798_, 0);
v_isSharedCheck_2806_ = !lean_is_exclusive(v___x_2798_);
if (v_isSharedCheck_2806_ == 0)
{
v___x_2801_ = v___x_2798_;
v_isShared_2802_ = v_isSharedCheck_2806_;
goto v_resetjp_2800_;
}
else
{
lean_inc(v_a_2799_);
lean_dec(v___x_2798_);
v___x_2801_ = lean_box(0);
v_isShared_2802_ = v_isSharedCheck_2806_;
goto v_resetjp_2800_;
}
v_resetjp_2800_:
{
lean_object* v___x_2804_; 
if (v_isShared_2802_ == 0)
{
v___x_2804_ = v___x_2801_;
goto v_reusejp_2803_;
}
else
{
lean_object* v_reuseFailAlloc_2805_; 
v_reuseFailAlloc_2805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2805_, 0, v_a_2799_);
v___x_2804_ = v_reuseFailAlloc_2805_;
goto v_reusejp_2803_;
}
v_reusejp_2803_:
{
return v___x_2804_;
}
}
}
else
{
lean_object* v_a_2807_; lean_object* v___x_2809_; uint8_t v_isShared_2810_; uint8_t v_isSharedCheck_2836_; 
v_a_2807_ = lean_ctor_get(v___x_2798_, 0);
v_isSharedCheck_2836_ = !lean_is_exclusive(v___x_2798_);
if (v_isSharedCheck_2836_ == 0)
{
v___x_2809_ = v___x_2798_;
v_isShared_2810_ = v_isSharedCheck_2836_;
goto v_resetjp_2808_;
}
else
{
lean_inc(v_a_2807_);
lean_dec(v___x_2798_);
v___x_2809_ = lean_box(0);
v_isShared_2810_ = v_isSharedCheck_2836_;
goto v_resetjp_2808_;
}
v_resetjp_2808_:
{
lean_object* v___f_2811_; lean_object* v___x_2813_; 
lean_inc(v_mul_2779_);
v___f_2811_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2811_, 0, v_mul_2779_);
lean_closure_set(v___f_2811_, 1, v_add_2778_);
lean_inc(v_zero_2775_);
if (v_isShared_2773_ == 0)
{
lean_ctor_set(v___x_2772_, 1, v_zero_2775_);
lean_ctor_set(v___x_2772_, 0, v_zero_2775_);
v___x_2813_ = v___x_2772_;
goto v_reusejp_2812_;
}
else
{
lean_object* v_reuseFailAlloc_2835_; 
v_reuseFailAlloc_2835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2835_, 0, v_zero_2775_);
lean_ctor_set(v_reuseFailAlloc_2835_, 1, v_zero_2775_);
v___x_2813_ = v_reuseFailAlloc_2835_;
goto v_reusejp_2812_;
}
v_reusejp_2812_:
{
lean_object* v___x_2814_; lean_object* v___x_2815_; lean_object* v_fst_2816_; lean_object* v_snd_2817_; lean_object* v___x_2819_; uint8_t v_isShared_2820_; uint8_t v_isSharedCheck_2834_; 
v___x_2814_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_eq3bs_2732_, v_a_2807_);
v___x_2815_ = l_List_foldl___redArg(v___f_2811_, v___x_2813_, v___x_2814_);
v_fst_2816_ = lean_ctor_get(v___x_2815_, 0);
v_snd_2817_ = lean_ctor_get(v___x_2815_, 1);
v_isSharedCheck_2834_ = !lean_is_exclusive(v___x_2815_);
if (v_isSharedCheck_2834_ == 0)
{
v___x_2819_ = v___x_2815_;
v_isShared_2820_ = v_isSharedCheck_2834_;
goto v_resetjp_2818_;
}
else
{
lean_inc(v_snd_2817_);
lean_inc(v_fst_2816_);
lean_dec(v___x_2815_);
v___x_2819_ = lean_box(0);
v_isShared_2820_ = v_isSharedCheck_2834_;
goto v_resetjp_2818_;
}
v_resetjp_2818_:
{
lean_object* v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2826_; 
lean_inc_n(v_mul_2779_, 3);
v___x_2821_ = lean_apply_2(v_mul_2779_, v_eqXiR_2733_, v_a_2793_);
v___x_2822_ = lean_apply_2(v_mul_2779_, v_fst_2816_, v_snd_2755_);
lean_inc(v_eqSharpXiR_2734_);
v___x_2823_ = lean_apply_2(v_mul_2779_, v___x_2822_, v_eqSharpXiR_2734_);
v___x_2824_ = lean_apply_2(v_mul_2779_, v_snd_2817_, v_eqSharpXiR_2734_);
if (v_isShared_2820_ == 0)
{
lean_ctor_set(v___x_2819_, 1, v___x_2821_);
lean_ctor_set(v___x_2819_, 0, v___x_2824_);
v___x_2826_ = v___x_2819_;
goto v_reusejp_2825_;
}
else
{
lean_object* v_reuseFailAlloc_2833_; 
v_reuseFailAlloc_2833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2833_, 0, v___x_2824_);
lean_ctor_set(v_reuseFailAlloc_2833_, 1, v___x_2821_);
v___x_2826_ = v_reuseFailAlloc_2833_;
goto v_reusejp_2825_;
}
v_reusejp_2825_:
{
lean_object* v___x_2828_; 
if (v_isShared_2765_ == 0)
{
lean_ctor_set(v___x_2764_, 1, v___x_2826_);
lean_ctor_set(v___x_2764_, 0, v___x_2823_);
v___x_2828_ = v___x_2764_;
goto v_reusejp_2827_;
}
else
{
lean_object* v_reuseFailAlloc_2832_; 
v_reuseFailAlloc_2832_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2832_, 0, v___x_2823_);
lean_ctor_set(v_reuseFailAlloc_2832_, 1, v___x_2826_);
v___x_2828_ = v_reuseFailAlloc_2832_;
goto v_reusejp_2827_;
}
v_reusejp_2827_:
{
lean_object* v___x_2830_; 
if (v_isShared_2810_ == 0)
{
lean_ctor_set(v___x_2809_, 0, v___x_2828_);
v___x_2830_ = v___x_2809_;
goto v_reusejp_2829_;
}
else
{
lean_object* v_reuseFailAlloc_2831_; 
v_reuseFailAlloc_2831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2831_, 0, v___x_2828_);
v___x_2830_ = v_reuseFailAlloc_2831_;
goto v_reusejp_2829_;
}
v_reusejp_2829_:
{
return v___x_2830_;
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
v___jp_2840_:
{
lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; 
v___x_2850_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v___y_2841_, v___y_2849_, v___y_2847_);
lean_dec(v___y_2849_);
v___x_2851_ = lean_box(0);
v___x_2852_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2852_, 0, v___x_2850_);
lean_ctor_set(v___x_2852_, 1, v___x_2851_);
v___x_2853_ = lean_unsigned_to_nat(2u);
v___x_2854_ = lean_apply_1(v_natCast_2842_, v___x_2853_);
v___x_2855_ = lean_apply_2(v_pow_2843_, v___x_2854_, v___y_2847_);
v___x_2856_ = lean_apply_1(v___y_2848_, v___x_2855_);
v___y_2751_ = v___y_2844_;
v___y_2752_ = v___y_2845_;
v_fst_2753_ = v___y_2846_;
v_fst_2754_ = v___x_2852_;
v_snd_2755_ = v___x_2856_;
goto v___jp_2750_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___boxed(lean_object* v_fo_2919_, lean_object* v_algMap_2920_, lean_object* v_inst_2921_, lean_object* v_vk_2922_, lean_object* v_columnOpenings_2923_, lean_object* v_publicValues_2924_, lean_object* v_lSkip_2925_, lean_object* v_rs_2926_, lean_object* v_lambda_2927_, lean_object* v_betaLogup_2928_, lean_object* v_traceIdx_2929_, lean_object* v_airIdx_2930_, lean_object* v_n_2931_, lean_object* v_eq3bs_2932_, lean_object* v_eqXiR_2933_, lean_object* v_eqSharpXiR_2934_){
_start:
{
lean_object* v_res_2935_; 
v_res_2935_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(v_fo_2919_, v_algMap_2920_, v_inst_2921_, v_vk_2922_, v_columnOpenings_2923_, v_publicValues_2924_, v_lSkip_2925_, v_rs_2926_, v_lambda_2927_, v_betaLogup_2928_, v_traceIdx_2929_, v_airIdx_2930_, v_n_2931_, v_eq3bs_2932_, v_eqXiR_2933_, v_eqSharpXiR_2934_);
lean_dec(v_n_2931_);
lean_dec(v_publicValues_2924_);
lean_dec(v_columnOpenings_2923_);
lean_dec_ref(v_vk_2922_);
return v_res_2935_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace(lean_object* v_F_2936_, lean_object* v_EF_2937_, lean_object* v_Digest_2938_, lean_object* v_fo_2939_, lean_object* v_algMap_2940_, lean_object* v_inst_2941_, lean_object* v_vk_2942_, lean_object* v_columnOpenings_2943_, lean_object* v_publicValues_2944_, lean_object* v_lSkip_2945_, lean_object* v_rs_2946_, lean_object* v_lambda_2947_, lean_object* v_betaLogup_2948_, lean_object* v_traceIdx_2949_, lean_object* v_airIdx_2950_, lean_object* v_n_2951_, lean_object* v_eq3bs_2952_, lean_object* v_eqXiR_2953_, lean_object* v_eqSharpXiR_2954_){
_start:
{
lean_object* v___x_2955_; 
v___x_2955_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(v_fo_2939_, v_algMap_2940_, v_inst_2941_, v_vk_2942_, v_columnOpenings_2943_, v_publicValues_2944_, v_lSkip_2945_, v_rs_2946_, v_lambda_2947_, v_betaLogup_2948_, v_traceIdx_2949_, v_airIdx_2950_, v_n_2951_, v_eq3bs_2952_, v_eqXiR_2953_, v_eqSharpXiR_2954_);
return v___x_2955_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___boxed(lean_object** _args){
lean_object* v_F_2956_ = _args[0];
lean_object* v_EF_2957_ = _args[1];
lean_object* v_Digest_2958_ = _args[2];
lean_object* v_fo_2959_ = _args[3];
lean_object* v_algMap_2960_ = _args[4];
lean_object* v_inst_2961_ = _args[5];
lean_object* v_vk_2962_ = _args[6];
lean_object* v_columnOpenings_2963_ = _args[7];
lean_object* v_publicValues_2964_ = _args[8];
lean_object* v_lSkip_2965_ = _args[9];
lean_object* v_rs_2966_ = _args[10];
lean_object* v_lambda_2967_ = _args[11];
lean_object* v_betaLogup_2968_ = _args[12];
lean_object* v_traceIdx_2969_ = _args[13];
lean_object* v_airIdx_2970_ = _args[14];
lean_object* v_n_2971_ = _args[15];
lean_object* v_eq3bs_2972_ = _args[16];
lean_object* v_eqXiR_2973_ = _args[17];
lean_object* v_eqSharpXiR_2974_ = _args[18];
_start:
{
lean_object* v_res_2975_; 
v_res_2975_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace(v_F_2956_, v_EF_2957_, v_Digest_2958_, v_fo_2959_, v_algMap_2960_, v_inst_2961_, v_vk_2962_, v_columnOpenings_2963_, v_publicValues_2964_, v_lSkip_2965_, v_rs_2966_, v_lambda_2967_, v_betaLogup_2968_, v_traceIdx_2969_, v_airIdx_2970_, v_n_2971_, v_eq3bs_2972_, v_eqXiR_2973_, v_eqSharpXiR_2974_);
lean_dec(v_n_2971_);
lean_dec(v_publicValues_2964_);
lean_dec(v_columnOpenings_2963_);
lean_dec_ref(v_vk_2962_);
return v_res_2975_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0(lean_object* v_fo_2976_, lean_object* v_00___2977_){
_start:
{
lean_object* v_toRingOps_2978_; lean_object* v_toSemiringOps_2979_; lean_object* v_zero_2980_; 
v_toRingOps_2978_ = lean_ctor_get(v_fo_2976_, 0);
v_toSemiringOps_2979_ = lean_ctor_get(v_toRingOps_2978_, 0);
v_zero_2980_ = lean_ctor_get(v_toSemiringOps_2979_, 0);
lean_inc(v_zero_2980_);
return v_zero_2980_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0___boxed(lean_object* v_fo_2981_, lean_object* v_00___2982_){
_start:
{
lean_object* v_res_2983_; 
v_res_2983_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0(v_fo_2981_, v_00___2982_);
lean_dec_ref(v_fo_2981_);
return v_res_2983_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1(lean_object* v_nPerTrace_2987_, lean_object* v_fo_2988_, lean_object* v_algMap_2989_, lean_object* v_inst_2990_, lean_object* v_vk_2991_, lean_object* v_columnOpenings_2992_, lean_object* v_publicValues_2993_, lean_object* v_lSkip_2994_, lean_object* v_rs_2995_, lean_object* v_lambda_2996_, lean_object* v_betaLogup_2997_, lean_object* v_eqSharpNs_2998_, lean_object* v___f_2999_, lean_object* v_eqNs_3000_, lean_object* v_eq3bPerTrace_3001_, lean_object* v_state_3002_, lean_object* v_entry_3003_){
_start:
{
lean_object* v_fst_3004_; lean_object* v_snd_3005_; lean_object* v___x_3007_; uint8_t v_isShared_3008_; uint8_t v_isSharedCheck_3081_; 
v_fst_3004_ = lean_ctor_get(v_state_3002_, 0);
v_snd_3005_ = lean_ctor_get(v_state_3002_, 1);
v_isSharedCheck_3081_ = !lean_is_exclusive(v_state_3002_);
if (v_isSharedCheck_3081_ == 0)
{
v___x_3007_ = v_state_3002_;
v_isShared_3008_ = v_isSharedCheck_3081_;
goto v_resetjp_3006_;
}
else
{
lean_inc(v_snd_3005_);
lean_inc(v_fst_3004_);
lean_dec(v_state_3002_);
v___x_3007_ = lean_box(0);
v_isShared_3008_ = v_isSharedCheck_3081_;
goto v_resetjp_3006_;
}
v_resetjp_3006_:
{
lean_object* v_fst_3009_; lean_object* v_snd_3010_; lean_object* v___x_3012_; uint8_t v_isShared_3013_; uint8_t v_isSharedCheck_3080_; 
v_fst_3009_ = lean_ctor_get(v_entry_3003_, 0);
v_snd_3010_ = lean_ctor_get(v_entry_3003_, 1);
v_isSharedCheck_3080_ = !lean_is_exclusive(v_entry_3003_);
if (v_isSharedCheck_3080_ == 0)
{
v___x_3012_ = v_entry_3003_;
v_isShared_3013_ = v_isSharedCheck_3080_;
goto v_resetjp_3011_;
}
else
{
lean_inc(v_snd_3010_);
lean_inc(v_fst_3009_);
lean_dec(v_entry_3003_);
v___x_3012_ = lean_box(0);
v_isShared_3013_ = v_isSharedCheck_3080_;
goto v_resetjp_3011_;
}
v_resetjp_3011_:
{
lean_object* v___x_3014_; 
lean_inc(v_fst_3009_);
v___x_3014_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_requireBatchElem___redArg(v_nPerTrace_2987_, v_fst_3009_);
if (lean_obj_tag(v___x_3014_) == 0)
{
lean_object* v___x_3015_; 
lean_dec_ref_known(v___x_3014_, 1);
lean_del_object(v___x_3012_);
lean_dec(v_snd_3010_);
lean_dec(v_fst_3009_);
lean_del_object(v___x_3007_);
lean_dec(v_snd_3005_);
lean_dec(v_fst_3004_);
lean_dec(v___f_2999_);
lean_dec(v_betaLogup_2997_);
lean_dec(v_lambda_2996_);
lean_dec(v_rs_2995_);
lean_dec(v_lSkip_2994_);
lean_dec(v_inst_2990_);
lean_dec(v_algMap_2989_);
lean_dec_ref(v_fo_2988_);
v___x_3015_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___closed__0));
return v___x_3015_;
}
else
{
lean_object* v_a_3016_; lean_object* v___y_3018_; lean_object* v___y_3019_; lean_object* v___y_3020_; lean_object* v___y_3063_; lean_object* v___y_3064_; lean_object* v___y_3065_; lean_object* v___y_3071_; lean_object* v___x_3077_; 
v_a_3016_ = lean_ctor_get(v___x_3014_, 0);
lean_inc(v_a_3016_);
lean_dec_ref_known(v___x_3014_, 1);
lean_inc(v_fst_3009_);
v___x_3077_ = l_List_get_x3fInternal___redArg(v_eq3bPerTrace_3001_, v_fst_3009_);
if (lean_obj_tag(v___x_3077_) == 0)
{
lean_object* v___x_3078_; 
v___x_3078_ = lean_box(0);
v___y_3071_ = v___x_3078_;
goto v___jp_3070_;
}
else
{
lean_object* v_val_3079_; 
v_val_3079_ = lean_ctor_get(v___x_3077_, 0);
lean_inc(v_val_3079_);
lean_dec_ref_known(v___x_3077_, 1);
v___y_3071_ = v_val_3079_;
goto v___jp_3070_;
}
v___jp_3017_:
{
lean_object* v___x_3021_; 
v___x_3021_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg(v_fo_2988_, v_algMap_2989_, v_inst_2990_, v_vk_2991_, v_columnOpenings_2992_, v_publicValues_2993_, v_lSkip_2994_, v_rs_2995_, v_lambda_2996_, v_betaLogup_2997_, v_fst_3009_, v_snd_3010_, v_a_3016_, v___y_3019_, v___y_3018_, v___y_3020_);
lean_dec(v_a_3016_);
if (lean_obj_tag(v___x_3021_) == 0)
{
lean_object* v_a_3022_; lean_object* v___x_3024_; uint8_t v_isShared_3025_; uint8_t v_isSharedCheck_3029_; 
lean_del_object(v___x_3012_);
lean_del_object(v___x_3007_);
lean_dec(v_snd_3005_);
lean_dec(v_fst_3004_);
v_a_3022_ = lean_ctor_get(v___x_3021_, 0);
v_isSharedCheck_3029_ = !lean_is_exclusive(v___x_3021_);
if (v_isSharedCheck_3029_ == 0)
{
v___x_3024_ = v___x_3021_;
v_isShared_3025_ = v_isSharedCheck_3029_;
goto v_resetjp_3023_;
}
else
{
lean_inc(v_a_3022_);
lean_dec(v___x_3021_);
v___x_3024_ = lean_box(0);
v_isShared_3025_ = v_isSharedCheck_3029_;
goto v_resetjp_3023_;
}
v_resetjp_3023_:
{
lean_object* v___x_3027_; 
if (v_isShared_3025_ == 0)
{
v___x_3027_ = v___x_3024_;
goto v_reusejp_3026_;
}
else
{
lean_object* v_reuseFailAlloc_3028_; 
v_reuseFailAlloc_3028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3028_, 0, v_a_3022_);
v___x_3027_ = v_reuseFailAlloc_3028_;
goto v_reusejp_3026_;
}
v_reusejp_3026_:
{
return v___x_3027_;
}
}
}
else
{
lean_object* v_a_3030_; lean_object* v___x_3032_; uint8_t v_isShared_3033_; uint8_t v_isSharedCheck_3061_; 
v_a_3030_ = lean_ctor_get(v___x_3021_, 0);
v_isSharedCheck_3061_ = !lean_is_exclusive(v___x_3021_);
if (v_isSharedCheck_3061_ == 0)
{
v___x_3032_ = v___x_3021_;
v_isShared_3033_ = v_isSharedCheck_3061_;
goto v_resetjp_3031_;
}
else
{
lean_inc(v_a_3030_);
lean_dec(v___x_3021_);
v___x_3032_ = lean_box(0);
v_isShared_3033_ = v_isSharedCheck_3061_;
goto v_resetjp_3031_;
}
v_resetjp_3031_:
{
lean_object* v_snd_3034_; lean_object* v_fst_3035_; lean_object* v___x_3037_; uint8_t v_isShared_3038_; uint8_t v_isSharedCheck_3060_; 
v_snd_3034_ = lean_ctor_get(v_a_3030_, 1);
v_fst_3035_ = lean_ctor_get(v_a_3030_, 0);
v_isSharedCheck_3060_ = !lean_is_exclusive(v_a_3030_);
if (v_isSharedCheck_3060_ == 0)
{
v___x_3037_ = v_a_3030_;
v_isShared_3038_ = v_isSharedCheck_3060_;
goto v_resetjp_3036_;
}
else
{
lean_inc(v_snd_3034_);
lean_inc(v_fst_3035_);
lean_dec(v_a_3030_);
v___x_3037_ = lean_box(0);
v_isShared_3038_ = v_isSharedCheck_3060_;
goto v_resetjp_3036_;
}
v_resetjp_3036_:
{
lean_object* v_fst_3039_; lean_object* v_snd_3040_; lean_object* v___x_3042_; uint8_t v_isShared_3043_; uint8_t v_isSharedCheck_3059_; 
v_fst_3039_ = lean_ctor_get(v_snd_3034_, 0);
v_snd_3040_ = lean_ctor_get(v_snd_3034_, 1);
v_isSharedCheck_3059_ = !lean_is_exclusive(v_snd_3034_);
if (v_isSharedCheck_3059_ == 0)
{
v___x_3042_ = v_snd_3034_;
v_isShared_3043_ = v_isSharedCheck_3059_;
goto v_resetjp_3041_;
}
else
{
lean_inc(v_snd_3040_);
lean_inc(v_fst_3039_);
lean_dec(v_snd_3034_);
v___x_3042_ = lean_box(0);
v_isShared_3043_ = v_isSharedCheck_3059_;
goto v_resetjp_3041_;
}
v_resetjp_3041_:
{
lean_object* v___x_3045_; 
if (v_isShared_3038_ == 0)
{
lean_ctor_set_tag(v___x_3037_, 1);
lean_ctor_set(v___x_3037_, 1, v_fst_3004_);
v___x_3045_ = v___x_3037_;
goto v_reusejp_3044_;
}
else
{
lean_object* v_reuseFailAlloc_3058_; 
v_reuseFailAlloc_3058_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3058_, 0, v_fst_3035_);
lean_ctor_set(v_reuseFailAlloc_3058_, 1, v_fst_3004_);
v___x_3045_ = v_reuseFailAlloc_3058_;
goto v_reusejp_3044_;
}
v_reusejp_3044_:
{
lean_object* v___x_3047_; 
if (v_isShared_3013_ == 0)
{
lean_ctor_set_tag(v___x_3012_, 1);
lean_ctor_set(v___x_3012_, 1, v___x_3045_);
lean_ctor_set(v___x_3012_, 0, v_fst_3039_);
v___x_3047_ = v___x_3012_;
goto v_reusejp_3046_;
}
else
{
lean_object* v_reuseFailAlloc_3057_; 
v_reuseFailAlloc_3057_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3057_, 0, v_fst_3039_);
lean_ctor_set(v_reuseFailAlloc_3057_, 1, v___x_3045_);
v___x_3047_ = v_reuseFailAlloc_3057_;
goto v_reusejp_3046_;
}
v_reusejp_3046_:
{
lean_object* v___x_3049_; 
if (v_isShared_3008_ == 0)
{
lean_ctor_set_tag(v___x_3007_, 1);
lean_ctor_set(v___x_3007_, 0, v_snd_3040_);
v___x_3049_ = v___x_3007_;
goto v_reusejp_3048_;
}
else
{
lean_object* v_reuseFailAlloc_3056_; 
v_reuseFailAlloc_3056_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3056_, 0, v_snd_3040_);
lean_ctor_set(v_reuseFailAlloc_3056_, 1, v_snd_3005_);
v___x_3049_ = v_reuseFailAlloc_3056_;
goto v_reusejp_3048_;
}
v_reusejp_3048_:
{
lean_object* v___x_3051_; 
if (v_isShared_3043_ == 0)
{
lean_ctor_set(v___x_3042_, 1, v___x_3049_);
lean_ctor_set(v___x_3042_, 0, v___x_3047_);
v___x_3051_ = v___x_3042_;
goto v_reusejp_3050_;
}
else
{
lean_object* v_reuseFailAlloc_3055_; 
v_reuseFailAlloc_3055_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3055_, 0, v___x_3047_);
lean_ctor_set(v_reuseFailAlloc_3055_, 1, v___x_3049_);
v___x_3051_ = v_reuseFailAlloc_3055_;
goto v_reusejp_3050_;
}
v_reusejp_3050_:
{
lean_object* v___x_3053_; 
if (v_isShared_3033_ == 0)
{
lean_ctor_set(v___x_3032_, 0, v___x_3051_);
v___x_3053_ = v___x_3032_;
goto v_reusejp_3052_;
}
else
{
lean_object* v_reuseFailAlloc_3054_; 
v_reuseFailAlloc_3054_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3054_, 0, v___x_3051_);
v___x_3053_ = v_reuseFailAlloc_3054_;
goto v_reusejp_3052_;
}
v_reusejp_3052_:
{
return v___x_3053_;
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
v___jp_3062_:
{
lean_object* v___x_3066_; 
v___x_3066_ = l_List_get_x3fInternal___redArg(v_eqSharpNs_2998_, v___y_3063_);
if (lean_obj_tag(v___x_3066_) == 0)
{
lean_object* v___x_3067_; lean_object* v___x_3068_; 
v___x_3067_ = lean_box(0);
v___x_3068_ = lean_apply_1(v___f_2999_, v___x_3067_);
v___y_3018_ = v___y_3065_;
v___y_3019_ = v___y_3064_;
v___y_3020_ = v___x_3068_;
goto v___jp_3017_;
}
else
{
lean_object* v_val_3069_; 
lean_dec(v___f_2999_);
v_val_3069_ = lean_ctor_get(v___x_3066_, 0);
lean_inc(v_val_3069_);
lean_dec_ref_known(v___x_3066_, 1);
v___y_3018_ = v___y_3065_;
v___y_3019_ = v___y_3064_;
v___y_3020_ = v_val_3069_;
goto v___jp_3017_;
}
}
v___jp_3070_:
{
lean_object* v___x_3072_; lean_object* v___x_3073_; 
v___x_3072_ = l_Int_toNat(v_a_3016_);
lean_inc(v___x_3072_);
v___x_3073_ = l_List_get_x3fInternal___redArg(v_eqNs_3000_, v___x_3072_);
if (lean_obj_tag(v___x_3073_) == 0)
{
lean_object* v___x_3074_; lean_object* v___x_3075_; 
v___x_3074_ = lean_box(0);
lean_inc(v___f_2999_);
v___x_3075_ = lean_apply_1(v___f_2999_, v___x_3074_);
v___y_3063_ = v___x_3072_;
v___y_3064_ = v___y_3071_;
v___y_3065_ = v___x_3075_;
goto v___jp_3062_;
}
else
{
lean_object* v_val_3076_; 
v_val_3076_ = lean_ctor_get(v___x_3073_, 0);
lean_inc(v_val_3076_);
lean_dec_ref_known(v___x_3073_, 1);
v___y_3063_ = v___x_3072_;
v___y_3064_ = v___y_3071_;
v___y_3065_ = v_val_3076_;
goto v___jp_3062_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___boxed(lean_object** _args){
lean_object* v_nPerTrace_3082_ = _args[0];
lean_object* v_fo_3083_ = _args[1];
lean_object* v_algMap_3084_ = _args[2];
lean_object* v_inst_3085_ = _args[3];
lean_object* v_vk_3086_ = _args[4];
lean_object* v_columnOpenings_3087_ = _args[5];
lean_object* v_publicValues_3088_ = _args[6];
lean_object* v_lSkip_3089_ = _args[7];
lean_object* v_rs_3090_ = _args[8];
lean_object* v_lambda_3091_ = _args[9];
lean_object* v_betaLogup_3092_ = _args[10];
lean_object* v_eqSharpNs_3093_ = _args[11];
lean_object* v___f_3094_ = _args[12];
lean_object* v_eqNs_3095_ = _args[13];
lean_object* v_eq3bPerTrace_3096_ = _args[14];
lean_object* v_state_3097_ = _args[15];
lean_object* v_entry_3098_ = _args[16];
_start:
{
lean_object* v_res_3099_; 
v_res_3099_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1(v_nPerTrace_3082_, v_fo_3083_, v_algMap_3084_, v_inst_3085_, v_vk_3086_, v_columnOpenings_3087_, v_publicValues_3088_, v_lSkip_3089_, v_rs_3090_, v_lambda_3091_, v_betaLogup_3092_, v_eqSharpNs_3093_, v___f_3094_, v_eqNs_3095_, v_eq3bPerTrace_3096_, v_state_3097_, v_entry_3098_);
lean_dec(v_eq3bPerTrace_3096_);
lean_dec(v_eqNs_3095_);
lean_dec(v_eqSharpNs_3093_);
lean_dec(v_publicValues_3088_);
lean_dec(v_columnOpenings_3087_);
lean_dec_ref(v_vk_3086_);
lean_dec(v_nPerTrace_3082_);
return v_res_3099_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(lean_object* v_fo_3102_, lean_object* v_algMap_3103_, lean_object* v_inst_3104_, lean_object* v_vk_3105_, lean_object* v_columnOpenings_3106_, lean_object* v_publicValues_3107_, lean_object* v_traceIdToAirId_3108_, lean_object* v_nPerTrace_3109_, lean_object* v_lSkip_3110_, lean_object* v_rs_3111_, lean_object* v_lambda_3112_, lean_object* v_mu_3113_, lean_object* v_betaLogup_3114_, lean_object* v_eq3bPerTrace_3115_, lean_object* v_eqNs_3116_, lean_object* v_eqSharpNs_3117_){
_start:
{
lean_object* v___f_3118_; lean_object* v___f_3119_; lean_object* v___x_3120_; lean_object* v___x_3121_; lean_object* v___x_3122_; lean_object* v_traces_3123_; lean_object* v___x_3124_; lean_object* v___x_3125_; 
lean_inc_ref_n(v_fo_3102_, 2);
v___f_3118_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3118_, 0, v_fo_3102_);
v___f_3119_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___lam__1___boxed), 17, 15);
lean_closure_set(v___f_3119_, 0, v_nPerTrace_3109_);
lean_closure_set(v___f_3119_, 1, v_fo_3102_);
lean_closure_set(v___f_3119_, 2, v_algMap_3103_);
lean_closure_set(v___f_3119_, 3, v_inst_3104_);
lean_closure_set(v___f_3119_, 4, v_vk_3105_);
lean_closure_set(v___f_3119_, 5, v_columnOpenings_3106_);
lean_closure_set(v___f_3119_, 6, v_publicValues_3107_);
lean_closure_set(v___f_3119_, 7, v_lSkip_3110_);
lean_closure_set(v___f_3119_, 8, v_rs_3111_);
lean_closure_set(v___f_3119_, 9, v_lambda_3112_);
lean_closure_set(v___f_3119_, 10, v_betaLogup_3114_);
lean_closure_set(v___f_3119_, 11, v_eqSharpNs_3117_);
lean_closure_set(v___f_3119_, 12, v___f_3118_);
lean_closure_set(v___f_3119_, 13, v_eqNs_3116_);
lean_closure_set(v___f_3119_, 14, v_eq3bPerTrace_3115_);
v___x_3120_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__9));
v___x_3121_ = l_List_lengthTR___redArg(v_traceIdToAirId_3108_);
v___x_3122_ = l_List_range(v___x_3121_);
v_traces_3123_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_3122_, v_traceIdToAirId_3108_);
v___x_3124_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg___closed__0));
v___x_3125_ = l_List_foldlM___redArg(v___x_3120_, v___f_3119_, v___x_3124_, v_traces_3123_);
if (lean_obj_tag(v___x_3125_) == 0)
{
lean_object* v_a_3126_; lean_object* v___x_3128_; uint8_t v_isShared_3129_; uint8_t v_isSharedCheck_3133_; 
lean_dec(v_mu_3113_);
lean_dec_ref(v_fo_3102_);
v_a_3126_ = lean_ctor_get(v___x_3125_, 0);
v_isSharedCheck_3133_ = !lean_is_exclusive(v___x_3125_);
if (v_isSharedCheck_3133_ == 0)
{
v___x_3128_ = v___x_3125_;
v_isShared_3129_ = v_isSharedCheck_3133_;
goto v_resetjp_3127_;
}
else
{
lean_inc(v_a_3126_);
lean_dec(v___x_3125_);
v___x_3128_ = lean_box(0);
v_isShared_3129_ = v_isSharedCheck_3133_;
goto v_resetjp_3127_;
}
v_resetjp_3127_:
{
lean_object* v___x_3131_; 
if (v_isShared_3129_ == 0)
{
v___x_3131_ = v___x_3128_;
goto v_reusejp_3130_;
}
else
{
lean_object* v_reuseFailAlloc_3132_; 
v_reuseFailAlloc_3132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3132_, 0, v_a_3126_);
v___x_3131_ = v_reuseFailAlloc_3132_;
goto v_reusejp_3130_;
}
v_reusejp_3130_:
{
return v___x_3131_;
}
}
}
else
{
lean_object* v_a_3134_; lean_object* v___x_3136_; uint8_t v_isShared_3137_; uint8_t v_isSharedCheck_3156_; 
v_a_3134_ = lean_ctor_get(v___x_3125_, 0);
v_isSharedCheck_3156_ = !lean_is_exclusive(v___x_3125_);
if (v_isSharedCheck_3156_ == 0)
{
v___x_3136_ = v___x_3125_;
v_isShared_3137_ = v_isSharedCheck_3156_;
goto v_resetjp_3135_;
}
else
{
lean_inc(v_a_3134_);
lean_dec(v___x_3125_);
v___x_3136_ = lean_box(0);
v_isShared_3137_ = v_isSharedCheck_3156_;
goto v_resetjp_3135_;
}
v_resetjp_3135_:
{
lean_object* v_toRingOps_3138_; lean_object* v_toSemiringOps_3139_; lean_object* v_fst_3140_; lean_object* v_snd_3141_; lean_object* v_zero_3142_; lean_object* v_add_3143_; lean_object* v_mul_3144_; lean_object* v___f_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v___x_3151_; lean_object* v___x_3152_; lean_object* v___x_3154_; 
v_toRingOps_3138_ = lean_ctor_get(v_fo_3102_, 0);
v_toSemiringOps_3139_ = lean_ctor_get(v_toRingOps_3138_, 0);
v_fst_3140_ = lean_ctor_get(v_a_3134_, 0);
lean_inc(v_fst_3140_);
v_snd_3141_ = lean_ctor_get(v_a_3134_, 1);
lean_inc(v_snd_3141_);
lean_dec(v_a_3134_);
v_zero_3142_ = lean_ctor_get(v_toSemiringOps_3139_, 0);
lean_inc(v_zero_3142_);
v_add_3143_ = lean_ctor_get(v_toSemiringOps_3139_, 3);
v_mul_3144_ = lean_ctor_get(v_toSemiringOps_3139_, 4);
lean_inc(v_add_3143_);
lean_inc(v_mul_3144_);
v___f_3145_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateOneTrace___redArg___lam__2), 4, 2);
lean_closure_set(v___f_3145_, 0, v_mul_3144_);
lean_closure_set(v___f_3145_, 1, v_add_3143_);
v___x_3146_ = l_List_reverse___redArg(v_fst_3140_);
v___x_3147_ = l_List_reverse___redArg(v_snd_3141_);
v___x_3148_ = l_List_appendTR___redArg(v___x_3146_, v___x_3147_);
v___x_3149_ = l_List_lengthTR___redArg(v___x_3148_);
v___x_3150_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_3102_, v_mu_3113_, v___x_3149_);
v___x_3151_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_3148_, v___x_3150_);
v___x_3152_ = l_List_foldl___redArg(v___f_3145_, v_zero_3142_, v___x_3151_);
if (v_isShared_3137_ == 0)
{
lean_ctor_set(v___x_3136_, 0, v___x_3152_);
v___x_3154_ = v___x_3136_;
goto v_reusejp_3153_;
}
else
{
lean_object* v_reuseFailAlloc_3155_; 
v_reuseFailAlloc_3155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3155_, 0, v___x_3152_);
v___x_3154_ = v_reuseFailAlloc_3155_;
goto v_reusejp_3153_;
}
v_reusejp_3153_:
{
return v___x_3154_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings(lean_object* v_F_3157_, lean_object* v_EF_3158_, lean_object* v_Digest_3159_, lean_object* v_fo_3160_, lean_object* v_algMap_3161_, lean_object* v_inst_3162_, lean_object* v_vk_3163_, lean_object* v_columnOpenings_3164_, lean_object* v_publicValues_3165_, lean_object* v_traceIdToAirId_3166_, lean_object* v_nPerTrace_3167_, lean_object* v_lSkip_3168_, lean_object* v_rs_3169_, lean_object* v_lambda_3170_, lean_object* v_mu_3171_, lean_object* v_betaLogup_3172_, lean_object* v_eq3bPerTrace_3173_, lean_object* v_eqNs_3174_, lean_object* v_eqSharpNs_3175_){
_start:
{
lean_object* v___x_3176_; 
v___x_3176_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(v_fo_3160_, v_algMap_3161_, v_inst_3162_, v_vk_3163_, v_columnOpenings_3164_, v_publicValues_3165_, v_traceIdToAirId_3166_, v_nPerTrace_3167_, v_lSkip_3168_, v_rs_3169_, v_lambda_3170_, v_mu_3171_, v_betaLogup_3172_, v_eq3bPerTrace_3173_, v_eqNs_3174_, v_eqSharpNs_3175_);
return v___x_3176_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___boxed(lean_object** _args){
lean_object* v_F_3177_ = _args[0];
lean_object* v_EF_3178_ = _args[1];
lean_object* v_Digest_3179_ = _args[2];
lean_object* v_fo_3180_ = _args[3];
lean_object* v_algMap_3181_ = _args[4];
lean_object* v_inst_3182_ = _args[5];
lean_object* v_vk_3183_ = _args[6];
lean_object* v_columnOpenings_3184_ = _args[7];
lean_object* v_publicValues_3185_ = _args[8];
lean_object* v_traceIdToAirId_3186_ = _args[9];
lean_object* v_nPerTrace_3187_ = _args[10];
lean_object* v_lSkip_3188_ = _args[11];
lean_object* v_rs_3189_ = _args[12];
lean_object* v_lambda_3190_ = _args[13];
lean_object* v_mu_3191_ = _args[14];
lean_object* v_betaLogup_3192_ = _args[15];
lean_object* v_eq3bPerTrace_3193_ = _args[16];
lean_object* v_eqNs_3194_ = _args[17];
lean_object* v_eqSharpNs_3195_ = _args[18];
_start:
{
lean_object* v_res_3196_; 
v_res_3196_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings(v_F_3177_, v_EF_3178_, v_Digest_3179_, v_fo_3180_, v_algMap_3181_, v_inst_3182_, v_vk_3183_, v_columnOpenings_3184_, v_publicValues_3185_, v_traceIdToAirId_3186_, v_nPerTrace_3187_, v_lSkip_3188_, v_rs_3189_, v_lambda_3190_, v_mu_3191_, v_betaLogup_3192_, v_eq3bPerTrace_3193_, v_eqNs_3194_, v_eqSharpNs_3195_);
return v_res_3196_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0(lean_object* v_vk_3197_, lean_object* v_airId_3198_){
_start:
{
lean_object* v___x_3199_; 
v___x_3199_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_airVKeyAt___redArg(v_vk_3197_, v_airId_3198_);
if (lean_obj_tag(v___x_3199_) == 0)
{
lean_object* v___x_3200_; 
v___x_3200_ = lean_unsigned_to_nat(0u);
return v___x_3200_;
}
else
{
lean_object* v_val_3201_; lean_object* v_symbolicConstraints_3202_; lean_object* v_interactions_3203_; lean_object* v___x_3204_; 
v_val_3201_ = lean_ctor_get(v___x_3199_, 0);
lean_inc(v_val_3201_);
lean_dec_ref_known(v___x_3199_, 1);
v_symbolicConstraints_3202_ = lean_ctor_get(v_val_3201_, 2);
lean_inc_ref(v_symbolicConstraints_3202_);
lean_dec(v_val_3201_);
v_interactions_3203_ = lean_ctor_get(v_symbolicConstraints_3202_, 1);
lean_inc(v_interactions_3203_);
lean_dec_ref(v_symbolicConstraints_3202_);
v___x_3204_ = l_List_lengthTR___redArg(v_interactions_3203_);
lean_dec(v_interactions_3203_);
return v___x_3204_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0___boxed(lean_object* v_vk_3205_, lean_object* v_airId_3206_){
_start:
{
lean_object* v_res_3207_; 
v_res_3207_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0(v_vk_3205_, v_airId_3206_);
lean_dec_ref(v_vk_3205_);
return v_res_3207_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__1(lean_object* v_inst_3208_, lean_object* v_inst_3209_, lean_object* v_fo_3210_, lean_object* v_acc_3211_, lean_object* v_roundPoly_3212_, lean_object* v___y_3213_){
_start:
{
lean_object* v_fst_3214_; lean_object* v_snd_3215_; lean_object* v___x_3217_; uint8_t v_isShared_3218_; uint8_t v_isSharedCheck_3260_; 
v_fst_3214_ = lean_ctor_get(v_acc_3211_, 0);
v_snd_3215_ = lean_ctor_get(v_acc_3211_, 1);
v_isSharedCheck_3260_ = !lean_is_exclusive(v_acc_3211_);
if (v_isSharedCheck_3260_ == 0)
{
v___x_3217_ = v_acc_3211_;
v_isShared_3218_ = v_isSharedCheck_3260_;
goto v_resetjp_3216_;
}
else
{
lean_inc(v_snd_3215_);
lean_inc(v_fst_3214_);
lean_dec(v_acc_3211_);
v___x_3217_ = lean_box(0);
v_isShared_3218_ = v_isSharedCheck_3260_;
goto v_resetjp_3216_;
}
v_resetjp_3216_:
{
lean_object* v___x_3219_; 
lean_inc(v_roundPoly_3212_);
lean_inc_ref(v_inst_3209_);
lean_inc_ref(v_inst_3208_);
v___x_3219_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(v_inst_3208_, v_inst_3209_, v_roundPoly_3212_, v___y_3213_);
if (lean_obj_tag(v___x_3219_) == 0)
{
lean_object* v_a_3220_; lean_object* v___x_3222_; uint8_t v_isShared_3223_; uint8_t v_isSharedCheck_3227_; 
lean_del_object(v___x_3217_);
lean_dec(v_snd_3215_);
lean_dec(v_fst_3214_);
lean_dec(v_roundPoly_3212_);
lean_dec_ref(v_fo_3210_);
lean_dec_ref(v_inst_3209_);
lean_dec_ref(v_inst_3208_);
v_a_3220_ = lean_ctor_get(v___x_3219_, 0);
v_isSharedCheck_3227_ = !lean_is_exclusive(v___x_3219_);
if (v_isSharedCheck_3227_ == 0)
{
v___x_3222_ = v___x_3219_;
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
else
{
lean_inc(v_a_3220_);
lean_dec(v___x_3219_);
v___x_3222_ = lean_box(0);
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
v_resetjp_3221_:
{
lean_object* v___x_3225_; 
if (v_isShared_3223_ == 0)
{
v___x_3225_ = v___x_3222_;
goto v_reusejp_3224_;
}
else
{
lean_object* v_reuseFailAlloc_3226_; 
v_reuseFailAlloc_3226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3226_, 0, v_a_3220_);
v___x_3225_ = v_reuseFailAlloc_3226_;
goto v_reusejp_3224_;
}
v_reusejp_3224_:
{
return v___x_3225_;
}
}
}
else
{
lean_object* v_a_3228_; lean_object* v_snd_3229_; lean_object* v___x_3231_; uint8_t v_isShared_3232_; uint8_t v_isSharedCheck_3258_; 
v_a_3228_ = lean_ctor_get(v___x_3219_, 0);
lean_inc(v_a_3228_);
lean_dec_ref_known(v___x_3219_, 1);
v_snd_3229_ = lean_ctor_get(v_a_3228_, 1);
v_isSharedCheck_3258_ = !lean_is_exclusive(v_a_3228_);
if (v_isSharedCheck_3258_ == 0)
{
lean_object* v_unused_3259_; 
v_unused_3259_ = lean_ctor_get(v_a_3228_, 0);
lean_dec(v_unused_3259_);
v___x_3231_ = v_a_3228_;
v_isShared_3232_ = v_isSharedCheck_3258_;
goto v_resetjp_3230_;
}
else
{
lean_inc(v_snd_3229_);
lean_dec(v_a_3228_);
v___x_3231_ = lean_box(0);
v_isShared_3232_ = v_isSharedCheck_3258_;
goto v_resetjp_3230_;
}
v_resetjp_3230_:
{
lean_object* v___x_3233_; lean_object* v_a_3234_; lean_object* v___x_3236_; uint8_t v_isShared_3237_; uint8_t v_isSharedCheck_3257_; 
v___x_3233_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3208_, v_inst_3209_, v_snd_3229_);
v_a_3234_ = lean_ctor_get(v___x_3233_, 0);
v_isSharedCheck_3257_ = !lean_is_exclusive(v___x_3233_);
if (v_isSharedCheck_3257_ == 0)
{
v___x_3236_ = v___x_3233_;
v_isShared_3237_ = v_isSharedCheck_3257_;
goto v_resetjp_3235_;
}
else
{
lean_inc(v_a_3234_);
lean_dec(v___x_3233_);
v___x_3236_ = lean_box(0);
v_isShared_3237_ = v_isSharedCheck_3257_;
goto v_resetjp_3235_;
}
v_resetjp_3235_:
{
lean_object* v_fst_3238_; lean_object* v_snd_3239_; lean_object* v___x_3241_; uint8_t v_isShared_3242_; uint8_t v_isSharedCheck_3256_; 
v_fst_3238_ = lean_ctor_get(v_a_3234_, 0);
v_snd_3239_ = lean_ctor_get(v_a_3234_, 1);
v_isSharedCheck_3256_ = !lean_is_exclusive(v_a_3234_);
if (v_isSharedCheck_3256_ == 0)
{
v___x_3241_ = v_a_3234_;
v_isShared_3242_ = v_isSharedCheck_3256_;
goto v_resetjp_3240_;
}
else
{
lean_inc(v_snd_3239_);
lean_inc(v_fst_3238_);
lean_dec(v_a_3234_);
v___x_3241_ = lean_box(0);
v_isShared_3242_ = v_isSharedCheck_3256_;
goto v_resetjp_3240_;
}
v_resetjp_3240_:
{
lean_object* v___x_3243_; lean_object* v___x_3245_; 
lean_inc(v_fst_3238_);
v___x_3243_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyMultilinearSumcheckRound___redArg(v_fo_3210_, v_fst_3214_, v_roundPoly_3212_, v_fst_3238_);
if (v_isShared_3218_ == 0)
{
lean_ctor_set_tag(v___x_3217_, 1);
lean_ctor_set(v___x_3217_, 0, v_fst_3238_);
v___x_3245_ = v___x_3217_;
goto v_reusejp_3244_;
}
else
{
lean_object* v_reuseFailAlloc_3255_; 
v_reuseFailAlloc_3255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3255_, 0, v_fst_3238_);
lean_ctor_set(v_reuseFailAlloc_3255_, 1, v_snd_3215_);
v___x_3245_ = v_reuseFailAlloc_3255_;
goto v_reusejp_3244_;
}
v_reusejp_3244_:
{
lean_object* v___x_3247_; 
if (v_isShared_3242_ == 0)
{
lean_ctor_set(v___x_3241_, 1, v___x_3245_);
lean_ctor_set(v___x_3241_, 0, v___x_3243_);
v___x_3247_ = v___x_3241_;
goto v_reusejp_3246_;
}
else
{
lean_object* v_reuseFailAlloc_3254_; 
v_reuseFailAlloc_3254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3254_, 0, v___x_3243_);
lean_ctor_set(v_reuseFailAlloc_3254_, 1, v___x_3245_);
v___x_3247_ = v_reuseFailAlloc_3254_;
goto v_reusejp_3246_;
}
v_reusejp_3246_:
{
lean_object* v___x_3249_; 
if (v_isShared_3232_ == 0)
{
lean_ctor_set(v___x_3231_, 1, v_snd_3239_);
lean_ctor_set(v___x_3231_, 0, v___x_3247_);
v___x_3249_ = v___x_3231_;
goto v_reusejp_3248_;
}
else
{
lean_object* v_reuseFailAlloc_3253_; 
v_reuseFailAlloc_3253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3253_, 0, v___x_3247_);
lean_ctor_set(v_reuseFailAlloc_3253_, 1, v_snd_3239_);
v___x_3249_ = v_reuseFailAlloc_3253_;
goto v_reusejp_3248_;
}
v_reusejp_3248_:
{
lean_object* v___x_3251_; 
if (v_isShared_3237_ == 0)
{
lean_ctor_set(v___x_3236_, 0, v___x_3249_);
v___x_3251_ = v___x_3236_;
goto v_reusejp_3250_;
}
else
{
lean_object* v_reuseFailAlloc_3252_; 
v_reuseFailAlloc_3252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3252_, 0, v___x_3249_);
v___x_3251_ = v_reuseFailAlloc_3252_;
goto v_reusejp_3250_;
}
v_reusejp_3250_:
{
return v___x_3251_;
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__2(lean_object* v_inst_3261_, lean_object* v_inst_3262_, lean_object* v_x_3263_, lean_object* v_entry_3264_, lean_object* v___y_3265_){
_start:
{
lean_object* v_fst_3266_; lean_object* v_snd_3267_; lean_object* v___x_3268_; lean_object* v_a_3269_; lean_object* v_snd_3270_; lean_object* v___x_3271_; 
v_fst_3266_ = lean_ctor_get(v_entry_3264_, 0);
lean_inc(v_fst_3266_);
v_snd_3267_ = lean_ctor_get(v_entry_3264_, 1);
lean_inc(v_snd_3267_);
lean_dec_ref(v_entry_3264_);
lean_inc_ref(v_inst_3262_);
lean_inc_ref(v_inst_3261_);
v___x_3268_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_3261_, v_inst_3262_, v_fst_3266_, v___y_3265_);
v_a_3269_ = lean_ctor_get(v___x_3268_, 0);
lean_inc(v_a_3269_);
lean_dec_ref(v___x_3268_);
v_snd_3270_ = lean_ctor_get(v_a_3269_, 1);
lean_inc(v_snd_3270_);
lean_dec(v_a_3269_);
v___x_3271_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_3261_, v_inst_3262_, v_snd_3267_, v_snd_3270_);
return v___x_3271_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3(lean_object* v_inst_3272_, lean_object* v_inst_3273_, lean_object* v_acc_3274_, lean_object* v_x_3275_, lean_object* v___y_3276_){
_start:
{
lean_object* v_fst_3277_; lean_object* v___x_3279_; uint8_t v_isShared_3280_; uint8_t v_isSharedCheck_3304_; 
v_fst_3277_ = lean_ctor_get(v_acc_3274_, 0);
v_isSharedCheck_3304_ = !lean_is_exclusive(v_acc_3274_);
if (v_isSharedCheck_3304_ == 0)
{
lean_object* v_unused_3305_; 
v_unused_3305_ = lean_ctor_get(v_acc_3274_, 1);
lean_dec(v_unused_3305_);
v___x_3279_ = v_acc_3274_;
v_isShared_3280_ = v_isSharedCheck_3304_;
goto v_resetjp_3278_;
}
else
{
lean_inc(v_fst_3277_);
lean_dec(v_acc_3274_);
v___x_3279_ = lean_box(0);
v_isShared_3280_ = v_isSharedCheck_3304_;
goto v_resetjp_3278_;
}
v_resetjp_3278_:
{
lean_object* v___x_3281_; lean_object* v_a_3282_; lean_object* v___x_3284_; uint8_t v_isShared_3285_; uint8_t v_isSharedCheck_3303_; 
v___x_3281_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3272_, v_inst_3273_, v___y_3276_);
v_a_3282_ = lean_ctor_get(v___x_3281_, 0);
v_isSharedCheck_3303_ = !lean_is_exclusive(v___x_3281_);
if (v_isSharedCheck_3303_ == 0)
{
v___x_3284_ = v___x_3281_;
v_isShared_3285_ = v_isSharedCheck_3303_;
goto v_resetjp_3283_;
}
else
{
lean_inc(v_a_3282_);
lean_dec(v___x_3281_);
v___x_3284_ = lean_box(0);
v_isShared_3285_ = v_isSharedCheck_3303_;
goto v_resetjp_3283_;
}
v_resetjp_3283_:
{
lean_object* v_fst_3286_; lean_object* v_snd_3287_; lean_object* v___x_3289_; uint8_t v_isShared_3290_; uint8_t v_isSharedCheck_3302_; 
v_fst_3286_ = lean_ctor_get(v_a_3282_, 0);
v_snd_3287_ = lean_ctor_get(v_a_3282_, 1);
v_isSharedCheck_3302_ = !lean_is_exclusive(v_a_3282_);
if (v_isSharedCheck_3302_ == 0)
{
v___x_3289_ = v_a_3282_;
v_isShared_3290_ = v_isSharedCheck_3302_;
goto v_resetjp_3288_;
}
else
{
lean_inc(v_snd_3287_);
lean_inc(v_fst_3286_);
lean_dec(v_a_3282_);
v___x_3289_ = lean_box(0);
v_isShared_3290_ = v_isSharedCheck_3302_;
goto v_resetjp_3288_;
}
v_resetjp_3288_:
{
lean_object* v___x_3291_; lean_object* v___x_3292_; lean_object* v___x_3294_; 
v___x_3291_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3291_, 0, v_fst_3286_);
lean_ctor_set(v___x_3291_, 1, v_fst_3277_);
v___x_3292_ = lean_box(0);
if (v_isShared_3290_ == 0)
{
lean_ctor_set(v___x_3289_, 1, v___x_3292_);
lean_ctor_set(v___x_3289_, 0, v___x_3291_);
v___x_3294_ = v___x_3289_;
goto v_reusejp_3293_;
}
else
{
lean_object* v_reuseFailAlloc_3301_; 
v_reuseFailAlloc_3301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3301_, 0, v___x_3291_);
lean_ctor_set(v_reuseFailAlloc_3301_, 1, v___x_3292_);
v___x_3294_ = v_reuseFailAlloc_3301_;
goto v_reusejp_3293_;
}
v_reusejp_3293_:
{
lean_object* v___x_3296_; 
if (v_isShared_3280_ == 0)
{
lean_ctor_set(v___x_3279_, 1, v_snd_3287_);
lean_ctor_set(v___x_3279_, 0, v___x_3294_);
v___x_3296_ = v___x_3279_;
goto v_reusejp_3295_;
}
else
{
lean_object* v_reuseFailAlloc_3300_; 
v_reuseFailAlloc_3300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3300_, 0, v___x_3294_);
lean_ctor_set(v_reuseFailAlloc_3300_, 1, v_snd_3287_);
v___x_3296_ = v_reuseFailAlloc_3300_;
goto v_reusejp_3295_;
}
v_reusejp_3295_:
{
lean_object* v___x_3298_; 
if (v_isShared_3285_ == 0)
{
lean_ctor_set(v___x_3284_, 0, v___x_3296_);
v___x_3298_ = v___x_3284_;
goto v_reusejp_3297_;
}
else
{
lean_object* v_reuseFailAlloc_3299_; 
v_reuseFailAlloc_3299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3299_, 0, v___x_3296_);
v___x_3298_ = v_reuseFailAlloc_3299_;
goto v_reusejp_3297_;
}
v_reusejp_3297_:
{
return v___x_3298_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3___boxed(lean_object* v_inst_3306_, lean_object* v_inst_3307_, lean_object* v_acc_3308_, lean_object* v_x_3309_, lean_object* v___y_3310_){
_start:
{
lean_object* v_res_3311_; 
v_res_3311_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3(v_inst_3306_, v_inst_3307_, v_acc_3308_, v_x_3309_, v___y_3310_);
lean_dec(v_x_3309_);
return v_res_3311_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(lean_object* v_inst_3318_, lean_object* v_inst_3319_, lean_object* v_inst_3320_, lean_object* v_inst_3321_, lean_object* v_inst_3322_, lean_object* v_fo_3323_, lean_object* v_algMap_3324_, lean_object* v_vk_3325_, lean_object* v_gkrProof_3326_, lean_object* v_batchProof_3327_, lean_object* v_traceVdata_3328_, lean_object* v_publicValues_3329_, lean_object* v_traceIdToAirId_3330_, lean_object* v_nPerTrace_3331_, lean_object* v_a_3332_){
_start:
{
lean_object* v_inner_3333_; lean_object* v_params_3334_; lean_object* v___x_3335_; lean_object* v_logupPowWitness_3336_; lean_object* v_q0Claim_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; 
v_inner_3333_ = lean_ctor_get(v_vk_3325_, 0);
v_params_3334_ = lean_ctor_get(v_inner_3333_, 0);
v___x_3335_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeOpeningPairsM___redArg___closed__19));
v_logupPowWitness_3336_ = lean_ctor_get(v_gkrProof_3326_, 0);
v_q0Claim_3337_ = lean_ctor_get(v_gkrProof_3326_, 1);
v___x_3338_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_SystemParams_logupPowBits(v_params_3334_);
lean_inc(v_logupPowWitness_3336_);
lean_inc_ref(v_inst_3320_);
v___x_3339_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observePowWitness___redArg(v_inst_3320_, v_inst_3321_, v___x_3338_, v_logupPowWitness_3336_, v_a_3332_);
lean_dec(v___x_3338_);
if (lean_obj_tag(v___x_3339_) == 0)
{
lean_object* v_a_3340_; lean_object* v___x_3342_; uint8_t v_isShared_3343_; uint8_t v_isSharedCheck_3347_; 
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_batchProof_3327_);
lean_dec_ref(v_gkrProof_3326_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3340_ = lean_ctor_get(v___x_3339_, 0);
v_isSharedCheck_3347_ = !lean_is_exclusive(v___x_3339_);
if (v_isSharedCheck_3347_ == 0)
{
v___x_3342_ = v___x_3339_;
v_isShared_3343_ = v_isSharedCheck_3347_;
goto v_resetjp_3341_;
}
else
{
lean_inc(v_a_3340_);
lean_dec(v___x_3339_);
v___x_3342_ = lean_box(0);
v_isShared_3343_ = v_isSharedCheck_3347_;
goto v_resetjp_3341_;
}
v_resetjp_3341_:
{
lean_object* v___x_3345_; 
if (v_isShared_3343_ == 0)
{
v___x_3345_ = v___x_3342_;
goto v_reusejp_3344_;
}
else
{
lean_object* v_reuseFailAlloc_3346_; 
v_reuseFailAlloc_3346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3346_, 0, v_a_3340_);
v___x_3345_ = v_reuseFailAlloc_3346_;
goto v_reusejp_3344_;
}
v_reusejp_3344_:
{
return v___x_3345_;
}
}
}
else
{
lean_object* v_a_3348_; lean_object* v_snd_3349_; lean_object* v___x_3350_; lean_object* v_a_3351_; lean_object* v_fst_3352_; lean_object* v_snd_3353_; lean_object* v___x_3354_; lean_object* v_a_3355_; lean_object* v_fst_3356_; lean_object* v_snd_3357_; lean_object* v_lSkip_3358_; lean_object* v___f_3359_; lean_object* v___f_3360_; lean_object* v___f_3361_; lean_object* v___f_3362_; lean_object* v_nMax_3363_; lean_object* v___x_3364_; lean_object* v___x_3365_; lean_object* v___y_3367_; lean_object* v___y_3368_; lean_object* v___y_3369_; lean_object* v___y_3370_; lean_object* v___y_3371_; lean_object* v_fst_3529_; lean_object* v_fst_3530_; lean_object* v_snd_3531_; lean_object* v___y_3532_; lean_object* v___x_3534_; uint8_t v___x_3535_; 
v_a_3348_ = lean_ctor_get(v___x_3339_, 0);
lean_inc(v_a_3348_);
lean_dec_ref_known(v___x_3339_, 1);
v_snd_3349_ = lean_ctor_get(v_a_3348_, 1);
lean_inc(v_snd_3349_);
lean_dec(v_a_3348_);
lean_inc_ref_n(v_inst_3322_, 5);
lean_inc_ref_n(v_inst_3320_, 5);
v___x_3350_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3320_, v_inst_3322_, v_snd_3349_);
v_a_3351_ = lean_ctor_get(v___x_3350_, 0);
lean_inc(v_a_3351_);
lean_dec_ref(v___x_3350_);
v_fst_3352_ = lean_ctor_get(v_a_3351_, 0);
lean_inc(v_fst_3352_);
v_snd_3353_ = lean_ctor_get(v_a_3351_, 1);
lean_inc(v_snd_3353_);
lean_dec(v_a_3351_);
v___x_3354_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3320_, v_inst_3322_, v_snd_3353_);
v_a_3355_ = lean_ctor_get(v___x_3354_, 0);
lean_inc(v_a_3355_);
lean_dec_ref(v___x_3354_);
v_fst_3356_ = lean_ctor_get(v_a_3355_, 0);
lean_inc(v_fst_3356_);
v_snd_3357_ = lean_ctor_get(v_a_3355_, 1);
lean_inc(v_snd_3357_);
lean_dec(v_a_3355_);
v_lSkip_3358_ = lean_ctor_get(v_params_3334_, 0);
lean_inc(v_lSkip_3358_);
lean_inc_ref(v_vk_3325_);
v___f_3359_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3359_, 0, v_vk_3325_);
lean_inc_ref(v_fo_3323_);
v___f_3360_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__1), 6, 3);
lean_closure_set(v___f_3360_, 0, v_inst_3320_);
lean_closure_set(v___f_3360_, 1, v_inst_3322_);
lean_closure_set(v___f_3360_, 2, v_fo_3323_);
v___f_3361_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__2), 5, 2);
lean_closure_set(v___f_3361_, 0, v_inst_3320_);
lean_closure_set(v___f_3361_, 1, v_inst_3322_);
v___f_3362_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___lam__3___boxed), 5, 2);
lean_closure_set(v___f_3362_, 0, v_inst_3320_);
lean_closure_set(v___f_3362_, 1, v_inst_3322_);
lean_inc_n(v_traceIdToAirId_3330_, 2);
v_nMax_3363_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeNMax___redArg(v_lSkip_3358_, v_traceVdata_3328_, v_traceIdToAirId_3330_);
lean_inc_ref(v_params_3334_);
v___x_3364_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_totalInteractions___redArg(v_params_3334_, v_vk_3325_, v_traceVdata_3328_, v_traceIdToAirId_3330_);
v___x_3365_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape_calculateNLogup(v_lSkip_3358_, v___x_3364_);
v___x_3534_ = lean_unsigned_to_nat(0u);
v___x_3535_ = lean_nat_dec_lt(v___x_3534_, v___x_3364_);
lean_dec(v___x_3364_);
if (v___x_3535_ == 0)
{
lean_object* v_toRingOps_3536_; lean_object* v_toSemiringOps_3537_; lean_object* v_zero_3538_; lean_object* v_one_3539_; lean_object* v___x_3540_; uint8_t v___x_3541_; 
lean_inc(v_q0Claim_3337_);
lean_dec_ref(v_gkrProof_3326_);
v_toRingOps_3536_ = lean_ctor_get(v_fo_3323_, 0);
v_toSemiringOps_3537_ = lean_ctor_get(v_toRingOps_3536_, 0);
v_zero_3538_ = lean_ctor_get(v_toSemiringOps_3537_, 0);
v_one_3539_ = lean_ctor_get(v_toSemiringOps_3537_, 1);
lean_inc_ref(v_inst_3318_);
lean_inc(v_one_3539_);
v___x_3540_ = lean_apply_2(v_inst_3318_, v_q0Claim_3337_, v_one_3539_);
v___x_3541_ = lean_unbox(v___x_3540_);
if (v___x_3541_ == 0)
{
lean_object* v___x_3542_; 
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3362_);
lean_dec_ref(v___f_3361_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_snd_3357_);
lean_dec(v_fst_3356_);
lean_dec(v_fst_3352_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_batchProof_3327_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v___x_3542_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3542_;
}
else
{
lean_object* v___x_3543_; 
v___x_3543_ = lean_box(0);
lean_inc(v_fst_3352_);
lean_inc(v_zero_3538_);
v_fst_3529_ = v_zero_3538_;
v_fst_3530_ = v_fst_3352_;
v_snd_3531_ = v___x_3543_;
v___y_3532_ = v_snd_3357_;
goto v___jp_3528_;
}
}
else
{
lean_object* v___x_3544_; lean_object* v___x_3545_; 
v___x_3544_ = lean_nat_add(v_lSkip_3358_, v___x_3365_);
lean_inc_ref(v_fo_3323_);
lean_inc_ref(v_inst_3322_);
lean_inc_ref(v_inst_3320_);
lean_inc_ref(v_inst_3318_);
v___x_3545_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyGkrM___redArg(v_inst_3318_, v_inst_3320_, v_inst_3322_, v_fo_3323_, v_gkrProof_3326_, v___x_3544_, v_snd_3357_);
lean_dec(v___x_3544_);
if (lean_obj_tag(v___x_3545_) == 0)
{
lean_object* v_a_3546_; lean_object* v___x_3548_; uint8_t v_isShared_3549_; uint8_t v_isSharedCheck_3553_; 
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3362_);
lean_dec_ref(v___f_3361_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_fst_3352_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_batchProof_3327_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3546_ = lean_ctor_get(v___x_3545_, 0);
v_isSharedCheck_3553_ = !lean_is_exclusive(v___x_3545_);
if (v_isSharedCheck_3553_ == 0)
{
v___x_3548_ = v___x_3545_;
v_isShared_3549_ = v_isSharedCheck_3553_;
goto v_resetjp_3547_;
}
else
{
lean_inc(v_a_3546_);
lean_dec(v___x_3545_);
v___x_3548_ = lean_box(0);
v_isShared_3549_ = v_isSharedCheck_3553_;
goto v_resetjp_3547_;
}
v_resetjp_3547_:
{
lean_object* v___x_3551_; 
if (v_isShared_3549_ == 0)
{
v___x_3551_ = v___x_3548_;
goto v_reusejp_3550_;
}
else
{
lean_object* v_reuseFailAlloc_3552_; 
v_reuseFailAlloc_3552_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3552_, 0, v_a_3546_);
v___x_3551_ = v_reuseFailAlloc_3552_;
goto v_reusejp_3550_;
}
v_reusejp_3550_:
{
return v___x_3551_;
}
}
}
else
{
lean_object* v_a_3554_; lean_object* v_fst_3555_; lean_object* v_snd_3556_; lean_object* v_numeratorClaim_3557_; lean_object* v_denominatorClaim_3558_; lean_object* v_xi_3559_; 
v_a_3554_ = lean_ctor_get(v___x_3545_, 0);
lean_inc(v_a_3554_);
lean_dec_ref_known(v___x_3545_, 1);
v_fst_3555_ = lean_ctor_get(v_a_3554_, 0);
lean_inc(v_fst_3555_);
v_snd_3556_ = lean_ctor_get(v_a_3554_, 1);
lean_inc(v_snd_3556_);
lean_dec(v_a_3554_);
v_numeratorClaim_3557_ = lean_ctor_get(v_fst_3555_, 0);
lean_inc(v_numeratorClaim_3557_);
v_denominatorClaim_3558_ = lean_ctor_get(v_fst_3555_, 1);
lean_inc(v_denominatorClaim_3558_);
v_xi_3559_ = lean_ctor_get(v_fst_3555_, 2);
lean_inc(v_xi_3559_);
lean_dec(v_fst_3555_);
v_fst_3529_ = v_numeratorClaim_3557_;
v_fst_3530_ = v_denominatorClaim_3558_;
v_snd_3531_ = v_xi_3559_;
v___y_3532_ = v_snd_3556_;
goto v___jp_3528_;
}
}
v___jp_3366_:
{
lean_object* v___x_3372_; lean_object* v___x_3373_; lean_object* v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; lean_object* v___x_3377_; lean_object* v___x_3378_; lean_object* v___x_9425__overap_3379_; lean_object* v___x_3380_; 
v___x_3372_ = lean_nat_add(v_lSkip_3358_, v___y_3371_);
lean_dec(v___y_3371_);
v___x_3373_ = l_List_lengthTR___redArg(v___y_3368_);
v___x_3374_ = lean_nat_sub(v___x_3372_, v___x_3373_);
lean_dec(v___x_3373_);
lean_dec(v___x_3372_);
v___x_3375_ = lean_box(0);
v___x_3376_ = lean_box(0);
v___x_3377_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__0));
v___x_3378_ = l_List_range(v___x_3374_);
v___x_9425__overap_3379_ = l_List_foldlM___redArg(v___x_3335_, v___f_3362_, v___x_3377_, v___x_3378_);
v___x_3380_ = lean_apply_1(v___x_9425__overap_3379_, v___y_3367_);
if (lean_obj_tag(v___x_3380_) == 0)
{
lean_object* v_a_3381_; lean_object* v___x_3383_; uint8_t v_isShared_3384_; uint8_t v_isSharedCheck_3388_; 
lean_dec(v___y_3370_);
lean_dec(v___y_3369_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3361_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_fst_3352_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_batchProof_3327_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3381_ = lean_ctor_get(v___x_3380_, 0);
v_isSharedCheck_3388_ = !lean_is_exclusive(v___x_3380_);
if (v_isSharedCheck_3388_ == 0)
{
v___x_3383_ = v___x_3380_;
v_isShared_3384_ = v_isSharedCheck_3388_;
goto v_resetjp_3382_;
}
else
{
lean_inc(v_a_3381_);
lean_dec(v___x_3380_);
v___x_3383_ = lean_box(0);
v_isShared_3384_ = v_isSharedCheck_3388_;
goto v_resetjp_3382_;
}
v_resetjp_3382_:
{
lean_object* v___x_3386_; 
if (v_isShared_3384_ == 0)
{
v___x_3386_ = v___x_3383_;
goto v_reusejp_3385_;
}
else
{
lean_object* v_reuseFailAlloc_3387_; 
v_reuseFailAlloc_3387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3387_, 0, v_a_3381_);
v___x_3386_ = v_reuseFailAlloc_3387_;
goto v_reusejp_3385_;
}
v_reusejp_3385_:
{
return v___x_3386_;
}
}
}
else
{
lean_object* v_a_3389_; lean_object* v_fst_3390_; lean_object* v_snd_3391_; lean_object* v_fst_3392_; lean_object* v___x_3393_; lean_object* v_a_3394_; lean_object* v_fst_3395_; lean_object* v_snd_3396_; lean_object* v_numeratorTermPerAir_3397_; lean_object* v_denominatorTermPerAir_3398_; lean_object* v_univariateRoundCoeffs_3399_; lean_object* v_sumcheckRoundPolys_3400_; lean_object* v_columnOpenings_3401_; lean_object* v___x_3402_; lean_object* v___x_3403_; uint8_t v___x_3404_; 
v_a_3389_ = lean_ctor_get(v___x_3380_, 0);
lean_inc(v_a_3389_);
lean_dec_ref_known(v___x_3380_, 1);
v_fst_3390_ = lean_ctor_get(v_a_3389_, 0);
lean_inc(v_fst_3390_);
v_snd_3391_ = lean_ctor_get(v_a_3389_, 1);
lean_inc(v_snd_3391_);
lean_dec(v_a_3389_);
v_fst_3392_ = lean_ctor_get(v_fst_3390_, 0);
lean_inc(v_fst_3392_);
lean_dec(v_fst_3390_);
lean_inc_ref(v_inst_3322_);
lean_inc_ref(v_inst_3320_);
v___x_3393_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3320_, v_inst_3322_, v_snd_3391_);
v_a_3394_ = lean_ctor_get(v___x_3393_, 0);
lean_inc(v_a_3394_);
lean_dec_ref(v___x_3393_);
v_fst_3395_ = lean_ctor_get(v_a_3394_, 0);
lean_inc(v_fst_3395_);
v_snd_3396_ = lean_ctor_get(v_a_3394_, 1);
lean_inc(v_snd_3396_);
lean_dec(v_a_3394_);
v_numeratorTermPerAir_3397_ = lean_ctor_get(v_batchProof_3327_, 0);
lean_inc(v_numeratorTermPerAir_3397_);
v_denominatorTermPerAir_3398_ = lean_ctor_get(v_batchProof_3327_, 1);
lean_inc(v_denominatorTermPerAir_3398_);
v_univariateRoundCoeffs_3399_ = lean_ctor_get(v_batchProof_3327_, 2);
lean_inc(v_univariateRoundCoeffs_3399_);
v_sumcheckRoundPolys_3400_ = lean_ctor_get(v_batchProof_3327_, 3);
lean_inc(v_sumcheckRoundPolys_3400_);
v_columnOpenings_3401_ = lean_ctor_get(v_batchProof_3327_, 4);
lean_inc(v_columnOpenings_3401_);
lean_dec_ref(v_batchProof_3327_);
v___x_3402_ = l_List_lengthTR___redArg(v_numeratorTermPerAir_3397_);
v___x_3403_ = l_List_lengthTR___redArg(v_denominatorTermPerAir_3398_);
v___x_3404_ = lean_nat_dec_eq(v___x_3402_, v___x_3403_);
lean_dec(v___x_3403_);
lean_dec(v___x_3402_);
if (v___x_3404_ == 0)
{
lean_object* v___x_3405_; 
lean_dec(v_columnOpenings_3401_);
lean_dec(v_sumcheckRoundPolys_3400_);
lean_dec(v_univariateRoundCoeffs_3399_);
lean_dec(v_denominatorTermPerAir_3398_);
lean_dec(v_numeratorTermPerAir_3397_);
lean_dec(v_snd_3396_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3370_);
lean_dec(v___y_3369_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3361_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_fst_3352_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v___x_3405_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3405_;
}
else
{
lean_object* v___x_3406_; lean_object* v___x_9704__overap_3407_; lean_object* v___x_3408_; 
lean_inc(v_denominatorTermPerAir_3398_);
lean_inc(v_numeratorTermPerAir_3397_);
v___x_3406_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_numeratorTermPerAir_3397_, v_denominatorTermPerAir_3398_);
v___x_9704__overap_3407_ = l_List_foldlM___redArg(v___x_3335_, v___f_3361_, v___x_3376_, v___x_3406_);
v___x_3408_ = lean_apply_1(v___x_9704__overap_3407_, v_snd_3396_);
if (lean_obj_tag(v___x_3408_) == 0)
{
lean_object* v_a_3409_; lean_object* v___x_3411_; uint8_t v_isShared_3412_; uint8_t v_isSharedCheck_3416_; 
lean_dec(v_columnOpenings_3401_);
lean_dec(v_sumcheckRoundPolys_3400_);
lean_dec(v_univariateRoundCoeffs_3399_);
lean_dec(v_denominatorTermPerAir_3398_);
lean_dec(v_numeratorTermPerAir_3397_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3370_);
lean_dec(v___y_3369_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_fst_3352_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3409_ = lean_ctor_get(v___x_3408_, 0);
v_isSharedCheck_3416_ = !lean_is_exclusive(v___x_3408_);
if (v_isSharedCheck_3416_ == 0)
{
v___x_3411_ = v___x_3408_;
v_isShared_3412_ = v_isSharedCheck_3416_;
goto v_resetjp_3410_;
}
else
{
lean_inc(v_a_3409_);
lean_dec(v___x_3408_);
v___x_3411_ = lean_box(0);
v_isShared_3412_ = v_isSharedCheck_3416_;
goto v_resetjp_3410_;
}
v_resetjp_3410_:
{
lean_object* v___x_3414_; 
if (v_isShared_3412_ == 0)
{
v___x_3414_ = v___x_3411_;
goto v_reusejp_3413_;
}
else
{
lean_object* v_reuseFailAlloc_3415_; 
v_reuseFailAlloc_3415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3415_, 0, v_a_3409_);
v___x_3414_ = v_reuseFailAlloc_3415_;
goto v_reusejp_3413_;
}
v_reusejp_3413_:
{
return v___x_3414_;
}
}
}
else
{
lean_object* v_a_3417_; lean_object* v_snd_3418_; lean_object* v___x_3419_; 
v_a_3417_ = lean_ctor_get(v___x_3408_, 0);
lean_inc(v_a_3417_);
lean_dec_ref_known(v___x_3408_, 1);
v_snd_3418_ = lean_ctor_get(v_a_3417_, 1);
lean_inc(v_snd_3418_);
lean_dec(v_a_3417_);
lean_inc(v_denominatorTermPerAir_3398_);
lean_inc(v_numeratorTermPerAir_3397_);
lean_inc_ref(v_fo_3323_);
lean_inc_ref(v_inst_3318_);
v___x_3419_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_foldSumClaimsAgainstXi___redArg(v_inst_3318_, v_fo_3323_, v_numeratorTermPerAir_3397_, v_denominatorTermPerAir_3398_, v_fst_3352_, v___y_3370_, v___y_3369_);
if (lean_obj_tag(v___x_3419_) == 0)
{
lean_object* v___x_3420_; 
lean_dec_ref_known(v___x_3419_, 1);
lean_dec(v_snd_3418_);
lean_dec(v_columnOpenings_3401_);
lean_dec(v_sumcheckRoundPolys_3400_);
lean_dec(v_univariateRoundCoeffs_3399_);
lean_dec(v_denominatorTermPerAir_3398_);
lean_dec(v_numeratorTermPerAir_3397_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v___x_3420_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3420_;
}
else
{
lean_object* v___x_3421_; lean_object* v_a_3422_; lean_object* v_fst_3423_; lean_object* v_snd_3424_; lean_object* v___x_3425_; 
lean_dec_ref_known(v___x_3419_, 1);
lean_inc_ref_n(v_inst_3322_, 2);
lean_inc_ref_n(v_inst_3320_, 2);
v___x_3421_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3320_, v_inst_3322_, v_snd_3418_);
v_a_3422_ = lean_ctor_get(v___x_3421_, 0);
lean_inc(v_a_3422_);
lean_dec_ref(v___x_3421_);
v_fst_3423_ = lean_ctor_get(v_a_3422_, 0);
lean_inc(v_fst_3423_);
v_snd_3424_ = lean_ctor_get(v_a_3422_, 1);
lean_inc(v_snd_3424_);
lean_dec(v_a_3422_);
lean_inc(v_univariateRoundCoeffs_3399_);
v___x_3425_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(v_inst_3320_, v_inst_3322_, v_univariateRoundCoeffs_3399_, v_snd_3424_);
if (lean_obj_tag(v___x_3425_) == 0)
{
lean_object* v_a_3426_; lean_object* v___x_3428_; uint8_t v_isShared_3429_; uint8_t v_isSharedCheck_3433_; 
lean_dec(v_fst_3423_);
lean_dec(v_columnOpenings_3401_);
lean_dec(v_sumcheckRoundPolys_3400_);
lean_dec(v_univariateRoundCoeffs_3399_);
lean_dec(v_denominatorTermPerAir_3398_);
lean_dec(v_numeratorTermPerAir_3397_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3426_ = lean_ctor_get(v___x_3425_, 0);
v_isSharedCheck_3433_ = !lean_is_exclusive(v___x_3425_);
if (v_isSharedCheck_3433_ == 0)
{
v___x_3428_ = v___x_3425_;
v_isShared_3429_ = v_isSharedCheck_3433_;
goto v_resetjp_3427_;
}
else
{
lean_inc(v_a_3426_);
lean_dec(v___x_3425_);
v___x_3428_ = lean_box(0);
v_isShared_3429_ = v_isSharedCheck_3433_;
goto v_resetjp_3427_;
}
v_resetjp_3427_:
{
lean_object* v___x_3431_; 
if (v_isShared_3429_ == 0)
{
v___x_3431_ = v___x_3428_;
goto v_reusejp_3430_;
}
else
{
lean_object* v_reuseFailAlloc_3432_; 
v_reuseFailAlloc_3432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3432_, 0, v_a_3426_);
v___x_3431_ = v_reuseFailAlloc_3432_;
goto v_reusejp_3430_;
}
v_reusejp_3430_:
{
return v___x_3431_;
}
}
}
else
{
lean_object* v_a_3434_; lean_object* v_snd_3435_; lean_object* v___x_3437_; uint8_t v_isShared_3438_; uint8_t v_isSharedCheck_3526_; 
v_a_3434_ = lean_ctor_get(v___x_3425_, 0);
lean_inc(v_a_3434_);
lean_dec_ref_known(v___x_3425_, 1);
v_snd_3435_ = lean_ctor_get(v_a_3434_, 1);
v_isSharedCheck_3526_ = !lean_is_exclusive(v_a_3434_);
if (v_isSharedCheck_3526_ == 0)
{
lean_object* v_unused_3527_; 
v_unused_3527_ = lean_ctor_get(v_a_3434_, 0);
lean_dec(v_unused_3527_);
v___x_3437_ = v_a_3434_;
v_isShared_3438_ = v_isSharedCheck_3526_;
goto v_resetjp_3436_;
}
else
{
lean_inc(v_snd_3435_);
lean_dec(v_a_3434_);
v___x_3437_ = lean_box(0);
v_isShared_3438_ = v_isSharedCheck_3526_;
goto v_resetjp_3436_;
}
v_resetjp_3436_:
{
lean_object* v___x_3439_; lean_object* v_a_3440_; lean_object* v_fst_3441_; lean_object* v_snd_3442_; lean_object* v___x_3444_; uint8_t v_isShared_3445_; uint8_t v_isSharedCheck_3525_; 
lean_inc_ref(v_inst_3322_);
lean_inc_ref(v_inst_3320_);
v___x_3439_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_3320_, v_inst_3322_, v_snd_3435_);
v_a_3440_ = lean_ctor_get(v___x_3439_, 0);
lean_inc(v_a_3440_);
lean_dec_ref(v___x_3439_);
v_fst_3441_ = lean_ctor_get(v_a_3440_, 0);
v_snd_3442_ = lean_ctor_get(v_a_3440_, 1);
v_isSharedCheck_3525_ = !lean_is_exclusive(v_a_3440_);
if (v_isSharedCheck_3525_ == 0)
{
v___x_3444_ = v_a_3440_;
v_isShared_3445_ = v_isSharedCheck_3525_;
goto v_resetjp_3443_;
}
else
{
lean_inc(v_snd_3442_);
lean_inc(v_fst_3441_);
lean_dec(v_a_3440_);
v___x_3444_ = lean_box(0);
v_isShared_3445_ = v_isSharedCheck_3525_;
goto v_resetjp_3443_;
}
v_resetjp_3443_:
{
lean_object* v___x_3446_; lean_object* v___x_3447_; 
lean_inc(v_fst_3423_);
lean_inc_ref_n(v_fo_3323_, 2);
v___x_3446_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_muRlcSumClaim___redArg(v_fo_3323_, v_numeratorTermPerAir_3397_, v_denominatorTermPerAir_3398_, v_fst_3423_);
lean_inc(v_fst_3441_);
lean_inc_ref(v_inst_3318_);
v___x_3447_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyUnivariateRound___redArg(v_inst_3318_, v_fo_3323_, v_univariateRoundCoeffs_3399_, v___x_3446_, v_lSkip_3358_, v_fst_3441_);
if (lean_obj_tag(v___x_3447_) == 0)
{
lean_object* v___x_3448_; 
lean_dec_ref_known(v___x_3447_, 1);
lean_del_object(v___x_3444_);
lean_dec(v_snd_3442_);
lean_dec(v_fst_3441_);
lean_del_object(v___x_3437_);
lean_dec(v_fst_3423_);
lean_dec(v_columnOpenings_3401_);
lean_dec(v_sumcheckRoundPolys_3400_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v___x_3448_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3448_;
}
else
{
lean_object* v_a_3449_; lean_object* v___x_3450_; uint8_t v___x_3451_; 
v_a_3449_ = lean_ctor_get(v___x_3447_, 0);
lean_inc(v_a_3449_);
lean_dec_ref_known(v___x_3447_, 1);
v___x_3450_ = l_List_lengthTR___redArg(v_sumcheckRoundPolys_3400_);
v___x_3451_ = lean_nat_dec_eq(v___x_3450_, v_nMax_3363_);
lean_dec(v___x_3450_);
if (v___x_3451_ == 0)
{
lean_object* v___x_3452_; 
lean_dec(v_a_3449_);
lean_del_object(v___x_3444_);
lean_dec(v_snd_3442_);
lean_dec(v_fst_3441_);
lean_del_object(v___x_3437_);
lean_dec(v_fst_3423_);
lean_dec(v_columnOpenings_3401_);
lean_dec(v_sumcheckRoundPolys_3400_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3360_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v___x_3452_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3452_;
}
else
{
lean_object* v___x_3454_; 
if (v_isShared_3438_ == 0)
{
lean_ctor_set_tag(v___x_3437_, 1);
lean_ctor_set(v___x_3437_, 1, v___x_3375_);
lean_ctor_set(v___x_3437_, 0, v_fst_3441_);
v___x_3454_ = v___x_3437_;
goto v_reusejp_3453_;
}
else
{
lean_object* v_reuseFailAlloc_3524_; 
v_reuseFailAlloc_3524_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3524_, 0, v_fst_3441_);
lean_ctor_set(v_reuseFailAlloc_3524_, 1, v___x_3375_);
v___x_3454_ = v_reuseFailAlloc_3524_;
goto v_reusejp_3453_;
}
v_reusejp_3453_:
{
lean_object* v___x_3456_; 
if (v_isShared_3445_ == 0)
{
lean_ctor_set(v___x_3444_, 1, v___x_3454_);
lean_ctor_set(v___x_3444_, 0, v_a_3449_);
v___x_3456_ = v___x_3444_;
goto v_reusejp_3455_;
}
else
{
lean_object* v_reuseFailAlloc_3523_; 
v_reuseFailAlloc_3523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3523_, 0, v_a_3449_);
lean_ctor_set(v_reuseFailAlloc_3523_, 1, v___x_3454_);
v___x_3456_ = v_reuseFailAlloc_3523_;
goto v_reusejp_3455_;
}
v_reusejp_3455_:
{
lean_object* v___x_9788__overap_3457_; lean_object* v___x_3458_; 
v___x_9788__overap_3457_ = l_List_foldlM___redArg(v___x_3335_, v___f_3360_, v___x_3456_, v_sumcheckRoundPolys_3400_);
v___x_3458_ = lean_apply_1(v___x_9788__overap_3457_, v_snd_3442_);
if (lean_obj_tag(v___x_3458_) == 0)
{
lean_object* v_a_3459_; lean_object* v___x_3461_; uint8_t v_isShared_3462_; uint8_t v_isSharedCheck_3466_; 
lean_dec(v_fst_3423_);
lean_dec(v_columnOpenings_3401_);
lean_dec(v_fst_3395_);
lean_dec(v_fst_3392_);
lean_dec(v___y_3368_);
lean_dec(v___x_3365_);
lean_dec(v_nMax_3363_);
lean_dec_ref(v___f_3359_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec_ref(v_inst_3322_);
lean_dec_ref(v_inst_3320_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3459_ = lean_ctor_get(v___x_3458_, 0);
v_isSharedCheck_3466_ = !lean_is_exclusive(v___x_3458_);
if (v_isSharedCheck_3466_ == 0)
{
v___x_3461_ = v___x_3458_;
v_isShared_3462_ = v_isSharedCheck_3466_;
goto v_resetjp_3460_;
}
else
{
lean_inc(v_a_3459_);
lean_dec(v___x_3458_);
v___x_3461_ = lean_box(0);
v_isShared_3462_ = v_isSharedCheck_3466_;
goto v_resetjp_3460_;
}
v_resetjp_3460_:
{
lean_object* v___x_3464_; 
if (v_isShared_3462_ == 0)
{
v___x_3464_ = v___x_3461_;
goto v_reusejp_3463_;
}
else
{
lean_object* v_reuseFailAlloc_3465_; 
v_reuseFailAlloc_3465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3465_, 0, v_a_3459_);
v___x_3464_ = v_reuseFailAlloc_3465_;
goto v_reusejp_3463_;
}
v_reusejp_3463_:
{
return v___x_3464_;
}
}
}
else
{
lean_object* v_a_3467_; lean_object* v_fst_3468_; lean_object* v_snd_3469_; lean_object* v_fst_3470_; lean_object* v_snd_3471_; lean_object* v___x_3472_; lean_object* v___x_3473_; lean_object* v___x_3474_; lean_object* v___x_3475_; lean_object* v___x_3476_; lean_object* v___x_3477_; lean_object* v___x_3478_; lean_object* v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; lean_object* v_fst_3482_; lean_object* v_snd_3483_; lean_object* v___x_3484_; 
v_a_3467_ = lean_ctor_get(v___x_3458_, 0);
lean_inc(v_a_3467_);
lean_dec_ref_known(v___x_3458_, 1);
v_fst_3468_ = lean_ctor_get(v_a_3467_, 0);
lean_inc(v_fst_3468_);
v_snd_3469_ = lean_ctor_get(v_a_3467_, 1);
lean_inc(v_snd_3469_);
lean_dec(v_a_3467_);
v_fst_3470_ = lean_ctor_get(v_fst_3468_, 0);
lean_inc(v_fst_3470_);
v_snd_3471_ = lean_ctor_get(v_fst_3468_, 1);
lean_inc(v_snd_3471_);
lean_dec(v_fst_3468_);
v___x_3472_ = l_List_reverse___redArg(v_fst_3392_);
v___x_3473_ = l_List_appendTR___redArg(v___y_3368_, v___x_3472_);
v___x_3474_ = l_List_reverse___redArg(v_snd_3471_);
lean_inc_n(v_traceIdToAirId_3330_, 2);
v___x_3475_ = l_List_mapTR_loop___redArg(v___f_3359_, v_traceIdToAirId_3330_, v___x_3375_);
lean_inc(v_nPerTrace_3331_);
lean_inc_ref_n(v_fo_3323_, 4);
v___x_3476_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeEq3bPerTrace___redArg(v_fo_3323_, v_lSkip_3358_, v___x_3365_, v___x_3473_, v___x_3475_, v_nPerTrace_3331_);
lean_dec(v___x_3365_);
lean_inc(v_inst_3319_);
lean_inc_n(v_lSkip_3358_, 2);
v___x_3477_ = lean_apply_1(v_inst_3319_, v_lSkip_3358_);
v___x_3478_ = lean_unsigned_to_nat(2u);
v___x_3479_ = lean_nat_pow(v___x_3478_, v_lSkip_3358_);
v___x_3480_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_fieldPowers___redArg(v_fo_3323_, v___x_3477_, v___x_3479_);
lean_inc(v___x_3474_);
v___x_3481_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_computeBatchEqs___redArg(v_fo_3323_, v_lSkip_3358_, v_nMax_3363_, v___x_3473_, v___x_3474_, v___x_3480_);
v_fst_3482_ = lean_ctor_get(v___x_3481_, 0);
lean_inc(v_fst_3482_);
v_snd_3483_ = lean_ctor_get(v___x_3481_, 1);
lean_inc(v_snd_3483_);
lean_dec_ref(v___x_3481_);
lean_inc(v_columnOpenings_3401_);
lean_inc_ref(v_vk_3325_);
v___x_3484_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_observeColumnOpeningsM___redArg(v_inst_3320_, v_inst_3322_, v_fo_3323_, v_vk_3325_, v_columnOpenings_3401_, v_traceIdToAirId_3330_, v_snd_3469_);
if (lean_obj_tag(v___x_3484_) == 0)
{
lean_object* v_a_3485_; lean_object* v___x_3487_; uint8_t v_isShared_3488_; uint8_t v_isSharedCheck_3492_; 
lean_dec(v_snd_3483_);
lean_dec(v_fst_3482_);
lean_dec(v___x_3476_);
lean_dec(v___x_3474_);
lean_dec(v_fst_3470_);
lean_dec(v_fst_3423_);
lean_dec(v_columnOpenings_3401_);
lean_dec(v_fst_3395_);
lean_dec(v_lSkip_3358_);
lean_dec(v_fst_3356_);
lean_dec(v_nPerTrace_3331_);
lean_dec(v_traceIdToAirId_3330_);
lean_dec(v_publicValues_3329_);
lean_dec_ref(v_vk_3325_);
lean_dec(v_algMap_3324_);
lean_dec_ref(v_fo_3323_);
lean_dec(v_inst_3319_);
lean_dec_ref(v_inst_3318_);
v_a_3485_ = lean_ctor_get(v___x_3484_, 0);
v_isSharedCheck_3492_ = !lean_is_exclusive(v___x_3484_);
if (v_isSharedCheck_3492_ == 0)
{
v___x_3487_ = v___x_3484_;
v_isShared_3488_ = v_isSharedCheck_3492_;
goto v_resetjp_3486_;
}
else
{
lean_inc(v_a_3485_);
lean_dec(v___x_3484_);
v___x_3487_ = lean_box(0);
v_isShared_3488_ = v_isSharedCheck_3492_;
goto v_resetjp_3486_;
}
v_resetjp_3486_:
{
lean_object* v___x_3490_; 
if (v_isShared_3488_ == 0)
{
v___x_3490_ = v___x_3487_;
goto v_reusejp_3489_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v_a_3485_);
v___x_3490_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3489_;
}
v_reusejp_3489_:
{
return v___x_3490_;
}
}
}
else
{
lean_object* v_a_3493_; lean_object* v_snd_3494_; lean_object* v___x_3496_; uint8_t v_isShared_3497_; uint8_t v_isSharedCheck_3521_; 
v_a_3493_ = lean_ctor_get(v___x_3484_, 0);
lean_inc(v_a_3493_);
lean_dec_ref_known(v___x_3484_, 1);
v_snd_3494_ = lean_ctor_get(v_a_3493_, 1);
v_isSharedCheck_3521_ = !lean_is_exclusive(v_a_3493_);
if (v_isSharedCheck_3521_ == 0)
{
lean_object* v_unused_3522_; 
v_unused_3522_ = lean_ctor_get(v_a_3493_, 0);
lean_dec(v_unused_3522_);
v___x_3496_ = v_a_3493_;
v_isShared_3497_ = v_isSharedCheck_3521_;
goto v_resetjp_3495_;
}
else
{
lean_inc(v_snd_3494_);
lean_dec(v_a_3493_);
v___x_3496_ = lean_box(0);
v_isShared_3497_ = v_isSharedCheck_3521_;
goto v_resetjp_3495_;
}
v_resetjp_3495_:
{
lean_object* v___x_3498_; 
lean_inc(v___x_3474_);
v___x_3498_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_evaluateColumnOpenings___redArg(v_fo_3323_, v_algMap_3324_, v_inst_3319_, v_vk_3325_, v_columnOpenings_3401_, v_publicValues_3329_, v_traceIdToAirId_3330_, v_nPerTrace_3331_, v_lSkip_3358_, v___x_3474_, v_fst_3395_, v_fst_3423_, v_fst_3356_, v___x_3476_, v_fst_3482_, v_snd_3483_);
if (lean_obj_tag(v___x_3498_) == 0)
{
lean_object* v_a_3499_; lean_object* v___x_3501_; uint8_t v_isShared_3502_; uint8_t v_isSharedCheck_3506_; 
lean_del_object(v___x_3496_);
lean_dec(v_snd_3494_);
lean_dec(v___x_3474_);
lean_dec(v_fst_3470_);
lean_dec_ref(v_inst_3318_);
v_a_3499_ = lean_ctor_get(v___x_3498_, 0);
v_isSharedCheck_3506_ = !lean_is_exclusive(v___x_3498_);
if (v_isSharedCheck_3506_ == 0)
{
v___x_3501_ = v___x_3498_;
v_isShared_3502_ = v_isSharedCheck_3506_;
goto v_resetjp_3500_;
}
else
{
lean_inc(v_a_3499_);
lean_dec(v___x_3498_);
v___x_3501_ = lean_box(0);
v_isShared_3502_ = v_isSharedCheck_3506_;
goto v_resetjp_3500_;
}
v_resetjp_3500_:
{
lean_object* v___x_3504_; 
if (v_isShared_3502_ == 0)
{
v___x_3504_ = v___x_3501_;
goto v_reusejp_3503_;
}
else
{
lean_object* v_reuseFailAlloc_3505_; 
v_reuseFailAlloc_3505_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3505_, 0, v_a_3499_);
v___x_3504_ = v_reuseFailAlloc_3505_;
goto v_reusejp_3503_;
}
v_reusejp_3503_:
{
return v___x_3504_;
}
}
}
else
{
lean_object* v_a_3507_; lean_object* v___x_3509_; uint8_t v_isShared_3510_; uint8_t v_isSharedCheck_3520_; 
v_a_3507_ = lean_ctor_get(v___x_3498_, 0);
v_isSharedCheck_3520_ = !lean_is_exclusive(v___x_3498_);
if (v_isSharedCheck_3520_ == 0)
{
v___x_3509_ = v___x_3498_;
v_isShared_3510_ = v_isSharedCheck_3520_;
goto v_resetjp_3508_;
}
else
{
lean_inc(v_a_3507_);
lean_dec(v___x_3498_);
v___x_3509_ = lean_box(0);
v_isShared_3510_ = v_isSharedCheck_3520_;
goto v_resetjp_3508_;
}
v_resetjp_3508_:
{
lean_object* v___x_3511_; uint8_t v___x_3512_; 
v___x_3511_ = lean_apply_2(v_inst_3318_, v_fst_3470_, v_a_3507_);
v___x_3512_ = lean_unbox(v___x_3511_);
if (v___x_3512_ == 0)
{
lean_object* v___x_3513_; 
lean_del_object(v___x_3509_);
lean_del_object(v___x_3496_);
lean_dec(v_snd_3494_);
lean_dec(v___x_3474_);
v___x_3513_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___closed__1));
return v___x_3513_;
}
else
{
lean_object* v___x_3515_; 
if (v_isShared_3497_ == 0)
{
lean_ctor_set(v___x_3496_, 0, v___x_3474_);
v___x_3515_ = v___x_3496_;
goto v_reusejp_3514_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v___x_3474_);
lean_ctor_set(v_reuseFailAlloc_3519_, 1, v_snd_3494_);
v___x_3515_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3514_;
}
v_reusejp_3514_:
{
lean_object* v___x_3517_; 
if (v_isShared_3510_ == 0)
{
lean_ctor_set(v___x_3509_, 0, v___x_3515_);
v___x_3517_ = v___x_3509_;
goto v_reusejp_3516_;
}
else
{
lean_object* v_reuseFailAlloc_3518_; 
v_reuseFailAlloc_3518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3518_, 0, v___x_3515_);
v___x_3517_ = v_reuseFailAlloc_3518_;
goto v_reusejp_3516_;
}
v_reusejp_3516_:
{
return v___x_3517_;
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
v___jp_3528_:
{
uint8_t v___x_3533_; 
v___x_3533_ = lean_nat_dec_le(v_nMax_3363_, v___x_3365_);
if (v___x_3533_ == 0)
{
lean_inc(v_nMax_3363_);
v___y_3367_ = v___y_3532_;
v___y_3368_ = v_snd_3531_;
v___y_3369_ = v_fst_3530_;
v___y_3370_ = v_fst_3529_;
v___y_3371_ = v_nMax_3363_;
goto v___jp_3366_;
}
else
{
lean_inc(v___x_3365_);
v___y_3367_ = v___y_3532_;
v___y_3368_ = v_snd_3531_;
v___y_3369_ = v_fst_3530_;
v___y_3370_ = v_fst_3529_;
v___y_3371_ = v___x_3365_;
goto v___jp_3366_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg___boxed(lean_object* v_inst_3560_, lean_object* v_inst_3561_, lean_object* v_inst_3562_, lean_object* v_inst_3563_, lean_object* v_inst_3564_, lean_object* v_fo_3565_, lean_object* v_algMap_3566_, lean_object* v_vk_3567_, lean_object* v_gkrProof_3568_, lean_object* v_batchProof_3569_, lean_object* v_traceVdata_3570_, lean_object* v_publicValues_3571_, lean_object* v_traceIdToAirId_3572_, lean_object* v_nPerTrace_3573_, lean_object* v_a_3574_){
_start:
{
lean_object* v_res_3575_; 
v_res_3575_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(v_inst_3560_, v_inst_3561_, v_inst_3562_, v_inst_3563_, v_inst_3564_, v_fo_3565_, v_algMap_3566_, v_vk_3567_, v_gkrProof_3568_, v_batchProof_3569_, v_traceVdata_3570_, v_publicValues_3571_, v_traceIdToAirId_3572_, v_nPerTrace_3573_, v_a_3574_);
lean_dec(v_traceVdata_3570_);
return v_res_3575_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM(lean_object* v_F_3576_, lean_object* v_EF_3577_, lean_object* v_Digest_3578_, lean_object* v_inst_3579_, lean_object* v_inst_3580_, lean_object* v_inst_3581_, lean_object* v_inst_3582_, lean_object* v_inst_3583_, lean_object* v_fo_3584_, lean_object* v_algMap_3585_, lean_object* v_vk_3586_, lean_object* v_gkrProof_3587_, lean_object* v_batchProof_3588_, lean_object* v_traceVdata_3589_, lean_object* v_publicValues_3590_, lean_object* v_traceIdToAirId_3591_, lean_object* v_nPerTrace_3592_, lean_object* v_a_3593_){
_start:
{
lean_object* v___x_3594_; 
v___x_3594_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(v_inst_3579_, v_inst_3580_, v_inst_3581_, v_inst_3582_, v_inst_3583_, v_fo_3584_, v_algMap_3585_, v_vk_3586_, v_gkrProof_3587_, v_batchProof_3588_, v_traceVdata_3589_, v_publicValues_3590_, v_traceIdToAirId_3591_, v_nPerTrace_3592_, v_a_3593_);
return v___x_3594_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___boxed(lean_object** _args){
lean_object* v_F_3595_ = _args[0];
lean_object* v_EF_3596_ = _args[1];
lean_object* v_Digest_3597_ = _args[2];
lean_object* v_inst_3598_ = _args[3];
lean_object* v_inst_3599_ = _args[4];
lean_object* v_inst_3600_ = _args[5];
lean_object* v_inst_3601_ = _args[6];
lean_object* v_inst_3602_ = _args[7];
lean_object* v_fo_3603_ = _args[8];
lean_object* v_algMap_3604_ = _args[9];
lean_object* v_vk_3605_ = _args[10];
lean_object* v_gkrProof_3606_ = _args[11];
lean_object* v_batchProof_3607_ = _args[12];
lean_object* v_traceVdata_3608_ = _args[13];
lean_object* v_publicValues_3609_ = _args[14];
lean_object* v_traceIdToAirId_3610_ = _args[15];
lean_object* v_nPerTrace_3611_ = _args[16];
lean_object* v_a_3612_ = _args[17];
_start:
{
lean_object* v_res_3613_; 
v_res_3613_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM(v_F_3595_, v_EF_3596_, v_Digest_3597_, v_inst_3598_, v_inst_3599_, v_inst_3600_, v_inst_3601_, v_inst_3602_, v_fo_3603_, v_algMap_3604_, v_vk_3605_, v_gkrProof_3606_, v_batchProof_3607_, v_traceVdata_3608_, v_publicValues_3609_, v_traceIdToAirId_3610_, v_nPerTrace_3611_, v_a_3612_);
lean_dec(v_traceVdata_3608_);
return v_res_3613_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(lean_object* v_inst_3614_, lean_object* v_inst_3615_, lean_object* v_inst_3616_, lean_object* v_inst_3617_, lean_object* v_inst_3618_, lean_object* v_fo_3619_, lean_object* v_algMap_3620_, lean_object* v_transcript_3621_, lean_object* v_vk_3622_, lean_object* v_publicValues_3623_, lean_object* v_gkrProof_3624_, lean_object* v_batchProof_3625_, lean_object* v_traceVdata_3626_, lean_object* v_traceIdToAirId_3627_, lean_object* v_nPerTrace_3628_){
_start:
{
lean_object* v___x_3629_; 
v___x_3629_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verifyM___redArg(v_inst_3614_, v_inst_3615_, v_inst_3616_, v_inst_3617_, v_inst_3618_, v_fo_3619_, v_algMap_3620_, v_vk_3622_, v_gkrProof_3624_, v_batchProof_3625_, v_traceVdata_3626_, v_publicValues_3623_, v_traceIdToAirId_3627_, v_nPerTrace_3628_, v_transcript_3621_);
if (lean_obj_tag(v___x_3629_) == 0)
{
lean_object* v_a_3630_; lean_object* v___x_3632_; uint8_t v_isShared_3633_; uint8_t v_isSharedCheck_3637_; 
v_a_3630_ = lean_ctor_get(v___x_3629_, 0);
v_isSharedCheck_3637_ = !lean_is_exclusive(v___x_3629_);
if (v_isSharedCheck_3637_ == 0)
{
v___x_3632_ = v___x_3629_;
v_isShared_3633_ = v_isSharedCheck_3637_;
goto v_resetjp_3631_;
}
else
{
lean_inc(v_a_3630_);
lean_dec(v___x_3629_);
v___x_3632_ = lean_box(0);
v_isShared_3633_ = v_isSharedCheck_3637_;
goto v_resetjp_3631_;
}
v_resetjp_3631_:
{
lean_object* v___x_3635_; 
if (v_isShared_3633_ == 0)
{
v___x_3635_ = v___x_3632_;
goto v_reusejp_3634_;
}
else
{
lean_object* v_reuseFailAlloc_3636_; 
v_reuseFailAlloc_3636_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3636_, 0, v_a_3630_);
v___x_3635_ = v_reuseFailAlloc_3636_;
goto v_reusejp_3634_;
}
v_reusejp_3634_:
{
return v___x_3635_;
}
}
}
else
{
lean_object* v_a_3638_; lean_object* v___x_3640_; uint8_t v_isShared_3641_; uint8_t v_isSharedCheck_3654_; 
v_a_3638_ = lean_ctor_get(v___x_3629_, 0);
v_isSharedCheck_3654_ = !lean_is_exclusive(v___x_3629_);
if (v_isSharedCheck_3654_ == 0)
{
v___x_3640_ = v___x_3629_;
v_isShared_3641_ = v_isSharedCheck_3654_;
goto v_resetjp_3639_;
}
else
{
lean_inc(v_a_3638_);
lean_dec(v___x_3629_);
v___x_3640_ = lean_box(0);
v_isShared_3641_ = v_isSharedCheck_3654_;
goto v_resetjp_3639_;
}
v_resetjp_3639_:
{
lean_object* v_fst_3642_; lean_object* v_snd_3643_; lean_object* v___x_3645_; uint8_t v_isShared_3646_; uint8_t v_isSharedCheck_3653_; 
v_fst_3642_ = lean_ctor_get(v_a_3638_, 0);
v_snd_3643_ = lean_ctor_get(v_a_3638_, 1);
v_isSharedCheck_3653_ = !lean_is_exclusive(v_a_3638_);
if (v_isSharedCheck_3653_ == 0)
{
v___x_3645_ = v_a_3638_;
v_isShared_3646_ = v_isSharedCheck_3653_;
goto v_resetjp_3644_;
}
else
{
lean_inc(v_snd_3643_);
lean_inc(v_fst_3642_);
lean_dec(v_a_3638_);
v___x_3645_ = lean_box(0);
v_isShared_3646_ = v_isSharedCheck_3653_;
goto v_resetjp_3644_;
}
v_resetjp_3644_:
{
lean_object* v___x_3648_; 
if (v_isShared_3646_ == 0)
{
v___x_3648_ = v___x_3645_;
goto v_reusejp_3647_;
}
else
{
lean_object* v_reuseFailAlloc_3652_; 
v_reuseFailAlloc_3652_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3652_, 0, v_fst_3642_);
lean_ctor_set(v_reuseFailAlloc_3652_, 1, v_snd_3643_);
v___x_3648_ = v_reuseFailAlloc_3652_;
goto v_reusejp_3647_;
}
v_reusejp_3647_:
{
lean_object* v___x_3650_; 
if (v_isShared_3641_ == 0)
{
lean_ctor_set(v___x_3640_, 0, v___x_3648_);
v___x_3650_ = v___x_3640_;
goto v_reusejp_3649_;
}
else
{
lean_object* v_reuseFailAlloc_3651_; 
v_reuseFailAlloc_3651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3651_, 0, v___x_3648_);
v___x_3650_ = v_reuseFailAlloc_3651_;
goto v_reusejp_3649_;
}
v_reusejp_3649_:
{
return v___x_3650_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg___boxed(lean_object* v_inst_3655_, lean_object* v_inst_3656_, lean_object* v_inst_3657_, lean_object* v_inst_3658_, lean_object* v_inst_3659_, lean_object* v_fo_3660_, lean_object* v_algMap_3661_, lean_object* v_transcript_3662_, lean_object* v_vk_3663_, lean_object* v_publicValues_3664_, lean_object* v_gkrProof_3665_, lean_object* v_batchProof_3666_, lean_object* v_traceVdata_3667_, lean_object* v_traceIdToAirId_3668_, lean_object* v_nPerTrace_3669_){
_start:
{
lean_object* v_res_3670_; 
v_res_3670_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(v_inst_3655_, v_inst_3656_, v_inst_3657_, v_inst_3658_, v_inst_3659_, v_fo_3660_, v_algMap_3661_, v_transcript_3662_, v_vk_3663_, v_publicValues_3664_, v_gkrProof_3665_, v_batchProof_3666_, v_traceVdata_3667_, v_traceIdToAirId_3668_, v_nPerTrace_3669_);
lean_dec(v_traceVdata_3667_);
return v_res_3670_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify(lean_object* v_F_3671_, lean_object* v_EF_3672_, lean_object* v_Digest_3673_, lean_object* v_inst_3674_, lean_object* v_inst_3675_, lean_object* v_inst_3676_, lean_object* v_inst_3677_, lean_object* v_inst_3678_, lean_object* v_fo_3679_, lean_object* v_algMap_3680_, lean_object* v_transcript_3681_, lean_object* v_vk_3682_, lean_object* v_publicValues_3683_, lean_object* v_gkrProof_3684_, lean_object* v_batchProof_3685_, lean_object* v_traceVdata_3686_, lean_object* v_traceIdToAirId_3687_, lean_object* v_nPerTrace_3688_){
_start:
{
lean_object* v___x_3689_; 
v___x_3689_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___redArg(v_inst_3674_, v_inst_3675_, v_inst_3676_, v_inst_3677_, v_inst_3678_, v_fo_3679_, v_algMap_3680_, v_transcript_3681_, v_vk_3682_, v_publicValues_3683_, v_gkrProof_3684_, v_batchProof_3685_, v_traceVdata_3686_, v_traceIdToAirId_3687_, v_nPerTrace_3688_);
return v___x_3689_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify___boxed(lean_object** _args){
lean_object* v_F_3690_ = _args[0];
lean_object* v_EF_3691_ = _args[1];
lean_object* v_Digest_3692_ = _args[2];
lean_object* v_inst_3693_ = _args[3];
lean_object* v_inst_3694_ = _args[4];
lean_object* v_inst_3695_ = _args[5];
lean_object* v_inst_3696_ = _args[6];
lean_object* v_inst_3697_ = _args[7];
lean_object* v_fo_3698_ = _args[8];
lean_object* v_algMap_3699_ = _args[9];
lean_object* v_transcript_3700_ = _args[10];
lean_object* v_vk_3701_ = _args[11];
lean_object* v_publicValues_3702_ = _args[12];
lean_object* v_gkrProof_3703_ = _args[13];
lean_object* v_batchProof_3704_ = _args[14];
lean_object* v_traceVdata_3705_ = _args[15];
lean_object* v_traceIdToAirId_3706_ = _args[16];
lean_object* v_nPerTrace_3707_ = _args[17];
_start:
{
lean_object* v_res_3708_; 
v_res_3708_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_verify(v_F_3690_, v_EF_3691_, v_Digest_3692_, v_inst_3693_, v_inst_3694_, v_inst_3695_, v_inst_3696_, v_inst_3697_, v_fo_3698_, v_algMap_3699_, v_transcript_3700_, v_vk_3701_, v_publicValues_3702_, v_gkrProof_3703_, v_batchProof_3704_, v_traceVdata_3705_, v_traceIdToAirId_3706_, v_nPerTrace_3707_);
lean_dec(v_traceVdata_3705_);
return v_res_3708_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_Core(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Ops(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Ops(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Shape(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
