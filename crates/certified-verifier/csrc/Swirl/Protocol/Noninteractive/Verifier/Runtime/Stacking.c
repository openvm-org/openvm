// Lean compiler output
// Module: Swirl.Protocol.Noninteractive.Verifier.Runtime.Stacking
// Imports: public import Init public meta import Init public import Swirl.Protocol.Noninteractive.Runtime.Core public import Swirl.Protocol.Noninteractive.Ops public import Swirl.Protocol.Noninteractive.Verifier.Runtime.Batch
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
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(lean_object*, lean_object*, lean_object*);
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
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
uint8_t l_Nat_testBit(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instReprTupleOfRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "uPrism"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__9_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "uCube"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__10_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__11_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__13_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__14;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "openingPairs"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "round0Claim"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "u0"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__9_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__10;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "nextClaim"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__11_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__12_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__13;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State(lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectRound0OpeningPairs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectRound0OpeningPairs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "claim"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState(lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__0_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__1_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__2_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__3_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__4_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__5_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__5_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__6_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__7_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__7_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__10_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__4, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__11_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__7, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__12_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_instMonad___redArg___lam__9, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__13_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_map, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__14_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__15_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_pure, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__15_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__16_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__11_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__12_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__13_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__17_value;
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateT_bind, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9_value)} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__18_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__17_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__18_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__19 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
lean_object* v_uPrism_4_; lean_object* v_uCube_5_; lean_object* v_uPrism_6_; lean_object* v_uCube_7_; uint8_t v___x_8_; 
v_uPrism_4_ = lean_ctor_get(v_x_2_, 0);
lean_inc(v_uPrism_4_);
v_uCube_5_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_uCube_5_);
lean_dec_ref(v_x_2_);
v_uPrism_6_ = lean_ctor_get(v_x_3_, 0);
lean_inc(v_uPrism_6_);
v_uCube_7_ = lean_ctor_get(v_x_3_, 1);
lean_inc(v_uCube_7_);
lean_dec_ref(v_x_3_);
lean_inc_ref(v_inst_1_);
v___x_8_ = l_instDecidableEqList___redArg(v_inst_1_, v_uPrism_4_, v_uPrism_6_);
if (v___x_8_ == 0)
{
lean_dec(v_uCube_7_);
lean_dec(v_uCube_5_);
lean_dec_ref(v_inst_1_);
return v___x_8_;
}
else
{
uint8_t v___x_9_; 
v___x_9_ = l_instDecidableEqList___redArg(v_inst_1_, v_uCube_5_, v_uCube_7_);
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg___boxed(lean_object* v_inst_10_, lean_object* v_x_11_, lean_object* v_x_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg(v_inst_10_, v_x_11_, v_x_12_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq(lean_object* v_EF_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v___x_19_; 
v___x_19_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg(v_inst_16_, v_x_17_, v_x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___boxed(lean_object* v_EF_20_, lean_object* v_inst_21_, lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
uint8_t v_res_24_; lean_object* v_r_25_; 
v_res_24_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq(v_EF_20_, v_inst_21_, v_x_22_, v_x_23_);
v_r_25_ = lean_box(v_res_24_);
return v_r_25_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___redArg(lean_object* v_inst_26_, lean_object* v_x_27_, lean_object* v_x_28_){
_start:
{
uint8_t v___x_29_; 
v___x_29_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg(v_inst_26_, v_x_27_, v_x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___redArg___boxed(lean_object* v_inst_30_, lean_object* v_x_31_, lean_object* v_x_32_){
_start:
{
uint8_t v_res_33_; lean_object* v_r_34_; 
v_res_33_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___redArg(v_inst_30_, v_x_31_, v_x_32_);
v_r_34_ = lean_box(v_res_33_);
return v_r_34_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput(lean_object* v_EF_35_, lean_object* v_inst_36_, lean_object* v_x_37_, lean_object* v_x_38_){
_start:
{
uint8_t v___x_39_; 
v___x_39_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput_decEq___redArg(v_inst_36_, v_x_37_, v_x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput___boxed(lean_object* v_EF_40_, lean_object* v_inst_41_, lean_object* v_x_42_, lean_object* v_x_43_){
_start:
{
uint8_t v_res_44_; lean_object* v_r_45_; 
v_res_44_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqOutput(v_EF_40_, v_inst_41_, v_x_42_, v_x_43_);
v_r_45_ = lean_box(v_res_44_);
return v_r_45_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_unsigned_to_nat(10u);
v___x_60_ = lean_nat_to_int(v___x_59_);
return v___x_60_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = lean_unsigned_to_nat(9u);
v___x_68_ = lean_nat_to_int(v___x_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__14(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__0));
v___x_71_ = lean_string_length(v___x_70_);
return v___x_71_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__14, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__14_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__14);
v___x_73_ = lean_nat_to_int(v___x_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg(lean_object* v_inst_78_, lean_object* v_x_79_){
_start:
{
lean_object* v_uPrism_80_; lean_object* v_uCube_81_; lean_object* v___x_83_; uint8_t v_isShared_84_; uint8_t v_isSharedCheck_114_; 
v_uPrism_80_ = lean_ctor_get(v_x_79_, 0);
v_uCube_81_ = lean_ctor_get(v_x_79_, 1);
v_isSharedCheck_114_ = !lean_is_exclusive(v_x_79_);
if (v_isSharedCheck_114_ == 0)
{
v___x_83_ = v_x_79_;
v_isShared_84_ = v_isSharedCheck_114_;
goto v_resetjp_82_;
}
else
{
lean_inc(v_uCube_81_);
lean_inc(v_uPrism_80_);
lean_dec(v_x_79_);
v___x_83_ = lean_box(0);
v_isShared_84_ = v_isSharedCheck_114_;
goto v_resetjp_82_;
}
v_resetjp_82_:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_90_; 
v___x_85_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5));
v___x_86_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__6));
v___x_87_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7);
lean_inc_ref(v_inst_78_);
v___x_88_ = l_List_repr___redArg(v_inst_78_, v_uPrism_80_);
if (v_isShared_84_ == 0)
{
lean_ctor_set_tag(v___x_83_, 4);
lean_ctor_set(v___x_83_, 1, v___x_88_);
lean_ctor_set(v___x_83_, 0, v___x_87_);
v___x_90_ = v___x_83_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v___x_87_);
lean_ctor_set(v_reuseFailAlloc_113_, 1, v___x_88_);
v___x_90_ = v_reuseFailAlloc_113_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
uint8_t v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_91_ = 0;
v___x_92_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_92_, 0, v___x_90_);
lean_ctor_set_uint8(v___x_92_, sizeof(void*)*1, v___x_91_);
v___x_93_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_86_);
lean_ctor_set(v___x_93_, 1, v___x_92_);
v___x_94_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__9));
v___x_95_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_93_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = lean_box(1);
v___x_97_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_95_);
lean_ctor_set(v___x_97_, 1, v___x_96_);
v___x_98_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__11));
v___x_99_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_97_);
lean_ctor_set(v___x_99_, 1, v___x_98_);
v___x_100_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v___x_85_);
v___x_101_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12);
v___x_102_ = l_List_repr___redArg(v_inst_78_, v_uCube_81_);
v___x_103_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_101_);
lean_ctor_set(v___x_103_, 1, v___x_102_);
v___x_104_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set_uint8(v___x_104_, sizeof(void*)*1, v___x_91_);
v___x_105_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_100_);
lean_ctor_set(v___x_105_, 1, v___x_104_);
v___x_106_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15);
v___x_107_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__16));
v___x_108_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v___x_105_);
v___x_109_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__17));
v___x_110_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_108_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_106_);
lean_ctor_set(v___x_111_, 1, v___x_110_);
v___x_112_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set_uint8(v___x_112_, sizeof(void*)*1, v___x_91_);
return v___x_112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr(lean_object* v_EF_115_, lean_object* v_inst_116_, lean_object* v_x_117_, lean_object* v_prec_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg(v_inst_116_, v_x_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___boxed(lean_object* v_EF_120_, lean_object* v_inst_121_, lean_object* v_x_122_, lean_object* v_prec_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr(v_EF_120_, v_inst_121_, v_x_122_, v_prec_123_);
lean_dec(v_prec_123_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput___redArg(lean_object* v_inst_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_126_, 0, lean_box(0));
lean_closure_set(v___x_126_, 1, v_inst_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput(lean_object* v_EF_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___boxed), 4, 2);
lean_closure_set(v___x_129_, 0, lean_box(0));
lean_closure_set(v___x_129_, 1, v_inst_128_);
return v___x_129_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___lam__0(lean_object* v_inst_130_, lean_object* v_a_131_, lean_object* v_b_132_){
_start:
{
uint8_t v___x_133_; 
lean_inc_ref(v_inst_130_);
v___x_133_ = l_instDecidableEqProd___redArg(v_inst_130_, v_inst_130_, v_a_131_, v_b_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___lam__0___boxed(lean_object* v_inst_134_, lean_object* v_a_135_, lean_object* v_b_136_){
_start:
{
uint8_t v_res_137_; lean_object* v_r_138_; 
v_res_137_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___lam__0(v_inst_134_, v_a_135_, v_b_136_);
v_r_138_ = lean_box(v_res_137_);
return v_r_138_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg(lean_object* v_inst_139_, lean_object* v_x_140_, lean_object* v_x_141_){
_start:
{
lean_object* v_openingPairs_142_; lean_object* v_round0Claim_143_; lean_object* v_u0_144_; lean_object* v_nextClaim_145_; lean_object* v_openingPairs_146_; lean_object* v_round0Claim_147_; lean_object* v_u0_148_; lean_object* v_nextClaim_149_; lean_object* v___f_150_; uint8_t v___x_151_; 
v_openingPairs_142_ = lean_ctor_get(v_x_140_, 0);
lean_inc(v_openingPairs_142_);
v_round0Claim_143_ = lean_ctor_get(v_x_140_, 1);
lean_inc(v_round0Claim_143_);
v_u0_144_ = lean_ctor_get(v_x_140_, 2);
lean_inc(v_u0_144_);
v_nextClaim_145_ = lean_ctor_get(v_x_140_, 3);
lean_inc(v_nextClaim_145_);
lean_dec_ref(v_x_140_);
v_openingPairs_146_ = lean_ctor_get(v_x_141_, 0);
lean_inc(v_openingPairs_146_);
v_round0Claim_147_ = lean_ctor_get(v_x_141_, 1);
lean_inc(v_round0Claim_147_);
v_u0_148_ = lean_ctor_get(v_x_141_, 2);
lean_inc(v_u0_148_);
v_nextClaim_149_ = lean_ctor_get(v_x_141_, 3);
lean_inc(v_nextClaim_149_);
lean_dec_ref(v_x_141_);
lean_inc_ref(v_inst_139_);
v___f_150_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_150_, 0, v_inst_139_);
v___x_151_ = l_instDecidableEqList___redArg(v___f_150_, v_openingPairs_142_, v_openingPairs_146_);
if (v___x_151_ == 0)
{
lean_dec(v_nextClaim_149_);
lean_dec(v_u0_148_);
lean_dec(v_round0Claim_147_);
lean_dec(v_nextClaim_145_);
lean_dec(v_u0_144_);
lean_dec(v_round0Claim_143_);
lean_dec_ref(v_inst_139_);
return v___x_151_;
}
else
{
lean_object* v___x_152_; uint8_t v___x_153_; 
lean_inc_ref(v_inst_139_);
v___x_152_ = lean_apply_2(v_inst_139_, v_round0Claim_143_, v_round0Claim_147_);
v___x_153_ = lean_unbox(v___x_152_);
if (v___x_153_ == 0)
{
uint8_t v___x_154_; 
lean_dec(v_nextClaim_149_);
lean_dec(v_u0_148_);
lean_dec(v_nextClaim_145_);
lean_dec(v_u0_144_);
lean_dec_ref(v_inst_139_);
v___x_154_ = lean_unbox(v___x_152_);
return v___x_154_;
}
else
{
lean_object* v___x_155_; uint8_t v___x_156_; 
lean_inc_ref(v_inst_139_);
v___x_155_ = lean_apply_2(v_inst_139_, v_u0_144_, v_u0_148_);
v___x_156_ = lean_unbox(v___x_155_);
if (v___x_156_ == 0)
{
uint8_t v___x_157_; 
lean_dec(v_nextClaim_149_);
lean_dec(v_nextClaim_145_);
lean_dec_ref(v_inst_139_);
v___x_157_ = lean_unbox(v___x_155_);
return v___x_157_;
}
else
{
lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_158_ = lean_apply_2(v_inst_139_, v_nextClaim_145_, v_nextClaim_149_);
v___x_159_ = lean_unbox(v___x_158_);
return v___x_159_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg___boxed(lean_object* v_inst_160_, lean_object* v_x_161_, lean_object* v_x_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg(v_inst_160_, v_x_161_, v_x_162_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq(lean_object* v_EF_165_, lean_object* v_inst_166_, lean_object* v_x_167_, lean_object* v_x_168_){
_start:
{
uint8_t v___x_169_; 
v___x_169_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg(v_inst_166_, v_x_167_, v_x_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___boxed(lean_object* v_EF_170_, lean_object* v_inst_171_, lean_object* v_x_172_, lean_object* v_x_173_){
_start:
{
uint8_t v_res_174_; lean_object* v_r_175_; 
v_res_174_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq(v_EF_170_, v_inst_171_, v_x_172_, v_x_173_);
v_r_175_ = lean_box(v_res_174_);
return v_r_175_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___redArg(lean_object* v_inst_176_, lean_object* v_x_177_, lean_object* v_x_178_){
_start:
{
uint8_t v___x_179_; 
v___x_179_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg(v_inst_176_, v_x_177_, v_x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___redArg___boxed(lean_object* v_inst_180_, lean_object* v_x_181_, lean_object* v_x_182_){
_start:
{
uint8_t v_res_183_; lean_object* v_r_184_; 
v_res_183_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___redArg(v_inst_180_, v_x_181_, v_x_182_);
v_r_184_ = lean_box(v_res_183_);
return v_r_184_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State(lean_object* v_EF_185_, lean_object* v_inst_186_, lean_object* v_x_187_, lean_object* v_x_188_){
_start:
{
uint8_t v___x_189_; 
v___x_189_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State_decEq___redArg(v_inst_186_, v_x_187_, v_x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State___boxed(lean_object* v_EF_190_, lean_object* v_inst_191_, lean_object* v_x_192_, lean_object* v_x_193_){
_start:
{
uint8_t v_res_194_; lean_object* v_r_195_; 
v_res_194_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRound0State(v_EF_190_, v_inst_191_, v_x_192_, v_x_193_);
v_r_195_ = lean_box(v_res_194_);
return v_r_195_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = lean_unsigned_to_nat(16u);
v___x_206_ = lean_nat_to_int(v___x_205_);
return v___x_206_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_210_ = lean_unsigned_to_nat(15u);
v___x_211_ = lean_nat_to_int(v___x_210_);
return v___x_211_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_215_ = lean_unsigned_to_nat(6u);
v___x_216_ = lean_nat_to_int(v___x_215_);
return v___x_216_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__13(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = lean_unsigned_to_nat(13u);
v___x_221_ = lean_nat_to_int(v___x_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg(lean_object* v_inst_222_, lean_object* v_x_223_){
_start:
{
lean_object* v_openingPairs_224_; lean_object* v_round0Claim_225_; lean_object* v_u0_226_; lean_object* v_nextClaim_227_; lean_object* v___f_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v_openingPairs_224_ = lean_ctor_get(v_x_223_, 0);
lean_inc(v_openingPairs_224_);
v_round0Claim_225_ = lean_ctor_get(v_x_223_, 1);
lean_inc(v_round0Claim_225_);
v_u0_226_ = lean_ctor_get(v_x_223_, 2);
lean_inc(v_u0_226_);
v_nextClaim_227_ = lean_ctor_get(v_x_223_, 3);
lean_inc(v_nextClaim_227_);
lean_dec_ref(v_x_223_);
lean_inc_ref_n(v_inst_222_, 4);
v___f_228_ = lean_alloc_closure((void*)(l_instReprTupleOfRepr___redArg___lam__0), 3, 1);
lean_closure_set(v___f_228_, 0, v_inst_222_);
v___x_229_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5));
v___x_230_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__3));
v___x_231_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__4);
v___x_232_ = lean_alloc_closure((void*)(l_Prod_repr___boxed), 6, 4);
lean_closure_set(v___x_232_, 0, lean_box(0));
lean_closure_set(v___x_232_, 1, lean_box(0));
lean_closure_set(v___x_232_, 2, v_inst_222_);
lean_closure_set(v___x_232_, 3, v___f_228_);
v___x_233_ = lean_unsigned_to_nat(0u);
v___x_234_ = l_List_repr___redArg(v___x_232_, v_openingPairs_224_);
v___x_235_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_231_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
v___x_236_ = 0;
v___x_237_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_237_, 0, v___x_235_);
lean_ctor_set_uint8(v___x_237_, sizeof(void*)*1, v___x_236_);
v___x_238_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_230_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__9));
v___x_240_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_238_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = lean_box(1);
v___x_242_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__6));
v___x_244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_242_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v___x_229_);
v___x_246_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__7);
v___x_247_ = lean_apply_2(v_inst_222_, v_round0Claim_225_, v___x_233_);
v___x_248_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set_uint8(v___x_249_, sizeof(void*)*1, v___x_236_);
v___x_250_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_245_);
lean_ctor_set(v___x_250_, 1, v___x_249_);
v___x_251_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___x_239_);
v___x_252_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v___x_241_);
v___x_253_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__9));
v___x_254_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_252_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
v___x_255_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v___x_229_);
v___x_256_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__10);
v___x_257_ = lean_apply_2(v_inst_222_, v_u0_226_, v___x_233_);
v___x_258_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_256_);
lean_ctor_set(v___x_258_, 1, v___x_257_);
v___x_259_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_259_, 0, v___x_258_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*1, v___x_236_);
v___x_260_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_255_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
v___x_261_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
lean_ctor_set(v___x_261_, 1, v___x_239_);
v___x_262_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_241_);
v___x_263_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__12));
v___x_264_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_262_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
lean_ctor_set(v___x_265_, 1, v___x_229_);
v___x_266_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__13, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__13_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg___closed__13);
v___x_267_ = lean_apply_2(v_inst_222_, v_nextClaim_227_, v___x_233_);
v___x_268_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_266_);
lean_ctor_set(v___x_268_, 1, v___x_267_);
v___x_269_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_269_, 0, v___x_268_);
lean_ctor_set_uint8(v___x_269_, sizeof(void*)*1, v___x_236_);
v___x_270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_270_, 0, v___x_265_);
lean_ctor_set(v___x_270_, 1, v___x_269_);
v___x_271_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15);
v___x_272_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__16));
v___x_273_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_272_);
lean_ctor_set(v___x_273_, 1, v___x_270_);
v___x_274_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__17));
v___x_275_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_273_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_271_);
lean_ctor_set(v___x_276_, 1, v___x_275_);
v___x_277_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set_uint8(v___x_277_, sizeof(void*)*1, v___x_236_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr(lean_object* v_EF_278_, lean_object* v_inst_279_, lean_object* v_x_280_, lean_object* v_prec_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___redArg(v_inst_279_, v_x_280_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___boxed(lean_object* v_EF_283_, lean_object* v_inst_284_, lean_object* v_x_285_, lean_object* v_prec_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr(v_EF_283_, v_inst_284_, v_x_285_, v_prec_286_);
lean_dec(v_prec_286_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State___redArg(lean_object* v_inst_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___boxed), 4, 2);
lean_closure_set(v___x_289_, 0, lean_box(0));
lean_closure_set(v___x_289_, 1, v_inst_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State(lean_object* v_EF_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRound0State_repr___boxed), 4, 2);
lean_closure_set(v___x_292_, 0, lean_box(0));
lean_closure_set(v___x_292_, 1, v_inst_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg(lean_object* v_values_298_){
_start:
{
if (lean_obj_tag(v_values_298_) == 0)
{
lean_object* v___x_299_; 
v___x_299_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__0));
return v___x_299_;
}
else
{
lean_object* v_tail_300_; 
v_tail_300_ = lean_ctor_get(v_values_298_, 1);
lean_inc(v_tail_300_);
if (lean_obj_tag(v_tail_300_) == 0)
{
lean_object* v___x_301_; 
lean_dec_ref_known(v_values_298_, 2);
v___x_301_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_301_;
}
else
{
lean_object* v_head_302_; lean_object* v___x_304_; uint8_t v_isShared_305_; uint8_t v_isSharedCheck_327_; 
v_head_302_ = lean_ctor_get(v_values_298_, 0);
v_isSharedCheck_327_ = !lean_is_exclusive(v_values_298_);
if (v_isSharedCheck_327_ == 0)
{
lean_object* v_unused_328_; 
v_unused_328_ = lean_ctor_get(v_values_298_, 1);
lean_dec(v_unused_328_);
v___x_304_ = v_values_298_;
v_isShared_305_ = v_isSharedCheck_327_;
goto v_resetjp_303_;
}
else
{
lean_inc(v_head_302_);
lean_dec(v_values_298_);
v___x_304_ = lean_box(0);
v_isShared_305_ = v_isSharedCheck_327_;
goto v_resetjp_303_;
}
v_resetjp_303_:
{
lean_object* v_head_306_; lean_object* v_tail_307_; lean_object* v___x_309_; uint8_t v_isShared_310_; uint8_t v_isSharedCheck_326_; 
v_head_306_ = lean_ctor_get(v_tail_300_, 0);
v_tail_307_ = lean_ctor_get(v_tail_300_, 1);
v_isSharedCheck_326_ = !lean_is_exclusive(v_tail_300_);
if (v_isSharedCheck_326_ == 0)
{
v___x_309_ = v_tail_300_;
v_isShared_310_ = v_isSharedCheck_326_;
goto v_resetjp_308_;
}
else
{
lean_inc(v_tail_307_);
lean_inc(v_head_306_);
lean_dec(v_tail_300_);
v___x_309_ = lean_box(0);
v_isShared_310_ = v_isSharedCheck_326_;
goto v_resetjp_308_;
}
v_resetjp_308_:
{
lean_object* v___x_311_; 
v___x_311_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg(v_tail_307_);
if (lean_obj_tag(v___x_311_) == 0)
{
lean_del_object(v___x_309_);
lean_dec(v_head_306_);
lean_del_object(v___x_304_);
lean_dec(v_head_302_);
return v___x_311_;
}
else
{
lean_object* v_a_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_325_; 
v_a_312_ = lean_ctor_get(v___x_311_, 0);
v_isSharedCheck_325_ = !lean_is_exclusive(v___x_311_);
if (v_isSharedCheck_325_ == 0)
{
v___x_314_ = v___x_311_;
v_isShared_315_ = v_isSharedCheck_325_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_a_312_);
lean_dec(v___x_311_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_325_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v___x_317_; 
if (v_isShared_305_ == 0)
{
lean_ctor_set_tag(v___x_304_, 0);
lean_ctor_set(v___x_304_, 1, v_head_306_);
v___x_317_ = v___x_304_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v_head_302_);
lean_ctor_set(v_reuseFailAlloc_324_, 1, v_head_306_);
v___x_317_ = v_reuseFailAlloc_324_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
lean_object* v___x_319_; 
if (v_isShared_310_ == 0)
{
lean_ctor_set(v___x_309_, 1, v_a_312_);
lean_ctor_set(v___x_309_, 0, v___x_317_);
v___x_319_ = v___x_309_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v___x_317_);
lean_ctor_set(v_reuseFailAlloc_323_, 1, v_a_312_);
v___x_319_ = v_reuseFailAlloc_323_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
lean_object* v___x_321_; 
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 0, v___x_319_);
v___x_321_ = v___x_314_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v___x_319_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent(lean_object* v_EF_329_, lean_object* v_values_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg(v_values_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg(lean_object* v_fo_332_, lean_object* v_openings_333_, uint8_t v_needRot_334_){
_start:
{
if (v_needRot_334_ == 0)
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_335_ = lean_box(0);
v___x_336_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch_columnOpeningsByRot_spec__0___redArg(v_fo_332_, v_openings_333_, v___x_335_);
v___x_337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_337_, 0, v___x_336_);
return v___x_337_;
}
else
{
lean_object* v___x_338_; 
lean_dec_ref(v_fo_332_);
v___x_338_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg(v_openings_333_);
return v___x_338_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg___boxed(lean_object* v_fo_339_, lean_object* v_openings_340_, lean_object* v_needRot_341_){
_start:
{
uint8_t v_needRot_boxed_342_; lean_object* v_res_343_; 
v_needRot_boxed_342_ = lean_unbox(v_needRot_341_);
v_res_343_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg(v_fo_339_, v_openings_340_, v_needRot_boxed_342_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot(lean_object* v_EF_344_, lean_object* v_fo_345_, lean_object* v_openings_346_, uint8_t v_needRot_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg(v_fo_345_, v_openings_346_, v_needRot_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___boxed(lean_object* v_EF_349_, lean_object* v_fo_350_, lean_object* v_openings_351_, lean_object* v_needRot_352_){
_start:
{
uint8_t v_needRot_boxed_353_; lean_object* v_res_354_; 
v_needRot_boxed_353_ = lean_unbox(v_needRot_352_);
v_res_354_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot(v_EF_349_, v_fo_350_, v_openings_351_, v_needRot_boxed_353_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg(lean_object* v_fo_355_, lean_object* v_columnOpenings_356_, lean_object* v_needRotPerTrace_357_){
_start:
{
if (lean_obj_tag(v_columnOpenings_356_) == 0)
{
lean_dec_ref(v_fo_355_);
if (lean_obj_tag(v_needRotPerTrace_357_) == 0)
{
lean_object* v___x_360_; 
v___x_360_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__0));
return v___x_360_;
}
else
{
goto v___jp_358_;
}
}
else
{
if (lean_obj_tag(v_needRotPerTrace_357_) == 1)
{
lean_object* v_head_361_; 
v_head_361_ = lean_ctor_get(v_columnOpenings_356_, 0);
lean_inc(v_head_361_);
if (lean_obj_tag(v_head_361_) == 0)
{
lean_object* v___x_362_; 
lean_dec_ref_known(v_columnOpenings_356_, 2);
lean_dec_ref(v_fo_355_);
v___x_362_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_362_;
}
else
{
lean_object* v_tail_363_; lean_object* v_head_364_; lean_object* v_tail_365_; lean_object* v_head_366_; uint8_t v___x_367_; lean_object* v___x_368_; 
v_tail_363_ = lean_ctor_get(v_columnOpenings_356_, 1);
lean_inc(v_tail_363_);
lean_dec_ref_known(v_columnOpenings_356_, 2);
v_head_364_ = lean_ctor_get(v_needRotPerTrace_357_, 0);
v_tail_365_ = lean_ctor_get(v_needRotPerTrace_357_, 1);
v_head_366_ = lean_ctor_get(v_head_361_, 0);
lean_inc(v_head_366_);
lean_dec_ref_known(v_head_361_, 2);
v___x_367_ = lean_unbox(v_head_364_);
lean_inc_ref(v_fo_355_);
v___x_368_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg(v_fo_355_, v_head_366_, v___x_367_);
if (lean_obj_tag(v___x_368_) == 0)
{
lean_dec(v_tail_363_);
lean_dec_ref(v_fo_355_);
return v___x_368_;
}
else
{
lean_object* v_a_369_; lean_object* v___x_370_; 
v_a_369_ = lean_ctor_get(v___x_368_, 0);
lean_inc(v_a_369_);
lean_dec_ref_known(v___x_368_, 1);
v___x_370_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg(v_fo_355_, v_tail_363_, v_tail_365_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_dec(v_a_369_);
return v___x_370_;
}
else
{
lean_object* v_a_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_379_; 
v_a_371_ = lean_ctor_get(v___x_370_, 0);
v_isSharedCheck_379_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_379_ == 0)
{
v___x_373_ = v___x_370_;
v_isShared_374_ = v_isSharedCheck_379_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_a_371_);
lean_dec(v___x_370_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_379_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; lean_object* v___x_377_; 
v___x_375_ = l_List_appendTR___redArg(v_a_369_, v_a_371_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_375_);
v___x_377_ = v___x_373_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
else
{
lean_dec_ref_known(v_columnOpenings_356_, 2);
lean_dec_ref(v_fo_355_);
goto v___jp_358_;
}
}
v___jp_358_:
{
lean_object* v___x_359_; 
v___x_359_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_359_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg___boxed(lean_object* v_fo_380_, lean_object* v_columnOpenings_381_, lean_object* v_needRotPerTrace_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg(v_fo_380_, v_columnOpenings_381_, v_needRotPerTrace_382_);
lean_dec(v_needRotPerTrace_382_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs(lean_object* v_EF_384_, lean_object* v_fo_385_, lean_object* v_columnOpenings_386_, lean_object* v_needRotPerTrace_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg(v_fo_385_, v_columnOpenings_386_, v_needRotPerTrace_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___boxed(lean_object* v_EF_389_, lean_object* v_fo_390_, lean_object* v_columnOpenings_391_, lean_object* v_needRotPerTrace_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs(v_EF_389_, v_fo_390_, v_columnOpenings_391_, v_needRotPerTrace_392_);
lean_dec(v_needRotPerTrace_392_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg(lean_object* v_fo_397_, lean_object* v_parts_398_, lean_object* v_needRot_399_){
_start:
{
if (lean_obj_tag(v_parts_398_) == 0)
{
lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; 
lean_dec_ref(v_fo_397_);
v___x_402_ = lean_box(0);
v___x_403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
lean_ctor_set(v___x_403_, 1, v_needRot_399_);
v___x_404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_404_, 0, v___x_403_);
return v___x_404_;
}
else
{
if (lean_obj_tag(v_needRot_399_) == 1)
{
lean_object* v_head_405_; 
v_head_405_ = lean_ctor_get(v_needRot_399_, 0);
lean_inc(v_head_405_);
if (lean_obj_tag(v_head_405_) == 1)
{
lean_object* v_tail_406_; 
v_tail_406_ = lean_ctor_get(v_head_405_, 1);
if (lean_obj_tag(v_tail_406_) == 0)
{
lean_object* v_head_407_; lean_object* v_tail_408_; lean_object* v_tail_409_; lean_object* v_head_410_; uint8_t v___x_411_; lean_object* v___x_412_; 
v_head_407_ = lean_ctor_get(v_parts_398_, 0);
lean_inc(v_head_407_);
v_tail_408_ = lean_ctor_get(v_parts_398_, 1);
lean_inc(v_tail_408_);
lean_dec_ref_known(v_parts_398_, 2);
v_tail_409_ = lean_ctor_get(v_needRot_399_, 1);
lean_inc(v_tail_409_);
lean_dec_ref_known(v_needRot_399_, 2);
v_head_410_ = lean_ctor_get(v_head_405_, 0);
lean_inc(v_head_410_);
lean_dec_ref_known(v_head_405_, 2);
v___x_411_ = lean_unbox(v_head_410_);
lean_dec(v_head_410_);
lean_inc_ref(v_fo_397_);
v___x_412_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_columnOpeningsByRot___redArg(v_fo_397_, v_head_407_, v___x_411_);
if (lean_obj_tag(v___x_412_) == 0)
{
lean_object* v___x_413_; 
lean_dec_ref_known(v___x_412_, 1);
lean_dec(v_tail_409_);
lean_dec(v_tail_408_);
lean_dec_ref(v_fo_397_);
v___x_413_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg___closed__0));
return v___x_413_;
}
else
{
lean_object* v_a_414_; lean_object* v___x_415_; 
v_a_414_ = lean_ctor_get(v___x_412_, 0);
lean_inc(v_a_414_);
lean_dec_ref_known(v___x_412_, 1);
v___x_415_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg(v_fo_397_, v_tail_408_, v_tail_409_);
if (lean_obj_tag(v___x_415_) == 0)
{
lean_dec(v_a_414_);
return v___x_415_;
}
else
{
lean_object* v_a_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_433_; 
v_a_416_ = lean_ctor_get(v___x_415_, 0);
v_isSharedCheck_433_ = !lean_is_exclusive(v___x_415_);
if (v_isSharedCheck_433_ == 0)
{
v___x_418_ = v___x_415_;
v_isShared_419_ = v_isSharedCheck_433_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_a_416_);
lean_dec(v___x_415_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_433_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
lean_object* v_fst_420_; lean_object* v_snd_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_432_; 
v_fst_420_ = lean_ctor_get(v_a_416_, 0);
v_snd_421_ = lean_ctor_get(v_a_416_, 1);
v_isSharedCheck_432_ = !lean_is_exclusive(v_a_416_);
if (v_isSharedCheck_432_ == 0)
{
v___x_423_ = v_a_416_;
v_isShared_424_ = v_isSharedCheck_432_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_snd_421_);
lean_inc(v_fst_420_);
lean_dec(v_a_416_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_432_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v___x_425_; lean_object* v___x_427_; 
v___x_425_ = l_List_appendTR___redArg(v_a_414_, v_fst_420_);
if (v_isShared_424_ == 0)
{
lean_ctor_set(v___x_423_, 0, v___x_425_);
v___x_427_ = v___x_423_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_431_, 1, v_snd_421_);
v___x_427_ = v_reuseFailAlloc_431_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
lean_object* v___x_429_; 
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 0, v___x_427_);
v___x_429_ = v___x_418_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_427_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_head_405_, 2);
lean_dec_ref_known(v_needRot_399_, 2);
lean_dec_ref_known(v_parts_398_, 2);
lean_dec_ref(v_fo_397_);
goto v___jp_400_;
}
}
else
{
lean_dec_ref_known(v_needRot_399_, 2);
lean_dec(v_head_405_);
lean_dec_ref_known(v_parts_398_, 2);
lean_dec_ref(v_fo_397_);
goto v___jp_400_;
}
}
else
{
lean_dec_ref_known(v_parts_398_, 2);
lean_dec(v_needRot_399_);
lean_dec_ref(v_fo_397_);
goto v___jp_400_;
}
}
v___jp_400_:
{
lean_object* v___x_401_; 
v___x_401_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg___closed__0));
return v___x_401_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts(lean_object* v_EF_434_, lean_object* v_fo_435_, lean_object* v_parts_436_, lean_object* v_needRot_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg(v_fo_435_, v_parts_436_, v_needRot_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg(lean_object* v_fo_439_, lean_object* v_allParts_440_, lean_object* v_needRot_441_){
_start:
{
if (lean_obj_tag(v_allParts_440_) == 0)
{
lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; 
lean_dec_ref(v_fo_439_);
v___x_442_ = lean_box(0);
v___x_443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
lean_ctor_set(v___x_443_, 1, v_needRot_441_);
v___x_444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_444_, 0, v___x_443_);
return v___x_444_;
}
else
{
lean_object* v_head_445_; lean_object* v_tail_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v_head_445_ = lean_ctor_get(v_allParts_440_, 0);
v_tail_446_ = lean_ctor_get(v_allParts_440_, 1);
v___x_447_ = lean_unsigned_to_nat(1u);
v___x_448_ = l_List_drop___redArg(v___x_447_, v_head_445_);
lean_inc_ref(v_fo_439_);
v___x_449_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectParts___redArg(v_fo_439_, v___x_448_, v_needRot_441_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_dec_ref(v_fo_439_);
return v___x_449_;
}
else
{
lean_object* v_a_450_; lean_object* v_fst_451_; lean_object* v_snd_452_; lean_object* v___x_453_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___x_449_, 1);
v_fst_451_ = lean_ctor_get(v_a_450_, 0);
lean_inc(v_fst_451_);
v_snd_452_ = lean_ctor_get(v_a_450_, 1);
lean_inc(v_snd_452_);
lean_dec(v_a_450_);
v___x_453_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg(v_fo_439_, v_tail_446_, v_snd_452_);
if (lean_obj_tag(v___x_453_) == 0)
{
lean_dec(v_fst_451_);
return v___x_453_;
}
else
{
lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_471_; 
v_a_454_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_471_ == 0)
{
v___x_456_ = v___x_453_;
v_isShared_457_ = v_isSharedCheck_471_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_453_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_471_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v_fst_458_; lean_object* v_snd_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_470_; 
v_fst_458_ = lean_ctor_get(v_a_454_, 0);
v_snd_459_ = lean_ctor_get(v_a_454_, 1);
v_isSharedCheck_470_ = !lean_is_exclusive(v_a_454_);
if (v_isSharedCheck_470_ == 0)
{
v___x_461_ = v_a_454_;
v_isShared_462_ = v_isSharedCheck_470_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_snd_459_);
lean_inc(v_fst_458_);
lean_dec(v_a_454_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_470_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_463_; lean_object* v___x_465_; 
v___x_463_ = l_List_appendTR___redArg(v_fst_451_, v_fst_458_);
if (v_isShared_462_ == 0)
{
lean_ctor_set(v___x_461_, 0, v___x_463_);
v___x_465_ = v___x_461_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v___x_463_);
lean_ctor_set(v_reuseFailAlloc_469_, 1, v_snd_459_);
v___x_465_ = v_reuseFailAlloc_469_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
lean_object* v___x_467_; 
if (v_isShared_457_ == 0)
{
lean_ctor_set(v___x_456_, 0, v___x_465_);
v___x_467_ = v___x_456_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_468_; 
v_reuseFailAlloc_468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v___x_465_);
v___x_467_ = v_reuseFailAlloc_468_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
return v___x_467_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg___boxed(lean_object* v_fo_472_, lean_object* v_allParts_473_, lean_object* v_needRot_474_){
_start:
{
lean_object* v_res_475_; 
v_res_475_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg(v_fo_472_, v_allParts_473_, v_needRot_474_);
lean_dec(v_allParts_473_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs(lean_object* v_EF_476_, lean_object* v_fo_477_, lean_object* v_allParts_478_, lean_object* v_needRot_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg(v_fo_477_, v_allParts_478_, v_needRot_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___boxed(lean_object* v_EF_481_, lean_object* v_fo_482_, lean_object* v_allParts_483_, lean_object* v_needRot_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs(v_EF_481_, v_fo_482_, v_allParts_483_, v_needRot_484_);
lean_dec(v_allParts_483_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg(lean_object* v_fo_486_, lean_object* v_columnOpenings_487_, lean_object* v_remainingNeedRotPerCommit_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs_collectAirs___redArg(v_fo_486_, v_columnOpenings_487_, v_remainingNeedRotPerCommit_488_);
if (lean_obj_tag(v___x_489_) == 0)
{
lean_object* v___x_490_; 
lean_dec_ref_known(v___x_489_, 1);
v___x_490_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_490_;
}
else
{
lean_object* v_a_491_; lean_object* v___x_493_; uint8_t v_isShared_494_; uint8_t v_isSharedCheck_502_; 
v_a_491_ = lean_ctor_get(v___x_489_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_489_);
if (v_isSharedCheck_502_ == 0)
{
v___x_493_ = v___x_489_;
v_isShared_494_ = v_isSharedCheck_502_;
goto v_resetjp_492_;
}
else
{
lean_inc(v_a_491_);
lean_dec(v___x_489_);
v___x_493_ = lean_box(0);
v_isShared_494_ = v_isSharedCheck_502_;
goto v_resetjp_492_;
}
v_resetjp_492_:
{
lean_object* v_fst_495_; lean_object* v_snd_496_; uint8_t v___x_497_; 
v_fst_495_ = lean_ctor_get(v_a_491_, 0);
lean_inc(v_fst_495_);
v_snd_496_ = lean_ctor_get(v_a_491_, 1);
lean_inc(v_snd_496_);
lean_dec(v_a_491_);
v___x_497_ = l_List_isEmpty___redArg(v_snd_496_);
lean_dec(v_snd_496_);
if (v___x_497_ == 0)
{
lean_object* v___x_498_; 
lean_dec(v_fst_495_);
lean_del_object(v___x_493_);
v___x_498_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_498_;
}
else
{
lean_object* v___x_500_; 
if (v_isShared_494_ == 0)
{
lean_ctor_set(v___x_493_, 0, v_fst_495_);
v___x_500_ = v___x_493_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_fst_495_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg___boxed(lean_object* v_fo_503_, lean_object* v_columnOpenings_504_, lean_object* v_remainingNeedRotPerCommit_505_){
_start:
{
lean_object* v_res_506_; 
v_res_506_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg(v_fo_503_, v_columnOpenings_504_, v_remainingNeedRotPerCommit_505_);
lean_dec(v_columnOpenings_504_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs(lean_object* v_EF_507_, lean_object* v_fo_508_, lean_object* v_columnOpenings_509_, lean_object* v_remainingNeedRotPerCommit_510_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg(v_fo_508_, v_columnOpenings_509_, v_remainingNeedRotPerCommit_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___boxed(lean_object* v_EF_512_, lean_object* v_fo_513_, lean_object* v_columnOpenings_514_, lean_object* v_remainingNeedRotPerCommit_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs(v_EF_512_, v_fo_513_, v_columnOpenings_514_, v_remainingNeedRotPerCommit_515_);
lean_dec(v_columnOpenings_514_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectRound0OpeningPairs___redArg(lean_object* v_fo_517_, lean_object* v_columnOpenings_518_, lean_object* v_needRotPerCommit_519_){
_start:
{
if (lean_obj_tag(v_needRotPerCommit_519_) == 0)
{
lean_object* v___x_520_; 
lean_dec(v_columnOpenings_518_);
lean_dec_ref(v_fo_517_);
v___x_520_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_520_;
}
else
{
lean_object* v_head_521_; lean_object* v_tail_522_; lean_object* v___x_523_; 
v_head_521_ = lean_ctor_get(v_needRotPerCommit_519_, 0);
lean_inc(v_head_521_);
v_tail_522_ = lean_ctor_get(v_needRotPerCommit_519_, 1);
lean_inc(v_tail_522_);
lean_dec_ref_known(v_needRotPerCommit_519_, 2);
lean_inc(v_columnOpenings_518_);
lean_inc_ref(v_fo_517_);
v___x_523_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectCommonMainOpeningPairs___redArg(v_fo_517_, v_columnOpenings_518_, v_head_521_);
lean_dec(v_head_521_);
if (lean_obj_tag(v___x_523_) == 0)
{
lean_dec(v_tail_522_);
lean_dec(v_columnOpenings_518_);
lean_dec_ref(v_fo_517_);
return v___x_523_;
}
else
{
lean_object* v_a_524_; lean_object* v___x_525_; 
v_a_524_ = lean_ctor_get(v___x_523_, 0);
lean_inc(v_a_524_);
lean_dec_ref_known(v___x_523_, 1);
v___x_525_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectAuxiliaryOpeningPairs___redArg(v_fo_517_, v_columnOpenings_518_, v_tail_522_);
lean_dec(v_columnOpenings_518_);
if (lean_obj_tag(v___x_525_) == 0)
{
lean_dec(v_a_524_);
return v___x_525_;
}
else
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_534_; 
v_a_526_ = lean_ctor_get(v___x_525_, 0);
v_isSharedCheck_534_ = !lean_is_exclusive(v___x_525_);
if (v_isSharedCheck_534_ == 0)
{
v___x_528_ = v___x_525_;
v_isShared_529_ = v_isSharedCheck_534_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_525_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_534_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_530_; lean_object* v___x_532_; 
v___x_530_ = l_List_appendTR___redArg(v_a_524_, v_a_526_);
if (v_isShared_529_ == 0)
{
lean_ctor_set(v___x_528_, 0, v___x_530_);
v___x_532_ = v___x_528_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectRound0OpeningPairs(lean_object* v_EF_535_, lean_object* v_fo_536_, lean_object* v_columnOpenings_537_, lean_object* v_needRotPerCommit_538_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectRound0OpeningPairs___redArg(v_fo_536_, v_columnOpenings_537_, v_needRotPerCommit_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings_spec__0___redArg(lean_object* v___x_540_, lean_object* v_lambdaSq_541_, lean_object* v___x_542_, lean_object* v_lambda_543_, lean_object* v_x_544_, lean_object* v_x_545_){
_start:
{
if (lean_obj_tag(v_x_545_) == 0)
{
lean_dec(v_lambda_543_);
lean_dec_ref(v___x_542_);
lean_dec(v_lambdaSq_541_);
lean_dec(v___x_540_);
return v_x_544_;
}
else
{
lean_object* v_head_546_; lean_object* v_tail_547_; lean_object* v_fst_548_; lean_object* v_snd_549_; lean_object* v_fst_550_; lean_object* v_snd_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_565_; 
v_head_546_ = lean_ctor_get(v_x_545_, 0);
lean_inc(v_head_546_);
v_tail_547_ = lean_ctor_get(v_x_545_, 1);
lean_inc(v_tail_547_);
lean_dec_ref_known(v_x_545_, 2);
v_fst_548_ = lean_ctor_get(v_x_544_, 0);
lean_inc(v_fst_548_);
v_snd_549_ = lean_ctor_get(v_x_544_, 1);
lean_inc(v_snd_549_);
lean_dec_ref(v_x_544_);
v_fst_550_ = lean_ctor_get(v_head_546_, 0);
v_snd_551_ = lean_ctor_get(v_head_546_, 1);
v_isSharedCheck_565_ = !lean_is_exclusive(v_head_546_);
if (v_isSharedCheck_565_ == 0)
{
v___x_553_ = v_head_546_;
v_isShared_554_ = v_isSharedCheck_565_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_snd_551_);
lean_inc(v_fst_550_);
lean_dec(v_head_546_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_565_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v_add_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_562_; 
v_add_555_ = lean_ctor_get(v___x_542_, 3);
lean_inc_n(v___x_540_, 3);
lean_inc(v_lambdaSq_541_);
lean_inc(v_fst_548_);
v___x_556_ = lean_apply_2(v___x_540_, v_fst_548_, v_lambdaSq_541_);
lean_inc(v_lambda_543_);
v___x_557_ = lean_apply_2(v___x_540_, v_snd_551_, v_lambda_543_);
lean_inc_n(v_add_555_, 2);
v___x_558_ = lean_apply_2(v_add_555_, v_fst_550_, v___x_557_);
v___x_559_ = lean_apply_2(v___x_540_, v___x_558_, v_fst_548_);
v___x_560_ = lean_apply_2(v_add_555_, v_snd_549_, v___x_559_);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 1, v___x_560_);
lean_ctor_set(v___x_553_, 0, v___x_556_);
v___x_562_ = v___x_553_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v___x_556_);
lean_ctor_set(v_reuseFailAlloc_564_, 1, v___x_560_);
v___x_562_ = v_reuseFailAlloc_564_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
v_x_544_ = v___x_562_;
v_x_545_ = v_tail_547_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings___redArg(lean_object* v_fo_566_, lean_object* v_openingPairs_567_, lean_object* v_lambda_568_){
_start:
{
lean_object* v_toRingOps_569_; lean_object* v_toSemiringOps_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_583_; 
v_toRingOps_569_ = lean_ctor_get(v_fo_566_, 0);
lean_inc_ref(v_toRingOps_569_);
lean_dec_ref(v_fo_566_);
v_toSemiringOps_570_ = lean_ctor_get(v_toRingOps_569_, 0);
v_isSharedCheck_583_ = !lean_is_exclusive(v_toRingOps_569_);
if (v_isSharedCheck_583_ == 0)
{
lean_object* v_unused_584_; 
v_unused_584_ = lean_ctor_get(v_toRingOps_569_, 1);
lean_dec(v_unused_584_);
v___x_572_ = v_toRingOps_569_;
v_isShared_573_ = v_isSharedCheck_583_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_toSemiringOps_570_);
lean_dec(v_toRingOps_569_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_583_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v_zero_574_; lean_object* v_one_575_; lean_object* v_mul_576_; lean_object* v_lambdaSq_577_; lean_object* v___x_579_; 
v_zero_574_ = lean_ctor_get(v_toSemiringOps_570_, 0);
v_one_575_ = lean_ctor_get(v_toSemiringOps_570_, 1);
v_mul_576_ = lean_ctor_get(v_toSemiringOps_570_, 4);
lean_inc_n(v_mul_576_, 2);
lean_inc_n(v_lambda_568_, 2);
v_lambdaSq_577_ = lean_apply_2(v_mul_576_, v_lambda_568_, v_lambda_568_);
lean_inc(v_zero_574_);
lean_inc(v_one_575_);
if (v_isShared_573_ == 0)
{
lean_ctor_set(v___x_572_, 1, v_zero_574_);
lean_ctor_set(v___x_572_, 0, v_one_575_);
v___x_579_ = v___x_572_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v_one_575_);
lean_ctor_set(v_reuseFailAlloc_582_, 1, v_zero_574_);
v___x_579_ = v_reuseFailAlloc_582_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
lean_object* v___x_580_; lean_object* v_snd_581_; 
v___x_580_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings_spec__0___redArg(v_mul_576_, v_lambdaSq_577_, v_toSemiringOps_570_, v_lambda_568_, v___x_579_, v_openingPairs_567_);
v_snd_581_ = lean_ctor_get(v___x_580_, 1);
lean_inc(v_snd_581_);
lean_dec_ref(v___x_580_);
return v_snd_581_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings(lean_object* v_EF_585_, lean_object* v_fo_586_, lean_object* v_openingPairs_587_, lean_object* v_lambda_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings___redArg(v_fo_586_, v_openingPairs_587_, v_lambda_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings_spec__0(lean_object* v_EF_590_, lean_object* v___x_591_, lean_object* v_lambdaSq_592_, lean_object* v___x_593_, lean_object* v_lambda_594_, lean_object* v_x_595_, lean_object* v_x_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings_spec__0___redArg(v___x_591_, v_lambdaSq_592_, v___x_593_, v_lambda_594_, v_x_595_, v_x_596_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg(lean_object* v_fo_598_, lean_object* v_univariateRoundCoeffs_599_, lean_object* v_lSkip_600_){
_start:
{
lean_object* v_toRingOps_601_; lean_object* v_toSemiringOps_602_; lean_object* v_natCast_603_; lean_object* v_mul_604_; lean_object* v___x_605_; lean_object* v_omegaOrder_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; 
v_toRingOps_601_ = lean_ctor_get(v_fo_598_, 0);
v_toSemiringOps_602_ = lean_ctor_get(v_toRingOps_601_, 0);
v_natCast_603_ = lean_ctor_get(v_toSemiringOps_602_, 2);
lean_inc(v_natCast_603_);
v_mul_604_ = lean_ctor_get(v_toSemiringOps_602_, 4);
lean_inc(v_mul_604_);
v___x_605_ = lean_unsigned_to_nat(2u);
v_omegaOrder_606_ = lean_nat_pow(v___x_605_, v_lSkip_600_);
v___x_607_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Util_sumEveryNth___redArg(v_fo_598_, v_univariateRoundCoeffs_599_, v_omegaOrder_606_);
v___x_608_ = lean_apply_1(v_natCast_603_, v_omegaOrder_606_);
v___x_609_ = lean_apply_2(v_mul_604_, v___x_607_, v___x_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg___boxed(lean_object* v_fo_610_, lean_object* v_univariateRoundCoeffs_611_, lean_object* v_lSkip_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg(v_fo_610_, v_univariateRoundCoeffs_611_, v_lSkip_612_);
lean_dec(v_lSkip_612_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial(lean_object* v_EF_614_, lean_object* v_fo_615_, lean_object* v_univariateRoundCoeffs_616_, lean_object* v_lSkip_617_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg(v_fo_615_, v_univariateRoundCoeffs_616_, v_lSkip_617_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___boxed(lean_object* v_EF_619_, lean_object* v_fo_620_, lean_object* v_univariateRoundCoeffs_621_, lean_object* v_lSkip_622_){
_start:
{
lean_object* v_res_623_; 
v_res_623_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial(v_EF_619_, v_fo_620_, v_univariateRoundCoeffs_621_, v_lSkip_622_);
lean_dec(v_lSkip_622_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg(lean_object* v_inst_624_, lean_object* v_fo_625_, lean_object* v_columnOpenings_626_, lean_object* v_univariateRoundCoeffs_627_, lean_object* v_needRotPerCommit_628_, lean_object* v_lSkip_629_, lean_object* v_lambda_630_){
_start:
{
lean_object* v___x_631_; 
lean_inc_ref(v_fo_625_);
v___x_631_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_collectRound0OpeningPairs___redArg(v_fo_625_, v_columnOpenings_626_, v_needRotPerCommit_628_);
if (lean_obj_tag(v___x_631_) == 0)
{
lean_dec(v_lambda_630_);
lean_dec(v_univariateRoundCoeffs_627_);
lean_dec_ref(v_fo_625_);
lean_dec_ref(v_inst_624_);
return v___x_631_;
}
else
{
lean_object* v_a_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; uint8_t v___x_636_; 
v_a_632_ = lean_ctor_get(v___x_631_, 0);
lean_inc(v_a_632_);
lean_inc_ref(v_fo_625_);
v___x_633_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromOpenings___redArg(v_fo_625_, v_a_632_, v_lambda_630_);
v___x_634_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg(v_fo_625_, v_univariateRoundCoeffs_627_, v_lSkip_629_);
v___x_635_ = lean_apply_2(v_inst_624_, v___x_633_, v___x_634_);
v___x_636_ = lean_unbox(v___x_635_);
if (v___x_636_ == 0)
{
lean_object* v___x_637_; 
lean_dec_ref_known(v___x_631_, 1);
v___x_637_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_pairAdjacent___redArg___closed__1));
return v___x_637_;
}
else
{
return v___x_631_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg___boxed(lean_object* v_inst_638_, lean_object* v_fo_639_, lean_object* v_columnOpenings_640_, lean_object* v_univariateRoundCoeffs_641_, lean_object* v_needRotPerCommit_642_, lean_object* v_lSkip_643_, lean_object* v_lambda_644_){
_start:
{
lean_object* v_res_645_; 
v_res_645_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg(v_inst_638_, v_fo_639_, v_columnOpenings_640_, v_univariateRoundCoeffs_641_, v_needRotPerCommit_642_, v_lSkip_643_, v_lambda_644_);
lean_dec(v_lSkip_643_);
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim(lean_object* v_EF_646_, lean_object* v_inst_647_, lean_object* v_fo_648_, lean_object* v_columnOpenings_649_, lean_object* v_univariateRoundCoeffs_650_, lean_object* v_needRotPerCommit_651_, lean_object* v_lSkip_652_, lean_object* v_lambda_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg(v_inst_647_, v_fo_648_, v_columnOpenings_649_, v_univariateRoundCoeffs_650_, v_needRotPerCommit_651_, v_lSkip_652_, v_lambda_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___boxed(lean_object* v_EF_655_, lean_object* v_inst_656_, lean_object* v_fo_657_, lean_object* v_columnOpenings_658_, lean_object* v_univariateRoundCoeffs_659_, lean_object* v_needRotPerCommit_660_, lean_object* v_lSkip_661_, lean_object* v_lambda_662_){
_start:
{
lean_object* v_res_663_; 
v_res_663_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim(v_EF_655_, v_inst_656_, v_fo_657_, v_columnOpenings_658_, v_univariateRoundCoeffs_659_, v_needRotPerCommit_660_, v_lSkip_661_, v_lambda_662_);
lean_dec(v_lSkip_661_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg(lean_object* v_fo_664_, lean_object* v_univariateRoundCoeffs_665_, lean_object* v_lSkip_666_, lean_object* v_u0_667_, lean_object* v_openingPairs_668_){
_start:
{
lean_object* v_toRingOps_669_; lean_object* v_toSemiringOps_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
v_toRingOps_669_ = lean_ctor_get(v_fo_664_, 0);
v_toSemiringOps_670_ = lean_ctor_get(v_toRingOps_669_, 0);
lean_inc_ref(v_toSemiringOps_670_);
lean_inc(v_univariateRoundCoeffs_665_);
v___x_671_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeRound0ClaimFromPolynomial___redArg(v_fo_664_, v_univariateRoundCoeffs_665_, v_lSkip_666_);
lean_inc(v_u0_667_);
v___x_672_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_toSemiringOps_670_, v_univariateRoundCoeffs_665_, v_u0_667_);
v___x_673_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_673_, 0, v_openingPairs_668_);
lean_ctor_set(v___x_673_, 1, v___x_671_);
lean_ctor_set(v___x_673_, 2, v_u0_667_);
lean_ctor_set(v___x_673_, 3, v___x_672_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg___boxed(lean_object* v_fo_674_, lean_object* v_univariateRoundCoeffs_675_, lean_object* v_lSkip_676_, lean_object* v_u0_677_, lean_object* v_openingPairs_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg(v_fo_674_, v_univariateRoundCoeffs_675_, v_lSkip_676_, v_u0_677_, v_openingPairs_678_);
lean_dec(v_lSkip_676_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState(lean_object* v_EF_680_, lean_object* v_fo_681_, lean_object* v_univariateRoundCoeffs_682_, lean_object* v_lSkip_683_, lean_object* v_u0_684_, lean_object* v_openingPairs_685_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg(v_fo_681_, v_univariateRoundCoeffs_682_, v_lSkip_683_, v_u0_684_, v_openingPairs_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___boxed(lean_object* v_EF_687_, lean_object* v_fo_688_, lean_object* v_univariateRoundCoeffs_689_, lean_object* v_lSkip_690_, lean_object* v_u0_691_, lean_object* v_openingPairs_692_){
_start:
{
lean_object* v_res_693_; 
v_res_693_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState(v_EF_687_, v_fo_688_, v_univariateRoundCoeffs_689_, v_lSkip_690_, v_u0_691_, v_openingPairs_692_);
lean_dec(v_lSkip_690_);
return v_res_693_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg(lean_object* v_inst_694_, lean_object* v_x_695_, lean_object* v_x_696_){
_start:
{
lean_object* v_uPrism_697_; lean_object* v_claim_698_; lean_object* v_uPrism_699_; lean_object* v_claim_700_; uint8_t v___x_701_; 
v_uPrism_697_ = lean_ctor_get(v_x_695_, 0);
lean_inc(v_uPrism_697_);
v_claim_698_ = lean_ctor_get(v_x_695_, 1);
lean_inc(v_claim_698_);
lean_dec_ref(v_x_695_);
v_uPrism_699_ = lean_ctor_get(v_x_696_, 0);
lean_inc(v_uPrism_699_);
v_claim_700_ = lean_ctor_get(v_x_696_, 1);
lean_inc(v_claim_700_);
lean_dec_ref(v_x_696_);
lean_inc_ref(v_inst_694_);
v___x_701_ = l_instDecidableEqList___redArg(v_inst_694_, v_uPrism_697_, v_uPrism_699_);
if (v___x_701_ == 0)
{
lean_dec(v_claim_700_);
lean_dec(v_claim_698_);
lean_dec_ref(v_inst_694_);
return v___x_701_;
}
else
{
lean_object* v___x_702_; uint8_t v___x_703_; 
v___x_702_ = lean_apply_2(v_inst_694_, v_claim_698_, v_claim_700_);
v___x_703_ = lean_unbox(v___x_702_);
return v___x_703_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg___boxed(lean_object* v_inst_704_, lean_object* v_x_705_, lean_object* v_x_706_){
_start:
{
uint8_t v_res_707_; lean_object* v_r_708_; 
v_res_707_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg(v_inst_704_, v_x_705_, v_x_706_);
v_r_708_ = lean_box(v_res_707_);
return v_r_708_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq(lean_object* v_EF_709_, lean_object* v_inst_710_, lean_object* v_x_711_, lean_object* v_x_712_){
_start:
{
uint8_t v___x_713_; 
v___x_713_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg(v_inst_710_, v_x_711_, v_x_712_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___boxed(lean_object* v_EF_714_, lean_object* v_inst_715_, lean_object* v_x_716_, lean_object* v_x_717_){
_start:
{
uint8_t v_res_718_; lean_object* v_r_719_; 
v_res_718_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq(v_EF_714_, v_inst_715_, v_x_716_, v_x_717_);
v_r_719_ = lean_box(v_res_718_);
return v_r_719_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___redArg(lean_object* v_inst_720_, lean_object* v_x_721_, lean_object* v_x_722_){
_start:
{
uint8_t v___x_723_; 
v___x_723_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg(v_inst_720_, v_x_721_, v_x_722_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___redArg___boxed(lean_object* v_inst_724_, lean_object* v_x_725_, lean_object* v_x_726_){
_start:
{
uint8_t v_res_727_; lean_object* v_r_728_; 
v_res_727_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___redArg(v_inst_724_, v_x_725_, v_x_726_);
v_r_728_ = lean_box(v_res_727_);
return v_r_728_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState(lean_object* v_EF_729_, lean_object* v_inst_730_, lean_object* v_x_731_, lean_object* v_x_732_){
_start:
{
uint8_t v___x_733_; 
v___x_733_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState_decEq___redArg(v_inst_730_, v_x_731_, v_x_732_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState___boxed(lean_object* v_EF_734_, lean_object* v_inst_735_, lean_object* v_x_736_, lean_object* v_x_737_){
_start:
{
uint8_t v_res_738_; lean_object* v_r_739_; 
v_res_738_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instDecidableEqRoundState(v_EF_734_, v_inst_735_, v_x_736_, v_x_737_);
v_r_739_ = lean_box(v_res_738_);
return v_r_739_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg(lean_object* v_inst_743_, lean_object* v_x_744_){
_start:
{
lean_object* v_uPrism_745_; lean_object* v_claim_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_780_; 
v_uPrism_745_ = lean_ctor_get(v_x_744_, 0);
v_claim_746_ = lean_ctor_get(v_x_744_, 1);
v_isSharedCheck_780_ = !lean_is_exclusive(v_x_744_);
if (v_isSharedCheck_780_ == 0)
{
v___x_748_ = v_x_744_;
v_isShared_749_ = v_isSharedCheck_780_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_claim_746_);
lean_inc(v_uPrism_745_);
lean_dec(v_x_744_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_780_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_756_; 
v___x_750_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__5));
v___x_751_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__6));
v___x_752_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__7);
v___x_753_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_inst_743_);
v___x_754_ = l_List_repr___redArg(v_inst_743_, v_uPrism_745_);
if (v_isShared_749_ == 0)
{
lean_ctor_set_tag(v___x_748_, 4);
lean_ctor_set(v___x_748_, 1, v___x_754_);
lean_ctor_set(v___x_748_, 0, v___x_752_);
v___x_756_ = v___x_748_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_779_; 
v_reuseFailAlloc_779_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_779_, 0, v___x_752_);
lean_ctor_set(v_reuseFailAlloc_779_, 1, v___x_754_);
v___x_756_ = v_reuseFailAlloc_779_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
uint8_t v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; 
v___x_757_ = 0;
v___x_758_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_758_, 0, v___x_756_);
lean_ctor_set_uint8(v___x_758_, sizeof(void*)*1, v___x_757_);
v___x_759_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_759_, 0, v___x_751_);
lean_ctor_set(v___x_759_, 1, v___x_758_);
v___x_760_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__9));
v___x_761_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_761_, 0, v___x_759_);
lean_ctor_set(v___x_761_, 1, v___x_760_);
v___x_762_ = lean_box(1);
v___x_763_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_763_, 0, v___x_761_);
lean_ctor_set(v___x_763_, 1, v___x_762_);
v___x_764_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg___closed__1));
v___x_765_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_765_, 0, v___x_763_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
v___x_766_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_766_, 0, v___x_765_);
lean_ctor_set(v___x_766_, 1, v___x_750_);
v___x_767_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__12);
v___x_768_ = lean_apply_2(v_inst_743_, v_claim_746_, v___x_753_);
v___x_769_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_769_, 0, v___x_767_);
lean_ctor_set(v___x_769_, 1, v___x_768_);
v___x_770_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_770_, 0, v___x_769_);
lean_ctor_set_uint8(v___x_770_, sizeof(void*)*1, v___x_757_);
v___x_771_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_771_, 0, v___x_766_);
lean_ctor_set(v___x_771_, 1, v___x_770_);
v___x_772_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__15);
v___x_773_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__16));
v___x_774_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
lean_ctor_set(v___x_774_, 1, v___x_771_);
v___x_775_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprOutput_repr___redArg___closed__17));
v___x_776_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_776_, 0, v___x_774_);
lean_ctor_set(v___x_776_, 1, v___x_775_);
v___x_777_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_777_, 0, v___x_772_);
lean_ctor_set(v___x_777_, 1, v___x_776_);
v___x_778_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_778_, 0, v___x_777_);
lean_ctor_set_uint8(v___x_778_, sizeof(void*)*1, v___x_757_);
return v___x_778_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr(lean_object* v_EF_781_, lean_object* v_inst_782_, lean_object* v_x_783_, lean_object* v_prec_784_){
_start:
{
lean_object* v___x_785_; 
v___x_785_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___redArg(v_inst_782_, v_x_783_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___boxed(lean_object* v_EF_786_, lean_object* v_inst_787_, lean_object* v_x_788_, lean_object* v_prec_789_){
_start:
{
lean_object* v_res_790_; 
v_res_790_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr(v_EF_786_, v_inst_787_, v_x_788_, v_prec_789_);
lean_dec(v_prec_789_);
return v_res_790_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState___redArg(lean_object* v_inst_791_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___boxed), 4, 2);
lean_closure_set(v___x_792_, 0, lean_box(0));
lean_closure_set(v___x_792_, 1, v_inst_791_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState(lean_object* v_EF_793_, lean_object* v_inst_794_){
_start:
{
lean_object* v___x_795_; 
v___x_795_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_instReprRoundState_repr___boxed), 4, 2);
lean_closure_set(v___x_795_, 0, lean_box(0));
lean_closure_set(v___x_795_, 1, v_inst_794_);
return v___x_795_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg(lean_object* v_values_801_, lean_object* v_len_802_){
_start:
{
lean_object* v___x_803_; uint8_t v___x_804_; 
v___x_803_ = l_List_lengthTR___redArg(v_values_801_);
v___x_804_ = lean_nat_dec_le(v_len_802_, v___x_803_);
lean_dec(v___x_803_);
if (v___x_804_ == 0)
{
lean_object* v___x_805_; 
lean_dec(v_len_802_);
lean_dec(v_values_801_);
v___x_805_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_805_;
}
else
{
lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_806_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__1));
lean_inc(v_values_801_);
v___x_807_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_values_801_, v_values_801_, v_len_802_, v___x_806_);
lean_dec(v_values_801_);
v___x_808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_808_, 0, v___x_807_);
return v___x_808_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix(lean_object* v_00_u03b1_809_, lean_object* v_values_810_, lean_object* v_len_811_){
_start:
{
lean_object* v___x_812_; 
v___x_812_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg(v_values_810_, v_len_811_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg(lean_object* v_fo_813_, lean_object* v_coeffs_814_, lean_object* v_idx_815_, lean_object* v_delta_816_){
_start:
{
if (lean_obj_tag(v_coeffs_814_) == 0)
{
lean_object* v___x_817_; 
lean_dec(v_delta_816_);
lean_dec_ref(v_fo_813_);
v___x_817_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_817_;
}
else
{
lean_object* v_head_818_; lean_object* v_tail_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_847_; 
v_head_818_ = lean_ctor_get(v_coeffs_814_, 0);
v_tail_819_ = lean_ctor_get(v_coeffs_814_, 1);
v_isSharedCheck_847_ = !lean_is_exclusive(v_coeffs_814_);
if (v_isSharedCheck_847_ == 0)
{
v___x_821_ = v_coeffs_814_;
v_isShared_822_ = v_isSharedCheck_847_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_tail_819_);
lean_inc(v_head_818_);
lean_dec(v_coeffs_814_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_847_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v_zero_823_; uint8_t v_isZero_824_; 
v_zero_823_ = lean_unsigned_to_nat(0u);
v_isZero_824_ = lean_nat_dec_eq(v_idx_815_, v_zero_823_);
if (v_isZero_824_ == 1)
{
lean_object* v_toRingOps_825_; lean_object* v_toSemiringOps_826_; lean_object* v_add_827_; lean_object* v___x_828_; lean_object* v___x_830_; 
v_toRingOps_825_ = lean_ctor_get(v_fo_813_, 0);
lean_inc_ref(v_toRingOps_825_);
lean_dec_ref(v_fo_813_);
v_toSemiringOps_826_ = lean_ctor_get(v_toRingOps_825_, 0);
lean_inc_ref(v_toSemiringOps_826_);
lean_dec_ref(v_toRingOps_825_);
v_add_827_ = lean_ctor_get(v_toSemiringOps_826_, 3);
lean_inc(v_add_827_);
lean_dec_ref(v_toSemiringOps_826_);
v___x_828_ = lean_apply_2(v_add_827_, v_head_818_, v_delta_816_);
if (v_isShared_822_ == 0)
{
lean_ctor_set(v___x_821_, 0, v___x_828_);
v___x_830_ = v___x_821_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_832_; 
v_reuseFailAlloc_832_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_832_, 0, v___x_828_);
lean_ctor_set(v_reuseFailAlloc_832_, 1, v_tail_819_);
v___x_830_ = v_reuseFailAlloc_832_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
lean_object* v___x_831_; 
v___x_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_831_, 0, v___x_830_);
return v___x_831_;
}
}
else
{
lean_object* v_one_833_; lean_object* v_n_834_; lean_object* v___x_835_; 
v_one_833_ = lean_unsigned_to_nat(1u);
v_n_834_ = lean_nat_sub(v_idx_815_, v_one_833_);
v___x_835_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg(v_fo_813_, v_tail_819_, v_n_834_, v_delta_816_);
lean_dec(v_n_834_);
if (lean_obj_tag(v___x_835_) == 0)
{
lean_del_object(v___x_821_);
lean_dec(v_head_818_);
return v___x_835_;
}
else
{
lean_object* v_a_836_; lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_846_; 
v_a_836_ = lean_ctor_get(v___x_835_, 0);
v_isSharedCheck_846_ = !lean_is_exclusive(v___x_835_);
if (v_isSharedCheck_846_ == 0)
{
v___x_838_ = v___x_835_;
v_isShared_839_ = v_isSharedCheck_846_;
goto v_resetjp_837_;
}
else
{
lean_inc(v_a_836_);
lean_dec(v___x_835_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_846_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
lean_object* v___x_841_; 
if (v_isShared_822_ == 0)
{
lean_ctor_set(v___x_821_, 1, v_a_836_);
v___x_841_ = v___x_821_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_head_818_);
lean_ctor_set(v_reuseFailAlloc_845_, 1, v_a_836_);
v___x_841_ = v_reuseFailAlloc_845_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
lean_object* v___x_843_; 
if (v_isShared_839_ == 0)
{
lean_ctor_set(v___x_838_, 0, v___x_841_);
v___x_843_ = v___x_838_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v___x_841_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg___boxed(lean_object* v_fo_848_, lean_object* v_coeffs_849_, lean_object* v_idx_850_, lean_object* v_delta_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg(v_fo_848_, v_coeffs_849_, v_idx_850_, v_delta_851_);
lean_dec(v_idx_850_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt(lean_object* v_EF_853_, lean_object* v_fo_854_, lean_object* v_coeffs_855_, lean_object* v_idx_856_, lean_object* v_delta_857_){
_start:
{
lean_object* v___x_858_; 
v___x_858_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg(v_fo_854_, v_coeffs_855_, v_idx_856_, v_delta_857_);
return v___x_858_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___boxed(lean_object* v_EF_859_, lean_object* v_fo_860_, lean_object* v_coeffs_861_, lean_object* v_idx_862_, lean_object* v_delta_863_){
_start:
{
lean_object* v_res_864_; 
v_res_864_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt(v_EF_859_, v_fo_860_, v_coeffs_861_, v_idx_862_, v_delta_863_);
lean_dec(v_idx_862_);
return v_res_864_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0(lean_object* v_head_868_, lean_object* v_x_869_, lean_object* v_x_870_){
_start:
{
if (lean_obj_tag(v_x_870_) == 0)
{
lean_object* v___x_871_; 
v___x_871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_871_, 0, v_x_869_);
return v___x_871_;
}
else
{
lean_object* v_head_872_; lean_object* v_tail_873_; lean_object* v___x_875_; uint8_t v_isShared_876_; uint8_t v_isSharedCheck_897_; 
v_head_872_ = lean_ctor_get(v_x_870_, 0);
v_tail_873_ = lean_ctor_get(v_x_870_, 1);
v_isSharedCheck_897_ = !lean_is_exclusive(v_x_870_);
if (v_isSharedCheck_897_ == 0)
{
v___x_875_ = v_x_870_;
v_isShared_876_ = v_isSharedCheck_897_;
goto v_resetjp_874_;
}
else
{
lean_inc(v_tail_873_);
lean_inc(v_head_872_);
lean_dec(v_x_870_);
v___x_875_ = lean_box(0);
v_isShared_876_ = v_isSharedCheck_897_;
goto v_resetjp_874_;
}
v_resetjp_874_:
{
lean_object* v_fst_877_; lean_object* v_snd_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_896_; 
v_fst_877_ = lean_ctor_get(v_x_869_, 0);
v_snd_878_ = lean_ctor_get(v_x_869_, 1);
v_isSharedCheck_896_ = !lean_is_exclusive(v_x_869_);
if (v_isSharedCheck_896_ == 0)
{
v___x_880_ = v_x_869_;
v_isShared_881_ = v_isSharedCheck_896_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_snd_878_);
lean_inc(v_fst_877_);
lean_dec(v_x_869_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_896_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v_matrixIdx_882_; lean_object* v___x_883_; 
v_matrixIdx_882_ = lean_ctor_get(v_head_872_, 0);
lean_inc(v_matrixIdx_882_);
lean_dec(v_head_872_);
v___x_883_ = l_List_get_x3fInternal___redArg(v_head_868_, v_matrixIdx_882_);
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v___x_884_; 
lean_del_object(v___x_880_);
lean_dec(v_snd_878_);
lean_dec(v_fst_877_);
lean_del_object(v___x_875_);
lean_dec(v_tail_873_);
v___x_884_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0___closed__0));
return v___x_884_;
}
else
{
lean_object* v_val_885_; lean_object* v___x_887_; 
v_val_885_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_val_885_);
lean_dec_ref_known(v___x_883_, 1);
lean_inc(v_snd_878_);
if (v_isShared_881_ == 0)
{
lean_ctor_set(v___x_880_, 1, v_val_885_);
lean_ctor_set(v___x_880_, 0, v_snd_878_);
v___x_887_ = v___x_880_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v_snd_878_);
lean_ctor_set(v_reuseFailAlloc_895_, 1, v_val_885_);
v___x_887_ = v_reuseFailAlloc_895_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
lean_object* v___x_889_; 
if (v_isShared_876_ == 0)
{
lean_ctor_set(v___x_875_, 1, v_fst_877_);
lean_ctor_set(v___x_875_, 0, v___x_887_);
v___x_889_ = v___x_875_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v___x_887_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v_fst_877_);
v___x_889_ = v_reuseFailAlloc_894_;
goto v_reusejp_888_;
}
v_reusejp_888_:
{
lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_890_ = lean_unsigned_to_nat(1u);
v___x_891_ = lean_nat_add(v_snd_878_, v___x_890_);
lean_dec(v_snd_878_);
v___x_892_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_892_, 0, v___x_889_);
lean_ctor_set(v___x_892_, 1, v___x_891_);
v_x_869_ = v___x_892_;
v_x_870_ = v_tail_873_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0___boxed(lean_object* v_head_898_, lean_object* v_x_899_, lean_object* v_x_900_){
_start:
{
lean_object* v_res_901_; 
v_res_901_ = lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0(v_head_898_, v_x_899_, v_x_900_);
lean_dec(v_head_898_);
return v_res_901_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go(lean_object* v_layouts_905_, lean_object* v_needRot_906_, lean_object* v_nextIdx_907_){
_start:
{
if (lean_obj_tag(v_layouts_905_) == 0)
{
if (lean_obj_tag(v_needRot_906_) == 0)
{
lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; 
v___x_910_ = lean_box(0);
v___x_911_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_911_, 0, v___x_910_);
lean_ctor_set(v___x_911_, 1, v_nextIdx_907_);
v___x_912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_912_, 0, v___x_911_);
return v___x_912_;
}
else
{
lean_dec(v_nextIdx_907_);
lean_dec(v_needRot_906_);
goto v___jp_908_;
}
}
else
{
if (lean_obj_tag(v_needRot_906_) == 1)
{
lean_object* v_head_913_; lean_object* v_tail_914_; lean_object* v___x_916_; uint8_t v_isShared_917_; uint8_t v_isSharedCheck_961_; 
v_head_913_ = lean_ctor_get(v_layouts_905_, 0);
v_tail_914_ = lean_ctor_get(v_layouts_905_, 1);
v_isSharedCheck_961_ = !lean_is_exclusive(v_layouts_905_);
if (v_isSharedCheck_961_ == 0)
{
v___x_916_ = v_layouts_905_;
v_isShared_917_ = v_isSharedCheck_961_;
goto v_resetjp_915_;
}
else
{
lean_inc(v_tail_914_);
lean_inc(v_head_913_);
lean_dec(v_layouts_905_);
v___x_916_ = lean_box(0);
v_isShared_917_ = v_isSharedCheck_961_;
goto v_resetjp_915_;
}
v_resetjp_915_:
{
lean_object* v_head_918_; lean_object* v_tail_919_; lean_object* v___x_921_; uint8_t v_isShared_922_; uint8_t v_isSharedCheck_960_; 
v_head_918_ = lean_ctor_get(v_needRot_906_, 0);
v_tail_919_ = lean_ctor_get(v_needRot_906_, 1);
v_isSharedCheck_960_ = !lean_is_exclusive(v_needRot_906_);
if (v_isSharedCheck_960_ == 0)
{
v___x_921_ = v_needRot_906_;
v_isShared_922_ = v_isSharedCheck_960_;
goto v_resetjp_920_;
}
else
{
lean_inc(v_tail_919_);
lean_inc(v_head_918_);
lean_dec(v_needRot_906_);
v___x_921_ = lean_box(0);
v_isShared_922_ = v_isSharedCheck_960_;
goto v_resetjp_920_;
}
v_resetjp_920_:
{
lean_object* v_sortedCols_923_; lean_object* v_matStarts_924_; lean_object* v___x_925_; lean_object* v___x_926_; uint8_t v___x_927_; 
v_sortedCols_923_ = lean_ctor_get(v_head_913_, 3);
lean_inc(v_sortedCols_923_);
v_matStarts_924_ = lean_ctor_get(v_head_913_, 4);
lean_inc(v_matStarts_924_);
lean_dec(v_head_913_);
v___x_925_ = l_List_lengthTR___redArg(v_head_918_);
v___x_926_ = l_List_lengthTR___redArg(v_matStarts_924_);
lean_dec(v_matStarts_924_);
v___x_927_ = lean_nat_dec_eq(v___x_925_, v___x_926_);
lean_dec(v___x_926_);
lean_dec(v___x_925_);
if (v___x_927_ == 0)
{
lean_object* v___x_928_; 
lean_dec(v_sortedCols_923_);
lean_del_object(v___x_921_);
lean_dec(v_tail_919_);
lean_dec(v_head_918_);
lean_del_object(v___x_916_);
lean_dec(v_tail_914_);
lean_dec(v_nextIdx_907_);
v___x_928_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go___closed__0));
return v___x_928_;
}
else
{
lean_object* v___x_929_; lean_object* v___x_931_; 
v___x_929_ = lean_box(0);
if (v_isShared_917_ == 0)
{
lean_ctor_set_tag(v___x_916_, 0);
lean_ctor_set(v___x_916_, 1, v_nextIdx_907_);
lean_ctor_set(v___x_916_, 0, v___x_929_);
v___x_931_ = v___x_916_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_959_; 
v_reuseFailAlloc_959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_959_, 0, v___x_929_);
lean_ctor_set(v_reuseFailAlloc_959_, 1, v_nextIdx_907_);
v___x_931_ = v_reuseFailAlloc_959_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
lean_object* v___x_932_; 
v___x_932_ = lp_swirl_x2drbr_x2dformal_List_foldlM___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go_spec__0(v_head_918_, v___x_931_, v_sortedCols_923_);
lean_dec(v_head_918_);
if (lean_obj_tag(v___x_932_) == 0)
{
lean_object* v___x_933_; 
lean_dec_ref_known(v___x_932_, 1);
lean_del_object(v___x_921_);
lean_dec(v_tail_919_);
lean_dec(v_tail_914_);
v___x_933_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go___closed__0));
return v___x_933_;
}
else
{
lean_object* v_a_934_; lean_object* v_fst_935_; lean_object* v_snd_936_; lean_object* v___x_937_; 
v_a_934_ = lean_ctor_get(v___x_932_, 0);
lean_inc(v_a_934_);
lean_dec_ref_known(v___x_932_, 1);
v_fst_935_ = lean_ctor_get(v_a_934_, 0);
lean_inc(v_fst_935_);
v_snd_936_ = lean_ctor_get(v_a_934_, 1);
lean_inc(v_snd_936_);
lean_dec(v_a_934_);
v___x_937_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go(v_tail_914_, v_tail_919_, v_snd_936_);
if (lean_obj_tag(v___x_937_) == 0)
{
lean_dec(v_fst_935_);
lean_del_object(v___x_921_);
return v___x_937_;
}
else
{
lean_object* v_a_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_958_; 
v_a_938_ = lean_ctor_get(v___x_937_, 0);
v_isSharedCheck_958_ = !lean_is_exclusive(v___x_937_);
if (v_isSharedCheck_958_ == 0)
{
v___x_940_ = v___x_937_;
v_isShared_941_ = v_isSharedCheck_958_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_a_938_);
lean_dec(v___x_937_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_958_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v_fst_942_; lean_object* v_snd_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_957_; 
v_fst_942_ = lean_ctor_get(v_a_938_, 0);
v_snd_943_ = lean_ctor_get(v_a_938_, 1);
v_isSharedCheck_957_ = !lean_is_exclusive(v_a_938_);
if (v_isSharedCheck_957_ == 0)
{
v___x_945_ = v_a_938_;
v_isShared_946_ = v_isSharedCheck_957_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_snd_943_);
lean_inc(v_fst_942_);
lean_dec(v_a_938_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_957_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
lean_object* v___x_947_; lean_object* v___x_949_; 
v___x_947_ = l_List_reverse___redArg(v_fst_935_);
if (v_isShared_922_ == 0)
{
lean_ctor_set(v___x_921_, 1, v_fst_942_);
lean_ctor_set(v___x_921_, 0, v___x_947_);
v___x_949_ = v___x_921_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_956_, 1, v_fst_942_);
v___x_949_ = v_reuseFailAlloc_956_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
lean_object* v___x_951_; 
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 0, v___x_949_);
v___x_951_ = v___x_945_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v___x_949_);
lean_ctor_set(v_reuseFailAlloc_955_, 1, v_snd_943_);
v___x_951_ = v_reuseFailAlloc_955_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
lean_object* v___x_953_; 
if (v_isShared_941_ == 0)
{
lean_ctor_set(v___x_940_, 0, v___x_951_);
v___x_953_ = v___x_940_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v___x_951_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
return v___x_953_;
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
lean_dec_ref_known(v_layouts_905_, 2);
lean_dec(v_nextIdx_907_);
lean_dec(v_needRot_906_);
goto v___jp_908_;
}
}
v___jp_908_:
{
lean_object* v___x_909_; 
v___x_909_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go___closed__0));
return v___x_909_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData(lean_object* v_layouts_962_, lean_object* v_needRotPerCommit_963_){
_start:
{
lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_964_ = lean_unsigned_to_nat(0u);
v___x_965_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData_go(v_layouts_962_, v_needRotPerCommit_963_, v___x_964_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg(lean_object* v_fo_966_, lean_object* v_state_967_, lean_object* v_roundPoly_968_, lean_object* v_u_969_){
_start:
{
lean_object* v_toRingOps_970_; lean_object* v_sub_971_; lean_object* v___x_973_; uint8_t v_isShared_974_; uint8_t v_isSharedCheck_995_; 
v_toRingOps_970_ = lean_ctor_get(v_fo_966_, 0);
lean_inc_ref(v_toRingOps_970_);
v_sub_971_ = lean_ctor_get(v_toRingOps_970_, 1);
v_isSharedCheck_995_ = !lean_is_exclusive(v_toRingOps_970_);
if (v_isSharedCheck_995_ == 0)
{
lean_object* v_unused_996_; 
v_unused_996_ = lean_ctor_get(v_toRingOps_970_, 0);
lean_dec(v_unused_996_);
v___x_973_ = v_toRingOps_970_;
v_isShared_974_ = v_isSharedCheck_995_;
goto v_resetjp_972_;
}
else
{
lean_inc(v_sub_971_);
lean_dec(v_toRingOps_970_);
v___x_973_ = lean_box(0);
v_isShared_974_ = v_isSharedCheck_995_;
goto v_resetjp_972_;
}
v_resetjp_972_:
{
lean_object* v_uPrism_975_; lean_object* v_claim_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_994_; 
v_uPrism_975_ = lean_ctor_get(v_state_967_, 0);
v_claim_976_ = lean_ctor_get(v_state_967_, 1);
v_isSharedCheck_994_ = !lean_is_exclusive(v_state_967_);
if (v_isSharedCheck_994_ == 0)
{
v___x_978_ = v_state_967_;
v_isShared_979_ = v_isSharedCheck_994_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_claim_976_);
lean_inc(v_uPrism_975_);
lean_dec(v_state_967_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_994_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_980_; lean_object* v_s1_981_; lean_object* v___x_982_; lean_object* v_s2_983_; lean_object* v_s0_984_; lean_object* v___x_985_; lean_object* v___x_987_; 
v___x_980_ = lean_unsigned_to_nat(0u);
v_s1_981_ = lean_array_fget_borrowed(v_roundPoly_968_, v___x_980_);
v___x_982_ = lean_unsigned_to_nat(1u);
v_s2_983_ = lean_array_fget_borrowed(v_roundPoly_968_, v___x_982_);
lean_inc(v_s1_981_);
v_s0_984_ = lean_apply_2(v_sub_971_, v_claim_976_, v_s1_981_);
v___x_985_ = lean_box(0);
lean_inc(v_u_969_);
if (v_isShared_974_ == 0)
{
lean_ctor_set_tag(v___x_973_, 1);
lean_ctor_set(v___x_973_, 1, v___x_985_);
lean_ctor_set(v___x_973_, 0, v_u_969_);
v___x_987_ = v___x_973_;
goto v_reusejp_986_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v_u_969_);
lean_ctor_set(v_reuseFailAlloc_993_, 1, v___x_985_);
v___x_987_ = v_reuseFailAlloc_993_;
goto v_reusejp_986_;
}
v_reusejp_986_:
{
lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_991_; 
v___x_988_ = l_List_appendTR___redArg(v_uPrism_975_, v___x_987_);
lean_inc(v_s2_983_);
lean_inc(v_s1_981_);
v___x_989_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(v_fo_966_, v_s0_984_, v_s1_981_, v_s2_983_, v_u_969_);
if (v_isShared_979_ == 0)
{
lean_ctor_set(v___x_978_, 1, v___x_989_);
lean_ctor_set(v___x_978_, 0, v___x_988_);
v___x_991_ = v___x_978_;
goto v_reusejp_990_;
}
else
{
lean_object* v_reuseFailAlloc_992_; 
v_reuseFailAlloc_992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_992_, 0, v___x_988_);
lean_ctor_set(v_reuseFailAlloc_992_, 1, v___x_989_);
v___x_991_ = v_reuseFailAlloc_992_;
goto v_reusejp_990_;
}
v_reusejp_990_:
{
return v___x_991_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg___boxed(lean_object* v_fo_997_, lean_object* v_state_998_, lean_object* v_roundPoly_999_, lean_object* v_u_1000_){
_start:
{
lean_object* v_res_1001_; 
v_res_1001_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg(v_fo_997_, v_state_998_, v_roundPoly_999_, v_u_1000_);
lean_dec_ref(v_roundPoly_999_);
return v_res_1001_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound(lean_object* v_EF_1002_, lean_object* v_fo_1003_, lean_object* v_state_1004_, lean_object* v_roundPoly_1005_, lean_object* v_u_1006_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg(v_fo_1003_, v_state_1004_, v_roundPoly_1005_, v_u_1006_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___boxed(lean_object* v_EF_1008_, lean_object* v_fo_1009_, lean_object* v_state_1010_, lean_object* v_roundPoly_1011_, lean_object* v_u_1012_){
_start:
{
lean_object* v_res_1013_; 
v_res_1013_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound(v_EF_1008_, v_fo_1009_, v_state_1010_, v_roundPoly_1011_, v_u_1012_);
lean_dec_ref(v_roundPoly_1011_);
return v_res_1013_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___lam__0(lean_object* v_inst_1014_, lean_object* v_inst_1015_, lean_object* v_fo_1016_, lean_object* v_state_1017_, lean_object* v_roundPoly_1018_, lean_object* v___y_1019_){
_start:
{
lean_object* v___x_1020_; lean_object* v_s1_1021_; lean_object* v___x_1022_; lean_object* v_a_1023_; lean_object* v_snd_1024_; lean_object* v___x_1025_; lean_object* v_s2_1026_; lean_object* v___x_1027_; lean_object* v_a_1028_; lean_object* v_snd_1029_; lean_object* v___x_1030_; lean_object* v_a_1031_; lean_object* v___x_1033_; uint8_t v_isShared_1034_; uint8_t v_isSharedCheck_1048_; 
v___x_1020_ = lean_unsigned_to_nat(0u);
v_s1_1021_ = lean_array_fget_borrowed(v_roundPoly_1018_, v___x_1020_);
lean_inc(v_s1_1021_);
lean_inc_ref_n(v_inst_1015_, 2);
lean_inc_ref_n(v_inst_1014_, 2);
v___x_1022_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_1014_, v_inst_1015_, v_s1_1021_, v___y_1019_);
v_a_1023_ = lean_ctor_get(v___x_1022_, 0);
lean_inc(v_a_1023_);
lean_dec_ref(v___x_1022_);
v_snd_1024_ = lean_ctor_get(v_a_1023_, 1);
lean_inc(v_snd_1024_);
lean_dec(v_a_1023_);
v___x_1025_ = lean_unsigned_to_nat(1u);
v_s2_1026_ = lean_array_fget_borrowed(v_roundPoly_1018_, v___x_1025_);
lean_inc(v_s2_1026_);
v___x_1027_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_1014_, v_inst_1015_, v_s2_1026_, v_snd_1024_);
v_a_1028_ = lean_ctor_get(v___x_1027_, 0);
lean_inc(v_a_1028_);
lean_dec_ref(v___x_1027_);
v_snd_1029_ = lean_ctor_get(v_a_1028_, 1);
lean_inc(v_snd_1029_);
lean_dec(v_a_1028_);
v___x_1030_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_1014_, v_inst_1015_, v_snd_1029_);
v_a_1031_ = lean_ctor_get(v___x_1030_, 0);
v_isSharedCheck_1048_ = !lean_is_exclusive(v___x_1030_);
if (v_isSharedCheck_1048_ == 0)
{
v___x_1033_ = v___x_1030_;
v_isShared_1034_ = v_isSharedCheck_1048_;
goto v_resetjp_1032_;
}
else
{
lean_inc(v_a_1031_);
lean_dec(v___x_1030_);
v___x_1033_ = lean_box(0);
v_isShared_1034_ = v_isSharedCheck_1048_;
goto v_resetjp_1032_;
}
v_resetjp_1032_:
{
lean_object* v_fst_1035_; lean_object* v_snd_1036_; lean_object* v___x_1038_; uint8_t v_isShared_1039_; uint8_t v_isSharedCheck_1047_; 
v_fst_1035_ = lean_ctor_get(v_a_1031_, 0);
v_snd_1036_ = lean_ctor_get(v_a_1031_, 1);
v_isSharedCheck_1047_ = !lean_is_exclusive(v_a_1031_);
if (v_isSharedCheck_1047_ == 0)
{
v___x_1038_ = v_a_1031_;
v_isShared_1039_ = v_isSharedCheck_1047_;
goto v_resetjp_1037_;
}
else
{
lean_inc(v_snd_1036_);
lean_inc(v_fst_1035_);
lean_dec(v_a_1031_);
v___x_1038_ = lean_box(0);
v_isShared_1039_ = v_isSharedCheck_1047_;
goto v_resetjp_1037_;
}
v_resetjp_1037_:
{
lean_object* v___x_1040_; lean_object* v___x_1042_; 
v___x_1040_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRound___redArg(v_fo_1016_, v_state_1017_, v_roundPoly_1018_, v_fst_1035_);
if (v_isShared_1039_ == 0)
{
lean_ctor_set(v___x_1038_, 0, v___x_1040_);
v___x_1042_ = v___x_1038_;
goto v_reusejp_1041_;
}
else
{
lean_object* v_reuseFailAlloc_1046_; 
v_reuseFailAlloc_1046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1046_, 0, v___x_1040_);
lean_ctor_set(v_reuseFailAlloc_1046_, 1, v_snd_1036_);
v___x_1042_ = v_reuseFailAlloc_1046_;
goto v_reusejp_1041_;
}
v_reusejp_1041_:
{
lean_object* v___x_1044_; 
if (v_isShared_1034_ == 0)
{
lean_ctor_set(v___x_1033_, 0, v___x_1042_);
v___x_1044_ = v___x_1033_;
goto v_reusejp_1043_;
}
else
{
lean_object* v_reuseFailAlloc_1045_; 
v_reuseFailAlloc_1045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1045_, 0, v___x_1042_);
v___x_1044_ = v_reuseFailAlloc_1045_;
goto v_reusejp_1043_;
}
v_reusejp_1043_:
{
return v___x_1044_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___lam__0___boxed(lean_object* v_inst_1049_, lean_object* v_inst_1050_, lean_object* v_fo_1051_, lean_object* v_state_1052_, lean_object* v_roundPoly_1053_, lean_object* v___y_1054_){
_start:
{
lean_object* v_res_1055_; 
v_res_1055_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___lam__0(v_inst_1049_, v_inst_1050_, v_fo_1051_, v_state_1052_, v_roundPoly_1053_, v___y_1054_);
lean_dec_ref(v_roundPoly_1053_);
return v_res_1055_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg(lean_object* v_inst_1101_, lean_object* v_inst_1102_, lean_object* v_fo_1103_, lean_object* v_roundPolys_1104_, lean_object* v_initial_1105_, lean_object* v_a_1106_){
_start:
{
lean_object* v___f_1107_; lean_object* v___x_1108_; lean_object* v___x_41__overap_1109_; lean_object* v___x_1110_; 
v___f_1107_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___lam__0___boxed), 6, 3);
lean_closure_set(v___f_1107_, 0, v_inst_1101_);
lean_closure_set(v___f_1107_, 1, v_inst_1102_);
lean_closure_set(v___f_1107_, 2, v_fo_1103_);
v___x_1108_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__19));
v___x_41__overap_1109_ = l_List_foldlM___redArg(v___x_1108_, v___f_1107_, v_initial_1105_, v_roundPolys_1104_);
v___x_1110_ = lean_apply_1(v___x_41__overap_1109_, v_a_1106_);
return v___x_1110_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM(lean_object* v_F_1111_, lean_object* v_EF_1112_, lean_object* v_inst_1113_, lean_object* v_inst_1114_, lean_object* v_fo_1115_, lean_object* v_roundPolys_1116_, lean_object* v_initial_1117_, lean_object* v_a_1118_){
_start:
{
lean_object* v___x_1119_; 
v___x_1119_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg(v_inst_1113_, v_inst_1114_, v_fo_1115_, v_roundPolys_1116_, v_initial_1117_, v_a_1118_);
return v___x_1119_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg(lean_object* v_lSkip_1120_, lean_object* v_nLift_1121_, lean_object* v_rowIdx_1122_, lean_object* v_fo_1123_, lean_object* v_a_1124_, lean_object* v_a_1125_){
_start:
{
if (lean_obj_tag(v_a_1124_) == 0)
{
lean_object* v___x_1126_; 
lean_dec_ref(v_fo_1123_);
v___x_1126_ = l_List_reverse___redArg(v_a_1125_);
return v___x_1126_;
}
else
{
lean_object* v_head_1127_; lean_object* v_tail_1128_; lean_object* v___x_1130_; uint8_t v_isShared_1131_; uint8_t v_isSharedCheck_1147_; 
v_head_1127_ = lean_ctor_get(v_a_1124_, 0);
v_tail_1128_ = lean_ctor_get(v_a_1124_, 1);
v_isSharedCheck_1147_ = !lean_is_exclusive(v_a_1124_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1130_ = v_a_1124_;
v_isShared_1131_ = v_isSharedCheck_1147_;
goto v_resetjp_1129_;
}
else
{
lean_inc(v_tail_1128_);
lean_inc(v_head_1127_);
lean_dec(v_a_1124_);
v___x_1130_ = lean_box(0);
v_isShared_1131_ = v_isSharedCheck_1147_;
goto v_resetjp_1129_;
}
v_resetjp_1129_:
{
lean_object* v___y_1133_; lean_object* v___x_1138_; lean_object* v___x_1139_; uint8_t v___x_1140_; 
v___x_1138_ = lean_nat_add(v_lSkip_1120_, v_nLift_1121_);
v___x_1139_ = lean_nat_add(v___x_1138_, v_head_1127_);
lean_dec(v_head_1127_);
lean_dec(v___x_1138_);
v___x_1140_ = l_Nat_testBit(v_rowIdx_1122_, v___x_1139_);
lean_dec(v___x_1139_);
if (v___x_1140_ == 0)
{
lean_object* v_toRingOps_1141_; lean_object* v_toSemiringOps_1142_; lean_object* v_zero_1143_; 
v_toRingOps_1141_ = lean_ctor_get(v_fo_1123_, 0);
v_toSemiringOps_1142_ = lean_ctor_get(v_toRingOps_1141_, 0);
v_zero_1143_ = lean_ctor_get(v_toSemiringOps_1142_, 0);
lean_inc(v_zero_1143_);
v___y_1133_ = v_zero_1143_;
goto v___jp_1132_;
}
else
{
lean_object* v_toRingOps_1144_; lean_object* v_toSemiringOps_1145_; lean_object* v_one_1146_; 
v_toRingOps_1144_ = lean_ctor_get(v_fo_1123_, 0);
v_toSemiringOps_1145_ = lean_ctor_get(v_toRingOps_1144_, 0);
v_one_1146_ = lean_ctor_get(v_toSemiringOps_1145_, 1);
lean_inc(v_one_1146_);
v___y_1133_ = v_one_1146_;
goto v___jp_1132_;
}
v___jp_1132_:
{
lean_object* v___x_1135_; 
if (v_isShared_1131_ == 0)
{
lean_ctor_set(v___x_1130_, 1, v_a_1125_);
lean_ctor_set(v___x_1130_, 0, v___y_1133_);
v___x_1135_ = v___x_1130_;
goto v_reusejp_1134_;
}
else
{
lean_object* v_reuseFailAlloc_1137_; 
v_reuseFailAlloc_1137_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1137_, 0, v___y_1133_);
lean_ctor_set(v_reuseFailAlloc_1137_, 1, v_a_1125_);
v___x_1135_ = v_reuseFailAlloc_1137_;
goto v_reusejp_1134_;
}
v_reusejp_1134_:
{
v_a_1124_ = v_tail_1128_;
v_a_1125_ = v___x_1135_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg___boxed(lean_object* v_lSkip_1148_, lean_object* v_nLift_1149_, lean_object* v_rowIdx_1150_, lean_object* v_fo_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_){
_start:
{
lean_object* v_res_1154_; 
v_res_1154_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg(v_lSkip_1148_, v_nLift_1149_, v_rowIdx_1150_, v_fo_1151_, v_a_1152_, v_a_1153_);
lean_dec(v_rowIdx_1150_);
lean_dec(v_nLift_1149_);
lean_dec(v_lSkip_1148_);
return v_res_1154_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg(lean_object* v_fo_1155_, lean_object* v_lSkip_1156_, lean_object* v_nLift_1157_, lean_object* v_nStack_1158_, lean_object* v_rowIdx_1159_){
_start:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1160_ = lean_nat_sub(v_nStack_1158_, v_nLift_1157_);
v___x_1161_ = l_List_range(v___x_1160_);
v___x_1162_ = lean_box(0);
v___x_1163_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg(v_lSkip_1156_, v_nLift_1157_, v_rowIdx_1159_, v_fo_1155_, v___x_1161_, v___x_1162_);
return v___x_1163_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg___boxed(lean_object* v_fo_1164_, lean_object* v_lSkip_1165_, lean_object* v_nLift_1166_, lean_object* v_nStack_1167_, lean_object* v_rowIdx_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg(v_fo_1164_, v_lSkip_1165_, v_nLift_1166_, v_nStack_1167_, v_rowIdx_1168_);
lean_dec(v_rowIdx_1168_);
lean_dec(v_nStack_1167_);
lean_dec(v_nLift_1166_);
lean_dec(v_lSkip_1165_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits(lean_object* v_EF_1170_, lean_object* v_fo_1171_, lean_object* v_lSkip_1172_, lean_object* v_nLift_1173_, lean_object* v_nStack_1174_, lean_object* v_rowIdx_1175_){
_start:
{
lean_object* v___x_1176_; 
v___x_1176_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg(v_fo_1171_, v_lSkip_1172_, v_nLift_1173_, v_nStack_1174_, v_rowIdx_1175_);
return v___x_1176_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___boxed(lean_object* v_EF_1177_, lean_object* v_fo_1178_, lean_object* v_lSkip_1179_, lean_object* v_nLift_1180_, lean_object* v_nStack_1181_, lean_object* v_rowIdx_1182_){
_start:
{
lean_object* v_res_1183_; 
v_res_1183_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits(v_EF_1177_, v_fo_1178_, v_lSkip_1179_, v_nLift_1180_, v_nStack_1181_, v_rowIdx_1182_);
lean_dec(v_rowIdx_1182_);
lean_dec(v_nStack_1181_);
lean_dec(v_nLift_1180_);
lean_dec(v_lSkip_1179_);
return v_res_1183_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0(lean_object* v_EF_1184_, lean_object* v_lSkip_1185_, lean_object* v_nLift_1186_, lean_object* v_rowIdx_1187_, lean_object* v_fo_1188_, lean_object* v_a_1189_, lean_object* v_a_1190_){
_start:
{
lean_object* v___x_1191_; 
v___x_1191_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___redArg(v_lSkip_1185_, v_nLift_1186_, v_rowIdx_1187_, v_fo_1188_, v_a_1189_, v_a_1190_);
return v___x_1191_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0___boxed(lean_object* v_EF_1192_, lean_object* v_lSkip_1193_, lean_object* v_nLift_1194_, lean_object* v_rowIdx_1195_, lean_object* v_fo_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_){
_start:
{
lean_object* v_res_1199_; 
v_res_1199_ = lp_swirl_x2drbr_x2dformal_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits_spec__0(v_EF_1192_, v_lSkip_1193_, v_nLift_1194_, v_rowIdx_1195_, v_fo_1196_, v_a_1197_, v_a_1198_);
lean_dec(v_rowIdx_1195_);
lean_dec(v_nLift_1194_);
lean_dec(v_lSkip_1193_);
return v_res_1199_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0(void){
_start:
{
lean_object* v___x_1200_; lean_object* v___x_1201_; 
v___x_1200_ = lean_unsigned_to_nat(0u);
v___x_1201_ = lean_nat_to_int(v___x_1200_);
return v___x_1201_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg(lean_object* v_fo_1205_, lean_object* v_r_1206_, lean_object* v_lSkip_1207_, lean_object* v_n_1208_, lean_object* v_nLift_1209_){
_start:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; uint8_t v___x_1212_; 
v___x_1210_ = lean_unsigned_to_nat(0u);
v___x_1211_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0);
v___x_1212_ = lean_int_dec_lt(v_n_1208_, v___x_1211_);
if (v___x_1212_ == 0)
{
lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; 
lean_dec_ref(v_fo_1205_);
v___x_1213_ = lean_unsigned_to_nat(1u);
v___x_1214_ = lean_nat_add(v_nLift_1209_, v___x_1213_);
v___x_1215_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg(v_r_1206_, v___x_1214_);
if (lean_obj_tag(v___x_1215_) == 0)
{
lean_object* v___x_1216_; 
lean_dec_ref_known(v___x_1215_, 1);
lean_dec(v_lSkip_1207_);
v___x_1216_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__1));
return v___x_1216_;
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1225_; 
v_a_1217_ = lean_ctor_get(v___x_1215_, 0);
v_isSharedCheck_1225_ = !lean_is_exclusive(v___x_1215_);
if (v_isSharedCheck_1225_ == 0)
{
v___x_1219_ = v___x_1215_;
v_isShared_1220_ = v_isSharedCheck_1225_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1215_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1225_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1221_; lean_object* v___x_1223_; 
v___x_1221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1221_, 0, v_lSkip_1207_);
lean_ctor_set(v___x_1221_, 1, v_a_1217_);
if (v_isShared_1220_ == 0)
{
lean_ctor_set(v___x_1219_, 0, v___x_1221_);
v___x_1223_ = v___x_1219_;
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
else
{
lean_object* v___x_1226_; 
v___x_1226_ = l_List_get_x3fInternal___redArg(v_r_1206_, v___x_1210_);
lean_dec(v_r_1206_);
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_object* v___x_1227_; 
lean_dec(v_lSkip_1207_);
lean_dec_ref(v_fo_1205_);
v___x_1227_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__1));
return v___x_1227_;
}
else
{
lean_object* v_toRingOps_1228_; lean_object* v___x_1230_; uint8_t v_isShared_1231_; uint8_t v_isSharedCheck_1258_; 
v_toRingOps_1228_ = lean_ctor_get(v_fo_1205_, 0);
v_isSharedCheck_1258_ = !lean_is_exclusive(v_fo_1205_);
if (v_isSharedCheck_1258_ == 0)
{
lean_object* v_unused_1259_; 
v_unused_1259_ = lean_ctor_get(v_fo_1205_, 1);
lean_dec(v_unused_1259_);
v___x_1230_ = v_fo_1205_;
v_isShared_1231_ = v_isSharedCheck_1258_;
goto v_resetjp_1229_;
}
else
{
lean_inc(v_toRingOps_1228_);
lean_dec(v_fo_1205_);
v___x_1230_ = lean_box(0);
v_isShared_1231_ = v_isSharedCheck_1258_;
goto v_resetjp_1229_;
}
v_resetjp_1229_:
{
lean_object* v_val_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1257_; 
v_val_1232_ = lean_ctor_get(v___x_1226_, 0);
v_isSharedCheck_1257_ = !lean_is_exclusive(v___x_1226_);
if (v_isSharedCheck_1257_ == 0)
{
v___x_1234_ = v___x_1226_;
v_isShared_1235_ = v_isSharedCheck_1257_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_val_1232_);
lean_dec(v___x_1226_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1257_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v_toSemiringOps_1236_; lean_object* v___x_1238_; uint8_t v_isShared_1239_; uint8_t v_isSharedCheck_1255_; 
v_toSemiringOps_1236_ = lean_ctor_get(v_toRingOps_1228_, 0);
v_isSharedCheck_1255_ = !lean_is_exclusive(v_toRingOps_1228_);
if (v_isSharedCheck_1255_ == 0)
{
lean_object* v_unused_1256_; 
v_unused_1256_ = lean_ctor_get(v_toRingOps_1228_, 1);
lean_dec(v_unused_1256_);
v___x_1238_ = v_toRingOps_1228_;
v_isShared_1239_ = v_isSharedCheck_1255_;
goto v_resetjp_1237_;
}
else
{
lean_inc(v_toSemiringOps_1236_);
lean_dec(v_toRingOps_1228_);
v___x_1238_ = lean_box(0);
v_isShared_1239_ = v_isSharedCheck_1255_;
goto v_resetjp_1237_;
}
v_resetjp_1237_:
{
lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1247_; 
v___x_1240_ = lean_nat_to_int(v_lSkip_1207_);
v___x_1241_ = lean_int_add(v___x_1240_, v_n_1208_);
lean_dec(v___x_1240_);
v___x_1242_ = l_Int_toNat(v___x_1241_);
lean_dec(v___x_1241_);
v___x_1243_ = lean_nat_abs(v_n_1208_);
v___x_1244_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_toSemiringOps_1236_, v_val_1232_, v___x_1243_);
lean_dec(v___x_1243_);
lean_dec(v_val_1232_);
v___x_1245_ = lean_box(0);
if (v_isShared_1239_ == 0)
{
lean_ctor_set_tag(v___x_1238_, 1);
lean_ctor_set(v___x_1238_, 1, v___x_1245_);
lean_ctor_set(v___x_1238_, 0, v___x_1244_);
v___x_1247_ = v___x_1238_;
goto v_reusejp_1246_;
}
else
{
lean_object* v_reuseFailAlloc_1254_; 
v_reuseFailAlloc_1254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1254_, 0, v___x_1244_);
lean_ctor_set(v_reuseFailAlloc_1254_, 1, v___x_1245_);
v___x_1247_ = v_reuseFailAlloc_1254_;
goto v_reusejp_1246_;
}
v_reusejp_1246_:
{
lean_object* v___x_1249_; 
if (v_isShared_1231_ == 0)
{
lean_ctor_set(v___x_1230_, 1, v___x_1247_);
lean_ctor_set(v___x_1230_, 0, v___x_1242_);
v___x_1249_ = v___x_1230_;
goto v_reusejp_1248_;
}
else
{
lean_object* v_reuseFailAlloc_1253_; 
v_reuseFailAlloc_1253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1253_, 0, v___x_1242_);
lean_ctor_set(v_reuseFailAlloc_1253_, 1, v___x_1247_);
v___x_1249_ = v_reuseFailAlloc_1253_;
goto v_reusejp_1248_;
}
v_reusejp_1248_:
{
lean_object* v___x_1251_; 
if (v_isShared_1235_ == 0)
{
lean_ctor_set(v___x_1234_, 0, v___x_1249_);
v___x_1251_ = v___x_1234_;
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
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___boxed(lean_object* v_fo_1260_, lean_object* v_r_1261_, lean_object* v_lSkip_1262_, lean_object* v_n_1263_, lean_object* v_nLift_1264_){
_start:
{
lean_object* v_res_1265_; 
v_res_1265_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg(v_fo_1260_, v_r_1261_, v_lSkip_1262_, v_n_1263_, v_nLift_1264_);
lean_dec(v_nLift_1264_);
lean_dec(v_n_1263_);
return v_res_1265_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice(lean_object* v_EF_1266_, lean_object* v_fo_1267_, lean_object* v_r_1268_, lean_object* v_lSkip_1269_, lean_object* v_n_1270_, lean_object* v_nLift_1271_){
_start:
{
lean_object* v___x_1272_; 
v___x_1272_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg(v_fo_1267_, v_r_1268_, v_lSkip_1269_, v_n_1270_, v_nLift_1271_);
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___boxed(lean_object* v_EF_1273_, lean_object* v_fo_1274_, lean_object* v_r_1275_, lean_object* v_lSkip_1276_, lean_object* v_n_1277_, lean_object* v_nLift_1278_){
_start:
{
lean_object* v_res_1279_; 
v_res_1279_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice(v_EF_1273_, v_fo_1274_, v_r_1275_, v_lSkip_1276_, v_n_1277_, v_nLift_1278_);
lean_dec(v_nLift_1278_);
lean_dec(v_n_1277_);
return v_res_1279_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg___lam__0(lean_object* v_pow_1280_, lean_object* v_lambdaSq_1281_, lean_object* v_mul_1282_, lean_object* v_fo_1283_, lean_object* v_lSkip_1284_, lean_object* v_uPrism_1285_, lean_object* v_nStack_1286_, lean_object* v_head_1287_, lean_object* v_r_1288_, lean_object* v_toRingOps_1289_, lean_object* v_inst_1290_, lean_object* v_lambda_1291_, lean_object* v_add_1292_, lean_object* v_coeffs_1293_, lean_object* v_entry_1294_){
_start:
{
lean_object* v_snd_1295_; lean_object* v_fst_1296_; lean_object* v_slice_1297_; lean_object* v_fst_1298_; lean_object* v_snd_1299_; lean_object* v_colIdx_1300_; lean_object* v_rowIdx_1301_; lean_object* v_logHeight_1302_; lean_object* v___y_1304_; lean_object* v___y_1305_; lean_object* v___y_1306_; lean_object* v_lambdaSqPow_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v_n_1313_; lean_object* v___y_1315_; lean_object* v___x_1342_; uint8_t v___x_1343_; 
v_snd_1295_ = lean_ctor_get(v_entry_1294_, 1);
lean_inc(v_snd_1295_);
v_fst_1296_ = lean_ctor_get(v_entry_1294_, 0);
lean_inc(v_fst_1296_);
lean_dec_ref(v_entry_1294_);
v_slice_1297_ = lean_ctor_get(v_fst_1296_, 2);
lean_inc_ref(v_slice_1297_);
lean_dec(v_fst_1296_);
v_fst_1298_ = lean_ctor_get(v_snd_1295_, 0);
lean_inc(v_fst_1298_);
v_snd_1299_ = lean_ctor_get(v_snd_1295_, 1);
lean_inc(v_snd_1299_);
lean_dec(v_snd_1295_);
v_colIdx_1300_ = lean_ctor_get(v_slice_1297_, 0);
lean_inc(v_colIdx_1300_);
v_rowIdx_1301_ = lean_ctor_get(v_slice_1297_, 1);
lean_inc(v_rowIdx_1301_);
v_logHeight_1302_ = lean_ctor_get(v_slice_1297_, 2);
lean_inc(v_logHeight_1302_);
lean_dec_ref(v_slice_1297_);
v_lambdaSqPow_1310_ = lean_apply_2(v_pow_1280_, v_lambdaSq_1281_, v_fst_1298_);
v___x_1311_ = lean_nat_to_int(v_logHeight_1302_);
lean_inc(v_lSkip_1284_);
v___x_1312_ = lean_nat_to_int(v_lSkip_1284_);
v_n_1313_ = lean_int_sub(v___x_1311_, v___x_1312_);
lean_dec(v___x_1312_);
lean_dec(v___x_1311_);
v___x_1342_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg___closed__0);
v___x_1343_ = lean_int_dec_le(v_n_1313_, v___x_1342_);
if (v___x_1343_ == 0)
{
lean_inc(v_n_1313_);
v___y_1315_ = v_n_1313_;
goto v___jp_1314_;
}
else
{
v___y_1315_ = v___x_1342_;
goto v___jp_1314_;
}
v___jp_1303_:
{
lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; 
lean_inc(v_mul_1282_);
v___x_1307_ = lean_apply_2(v_mul_1282_, v___y_1304_, v___y_1306_);
v___x_1308_ = lean_apply_2(v_mul_1282_, v___x_1307_, v___y_1305_);
v___x_1309_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_addAt___redArg(v_fo_1283_, v_coeffs_1293_, v_colIdx_1300_, v___x_1308_);
lean_dec(v_colIdx_1300_);
return v___x_1309_;
}
v___jp_1314_:
{
lean_object* v_nLift_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; 
v_nLift_1316_ = l_Int_toNat(v___y_1315_);
lean_dec(v___y_1315_);
v___x_1317_ = lean_unsigned_to_nat(1u);
v___x_1318_ = lean_nat_add(v_nLift_1316_, v___x_1317_);
lean_inc(v___x_1318_);
lean_inc(v_uPrism_1285_);
v___x_1319_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg(v_uPrism_1285_, v___x_1318_);
if (lean_obj_tag(v___x_1319_) == 0)
{
lean_dec(v___x_1318_);
lean_dec(v_nLift_1316_);
lean_dec(v_n_1313_);
lean_dec(v_lambdaSqPow_1310_);
lean_dec(v_rowIdx_1301_);
lean_dec(v_colIdx_1300_);
lean_dec(v_snd_1299_);
lean_dec(v_coeffs_1293_);
lean_dec(v_add_1292_);
lean_dec(v_lambda_1291_);
lean_dec(v_inst_1290_);
lean_dec_ref(v_toRingOps_1289_);
lean_dec(v_r_1288_);
lean_dec(v_uPrism_1285_);
lean_dec(v_lSkip_1284_);
lean_dec_ref(v_fo_1283_);
lean_dec(v_mul_1282_);
return v___x_1319_;
}
else
{
lean_object* v_a_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; uint8_t v___x_1325_; 
v_a_1320_ = lean_ctor_get(v___x_1319_, 0);
lean_inc(v_a_1320_);
lean_dec_ref_known(v___x_1319_, 1);
v___x_1321_ = l_List_drop___redArg(v___x_1318_, v_uPrism_1285_);
lean_dec(v_uPrism_1285_);
lean_inc_ref(v_fo_1283_);
v___x_1322_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_sliceTailBits___redArg(v_fo_1283_, v_lSkip_1284_, v_nLift_1316_, v_nStack_1286_, v_rowIdx_1301_);
lean_dec(v_rowIdx_1301_);
v___x_1323_ = l_List_lengthTR___redArg(v___x_1321_);
v___x_1324_ = l_List_lengthTR___redArg(v___x_1322_);
v___x_1325_ = lean_nat_dec_eq(v___x_1323_, v___x_1324_);
lean_dec(v___x_1324_);
lean_dec(v___x_1323_);
if (v___x_1325_ == 0)
{
lean_object* v___x_1326_; 
lean_dec(v___x_1322_);
lean_dec(v___x_1321_);
lean_dec(v_a_1320_);
lean_dec(v_nLift_1316_);
lean_dec(v_n_1313_);
lean_dec(v_lambdaSqPow_1310_);
lean_dec(v_colIdx_1300_);
lean_dec(v_snd_1299_);
lean_dec(v_coeffs_1293_);
lean_dec(v_add_1292_);
lean_dec(v_lambda_1291_);
lean_dec(v_inst_1290_);
lean_dec_ref(v_toRingOps_1289_);
lean_dec(v_r_1288_);
lean_dec(v_lSkip_1284_);
lean_dec_ref(v_fo_1283_);
lean_dec(v_mul_1282_);
v___x_1326_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_1326_;
}
else
{
lean_object* v___x_1327_; lean_object* v___x_1328_; 
lean_inc(v_lSkip_1284_);
lean_inc_ref_n(v_fo_1283_, 2);
v___x_1327_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg(v_fo_1283_, v_lSkip_1284_, v_n_1313_, v_head_1287_);
v___x_1328_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_rCoordinatesForSlice___redArg(v_fo_1283_, v_r_1288_, v_lSkip_1284_, v_n_1313_, v_nLift_1316_);
lean_dec(v_nLift_1316_);
lean_dec(v_n_1313_);
if (lean_obj_tag(v___x_1328_) == 0)
{
lean_object* v___x_1329_; 
lean_dec_ref_known(v___x_1328_, 1);
lean_dec(v___x_1327_);
lean_dec(v___x_1322_);
lean_dec(v___x_1321_);
lean_dec(v_a_1320_);
lean_dec(v_lambdaSqPow_1310_);
lean_dec(v_colIdx_1300_);
lean_dec(v_snd_1299_);
lean_dec(v_coeffs_1293_);
lean_dec(v_add_1292_);
lean_dec(v_lambda_1291_);
lean_dec(v_inst_1290_);
lean_dec_ref(v_toRingOps_1289_);
lean_dec_ref(v_fo_1283_);
lean_dec(v_mul_1282_);
v___x_1329_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_1329_;
}
else
{
lean_object* v_a_1330_; lean_object* v_fst_1331_; lean_object* v_snd_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; uint8_t v___x_1335_; 
v_a_1330_ = lean_ctor_get(v___x_1328_, 0);
lean_inc(v_a_1330_);
lean_dec_ref_known(v___x_1328_, 1);
v_fst_1331_ = lean_ctor_get(v_a_1330_, 0);
lean_inc_n(v_fst_1331_, 2);
v_snd_1332_ = lean_ctor_get(v_a_1330_, 1);
lean_inc_n(v_snd_1332_, 2);
lean_dec(v_a_1330_);
v___x_1333_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_1289_, v___x_1321_, v___x_1322_);
lean_inc(v_a_1320_);
lean_inc_ref(v_fo_1283_);
v___x_1334_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism___redArg(v_fo_1283_, v_fst_1331_, v_a_1320_, v_snd_1332_);
v___x_1335_ = lean_unbox(v_snd_1299_);
lean_dec(v_snd_1299_);
if (v___x_1335_ == 0)
{
lean_object* v___x_1336_; 
lean_dec(v_snd_1332_);
lean_dec(v_fst_1331_);
lean_dec(v_a_1320_);
lean_dec(v_add_1292_);
lean_dec(v_lambda_1291_);
lean_dec(v_inst_1290_);
lean_inc(v_mul_1282_);
v___x_1336_ = lean_apply_2(v_mul_1282_, v_lambdaSqPow_1310_, v___x_1334_);
v___y_1304_ = v___x_1333_;
v___y_1305_ = v___x_1327_;
v___y_1306_ = v___x_1336_;
goto v___jp_1303_;
}
else
{
lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; 
lean_inc_ref(v_fo_1283_);
v___x_1337_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism___redArg(v_fo_1283_, v_inst_1290_, v_fst_1331_, v_a_1320_, v_snd_1332_);
lean_inc_n(v_mul_1282_, 3);
lean_inc(v_lambdaSqPow_1310_);
v___x_1338_ = lean_apply_2(v_mul_1282_, v_lambdaSqPow_1310_, v___x_1334_);
v___x_1339_ = lean_apply_2(v_mul_1282_, v_lambdaSqPow_1310_, v_lambda_1291_);
v___x_1340_ = lean_apply_2(v_mul_1282_, v___x_1339_, v___x_1337_);
v___x_1341_ = lean_apply_2(v_add_1292_, v___x_1338_, v___x_1340_);
v___y_1304_ = v___x_1333_;
v___y_1305_ = v___x_1327_;
v___y_1306_ = v___x_1341_;
goto v___jp_1303_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg___lam__0___boxed(lean_object* v_pow_1344_, lean_object* v_lambdaSq_1345_, lean_object* v_mul_1346_, lean_object* v_fo_1347_, lean_object* v_lSkip_1348_, lean_object* v_uPrism_1349_, lean_object* v_nStack_1350_, lean_object* v_head_1351_, lean_object* v_r_1352_, lean_object* v_toRingOps_1353_, lean_object* v_inst_1354_, lean_object* v_lambda_1355_, lean_object* v_add_1356_, lean_object* v_coeffs_1357_, lean_object* v_entry_1358_){
_start:
{
lean_object* v_res_1359_; 
v_res_1359_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg___lam__0(v_pow_1344_, v_lambdaSq_1345_, v_mul_1346_, v_fo_1347_, v_lSkip_1348_, v_uPrism_1349_, v_nStack_1350_, v_head_1351_, v_r_1352_, v_toRingOps_1353_, v_inst_1354_, v_lambda_1355_, v_add_1356_, v_coeffs_1357_, v_entry_1358_);
lean_dec(v_head_1351_);
lean_dec(v_nStack_1350_);
return v_res_1359_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg(lean_object* v_inst_1360_, lean_object* v_fo_1361_, lean_object* v_layout_1362_, lean_object* v_lambdaData_1363_, lean_object* v_lambda_1364_, lean_object* v_lambdaSq_1365_, lean_object* v_lSkip_1366_, lean_object* v_nStack_1367_, lean_object* v_uPrism_1368_, lean_object* v_r_1369_){
_start:
{
lean_object* v_width_1370_; lean_object* v_sortedCols_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; uint8_t v___x_1374_; 
v_width_1370_ = lean_ctor_get(v_layout_1362_, 2);
lean_inc(v_width_1370_);
v_sortedCols_1371_ = lean_ctor_get(v_layout_1362_, 3);
lean_inc(v_sortedCols_1371_);
lean_dec_ref(v_layout_1362_);
v___x_1372_ = l_List_lengthTR___redArg(v_sortedCols_1371_);
v___x_1373_ = l_List_lengthTR___redArg(v_lambdaData_1363_);
v___x_1374_ = lean_nat_dec_eq(v___x_1372_, v___x_1373_);
lean_dec(v___x_1373_);
lean_dec(v___x_1372_);
if (v___x_1374_ == 0)
{
lean_object* v___x_1375_; 
lean_dec(v_sortedCols_1371_);
lean_dec(v_width_1370_);
lean_dec(v_r_1369_);
lean_dec(v_uPrism_1368_);
lean_dec(v_nStack_1367_);
lean_dec(v_lSkip_1366_);
lean_dec(v_lambdaSq_1365_);
lean_dec(v_lambda_1364_);
lean_dec(v_lambdaData_1363_);
lean_dec_ref(v_fo_1361_);
lean_dec(v_inst_1360_);
v___x_1375_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_1375_;
}
else
{
if (lean_obj_tag(v_uPrism_1368_) == 0)
{
lean_object* v___x_1376_; 
lean_dec(v_sortedCols_1371_);
lean_dec(v_width_1370_);
lean_dec(v_r_1369_);
lean_dec(v_nStack_1367_);
lean_dec(v_lSkip_1366_);
lean_dec(v_lambdaSq_1365_);
lean_dec(v_lambda_1364_);
lean_dec(v_lambdaData_1363_);
lean_dec_ref(v_fo_1361_);
lean_dec(v_inst_1360_);
v___x_1376_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_1376_;
}
else
{
lean_object* v_head_1377_; lean_object* v___x_1378_; lean_object* v_toRingOps_1379_; lean_object* v_toSemiringOps_1380_; lean_object* v_zero_1381_; lean_object* v_add_1382_; lean_object* v_mul_1383_; lean_object* v_pow_1384_; lean_object* v___f_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; 
v_head_1377_ = lean_ctor_get(v_uPrism_1368_, 0);
lean_inc(v_head_1377_);
v___x_1378_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__9));
v_toRingOps_1379_ = lean_ctor_get(v_fo_1361_, 0);
lean_inc_ref(v_toRingOps_1379_);
v_toSemiringOps_1380_ = lean_ctor_get(v_toRingOps_1379_, 0);
v_zero_1381_ = lean_ctor_get(v_toSemiringOps_1380_, 0);
lean_inc(v_zero_1381_);
v_add_1382_ = lean_ctor_get(v_toSemiringOps_1380_, 3);
lean_inc(v_add_1382_);
v_mul_1383_ = lean_ctor_get(v_toSemiringOps_1380_, 4);
lean_inc(v_mul_1383_);
v_pow_1384_ = lean_ctor_get(v_toSemiringOps_1380_, 5);
lean_inc(v_pow_1384_);
v___f_1385_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg___lam__0___boxed), 15, 13);
lean_closure_set(v___f_1385_, 0, v_pow_1384_);
lean_closure_set(v___f_1385_, 1, v_lambdaSq_1365_);
lean_closure_set(v___f_1385_, 2, v_mul_1383_);
lean_closure_set(v___f_1385_, 3, v_fo_1361_);
lean_closure_set(v___f_1385_, 4, v_lSkip_1366_);
lean_closure_set(v___f_1385_, 5, v_uPrism_1368_);
lean_closure_set(v___f_1385_, 6, v_nStack_1367_);
lean_closure_set(v___f_1385_, 7, v_head_1377_);
lean_closure_set(v___f_1385_, 8, v_r_1369_);
lean_closure_set(v___f_1385_, 9, v_toRingOps_1379_);
lean_closure_set(v___f_1385_, 10, v_inst_1360_);
lean_closure_set(v___f_1385_, 11, v_lambda_1364_);
lean_closure_set(v___f_1385_, 12, v_add_1382_);
v___x_1386_ = l_List_replicateTR___redArg(v_width_1370_, v_zero_1381_);
v___x_1387_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_sortedCols_1371_, v_lambdaData_1363_);
v___x_1388_ = l_List_foldlM___redArg(v___x_1378_, v___f_1385_, v___x_1386_, v___x_1387_);
return v___x_1388_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs(lean_object* v_EF_1389_, lean_object* v_inst_1390_, lean_object* v_fo_1391_, lean_object* v_layout_1392_, lean_object* v_lambdaData_1393_, lean_object* v_lambda_1394_, lean_object* v_lambdaSq_1395_, lean_object* v_lSkip_1396_, lean_object* v_nStack_1397_, lean_object* v_uPrism_1398_, lean_object* v_r_1399_){
_start:
{
lean_object* v___x_1400_; 
v___x_1400_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg(v_inst_1390_, v_fo_1391_, v_layout_1392_, v_lambdaData_1393_, v_lambda_1394_, v_lambdaSq_1395_, v_lSkip_1396_, v_nStack_1397_, v_uPrism_1398_, v_r_1399_);
return v___x_1400_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__0(lean_object* v_inst_1401_, lean_object* v_inst_1402_, lean_object* v_mul_1403_, lean_object* v_add_1404_, lean_object* v_inner_1405_, lean_object* v_term_1406_, lean_object* v___y_1407_){
_start:
{
lean_object* v_fst_1408_; lean_object* v_snd_1409_; lean_object* v___x_1410_; lean_object* v_a_1411_; lean_object* v___x_1413_; uint8_t v_isShared_1414_; uint8_t v_isSharedCheck_1429_; 
v_fst_1408_ = lean_ctor_get(v_term_1406_, 0);
lean_inc(v_fst_1408_);
v_snd_1409_ = lean_ctor_get(v_term_1406_, 1);
lean_inc_n(v_snd_1409_, 2);
lean_dec_ref(v_term_1406_);
v___x_1410_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExt___redArg(v_inst_1401_, v_inst_1402_, v_snd_1409_, v___y_1407_);
v_a_1411_ = lean_ctor_get(v___x_1410_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1410_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1413_ = v___x_1410_;
v_isShared_1414_ = v_isSharedCheck_1429_;
goto v_resetjp_1412_;
}
else
{
lean_inc(v_a_1411_);
lean_dec(v___x_1410_);
v___x_1413_ = lean_box(0);
v_isShared_1414_ = v_isSharedCheck_1429_;
goto v_resetjp_1412_;
}
v_resetjp_1412_:
{
lean_object* v_snd_1415_; lean_object* v___x_1417_; uint8_t v_isShared_1418_; uint8_t v_isSharedCheck_1427_; 
v_snd_1415_ = lean_ctor_get(v_a_1411_, 1);
v_isSharedCheck_1427_ = !lean_is_exclusive(v_a_1411_);
if (v_isSharedCheck_1427_ == 0)
{
lean_object* v_unused_1428_; 
v_unused_1428_ = lean_ctor_get(v_a_1411_, 0);
lean_dec(v_unused_1428_);
v___x_1417_ = v_a_1411_;
v_isShared_1418_ = v_isSharedCheck_1427_;
goto v_resetjp_1416_;
}
else
{
lean_inc(v_snd_1415_);
lean_dec(v_a_1411_);
v___x_1417_ = lean_box(0);
v_isShared_1418_ = v_isSharedCheck_1427_;
goto v_resetjp_1416_;
}
v_resetjp_1416_:
{
lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1422_; 
v___x_1419_ = lean_apply_2(v_mul_1403_, v_fst_1408_, v_snd_1409_);
v___x_1420_ = lean_apply_2(v_add_1404_, v_inner_1405_, v___x_1419_);
if (v_isShared_1418_ == 0)
{
lean_ctor_set(v___x_1417_, 0, v___x_1420_);
v___x_1422_ = v___x_1417_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1426_; 
v_reuseFailAlloc_1426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1426_, 0, v___x_1420_);
lean_ctor_set(v_reuseFailAlloc_1426_, 1, v_snd_1415_);
v___x_1422_ = v_reuseFailAlloc_1426_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
lean_object* v___x_1424_; 
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 0, v___x_1422_);
v___x_1424_ = v___x_1413_;
goto v_reusejp_1423_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v___x_1422_);
v___x_1424_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1423_;
}
v_reusejp_1423_:
{
return v___x_1424_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1(lean_object* v___x_1433_, lean_object* v___f_1434_, lean_object* v_acc_1435_, lean_object* v_entry_1436_, lean_object* v___y_1437_){
_start:
{
lean_object* v_fst_1438_; lean_object* v_snd_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; uint8_t v___x_1442_; 
v_fst_1438_ = lean_ctor_get(v_entry_1436_, 0);
lean_inc(v_fst_1438_);
v_snd_1439_ = lean_ctor_get(v_entry_1436_, 1);
lean_inc(v_snd_1439_);
lean_dec_ref(v_entry_1436_);
v___x_1440_ = l_List_lengthTR___redArg(v_fst_1438_);
v___x_1441_ = l_List_lengthTR___redArg(v_snd_1439_);
v___x_1442_ = lean_nat_dec_eq(v___x_1440_, v___x_1441_);
lean_dec(v___x_1441_);
lean_dec(v___x_1440_);
if (v___x_1442_ == 0)
{
lean_object* v___x_1443_; 
lean_dec(v_snd_1439_);
lean_dec(v_fst_1438_);
lean_dec_ref(v___y_1437_);
lean_dec(v_acc_1435_);
lean_dec_ref(v___f_1434_);
lean_dec_ref(v___x_1433_);
v___x_1443_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1___closed__0));
return v___x_1443_;
}
else
{
lean_object* v___x_1444_; lean_object* v___x_866__overap_1445_; lean_object* v___x_1446_; 
v___x_1444_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_fst_1438_, v_snd_1439_);
v___x_866__overap_1445_ = l_List_foldlM___redArg(v___x_1433_, v___f_1434_, v_acc_1435_, v___x_1444_);
v___x_1446_ = lean_apply_1(v___x_866__overap_1445_, v___y_1437_);
return v___x_1446_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg(lean_object* v_inst_1447_, lean_object* v_inst_1448_, lean_object* v_fo_1449_, lean_object* v_qCoeffs_1450_, lean_object* v_stackingOpenings_1451_, lean_object* v_a_1452_){
_start:
{
lean_object* v___x_1453_; lean_object* v___x_1454_; uint8_t v___x_1455_; 
v___x_1453_ = l_List_lengthTR___redArg(v_qCoeffs_1450_);
v___x_1454_ = l_List_lengthTR___redArg(v_stackingOpenings_1451_);
v___x_1455_ = lean_nat_dec_eq(v___x_1453_, v___x_1454_);
lean_dec(v___x_1454_);
lean_dec(v___x_1453_);
if (v___x_1455_ == 0)
{
lean_object* v___x_1456_; 
lean_dec_ref(v_a_1452_);
lean_dec(v_stackingOpenings_1451_);
lean_dec(v_qCoeffs_1450_);
lean_dec_ref(v_fo_1449_);
lean_dec_ref(v_inst_1448_);
lean_dec_ref(v_inst_1447_);
v___x_1456_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1___closed__0));
return v___x_1456_;
}
else
{
lean_object* v___x_1457_; lean_object* v_toRingOps_1458_; lean_object* v_toSemiringOps_1459_; lean_object* v_zero_1460_; lean_object* v_add_1461_; lean_object* v_mul_1462_; lean_object* v___f_1463_; lean_object* v___f_1464_; lean_object* v___x_1465_; lean_object* v___x_124__overap_1466_; lean_object* v___x_1467_; 
v___x_1457_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__19));
v_toRingOps_1458_ = lean_ctor_get(v_fo_1449_, 0);
lean_inc_ref(v_toRingOps_1458_);
lean_dec_ref(v_fo_1449_);
v_toSemiringOps_1459_ = lean_ctor_get(v_toRingOps_1458_, 0);
lean_inc_ref(v_toSemiringOps_1459_);
lean_dec_ref(v_toRingOps_1458_);
v_zero_1460_ = lean_ctor_get(v_toSemiringOps_1459_, 0);
lean_inc(v_zero_1460_);
v_add_1461_ = lean_ctor_get(v_toSemiringOps_1459_, 3);
lean_inc(v_add_1461_);
v_mul_1462_ = lean_ctor_get(v_toSemiringOps_1459_, 4);
lean_inc(v_mul_1462_);
lean_dec_ref(v_toSemiringOps_1459_);
v___f_1463_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__0), 7, 4);
lean_closure_set(v___f_1463_, 0, v_inst_1447_);
lean_closure_set(v___f_1463_, 1, v_inst_1448_);
lean_closure_set(v___f_1463_, 2, v_mul_1462_);
lean_closure_set(v___f_1463_, 3, v_add_1461_);
v___f_1464_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg___lam__1), 5, 2);
lean_closure_set(v___f_1464_, 0, v___x_1457_);
lean_closure_set(v___f_1464_, 1, v___f_1463_);
v___x_1465_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_qCoeffs_1450_, v_stackingOpenings_1451_);
v___x_124__overap_1466_ = l_List_foldlM___redArg(v___x_1457_, v___f_1464_, v_zero_1460_, v___x_1465_);
v___x_1467_ = lean_apply_1(v___x_124__overap_1466_, v_a_1452_);
return v___x_1467_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM(lean_object* v_F_1468_, lean_object* v_EF_1469_, lean_object* v_inst_1470_, lean_object* v_inst_1471_, lean_object* v_fo_1472_, lean_object* v_qCoeffs_1473_, lean_object* v_stackingOpenings_1474_, lean_object* v_a_1475_){
_start:
{
lean_object* v___x_1476_; 
v___x_1476_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg(v_inst_1470_, v_inst_1471_, v_fo_1472_, v_qCoeffs_1473_, v_stackingOpenings_1474_, v_a_1475_);
return v___x_1476_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube___redArg(lean_object* v_fo_1477_, lean_object* v_lSkip_1478_, lean_object* v_uPrism_1479_){
_start:
{
if (lean_obj_tag(v_uPrism_1479_) == 0)
{
lean_object* v___x_1480_; 
lean_dec(v_lSkip_1478_);
lean_dec_ref(v_fo_1477_);
v___x_1480_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_requirePrefix___redArg___closed__0));
return v___x_1480_;
}
else
{
lean_object* v_toRingOps_1481_; lean_object* v_head_1482_; lean_object* v_tail_1483_; lean_object* v_toSemiringOps_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; 
v_toRingOps_1481_ = lean_ctor_get(v_fo_1477_, 0);
lean_inc_ref(v_toRingOps_1481_);
lean_dec_ref(v_fo_1477_);
v_head_1482_ = lean_ctor_get(v_uPrism_1479_, 0);
lean_inc(v_head_1482_);
v_tail_1483_ = lean_ctor_get(v_uPrism_1479_, 1);
lean_inc(v_tail_1483_);
lean_dec_ref_known(v_uPrism_1479_, 2);
v_toSemiringOps_1484_ = lean_ctor_get(v_toRingOps_1481_, 0);
lean_inc_ref(v_toSemiringOps_1484_);
lean_dec_ref(v_toRingOps_1481_);
v___x_1485_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_1484_, v_head_1482_, v_lSkip_1478_);
lean_dec(v_head_1482_);
v___x_1486_ = l_List_appendTR___redArg(v___x_1485_, v_tail_1483_);
v___x_1487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1487_, 0, v___x_1486_);
return v___x_1487_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube(lean_object* v_EF_1488_, lean_object* v_fo_1489_, lean_object* v_lSkip_1490_, lean_object* v_uPrism_1491_){
_start:
{
lean_object* v___x_1492_; 
v___x_1492_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube___redArg(v_fo_1489_, v_lSkip_1490_, v_uPrism_1491_);
return v___x_1492_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0(lean_object* v_fst_1496_, lean_object* v_inst_1497_, lean_object* v_fo_1498_, lean_object* v_fst_1499_, lean_object* v___x_1500_, lean_object* v_lSkip_1501_, lean_object* v_nStack_1502_, lean_object* v_r_1503_, lean_object* v_acc_1504_, lean_object* v_entry_1505_, lean_object* v___y_1506_){
_start:
{
lean_object* v_fst_1507_; lean_object* v_fst_1508_; lean_object* v_snd_1509_; lean_object* v_snd_1510_; lean_object* v___x_1512_; uint8_t v_isShared_1513_; uint8_t v_isSharedCheck_1549_; 
v_fst_1507_ = lean_ctor_get(v_entry_1505_, 0);
lean_inc(v_fst_1507_);
v_fst_1508_ = lean_ctor_get(v_fst_1507_, 0);
lean_inc(v_fst_1508_);
v_snd_1509_ = lean_ctor_get(v_entry_1505_, 1);
lean_inc(v_snd_1509_);
lean_dec_ref(v_entry_1505_);
v_snd_1510_ = lean_ctor_get(v_fst_1507_, 1);
v_isSharedCheck_1549_ = !lean_is_exclusive(v_fst_1507_);
if (v_isSharedCheck_1549_ == 0)
{
lean_object* v_unused_1550_; 
v_unused_1550_ = lean_ctor_get(v_fst_1507_, 0);
lean_dec(v_unused_1550_);
v___x_1512_ = v_fst_1507_;
v_isShared_1513_ = v_isSharedCheck_1549_;
goto v_resetjp_1511_;
}
else
{
lean_inc(v_snd_1510_);
lean_dec(v_fst_1507_);
v___x_1512_ = lean_box(0);
v_isShared_1513_ = v_isSharedCheck_1549_;
goto v_resetjp_1511_;
}
v_resetjp_1511_:
{
lean_object* v_width_1514_; lean_object* v___x_1515_; uint8_t v___x_1516_; 
v_width_1514_ = lean_ctor_get(v_fst_1508_, 2);
v___x_1515_ = l_List_lengthTR___redArg(v_snd_1509_);
lean_dec(v_snd_1509_);
v___x_1516_ = lean_nat_dec_eq(v___x_1515_, v_width_1514_);
lean_dec(v___x_1515_);
if (v___x_1516_ == 0)
{
lean_object* v___x_1517_; 
lean_del_object(v___x_1512_);
lean_dec(v_snd_1510_);
lean_dec(v_fst_1508_);
lean_dec_ref(v___y_1506_);
lean_dec(v_acc_1504_);
lean_dec(v_r_1503_);
lean_dec(v_nStack_1502_);
lean_dec(v_lSkip_1501_);
lean_dec(v___x_1500_);
lean_dec(v_fst_1499_);
lean_dec_ref(v_fo_1498_);
lean_dec(v_inst_1497_);
lean_dec_ref(v_fst_1496_);
v___x_1517_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0___closed__0));
return v___x_1517_;
}
else
{
lean_object* v_uPrism_1518_; lean_object* v___x_1520_; uint8_t v_isShared_1521_; uint8_t v_isSharedCheck_1547_; 
v_uPrism_1518_ = lean_ctor_get(v_fst_1496_, 0);
v_isSharedCheck_1547_ = !lean_is_exclusive(v_fst_1496_);
if (v_isSharedCheck_1547_ == 0)
{
lean_object* v_unused_1548_; 
v_unused_1548_ = lean_ctor_get(v_fst_1496_, 1);
lean_dec(v_unused_1548_);
v___x_1520_ = v_fst_1496_;
v_isShared_1521_ = v_isSharedCheck_1547_;
goto v_resetjp_1519_;
}
else
{
lean_inc(v_uPrism_1518_);
lean_dec(v_fst_1496_);
v___x_1520_ = lean_box(0);
v_isShared_1521_ = v_isSharedCheck_1547_;
goto v_resetjp_1519_;
}
v_resetjp_1519_:
{
lean_object* v___x_1522_; 
v___x_1522_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeCommitQCoeffs___redArg(v_inst_1497_, v_fo_1498_, v_fst_1508_, v_snd_1510_, v_fst_1499_, v___x_1500_, v_lSkip_1501_, v_nStack_1502_, v_uPrism_1518_, v_r_1503_);
if (lean_obj_tag(v___x_1522_) == 0)
{
lean_object* v_a_1523_; lean_object* v___x_1525_; uint8_t v_isShared_1526_; uint8_t v_isSharedCheck_1530_; 
lean_del_object(v___x_1520_);
lean_del_object(v___x_1512_);
lean_dec_ref(v___y_1506_);
lean_dec(v_acc_1504_);
v_a_1523_ = lean_ctor_get(v___x_1522_, 0);
v_isSharedCheck_1530_ = !lean_is_exclusive(v___x_1522_);
if (v_isSharedCheck_1530_ == 0)
{
v___x_1525_ = v___x_1522_;
v_isShared_1526_ = v_isSharedCheck_1530_;
goto v_resetjp_1524_;
}
else
{
lean_inc(v_a_1523_);
lean_dec(v___x_1522_);
v___x_1525_ = lean_box(0);
v_isShared_1526_ = v_isSharedCheck_1530_;
goto v_resetjp_1524_;
}
v_resetjp_1524_:
{
lean_object* v___x_1528_; 
if (v_isShared_1526_ == 0)
{
v___x_1528_ = v___x_1525_;
goto v_reusejp_1527_;
}
else
{
lean_object* v_reuseFailAlloc_1529_; 
v_reuseFailAlloc_1529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1529_, 0, v_a_1523_);
v___x_1528_ = v_reuseFailAlloc_1529_;
goto v_reusejp_1527_;
}
v_reusejp_1527_:
{
return v___x_1528_;
}
}
}
else
{
lean_object* v_a_1531_; lean_object* v___x_1533_; uint8_t v_isShared_1534_; uint8_t v_isSharedCheck_1546_; 
v_a_1531_ = lean_ctor_get(v___x_1522_, 0);
v_isSharedCheck_1546_ = !lean_is_exclusive(v___x_1522_);
if (v_isSharedCheck_1546_ == 0)
{
v___x_1533_ = v___x_1522_;
v_isShared_1534_ = v_isSharedCheck_1546_;
goto v_resetjp_1532_;
}
else
{
lean_inc(v_a_1531_);
lean_dec(v___x_1522_);
v___x_1533_ = lean_box(0);
v_isShared_1534_ = v_isSharedCheck_1546_;
goto v_resetjp_1532_;
}
v_resetjp_1532_:
{
lean_object* v___x_1535_; lean_object* v___x_1537_; 
v___x_1535_ = lean_box(0);
if (v_isShared_1521_ == 0)
{
lean_ctor_set_tag(v___x_1520_, 1);
lean_ctor_set(v___x_1520_, 1, v___x_1535_);
lean_ctor_set(v___x_1520_, 0, v_a_1531_);
v___x_1537_ = v___x_1520_;
goto v_reusejp_1536_;
}
else
{
lean_object* v_reuseFailAlloc_1545_; 
v_reuseFailAlloc_1545_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1545_, 0, v_a_1531_);
lean_ctor_set(v_reuseFailAlloc_1545_, 1, v___x_1535_);
v___x_1537_ = v_reuseFailAlloc_1545_;
goto v_reusejp_1536_;
}
v_reusejp_1536_:
{
lean_object* v___x_1538_; lean_object* v___x_1540_; 
v___x_1538_ = l_List_appendTR___redArg(v_acc_1504_, v___x_1537_);
if (v_isShared_1513_ == 0)
{
lean_ctor_set(v___x_1512_, 1, v___y_1506_);
lean_ctor_set(v___x_1512_, 0, v___x_1538_);
v___x_1540_ = v___x_1512_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1544_; 
v_reuseFailAlloc_1544_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1544_, 0, v___x_1538_);
lean_ctor_set(v_reuseFailAlloc_1544_, 1, v___y_1506_);
v___x_1540_ = v_reuseFailAlloc_1544_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
lean_object* v___x_1542_; 
if (v_isShared_1534_ == 0)
{
lean_ctor_set(v___x_1533_, 0, v___x_1540_);
v___x_1542_ = v___x_1533_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v___x_1540_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
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
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg(lean_object* v_inst_1554_, lean_object* v_inst_1555_, lean_object* v_inst_1556_, lean_object* v_inst_1557_, lean_object* v_fo_1558_, lean_object* v_stackingProof_1559_, lean_object* v_layouts_1560_, lean_object* v_needRotPerCommit_1561_, lean_object* v_lSkip_1562_, lean_object* v_nStack_1563_, lean_object* v_columnOpenings_1564_, lean_object* v_r_1565_, lean_object* v_a_1566_){
_start:
{
lean_object* v___x_1567_; lean_object* v_univariateRoundCoeffs_1568_; lean_object* v_sumcheckRoundPolys_1569_; lean_object* v_stackingOpenings_1570_; lean_object* v___x_1571_; uint8_t v___x_1572_; 
v___x_1567_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg___closed__19));
v_univariateRoundCoeffs_1568_ = lean_ctor_get(v_stackingProof_1559_, 0);
lean_inc(v_univariateRoundCoeffs_1568_);
v_sumcheckRoundPolys_1569_ = lean_ctor_get(v_stackingProof_1559_, 1);
lean_inc(v_sumcheckRoundPolys_1569_);
v_stackingOpenings_1570_ = lean_ctor_get(v_stackingProof_1559_, 2);
lean_inc(v_stackingOpenings_1570_);
lean_dec_ref(v_stackingProof_1559_);
v___x_1571_ = l_List_lengthTR___redArg(v_sumcheckRoundPolys_1569_);
v___x_1572_ = lean_nat_dec_eq(v___x_1571_, v_nStack_1563_);
lean_dec(v___x_1571_);
if (v___x_1572_ == 0)
{
lean_object* v___x_1573_; 
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_sumcheckRoundPolys_1569_);
lean_dec(v_univariateRoundCoeffs_1568_);
lean_dec_ref(v_a_1566_);
lean_dec(v_r_1565_);
lean_dec(v_columnOpenings_1564_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_needRotPerCommit_1561_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v___x_1573_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1573_;
}
else
{
lean_object* v___x_1574_; lean_object* v_a_1575_; lean_object* v_fst_1576_; lean_object* v_snd_1577_; lean_object* v___x_1578_; 
lean_inc_ref(v_inst_1557_);
lean_inc_ref(v_inst_1556_);
v___x_1574_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_1556_, v_inst_1557_, v_a_1566_);
v_a_1575_ = lean_ctor_get(v___x_1574_, 0);
lean_inc(v_a_1575_);
lean_dec_ref(v___x_1574_);
v_fst_1576_ = lean_ctor_get(v_a_1575_, 0);
lean_inc(v_fst_1576_);
v_snd_1577_ = lean_ctor_get(v_a_1575_, 1);
lean_inc(v_snd_1577_);
lean_dec(v_a_1575_);
lean_inc(v_needRotPerCommit_1561_);
lean_inc(v_layouts_1560_);
v___x_1578_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeLambdaIndexData(v_layouts_1560_, v_needRotPerCommit_1561_);
if (lean_obj_tag(v___x_1578_) == 0)
{
lean_object* v___x_1579_; 
lean_dec_ref_known(v___x_1578_, 1);
lean_dec(v_snd_1577_);
lean_dec(v_fst_1576_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_sumcheckRoundPolys_1569_);
lean_dec(v_univariateRoundCoeffs_1568_);
lean_dec(v_r_1565_);
lean_dec(v_columnOpenings_1564_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_needRotPerCommit_1561_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v___x_1579_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1579_;
}
else
{
lean_object* v_a_1580_; lean_object* v_fst_1581_; lean_object* v_snd_1582_; lean_object* v___x_1583_; 
v_a_1580_ = lean_ctor_get(v___x_1578_, 0);
lean_inc(v_a_1580_);
lean_dec_ref_known(v___x_1578_, 1);
v_fst_1581_ = lean_ctor_get(v_a_1580_, 0);
lean_inc(v_fst_1581_);
v_snd_1582_ = lean_ctor_get(v_a_1580_, 1);
lean_inc(v_snd_1582_);
lean_dec(v_a_1580_);
lean_inc(v_fst_1576_);
lean_inc(v_univariateRoundCoeffs_1568_);
lean_inc_ref(v_fo_1558_);
lean_inc_ref(v_inst_1555_);
v___x_1583_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_checkRound0Claim___redArg(v_inst_1555_, v_fo_1558_, v_columnOpenings_1564_, v_univariateRoundCoeffs_1568_, v_needRotPerCommit_1561_, v_lSkip_1562_, v_fst_1576_);
if (lean_obj_tag(v___x_1583_) == 0)
{
lean_object* v___x_1584_; 
lean_dec_ref_known(v___x_1583_, 1);
lean_dec(v_snd_1582_);
lean_dec(v_fst_1581_);
lean_dec(v_snd_1577_);
lean_dec(v_fst_1576_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_sumcheckRoundPolys_1569_);
lean_dec(v_univariateRoundCoeffs_1568_);
lean_dec(v_r_1565_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v___x_1584_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1584_;
}
else
{
lean_object* v_a_1585_; lean_object* v___x_1586_; uint8_t v___x_1587_; 
v_a_1585_ = lean_ctor_get(v___x_1583_, 0);
lean_inc(v_a_1585_);
lean_dec_ref_known(v___x_1583_, 1);
v___x_1586_ = l_List_lengthTR___redArg(v_a_1585_);
v___x_1587_ = lean_nat_dec_eq(v___x_1586_, v_snd_1582_);
lean_dec(v_snd_1582_);
lean_dec(v___x_1586_);
if (v___x_1587_ == 0)
{
lean_object* v___x_1588_; 
lean_dec(v_a_1585_);
lean_dec(v_fst_1581_);
lean_dec(v_snd_1577_);
lean_dec(v_fst_1576_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_sumcheckRoundPolys_1569_);
lean_dec(v_univariateRoundCoeffs_1568_);
lean_dec(v_r_1565_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v___x_1588_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1588_;
}
else
{
lean_object* v___x_1589_; 
lean_inc(v_univariateRoundCoeffs_1568_);
lean_inc_ref(v_inst_1557_);
lean_inc_ref(v_inst_1556_);
v___x_1589_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_observeExtList___redArg(v_inst_1556_, v_inst_1557_, v_univariateRoundCoeffs_1568_, v_snd_1577_);
if (lean_obj_tag(v___x_1589_) == 0)
{
lean_object* v_a_1590_; lean_object* v___x_1592_; uint8_t v_isShared_1593_; uint8_t v_isSharedCheck_1597_; 
lean_dec(v_a_1585_);
lean_dec(v_fst_1581_);
lean_dec(v_fst_1576_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_sumcheckRoundPolys_1569_);
lean_dec(v_univariateRoundCoeffs_1568_);
lean_dec(v_r_1565_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v_a_1590_ = lean_ctor_get(v___x_1589_, 0);
v_isSharedCheck_1597_ = !lean_is_exclusive(v___x_1589_);
if (v_isSharedCheck_1597_ == 0)
{
v___x_1592_ = v___x_1589_;
v_isShared_1593_ = v_isSharedCheck_1597_;
goto v_resetjp_1591_;
}
else
{
lean_inc(v_a_1590_);
lean_dec(v___x_1589_);
v___x_1592_ = lean_box(0);
v_isShared_1593_ = v_isSharedCheck_1597_;
goto v_resetjp_1591_;
}
v_resetjp_1591_:
{
lean_object* v___x_1595_; 
if (v_isShared_1593_ == 0)
{
v___x_1595_ = v___x_1592_;
goto v_reusejp_1594_;
}
else
{
lean_object* v_reuseFailAlloc_1596_; 
v_reuseFailAlloc_1596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1596_, 0, v_a_1590_);
v___x_1595_ = v_reuseFailAlloc_1596_;
goto v_reusejp_1594_;
}
v_reusejp_1594_:
{
return v___x_1595_;
}
}
}
else
{
lean_object* v_a_1598_; lean_object* v_snd_1599_; lean_object* v___x_1601_; uint8_t v_isShared_1602_; uint8_t v_isSharedCheck_1698_; 
v_a_1598_ = lean_ctor_get(v___x_1589_, 0);
lean_inc(v_a_1598_);
lean_dec_ref_known(v___x_1589_, 1);
v_snd_1599_ = lean_ctor_get(v_a_1598_, 1);
v_isSharedCheck_1698_ = !lean_is_exclusive(v_a_1598_);
if (v_isSharedCheck_1698_ == 0)
{
lean_object* v_unused_1699_; 
v_unused_1699_ = lean_ctor_get(v_a_1598_, 0);
lean_dec(v_unused_1699_);
v___x_1601_ = v_a_1598_;
v_isShared_1602_ = v_isSharedCheck_1698_;
goto v_resetjp_1600_;
}
else
{
lean_inc(v_snd_1599_);
lean_dec(v_a_1598_);
v___x_1601_ = lean_box(0);
v_isShared_1602_ = v_isSharedCheck_1698_;
goto v_resetjp_1600_;
}
v_resetjp_1600_:
{
lean_object* v___x_1603_; lean_object* v_a_1604_; lean_object* v_fst_1605_; lean_object* v_snd_1606_; lean_object* v___x_1608_; uint8_t v_isShared_1609_; uint8_t v_isSharedCheck_1697_; 
lean_inc_ref(v_inst_1557_);
lean_inc_ref(v_inst_1556_);
v___x_1603_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_TranscriptM_sampleExt___redArg(v_inst_1556_, v_inst_1557_, v_snd_1599_);
v_a_1604_ = lean_ctor_get(v___x_1603_, 0);
lean_inc(v_a_1604_);
lean_dec_ref(v___x_1603_);
v_fst_1605_ = lean_ctor_get(v_a_1604_, 0);
v_snd_1606_ = lean_ctor_get(v_a_1604_, 1);
v_isSharedCheck_1697_ = !lean_is_exclusive(v_a_1604_);
if (v_isSharedCheck_1697_ == 0)
{
v___x_1608_ = v_a_1604_;
v_isShared_1609_ = v_isSharedCheck_1697_;
goto v_resetjp_1607_;
}
else
{
lean_inc(v_snd_1606_);
lean_inc(v_fst_1605_);
lean_dec(v_a_1604_);
v___x_1608_ = lean_box(0);
v_isShared_1609_ = v_isSharedCheck_1697_;
goto v_resetjp_1607_;
}
v_resetjp_1607_:
{
lean_object* v___x_1610_; lean_object* v_u0_1611_; lean_object* v_nextClaim_1612_; lean_object* v___x_1613_; lean_object* v___x_1615_; 
lean_inc_ref(v_fo_1558_);
v___x_1610_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_initialRoundState___redArg(v_fo_1558_, v_univariateRoundCoeffs_1568_, v_lSkip_1562_, v_fst_1605_, v_a_1585_);
v_u0_1611_ = lean_ctor_get(v___x_1610_, 2);
lean_inc(v_u0_1611_);
v_nextClaim_1612_ = lean_ctor_get(v___x_1610_, 3);
lean_inc(v_nextClaim_1612_);
lean_dec_ref(v___x_1610_);
v___x_1613_ = lean_box(0);
if (v_isShared_1609_ == 0)
{
lean_ctor_set_tag(v___x_1608_, 1);
lean_ctor_set(v___x_1608_, 1, v___x_1613_);
lean_ctor_set(v___x_1608_, 0, v_u0_1611_);
v___x_1615_ = v___x_1608_;
goto v_reusejp_1614_;
}
else
{
lean_object* v_reuseFailAlloc_1696_; 
v_reuseFailAlloc_1696_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1696_, 0, v_u0_1611_);
lean_ctor_set(v_reuseFailAlloc_1696_, 1, v___x_1613_);
v___x_1615_ = v_reuseFailAlloc_1696_;
goto v_reusejp_1614_;
}
v_reusejp_1614_:
{
lean_object* v___x_1617_; 
if (v_isShared_1602_ == 0)
{
lean_ctor_set(v___x_1601_, 1, v_nextClaim_1612_);
lean_ctor_set(v___x_1601_, 0, v___x_1615_);
v___x_1617_ = v___x_1601_;
goto v_reusejp_1616_;
}
else
{
lean_object* v_reuseFailAlloc_1695_; 
v_reuseFailAlloc_1695_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1695_, 0, v___x_1615_);
lean_ctor_set(v_reuseFailAlloc_1695_, 1, v_nextClaim_1612_);
v___x_1617_ = v_reuseFailAlloc_1695_;
goto v_reusejp_1616_;
}
v_reusejp_1616_:
{
lean_object* v___x_1618_; 
lean_inc_ref(v_fo_1558_);
lean_inc_ref(v_inst_1557_);
lean_inc_ref(v_inst_1556_);
v___x_1618_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyRemainingRoundsM___redArg(v_inst_1556_, v_inst_1557_, v_fo_1558_, v_sumcheckRoundPolys_1569_, v___x_1617_, v_snd_1606_);
if (lean_obj_tag(v___x_1618_) == 0)
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1626_; 
lean_dec(v_fst_1581_);
lean_dec(v_fst_1576_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_r_1565_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v_a_1619_ = lean_ctor_get(v___x_1618_, 0);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1618_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1621_ = v___x_1618_;
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1618_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1624_; 
if (v_isShared_1622_ == 0)
{
v___x_1624_ = v___x_1621_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_a_1619_);
v___x_1624_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1623_;
}
v_reusejp_1623_:
{
return v___x_1624_;
}
}
}
else
{
lean_object* v_a_1627_; lean_object* v_toRingOps_1628_; lean_object* v_toSemiringOps_1629_; lean_object* v_fst_1630_; lean_object* v_snd_1631_; lean_object* v_mul_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; uint8_t v___x_1636_; 
v_a_1627_ = lean_ctor_get(v___x_1618_, 0);
lean_inc(v_a_1627_);
lean_dec_ref_known(v___x_1618_, 1);
v_toRingOps_1628_ = lean_ctor_get(v_fo_1558_, 0);
v_toSemiringOps_1629_ = lean_ctor_get(v_toRingOps_1628_, 0);
v_fst_1630_ = lean_ctor_get(v_a_1627_, 0);
lean_inc(v_fst_1630_);
v_snd_1631_ = lean_ctor_get(v_a_1627_, 1);
lean_inc(v_snd_1631_);
lean_dec(v_a_1627_);
v_mul_1632_ = lean_ctor_get(v_toSemiringOps_1629_, 4);
lean_inc(v_mul_1632_);
lean_inc_n(v_fst_1576_, 2);
v___x_1633_ = lean_apply_2(v_mul_1632_, v_fst_1576_, v_fst_1576_);
v___x_1634_ = l_List_lengthTR___redArg(v_stackingOpenings_1570_);
v___x_1635_ = l_List_lengthTR___redArg(v_layouts_1560_);
v___x_1636_ = lean_nat_dec_eq(v___x_1634_, v___x_1635_);
lean_dec(v___x_1635_);
lean_dec(v___x_1634_);
if (v___x_1636_ == 0)
{
lean_object* v___x_1637_; 
lean_dec(v___x_1633_);
lean_dec(v_snd_1631_);
lean_dec(v_fst_1630_);
lean_dec(v_fst_1581_);
lean_dec(v_fst_1576_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_r_1565_);
lean_dec(v_nStack_1563_);
lean_dec(v_lSkip_1562_);
lean_dec(v_layouts_1560_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
lean_dec(v_inst_1554_);
v___x_1637_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1637_;
}
else
{
lean_object* v___f_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_5592__overap_1641_; lean_object* v___x_1642_; 
lean_inc(v_lSkip_1562_);
lean_inc_ref(v_fo_1558_);
lean_inc(v_fst_1630_);
v___f_1638_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___lam__0), 11, 8);
lean_closure_set(v___f_1638_, 0, v_fst_1630_);
lean_closure_set(v___f_1638_, 1, v_inst_1554_);
lean_closure_set(v___f_1638_, 2, v_fo_1558_);
lean_closure_set(v___f_1638_, 3, v_fst_1576_);
lean_closure_set(v___f_1638_, 4, v___x_1633_);
lean_closure_set(v___f_1638_, 5, v_lSkip_1562_);
lean_closure_set(v___f_1638_, 6, v_nStack_1563_);
lean_closure_set(v___f_1638_, 7, v_r_1565_);
v___x_1639_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_layouts_1560_, v_fst_1581_);
lean_inc(v_stackingOpenings_1570_);
v___x_1640_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_1639_, v_stackingOpenings_1570_);
v___x_5592__overap_1641_ = l_List_foldlM___redArg(v___x_1567_, v___f_1638_, v___x_1613_, v___x_1640_);
v___x_1642_ = lean_apply_1(v___x_5592__overap_1641_, v_snd_1631_);
if (lean_obj_tag(v___x_1642_) == 0)
{
lean_object* v_a_1643_; lean_object* v___x_1645_; uint8_t v_isShared_1646_; uint8_t v_isSharedCheck_1650_; 
lean_dec(v_fst_1630_);
lean_dec(v_stackingOpenings_1570_);
lean_dec(v_lSkip_1562_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1557_);
lean_dec_ref(v_inst_1556_);
lean_dec_ref(v_inst_1555_);
v_a_1643_ = lean_ctor_get(v___x_1642_, 0);
v_isSharedCheck_1650_ = !lean_is_exclusive(v___x_1642_);
if (v_isSharedCheck_1650_ == 0)
{
v___x_1645_ = v___x_1642_;
v_isShared_1646_ = v_isSharedCheck_1650_;
goto v_resetjp_1644_;
}
else
{
lean_inc(v_a_1643_);
lean_dec(v___x_1642_);
v___x_1645_ = lean_box(0);
v_isShared_1646_ = v_isSharedCheck_1650_;
goto v_resetjp_1644_;
}
v_resetjp_1644_:
{
lean_object* v___x_1648_; 
if (v_isShared_1646_ == 0)
{
v___x_1648_ = v___x_1645_;
goto v_reusejp_1647_;
}
else
{
lean_object* v_reuseFailAlloc_1649_; 
v_reuseFailAlloc_1649_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1649_, 0, v_a_1643_);
v___x_1648_ = v_reuseFailAlloc_1649_;
goto v_reusejp_1647_;
}
v_reusejp_1647_:
{
return v___x_1648_;
}
}
}
else
{
lean_object* v_a_1651_; lean_object* v_fst_1652_; lean_object* v_snd_1653_; lean_object* v___x_1654_; 
v_a_1651_ = lean_ctor_get(v___x_1642_, 0);
lean_inc(v_a_1651_);
lean_dec_ref_known(v___x_1642_, 1);
v_fst_1652_ = lean_ctor_get(v_a_1651_, 0);
lean_inc(v_fst_1652_);
v_snd_1653_ = lean_ctor_get(v_a_1651_, 1);
lean_inc(v_snd_1653_);
lean_dec(v_a_1651_);
lean_inc_ref(v_fo_1558_);
v___x_1654_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeFinalSumM___redArg(v_inst_1556_, v_inst_1557_, v_fo_1558_, v_fst_1652_, v_stackingOpenings_1570_, v_snd_1653_);
if (lean_obj_tag(v___x_1654_) == 0)
{
lean_object* v_a_1655_; lean_object* v___x_1657_; uint8_t v_isShared_1658_; uint8_t v_isSharedCheck_1662_; 
lean_dec(v_fst_1630_);
lean_dec(v_lSkip_1562_);
lean_dec_ref(v_fo_1558_);
lean_dec_ref(v_inst_1555_);
v_a_1655_ = lean_ctor_get(v___x_1654_, 0);
v_isSharedCheck_1662_ = !lean_is_exclusive(v___x_1654_);
if (v_isSharedCheck_1662_ == 0)
{
v___x_1657_ = v___x_1654_;
v_isShared_1658_ = v_isSharedCheck_1662_;
goto v_resetjp_1656_;
}
else
{
lean_inc(v_a_1655_);
lean_dec(v___x_1654_);
v___x_1657_ = lean_box(0);
v_isShared_1658_ = v_isSharedCheck_1662_;
goto v_resetjp_1656_;
}
v_resetjp_1656_:
{
lean_object* v___x_1660_; 
if (v_isShared_1658_ == 0)
{
v___x_1660_ = v___x_1657_;
goto v_reusejp_1659_;
}
else
{
lean_object* v_reuseFailAlloc_1661_; 
v_reuseFailAlloc_1661_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1661_, 0, v_a_1655_);
v___x_1660_ = v_reuseFailAlloc_1661_;
goto v_reusejp_1659_;
}
v_reusejp_1659_:
{
return v___x_1660_;
}
}
}
else
{
lean_object* v_a_1663_; lean_object* v_fst_1664_; lean_object* v_snd_1665_; lean_object* v___x_1667_; uint8_t v_isShared_1668_; uint8_t v_isSharedCheck_1694_; 
v_a_1663_ = lean_ctor_get(v___x_1654_, 0);
lean_inc(v_a_1663_);
lean_dec_ref_known(v___x_1654_, 1);
v_fst_1664_ = lean_ctor_get(v_a_1663_, 0);
v_snd_1665_ = lean_ctor_get(v_a_1663_, 1);
v_isSharedCheck_1694_ = !lean_is_exclusive(v_a_1663_);
if (v_isSharedCheck_1694_ == 0)
{
v___x_1667_ = v_a_1663_;
v_isShared_1668_ = v_isSharedCheck_1694_;
goto v_resetjp_1666_;
}
else
{
lean_inc(v_snd_1665_);
lean_inc(v_fst_1664_);
lean_dec(v_a_1663_);
v___x_1667_ = lean_box(0);
v_isShared_1668_ = v_isSharedCheck_1694_;
goto v_resetjp_1666_;
}
v_resetjp_1666_:
{
lean_object* v_uPrism_1669_; lean_object* v_claim_1670_; lean_object* v___x_1672_; uint8_t v_isShared_1673_; uint8_t v_isSharedCheck_1693_; 
v_uPrism_1669_ = lean_ctor_get(v_fst_1630_, 0);
v_claim_1670_ = lean_ctor_get(v_fst_1630_, 1);
v_isSharedCheck_1693_ = !lean_is_exclusive(v_fst_1630_);
if (v_isSharedCheck_1693_ == 0)
{
v___x_1672_ = v_fst_1630_;
v_isShared_1673_ = v_isSharedCheck_1693_;
goto v_resetjp_1671_;
}
else
{
lean_inc(v_claim_1670_);
lean_inc(v_uPrism_1669_);
lean_dec(v_fst_1630_);
v___x_1672_ = lean_box(0);
v_isShared_1673_ = v_isSharedCheck_1693_;
goto v_resetjp_1671_;
}
v_resetjp_1671_:
{
lean_object* v___x_1674_; uint8_t v___x_1675_; 
v___x_1674_ = lean_apply_2(v_inst_1555_, v_claim_1670_, v_fst_1664_);
v___x_1675_ = lean_unbox(v___x_1674_);
if (v___x_1675_ == 0)
{
lean_object* v___x_1676_; 
lean_del_object(v___x_1672_);
lean_dec(v_uPrism_1669_);
lean_del_object(v___x_1667_);
lean_dec(v_snd_1665_);
lean_dec(v_lSkip_1562_);
lean_dec_ref(v_fo_1558_);
v___x_1676_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1676_;
}
else
{
lean_object* v___x_1677_; 
lean_inc(v_uPrism_1669_);
v___x_1677_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_computeUCube___redArg(v_fo_1558_, v_lSkip_1562_, v_uPrism_1669_);
if (lean_obj_tag(v___x_1677_) == 0)
{
lean_object* v___x_1678_; 
lean_dec_ref_known(v___x_1677_, 1);
lean_del_object(v___x_1672_);
lean_dec(v_uPrism_1669_);
lean_del_object(v___x_1667_);
lean_dec(v_snd_1665_);
v___x_1678_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg___closed__0));
return v___x_1678_;
}
else
{
lean_object* v_a_1679_; lean_object* v___x_1681_; uint8_t v_isShared_1682_; uint8_t v_isSharedCheck_1692_; 
v_a_1679_ = lean_ctor_get(v___x_1677_, 0);
v_isSharedCheck_1692_ = !lean_is_exclusive(v___x_1677_);
if (v_isSharedCheck_1692_ == 0)
{
v___x_1681_ = v___x_1677_;
v_isShared_1682_ = v_isSharedCheck_1692_;
goto v_resetjp_1680_;
}
else
{
lean_inc(v_a_1679_);
lean_dec(v___x_1677_);
v___x_1681_ = lean_box(0);
v_isShared_1682_ = v_isSharedCheck_1692_;
goto v_resetjp_1680_;
}
v_resetjp_1680_:
{
lean_object* v___x_1684_; 
if (v_isShared_1673_ == 0)
{
lean_ctor_set(v___x_1672_, 1, v_a_1679_);
v___x_1684_ = v___x_1672_;
goto v_reusejp_1683_;
}
else
{
lean_object* v_reuseFailAlloc_1691_; 
v_reuseFailAlloc_1691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1691_, 0, v_uPrism_1669_);
lean_ctor_set(v_reuseFailAlloc_1691_, 1, v_a_1679_);
v___x_1684_ = v_reuseFailAlloc_1691_;
goto v_reusejp_1683_;
}
v_reusejp_1683_:
{
lean_object* v___x_1686_; 
if (v_isShared_1668_ == 0)
{
lean_ctor_set(v___x_1667_, 0, v___x_1684_);
v___x_1686_ = v___x_1667_;
goto v_reusejp_1685_;
}
else
{
lean_object* v_reuseFailAlloc_1690_; 
v_reuseFailAlloc_1690_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1690_, 0, v___x_1684_);
lean_ctor_set(v_reuseFailAlloc_1690_, 1, v_snd_1665_);
v___x_1686_ = v_reuseFailAlloc_1690_;
goto v_reusejp_1685_;
}
v_reusejp_1685_:
{
lean_object* v___x_1688_; 
if (v_isShared_1682_ == 0)
{
lean_ctor_set(v___x_1681_, 0, v___x_1686_);
v___x_1688_ = v___x_1681_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v___x_1686_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
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
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM(lean_object* v_F_1700_, lean_object* v_EF_1701_, lean_object* v_inst_1702_, lean_object* v_inst_1703_, lean_object* v_inst_1704_, lean_object* v_inst_1705_, lean_object* v_fo_1706_, lean_object* v_stackingProof_1707_, lean_object* v_layouts_1708_, lean_object* v_needRotPerCommit_1709_, lean_object* v_lSkip_1710_, lean_object* v_nStack_1711_, lean_object* v_columnOpenings_1712_, lean_object* v_r_1713_, lean_object* v_a_1714_){
_start:
{
lean_object* v___x_1715_; 
v___x_1715_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg(v_inst_1702_, v_inst_1703_, v_inst_1704_, v_inst_1705_, v_fo_1706_, v_stackingProof_1707_, v_layouts_1708_, v_needRotPerCommit_1709_, v_lSkip_1710_, v_nStack_1711_, v_columnOpenings_1712_, v_r_1713_, v_a_1714_);
return v___x_1715_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify___redArg(lean_object* v_inst_1716_, lean_object* v_inst_1717_, lean_object* v_inst_1718_, lean_object* v_inst_1719_, lean_object* v_fo_1720_, lean_object* v_transcript_1721_, lean_object* v_stackingProof_1722_, lean_object* v_layouts_1723_, lean_object* v_needRotPerCommit_1724_, lean_object* v_lSkip_1725_, lean_object* v_nStack_1726_, lean_object* v_columnOpenings_1727_, lean_object* v_r_1728_){
_start:
{
lean_object* v___x_1729_; 
v___x_1729_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verifyM___redArg(v_inst_1716_, v_inst_1717_, v_inst_1718_, v_inst_1719_, v_fo_1720_, v_stackingProof_1722_, v_layouts_1723_, v_needRotPerCommit_1724_, v_lSkip_1725_, v_nStack_1726_, v_columnOpenings_1727_, v_r_1728_, v_transcript_1721_);
if (lean_obj_tag(v___x_1729_) == 0)
{
lean_object* v_a_1730_; lean_object* v___x_1732_; uint8_t v_isShared_1733_; uint8_t v_isSharedCheck_1737_; 
v_a_1730_ = lean_ctor_get(v___x_1729_, 0);
v_isSharedCheck_1737_ = !lean_is_exclusive(v___x_1729_);
if (v_isSharedCheck_1737_ == 0)
{
v___x_1732_ = v___x_1729_;
v_isShared_1733_ = v_isSharedCheck_1737_;
goto v_resetjp_1731_;
}
else
{
lean_inc(v_a_1730_);
lean_dec(v___x_1729_);
v___x_1732_ = lean_box(0);
v_isShared_1733_ = v_isSharedCheck_1737_;
goto v_resetjp_1731_;
}
v_resetjp_1731_:
{
lean_object* v___x_1735_; 
if (v_isShared_1733_ == 0)
{
v___x_1735_ = v___x_1732_;
goto v_reusejp_1734_;
}
else
{
lean_object* v_reuseFailAlloc_1736_; 
v_reuseFailAlloc_1736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1736_, 0, v_a_1730_);
v___x_1735_ = v_reuseFailAlloc_1736_;
goto v_reusejp_1734_;
}
v_reusejp_1734_:
{
return v___x_1735_;
}
}
}
else
{
lean_object* v_a_1738_; lean_object* v___x_1740_; uint8_t v_isShared_1741_; uint8_t v_isSharedCheck_1755_; 
v_a_1738_ = lean_ctor_get(v___x_1729_, 0);
v_isSharedCheck_1755_ = !lean_is_exclusive(v___x_1729_);
if (v_isSharedCheck_1755_ == 0)
{
v___x_1740_ = v___x_1729_;
v_isShared_1741_ = v_isSharedCheck_1755_;
goto v_resetjp_1739_;
}
else
{
lean_inc(v_a_1738_);
lean_dec(v___x_1729_);
v___x_1740_ = lean_box(0);
v_isShared_1741_ = v_isSharedCheck_1755_;
goto v_resetjp_1739_;
}
v_resetjp_1739_:
{
lean_object* v_fst_1742_; lean_object* v_snd_1743_; lean_object* v___x_1745_; uint8_t v_isShared_1746_; uint8_t v_isSharedCheck_1754_; 
v_fst_1742_ = lean_ctor_get(v_a_1738_, 0);
v_snd_1743_ = lean_ctor_get(v_a_1738_, 1);
v_isSharedCheck_1754_ = !lean_is_exclusive(v_a_1738_);
if (v_isSharedCheck_1754_ == 0)
{
v___x_1745_ = v_a_1738_;
v_isShared_1746_ = v_isSharedCheck_1754_;
goto v_resetjp_1744_;
}
else
{
lean_inc(v_snd_1743_);
lean_inc(v_fst_1742_);
lean_dec(v_a_1738_);
v___x_1745_ = lean_box(0);
v_isShared_1746_ = v_isSharedCheck_1754_;
goto v_resetjp_1744_;
}
v_resetjp_1744_:
{
lean_object* v_uPrism_1747_; lean_object* v___x_1749_; 
v_uPrism_1747_ = lean_ctor_get(v_fst_1742_, 0);
lean_inc(v_uPrism_1747_);
lean_dec(v_fst_1742_);
if (v_isShared_1746_ == 0)
{
lean_ctor_set(v___x_1745_, 0, v_uPrism_1747_);
v___x_1749_ = v___x_1745_;
goto v_reusejp_1748_;
}
else
{
lean_object* v_reuseFailAlloc_1753_; 
v_reuseFailAlloc_1753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1753_, 0, v_uPrism_1747_);
lean_ctor_set(v_reuseFailAlloc_1753_, 1, v_snd_1743_);
v___x_1749_ = v_reuseFailAlloc_1753_;
goto v_reusejp_1748_;
}
v_reusejp_1748_:
{
lean_object* v___x_1751_; 
if (v_isShared_1741_ == 0)
{
lean_ctor_set(v___x_1740_, 0, v___x_1749_);
v___x_1751_ = v___x_1740_;
goto v_reusejp_1750_;
}
else
{
lean_object* v_reuseFailAlloc_1752_; 
v_reuseFailAlloc_1752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1752_, 0, v___x_1749_);
v___x_1751_ = v_reuseFailAlloc_1752_;
goto v_reusejp_1750_;
}
v_reusejp_1750_:
{
return v___x_1751_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify(lean_object* v_F_1756_, lean_object* v_EF_1757_, lean_object* v_inst_1758_, lean_object* v_inst_1759_, lean_object* v_inst_1760_, lean_object* v_inst_1761_, lean_object* v_fo_1762_, lean_object* v_transcript_1763_, lean_object* v_stackingProof_1764_, lean_object* v_layouts_1765_, lean_object* v_needRotPerCommit_1766_, lean_object* v_lSkip_1767_, lean_object* v_nStack_1768_, lean_object* v_columnOpenings_1769_, lean_object* v_r_1770_){
_start:
{
lean_object* v___x_1771_; 
v___x_1771_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking_verify___redArg(v_inst_1758_, v_inst_1759_, v_inst_1760_, v_inst_1761_, v_fo_1762_, v_transcript_1763_, v_stackingProof_1764_, v_layouts_1765_, v_needRotPerCommit_1766_, v_lSkip_1767_, v_nStack_1768_, v_columnOpenings_1769_, v_r_1770_);
return v___x_1771_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Runtime_Core(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Ops(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Stacking(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Verifier_Runtime_Batch(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
