// Lean compiler output
// Module: Swirl.Spec.ReferenceVerifier.Runtime.PolyCommon
// Imports: public import Init public meta import Init public import Fundamentals.Spec.FieldOps
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
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* lean_nat_abs(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "coeffs"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__9;
static lean_once_cell_t lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__10;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__12_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_UnivariatePoly_evalAtPoint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_UnivariatePoly_evalAtPoint(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(lean_object* v_fo_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
lean_object* v_zero_4_; uint8_t v_isZero_5_; 
v_zero_4_ = lean_unsigned_to_nat(0u);
v_isZero_5_ = lean_nat_dec_eq(v_x_3_, v_zero_4_);
if (v_isZero_5_ == 1)
{
lean_dec_ref(v_fo_1_);
lean_inc(v_x_2_);
return v_x_2_;
}
else
{
lean_object* v_mul_6_; lean_object* v_one_7_; lean_object* v_n_8_; lean_object* v_y_9_; lean_object* v___x_10_; 
v_mul_6_ = lean_ctor_get(v_fo_1_, 4);
lean_inc(v_mul_6_);
v_one_7_ = lean_unsigned_to_nat(1u);
v_n_8_ = lean_nat_sub(v_x_3_, v_one_7_);
v_y_9_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_fo_1_, v_x_2_, v_n_8_);
lean_dec(v_n_8_);
lean_inc(v_y_9_);
v___x_10_ = lean_apply_2(v_mul_6_, v_y_9_, v_y_9_);
return v___x_10_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg___boxed(lean_object* v_fo_11_, lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_fo_11_, v_x_12_, v_x_13_);
lean_dec(v_x_13_);
lean_dec(v_x_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo(lean_object* v_EF_15_, lean_object* v_fo_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_fo_16_, v_x_17_, v_x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___boxed(lean_object* v_EF_20_, lean_object* v_fo_21_, lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo(v_EF_20_, v_fo_21_, v_x_22_, v_x_23_);
lean_dec(v_x_23_);
lean_dec(v_x_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg(lean_object* v_fo_25_, lean_object* v_x_26_, lean_object* v_a_27_, lean_object* v_a_28_){
_start:
{
if (lean_obj_tag(v_a_27_) == 0)
{
lean_object* v___x_29_; 
lean_dec_ref(v_fo_25_);
v___x_29_ = l_List_reverse___redArg(v_a_28_);
return v___x_29_;
}
else
{
lean_object* v_head_30_; lean_object* v_tail_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_40_; 
v_head_30_ = lean_ctor_get(v_a_27_, 0);
v_tail_31_ = lean_ctor_get(v_a_27_, 1);
v_isSharedCheck_40_ = !lean_is_exclusive(v_a_27_);
if (v_isSharedCheck_40_ == 0)
{
v___x_33_ = v_a_27_;
v_isShared_34_ = v_isSharedCheck_40_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_tail_31_);
lean_inc(v_head_30_);
lean_dec(v_a_27_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_40_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc_ref(v_fo_25_);
v___x_35_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_fo_25_, v_x_26_, v_head_30_);
lean_dec(v_head_30_);
if (v_isShared_34_ == 0)
{
lean_ctor_set(v___x_33_, 1, v_a_28_);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v___x_35_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v_a_28_);
v___x_37_ = v_reuseFailAlloc_39_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
v_a_27_ = v_tail_31_;
v_a_28_ = v___x_37_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg___boxed(lean_object* v_fo_41_, lean_object* v_x_42_, lean_object* v_a_43_, lean_object* v_a_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg(v_fo_41_, v_x_42_, v_a_43_, v_a_44_);
lean_dec(v_x_42_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(lean_object* v_fo_46_, lean_object* v_x_47_, lean_object* v_count_48_){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = l_List_range(v_count_48_);
v___x_50_ = lean_box(0);
v___x_51_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg(v_fo_46_, v_x_47_, v___x_49_, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg___boxed(lean_object* v_fo_52_, lean_object* v_x_53_, lean_object* v_count_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_fo_52_, v_x_53_, v_count_54_);
lean_dec(v_x_53_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo(lean_object* v_EF_56_, lean_object* v_fo_57_, lean_object* v_x_58_, lean_object* v_count_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_fo_57_, v_x_58_, v_count_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___boxed(lean_object* v_EF_61_, lean_object* v_fo_62_, lean_object* v_x_63_, lean_object* v_count_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo(v_EF_61_, v_fo_62_, v_x_63_, v_count_64_);
lean_dec(v_x_63_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0(lean_object* v_EF_66_, lean_object* v_fo_67_, lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___redArg(v_fo_67_, v_x_68_, v_a_69_, v_a_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0___boxed(lean_object* v_EF_72_, lean_object* v_fo_73_, lean_object* v_x_74_, lean_object* v_a_75_, lean_object* v_a_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo_spec__0(v_EF_72_, v_fo_73_, v_x_74_, v_a_75_, v_a_76_);
lean_dec(v_x_74_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg(lean_object* v_fo_78_, lean_object* v_x_79_, lean_object* v_as_80_, size_t v_i_81_, size_t v_stop_82_, lean_object* v_b_83_){
_start:
{
uint8_t v___x_84_; 
v___x_84_ = lean_usize_dec_eq(v_i_81_, v_stop_82_);
if (v___x_84_ == 0)
{
lean_object* v_add_85_; lean_object* v_mul_86_; size_t v___x_87_; size_t v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v_add_85_ = lean_ctor_get(v_fo_78_, 3);
v_mul_86_ = lean_ctor_get(v_fo_78_, 4);
v___x_87_ = ((size_t)1ULL);
v___x_88_ = lean_usize_sub(v_i_81_, v___x_87_);
v___x_89_ = lean_array_uget_borrowed(v_as_80_, v___x_88_);
lean_inc(v_mul_86_);
lean_inc(v_x_79_);
v___x_90_ = lean_apply_2(v_mul_86_, v_x_79_, v_b_83_);
lean_inc(v_add_85_);
lean_inc(v___x_89_);
v___x_91_ = lean_apply_2(v_add_85_, v___x_89_, v___x_90_);
v_i_81_ = v___x_88_;
v_b_83_ = v___x_91_;
goto _start;
}
else
{
lean_dec(v_x_79_);
lean_dec_ref(v_fo_78_);
return v_b_83_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg___boxed(lean_object* v_fo_93_, lean_object* v_x_94_, lean_object* v_as_95_, lean_object* v_i_96_, lean_object* v_stop_97_, lean_object* v_b_98_){
_start:
{
size_t v_i_boxed_99_; size_t v_stop_boxed_100_; lean_object* v_res_101_; 
v_i_boxed_99_ = lean_unbox_usize(v_i_96_);
lean_dec(v_i_96_);
v_stop_boxed_100_ = lean_unbox_usize(v_stop_97_);
lean_dec(v_stop_97_);
v_res_101_ = lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg(v_fo_93_, v_x_94_, v_as_95_, v_i_boxed_99_, v_stop_boxed_100_, v_b_98_);
lean_dec_ref(v_as_95_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0___redArg(lean_object* v_fo_102_, lean_object* v_x_103_, lean_object* v_init_104_, lean_object* v_l_105_){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_106_ = lean_array_mk(v_l_105_);
v___x_107_ = lean_array_get_size(v___x_106_);
v___x_108_ = lean_unsigned_to_nat(0u);
v___x_109_ = lean_nat_dec_lt(v___x_108_, v___x_107_);
if (v___x_109_ == 0)
{
lean_dec_ref(v___x_106_);
lean_dec(v_x_103_);
lean_dec_ref(v_fo_102_);
return v_init_104_;
}
else
{
size_t v___x_110_; size_t v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_usize_of_nat(v___x_107_);
v___x_111_ = ((size_t)0ULL);
v___x_112_ = lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg(v_fo_102_, v_x_103_, v___x_106_, v___x_110_, v___x_111_, v_init_104_);
lean_dec_ref(v___x_106_);
return v___x_112_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(lean_object* v_fo_113_, lean_object* v_coeffs_114_, lean_object* v_x_115_){
_start:
{
lean_object* v_zero_116_; lean_object* v___x_117_; 
v_zero_116_ = lean_ctor_get(v_fo_113_, 0);
lean_inc(v_zero_116_);
v___x_117_ = lp_swirl_x2drbr_x2dfv_List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0___redArg(v_fo_113_, v_x_115_, v_zero_116_, v_coeffs_114_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval(lean_object* v_EF_118_, lean_object* v_fo_119_, lean_object* v_coeffs_120_, lean_object* v_x_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_fo_119_, v_coeffs_120_, v_x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0(lean_object* v_EF_123_, lean_object* v_fo_124_, lean_object* v_x_125_, lean_object* v_init_126_, lean_object* v_l_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_swirl_x2drbr_x2dfv_List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0___redArg(v_fo_124_, v_x_125_, v_init_126_, v_l_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0(lean_object* v_EF_129_, lean_object* v_fo_130_, lean_object* v_x_131_, lean_object* v_as_132_, size_t v_i_133_, size_t v_stop_134_, lean_object* v_b_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___redArg(v_fo_130_, v_x_131_, v_as_132_, v_i_133_, v_stop_134_, v_b_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0___boxed(lean_object* v_EF_137_, lean_object* v_fo_138_, lean_object* v_x_139_, lean_object* v_as_140_, lean_object* v_i_141_, lean_object* v_stop_142_, lean_object* v_b_143_){
_start:
{
size_t v_i_boxed_144_; size_t v_stop_boxed_145_; lean_object* v_res_146_; 
v_i_boxed_144_ = lean_unbox_usize(v_i_141_);
lean_dec(v_i_141_);
v_stop_boxed_145_ = lean_unbox_usize(v_stop_142_);
lean_dec(v_stop_142_);
v_res_146_ = lp_swirl_x2drbr_x2dfv___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval_spec__0_spec__0(v_EF_137_, v_fo_138_, v_x_139_, v_as_140_, v_i_boxed_144_, v_stop_boxed_145_, v_b_143_);
lean_dec_ref(v_as_140_);
return v_res_146_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_160_ = lean_unsigned_to_nat(10u);
v___x_161_ = lean_nat_to_int(v___x_160_);
return v___x_161_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__0));
v___x_164_ = lean_string_length(v___x_163_);
return v___x_164_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__9);
v___x_166_ = lean_nat_to_int(v___x_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg(lean_object* v_inst_171_, lean_object* v_x_172_){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; uint8_t v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_173_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__6));
v___x_174_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__7);
v___x_175_ = l_List_repr___redArg(v_inst_171_, v_x_172_);
v___x_176_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_174_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
v___x_177_ = 0;
v___x_178_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_178_, 0, v___x_176_);
lean_ctor_set_uint8(v___x_178_, sizeof(void*)*1, v___x_177_);
v___x_179_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_173_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
v___x_180_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__10);
v___x_181_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__11));
v___x_182_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v___x_179_);
v___x_183_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg___closed__12));
v___x_184_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_182_);
lean_ctor_set(v___x_184_, 1, v___x_183_);
v___x_185_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_180_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
v___x_186_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set_uint8(v___x_186_, sizeof(void*)*1, v___x_177_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr(lean_object* v_EF_187_, lean_object* v_inst_188_, lean_object* v_x_189_, lean_object* v_prec_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___redArg(v_inst_188_, v_x_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___boxed(lean_object* v_EF_192_, lean_object* v_inst_193_, lean_object* v_x_194_, lean_object* v_prec_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr(v_EF_192_, v_inst_193_, v_x_194_, v_prec_195_);
lean_dec(v_prec_195_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly___redArg(lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___boxed), 4, 2);
lean_closure_set(v___x_198_, 0, lean_box(0));
lean_closure_set(v___x_198_, 1, v_inst_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly(lean_object* v_EF_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_instReprUnivariatePoly_repr___boxed), 4, 2);
lean_closure_set(v___x_201_, 0, lean_box(0));
lean_closure_set(v___x_201_, 1, v_inst_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_UnivariatePoly_evalAtPoint___redArg(lean_object* v_fo_202_, lean_object* v_p_203_, lean_object* v_x_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_fo_202_, v_p_203_, v_x_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_UnivariatePoly_evalAtPoint(lean_object* v_EF_206_, lean_object* v_fo_207_, lean_object* v_p_208_, lean_object* v_x_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_hornerEval___redArg(v_fo_207_, v_p_208_, v_x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle_spec__0___redArg(lean_object* v_fo_211_, lean_object* v_x_212_, lean_object* v_x_213_){
_start:
{
if (lean_obj_tag(v_x_213_) == 0)
{
lean_dec_ref(v_fo_211_);
return v_x_212_;
}
else
{
lean_object* v_head_214_; lean_object* v_toSemiringOps_215_; lean_object* v_tail_216_; lean_object* v_fst_217_; lean_object* v_snd_218_; lean_object* v_sub_219_; lean_object* v_one_220_; lean_object* v_add_221_; lean_object* v_mul_222_; lean_object* v___x_223_; lean_object* v_xyDouble_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v_head_214_ = lean_ctor_get(v_x_213_, 0);
lean_inc(v_head_214_);
v_toSemiringOps_215_ = lean_ctor_get(v_fo_211_, 0);
v_tail_216_ = lean_ctor_get(v_x_213_, 1);
lean_inc(v_tail_216_);
lean_dec_ref_known(v_x_213_, 2);
v_fst_217_ = lean_ctor_get(v_head_214_, 0);
lean_inc_n(v_fst_217_, 2);
v_snd_218_ = lean_ctor_get(v_head_214_, 1);
lean_inc_n(v_snd_218_, 2);
lean_dec(v_head_214_);
v_sub_219_ = lean_ctor_get(v_fo_211_, 1);
v_one_220_ = lean_ctor_get(v_toSemiringOps_215_, 1);
v_add_221_ = lean_ctor_get(v_toSemiringOps_215_, 3);
v_mul_222_ = lean_ctor_get(v_toSemiringOps_215_, 4);
lean_inc_n(v_mul_222_, 2);
v___x_223_ = lean_apply_2(v_mul_222_, v_fst_217_, v_snd_218_);
lean_inc_n(v_add_221_, 2);
lean_inc(v___x_223_);
v_xyDouble_224_ = lean_apply_2(v_add_221_, v___x_223_, v___x_223_);
lean_inc_n(v_sub_219_, 2);
lean_inc(v_one_220_);
v___x_225_ = lean_apply_2(v_sub_219_, v_one_220_, v_snd_218_);
v___x_226_ = lean_apply_2(v_sub_219_, v___x_225_, v_fst_217_);
v___x_227_ = lean_apply_2(v_add_221_, v___x_226_, v_xyDouble_224_);
v___x_228_ = lean_apply_2(v_mul_222_, v_x_212_, v___x_227_);
v_x_212_ = v___x_228_;
v_x_213_ = v_tail_216_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(lean_object* v_fo_230_, lean_object* v_x_231_, lean_object* v_y_232_){
_start:
{
lean_object* v_toSemiringOps_233_; lean_object* v_one_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v_toSemiringOps_233_ = lean_ctor_get(v_fo_230_, 0);
v_one_234_ = lean_ctor_get(v_toSemiringOps_233_, 1);
lean_inc(v_one_234_);
v___x_235_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_x_231_, v_y_232_);
v___x_236_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle_spec__0___redArg(v_fo_230_, v_one_234_, v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle(lean_object* v_EF_237_, lean_object* v_fo_238_, lean_object* v_x_239_, lean_object* v_y_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_fo_238_, v_x_239_, v_y_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle_spec__0(lean_object* v_EF_242_, lean_object* v_fo_243_, lean_object* v_x_244_, lean_object* v_x_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle_spec__0___redArg(v_fo_243_, v_x_244_, v_x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube___redArg(lean_object* v_fo_247_, lean_object* v_x_248_, lean_object* v_y_249_){
_start:
{
if (lean_obj_tag(v_x_248_) == 0)
{
if (lean_obj_tag(v_y_249_) == 0)
{
lean_object* v_toSemiringOps_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_269_; 
v_toSemiringOps_261_ = lean_ctor_get(v_fo_247_, 0);
v_isSharedCheck_269_ = !lean_is_exclusive(v_fo_247_);
if (v_isSharedCheck_269_ == 0)
{
lean_object* v_unused_270_; 
v_unused_270_ = lean_ctor_get(v_fo_247_, 1);
lean_dec(v_unused_270_);
v___x_263_ = v_fo_247_;
v_isShared_264_ = v_isSharedCheck_269_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_toSemiringOps_261_);
lean_dec(v_fo_247_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_269_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
lean_object* v_one_265_; lean_object* v___x_267_; 
v_one_265_ = lean_ctor_get(v_toSemiringOps_261_, 1);
lean_inc_n(v_one_265_, 2);
lean_dec_ref(v_toSemiringOps_261_);
if (v_isShared_264_ == 0)
{
lean_ctor_set(v___x_263_, 1, v_one_265_);
lean_ctor_set(v___x_263_, 0, v_one_265_);
v___x_267_ = v___x_263_;
goto v_reusejp_266_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v_one_265_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v_one_265_);
v___x_267_ = v_reuseFailAlloc_268_;
goto v_reusejp_266_;
}
v_reusejp_266_:
{
return v___x_267_;
}
}
}
else
{
lean_dec(v_y_249_);
goto v___jp_250_;
}
}
else
{
if (lean_obj_tag(v_y_249_) == 1)
{
lean_object* v_head_271_; lean_object* v_tail_272_; lean_object* v_head_273_; lean_object* v_tail_274_; lean_object* v___x_275_; lean_object* v_toSemiringOps_276_; lean_object* v_fst_277_; lean_object* v_snd_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_300_; 
v_head_271_ = lean_ctor_get(v_x_248_, 0);
lean_inc(v_head_271_);
v_tail_272_ = lean_ctor_get(v_x_248_, 1);
lean_inc(v_tail_272_);
lean_dec_ref_known(v_x_248_, 2);
v_head_273_ = lean_ctor_get(v_y_249_, 0);
lean_inc(v_head_273_);
v_tail_274_ = lean_ctor_get(v_y_249_, 1);
lean_inc(v_tail_274_);
lean_dec_ref_known(v_y_249_, 2);
lean_inc_ref(v_fo_247_);
v___x_275_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube___redArg(v_fo_247_, v_tail_272_, v_tail_274_);
v_toSemiringOps_276_ = lean_ctor_get(v_fo_247_, 0);
lean_inc_ref(v_toSemiringOps_276_);
v_fst_277_ = lean_ctor_get(v___x_275_, 0);
v_snd_278_ = lean_ctor_get(v___x_275_, 1);
v_isSharedCheck_300_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_300_ == 0)
{
v___x_280_ = v___x_275_;
v_isShared_281_ = v_isSharedCheck_300_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_snd_278_);
lean_inc(v_fst_277_);
lean_dec(v___x_275_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_300_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v_sub_282_; lean_object* v_one_283_; lean_object* v_add_284_; lean_object* v_mul_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v_rot_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v_eq_296_; lean_object* v___x_298_; 
v_sub_282_ = lean_ctor_get(v_fo_247_, 1);
lean_inc_n(v_sub_282_, 2);
lean_dec_ref(v_fo_247_);
v_one_283_ = lean_ctor_get(v_toSemiringOps_276_, 1);
lean_inc_n(v_one_283_, 2);
v_add_284_ = lean_ctor_get(v_toSemiringOps_276_, 3);
lean_inc_n(v_add_284_, 2);
v_mul_285_ = lean_ctor_get(v_toSemiringOps_276_, 4);
lean_inc_n(v_mul_285_, 7);
lean_dec_ref(v_toSemiringOps_276_);
lean_inc_n(v_head_273_, 2);
v___x_286_ = lean_apply_2(v_sub_282_, v_one_283_, v_head_273_);
lean_inc(v___x_286_);
lean_inc_n(v_head_271_, 2);
v___x_287_ = lean_apply_2(v_mul_285_, v_head_271_, v___x_286_);
lean_inc(v_fst_277_);
v___x_288_ = lean_apply_2(v_mul_285_, v___x_287_, v_fst_277_);
v___x_289_ = lean_apply_2(v_sub_282_, v_one_283_, v_head_271_);
lean_inc(v___x_289_);
v___x_290_ = lean_apply_2(v_mul_285_, v___x_289_, v_head_273_);
v___x_291_ = lean_apply_2(v_mul_285_, v___x_290_, v_snd_278_);
v_rot_292_ = lean_apply_2(v_add_284_, v___x_288_, v___x_291_);
v___x_293_ = lean_apply_2(v_mul_285_, v_head_271_, v_head_273_);
v___x_294_ = lean_apply_2(v_mul_285_, v___x_289_, v___x_286_);
v___x_295_ = lean_apply_2(v_add_284_, v___x_293_, v___x_294_);
v_eq_296_ = lean_apply_2(v_mul_285_, v_fst_277_, v___x_295_);
if (v_isShared_281_ == 0)
{
lean_ctor_set(v___x_280_, 1, v_rot_292_);
lean_ctor_set(v___x_280_, 0, v_eq_296_);
v___x_298_ = v___x_280_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_eq_296_);
lean_ctor_set(v_reuseFailAlloc_299_, 1, v_rot_292_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
}
else
{
lean_dec_ref_known(v_x_248_, 2);
lean_dec(v_y_249_);
goto v___jp_250_;
}
}
v___jp_250_:
{
lean_object* v_toSemiringOps_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_259_; 
v_toSemiringOps_251_ = lean_ctor_get(v_fo_247_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v_fo_247_);
if (v_isSharedCheck_259_ == 0)
{
lean_object* v_unused_260_; 
v_unused_260_ = lean_ctor_get(v_fo_247_, 1);
lean_dec(v_unused_260_);
v___x_253_ = v_fo_247_;
v_isShared_254_ = v_isSharedCheck_259_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_toSemiringOps_251_);
lean_dec(v_fo_247_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_259_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v_zero_255_; lean_object* v___x_257_; 
v_zero_255_ = lean_ctor_get(v_toSemiringOps_251_, 0);
lean_inc_n(v_zero_255_, 2);
lean_dec_ref(v_toSemiringOps_251_);
if (v_isShared_254_ == 0)
{
lean_ctor_set(v___x_253_, 1, v_zero_255_);
lean_ctor_set(v___x_253_, 0, v_zero_255_);
v___x_257_ = v___x_253_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_zero_255_);
lean_ctor_set(v_reuseFailAlloc_258_, 1, v_zero_255_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube(lean_object* v_EF_301_, lean_object* v_fo_302_, lean_object* v_x_303_, lean_object* v_y_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube___redArg(v_fo_302_, v_x_303_, v_y_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(lean_object* v_fo_306_, lean_object* v_e0_307_, lean_object* v_e1_308_, lean_object* v_x_309_){
_start:
{
lean_object* v_toSemiringOps_310_; lean_object* v_sub_311_; lean_object* v_add_312_; lean_object* v_mul_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v_toSemiringOps_310_ = lean_ctor_get(v_fo_306_, 0);
lean_inc_ref(v_toSemiringOps_310_);
v_sub_311_ = lean_ctor_get(v_fo_306_, 1);
lean_inc(v_sub_311_);
lean_dec_ref(v_fo_306_);
v_add_312_ = lean_ctor_get(v_toSemiringOps_310_, 3);
lean_inc(v_add_312_);
v_mul_313_ = lean_ctor_get(v_toSemiringOps_310_, 4);
lean_inc(v_mul_313_);
lean_dec_ref(v_toSemiringOps_310_);
lean_inc(v_e0_307_);
v___x_314_ = lean_apply_2(v_sub_311_, v_e1_308_, v_e0_307_);
v___x_315_ = lean_apply_2(v_mul_313_, v___x_314_, v_x_309_);
v___x_316_ = lean_apply_2(v_add_312_, v___x_315_, v_e0_307_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01(lean_object* v_EF_317_, lean_object* v_fo_318_, lean_object* v_e0_319_, lean_object* v_e1_320_, lean_object* v_x_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateLinearAt01___redArg(v_fo_318_, v_e0_319_, v_e1_320_, v_x_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__0___redArg(lean_object* v_fo_323_, lean_object* v_xi_324_, lean_object* v_a_325_, lean_object* v_a_326_){
_start:
{
if (lean_obj_tag(v_a_325_) == 0)
{
lean_object* v___x_327_; 
lean_dec(v_xi_324_);
lean_dec_ref(v_fo_323_);
v___x_327_ = l_List_reverse___redArg(v_a_326_);
return v___x_327_;
}
else
{
lean_object* v_toSemiringOps_328_; lean_object* v_head_329_; lean_object* v_tail_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_343_; 
v_toSemiringOps_328_ = lean_ctor_get(v_fo_323_, 0);
v_head_329_ = lean_ctor_get(v_a_325_, 0);
v_tail_330_ = lean_ctor_get(v_a_325_, 1);
v_isSharedCheck_343_ = !lean_is_exclusive(v_a_325_);
if (v_isSharedCheck_343_ == 0)
{
v___x_332_ = v_a_325_;
v_isShared_333_ = v_isSharedCheck_343_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_tail_330_);
lean_inc(v_head_329_);
lean_dec(v_a_325_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_343_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v_sub_334_; lean_object* v_one_335_; lean_object* v_mul_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_340_; 
v_sub_334_ = lean_ctor_get(v_fo_323_, 1);
v_one_335_ = lean_ctor_get(v_toSemiringOps_328_, 1);
v_mul_336_ = lean_ctor_get(v_toSemiringOps_328_, 4);
lean_inc(v_sub_334_);
lean_inc(v_xi_324_);
lean_inc(v_one_335_);
v___x_337_ = lean_apply_2(v_sub_334_, v_one_335_, v_xi_324_);
lean_inc(v_mul_336_);
v___x_338_ = lean_apply_2(v_mul_336_, v_head_329_, v___x_337_);
if (v_isShared_333_ == 0)
{
lean_ctor_set(v___x_332_, 1, v_a_326_);
lean_ctor_set(v___x_332_, 0, v___x_338_);
v___x_340_ = v___x_332_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_342_; 
v_reuseFailAlloc_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_342_, 0, v___x_338_);
lean_ctor_set(v_reuseFailAlloc_342_, 1, v_a_326_);
v___x_340_ = v_reuseFailAlloc_342_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
v_a_325_ = v_tail_330_;
v_a_326_ = v___x_340_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__1___redArg(lean_object* v_fo_344_, lean_object* v_xi_345_, lean_object* v_a_346_, lean_object* v_a_347_){
_start:
{
if (lean_obj_tag(v_a_346_) == 0)
{
lean_object* v___x_348_; 
lean_dec(v_xi_345_);
lean_dec_ref(v_fo_344_);
v___x_348_ = l_List_reverse___redArg(v_a_347_);
return v___x_348_;
}
else
{
lean_object* v_toSemiringOps_349_; lean_object* v_head_350_; lean_object* v_tail_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_361_; 
v_toSemiringOps_349_ = lean_ctor_get(v_fo_344_, 0);
v_head_350_ = lean_ctor_get(v_a_346_, 0);
v_tail_351_ = lean_ctor_get(v_a_346_, 1);
v_isSharedCheck_361_ = !lean_is_exclusive(v_a_346_);
if (v_isSharedCheck_361_ == 0)
{
v___x_353_ = v_a_346_;
v_isShared_354_ = v_isSharedCheck_361_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_tail_351_);
lean_inc(v_head_350_);
lean_dec(v_a_346_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_361_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v_mul_355_; lean_object* v___x_356_; lean_object* v___x_358_; 
v_mul_355_ = lean_ctor_get(v_toSemiringOps_349_, 4);
lean_inc(v_mul_355_);
lean_inc(v_xi_345_);
v___x_356_ = lean_apply_2(v_mul_355_, v_head_350_, v_xi_345_);
if (v_isShared_354_ == 0)
{
lean_ctor_set(v___x_353_, 1, v_a_347_);
lean_ctor_set(v___x_353_, 0, v___x_356_);
v___x_358_ = v___x_353_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_356_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v_a_347_);
v___x_358_ = v_reuseFailAlloc_360_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
v_a_346_ = v_tail_351_;
v_a_347_ = v___x_358_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__2___redArg(lean_object* v_fo_362_, lean_object* v_x_363_, lean_object* v_x_364_){
_start:
{
if (lean_obj_tag(v_x_364_) == 0)
{
lean_dec_ref(v_fo_362_);
return v_x_363_;
}
else
{
lean_object* v_head_365_; lean_object* v_tail_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
v_head_365_ = lean_ctor_get(v_x_364_, 0);
lean_inc_n(v_head_365_, 2);
v_tail_366_ = lean_ctor_get(v_x_364_, 1);
lean_inc(v_tail_366_);
lean_dec_ref_known(v_x_364_, 2);
v___x_367_ = lean_box(0);
lean_inc(v_x_363_);
lean_inc_ref_n(v_fo_362_, 2);
v___x_368_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__0___redArg(v_fo_362_, v_head_365_, v_x_363_, v___x_367_);
v___x_369_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__1___redArg(v_fo_362_, v_head_365_, v_x_363_, v___x_367_);
v___x_370_ = l_List_appendTR___redArg(v___x_368_, v___x_369_);
v_x_363_ = v___x_370_;
v_x_364_ = v_tail_366_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial___redArg(lean_object* v_fo_372_, lean_object* v_x_373_){
_start:
{
lean_object* v_toSemiringOps_374_; lean_object* v_one_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v_toSemiringOps_374_ = lean_ctor_get(v_fo_372_, 0);
v_one_375_ = lean_ctor_get(v_toSemiringOps_374_, 1);
v___x_376_ = lean_box(0);
lean_inc(v_one_375_);
v___x_377_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_377_, 0, v_one_375_);
lean_ctor_set(v___x_377_, 1, v___x_376_);
v___x_378_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__2___redArg(v_fo_372_, v___x_377_, v_x_373_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial(lean_object* v_EF_379_, lean_object* v_fo_380_, lean_object* v_x_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial___redArg(v_fo_380_, v_x_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__0(lean_object* v_EF_383_, lean_object* v_fo_384_, lean_object* v_xi_385_, lean_object* v_a_386_, lean_object* v_a_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__0___redArg(v_fo_384_, v_xi_385_, v_a_386_, v_a_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__1(lean_object* v_EF_389_, lean_object* v_fo_390_, lean_object* v_xi_391_, lean_object* v_a_392_, lean_object* v_a_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_swirl_x2drbr_x2dfv_List_mapTR_loop___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__1___redArg(v_fo_390_, v_xi_391_, v_a_392_, v_a_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__2(lean_object* v_EF_395_, lean_object* v_fo_396_, lean_object* v_x_397_, lean_object* v_x_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial_spec__2___redArg(v_fo_396_, v_x_397_, v_x_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle_spec__0___redArg(lean_object* v_fo_400_, lean_object* v_x_401_, lean_object* v_x_402_){
_start:
{
if (lean_obj_tag(v_x_402_) == 0)
{
lean_dec_ref(v_fo_400_);
return v_x_401_;
}
else
{
lean_object* v_head_403_; lean_object* v_toSemiringOps_404_; lean_object* v_tail_405_; lean_object* v_fst_406_; lean_object* v_snd_407_; lean_object* v_sub_408_; lean_object* v_one_409_; lean_object* v_add_410_; lean_object* v_mul_411_; lean_object* v___x_412_; lean_object* v_w0_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v_head_403_ = lean_ctor_get(v_x_402_, 0);
lean_inc(v_head_403_);
v_toSemiringOps_404_ = lean_ctor_get(v_fo_400_, 0);
v_tail_405_ = lean_ctor_get(v_x_402_, 1);
lean_inc(v_tail_405_);
lean_dec_ref_known(v_x_402_, 2);
v_fst_406_ = lean_ctor_get(v_head_403_, 0);
lean_inc_n(v_fst_406_, 3);
v_snd_407_ = lean_ctor_get(v_head_403_, 1);
lean_inc_n(v_snd_407_, 2);
lean_dec(v_head_403_);
v_sub_408_ = lean_ctor_get(v_fo_400_, 1);
v_one_409_ = lean_ctor_get(v_toSemiringOps_404_, 1);
v_add_410_ = lean_ctor_get(v_toSemiringOps_404_, 3);
v_mul_411_ = lean_ctor_get(v_toSemiringOps_404_, 4);
lean_inc_n(v_add_410_, 2);
v___x_412_ = lean_apply_2(v_add_410_, v_fst_406_, v_fst_406_);
lean_inc_n(v_sub_408_, 2);
lean_inc_n(v_one_409_, 2);
v_w0_413_ = lean_apply_2(v_sub_408_, v_one_409_, v___x_412_);
v___x_414_ = lean_apply_2(v_sub_408_, v_one_409_, v_snd_407_);
lean_inc_n(v_mul_411_, 3);
v___x_415_ = lean_apply_2(v_mul_411_, v_w0_413_, v___x_414_);
v___x_416_ = lean_apply_2(v_mul_411_, v_fst_406_, v_snd_407_);
v___x_417_ = lean_apply_2(v_add_410_, v___x_415_, v___x_416_);
v___x_418_ = lean_apply_2(v_mul_411_, v_x_401_, v___x_417_);
v_x_401_ = v___x_418_;
v_x_402_ = v_tail_405_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle___redArg(lean_object* v_fo_420_, lean_object* v_u_421_, lean_object* v_x_422_){
_start:
{
lean_object* v_toSemiringOps_423_; lean_object* v_one_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v_toSemiringOps_423_ = lean_ctor_get(v_fo_420_, 0);
v_one_424_ = lean_ctor_get(v_toSemiringOps_423_, 1);
lean_inc(v_one_424_);
v___x_425_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_u_421_, v_x_422_);
v___x_426_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle_spec__0___redArg(v_fo_420_, v_one_424_, v___x_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle(lean_object* v_EF_427_, lean_object* v_fo_428_, lean_object* v_u_429_, lean_object* v_x_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle___redArg(v_fo_428_, v_u_429_, v_x_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle_spec__0(lean_object* v_EF_432_, lean_object* v_fo_433_, lean_object* v_x_434_, lean_object* v_x_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMobiusEqMle_spec__0___redArg(v_fo_433_, v_x_434_, v_x_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___lam__0(lean_object* v_fo_437_, lean_object* v_head_438_, lean_object* v_l_439_, lean_object* v_h_440_){
_start:
{
lean_object* v_toSemiringOps_441_; lean_object* v_sub_442_; lean_object* v_one_443_; lean_object* v_add_444_; lean_object* v_mul_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v_toSemiringOps_441_ = lean_ctor_get(v_fo_437_, 0);
lean_inc_ref(v_toSemiringOps_441_);
v_sub_442_ = lean_ctor_get(v_fo_437_, 1);
lean_inc(v_sub_442_);
lean_dec_ref(v_fo_437_);
v_one_443_ = lean_ctor_get(v_toSemiringOps_441_, 1);
lean_inc(v_one_443_);
v_add_444_ = lean_ctor_get(v_toSemiringOps_441_, 3);
lean_inc(v_add_444_);
v_mul_445_ = lean_ctor_get(v_toSemiringOps_441_, 4);
lean_inc_n(v_mul_445_, 2);
lean_dec_ref(v_toSemiringOps_441_);
lean_inc(v_head_438_);
v___x_446_ = lean_apply_2(v_sub_442_, v_one_443_, v_head_438_);
v___x_447_ = lean_apply_2(v_mul_445_, v_l_439_, v___x_446_);
v___x_448_ = lean_apply_2(v_mul_445_, v_h_440_, v_head_438_);
v___x_449_ = lean_apply_2(v_add_444_, v___x_447_, v___x_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg(lean_object* v_fo_452_, lean_object* v_x_453_, lean_object* v_x_454_){
_start:
{
if (lean_obj_tag(v_x_454_) == 0)
{
lean_dec_ref(v_fo_452_);
return v_x_453_;
}
else
{
lean_object* v_head_455_; lean_object* v_tail_456_; lean_object* v___f_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v_len_460_; lean_object* v___x_461_; lean_object* v_lo_462_; lean_object* v_hi_463_; lean_object* v___x_464_; 
v_head_455_ = lean_ctor_get(v_x_454_, 0);
lean_inc(v_head_455_);
v_tail_456_ = lean_ctor_get(v_x_454_, 1);
lean_inc(v_tail_456_);
lean_dec_ref_known(v_x_454_, 2);
lean_inc_ref(v_fo_452_);
v___f_457_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___lam__0), 4, 2);
lean_closure_set(v___f_457_, 0, v_fo_452_);
lean_closure_set(v___f_457_, 1, v_head_455_);
v___x_458_ = l_List_lengthTR___redArg(v_x_453_);
v___x_459_ = lean_unsigned_to_nat(1u);
v_len_460_ = lean_nat_shiftr(v___x_458_, v___x_459_);
lean_dec(v___x_458_);
v___x_461_ = ((lean_object*)(lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg___closed__0));
lean_inc(v_len_460_);
lean_inc(v_x_453_);
v_lo_462_ = l___private_Init_Data_List_Impl_0__List_takeTR_go___redArg(v_x_453_, v_x_453_, v_len_460_, v___x_461_);
v_hi_463_ = l_List_drop___redArg(v_len_460_, v_x_453_);
lean_dec(v_x_453_);
v___x_464_ = l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(v___f_457_, v_lo_462_, v_hi_463_, v___x_461_);
v_x_453_ = v___x_464_;
v_x_454_ = v_tail_456_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint___redArg(lean_object* v_fo_466_, lean_object* v_evals_467_, lean_object* v_x_468_){
_start:
{
lean_object* v___x_469_; lean_object* v_folded_470_; 
v___x_469_ = l_List_reverse___redArg(v_x_468_);
lean_inc_ref(v_fo_466_);
v_folded_470_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg(v_fo_466_, v_evals_467_, v___x_469_);
if (lean_obj_tag(v_folded_470_) == 0)
{
lean_object* v_toSemiringOps_471_; lean_object* v_zero_472_; 
v_toSemiringOps_471_ = lean_ctor_get(v_fo_466_, 0);
lean_inc_ref(v_toSemiringOps_471_);
lean_dec_ref(v_fo_466_);
v_zero_472_ = lean_ctor_get(v_toSemiringOps_471_, 0);
lean_inc(v_zero_472_);
lean_dec_ref(v_toSemiringOps_471_);
return v_zero_472_;
}
else
{
lean_object* v_head_473_; 
lean_dec_ref(v_fo_466_);
v_head_473_ = lean_ctor_get(v_folded_470_, 0);
lean_inc(v_head_473_);
lean_dec_ref_known(v_folded_470_, 2);
return v_head_473_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint(lean_object* v_EF_474_, lean_object* v_fo_475_, lean_object* v_evals_476_, lean_object* v_x_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint___redArg(v_fo_475_, v_evals_476_, v_x_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0(lean_object* v_EF_479_, lean_object* v_fo_480_, lean_object* v_x_481_, lean_object* v_x_482_){
_start:
{
lean_object* v___x_483_; 
v___x_483_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalMleEvalsAtPoint_spec__0___redArg(v_fo_480_, v_x_481_, v_x_482_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni_spec__0___redArg(lean_object* v_fo_484_, lean_object* v_x_485_, lean_object* v_x_486_){
_start:
{
if (lean_obj_tag(v_x_486_) == 0)
{
lean_dec_ref(v_fo_484_);
return v_x_485_;
}
else
{
lean_object* v_head_487_; lean_object* v_toRingOps_488_; lean_object* v_toSemiringOps_489_; lean_object* v_tail_490_; lean_object* v_fst_491_; lean_object* v_snd_492_; lean_object* v_sub_493_; lean_object* v_one_494_; lean_object* v_add_495_; lean_object* v_mul_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; 
v_head_487_ = lean_ctor_get(v_x_486_, 0);
lean_inc(v_head_487_);
v_toRingOps_488_ = lean_ctor_get(v_fo_484_, 0);
v_toSemiringOps_489_ = lean_ctor_get(v_toRingOps_488_, 0);
v_tail_490_ = lean_ctor_get(v_x_486_, 1);
lean_inc(v_tail_490_);
lean_dec_ref_known(v_x_486_, 2);
v_fst_491_ = lean_ctor_get(v_head_487_, 0);
lean_inc_n(v_fst_491_, 2);
v_snd_492_ = lean_ctor_get(v_head_487_, 1);
lean_inc_n(v_snd_492_, 2);
lean_dec(v_head_487_);
v_sub_493_ = lean_ctor_get(v_toRingOps_488_, 1);
v_one_494_ = lean_ctor_get(v_toSemiringOps_489_, 1);
v_add_495_ = lean_ctor_get(v_toSemiringOps_489_, 3);
v_mul_496_ = lean_ctor_get(v_toSemiringOps_489_, 4);
lean_inc_n(v_add_495_, 2);
v___x_497_ = lean_apply_2(v_add_495_, v_fst_491_, v_snd_492_);
lean_inc_n(v_mul_496_, 2);
v___x_498_ = lean_apply_2(v_mul_496_, v___x_497_, v_x_485_);
lean_inc_n(v_sub_493_, 2);
lean_inc_n(v_one_494_, 2);
v___x_499_ = lean_apply_2(v_sub_493_, v_fst_491_, v_one_494_);
v___x_500_ = lean_apply_2(v_sub_493_, v_snd_492_, v_one_494_);
v___x_501_ = lean_apply_2(v_mul_496_, v___x_499_, v___x_500_);
v___x_502_ = lean_apply_2(v_add_495_, v___x_498_, v___x_501_);
v_x_485_ = v___x_502_;
v_x_486_ = v_tail_490_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(lean_object* v_fo_504_, lean_object* v_lSkip_505_, lean_object* v_x_506_, lean_object* v_y_507_){
_start:
{
lean_object* v_toRingOps_508_; lean_object* v_toSemiringOps_509_; lean_object* v_inv_510_; lean_object* v_one_511_; lean_object* v_natCast_512_; lean_object* v_mul_513_; lean_object* v_pow_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v_res_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; 
v_toRingOps_508_ = lean_ctor_get(v_fo_504_, 0);
v_toSemiringOps_509_ = lean_ctor_get(v_toRingOps_508_, 0);
v_inv_510_ = lean_ctor_get(v_fo_504_, 1);
lean_inc(v_inv_510_);
v_one_511_ = lean_ctor_get(v_toSemiringOps_509_, 1);
lean_inc(v_one_511_);
v_natCast_512_ = lean_ctor_get(v_toSemiringOps_509_, 2);
lean_inc(v_natCast_512_);
v_mul_513_ = lean_ctor_get(v_toSemiringOps_509_, 4);
lean_inc(v_mul_513_);
v_pow_514_ = lean_ctor_get(v_toSemiringOps_509_, 5);
lean_inc(v_pow_514_);
lean_inc_n(v_lSkip_505_, 2);
lean_inc_ref_n(v_toSemiringOps_509_, 2);
v___x_515_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_509_, v_x_506_, v_lSkip_505_);
v___x_516_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_509_, v_y_507_, v_lSkip_505_);
v___x_517_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v___x_515_, v___x_516_);
v_res_518_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni_spec__0___redArg(v_fo_504_, v_one_511_, v___x_517_);
v___x_519_ = lean_unsigned_to_nat(2u);
v___x_520_ = lean_apply_1(v_natCast_512_, v___x_519_);
v___x_521_ = lean_apply_1(v_inv_510_, v___x_520_);
v___x_522_ = lean_apply_2(v_pow_514_, v___x_521_, v_lSkip_505_);
v___x_523_ = lean_apply_2(v_mul_513_, v_res_518_, v___x_522_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg___boxed(lean_object* v_fo_524_, lean_object* v_lSkip_525_, lean_object* v_x_526_, lean_object* v_y_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_524_, v_lSkip_525_, v_x_526_, v_y_527_);
lean_dec(v_y_527_);
lean_dec(v_x_526_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni(lean_object* v_EF_529_, lean_object* v_fo_530_, lean_object* v_lSkip_531_, lean_object* v_x_532_, lean_object* v_y_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_530_, v_lSkip_531_, v_x_532_, v_y_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___boxed(lean_object* v_EF_535_, lean_object* v_fo_536_, lean_object* v_lSkip_537_, lean_object* v_x_538_, lean_object* v_y_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni(v_EF_535_, v_fo_536_, v_lSkip_537_, v_x_538_, v_y_539_);
lean_dec(v_y_539_);
lean_dec(v_x_538_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni_spec__0(lean_object* v_EF_541_, lean_object* v_fo_542_, lean_object* v_x_543_, lean_object* v_x_544_){
_start:
{
lean_object* v___x_545_; 
v___x_545_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni_spec__0___redArg(v_fo_542_, v_x_543_, v_x_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne_spec__0___redArg(lean_object* v_fo_546_, lean_object* v_x_547_, lean_object* v_x_548_){
_start:
{
if (lean_obj_tag(v_x_548_) == 0)
{
lean_dec_ref(v_fo_546_);
return v_x_547_;
}
else
{
lean_object* v_toRingOps_549_; lean_object* v_toSemiringOps_550_; lean_object* v_head_551_; lean_object* v_tail_552_; lean_object* v_one_553_; lean_object* v_add_554_; lean_object* v_mul_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v_toRingOps_549_ = lean_ctor_get(v_fo_546_, 0);
v_toSemiringOps_550_ = lean_ctor_get(v_toRingOps_549_, 0);
v_head_551_ = lean_ctor_get(v_x_548_, 0);
lean_inc(v_head_551_);
v_tail_552_ = lean_ctor_get(v_x_548_, 1);
lean_inc(v_tail_552_);
lean_dec_ref_known(v_x_548_, 2);
v_one_553_ = lean_ctor_get(v_toSemiringOps_550_, 1);
v_add_554_ = lean_ctor_get(v_toSemiringOps_550_, 3);
v_mul_555_ = lean_ctor_get(v_toSemiringOps_550_, 4);
lean_inc(v_add_554_);
lean_inc(v_one_553_);
v___x_556_ = lean_apply_2(v_add_554_, v_head_551_, v_one_553_);
lean_inc(v_mul_555_);
v___x_557_ = lean_apply_2(v_mul_555_, v_x_547_, v___x_556_);
v_x_547_ = v___x_557_;
v_x_548_ = v_tail_552_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(lean_object* v_fo_559_, lean_object* v_lSkip_560_, lean_object* v_x_561_){
_start:
{
lean_object* v_toRingOps_562_; lean_object* v_toSemiringOps_563_; lean_object* v_inv_564_; lean_object* v_one_565_; lean_object* v_natCast_566_; lean_object* v_mul_567_; lean_object* v_pow_568_; lean_object* v___x_569_; lean_object* v_res_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; 
v_toRingOps_562_ = lean_ctor_get(v_fo_559_, 0);
v_toSemiringOps_563_ = lean_ctor_get(v_toRingOps_562_, 0);
v_inv_564_ = lean_ctor_get(v_fo_559_, 1);
lean_inc(v_inv_564_);
v_one_565_ = lean_ctor_get(v_toSemiringOps_563_, 1);
lean_inc(v_one_565_);
v_natCast_566_ = lean_ctor_get(v_toSemiringOps_563_, 2);
lean_inc(v_natCast_566_);
v_mul_567_ = lean_ctor_get(v_toSemiringOps_563_, 4);
lean_inc(v_mul_567_);
v_pow_568_ = lean_ctor_get(v_toSemiringOps_563_, 5);
lean_inc(v_pow_568_);
lean_inc(v_lSkip_560_);
lean_inc_ref(v_toSemiringOps_563_);
v___x_569_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowersOfTwo___redArg(v_toSemiringOps_563_, v_x_561_, v_lSkip_560_);
v_res_570_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne_spec__0___redArg(v_fo_559_, v_one_565_, v___x_569_);
v___x_571_ = lean_unsigned_to_nat(2u);
v___x_572_ = lean_apply_1(v_natCast_566_, v___x_571_);
v___x_573_ = lean_apply_1(v_inv_564_, v___x_572_);
v___x_574_ = lean_apply_2(v_pow_568_, v___x_573_, v_lSkip_560_);
v___x_575_ = lean_apply_2(v_mul_567_, v_res_570_, v___x_574_);
return v___x_575_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg___boxed(lean_object* v_fo_576_, lean_object* v_lSkip_577_, lean_object* v_x_578_){
_start:
{
lean_object* v_res_579_; 
v_res_579_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(v_fo_576_, v_lSkip_577_, v_x_578_);
lean_dec(v_x_578_);
return v_res_579_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne(lean_object* v_EF_580_, lean_object* v_fo_581_, lean_object* v_lSkip_582_, lean_object* v_x_583_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(v_fo_581_, v_lSkip_582_, v_x_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___boxed(lean_object* v_EF_585_, lean_object* v_fo_586_, lean_object* v_lSkip_587_, lean_object* v_x_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne(v_EF_585_, v_fo_586_, v_lSkip_587_, v_x_588_);
lean_dec(v_x_588_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne_spec__0(lean_object* v_EF_590_, lean_object* v_fo_591_, lean_object* v_x_592_, lean_object* v_x_593_){
_start:
{
lean_object* v___x_594_; 
v___x_594_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne_spec__0___redArg(v_fo_591_, v_x_592_, v_x_593_);
return v___x_594_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___closed__0(void){
_start:
{
lean_object* v___x_595_; lean_object* v___x_596_; 
v___x_595_ = lean_unsigned_to_nat(0u);
v___x_596_ = lean_nat_to_int(v___x_595_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg(lean_object* v_fo_597_, lean_object* v_lSkip_598_, lean_object* v_n_599_, lean_object* v_z_600_){
_start:
{
lean_object* v___x_601_; uint8_t v___x_602_; 
v___x_601_ = lean_obj_once(&lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___closed__0, &lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___closed__0_once, _init_lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___closed__0);
v___x_602_ = lean_int_dec_lt(v_n_599_, v___x_601_);
if (v___x_602_ == 0)
{
lean_object* v_toRingOps_603_; lean_object* v_toSemiringOps_604_; lean_object* v_one_605_; 
lean_dec(v_lSkip_598_);
v_toRingOps_603_ = lean_ctor_get(v_fo_597_, 0);
lean_inc_ref(v_toRingOps_603_);
lean_dec_ref(v_fo_597_);
v_toSemiringOps_604_ = lean_ctor_get(v_toRingOps_603_, 0);
lean_inc_ref(v_toSemiringOps_604_);
lean_dec_ref(v_toRingOps_603_);
v_one_605_ = lean_ctor_get(v_toSemiringOps_604_, 1);
lean_inc(v_one_605_);
lean_dec_ref(v_toSemiringOps_604_);
return v_one_605_;
}
else
{
lean_object* v_toRingOps_606_; lean_object* v_toSemiringOps_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v_shift_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; 
v_toRingOps_606_ = lean_ctor_get(v_fo_597_, 0);
v_toSemiringOps_607_ = lean_ctor_get(v_toRingOps_606_, 0);
v___x_608_ = lean_nat_to_int(v_lSkip_598_);
v___x_609_ = lean_int_add(v___x_608_, v_n_599_);
lean_dec(v___x_608_);
v_shift_610_ = l_Int_toNat(v___x_609_);
lean_dec(v___x_609_);
v___x_611_ = lean_nat_abs(v_n_599_);
lean_inc_ref(v_toSemiringOps_607_);
v___x_612_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_expPowerOfTwo___redArg(v_toSemiringOps_607_, v_z_600_, v_shift_610_);
lean_dec(v_shift_610_);
v___x_613_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(v_fo_597_, v___x_611_, v___x_612_);
lean_dec(v___x_612_);
return v___x_613_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg___boxed(lean_object* v_fo_614_, lean_object* v_lSkip_615_, lean_object* v_n_616_, lean_object* v_z_617_){
_start:
{
lean_object* v_res_618_; 
v_res_618_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg(v_fo_614_, v_lSkip_615_, v_n_616_, v_z_617_);
lean_dec(v_z_617_);
lean_dec(v_n_616_);
return v_res_618_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni(lean_object* v_EF_619_, lean_object* v_fo_620_, lean_object* v_lSkip_621_, lean_object* v_n_622_, lean_object* v_z_623_){
_start:
{
lean_object* v___x_624_; 
v___x_624_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___redArg(v_fo_620_, v_lSkip_621_, v_n_622_, v_z_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni___boxed(lean_object* v_EF_625_, lean_object* v_fo_626_, lean_object* v_lSkip_627_, lean_object* v_n_628_, lean_object* v_z_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalInUni(v_EF_625_, v_fo_626_, v_lSkip_627_, v_n_628_, v_z_629_);
lean_dec(v_z_629_);
lean_dec(v_n_628_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism___redArg(lean_object* v_fo_631_, lean_object* v_lSkip_632_, lean_object* v_x_633_, lean_object* v_y_634_){
_start:
{
if (lean_obj_tag(v_x_633_) == 1)
{
if (lean_obj_tag(v_y_634_) == 1)
{
lean_object* v_toRingOps_639_; lean_object* v_toSemiringOps_640_; lean_object* v_head_641_; lean_object* v_tail_642_; lean_object* v_head_643_; lean_object* v_tail_644_; lean_object* v_mul_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; 
v_toRingOps_639_ = lean_ctor_get(v_fo_631_, 0);
lean_inc_ref(v_toRingOps_639_);
v_toSemiringOps_640_ = lean_ctor_get(v_toRingOps_639_, 0);
v_head_641_ = lean_ctor_get(v_x_633_, 0);
lean_inc(v_head_641_);
v_tail_642_ = lean_ctor_get(v_x_633_, 1);
lean_inc(v_tail_642_);
lean_dec_ref_known(v_x_633_, 2);
v_head_643_ = lean_ctor_get(v_y_634_, 0);
lean_inc(v_head_643_);
v_tail_644_ = lean_ctor_get(v_y_634_, 1);
lean_inc(v_tail_644_);
lean_dec_ref_known(v_y_634_, 2);
v_mul_645_ = lean_ctor_get(v_toSemiringOps_640_, 4);
lean_inc(v_mul_645_);
v___x_646_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_631_, v_lSkip_632_, v_head_641_, v_head_643_);
lean_dec(v_head_643_);
lean_dec(v_head_641_);
v___x_647_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqMle___redArg(v_toRingOps_639_, v_tail_642_, v_tail_644_);
v___x_648_ = lean_apply_2(v_mul_645_, v___x_646_, v___x_647_);
return v___x_648_;
}
else
{
lean_dec_ref_known(v_x_633_, 2);
lean_dec(v_y_634_);
lean_dec(v_lSkip_632_);
goto v___jp_635_;
}
}
else
{
lean_dec(v_y_634_);
lean_dec(v_x_633_);
lean_dec(v_lSkip_632_);
goto v___jp_635_;
}
v___jp_635_:
{
lean_object* v_toRingOps_636_; lean_object* v_toSemiringOps_637_; lean_object* v_zero_638_; 
v_toRingOps_636_ = lean_ctor_get(v_fo_631_, 0);
lean_inc_ref(v_toRingOps_636_);
lean_dec_ref(v_fo_631_);
v_toSemiringOps_637_ = lean_ctor_get(v_toRingOps_636_, 0);
lean_inc_ref(v_toSemiringOps_637_);
lean_dec_ref(v_toRingOps_636_);
v_zero_638_ = lean_ctor_get(v_toSemiringOps_637_, 0);
lean_inc(v_zero_638_);
lean_dec_ref(v_toSemiringOps_637_);
return v_zero_638_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism(lean_object* v_EF_649_, lean_object* v_fo_650_, lean_object* v_lSkip_651_, lean_object* v_x_652_, lean_object* v_y_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqPrism___redArg(v_fo_650_, v_lSkip_651_, v_x_652_, v_y_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(lean_object* v_fo_655_, lean_object* v_s0_656_, lean_object* v_s1_657_, lean_object* v_s2_658_, lean_object* v_x_659_){
_start:
{
lean_object* v_toRingOps_660_; lean_object* v_toSemiringOps_661_; lean_object* v_inv_662_; lean_object* v_sub_663_; lean_object* v_natCast_664_; lean_object* v_add_665_; lean_object* v_mul_666_; lean_object* v_slope1_667_; lean_object* v_slope2_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v_p_673_; lean_object* v_q_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; 
v_toRingOps_660_ = lean_ctor_get(v_fo_655_, 0);
lean_inc_ref(v_toRingOps_660_);
v_toSemiringOps_661_ = lean_ctor_get(v_toRingOps_660_, 0);
lean_inc_ref(v_toSemiringOps_661_);
v_inv_662_ = lean_ctor_get(v_fo_655_, 1);
lean_inc(v_inv_662_);
lean_dec_ref(v_fo_655_);
v_sub_663_ = lean_ctor_get(v_toRingOps_660_, 1);
lean_inc_n(v_sub_663_, 4);
lean_dec_ref(v_toRingOps_660_);
v_natCast_664_ = lean_ctor_get(v_toSemiringOps_661_, 2);
lean_inc(v_natCast_664_);
v_add_665_ = lean_ctor_get(v_toSemiringOps_661_, 3);
lean_inc_n(v_add_665_, 2);
v_mul_666_ = lean_ctor_get(v_toSemiringOps_661_, 4);
lean_inc_n(v_mul_666_, 3);
lean_dec_ref(v_toSemiringOps_661_);
lean_inc(v_s0_656_);
lean_inc(v_s1_657_);
v_slope1_667_ = lean_apply_2(v_sub_663_, v_s1_657_, v_s0_656_);
v_slope2_668_ = lean_apply_2(v_sub_663_, v_s2_658_, v_s1_657_);
lean_inc(v_slope1_667_);
v___x_669_ = lean_apply_2(v_sub_663_, v_slope2_668_, v_slope1_667_);
v___x_670_ = lean_unsigned_to_nat(2u);
v___x_671_ = lean_apply_1(v_natCast_664_, v___x_670_);
v___x_672_ = lean_apply_1(v_inv_662_, v___x_671_);
v_p_673_ = lean_apply_2(v_mul_666_, v___x_669_, v___x_672_);
lean_inc(v_p_673_);
v_q_674_ = lean_apply_2(v_sub_663_, v_slope1_667_, v_p_673_);
lean_inc(v_x_659_);
v___x_675_ = lean_apply_2(v_mul_666_, v_p_673_, v_x_659_);
v___x_676_ = lean_apply_2(v_add_665_, v___x_675_, v_q_674_);
v___x_677_ = lean_apply_2(v_mul_666_, v___x_676_, v_x_659_);
v___x_678_ = lean_apply_2(v_add_665_, v___x_677_, v_s0_656_);
return v___x_678_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012(lean_object* v_EF_679_, lean_object* v_fo_680_, lean_object* v_s0_681_, lean_object* v_s1_682_, lean_object* v_s2_683_, lean_object* v_x_684_){
_start:
{
lean_object* v___x_685_; 
v___x_685_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateQuadraticAt012___redArg(v_fo_680_, v_s0_681_, v_s1_682_, v_s2_683_, v_x_684_);
return v___x_685_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(lean_object* v_fo_686_, lean_object* v_e0_687_, lean_object* v_e1_688_, lean_object* v_e2_689_, lean_object* v_e3_690_, lean_object* v_x_691_){
_start:
{
lean_object* v_toRingOps_692_; lean_object* v_toSemiringOps_693_; lean_object* v_inv_694_; lean_object* v_sub_695_; lean_object* v_one_696_; lean_object* v_natCast_697_; lean_object* v_add_698_; lean_object* v_mul_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v_half_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v_sixth_705_; lean_object* v_delta1_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v_delta2_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v_delta3_716_; lean_object* v_xMinus1_717_; lean_object* v_xMinus2_718_; lean_object* v_term2_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v_term3_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v_term4_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; 
v_toRingOps_692_ = lean_ctor_get(v_fo_686_, 0);
lean_inc_ref(v_toRingOps_692_);
v_toSemiringOps_693_ = lean_ctor_get(v_toRingOps_692_, 0);
lean_inc_ref(v_toSemiringOps_693_);
v_inv_694_ = lean_ctor_get(v_fo_686_, 1);
lean_inc_n(v_inv_694_, 2);
lean_dec_ref(v_fo_686_);
v_sub_695_ = lean_ctor_get(v_toRingOps_692_, 1);
lean_inc_n(v_sub_695_, 6);
lean_dec_ref(v_toRingOps_692_);
v_one_696_ = lean_ctor_get(v_toSemiringOps_693_, 1);
lean_inc(v_one_696_);
v_natCast_697_ = lean_ctor_get(v_toSemiringOps_693_, 2);
lean_inc_n(v_natCast_697_, 3);
v_add_698_ = lean_ctor_get(v_toSemiringOps_693_, 3);
lean_inc_n(v_add_698_, 5);
v_mul_699_ = lean_ctor_get(v_toSemiringOps_693_, 4);
lean_inc_n(v_mul_699_, 11);
lean_dec_ref(v_toSemiringOps_693_);
v___x_700_ = lean_unsigned_to_nat(2u);
v___x_701_ = lean_apply_1(v_natCast_697_, v___x_700_);
lean_inc_n(v___x_701_, 2);
v_half_702_ = lean_apply_1(v_inv_694_, v___x_701_);
v___x_703_ = lean_unsigned_to_nat(6u);
v___x_704_ = lean_apply_1(v_natCast_697_, v___x_703_);
v_sixth_705_ = lean_apply_1(v_inv_694_, v___x_704_);
lean_inc_n(v_e0_687_, 3);
lean_inc_n(v_e1_688_, 2);
v_delta1_706_ = lean_apply_2(v_sub_695_, v_e1_688_, v_e0_687_);
v___x_707_ = lean_apply_2(v_mul_699_, v___x_701_, v_e1_688_);
lean_inc(v_e2_689_);
v___x_708_ = lean_apply_2(v_sub_695_, v_e2_689_, v___x_707_);
v_delta2_709_ = lean_apply_2(v_add_698_, v___x_708_, v_e0_687_);
v___x_710_ = lean_unsigned_to_nat(3u);
v___x_711_ = lean_apply_1(v_natCast_697_, v___x_710_);
lean_inc(v___x_711_);
v___x_712_ = lean_apply_2(v_mul_699_, v___x_711_, v_e2_689_);
v___x_713_ = lean_apply_2(v_sub_695_, v_e3_690_, v___x_712_);
v___x_714_ = lean_apply_2(v_mul_699_, v___x_711_, v_e1_688_);
v___x_715_ = lean_apply_2(v_add_698_, v___x_713_, v___x_714_);
v_delta3_716_ = lean_apply_2(v_sub_695_, v___x_715_, v_e0_687_);
lean_inc_n(v_x_691_, 4);
v_xMinus1_717_ = lean_apply_2(v_sub_695_, v_x_691_, v_one_696_);
v_xMinus2_718_ = lean_apply_2(v_sub_695_, v_x_691_, v___x_701_);
v_term2_719_ = lean_apply_2(v_mul_699_, v_delta1_706_, v_x_691_);
v___x_720_ = lean_apply_2(v_mul_699_, v_delta2_709_, v_half_702_);
v___x_721_ = lean_apply_2(v_mul_699_, v___x_720_, v_x_691_);
lean_inc(v_xMinus1_717_);
v_term3_722_ = lean_apply_2(v_mul_699_, v___x_721_, v_xMinus1_717_);
v___x_723_ = lean_apply_2(v_mul_699_, v_delta3_716_, v_sixth_705_);
v___x_724_ = lean_apply_2(v_mul_699_, v___x_723_, v_x_691_);
v___x_725_ = lean_apply_2(v_mul_699_, v___x_724_, v_xMinus1_717_);
v_term4_726_ = lean_apply_2(v_mul_699_, v___x_725_, v_xMinus2_718_);
v___x_727_ = lean_apply_2(v_add_698_, v_e0_687_, v_term2_719_);
v___x_728_ = lean_apply_2(v_add_698_, v___x_727_, v_term3_722_);
v___x_729_ = lean_apply_2(v_add_698_, v___x_728_, v_term4_726_);
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123(lean_object* v_EF_730_, lean_object* v_fo_731_, lean_object* v_e0_732_, lean_object* v_e1_733_, lean_object* v_e2_734_, lean_object* v_e3_735_, lean_object* v_x_736_){
_start:
{
lean_object* v___x_737_; 
v___x_737_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_interpolateCubicAt0123___redArg(v_fo_731_, v_e0_732_, v_e1_733_, v_e2_734_, v_e3_735_, v_x_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg(lean_object* v___x_738_, lean_object* v_fo_739_, lean_object* v_lSkip_740_, lean_object* v_z_741_, lean_object* v_x_742_, lean_object* v_x_743_){
_start:
{
if (lean_obj_tag(v_x_743_) == 0)
{
lean_dec(v_lSkip_740_);
lean_dec_ref(v_fo_739_);
lean_dec_ref(v___x_738_);
return v_x_742_;
}
else
{
lean_object* v_head_744_; lean_object* v_toSemiringOps_745_; lean_object* v_tail_746_; lean_object* v_fst_747_; lean_object* v_snd_748_; lean_object* v_add_749_; lean_object* v_mul_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
v_head_744_ = lean_ctor_get(v_x_743_, 0);
lean_inc(v_head_744_);
v_toSemiringOps_745_ = lean_ctor_get(v___x_738_, 0);
v_tail_746_ = lean_ctor_get(v_x_743_, 1);
lean_inc(v_tail_746_);
lean_dec_ref_known(v_x_743_, 2);
v_fst_747_ = lean_ctor_get(v_head_744_, 0);
lean_inc(v_fst_747_);
v_snd_748_ = lean_ctor_get(v_head_744_, 1);
lean_inc(v_snd_748_);
lean_dec(v_head_744_);
v_add_749_ = lean_ctor_get(v_toSemiringOps_745_, 3);
v_mul_750_ = lean_ctor_get(v_toSemiringOps_745_, 4);
lean_inc(v_lSkip_740_);
lean_inc_ref(v_fo_739_);
v___x_751_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_739_, v_lSkip_740_, v_z_741_, v_fst_747_);
lean_dec(v_fst_747_);
lean_inc(v_mul_750_);
v___x_752_ = lean_apply_2(v_mul_750_, v___x_751_, v_snd_748_);
lean_inc(v_add_749_);
v___x_753_ = lean_apply_2(v_add_749_, v_x_742_, v___x_752_);
v_x_742_ = v___x_753_;
v_x_743_ = v_tail_746_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg___boxed(lean_object* v___x_755_, lean_object* v_fo_756_, lean_object* v_lSkip_757_, lean_object* v_z_758_, lean_object* v_x_759_, lean_object* v_x_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg(v___x_755_, v_fo_756_, v_lSkip_757_, v_z_758_, v_x_759_, v_x_760_);
lean_dec(v_z_758_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(lean_object* v_fo_762_, lean_object* v_lSkip_763_, lean_object* v_omegaSkipPows_764_, lean_object* v_xi1_765_, lean_object* v_z_766_){
_start:
{
lean_object* v_toRingOps_767_; lean_object* v_toSemiringOps_768_; lean_object* v_zero_769_; lean_object* v_eqXiEvals_770_; lean_object* v___x_771_; lean_object* v___x_772_; 
v_toRingOps_767_ = lean_ctor_get(v_fo_762_, 0);
lean_inc_ref_n(v_toRingOps_767_, 2);
v_toSemiringOps_768_ = lean_ctor_get(v_toRingOps_767_, 0);
v_zero_769_ = lean_ctor_get(v_toSemiringOps_768_, 0);
lean_inc(v_zero_769_);
v_eqXiEvals_770_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalsEqHypercubeSerial___redArg(v_toRingOps_767_, v_xi1_765_);
v___x_771_ = l_List_zipWith___at___00List_zip_spec__0___redArg(v_omegaSkipPows_764_, v_eqXiEvals_770_);
v___x_772_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg(v_toRingOps_767_, v_fo_762_, v_lSkip_763_, v_z_766_, v_zero_769_, v___x_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg___boxed(lean_object* v_fo_773_, lean_object* v_lSkip_774_, lean_object* v_omegaSkipPows_775_, lean_object* v_xi1_776_, lean_object* v_z_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(v_fo_773_, v_lSkip_774_, v_omegaSkipPows_775_, v_xi1_776_, v_z_777_);
lean_dec(v_z_777_);
return v_res_778_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni(lean_object* v_EF_779_, lean_object* v_fo_780_, lean_object* v_lSkip_781_, lean_object* v_omegaSkipPows_782_, lean_object* v_xi1_783_, lean_object* v_z_784_){
_start:
{
lean_object* v___x_785_; 
v___x_785_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___redArg(v_fo_780_, v_lSkip_781_, v_omegaSkipPows_782_, v_xi1_783_, v_z_784_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni___boxed(lean_object* v_EF_786_, lean_object* v_fo_787_, lean_object* v_lSkip_788_, lean_object* v_omegaSkipPows_789_, lean_object* v_xi1_790_, lean_object* v_z_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni(v_EF_786_, v_fo_787_, v_lSkip_788_, v_omegaSkipPows_789_, v_xi1_790_, v_z_791_);
lean_dec(v_z_791_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0(lean_object* v_EF_793_, lean_object* v___x_794_, lean_object* v_fo_795_, lean_object* v_lSkip_796_, lean_object* v_z_797_, lean_object* v_x_798_, lean_object* v_x_799_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___redArg(v___x_794_, v_fo_795_, v_lSkip_796_, v_z_797_, v_x_798_, v_x_799_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0___boxed(lean_object* v_EF_801_, lean_object* v___x_802_, lean_object* v_fo_803_, lean_object* v_lSkip_804_, lean_object* v_z_805_, lean_object* v_x_806_, lean_object* v_x_807_){
_start:
{
lean_object* v_res_808_; 
v_res_808_ = lp_swirl_x2drbr_x2dfv_List_foldl___at___00Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqSharpUni_spec__0(v_EF_801_, v___x_802_, v_fo_803_, v_lSkip_804_, v_z_805_, v_x_806_, v_x_807_);
lean_dec(v_z_805_);
return v_res_808_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism___redArg(lean_object* v_fo_809_, lean_object* v_inst_810_, lean_object* v_lSkip_811_, lean_object* v_x_812_, lean_object* v_y_813_){
_start:
{
if (lean_obj_tag(v_x_812_) == 1)
{
if (lean_obj_tag(v_y_813_) == 1)
{
lean_object* v_head_818_; lean_object* v_tail_819_; lean_object* v_head_820_; lean_object* v_tail_821_; lean_object* v_toRingOps_822_; lean_object* v___x_823_; lean_object* v_toSemiringOps_824_; lean_object* v_fst_825_; lean_object* v_snd_826_; lean_object* v_sub_827_; lean_object* v_add_828_; lean_object* v_mul_829_; lean_object* v_omega_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; 
v_head_818_ = lean_ctor_get(v_x_812_, 0);
lean_inc(v_head_818_);
v_tail_819_ = lean_ctor_get(v_x_812_, 1);
lean_inc(v_tail_819_);
lean_dec_ref_known(v_x_812_, 2);
v_head_820_ = lean_ctor_get(v_y_813_, 0);
lean_inc(v_head_820_);
v_tail_821_ = lean_ctor_get(v_y_813_, 1);
lean_inc(v_tail_821_);
lean_dec_ref_known(v_y_813_, 2);
v_toRingOps_822_ = lean_ctor_get(v_fo_809_, 0);
lean_inc_ref(v_toRingOps_822_);
v___x_823_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqRotCube___redArg(v_toRingOps_822_, v_tail_819_, v_tail_821_);
v_toSemiringOps_824_ = lean_ctor_get(v_toRingOps_822_, 0);
v_fst_825_ = lean_ctor_get(v___x_823_, 0);
lean_inc_n(v_fst_825_, 2);
v_snd_826_ = lean_ctor_get(v___x_823_, 1);
lean_inc(v_snd_826_);
lean_dec_ref(v___x_823_);
v_sub_827_ = lean_ctor_get(v_toRingOps_822_, 1);
lean_inc(v_sub_827_);
v_add_828_ = lean_ctor_get(v_toSemiringOps_824_, 3);
lean_inc(v_add_828_);
v_mul_829_ = lean_ctor_get(v_toSemiringOps_824_, 4);
lean_inc_n(v_mul_829_, 4);
lean_inc_n(v_lSkip_811_, 3);
v_omega_830_ = lean_apply_1(v_inst_810_, v_lSkip_811_);
v___x_831_ = lean_apply_2(v_mul_829_, v_head_820_, v_omega_830_);
lean_inc_ref_n(v_fo_809_, 2);
v___x_832_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUni___redArg(v_fo_809_, v_lSkip_811_, v_head_818_, v___x_831_);
v___x_833_ = lean_apply_2(v_mul_829_, v___x_832_, v_fst_825_);
v___x_834_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(v_fo_809_, v_lSkip_811_, v_head_818_);
lean_dec(v_head_818_);
v___x_835_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalEqUniAtOne___redArg(v_fo_809_, v_lSkip_811_, v___x_831_);
lean_dec(v___x_831_);
v___x_836_ = lean_apply_2(v_mul_829_, v___x_834_, v___x_835_);
v___x_837_ = lean_apply_2(v_sub_827_, v_snd_826_, v_fst_825_);
v___x_838_ = lean_apply_2(v_mul_829_, v___x_836_, v___x_837_);
v___x_839_ = lean_apply_2(v_add_828_, v___x_833_, v___x_838_);
return v___x_839_;
}
else
{
lean_dec_ref_known(v_x_812_, 2);
lean_dec(v_y_813_);
lean_dec(v_lSkip_811_);
lean_dec(v_inst_810_);
goto v___jp_814_;
}
}
else
{
lean_dec(v_y_813_);
lean_dec(v_x_812_);
lean_dec(v_lSkip_811_);
lean_dec(v_inst_810_);
goto v___jp_814_;
}
v___jp_814_:
{
lean_object* v_toRingOps_815_; lean_object* v_toSemiringOps_816_; lean_object* v_zero_817_; 
v_toRingOps_815_ = lean_ctor_get(v_fo_809_, 0);
lean_inc_ref(v_toRingOps_815_);
lean_dec_ref(v_fo_809_);
v_toSemiringOps_816_ = lean_ctor_get(v_toRingOps_815_, 0);
lean_inc_ref(v_toSemiringOps_816_);
lean_dec_ref(v_toRingOps_815_);
v_zero_817_ = lean_ctor_get(v_toSemiringOps_816_, 0);
lean_inc(v_zero_817_);
lean_dec_ref(v_toSemiringOps_816_);
return v_zero_817_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism(lean_object* v_EF_840_, lean_object* v_fo_841_, lean_object* v_inst_842_, lean_object* v_lSkip_843_, lean_object* v_x_844_, lean_object* v_y_845_){
_start:
{
lean_object* v___x_846_; 
v___x_846_ = lp_swirl_x2drbr_x2dfv_Swirl_Protocol_Noninteractive_Runtime_PolyCommon_evalRotKernelPrism___redArg(v_fo_841_, v_inst_842_, v_lSkip_843_, v_x_844_, v_y_845_);
return v___x_846_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_FieldOps(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dfv_Swirl_Spec_ReferenceVerifier_Runtime_PolyCommon(uint8_t builtin) {
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
res = initialize_swirl_x2drbr_x2dfv_Fundamentals_Spec_FieldOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
