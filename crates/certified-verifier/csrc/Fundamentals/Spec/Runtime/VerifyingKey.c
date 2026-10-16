// Lean compiler output
// Module: Fundamentals.Spec.Runtime.VerifyingKey
// Imports: public import Init public meta import Init public import Fundamentals.Spec.Runtime.Config public import Fundamentals.Spec.FieldOps
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
lean_object* l_Nat_reprFast(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Option_repr___at___00Lean_Meta_instReprConfig__1_repr_spec__0(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Bool_repr___redArg(uint8_t);
lean_object* l_instReprNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* l_List_repr_x27___redArg(lean_object*, lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Runtime_instReprSystemParams_repr___redArg(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* l_Option_repr___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_swirl_x2dfv_Fundamentals_Runtime_instDecidableEqSystemParams_decEq(lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__2_value;
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__5_value;
static const lean_string_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__7;
static lean_once_cell_t lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__8;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "preprocessed"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "cachedMains"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__9_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "commonMain"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__11_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__12_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__14_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__14_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_TraceWidth_totalWidth(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_TraceWidth_totalWidth___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "coefficients"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "threshold"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__5_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__6;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint___closed__0_value;
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "width"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "numPublicValues"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "needRot"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__9_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams___closed__0_value;
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "commit"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "hypercubeDim"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "stackingWidth"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__11;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_preprocessed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_preprocessed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_main_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_main_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_publicInput_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_publicInput_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_challenge_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_challenge_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Fundamentals.VerifyingKey.Entry.publicInput"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Fundamentals.VerifyingKey.Entry.challenge"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Fundamentals.VerifyingKey.Entry.preprocessed"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__6_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Fundamentals.VerifyingKey.Entry.main"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__9_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__10_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__11_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry___closed__0_value;
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "entry"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "index"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_variable_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_variable_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isFirstRow_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isFirstRow_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isLastRow_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isLastRow_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isTransition_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isTransition_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_constant_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_constant_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_add_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_add_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_sub_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_sub_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_neg_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_neg_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_mul_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_mul_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_variable_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_variable_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isFirstRow_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isFirstRow_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isLastRow_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isLastRow_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isTransition_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isTransition_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_constant_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_constant_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_add_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_add_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_sub_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_sub_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_neg_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_neg_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_mul_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_mul_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.isTransition"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.isLastRow"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.isFirstRow"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__5_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.variable"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__7_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__8_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.constant"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__9_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__10_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__11_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.add"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__12_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__12_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__13_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__13_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__14_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.sub"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__15_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__15_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__16_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__17_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.neg"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__18 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__18_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__18_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__19 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__19_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__19_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__20 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__20_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Fundamentals.VerifyingKey.SymbolicExpressionNode.mul"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__21 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__21_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__21_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__22 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__22_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__22_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__23 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__23_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instReprNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "nodes"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__4_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "constraintIdx"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction___boxed(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "message"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "count"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__5_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "busIndex"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__7_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__8;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "countWeight"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__9_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constraints"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "interactions"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalNodes___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalNodes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalConstraints___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalConstraints(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "preprocessedData"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "params"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "symbolicConstraints"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__9;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "maxConstraintDegree"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__11_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "isRequired"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__12_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__12_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__13_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "unusedVariables"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__14_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__15_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__0_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__1_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "perAir"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "traceHeightConstraints"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__5_value;
static lean_once_cell_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__6;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_VerifierSinglePreprocessedData_mapDigest___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_VerifierSinglePreprocessedData_mapDigest(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_mapDigest___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_mapDigest(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "inner"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "preHash"};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_mapDigest___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_mapDigest(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalConstraints___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalConstraints(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalInteractions___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalInteractions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
lean_object* v_preprocessed_3_; lean_object* v_cachedMains_4_; lean_object* v_commonMain_5_; lean_object* v_preprocessed_6_; lean_object* v_cachedMains_7_; lean_object* v_commonMain_8_; lean_object* v___x_9_; uint8_t v___x_10_; 
v_preprocessed_3_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_preprocessed_3_);
v_cachedMains_4_ = lean_ctor_get(v_x_1_, 1);
lean_inc(v_cachedMains_4_);
v_commonMain_5_ = lean_ctor_get(v_x_1_, 2);
lean_inc(v_commonMain_5_);
lean_dec_ref(v_x_1_);
v_preprocessed_6_ = lean_ctor_get(v_x_2_, 0);
lean_inc(v_preprocessed_6_);
v_cachedMains_7_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_cachedMains_7_);
v_commonMain_8_ = lean_ctor_get(v_x_2_, 2);
lean_inc(v_commonMain_8_);
lean_dec_ref(v_x_2_);
v___x_9_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
lean_inc_ref(v___x_9_);
v___x_10_ = l_Option_instDecidableEq___redArg(v___x_9_, v_preprocessed_3_, v_preprocessed_6_);
if (v___x_10_ == 0)
{
lean_dec_ref(v___x_9_);
lean_dec(v_commonMain_8_);
lean_dec(v_cachedMains_7_);
lean_dec(v_commonMain_5_);
lean_dec(v_cachedMains_4_);
return v___x_10_;
}
else
{
uint8_t v___x_11_; 
v___x_11_ = l_instDecidableEqList___redArg(v___x_9_, v_cachedMains_4_, v_cachedMains_7_);
if (v___x_11_ == 0)
{
lean_dec(v_commonMain_8_);
lean_dec(v_commonMain_5_);
return v___x_11_;
}
else
{
uint8_t v___x_12_; 
v___x_12_ = lean_nat_dec_eq(v_commonMain_5_, v_commonMain_8_);
lean_dec(v_commonMain_8_);
lean_dec(v_commonMain_5_);
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq___boxed(lean_object* v_x_13_, lean_object* v_x_14_){
_start:
{
uint8_t v_res_15_; lean_object* v_r_16_; 
v_res_15_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq(v_x_13_, v_x_14_);
v_r_16_ = lean_box(v_res_15_);
return v_r_16_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth(lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v___x_19_; 
v___x_19_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq(v_x_17_, v_x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth___boxed(lean_object* v_x_20_, lean_object* v_x_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth(v_x_20_, v_x_21_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0___lam__0(lean_object* v___y_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = l_Nat_reprFast(v___y_24_);
v___x_26_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_26_, 0, v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0_spec__1_spec__2(lean_object* v_x_27_, lean_object* v_x_28_, lean_object* v_x_29_){
_start:
{
if (lean_obj_tag(v_x_29_) == 0)
{
lean_dec(v_x_27_);
return v_x_28_;
}
else
{
lean_object* v_head_30_; lean_object* v_tail_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_42_; 
v_head_30_ = lean_ctor_get(v_x_29_, 0);
v_tail_31_ = lean_ctor_get(v_x_29_, 1);
v_isSharedCheck_42_ = !lean_is_exclusive(v_x_29_);
if (v_isSharedCheck_42_ == 0)
{
v___x_33_ = v_x_29_;
v_isShared_34_ = v_isSharedCheck_42_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_tail_31_);
lean_inc(v_head_30_);
lean_dec(v_x_29_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_42_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_36_; 
lean_inc(v_x_27_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 5);
lean_ctor_set(v___x_33_, 1, v_x_27_);
lean_ctor_set(v___x_33_, 0, v_x_28_);
v___x_36_ = v___x_33_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v_x_28_);
lean_ctor_set(v_reuseFailAlloc_41_, 1, v_x_27_);
v___x_36_ = v_reuseFailAlloc_41_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = l_Nat_reprFast(v_head_30_);
v___x_38_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
v___x_39_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_39_, 0, v___x_36_);
lean_ctor_set(v___x_39_, 1, v___x_38_);
v_x_28_ = v___x_39_;
v_x_29_ = v_tail_31_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0_spec__1(lean_object* v_x_43_, lean_object* v_x_44_, lean_object* v_x_45_){
_start:
{
if (lean_obj_tag(v_x_45_) == 0)
{
lean_dec(v_x_43_);
return v_x_44_;
}
else
{
lean_object* v_head_46_; lean_object* v_tail_47_; lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_58_; 
v_head_46_ = lean_ctor_get(v_x_45_, 0);
v_tail_47_ = lean_ctor_get(v_x_45_, 1);
v_isSharedCheck_58_ = !lean_is_exclusive(v_x_45_);
if (v_isSharedCheck_58_ == 0)
{
v___x_49_ = v_x_45_;
v_isShared_50_ = v_isSharedCheck_58_;
goto v_resetjp_48_;
}
else
{
lean_inc(v_tail_47_);
lean_inc(v_head_46_);
lean_dec(v_x_45_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_58_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v___x_52_; 
lean_inc(v_x_43_);
if (v_isShared_50_ == 0)
{
lean_ctor_set_tag(v___x_49_, 5);
lean_ctor_set(v___x_49_, 1, v_x_43_);
lean_ctor_set(v___x_49_, 0, v_x_44_);
v___x_52_ = v___x_49_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v_x_44_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v_x_43_);
v___x_52_ = v_reuseFailAlloc_57_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = l_Nat_reprFast(v_head_46_);
v___x_54_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
v___x_55_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_55_, 0, v___x_52_);
lean_ctor_set(v___x_55_, 1, v___x_54_);
v___x_56_ = lp_swirl_x2dfv_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0_spec__1_spec__2(v_x_43_, v___x_55_, v_tail_47_);
return v___x_56_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0(lean_object* v_x_59_, lean_object* v_x_60_){
_start:
{
if (lean_obj_tag(v_x_59_) == 0)
{
lean_object* v___x_61_; 
lean_dec(v_x_60_);
v___x_61_ = lean_box(0);
return v___x_61_;
}
else
{
lean_object* v_tail_62_; 
v_tail_62_ = lean_ctor_get(v_x_59_, 1);
if (lean_obj_tag(v_tail_62_) == 0)
{
lean_object* v_head_63_; lean_object* v___x_64_; 
lean_dec(v_x_60_);
v_head_63_ = lean_ctor_get(v_x_59_, 0);
lean_inc(v_head_63_);
lean_dec_ref_known(v_x_59_, 2);
v___x_64_ = lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0___lam__0(v_head_63_);
return v___x_64_;
}
else
{
lean_object* v_head_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
lean_inc(v_tail_62_);
v_head_65_ = lean_ctor_get(v_x_59_, 0);
lean_inc(v_head_65_);
lean_dec_ref_known(v_x_59_, 2);
v___x_66_ = lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0___lam__0(v_head_65_);
v___x_67_ = lp_swirl_x2dfv_List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0_spec__1(v_x_60_, v___x_66_, v_tail_62_);
return v___x_67_;
}
}
}
}
static lean_object* _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__7(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__2));
v___x_80_ = lean_string_length(v___x_79_);
return v___x_80_;
}
}
static lean_object* _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__8(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_81_ = lean_obj_once(&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__7, &lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__7_once, _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__7);
v___x_82_ = lean_nat_to_int(v___x_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg(lean_object* v_a_87_){
_start:
{
if (lean_obj_tag(v_a_87_) == 0)
{
lean_object* v___x_88_; 
v___x_88_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__1));
return v___x_88_;
}
else
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_89_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__5));
v___x_90_ = lp_swirl_x2dfv_Std_Format_joinSep___at___00List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0_spec__0(v_a_87_, v___x_89_);
v___x_91_ = lean_obj_once(&lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__8, &lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__8_once, _init_lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__8);
v___x_92_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__9));
v___x_93_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v___x_90_);
v___x_94_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__10));
v___x_95_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_93_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_91_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v___x_97_ = l_Std_Format_fill(v___x_96_);
return v___x_97_;
}
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_111_ = lean_unsigned_to_nat(16u);
v___x_112_ = lean_nat_to_int(v___x_111_);
return v___x_112_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = lean_unsigned_to_nat(15u);
v___x_117_ = lean_nat_to_int(v___x_116_);
return v___x_117_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = lean_unsigned_to_nat(14u);
v___x_122_ = lean_nat_to_int(v___x_121_);
return v___x_122_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__0));
v___x_125_ = lean_string_length(v___x_124_);
return v___x_125_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15);
v___x_127_ = lean_nat_to_int(v___x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg(lean_object* v_x_132_){
_start:
{
lean_object* v_preprocessed_133_; lean_object* v_cachedMains_134_; lean_object* v_commonMain_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; uint8_t v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v_preprocessed_133_ = lean_ctor_get(v_x_132_, 0);
lean_inc(v_preprocessed_133_);
v_cachedMains_134_ = lean_ctor_get(v_x_132_, 1);
lean_inc(v_cachedMains_134_);
v_commonMain_135_ = lean_ctor_get(v_x_132_, 2);
lean_inc(v_commonMain_135_);
lean_dec_ref(v_x_132_);
v___x_136_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_137_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__6));
v___x_138_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7);
v___x_139_ = lean_unsigned_to_nat(0u);
v___x_140_ = l_Option_repr___at___00Lean_Meta_instReprConfig__1_repr_spec__0(v_preprocessed_133_, v___x_139_);
v___x_141_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_138_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = 0;
v___x_143_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_143_, 0, v___x_141_);
lean_ctor_set_uint8(v___x_143_, sizeof(void*)*1, v___x_142_);
v___x_144_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_137_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v___x_145_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_146_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_144_);
lean_ctor_set(v___x_146_, 1, v___x_145_);
v___x_147_ = lean_box(1);
v___x_148_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_146_);
lean_ctor_set(v___x_148_, 1, v___x_147_);
v___x_149_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__9));
v___x_150_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_150_, 0, v___x_148_);
lean_ctor_set(v___x_150_, 1, v___x_149_);
v___x_151_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_150_);
lean_ctor_set(v___x_151_, 1, v___x_136_);
v___x_152_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10);
v___x_153_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg(v_cachedMains_134_);
v___x_154_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_152_);
lean_ctor_set(v___x_154_, 1, v___x_153_);
v___x_155_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_155_, 0, v___x_154_);
lean_ctor_set_uint8(v___x_155_, sizeof(void*)*1, v___x_142_);
v___x_156_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_156_, 0, v___x_151_);
lean_ctor_set(v___x_156_, 1, v___x_155_);
v___x_157_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v___x_145_);
v___x_158_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v___x_147_);
v___x_159_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__12));
v___x_160_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_158_);
lean_ctor_set(v___x_160_, 1, v___x_159_);
v___x_161_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v___x_136_);
v___x_162_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13);
v___x_163_ = l_Nat_reprFast(v_commonMain_135_);
v___x_164_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
v___x_165_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_162_);
lean_ctor_set(v___x_165_, 1, v___x_164_);
v___x_166_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set_uint8(v___x_166_, sizeof(void*)*1, v___x_142_);
v___x_167_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_161_);
lean_ctor_set(v___x_167_, 1, v___x_166_);
v___x_168_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16);
v___x_169_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_170_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v___x_167_);
v___x_171_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_172_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_170_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_168_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
v___x_174_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set_uint8(v___x_174_, sizeof(void*)*1, v___x_142_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr(lean_object* v_x_175_, lean_object* v_prec_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg(v_x_175_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___boxed(lean_object* v_x_178_, lean_object* v_prec_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr(v_x_178_, v_prec_179_);
lean_dec(v_prec_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0(lean_object* v_a_181_, lean_object* v_n_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg(v_a_181_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___boxed(lean_object* v_a_184_, lean_object* v_n_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0(v_a_184_, v_n_185_);
lean_dec(v_n_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0(lean_object* v_init_189_, lean_object* v_x_190_){
_start:
{
if (lean_obj_tag(v_x_190_) == 0)
{
lean_inc(v_init_189_);
return v_init_189_;
}
else
{
lean_object* v_head_191_; lean_object* v_tail_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v_head_191_ = lean_ctor_get(v_x_190_, 0);
v_tail_192_ = lean_ctor_get(v_x_190_, 1);
v___x_193_ = lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0(v_init_189_, v_tail_192_);
v___x_194_ = lean_nat_add(v_head_191_, v___x_193_);
lean_dec(v___x_193_);
return v___x_194_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0___boxed(lean_object* v_init_195_, lean_object* v_x_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0(v_init_195_, v_x_196_);
lean_dec(v_x_196_);
lean_dec(v_init_195_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0(lean_object* v_l_198_){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = lean_unsigned_to_nat(0u);
v___x_200_ = lp_swirl_x2dfv_List_foldr___at___00List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0_spec__0(v___x_199_, v_l_198_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0___boxed(lean_object* v_l_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_swirl_x2dfv_List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0(v_l_201_);
lean_dec(v_l_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_TraceWidth_totalWidth(lean_object* v_width_203_){
_start:
{
lean_object* v_preprocessed_204_; lean_object* v_cachedMains_205_; lean_object* v_commonMain_206_; lean_object* v___y_208_; 
v_preprocessed_204_ = lean_ctor_get(v_width_203_, 0);
v_cachedMains_205_ = lean_ctor_get(v_width_203_, 1);
v_commonMain_206_ = lean_ctor_get(v_width_203_, 2);
if (lean_obj_tag(v_preprocessed_204_) == 0)
{
lean_object* v___x_212_; 
v___x_212_ = lean_unsigned_to_nat(0u);
v___y_208_ = v___x_212_;
goto v___jp_207_;
}
else
{
lean_object* v_val_213_; 
v_val_213_ = lean_ctor_get(v_preprocessed_204_, 0);
v___y_208_ = v_val_213_;
goto v___jp_207_;
}
v___jp_207_:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = lean_nat_add(v___y_208_, v_commonMain_206_);
v___x_210_ = lp_swirl_x2dfv_List_sum___at___00Fundamentals_VerifyingKey_TraceWidth_totalWidth_spec__0(v_cachedMains_205_);
v___x_211_ = lean_nat_add(v___x_209_, v___x_210_);
lean_dec(v___x_210_);
lean_dec(v___x_209_);
return v___x_211_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_TraceWidth_totalWidth___boxed(lean_object* v_width_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_TraceWidth_totalWidth(v_width_214_);
lean_dec_ref(v_width_214_);
return v_res_215_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint_decEq(lean_object* v_x_216_, lean_object* v_x_217_){
_start:
{
lean_object* v_coefficients_218_; lean_object* v_threshold_219_; lean_object* v_coefficients_220_; lean_object* v_threshold_221_; lean_object* v___x_222_; uint8_t v___x_223_; 
v_coefficients_218_ = lean_ctor_get(v_x_216_, 0);
lean_inc(v_coefficients_218_);
v_threshold_219_ = lean_ctor_get(v_x_216_, 1);
lean_inc(v_threshold_219_);
lean_dec_ref(v_x_216_);
v_coefficients_220_ = lean_ctor_get(v_x_217_, 0);
lean_inc(v_coefficients_220_);
v_threshold_221_ = lean_ctor_get(v_x_217_, 1);
lean_inc(v_threshold_221_);
lean_dec_ref(v_x_217_);
v___x_222_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_223_ = l_instDecidableEqList___redArg(v___x_222_, v_coefficients_218_, v_coefficients_220_);
if (v___x_223_ == 0)
{
lean_dec(v_threshold_221_);
lean_dec(v_threshold_219_);
return v___x_223_;
}
else
{
uint8_t v___x_224_; 
v___x_224_ = lean_nat_dec_eq(v_threshold_219_, v_threshold_221_);
lean_dec(v_threshold_221_);
lean_dec(v_threshold_219_);
return v___x_224_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint_decEq___boxed(lean_object* v_x_225_, lean_object* v_x_226_){
_start:
{
uint8_t v_res_227_; lean_object* v_r_228_; 
v_res_227_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint_decEq(v_x_225_, v_x_226_);
v_r_228_ = lean_box(v_res_227_);
return v_r_228_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint(lean_object* v_x_229_, lean_object* v_x_230_){
_start:
{
uint8_t v___x_231_; 
v___x_231_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint_decEq(v_x_229_, v_x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint___boxed(lean_object* v_x_232_, lean_object* v_x_233_){
_start:
{
uint8_t v_res_234_; lean_object* v_r_235_; 
v_res_234_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint(v_x_232_, v_x_233_);
v_r_235_ = lean_box(v_res_234_);
return v_r_235_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__6(void){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_248_ = lean_unsigned_to_nat(13u);
v___x_249_ = lean_nat_to_int(v___x_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg(lean_object* v_x_250_){
_start:
{
lean_object* v_coefficients_251_; lean_object* v_threshold_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_286_; 
v_coefficients_251_ = lean_ctor_get(v_x_250_, 0);
v_threshold_252_ = lean_ctor_get(v_x_250_, 1);
v_isSharedCheck_286_ = !lean_is_exclusive(v_x_250_);
if (v_isSharedCheck_286_ == 0)
{
v___x_254_ = v_x_250_;
v_isShared_255_ = v_isSharedCheck_286_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_threshold_252_);
lean_inc(v_coefficients_251_);
lean_dec(v_x_250_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_286_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_261_; 
v___x_256_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_257_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__3));
v___x_258_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7);
v___x_259_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg(v_coefficients_251_);
if (v_isShared_255_ == 0)
{
lean_ctor_set_tag(v___x_254_, 4);
lean_ctor_set(v___x_254_, 1, v___x_259_);
lean_ctor_set(v___x_254_, 0, v___x_258_);
v___x_261_ = v___x_254_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v___x_258_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v___x_259_);
v___x_261_ = v_reuseFailAlloc_285_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
uint8_t v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_262_ = 0;
v___x_263_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_263_, 0, v___x_261_);
lean_ctor_set_uint8(v___x_263_, sizeof(void*)*1, v___x_262_);
v___x_264_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_257_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_266_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_264_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = lean_box(1);
v___x_268_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_266_);
lean_ctor_set(v___x_268_, 1, v___x_267_);
v___x_269_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__5));
v___x_270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_270_, 0, v___x_268_);
lean_ctor_set(v___x_270_, 1, v___x_269_);
v___x_271_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
lean_ctor_set(v___x_271_, 1, v___x_256_);
v___x_272_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__6, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__6_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg___closed__6);
v___x_273_ = l_Nat_reprFast(v_threshold_252_);
v___x_274_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_274_, 0, v___x_273_);
v___x_275_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_272_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_276_, 0, v___x_275_);
lean_ctor_set_uint8(v___x_276_, sizeof(void*)*1, v___x_262_);
v___x_277_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_271_);
lean_ctor_set(v___x_277_, 1, v___x_276_);
v___x_278_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16);
v___x_279_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_280_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_280_, 0, v___x_279_);
lean_ctor_set(v___x_280_, 1, v___x_277_);
v___x_281_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_282_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_282_, 0, v___x_280_);
lean_ctor_set(v___x_282_, 1, v___x_281_);
v___x_283_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_278_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
v___x_284_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_284_, 0, v___x_283_);
lean_ctor_set_uint8(v___x_284_, sizeof(void*)*1, v___x_262_);
return v___x_284_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr(lean_object* v_x_287_, lean_object* v_prec_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___redArg(v_x_287_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr___boxed(lean_object* v_x_290_, lean_object* v_prec_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint_repr(v_x_290_, v_prec_291_);
lean_dec(v_prec_291_);
return v_res_292_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq(lean_object* v_x_295_, lean_object* v_x_296_){
_start:
{
lean_object* v_width_297_; lean_object* v_numPublicValues_298_; uint8_t v_needRot_299_; lean_object* v_width_300_; lean_object* v_numPublicValues_301_; uint8_t v_needRot_302_; uint8_t v___x_303_; 
v_width_297_ = lean_ctor_get(v_x_295_, 0);
lean_inc_ref(v_width_297_);
v_numPublicValues_298_ = lean_ctor_get(v_x_295_, 1);
lean_inc(v_numPublicValues_298_);
v_needRot_299_ = lean_ctor_get_uint8(v_x_295_, sizeof(void*)*2);
lean_dec_ref(v_x_295_);
v_width_300_ = lean_ctor_get(v_x_296_, 0);
lean_inc_ref(v_width_300_);
v_numPublicValues_301_ = lean_ctor_get(v_x_296_, 1);
lean_inc(v_numPublicValues_301_);
v_needRot_302_ = lean_ctor_get_uint8(v_x_296_, sizeof(void*)*2);
lean_dec_ref(v_x_296_);
v___x_303_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqTraceWidth_decEq(v_width_297_, v_width_300_);
if (v___x_303_ == 0)
{
lean_dec(v_numPublicValues_301_);
lean_dec(v_numPublicValues_298_);
return v___x_303_;
}
else
{
uint8_t v___x_304_; 
v___x_304_ = lean_nat_dec_eq(v_numPublicValues_298_, v_numPublicValues_301_);
lean_dec(v_numPublicValues_301_);
lean_dec(v_numPublicValues_298_);
if (v___x_304_ == 0)
{
return v___x_304_;
}
else
{
if (v_needRot_299_ == 0)
{
if (v_needRot_302_ == 0)
{
return v___x_304_;
}
else
{
return v_needRot_299_;
}
}
else
{
return v_needRot_302_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq___boxed(lean_object* v_x_305_, lean_object* v_x_306_){
_start:
{
uint8_t v_res_307_; lean_object* v_r_308_; 
v_res_307_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq(v_x_305_, v_x_306_);
v_r_308_ = lean_box(v_res_307_);
return v_r_308_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams(lean_object* v_x_309_, lean_object* v_x_310_){
_start:
{
uint8_t v___x_311_; 
v___x_311_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq(v_x_309_, v_x_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams___boxed(lean_object* v_x_312_, lean_object* v_x_313_){
_start:
{
uint8_t v_res_314_; lean_object* v_r_315_; 
v_res_314_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams(v_x_312_, v_x_313_);
v_r_315_ = lean_box(v_res_314_);
return v_r_315_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = lean_unsigned_to_nat(9u);
v___x_326_ = lean_nat_to_int(v___x_325_);
return v___x_326_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = lean_unsigned_to_nat(19u);
v___x_331_ = lean_nat_to_int(v___x_330_);
return v___x_331_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = lean_unsigned_to_nat(11u);
v___x_336_ = lean_nat_to_int(v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg(lean_object* v_x_337_){
_start:
{
lean_object* v_width_338_; lean_object* v_numPublicValues_339_; uint8_t v_needRot_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; uint8_t v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v_width_338_ = lean_ctor_get(v_x_337_, 0);
lean_inc_ref(v_width_338_);
v_numPublicValues_339_ = lean_ctor_get(v_x_337_, 1);
lean_inc(v_numPublicValues_339_);
v_needRot_340_ = lean_ctor_get_uint8(v_x_337_, sizeof(void*)*2);
lean_dec_ref(v_x_337_);
v___x_341_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_342_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__3));
v___x_343_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4);
v___x_344_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg(v_width_338_);
v___x_345_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_343_);
lean_ctor_set(v___x_345_, 1, v___x_344_);
v___x_346_ = 0;
v___x_347_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_347_, 0, v___x_345_);
lean_ctor_set_uint8(v___x_347_, sizeof(void*)*1, v___x_346_);
v___x_348_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_342_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
v___x_349_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_348_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = lean_box(1);
v___x_352_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_350_);
lean_ctor_set(v___x_352_, 1, v___x_351_);
v___x_353_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__6));
v___x_354_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_352_);
lean_ctor_set(v___x_354_, 1, v___x_353_);
v___x_355_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
lean_ctor_set(v___x_355_, 1, v___x_341_);
v___x_356_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7);
v___x_357_ = l_Nat_reprFast(v_numPublicValues_339_);
v___x_358_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
v___x_359_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_359_, 0, v___x_356_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
v___x_360_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set_uint8(v___x_360_, sizeof(void*)*1, v___x_346_);
v___x_361_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_355_);
lean_ctor_set(v___x_361_, 1, v___x_360_);
v___x_362_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_361_);
lean_ctor_set(v___x_362_, 1, v___x_349_);
v___x_363_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v___x_351_);
v___x_364_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__9));
v___x_365_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_363_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
v___x_366_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_366_, 0, v___x_365_);
lean_ctor_set(v___x_366_, 1, v___x_341_);
v___x_367_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10);
v___x_368_ = l_Bool_repr___redArg(v_needRot_340_);
v___x_369_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_367_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_370_, 0, v___x_369_);
lean_ctor_set_uint8(v___x_370_, sizeof(void*)*1, v___x_346_);
v___x_371_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_366_);
lean_ctor_set(v___x_371_, 1, v___x_370_);
v___x_372_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16);
v___x_373_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_374_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v___x_371_);
v___x_375_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_376_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_376_, 0, v___x_374_);
lean_ctor_set(v___x_376_, 1, v___x_375_);
v___x_377_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_377_, 0, v___x_372_);
lean_ctor_set(v___x_377_, 1, v___x_376_);
v___x_378_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set_uint8(v___x_378_, sizeof(void*)*1, v___x_346_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr(lean_object* v_x_379_, lean_object* v_prec_380_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg(v_x_379_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___boxed(lean_object* v_x_382_, lean_object* v_prec_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr(v_x_382_, v_prec_383_);
lean_dec(v_prec_383_);
return v_res_384_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(lean_object* v_inst_387_, lean_object* v_x_388_, lean_object* v_x_389_){
_start:
{
lean_object* v_commit_390_; lean_object* v_hypercubeDim_391_; lean_object* v_stackingWidth_392_; lean_object* v_commit_393_; lean_object* v_hypercubeDim_394_; lean_object* v_stackingWidth_395_; lean_object* v___x_396_; uint8_t v___x_397_; 
v_commit_390_ = lean_ctor_get(v_x_388_, 0);
lean_inc(v_commit_390_);
v_hypercubeDim_391_ = lean_ctor_get(v_x_388_, 1);
lean_inc(v_hypercubeDim_391_);
v_stackingWidth_392_ = lean_ctor_get(v_x_388_, 2);
lean_inc(v_stackingWidth_392_);
lean_dec_ref(v_x_388_);
v_commit_393_ = lean_ctor_get(v_x_389_, 0);
lean_inc(v_commit_393_);
v_hypercubeDim_394_ = lean_ctor_get(v_x_389_, 1);
lean_inc(v_hypercubeDim_394_);
v_stackingWidth_395_ = lean_ctor_get(v_x_389_, 2);
lean_inc(v_stackingWidth_395_);
lean_dec_ref(v_x_389_);
v___x_396_ = lean_apply_2(v_inst_387_, v_commit_390_, v_commit_393_);
v___x_397_ = lean_unbox(v___x_396_);
if (v___x_397_ == 0)
{
uint8_t v___x_398_; 
lean_dec(v_stackingWidth_395_);
lean_dec(v_hypercubeDim_394_);
lean_dec(v_stackingWidth_392_);
lean_dec(v_hypercubeDim_391_);
v___x_398_ = lean_unbox(v___x_396_);
return v___x_398_;
}
else
{
uint8_t v___x_399_; 
v___x_399_ = lean_int_dec_eq(v_hypercubeDim_391_, v_hypercubeDim_394_);
lean_dec(v_hypercubeDim_394_);
lean_dec(v_hypercubeDim_391_);
if (v___x_399_ == 0)
{
lean_dec(v_stackingWidth_395_);
lean_dec(v_stackingWidth_392_);
return v___x_399_;
}
else
{
uint8_t v___x_400_; 
v___x_400_ = lean_nat_dec_eq(v_stackingWidth_392_, v_stackingWidth_395_);
lean_dec(v_stackingWidth_395_);
lean_dec(v_stackingWidth_392_);
return v___x_400_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg___boxed(lean_object* v_inst_401_, lean_object* v_x_402_, lean_object* v_x_403_){
_start:
{
uint8_t v_res_404_; lean_object* v_r_405_; 
v_res_404_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(v_inst_401_, v_x_402_, v_x_403_);
v_r_405_ = lean_box(v_res_404_);
return v_r_405_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq(lean_object* v_Digest_406_, lean_object* v_inst_407_, lean_object* v_x_408_, lean_object* v_x_409_){
_start:
{
uint8_t v___x_410_; 
v___x_410_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(v_inst_407_, v_x_408_, v_x_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___boxed(lean_object* v_Digest_411_, lean_object* v_inst_412_, lean_object* v_x_413_, lean_object* v_x_414_){
_start:
{
uint8_t v_res_415_; lean_object* v_r_416_; 
v_res_415_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq(v_Digest_411_, v_inst_412_, v_x_413_, v_x_414_);
v_r_416_ = lean_box(v_res_415_);
return v_r_416_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___redArg(lean_object* v_inst_417_, lean_object* v_x_418_, lean_object* v_x_419_){
_start:
{
uint8_t v___x_420_; 
v___x_420_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(v_inst_417_, v_x_418_, v_x_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___redArg___boxed(lean_object* v_inst_421_, lean_object* v_x_422_, lean_object* v_x_423_){
_start:
{
uint8_t v_res_424_; lean_object* v_r_425_; 
v_res_424_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___redArg(v_inst_421_, v_x_422_, v_x_423_);
v_r_425_ = lean_box(v_res_424_);
return v_r_425_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData(lean_object* v_Digest_426_, lean_object* v_inst_427_, lean_object* v_x_428_, lean_object* v_x_429_){
_start:
{
uint8_t v___x_430_; 
v___x_430_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(v_inst_427_, v_x_428_, v_x_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData___boxed(lean_object* v_Digest_431_, lean_object* v_inst_432_, lean_object* v_x_433_, lean_object* v_x_434_){
_start:
{
uint8_t v_res_435_; lean_object* v_r_436_; 
v_res_435_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData(v_Digest_431_, v_inst_432_, v_x_433_, v_x_434_);
v_r_436_ = lean_box(v_res_435_);
return v_r_436_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_446_ = lean_unsigned_to_nat(10u);
v___x_447_ = lean_nat_to_int(v___x_446_);
return v___x_447_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_454_ = lean_unsigned_to_nat(17u);
v___x_455_ = lean_nat_to_int(v___x_454_);
return v___x_455_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_456_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__15);
v___x_457_ = lean_nat_to_int(v___x_456_);
return v___x_457_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__11(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_458_ = lean_unsigned_to_nat(0u);
v___x_459_ = lean_nat_to_int(v___x_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg(lean_object* v_inst_460_, lean_object* v_x_461_){
_start:
{
lean_object* v_commit_462_; lean_object* v_hypercubeDim_463_; lean_object* v_stackingWidth_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; uint8_t v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___y_483_; lean_object* v___x_505_; uint8_t v___x_506_; 
v_commit_462_ = lean_ctor_get(v_x_461_, 0);
lean_inc(v_commit_462_);
v_hypercubeDim_463_ = lean_ctor_get(v_x_461_, 1);
lean_inc(v_hypercubeDim_463_);
v_stackingWidth_464_ = lean_ctor_get(v_x_461_, 2);
lean_inc(v_stackingWidth_464_);
lean_dec_ref(v_x_461_);
v___x_465_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_466_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__3));
v___x_467_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4);
v___x_468_ = lean_unsigned_to_nat(0u);
v___x_469_ = lean_apply_2(v_inst_460_, v_commit_462_, v___x_468_);
v___x_470_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_467_);
lean_ctor_set(v___x_470_, 1, v___x_469_);
v___x_471_ = 0;
v___x_472_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_472_, 0, v___x_470_);
lean_ctor_set_uint8(v___x_472_, sizeof(void*)*1, v___x_471_);
v___x_473_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_473_, 0, v___x_466_);
lean_ctor_set(v___x_473_, 1, v___x_472_);
v___x_474_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_475_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_475_, 0, v___x_473_);
lean_ctor_set(v___x_475_, 1, v___x_474_);
v___x_476_ = lean_box(1);
v___x_477_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_477_, 0, v___x_475_);
lean_ctor_set(v___x_477_, 1, v___x_476_);
v___x_478_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__6));
v___x_479_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_479_, 0, v___x_477_);
lean_ctor_set(v___x_479_, 1, v___x_478_);
v___x_480_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_480_, 0, v___x_479_);
lean_ctor_set(v___x_480_, 1, v___x_465_);
v___x_481_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7);
v___x_505_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__11, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__11_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__11);
v___x_506_ = lean_int_dec_lt(v_hypercubeDim_463_, v___x_505_);
if (v___x_506_ == 0)
{
lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_507_ = l_Int_repr(v_hypercubeDim_463_);
lean_dec(v_hypercubeDim_463_);
v___x_508_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_508_, 0, v___x_507_);
v___y_483_ = v___x_508_;
goto v___jp_482_;
}
else
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_509_ = l_Int_repr(v_hypercubeDim_463_);
lean_dec(v_hypercubeDim_463_);
v___x_510_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
v___x_511_ = l_Repr_addAppParen(v___x_510_, v___x_468_);
v___y_483_ = v___x_511_;
goto v___jp_482_;
}
v___jp_482_:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_484_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_484_, 0, v___x_481_);
lean_ctor_set(v___x_484_, 1, v___y_483_);
v___x_485_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_485_, 0, v___x_484_);
lean_ctor_set_uint8(v___x_485_, sizeof(void*)*1, v___x_471_);
v___x_486_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_480_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set(v___x_487_, 1, v___x_474_);
v___x_488_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_487_);
lean_ctor_set(v___x_488_, 1, v___x_476_);
v___x_489_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__8));
v___x_490_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_488_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_491_, 0, v___x_490_);
lean_ctor_set(v___x_491_, 1, v___x_465_);
v___x_492_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9);
v___x_493_ = l_Nat_reprFast(v_stackingWidth_464_);
v___x_494_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_494_, 0, v___x_493_);
v___x_495_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_495_, 0, v___x_492_);
lean_ctor_set(v___x_495_, 1, v___x_494_);
v___x_496_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_496_, 0, v___x_495_);
lean_ctor_set_uint8(v___x_496_, sizeof(void*)*1, v___x_471_);
v___x_497_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_497_, 0, v___x_491_);
lean_ctor_set(v___x_497_, 1, v___x_496_);
v___x_498_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10);
v___x_499_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_500_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
lean_ctor_set(v___x_500_, 1, v___x_497_);
v___x_501_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_502_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
lean_ctor_set(v___x_502_, 1, v___x_501_);
v___x_503_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_498_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
v___x_504_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_504_, 0, v___x_503_);
lean_ctor_set_uint8(v___x_504_, sizeof(void*)*1, v___x_471_);
return v___x_504_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr(lean_object* v_Digest_512_, lean_object* v_inst_513_, lean_object* v_x_514_, lean_object* v_prec_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg(v_inst_513_, v_x_514_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___boxed(lean_object* v_Digest_517_, lean_object* v_inst_518_, lean_object* v_x_519_, lean_object* v_prec_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr(v_Digest_517_, v_inst_518_, v_x_519_, v_prec_520_);
lean_dec(v_prec_520_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData___redArg(lean_object* v_inst_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___boxed), 4, 2);
lean_closure_set(v___x_523_, 0, lean_box(0));
lean_closure_set(v___x_523_, 1, v_inst_522_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData(lean_object* v_Digest_524_, lean_object* v_inst_525_){
_start:
{
lean_object* v___x_526_; 
v___x_526_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___boxed), 4, 2);
lean_closure_set(v___x_526_, 0, lean_box(0));
lean_closure_set(v___x_526_, 1, v_inst_525_);
return v___x_526_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorIdx(lean_object* v_x_527_){
_start:
{
switch(lean_obj_tag(v_x_527_))
{
case 0:
{
lean_object* v___x_528_; 
v___x_528_ = lean_unsigned_to_nat(0u);
return v___x_528_;
}
case 1:
{
lean_object* v___x_529_; 
v___x_529_ = lean_unsigned_to_nat(1u);
return v___x_529_;
}
case 2:
{
lean_object* v___x_530_; 
v___x_530_ = lean_unsigned_to_nat(2u);
return v___x_530_;
}
default: 
{
lean_object* v___x_531_; 
v___x_531_ = lean_unsigned_to_nat(3u);
return v___x_531_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorIdx___boxed(lean_object* v_x_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorIdx(v_x_532_);
lean_dec(v_x_532_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(lean_object* v_t_534_, lean_object* v_k_535_){
_start:
{
switch(lean_obj_tag(v_t_534_))
{
case 0:
{
lean_object* v_offset_536_; lean_object* v___x_537_; 
v_offset_536_ = lean_ctor_get(v_t_534_, 0);
lean_inc(v_offset_536_);
lean_dec_ref_known(v_t_534_, 1);
v___x_537_ = lean_apply_1(v_k_535_, v_offset_536_);
return v___x_537_;
}
case 1:
{
lean_object* v_partIndex_538_; lean_object* v_offset_539_; lean_object* v___x_540_; 
v_partIndex_538_ = lean_ctor_get(v_t_534_, 0);
lean_inc(v_partIndex_538_);
v_offset_539_ = lean_ctor_get(v_t_534_, 1);
lean_inc(v_offset_539_);
lean_dec_ref_known(v_t_534_, 2);
v___x_540_ = lean_apply_2(v_k_535_, v_partIndex_538_, v_offset_539_);
return v___x_540_;
}
default: 
{
lean_dec(v_t_534_);
return v_k_535_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim(lean_object* v_motive_541_, lean_object* v_ctorIdx_542_, lean_object* v_t_543_, lean_object* v_h_544_, lean_object* v_k_545_){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_543_, v_k_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___boxed(lean_object* v_motive_547_, lean_object* v_ctorIdx_548_, lean_object* v_t_549_, lean_object* v_h_550_, lean_object* v_k_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim(v_motive_547_, v_ctorIdx_548_, v_t_549_, v_h_550_, v_k_551_);
lean_dec(v_ctorIdx_548_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_preprocessed_elim___redArg(lean_object* v_t_553_, lean_object* v_preprocessed_554_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_553_, v_preprocessed_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_preprocessed_elim(lean_object* v_motive_556_, lean_object* v_t_557_, lean_object* v_h_558_, lean_object* v_preprocessed_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_557_, v_preprocessed_559_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_main_elim___redArg(lean_object* v_t_561_, lean_object* v_main_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_561_, v_main_562_);
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_main_elim(lean_object* v_motive_564_, lean_object* v_t_565_, lean_object* v_h_566_, lean_object* v_main_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_565_, v_main_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_publicInput_elim___redArg(lean_object* v_t_569_, lean_object* v_publicInput_570_){
_start:
{
lean_object* v___x_571_; 
v___x_571_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_569_, v_publicInput_570_);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_publicInput_elim(lean_object* v_motive_572_, lean_object* v_t_573_, lean_object* v_h_574_, lean_object* v_publicInput_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_573_, v_publicInput_575_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_challenge_elim___redArg(lean_object* v_t_577_, lean_object* v_challenge_578_){
_start:
{
lean_object* v___x_579_; 
v___x_579_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_577_, v_challenge_578_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_challenge_elim(lean_object* v_motive_580_, lean_object* v_t_581_, lean_object* v_h_582_, lean_object* v_challenge_583_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_Entry_ctorElim___redArg(v_t_581_, v_challenge_583_);
return v___x_584_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq(lean_object* v_x_585_, lean_object* v_x_586_){
_start:
{
switch(lean_obj_tag(v_x_585_))
{
case 0:
{
lean_object* v_offset_587_; uint8_t v___x_588_; 
v_offset_587_ = lean_ctor_get(v_x_585_, 0);
v___x_588_ = 0;
if (lean_obj_tag(v_x_586_) == 0)
{
lean_object* v_offset_589_; uint8_t v___x_590_; 
v_offset_589_ = lean_ctor_get(v_x_586_, 0);
v___x_590_ = lean_nat_dec_eq(v_offset_587_, v_offset_589_);
if (v___x_590_ == 0)
{
return v___x_588_;
}
else
{
return v___x_590_;
}
}
else
{
return v___x_588_;
}
}
case 1:
{
lean_object* v_partIndex_591_; lean_object* v_offset_592_; uint8_t v___x_593_; 
v_partIndex_591_ = lean_ctor_get(v_x_585_, 0);
v_offset_592_ = lean_ctor_get(v_x_585_, 1);
v___x_593_ = 0;
if (lean_obj_tag(v_x_586_) == 1)
{
lean_object* v_partIndex_594_; lean_object* v_offset_595_; uint8_t v___x_596_; 
v_partIndex_594_ = lean_ctor_get(v_x_586_, 0);
v_offset_595_ = lean_ctor_get(v_x_586_, 1);
v___x_596_ = lean_nat_dec_eq(v_partIndex_591_, v_partIndex_594_);
if (v___x_596_ == 0)
{
return v___x_593_;
}
else
{
uint8_t v___x_597_; 
v___x_597_ = lean_nat_dec_eq(v_offset_592_, v_offset_595_);
if (v___x_597_ == 0)
{
return v___x_593_;
}
else
{
return v___x_597_;
}
}
}
else
{
return v___x_593_;
}
}
case 2:
{
switch(lean_obj_tag(v_x_586_))
{
case 2:
{
uint8_t v___x_598_; 
v___x_598_ = 1;
return v___x_598_;
}
case 3:
{
uint8_t v___x_599_; 
v___x_599_ = 0;
return v___x_599_;
}
default: 
{
uint8_t v___x_600_; 
v___x_600_ = 0;
return v___x_600_;
}
}
}
default: 
{
switch(lean_obj_tag(v_x_586_))
{
case 2:
{
uint8_t v___x_601_; 
v___x_601_ = 0;
return v___x_601_;
}
case 3:
{
uint8_t v___x_602_; 
v___x_602_ = 1;
return v___x_602_;
}
default: 
{
uint8_t v___x_603_; 
v___x_603_ = 0;
return v___x_603_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq___boxed(lean_object* v_x_604_, lean_object* v_x_605_){
_start:
{
uint8_t v_res_606_; lean_object* v_r_607_; 
v_res_606_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq(v_x_604_, v_x_605_);
lean_dec(v_x_605_);
lean_dec(v_x_604_);
v_r_607_ = lean_box(v_res_606_);
return v_r_607_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry(lean_object* v_x_608_, lean_object* v_x_609_){
_start:
{
uint8_t v___x_610_; 
v___x_610_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq(v_x_608_, v_x_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry___boxed(lean_object* v_x_611_, lean_object* v_x_612_){
_start:
{
uint8_t v_res_613_; lean_object* v_r_614_; 
v_res_613_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry(v_x_611_, v_x_612_);
lean_dec(v_x_612_);
lean_dec(v_x_611_);
v_r_614_ = lean_box(v_res_613_);
return v_r_614_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7(void){
_start:
{
lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_627_ = lean_unsigned_to_nat(2u);
v___x_628_ = lean_nat_to_int(v___x_627_);
return v___x_628_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_629_ = lean_unsigned_to_nat(1u);
v___x_630_ = lean_nat_to_int(v___x_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr(lean_object* v_x_637_, lean_object* v_prec_638_){
_start:
{
lean_object* v___y_640_; lean_object* v___y_647_; 
switch(lean_obj_tag(v_x_637_))
{
case 0:
{
lean_object* v_offset_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_673_; 
v_offset_653_ = lean_ctor_get(v_x_637_, 0);
v_isSharedCheck_673_ = !lean_is_exclusive(v_x_637_);
if (v_isSharedCheck_673_ == 0)
{
v___x_655_ = v_x_637_;
v_isShared_656_ = v_isSharedCheck_673_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_offset_653_);
lean_dec(v_x_637_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_673_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___y_658_; lean_object* v___x_669_; uint8_t v___x_670_; 
v___x_669_ = lean_unsigned_to_nat(1024u);
v___x_670_ = lean_nat_dec_le(v___x_669_, v_prec_638_);
if (v___x_670_ == 0)
{
lean_object* v___x_671_; 
v___x_671_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_658_ = v___x_671_;
goto v___jp_657_;
}
else
{
lean_object* v___x_672_; 
v___x_672_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_658_ = v___x_672_;
goto v___jp_657_;
}
v___jp_657_:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_662_; 
v___x_659_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__6));
v___x_660_ = l_Nat_reprFast(v_offset_653_);
if (v_isShared_656_ == 0)
{
lean_ctor_set_tag(v___x_655_, 3);
lean_ctor_set(v___x_655_, 0, v___x_660_);
v___x_662_ = v___x_655_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_668_; 
v_reuseFailAlloc_668_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_668_, 0, v___x_660_);
v___x_662_ = v_reuseFailAlloc_668_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
lean_object* v___x_663_; lean_object* v___x_664_; uint8_t v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_663_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_663_, 0, v___x_659_);
lean_ctor_set(v___x_663_, 1, v___x_662_);
lean_inc(v___y_658_);
v___x_664_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_664_, 0, v___y_658_);
lean_ctor_set(v___x_664_, 1, v___x_663_);
v___x_665_ = 0;
v___x_666_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_666_, 0, v___x_664_);
lean_ctor_set_uint8(v___x_666_, sizeof(void*)*1, v___x_665_);
v___x_667_ = l_Repr_addAppParen(v___x_666_, v_prec_638_);
return v___x_667_;
}
}
}
}
case 1:
{
lean_object* v_partIndex_674_; lean_object* v_offset_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_700_; 
v_partIndex_674_ = lean_ctor_get(v_x_637_, 0);
v_offset_675_ = lean_ctor_get(v_x_637_, 1);
v_isSharedCheck_700_ = !lean_is_exclusive(v_x_637_);
if (v_isSharedCheck_700_ == 0)
{
v___x_677_ = v_x_637_;
v_isShared_678_ = v_isSharedCheck_700_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_offset_675_);
lean_inc(v_partIndex_674_);
lean_dec(v_x_637_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_700_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v___y_680_; lean_object* v___x_696_; uint8_t v___x_697_; 
v___x_696_ = lean_unsigned_to_nat(1024u);
v___x_697_ = lean_nat_dec_le(v___x_696_, v_prec_638_);
if (v___x_697_ == 0)
{
lean_object* v___x_698_; 
v___x_698_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_680_ = v___x_698_;
goto v___jp_679_;
}
else
{
lean_object* v___x_699_; 
v___x_699_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_680_ = v___x_699_;
goto v___jp_679_;
}
v___jp_679_:
{
lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_686_; 
v___x_681_ = lean_box(1);
v___x_682_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__11));
v___x_683_ = l_Nat_reprFast(v_partIndex_674_);
v___x_684_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_684_, 0, v___x_683_);
if (v_isShared_678_ == 0)
{
lean_ctor_set_tag(v___x_677_, 5);
lean_ctor_set(v___x_677_, 1, v___x_684_);
lean_ctor_set(v___x_677_, 0, v___x_682_);
v___x_686_ = v___x_677_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v___x_682_);
lean_ctor_set(v_reuseFailAlloc_695_, 1, v___x_684_);
v___x_686_ = v_reuseFailAlloc_695_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; uint8_t v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_687_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_687_, 0, v___x_686_);
lean_ctor_set(v___x_687_, 1, v___x_681_);
v___x_688_ = l_Nat_reprFast(v_offset_675_);
v___x_689_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_689_, 0, v___x_688_);
v___x_690_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_690_, 0, v___x_687_);
lean_ctor_set(v___x_690_, 1, v___x_689_);
lean_inc(v___y_680_);
v___x_691_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_691_, 0, v___y_680_);
lean_ctor_set(v___x_691_, 1, v___x_690_);
v___x_692_ = 0;
v___x_693_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_693_, 0, v___x_691_);
lean_ctor_set_uint8(v___x_693_, sizeof(void*)*1, v___x_692_);
v___x_694_ = l_Repr_addAppParen(v___x_693_, v_prec_638_);
return v___x_694_;
}
}
}
}
case 2:
{
lean_object* v___x_701_; uint8_t v___x_702_; 
v___x_701_ = lean_unsigned_to_nat(1024u);
v___x_702_ = lean_nat_dec_le(v___x_701_, v_prec_638_);
if (v___x_702_ == 0)
{
lean_object* v___x_703_; 
v___x_703_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_640_ = v___x_703_;
goto v___jp_639_;
}
else
{
lean_object* v___x_704_; 
v___x_704_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_640_ = v___x_704_;
goto v___jp_639_;
}
}
default: 
{
lean_object* v___x_705_; uint8_t v___x_706_; 
v___x_705_ = lean_unsigned_to_nat(1024u);
v___x_706_ = lean_nat_dec_le(v___x_705_, v_prec_638_);
if (v___x_706_ == 0)
{
lean_object* v___x_707_; 
v___x_707_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_647_ = v___x_707_;
goto v___jp_646_;
}
else
{
lean_object* v___x_708_; 
v___x_708_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_647_ = v___x_708_;
goto v___jp_646_;
}
}
}
v___jp_639_:
{
lean_object* v___x_641_; lean_object* v___x_642_; uint8_t v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; 
v___x_641_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__1));
lean_inc(v___y_640_);
v___x_642_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_642_, 0, v___y_640_);
lean_ctor_set(v___x_642_, 1, v___x_641_);
v___x_643_ = 0;
v___x_644_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_644_, 0, v___x_642_);
lean_ctor_set_uint8(v___x_644_, sizeof(void*)*1, v___x_643_);
v___x_645_ = l_Repr_addAppParen(v___x_644_, v_prec_638_);
return v___x_645_;
}
v___jp_646_:
{
lean_object* v___x_648_; lean_object* v___x_649_; uint8_t v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_648_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__3));
lean_inc(v___y_647_);
v___x_649_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_649_, 0, v___y_647_);
lean_ctor_set(v___x_649_, 1, v___x_648_);
v___x_650_ = 0;
v___x_651_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_651_, 0, v___x_649_);
lean_ctor_set_uint8(v___x_651_, sizeof(void*)*1, v___x_650_);
v___x_652_ = l_Repr_addAppParen(v___x_651_, v_prec_638_);
return v___x_652_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___boxed(lean_object* v_x_709_, lean_object* v_prec_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr(v_x_709_, v_prec_710_);
lean_dec(v_prec_710_);
return v_res_711_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq(lean_object* v_x_714_, lean_object* v_x_715_){
_start:
{
lean_object* v_entry_716_; lean_object* v_index_717_; lean_object* v_entry_718_; lean_object* v_index_719_; uint8_t v___x_720_; 
v_entry_716_ = lean_ctor_get(v_x_714_, 0);
v_index_717_ = lean_ctor_get(v_x_714_, 1);
v_entry_718_ = lean_ctor_get(v_x_715_, 0);
v_index_719_ = lean_ctor_get(v_x_715_, 1);
v___x_720_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqEntry_decEq(v_entry_716_, v_entry_718_);
if (v___x_720_ == 0)
{
return v___x_720_;
}
else
{
uint8_t v___x_721_; 
v___x_721_ = lean_nat_dec_eq(v_index_717_, v_index_719_);
return v___x_721_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq___boxed(lean_object* v_x_722_, lean_object* v_x_723_){
_start:
{
uint8_t v_res_724_; lean_object* v_r_725_; 
v_res_724_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq(v_x_722_, v_x_723_);
lean_dec_ref(v_x_723_);
lean_dec_ref(v_x_722_);
v_r_725_ = lean_box(v_res_724_);
return v_r_725_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable(lean_object* v_x_726_, lean_object* v_x_727_){
_start:
{
uint8_t v___x_728_; 
v___x_728_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq(v_x_726_, v_x_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable___boxed(lean_object* v_x_729_, lean_object* v_x_730_){
_start:
{
uint8_t v_res_731_; lean_object* v_r_732_; 
v_res_731_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable(v_x_729_, v_x_730_);
lean_dec_ref(v_x_730_);
lean_dec_ref(v_x_729_);
v_r_732_ = lean_box(v_res_731_);
return v_r_732_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg(lean_object* v_x_745_){
_start:
{
lean_object* v_entry_746_; lean_object* v_index_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_781_; 
v_entry_746_ = lean_ctor_get(v_x_745_, 0);
v_index_747_ = lean_ctor_get(v_x_745_, 1);
v_isSharedCheck_781_ = !lean_is_exclusive(v_x_745_);
if (v_isSharedCheck_781_ == 0)
{
v___x_749_ = v_x_745_;
v_isShared_750_ = v_isSharedCheck_781_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_index_747_);
lean_inc(v_entry_746_);
lean_dec(v_x_745_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_781_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_757_; 
v___x_751_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_752_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__3));
v___x_753_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4);
v___x_754_ = lean_unsigned_to_nat(0u);
v___x_755_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr(v_entry_746_, v___x_754_);
if (v_isShared_750_ == 0)
{
lean_ctor_set_tag(v___x_749_, 4);
lean_ctor_set(v___x_749_, 1, v___x_755_);
lean_ctor_set(v___x_749_, 0, v___x_753_);
v___x_757_ = v___x_749_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v___x_753_);
lean_ctor_set(v_reuseFailAlloc_780_, 1, v___x_755_);
v___x_757_ = v_reuseFailAlloc_780_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
uint8_t v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; 
v___x_758_ = 0;
v___x_759_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_759_, 0, v___x_757_);
lean_ctor_set_uint8(v___x_759_, sizeof(void*)*1, v___x_758_);
v___x_760_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_760_, 0, v___x_752_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
v___x_761_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_762_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_762_, 0, v___x_760_);
lean_ctor_set(v___x_762_, 1, v___x_761_);
v___x_763_ = lean_box(1);
v___x_764_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_764_, 0, v___x_762_);
lean_ctor_set(v___x_764_, 1, v___x_763_);
v___x_765_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg___closed__5));
v___x_766_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_766_, 0, v___x_764_);
lean_ctor_set(v___x_766_, 1, v___x_765_);
v___x_767_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_767_, 0, v___x_766_);
lean_ctor_set(v___x_767_, 1, v___x_751_);
v___x_768_ = l_Nat_reprFast(v_index_747_);
v___x_769_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_769_, 0, v___x_768_);
v___x_770_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_770_, 0, v___x_753_);
lean_ctor_set(v___x_770_, 1, v___x_769_);
v___x_771_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_771_, 0, v___x_770_);
lean_ctor_set_uint8(v___x_771_, sizeof(void*)*1, v___x_758_);
v___x_772_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_772_, 0, v___x_767_);
lean_ctor_set(v___x_772_, 1, v___x_771_);
v___x_773_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16);
v___x_774_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_775_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_775_, 0, v___x_774_);
lean_ctor_set(v___x_775_, 1, v___x_772_);
v___x_776_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_777_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_777_, 0, v___x_775_);
lean_ctor_set(v___x_777_, 1, v___x_776_);
v___x_778_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_778_, 0, v___x_773_);
lean_ctor_set(v___x_778_, 1, v___x_777_);
v___x_779_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_779_, 0, v___x_778_);
lean_ctor_set_uint8(v___x_779_, sizeof(void*)*1, v___x_758_);
return v___x_779_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr(lean_object* v_x_782_, lean_object* v_prec_783_){
_start:
{
lean_object* v___x_784_; 
v___x_784_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg(v_x_782_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___boxed(lean_object* v_x_785_, lean_object* v_prec_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr(v_x_785_, v_prec_786_);
lean_dec(v_prec_786_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___redArg(lean_object* v_x_790_){
_start:
{
switch(lean_obj_tag(v_x_790_))
{
case 0:
{
lean_object* v___x_791_; 
v___x_791_ = lean_unsigned_to_nat(0u);
return v___x_791_;
}
case 1:
{
lean_object* v___x_792_; 
v___x_792_ = lean_unsigned_to_nat(1u);
return v___x_792_;
}
case 2:
{
lean_object* v___x_793_; 
v___x_793_ = lean_unsigned_to_nat(2u);
return v___x_793_;
}
case 3:
{
lean_object* v___x_794_; 
v___x_794_ = lean_unsigned_to_nat(3u);
return v___x_794_;
}
case 4:
{
lean_object* v___x_795_; 
v___x_795_ = lean_unsigned_to_nat(4u);
return v___x_795_;
}
case 5:
{
lean_object* v___x_796_; 
v___x_796_ = lean_unsigned_to_nat(5u);
return v___x_796_;
}
case 6:
{
lean_object* v___x_797_; 
v___x_797_ = lean_unsigned_to_nat(6u);
return v___x_797_;
}
case 7:
{
lean_object* v___x_798_; 
v___x_798_ = lean_unsigned_to_nat(7u);
return v___x_798_;
}
default: 
{
lean_object* v___x_799_; 
v___x_799_ = lean_unsigned_to_nat(8u);
return v___x_799_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___redArg___boxed(lean_object* v_x_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___redArg(v_x_800_);
lean_dec(v_x_800_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx(lean_object* v_F_802_, lean_object* v_x_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___redArg(v_x_803_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx___boxed(lean_object* v_F_805_, lean_object* v_x_806_){
_start:
{
lean_object* v_res_807_; 
v_res_807_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorIdx(v_F_805_, v_x_806_);
lean_dec(v_x_806_);
return v_res_807_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(lean_object* v_t_808_, lean_object* v_k_809_){
_start:
{
switch(lean_obj_tag(v_t_808_))
{
case 0:
{
lean_object* v_v_810_; lean_object* v___x_811_; 
v_v_810_ = lean_ctor_get(v_t_808_, 0);
lean_inc_ref(v_v_810_);
lean_dec_ref_known(v_t_808_, 1);
v___x_811_ = lean_apply_1(v_k_809_, v_v_810_);
return v___x_811_;
}
case 4:
{
lean_object* v_c_812_; lean_object* v___x_813_; 
v_c_812_ = lean_ctor_get(v_t_808_, 0);
lean_inc(v_c_812_);
lean_dec_ref_known(v_t_808_, 1);
v___x_813_ = lean_apply_1(v_k_809_, v_c_812_);
return v___x_813_;
}
case 5:
{
lean_object* v_x_814_; lean_object* v_y_815_; lean_object* v___x_816_; 
v_x_814_ = lean_ctor_get(v_t_808_, 0);
lean_inc(v_x_814_);
v_y_815_ = lean_ctor_get(v_t_808_, 1);
lean_inc(v_y_815_);
lean_dec_ref_known(v_t_808_, 2);
v___x_816_ = lean_apply_2(v_k_809_, v_x_814_, v_y_815_);
return v___x_816_;
}
case 6:
{
lean_object* v_x_817_; lean_object* v_y_818_; lean_object* v___x_819_; 
v_x_817_ = lean_ctor_get(v_t_808_, 0);
lean_inc(v_x_817_);
v_y_818_ = lean_ctor_get(v_t_808_, 1);
lean_inc(v_y_818_);
lean_dec_ref_known(v_t_808_, 2);
v___x_819_ = lean_apply_2(v_k_809_, v_x_817_, v_y_818_);
return v___x_819_;
}
case 7:
{
lean_object* v_x_820_; lean_object* v___x_821_; 
v_x_820_ = lean_ctor_get(v_t_808_, 0);
lean_inc(v_x_820_);
lean_dec_ref_known(v_t_808_, 1);
v___x_821_ = lean_apply_1(v_k_809_, v_x_820_);
return v___x_821_;
}
case 8:
{
lean_object* v_x_822_; lean_object* v_y_823_; lean_object* v___x_824_; 
v_x_822_ = lean_ctor_get(v_t_808_, 0);
lean_inc(v_x_822_);
v_y_823_ = lean_ctor_get(v_t_808_, 1);
lean_inc(v_y_823_);
lean_dec_ref_known(v_t_808_, 2);
v___x_824_ = lean_apply_2(v_k_809_, v_x_822_, v_y_823_);
return v___x_824_;
}
default: 
{
lean_dec(v_t_808_);
return v_k_809_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim(lean_object* v_F_825_, lean_object* v_motive_826_, lean_object* v_ctorIdx_827_, lean_object* v_t_828_, lean_object* v_h_829_, lean_object* v_k_830_){
_start:
{
lean_object* v___x_831_; 
v___x_831_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_828_, v_k_830_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___boxed(lean_object* v_F_832_, lean_object* v_motive_833_, lean_object* v_ctorIdx_834_, lean_object* v_t_835_, lean_object* v_h_836_, lean_object* v_k_837_){
_start:
{
lean_object* v_res_838_; 
v_res_838_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim(v_F_832_, v_motive_833_, v_ctorIdx_834_, v_t_835_, v_h_836_, v_k_837_);
lean_dec(v_ctorIdx_834_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_variable_elim___redArg(lean_object* v_t_839_, lean_object* v_variable_840_){
_start:
{
lean_object* v___x_841_; 
v___x_841_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_839_, v_variable_840_);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_variable_elim(lean_object* v_F_842_, lean_object* v_motive_843_, lean_object* v_t_844_, lean_object* v_h_845_, lean_object* v_variable_846_){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_844_, v_variable_846_);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isFirstRow_elim___redArg(lean_object* v_t_848_, lean_object* v_isFirstRow_849_){
_start:
{
lean_object* v___x_850_; 
v___x_850_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_848_, v_isFirstRow_849_);
return v___x_850_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isFirstRow_elim(lean_object* v_F_851_, lean_object* v_motive_852_, lean_object* v_t_853_, lean_object* v_h_854_, lean_object* v_isFirstRow_855_){
_start:
{
lean_object* v___x_856_; 
v___x_856_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_853_, v_isFirstRow_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isLastRow_elim___redArg(lean_object* v_t_857_, lean_object* v_isLastRow_858_){
_start:
{
lean_object* v___x_859_; 
v___x_859_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_857_, v_isLastRow_858_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isLastRow_elim(lean_object* v_F_860_, lean_object* v_motive_861_, lean_object* v_t_862_, lean_object* v_h_863_, lean_object* v_isLastRow_864_){
_start:
{
lean_object* v___x_865_; 
v___x_865_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_862_, v_isLastRow_864_);
return v___x_865_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isTransition_elim___redArg(lean_object* v_t_866_, lean_object* v_isTransition_867_){
_start:
{
lean_object* v___x_868_; 
v___x_868_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_866_, v_isTransition_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_isTransition_elim(lean_object* v_F_869_, lean_object* v_motive_870_, lean_object* v_t_871_, lean_object* v_h_872_, lean_object* v_isTransition_873_){
_start:
{
lean_object* v___x_874_; 
v___x_874_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_871_, v_isTransition_873_);
return v___x_874_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_constant_elim___redArg(lean_object* v_t_875_, lean_object* v_constant_876_){
_start:
{
lean_object* v___x_877_; 
v___x_877_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_875_, v_constant_876_);
return v___x_877_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_constant_elim(lean_object* v_F_878_, lean_object* v_motive_879_, lean_object* v_t_880_, lean_object* v_h_881_, lean_object* v_constant_882_){
_start:
{
lean_object* v___x_883_; 
v___x_883_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_880_, v_constant_882_);
return v___x_883_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_add_elim___redArg(lean_object* v_t_884_, lean_object* v_add_885_){
_start:
{
lean_object* v___x_886_; 
v___x_886_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_884_, v_add_885_);
return v___x_886_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_add_elim(lean_object* v_F_887_, lean_object* v_motive_888_, lean_object* v_t_889_, lean_object* v_h_890_, lean_object* v_add_891_){
_start:
{
lean_object* v___x_892_; 
v___x_892_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_889_, v_add_891_);
return v___x_892_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_sub_elim___redArg(lean_object* v_t_893_, lean_object* v_sub_894_){
_start:
{
lean_object* v___x_895_; 
v___x_895_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_893_, v_sub_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_sub_elim(lean_object* v_F_896_, lean_object* v_motive_897_, lean_object* v_t_898_, lean_object* v_h_899_, lean_object* v_sub_900_){
_start:
{
lean_object* v___x_901_; 
v___x_901_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_898_, v_sub_900_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_neg_elim___redArg(lean_object* v_t_902_, lean_object* v_neg_903_){
_start:
{
lean_object* v___x_904_; 
v___x_904_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_902_, v_neg_903_);
return v___x_904_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_neg_elim(lean_object* v_F_905_, lean_object* v_motive_906_, lean_object* v_t_907_, lean_object* v_h_908_, lean_object* v_neg_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_907_, v_neg_909_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_mul_elim___redArg(lean_object* v_t_911_, lean_object* v_mul_912_){
_start:
{
lean_object* v___x_913_; 
v___x_913_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_911_, v_mul_912_);
return v___x_913_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_mul_elim(lean_object* v_F_914_, lean_object* v_motive_915_, lean_object* v_t_916_, lean_object* v_h_917_, lean_object* v_mul_918_){
_start:
{
lean_object* v___x_919_; 
v___x_919_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpression_ctorElim___redArg(v_t_916_, v_mul_918_);
return v___x_919_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___redArg(lean_object* v_x_920_){
_start:
{
switch(lean_obj_tag(v_x_920_))
{
case 0:
{
lean_object* v___x_921_; 
v___x_921_ = lean_unsigned_to_nat(0u);
return v___x_921_;
}
case 1:
{
lean_object* v___x_922_; 
v___x_922_ = lean_unsigned_to_nat(1u);
return v___x_922_;
}
case 2:
{
lean_object* v___x_923_; 
v___x_923_ = lean_unsigned_to_nat(2u);
return v___x_923_;
}
case 3:
{
lean_object* v___x_924_; 
v___x_924_ = lean_unsigned_to_nat(3u);
return v___x_924_;
}
case 4:
{
lean_object* v___x_925_; 
v___x_925_ = lean_unsigned_to_nat(4u);
return v___x_925_;
}
case 5:
{
lean_object* v___x_926_; 
v___x_926_ = lean_unsigned_to_nat(5u);
return v___x_926_;
}
case 6:
{
lean_object* v___x_927_; 
v___x_927_ = lean_unsigned_to_nat(6u);
return v___x_927_;
}
case 7:
{
lean_object* v___x_928_; 
v___x_928_ = lean_unsigned_to_nat(7u);
return v___x_928_;
}
default: 
{
lean_object* v___x_929_; 
v___x_929_ = lean_unsigned_to_nat(8u);
return v___x_929_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___redArg___boxed(lean_object* v_x_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___redArg(v_x_930_);
lean_dec(v_x_930_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx(lean_object* v_F_932_, lean_object* v_x_933_){
_start:
{
lean_object* v___x_934_; 
v___x_934_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___redArg(v_x_933_);
return v___x_934_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx___boxed(lean_object* v_F_935_, lean_object* v_x_936_){
_start:
{
lean_object* v_res_937_; 
v_res_937_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorIdx(v_F_935_, v_x_936_);
lean_dec(v_x_936_);
return v_res_937_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(lean_object* v_t_938_, lean_object* v_k_939_){
_start:
{
switch(lean_obj_tag(v_t_938_))
{
case 0:
{
lean_object* v_v_940_; lean_object* v___x_941_; 
v_v_940_ = lean_ctor_get(v_t_938_, 0);
lean_inc_ref(v_v_940_);
lean_dec_ref_known(v_t_938_, 1);
v___x_941_ = lean_apply_1(v_k_939_, v_v_940_);
return v___x_941_;
}
case 4:
{
lean_object* v_c_942_; lean_object* v___x_943_; 
v_c_942_ = lean_ctor_get(v_t_938_, 0);
lean_inc(v_c_942_);
lean_dec_ref_known(v_t_938_, 1);
v___x_943_ = lean_apply_1(v_k_939_, v_c_942_);
return v___x_943_;
}
case 5:
{
lean_object* v_leftIdx_944_; lean_object* v_rightIdx_945_; lean_object* v_degreeMultiple_946_; lean_object* v___x_947_; 
v_leftIdx_944_ = lean_ctor_get(v_t_938_, 0);
lean_inc(v_leftIdx_944_);
v_rightIdx_945_ = lean_ctor_get(v_t_938_, 1);
lean_inc(v_rightIdx_945_);
v_degreeMultiple_946_ = lean_ctor_get(v_t_938_, 2);
lean_inc(v_degreeMultiple_946_);
lean_dec_ref_known(v_t_938_, 3);
v___x_947_ = lean_apply_3(v_k_939_, v_leftIdx_944_, v_rightIdx_945_, v_degreeMultiple_946_);
return v___x_947_;
}
case 6:
{
lean_object* v_leftIdx_948_; lean_object* v_rightIdx_949_; lean_object* v_degreeMultiple_950_; lean_object* v___x_951_; 
v_leftIdx_948_ = lean_ctor_get(v_t_938_, 0);
lean_inc(v_leftIdx_948_);
v_rightIdx_949_ = lean_ctor_get(v_t_938_, 1);
lean_inc(v_rightIdx_949_);
v_degreeMultiple_950_ = lean_ctor_get(v_t_938_, 2);
lean_inc(v_degreeMultiple_950_);
lean_dec_ref_known(v_t_938_, 3);
v___x_951_ = lean_apply_3(v_k_939_, v_leftIdx_948_, v_rightIdx_949_, v_degreeMultiple_950_);
return v___x_951_;
}
case 7:
{
lean_object* v_idx_952_; lean_object* v_degreeMultiple_953_; lean_object* v___x_954_; 
v_idx_952_ = lean_ctor_get(v_t_938_, 0);
lean_inc(v_idx_952_);
v_degreeMultiple_953_ = lean_ctor_get(v_t_938_, 1);
lean_inc(v_degreeMultiple_953_);
lean_dec_ref_known(v_t_938_, 2);
v___x_954_ = lean_apply_2(v_k_939_, v_idx_952_, v_degreeMultiple_953_);
return v___x_954_;
}
case 8:
{
lean_object* v_leftIdx_955_; lean_object* v_rightIdx_956_; lean_object* v_degreeMultiple_957_; lean_object* v___x_958_; 
v_leftIdx_955_ = lean_ctor_get(v_t_938_, 0);
lean_inc(v_leftIdx_955_);
v_rightIdx_956_ = lean_ctor_get(v_t_938_, 1);
lean_inc(v_rightIdx_956_);
v_degreeMultiple_957_ = lean_ctor_get(v_t_938_, 2);
lean_inc(v_degreeMultiple_957_);
lean_dec_ref_known(v_t_938_, 3);
v___x_958_ = lean_apply_3(v_k_939_, v_leftIdx_955_, v_rightIdx_956_, v_degreeMultiple_957_);
return v___x_958_;
}
default: 
{
lean_dec(v_t_938_);
return v_k_939_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim(lean_object* v_F_959_, lean_object* v_motive_960_, lean_object* v_ctorIdx_961_, lean_object* v_t_962_, lean_object* v_h_963_, lean_object* v_k_964_){
_start:
{
lean_object* v___x_965_; 
v___x_965_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_962_, v_k_964_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___boxed(lean_object* v_F_966_, lean_object* v_motive_967_, lean_object* v_ctorIdx_968_, lean_object* v_t_969_, lean_object* v_h_970_, lean_object* v_k_971_){
_start:
{
lean_object* v_res_972_; 
v_res_972_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim(v_F_966_, v_motive_967_, v_ctorIdx_968_, v_t_969_, v_h_970_, v_k_971_);
lean_dec(v_ctorIdx_968_);
return v_res_972_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_variable_elim___redArg(lean_object* v_t_973_, lean_object* v_variable_974_){
_start:
{
lean_object* v___x_975_; 
v___x_975_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_973_, v_variable_974_);
return v___x_975_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_variable_elim(lean_object* v_F_976_, lean_object* v_motive_977_, lean_object* v_t_978_, lean_object* v_h_979_, lean_object* v_variable_980_){
_start:
{
lean_object* v___x_981_; 
v___x_981_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_978_, v_variable_980_);
return v___x_981_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isFirstRow_elim___redArg(lean_object* v_t_982_, lean_object* v_isFirstRow_983_){
_start:
{
lean_object* v___x_984_; 
v___x_984_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_982_, v_isFirstRow_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isFirstRow_elim(lean_object* v_F_985_, lean_object* v_motive_986_, lean_object* v_t_987_, lean_object* v_h_988_, lean_object* v_isFirstRow_989_){
_start:
{
lean_object* v___x_990_; 
v___x_990_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_987_, v_isFirstRow_989_);
return v___x_990_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isLastRow_elim___redArg(lean_object* v_t_991_, lean_object* v_isLastRow_992_){
_start:
{
lean_object* v___x_993_; 
v___x_993_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_991_, v_isLastRow_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isLastRow_elim(lean_object* v_F_994_, lean_object* v_motive_995_, lean_object* v_t_996_, lean_object* v_h_997_, lean_object* v_isLastRow_998_){
_start:
{
lean_object* v___x_999_; 
v___x_999_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_996_, v_isLastRow_998_);
return v___x_999_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isTransition_elim___redArg(lean_object* v_t_1000_, lean_object* v_isTransition_1001_){
_start:
{
lean_object* v___x_1002_; 
v___x_1002_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1000_, v_isTransition_1001_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_isTransition_elim(lean_object* v_F_1003_, lean_object* v_motive_1004_, lean_object* v_t_1005_, lean_object* v_h_1006_, lean_object* v_isTransition_1007_){
_start:
{
lean_object* v___x_1008_; 
v___x_1008_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1005_, v_isTransition_1007_);
return v___x_1008_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_constant_elim___redArg(lean_object* v_t_1009_, lean_object* v_constant_1010_){
_start:
{
lean_object* v___x_1011_; 
v___x_1011_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1009_, v_constant_1010_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_constant_elim(lean_object* v_F_1012_, lean_object* v_motive_1013_, lean_object* v_t_1014_, lean_object* v_h_1015_, lean_object* v_constant_1016_){
_start:
{
lean_object* v___x_1017_; 
v___x_1017_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1014_, v_constant_1016_);
return v___x_1017_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_add_elim___redArg(lean_object* v_t_1018_, lean_object* v_add_1019_){
_start:
{
lean_object* v___x_1020_; 
v___x_1020_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1018_, v_add_1019_);
return v___x_1020_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_add_elim(lean_object* v_F_1021_, lean_object* v_motive_1022_, lean_object* v_t_1023_, lean_object* v_h_1024_, lean_object* v_add_1025_){
_start:
{
lean_object* v___x_1026_; 
v___x_1026_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1023_, v_add_1025_);
return v___x_1026_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_sub_elim___redArg(lean_object* v_t_1027_, lean_object* v_sub_1028_){
_start:
{
lean_object* v___x_1029_; 
v___x_1029_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1027_, v_sub_1028_);
return v___x_1029_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_sub_elim(lean_object* v_F_1030_, lean_object* v_motive_1031_, lean_object* v_t_1032_, lean_object* v_h_1033_, lean_object* v_sub_1034_){
_start:
{
lean_object* v___x_1035_; 
v___x_1035_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1032_, v_sub_1034_);
return v___x_1035_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_neg_elim___redArg(lean_object* v_t_1036_, lean_object* v_neg_1037_){
_start:
{
lean_object* v___x_1038_; 
v___x_1038_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1036_, v_neg_1037_);
return v___x_1038_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_neg_elim(lean_object* v_F_1039_, lean_object* v_motive_1040_, lean_object* v_t_1041_, lean_object* v_h_1042_, lean_object* v_neg_1043_){
_start:
{
lean_object* v___x_1044_; 
v___x_1044_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1041_, v_neg_1043_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_mul_elim___redArg(lean_object* v_t_1045_, lean_object* v_mul_1046_){
_start:
{
lean_object* v___x_1047_; 
v___x_1047_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1045_, v_mul_1046_);
return v___x_1047_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_mul_elim(lean_object* v_F_1048_, lean_object* v_motive_1049_, lean_object* v_t_1050_, lean_object* v_h_1051_, lean_object* v_mul_1052_){
_start:
{
lean_object* v___x_1053_; 
v___x_1053_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionNode_ctorElim___redArg(v_t_1050_, v_mul_1052_);
return v___x_1053_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(lean_object* v_inst_1054_, lean_object* v_x_1055_, lean_object* v_x_1056_){
_start:
{
switch(lean_obj_tag(v_x_1055_))
{
case 0:
{
lean_object* v_v_1057_; uint8_t v___x_1058_; 
lean_dec_ref(v_inst_1054_);
v_v_1057_ = lean_ctor_get(v_x_1055_, 0);
lean_inc_ref(v_v_1057_);
lean_dec_ref_known(v_x_1055_, 1);
v___x_1058_ = 0;
switch(lean_obj_tag(v_x_1056_))
{
case 0:
{
lean_object* v_v_1059_; uint8_t v___x_1060_; 
v_v_1059_ = lean_ctor_get(v_x_1056_, 0);
lean_inc_ref(v_v_1059_);
lean_dec_ref_known(v_x_1056_, 1);
v___x_1060_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable_decEq(v_v_1057_, v_v_1059_);
lean_dec_ref(v_v_1059_);
lean_dec_ref(v_v_1057_);
if (v___x_1060_ == 0)
{
return v___x_1058_;
}
else
{
return v___x_1060_;
}
}
case 1:
{
lean_dec_ref(v_v_1057_);
return v___x_1058_;
}
case 2:
{
lean_dec_ref(v_v_1057_);
return v___x_1058_;
}
case 3:
{
lean_dec_ref(v_v_1057_);
return v___x_1058_;
}
default: 
{
lean_dec_ref(v_v_1057_);
lean_dec(v_x_1056_);
return v___x_1058_;
}
}
}
case 1:
{
lean_dec_ref(v_inst_1054_);
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
uint8_t v___x_1061_; 
v___x_1061_ = 1;
return v___x_1061_;
}
case 2:
{
uint8_t v___x_1062_; 
v___x_1062_ = 0;
return v___x_1062_;
}
case 3:
{
uint8_t v___x_1063_; 
v___x_1063_ = 0;
return v___x_1063_;
}
default: 
{
uint8_t v___x_1064_; 
lean_dec(v_x_1056_);
v___x_1064_ = 0;
return v___x_1064_;
}
}
}
case 2:
{
lean_dec_ref(v_inst_1054_);
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
uint8_t v___x_1065_; 
v___x_1065_ = 0;
return v___x_1065_;
}
case 2:
{
uint8_t v___x_1066_; 
v___x_1066_ = 1;
return v___x_1066_;
}
case 3:
{
uint8_t v___x_1067_; 
v___x_1067_ = 0;
return v___x_1067_;
}
default: 
{
uint8_t v___x_1068_; 
lean_dec(v_x_1056_);
v___x_1068_ = 0;
return v___x_1068_;
}
}
}
case 3:
{
lean_dec_ref(v_inst_1054_);
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
uint8_t v___x_1069_; 
v___x_1069_ = 0;
return v___x_1069_;
}
case 2:
{
uint8_t v___x_1070_; 
v___x_1070_ = 0;
return v___x_1070_;
}
case 3:
{
uint8_t v___x_1071_; 
v___x_1071_ = 1;
return v___x_1071_;
}
default: 
{
uint8_t v___x_1072_; 
lean_dec(v_x_1056_);
v___x_1072_ = 0;
return v___x_1072_;
}
}
}
case 4:
{
lean_object* v_c_1073_; uint8_t v___x_1074_; 
v_c_1073_ = lean_ctor_get(v_x_1055_, 0);
lean_inc(v_c_1073_);
lean_dec_ref_known(v_x_1055_, 1);
v___x_1074_ = 0;
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
lean_dec(v_c_1073_);
lean_dec_ref(v_inst_1054_);
return v___x_1074_;
}
case 2:
{
lean_dec(v_c_1073_);
lean_dec_ref(v_inst_1054_);
return v___x_1074_;
}
case 3:
{
lean_dec(v_c_1073_);
lean_dec_ref(v_inst_1054_);
return v___x_1074_;
}
case 4:
{
lean_object* v_c_1075_; lean_object* v___x_1076_; uint8_t v___x_1077_; 
v_c_1075_ = lean_ctor_get(v_x_1056_, 0);
lean_inc(v_c_1075_);
lean_dec_ref_known(v_x_1056_, 1);
v___x_1076_ = lean_apply_2(v_inst_1054_, v_c_1073_, v_c_1075_);
v___x_1077_ = lean_unbox(v___x_1076_);
if (v___x_1077_ == 0)
{
return v___x_1074_;
}
else
{
uint8_t v___x_1078_; 
v___x_1078_ = lean_unbox(v___x_1076_);
return v___x_1078_;
}
}
default: 
{
lean_dec(v_c_1073_);
lean_dec(v_x_1056_);
lean_dec_ref(v_inst_1054_);
return v___x_1074_;
}
}
}
case 5:
{
lean_object* v_leftIdx_1079_; lean_object* v_rightIdx_1080_; lean_object* v_degreeMultiple_1081_; uint8_t v___x_1082_; 
lean_dec_ref(v_inst_1054_);
v_leftIdx_1079_ = lean_ctor_get(v_x_1055_, 0);
lean_inc(v_leftIdx_1079_);
v_rightIdx_1080_ = lean_ctor_get(v_x_1055_, 1);
lean_inc(v_rightIdx_1080_);
v_degreeMultiple_1081_ = lean_ctor_get(v_x_1055_, 2);
lean_inc(v_degreeMultiple_1081_);
lean_dec_ref_known(v_x_1055_, 3);
v___x_1082_ = 0;
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
lean_dec(v_degreeMultiple_1081_);
lean_dec(v_rightIdx_1080_);
lean_dec(v_leftIdx_1079_);
return v___x_1082_;
}
case 2:
{
lean_dec(v_degreeMultiple_1081_);
lean_dec(v_rightIdx_1080_);
lean_dec(v_leftIdx_1079_);
return v___x_1082_;
}
case 3:
{
lean_dec(v_degreeMultiple_1081_);
lean_dec(v_rightIdx_1080_);
lean_dec(v_leftIdx_1079_);
return v___x_1082_;
}
case 5:
{
lean_object* v_leftIdx_1083_; lean_object* v_rightIdx_1084_; lean_object* v_degreeMultiple_1085_; uint8_t v___x_1086_; 
v_leftIdx_1083_ = lean_ctor_get(v_x_1056_, 0);
lean_inc(v_leftIdx_1083_);
v_rightIdx_1084_ = lean_ctor_get(v_x_1056_, 1);
lean_inc(v_rightIdx_1084_);
v_degreeMultiple_1085_ = lean_ctor_get(v_x_1056_, 2);
lean_inc(v_degreeMultiple_1085_);
lean_dec_ref_known(v_x_1056_, 3);
v___x_1086_ = lean_nat_dec_eq(v_leftIdx_1079_, v_leftIdx_1083_);
lean_dec(v_leftIdx_1083_);
lean_dec(v_leftIdx_1079_);
if (v___x_1086_ == 0)
{
lean_dec(v_degreeMultiple_1085_);
lean_dec(v_rightIdx_1084_);
lean_dec(v_degreeMultiple_1081_);
lean_dec(v_rightIdx_1080_);
return v___x_1082_;
}
else
{
uint8_t v___x_1087_; 
v___x_1087_ = lean_nat_dec_eq(v_rightIdx_1080_, v_rightIdx_1084_);
lean_dec(v_rightIdx_1084_);
lean_dec(v_rightIdx_1080_);
if (v___x_1087_ == 0)
{
lean_dec(v_degreeMultiple_1085_);
lean_dec(v_degreeMultiple_1081_);
return v___x_1082_;
}
else
{
uint8_t v___x_1088_; 
v___x_1088_ = lean_nat_dec_eq(v_degreeMultiple_1081_, v_degreeMultiple_1085_);
lean_dec(v_degreeMultiple_1085_);
lean_dec(v_degreeMultiple_1081_);
if (v___x_1088_ == 0)
{
return v___x_1082_;
}
else
{
return v___x_1088_;
}
}
}
}
default: 
{
lean_dec(v_degreeMultiple_1081_);
lean_dec(v_rightIdx_1080_);
lean_dec(v_leftIdx_1079_);
lean_dec(v_x_1056_);
return v___x_1082_;
}
}
}
case 6:
{
lean_object* v_leftIdx_1089_; lean_object* v_rightIdx_1090_; lean_object* v_degreeMultiple_1091_; uint8_t v___x_1092_; 
lean_dec_ref(v_inst_1054_);
v_leftIdx_1089_ = lean_ctor_get(v_x_1055_, 0);
lean_inc(v_leftIdx_1089_);
v_rightIdx_1090_ = lean_ctor_get(v_x_1055_, 1);
lean_inc(v_rightIdx_1090_);
v_degreeMultiple_1091_ = lean_ctor_get(v_x_1055_, 2);
lean_inc(v_degreeMultiple_1091_);
lean_dec_ref_known(v_x_1055_, 3);
v___x_1092_ = 0;
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
lean_dec(v_degreeMultiple_1091_);
lean_dec(v_rightIdx_1090_);
lean_dec(v_leftIdx_1089_);
return v___x_1092_;
}
case 2:
{
lean_dec(v_degreeMultiple_1091_);
lean_dec(v_rightIdx_1090_);
lean_dec(v_leftIdx_1089_);
return v___x_1092_;
}
case 3:
{
lean_dec(v_degreeMultiple_1091_);
lean_dec(v_rightIdx_1090_);
lean_dec(v_leftIdx_1089_);
return v___x_1092_;
}
case 6:
{
lean_object* v_leftIdx_1093_; lean_object* v_rightIdx_1094_; lean_object* v_degreeMultiple_1095_; uint8_t v___x_1096_; 
v_leftIdx_1093_ = lean_ctor_get(v_x_1056_, 0);
lean_inc(v_leftIdx_1093_);
v_rightIdx_1094_ = lean_ctor_get(v_x_1056_, 1);
lean_inc(v_rightIdx_1094_);
v_degreeMultiple_1095_ = lean_ctor_get(v_x_1056_, 2);
lean_inc(v_degreeMultiple_1095_);
lean_dec_ref_known(v_x_1056_, 3);
v___x_1096_ = lean_nat_dec_eq(v_leftIdx_1089_, v_leftIdx_1093_);
lean_dec(v_leftIdx_1093_);
lean_dec(v_leftIdx_1089_);
if (v___x_1096_ == 0)
{
lean_dec(v_degreeMultiple_1095_);
lean_dec(v_rightIdx_1094_);
lean_dec(v_degreeMultiple_1091_);
lean_dec(v_rightIdx_1090_);
return v___x_1092_;
}
else
{
uint8_t v___x_1097_; 
v___x_1097_ = lean_nat_dec_eq(v_rightIdx_1090_, v_rightIdx_1094_);
lean_dec(v_rightIdx_1094_);
lean_dec(v_rightIdx_1090_);
if (v___x_1097_ == 0)
{
lean_dec(v_degreeMultiple_1095_);
lean_dec(v_degreeMultiple_1091_);
return v___x_1092_;
}
else
{
uint8_t v___x_1098_; 
v___x_1098_ = lean_nat_dec_eq(v_degreeMultiple_1091_, v_degreeMultiple_1095_);
lean_dec(v_degreeMultiple_1095_);
lean_dec(v_degreeMultiple_1091_);
if (v___x_1098_ == 0)
{
return v___x_1092_;
}
else
{
return v___x_1098_;
}
}
}
}
default: 
{
lean_dec(v_degreeMultiple_1091_);
lean_dec(v_rightIdx_1090_);
lean_dec(v_leftIdx_1089_);
lean_dec(v_x_1056_);
return v___x_1092_;
}
}
}
case 7:
{
lean_object* v_idx_1099_; lean_object* v_degreeMultiple_1100_; uint8_t v___x_1101_; 
lean_dec_ref(v_inst_1054_);
v_idx_1099_ = lean_ctor_get(v_x_1055_, 0);
lean_inc(v_idx_1099_);
v_degreeMultiple_1100_ = lean_ctor_get(v_x_1055_, 1);
lean_inc(v_degreeMultiple_1100_);
lean_dec_ref_known(v_x_1055_, 2);
v___x_1101_ = 0;
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
lean_dec(v_degreeMultiple_1100_);
lean_dec(v_idx_1099_);
return v___x_1101_;
}
case 2:
{
lean_dec(v_degreeMultiple_1100_);
lean_dec(v_idx_1099_);
return v___x_1101_;
}
case 3:
{
lean_dec(v_degreeMultiple_1100_);
lean_dec(v_idx_1099_);
return v___x_1101_;
}
case 7:
{
lean_object* v_idx_1102_; lean_object* v_degreeMultiple_1103_; uint8_t v___x_1104_; 
v_idx_1102_ = lean_ctor_get(v_x_1056_, 0);
lean_inc(v_idx_1102_);
v_degreeMultiple_1103_ = lean_ctor_get(v_x_1056_, 1);
lean_inc(v_degreeMultiple_1103_);
lean_dec_ref_known(v_x_1056_, 2);
v___x_1104_ = lean_nat_dec_eq(v_idx_1099_, v_idx_1102_);
lean_dec(v_idx_1102_);
lean_dec(v_idx_1099_);
if (v___x_1104_ == 0)
{
lean_dec(v_degreeMultiple_1103_);
lean_dec(v_degreeMultiple_1100_);
return v___x_1101_;
}
else
{
uint8_t v___x_1105_; 
v___x_1105_ = lean_nat_dec_eq(v_degreeMultiple_1100_, v_degreeMultiple_1103_);
lean_dec(v_degreeMultiple_1103_);
lean_dec(v_degreeMultiple_1100_);
if (v___x_1105_ == 0)
{
return v___x_1101_;
}
else
{
return v___x_1105_;
}
}
}
default: 
{
lean_dec(v_degreeMultiple_1100_);
lean_dec(v_idx_1099_);
lean_dec(v_x_1056_);
return v___x_1101_;
}
}
}
default: 
{
lean_object* v_leftIdx_1106_; lean_object* v_rightIdx_1107_; lean_object* v_degreeMultiple_1108_; uint8_t v___x_1109_; 
lean_dec_ref(v_inst_1054_);
v_leftIdx_1106_ = lean_ctor_get(v_x_1055_, 0);
lean_inc(v_leftIdx_1106_);
v_rightIdx_1107_ = lean_ctor_get(v_x_1055_, 1);
lean_inc(v_rightIdx_1107_);
v_degreeMultiple_1108_ = lean_ctor_get(v_x_1055_, 2);
lean_inc(v_degreeMultiple_1108_);
lean_dec_ref_known(v_x_1055_, 3);
v___x_1109_ = 0;
switch(lean_obj_tag(v_x_1056_))
{
case 1:
{
lean_dec(v_degreeMultiple_1108_);
lean_dec(v_rightIdx_1107_);
lean_dec(v_leftIdx_1106_);
return v___x_1109_;
}
case 2:
{
lean_dec(v_degreeMultiple_1108_);
lean_dec(v_rightIdx_1107_);
lean_dec(v_leftIdx_1106_);
return v___x_1109_;
}
case 3:
{
lean_dec(v_degreeMultiple_1108_);
lean_dec(v_rightIdx_1107_);
lean_dec(v_leftIdx_1106_);
return v___x_1109_;
}
case 8:
{
lean_object* v_leftIdx_1110_; lean_object* v_rightIdx_1111_; lean_object* v_degreeMultiple_1112_; uint8_t v___x_1113_; 
v_leftIdx_1110_ = lean_ctor_get(v_x_1056_, 0);
lean_inc(v_leftIdx_1110_);
v_rightIdx_1111_ = lean_ctor_get(v_x_1056_, 1);
lean_inc(v_rightIdx_1111_);
v_degreeMultiple_1112_ = lean_ctor_get(v_x_1056_, 2);
lean_inc(v_degreeMultiple_1112_);
lean_dec_ref_known(v_x_1056_, 3);
v___x_1113_ = lean_nat_dec_eq(v_leftIdx_1106_, v_leftIdx_1110_);
lean_dec(v_leftIdx_1110_);
lean_dec(v_leftIdx_1106_);
if (v___x_1113_ == 0)
{
lean_dec(v_degreeMultiple_1112_);
lean_dec(v_rightIdx_1111_);
lean_dec(v_degreeMultiple_1108_);
lean_dec(v_rightIdx_1107_);
return v___x_1109_;
}
else
{
uint8_t v___x_1114_; 
v___x_1114_ = lean_nat_dec_eq(v_rightIdx_1107_, v_rightIdx_1111_);
lean_dec(v_rightIdx_1111_);
lean_dec(v_rightIdx_1107_);
if (v___x_1114_ == 0)
{
lean_dec(v_degreeMultiple_1112_);
lean_dec(v_degreeMultiple_1108_);
return v___x_1109_;
}
else
{
uint8_t v___x_1115_; 
v___x_1115_ = lean_nat_dec_eq(v_degreeMultiple_1108_, v_degreeMultiple_1112_);
lean_dec(v_degreeMultiple_1112_);
lean_dec(v_degreeMultiple_1108_);
if (v___x_1115_ == 0)
{
return v___x_1109_;
}
else
{
return v___x_1115_;
}
}
}
}
default: 
{
lean_dec(v_degreeMultiple_1108_);
lean_dec(v_rightIdx_1107_);
lean_dec(v_leftIdx_1106_);
lean_dec(v_x_1056_);
return v___x_1109_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg___boxed(lean_object* v_inst_1116_, lean_object* v_x_1117_, lean_object* v_x_1118_){
_start:
{
uint8_t v_res_1119_; lean_object* v_r_1120_; 
v_res_1119_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(v_inst_1116_, v_x_1117_, v_x_1118_);
v_r_1120_ = lean_box(v_res_1119_);
return v_r_1120_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq(lean_object* v_F_1121_, lean_object* v_inst_1122_, lean_object* v_x_1123_, lean_object* v_x_1124_){
_start:
{
uint8_t v___x_1125_; 
v___x_1125_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(v_inst_1122_, v_x_1123_, v_x_1124_);
return v___x_1125_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___boxed(lean_object* v_F_1126_, lean_object* v_inst_1127_, lean_object* v_x_1128_, lean_object* v_x_1129_){
_start:
{
uint8_t v_res_1130_; lean_object* v_r_1131_; 
v_res_1130_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq(v_F_1126_, v_inst_1127_, v_x_1128_, v_x_1129_);
v_r_1131_ = lean_box(v_res_1130_);
return v_r_1131_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___redArg(lean_object* v_inst_1132_, lean_object* v_x_1133_, lean_object* v_x_1134_){
_start:
{
uint8_t v___x_1135_; 
v___x_1135_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(v_inst_1132_, v_x_1133_, v_x_1134_);
return v___x_1135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___redArg___boxed(lean_object* v_inst_1136_, lean_object* v_x_1137_, lean_object* v_x_1138_){
_start:
{
uint8_t v_res_1139_; lean_object* v_r_1140_; 
v_res_1139_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___redArg(v_inst_1136_, v_x_1137_, v_x_1138_);
v_r_1140_ = lean_box(v_res_1139_);
return v_r_1140_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode(lean_object* v_F_1141_, lean_object* v_inst_1142_, lean_object* v_x_1143_, lean_object* v_x_1144_){
_start:
{
uint8_t v___x_1145_; 
v___x_1145_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(v_inst_1142_, v_x_1143_, v_x_1144_);
return v___x_1145_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode___boxed(lean_object* v_F_1146_, lean_object* v_inst_1147_, lean_object* v_x_1148_, lean_object* v_x_1149_){
_start:
{
uint8_t v_res_1150_; lean_object* v_r_1151_; 
v_res_1150_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode(v_F_1146_, v_inst_1147_, v_x_1148_, v_x_1149_);
v_r_1151_ = lean_box(v_res_1150_);
return v_r_1151_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg(lean_object* v_inst_1197_, lean_object* v_x_1198_, lean_object* v_prec_1199_){
_start:
{
lean_object* v___y_1201_; lean_object* v___y_1208_; lean_object* v___y_1215_; 
switch(lean_obj_tag(v_x_1198_))
{
case 0:
{
lean_object* v_v_1221_; lean_object* v___y_1223_; lean_object* v___x_1231_; uint8_t v___x_1232_; 
lean_dec_ref(v_inst_1197_);
v_v_1221_ = lean_ctor_get(v_x_1198_, 0);
lean_inc_ref(v_v_1221_);
lean_dec_ref_known(v_x_1198_, 1);
v___x_1231_ = lean_unsigned_to_nat(1024u);
v___x_1232_ = lean_nat_dec_le(v___x_1231_, v_prec_1199_);
if (v___x_1232_ == 0)
{
lean_object* v___x_1233_; 
v___x_1233_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1223_ = v___x_1233_;
goto v___jp_1222_;
}
else
{
lean_object* v___x_1234_; 
v___x_1234_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1223_ = v___x_1234_;
goto v___jp_1222_;
}
v___jp_1222_:
{
lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; uint8_t v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; 
v___x_1224_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__8));
v___x_1225_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable_repr___redArg(v_v_1221_);
v___x_1226_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1226_, 0, v___x_1224_);
lean_ctor_set(v___x_1226_, 1, v___x_1225_);
lean_inc(v___y_1223_);
v___x_1227_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1227_, 0, v___y_1223_);
lean_ctor_set(v___x_1227_, 1, v___x_1226_);
v___x_1228_ = 0;
v___x_1229_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1229_, 0, v___x_1227_);
lean_ctor_set_uint8(v___x_1229_, sizeof(void*)*1, v___x_1228_);
v___x_1230_ = l_Repr_addAppParen(v___x_1229_, v_prec_1199_);
return v___x_1230_;
}
}
case 1:
{
lean_object* v___x_1235_; uint8_t v___x_1236_; 
lean_dec_ref(v_inst_1197_);
v___x_1235_ = lean_unsigned_to_nat(1024u);
v___x_1236_ = lean_nat_dec_le(v___x_1235_, v_prec_1199_);
if (v___x_1236_ == 0)
{
lean_object* v___x_1237_; 
v___x_1237_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1215_ = v___x_1237_;
goto v___jp_1214_;
}
else
{
lean_object* v___x_1238_; 
v___x_1238_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1215_ = v___x_1238_;
goto v___jp_1214_;
}
}
case 2:
{
lean_object* v___x_1239_; uint8_t v___x_1240_; 
lean_dec_ref(v_inst_1197_);
v___x_1239_ = lean_unsigned_to_nat(1024u);
v___x_1240_ = lean_nat_dec_le(v___x_1239_, v_prec_1199_);
if (v___x_1240_ == 0)
{
lean_object* v___x_1241_; 
v___x_1241_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1208_ = v___x_1241_;
goto v___jp_1207_;
}
else
{
lean_object* v___x_1242_; 
v___x_1242_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1208_ = v___x_1242_;
goto v___jp_1207_;
}
}
case 3:
{
lean_object* v___x_1243_; uint8_t v___x_1244_; 
lean_dec_ref(v_inst_1197_);
v___x_1243_ = lean_unsigned_to_nat(1024u);
v___x_1244_ = lean_nat_dec_le(v___x_1243_, v_prec_1199_);
if (v___x_1244_ == 0)
{
lean_object* v___x_1245_; 
v___x_1245_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1201_ = v___x_1245_;
goto v___jp_1200_;
}
else
{
lean_object* v___x_1246_; 
v___x_1246_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1201_ = v___x_1246_;
goto v___jp_1200_;
}
}
case 4:
{
lean_object* v_c_1247_; lean_object* v___y_1249_; lean_object* v___x_1258_; uint8_t v___x_1259_; 
v_c_1247_ = lean_ctor_get(v_x_1198_, 0);
lean_inc(v_c_1247_);
lean_dec_ref_known(v_x_1198_, 1);
v___x_1258_ = lean_unsigned_to_nat(1024u);
v___x_1259_ = lean_nat_dec_le(v___x_1258_, v_prec_1199_);
if (v___x_1259_ == 0)
{
lean_object* v___x_1260_; 
v___x_1260_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1249_ = v___x_1260_;
goto v___jp_1248_;
}
else
{
lean_object* v___x_1261_; 
v___x_1261_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1249_ = v___x_1261_;
goto v___jp_1248_;
}
v___jp_1248_:
{
lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; uint8_t v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; 
v___x_1250_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__11));
v___x_1251_ = lean_unsigned_to_nat(1024u);
v___x_1252_ = lean_apply_2(v_inst_1197_, v_c_1247_, v___x_1251_);
v___x_1253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1253_, 0, v___x_1250_);
lean_ctor_set(v___x_1253_, 1, v___x_1252_);
lean_inc(v___y_1249_);
v___x_1254_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1254_, 0, v___y_1249_);
lean_ctor_set(v___x_1254_, 1, v___x_1253_);
v___x_1255_ = 0;
v___x_1256_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1256_, 0, v___x_1254_);
lean_ctor_set_uint8(v___x_1256_, sizeof(void*)*1, v___x_1255_);
v___x_1257_ = l_Repr_addAppParen(v___x_1256_, v_prec_1199_);
return v___x_1257_;
}
}
case 5:
{
lean_object* v_leftIdx_1262_; lean_object* v_rightIdx_1263_; lean_object* v_degreeMultiple_1264_; lean_object* v___y_1266_; lean_object* v___x_1284_; uint8_t v___x_1285_; 
lean_dec_ref(v_inst_1197_);
v_leftIdx_1262_ = lean_ctor_get(v_x_1198_, 0);
lean_inc(v_leftIdx_1262_);
v_rightIdx_1263_ = lean_ctor_get(v_x_1198_, 1);
lean_inc(v_rightIdx_1263_);
v_degreeMultiple_1264_ = lean_ctor_get(v_x_1198_, 2);
lean_inc(v_degreeMultiple_1264_);
lean_dec_ref_known(v_x_1198_, 3);
v___x_1284_ = lean_unsigned_to_nat(1024u);
v___x_1285_ = lean_nat_dec_le(v___x_1284_, v_prec_1199_);
if (v___x_1285_ == 0)
{
lean_object* v___x_1286_; 
v___x_1286_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1266_ = v___x_1286_;
goto v___jp_1265_;
}
else
{
lean_object* v___x_1287_; 
v___x_1287_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1266_ = v___x_1287_;
goto v___jp_1265_;
}
v___jp_1265_:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; uint8_t v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; 
v___x_1267_ = lean_box(1);
v___x_1268_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__14));
v___x_1269_ = l_Nat_reprFast(v_leftIdx_1262_);
v___x_1270_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1269_);
v___x_1271_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1268_);
lean_ctor_set(v___x_1271_, 1, v___x_1270_);
v___x_1272_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1271_);
lean_ctor_set(v___x_1272_, 1, v___x_1267_);
v___x_1273_ = l_Nat_reprFast(v_rightIdx_1263_);
v___x_1274_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1273_);
v___x_1275_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1275_, 0, v___x_1272_);
lean_ctor_set(v___x_1275_, 1, v___x_1274_);
v___x_1276_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1275_);
lean_ctor_set(v___x_1276_, 1, v___x_1267_);
v___x_1277_ = l_Nat_reprFast(v_degreeMultiple_1264_);
v___x_1278_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1277_);
v___x_1279_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1279_, 0, v___x_1276_);
lean_ctor_set(v___x_1279_, 1, v___x_1278_);
lean_inc(v___y_1266_);
v___x_1280_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1280_, 0, v___y_1266_);
lean_ctor_set(v___x_1280_, 1, v___x_1279_);
v___x_1281_ = 0;
v___x_1282_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1282_, 0, v___x_1280_);
lean_ctor_set_uint8(v___x_1282_, sizeof(void*)*1, v___x_1281_);
v___x_1283_ = l_Repr_addAppParen(v___x_1282_, v_prec_1199_);
return v___x_1283_;
}
}
case 6:
{
lean_object* v_leftIdx_1288_; lean_object* v_rightIdx_1289_; lean_object* v_degreeMultiple_1290_; lean_object* v___y_1292_; lean_object* v___x_1310_; uint8_t v___x_1311_; 
lean_dec_ref(v_inst_1197_);
v_leftIdx_1288_ = lean_ctor_get(v_x_1198_, 0);
lean_inc(v_leftIdx_1288_);
v_rightIdx_1289_ = lean_ctor_get(v_x_1198_, 1);
lean_inc(v_rightIdx_1289_);
v_degreeMultiple_1290_ = lean_ctor_get(v_x_1198_, 2);
lean_inc(v_degreeMultiple_1290_);
lean_dec_ref_known(v_x_1198_, 3);
v___x_1310_ = lean_unsigned_to_nat(1024u);
v___x_1311_ = lean_nat_dec_le(v___x_1310_, v_prec_1199_);
if (v___x_1311_ == 0)
{
lean_object* v___x_1312_; 
v___x_1312_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1292_ = v___x_1312_;
goto v___jp_1291_;
}
else
{
lean_object* v___x_1313_; 
v___x_1313_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1292_ = v___x_1313_;
goto v___jp_1291_;
}
v___jp_1291_:
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; uint8_t v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; 
v___x_1293_ = lean_box(1);
v___x_1294_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__17));
v___x_1295_ = l_Nat_reprFast(v_leftIdx_1288_);
v___x_1296_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1296_, 0, v___x_1295_);
v___x_1297_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1297_, 0, v___x_1294_);
lean_ctor_set(v___x_1297_, 1, v___x_1296_);
v___x_1298_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1297_);
lean_ctor_set(v___x_1298_, 1, v___x_1293_);
v___x_1299_ = l_Nat_reprFast(v_rightIdx_1289_);
v___x_1300_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1300_, 0, v___x_1299_);
v___x_1301_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1301_, 0, v___x_1298_);
lean_ctor_set(v___x_1301_, 1, v___x_1300_);
v___x_1302_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1302_, 0, v___x_1301_);
lean_ctor_set(v___x_1302_, 1, v___x_1293_);
v___x_1303_ = l_Nat_reprFast(v_degreeMultiple_1290_);
v___x_1304_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1304_, 0, v___x_1303_);
v___x_1305_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1305_, 0, v___x_1302_);
lean_ctor_set(v___x_1305_, 1, v___x_1304_);
lean_inc(v___y_1292_);
v___x_1306_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1306_, 0, v___y_1292_);
lean_ctor_set(v___x_1306_, 1, v___x_1305_);
v___x_1307_ = 0;
v___x_1308_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1308_, 0, v___x_1306_);
lean_ctor_set_uint8(v___x_1308_, sizeof(void*)*1, v___x_1307_);
v___x_1309_ = l_Repr_addAppParen(v___x_1308_, v_prec_1199_);
return v___x_1309_;
}
}
case 7:
{
lean_object* v_idx_1314_; lean_object* v_degreeMultiple_1315_; lean_object* v___x_1317_; uint8_t v_isShared_1318_; uint8_t v_isSharedCheck_1340_; 
lean_dec_ref(v_inst_1197_);
v_idx_1314_ = lean_ctor_get(v_x_1198_, 0);
v_degreeMultiple_1315_ = lean_ctor_get(v_x_1198_, 1);
v_isSharedCheck_1340_ = !lean_is_exclusive(v_x_1198_);
if (v_isSharedCheck_1340_ == 0)
{
v___x_1317_ = v_x_1198_;
v_isShared_1318_ = v_isSharedCheck_1340_;
goto v_resetjp_1316_;
}
else
{
lean_inc(v_degreeMultiple_1315_);
lean_inc(v_idx_1314_);
lean_dec(v_x_1198_);
v___x_1317_ = lean_box(0);
v_isShared_1318_ = v_isSharedCheck_1340_;
goto v_resetjp_1316_;
}
v_resetjp_1316_:
{
lean_object* v___y_1320_; lean_object* v___x_1336_; uint8_t v___x_1337_; 
v___x_1336_ = lean_unsigned_to_nat(1024u);
v___x_1337_ = lean_nat_dec_le(v___x_1336_, v_prec_1199_);
if (v___x_1337_ == 0)
{
lean_object* v___x_1338_; 
v___x_1338_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1320_ = v___x_1338_;
goto v___jp_1319_;
}
else
{
lean_object* v___x_1339_; 
v___x_1339_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1320_ = v___x_1339_;
goto v___jp_1319_;
}
v___jp_1319_:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1326_; 
v___x_1321_ = lean_box(1);
v___x_1322_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__20));
v___x_1323_ = l_Nat_reprFast(v_idx_1314_);
v___x_1324_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1323_);
if (v_isShared_1318_ == 0)
{
lean_ctor_set_tag(v___x_1317_, 5);
lean_ctor_set(v___x_1317_, 1, v___x_1324_);
lean_ctor_set(v___x_1317_, 0, v___x_1322_);
v___x_1326_ = v___x_1317_;
goto v_reusejp_1325_;
}
else
{
lean_object* v_reuseFailAlloc_1335_; 
v_reuseFailAlloc_1335_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1335_, 0, v___x_1322_);
lean_ctor_set(v_reuseFailAlloc_1335_, 1, v___x_1324_);
v___x_1326_ = v_reuseFailAlloc_1335_;
goto v_reusejp_1325_;
}
v_reusejp_1325_:
{
lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; uint8_t v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1327_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
lean_ctor_set(v___x_1327_, 1, v___x_1321_);
v___x_1328_ = l_Nat_reprFast(v_degreeMultiple_1315_);
v___x_1329_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1329_, 0, v___x_1328_);
v___x_1330_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1327_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
lean_inc(v___y_1320_);
v___x_1331_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1331_, 0, v___y_1320_);
lean_ctor_set(v___x_1331_, 1, v___x_1330_);
v___x_1332_ = 0;
v___x_1333_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1333_, 0, v___x_1331_);
lean_ctor_set_uint8(v___x_1333_, sizeof(void*)*1, v___x_1332_);
v___x_1334_ = l_Repr_addAppParen(v___x_1333_, v_prec_1199_);
return v___x_1334_;
}
}
}
}
default: 
{
lean_object* v_leftIdx_1341_; lean_object* v_rightIdx_1342_; lean_object* v_degreeMultiple_1343_; lean_object* v___y_1345_; lean_object* v___x_1363_; uint8_t v___x_1364_; 
lean_dec_ref(v_inst_1197_);
v_leftIdx_1341_ = lean_ctor_get(v_x_1198_, 0);
lean_inc(v_leftIdx_1341_);
v_rightIdx_1342_ = lean_ctor_get(v_x_1198_, 1);
lean_inc(v_rightIdx_1342_);
v_degreeMultiple_1343_ = lean_ctor_get(v_x_1198_, 2);
lean_inc(v_degreeMultiple_1343_);
lean_dec_ref_known(v_x_1198_, 3);
v___x_1363_ = lean_unsigned_to_nat(1024u);
v___x_1364_ = lean_nat_dec_le(v___x_1363_, v_prec_1199_);
if (v___x_1364_ == 0)
{
lean_object* v___x_1365_; 
v___x_1365_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__7);
v___y_1345_ = v___x_1365_;
goto v___jp_1344_;
}
else
{
lean_object* v___x_1366_; 
v___x_1366_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprEntry_repr___closed__8);
v___y_1345_ = v___x_1366_;
goto v___jp_1344_;
}
v___jp_1344_:
{
lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; uint8_t v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; 
v___x_1346_ = lean_box(1);
v___x_1347_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__23));
v___x_1348_ = l_Nat_reprFast(v_leftIdx_1341_);
v___x_1349_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1349_, 0, v___x_1348_);
v___x_1350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1350_, 0, v___x_1347_);
lean_ctor_set(v___x_1350_, 1, v___x_1349_);
v___x_1351_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1351_, 0, v___x_1350_);
lean_ctor_set(v___x_1351_, 1, v___x_1346_);
v___x_1352_ = l_Nat_reprFast(v_rightIdx_1342_);
v___x_1353_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1352_);
v___x_1354_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1354_, 0, v___x_1351_);
lean_ctor_set(v___x_1354_, 1, v___x_1353_);
v___x_1355_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1355_, 0, v___x_1354_);
lean_ctor_set(v___x_1355_, 1, v___x_1346_);
v___x_1356_ = l_Nat_reprFast(v_degreeMultiple_1343_);
v___x_1357_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1357_, 0, v___x_1356_);
v___x_1358_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1358_, 0, v___x_1355_);
lean_ctor_set(v___x_1358_, 1, v___x_1357_);
lean_inc(v___y_1345_);
v___x_1359_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1359_, 0, v___y_1345_);
lean_ctor_set(v___x_1359_, 1, v___x_1358_);
v___x_1360_ = 0;
v___x_1361_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1361_, 0, v___x_1359_);
lean_ctor_set_uint8(v___x_1361_, sizeof(void*)*1, v___x_1360_);
v___x_1362_ = l_Repr_addAppParen(v___x_1361_, v_prec_1199_);
return v___x_1362_;
}
}
}
v___jp_1200_:
{
lean_object* v___x_1202_; lean_object* v___x_1203_; uint8_t v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; 
v___x_1202_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__1));
lean_inc(v___y_1201_);
v___x_1203_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___y_1201_);
lean_ctor_set(v___x_1203_, 1, v___x_1202_);
v___x_1204_ = 0;
v___x_1205_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1205_, 0, v___x_1203_);
lean_ctor_set_uint8(v___x_1205_, sizeof(void*)*1, v___x_1204_);
v___x_1206_ = l_Repr_addAppParen(v___x_1205_, v_prec_1199_);
return v___x_1206_;
}
v___jp_1207_:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; uint8_t v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; 
v___x_1209_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__3));
lean_inc(v___y_1208_);
v___x_1210_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1210_, 0, v___y_1208_);
lean_ctor_set(v___x_1210_, 1, v___x_1209_);
v___x_1211_ = 0;
v___x_1212_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1212_, 0, v___x_1210_);
lean_ctor_set_uint8(v___x_1212_, sizeof(void*)*1, v___x_1211_);
v___x_1213_ = l_Repr_addAppParen(v___x_1212_, v_prec_1199_);
return v___x_1213_;
}
v___jp_1214_:
{
lean_object* v___x_1216_; lean_object* v___x_1217_; uint8_t v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1216_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___closed__5));
lean_inc(v___y_1215_);
v___x_1217_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1217_, 0, v___y_1215_);
lean_ctor_set(v___x_1217_, 1, v___x_1216_);
v___x_1218_ = 0;
v___x_1219_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1219_, 0, v___x_1217_);
lean_ctor_set_uint8(v___x_1219_, sizeof(void*)*1, v___x_1218_);
v___x_1220_ = l_Repr_addAppParen(v___x_1219_, v_prec_1199_);
return v___x_1220_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg___boxed(lean_object* v_inst_1367_, lean_object* v_x_1368_, lean_object* v_prec_1369_){
_start:
{
lean_object* v_res_1370_; 
v_res_1370_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg(v_inst_1367_, v_x_1368_, v_prec_1369_);
lean_dec(v_prec_1369_);
return v_res_1370_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr(lean_object* v_F_1371_, lean_object* v_inst_1372_, lean_object* v_x_1373_, lean_object* v_prec_1374_){
_start:
{
lean_object* v___x_1375_; 
v___x_1375_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___redArg(v_inst_1372_, v_x_1373_, v_prec_1374_);
return v___x_1375_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___boxed(lean_object* v_F_1376_, lean_object* v_inst_1377_, lean_object* v_x_1378_, lean_object* v_prec_1379_){
_start:
{
lean_object* v_res_1380_; 
v_res_1380_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr(v_F_1376_, v_inst_1377_, v_x_1378_, v_prec_1379_);
lean_dec(v_prec_1379_);
return v_res_1380_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode___redArg(lean_object* v_inst_1381_){
_start:
{
lean_object* v___x_1382_; 
v___x_1382_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___boxed), 4, 2);
lean_closure_set(v___x_1382_, 0, lean_box(0));
lean_closure_set(v___x_1382_, 1, v_inst_1381_);
return v___x_1382_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode(lean_object* v_F_1383_, lean_object* v_inst_1384_){
_start:
{
lean_object* v___x_1385_; 
v___x_1385_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___boxed), 4, 2);
lean_closure_set(v___x_1385_, 0, lean_box(0));
lean_closure_set(v___x_1385_, 1, v_inst_1384_);
return v___x_1385_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___lam__0(lean_object* v_inst_1386_, lean_object* v_a_1387_, lean_object* v_b_1388_){
_start:
{
uint8_t v___x_1389_; 
v___x_1389_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionNode_decEq___redArg(v_inst_1386_, v_a_1387_, v_b_1388_);
return v___x_1389_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___lam__0___boxed(lean_object* v_inst_1390_, lean_object* v_a_1391_, lean_object* v_b_1392_){
_start:
{
uint8_t v_res_1393_; lean_object* v_r_1394_; 
v_res_1393_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___lam__0(v_inst_1390_, v_a_1391_, v_b_1392_);
v_r_1394_ = lean_box(v_res_1393_);
return v_r_1394_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(lean_object* v_inst_1395_, lean_object* v_x_1396_, lean_object* v_x_1397_){
_start:
{
lean_object* v_nodes_1398_; lean_object* v_constraintIdx_1399_; lean_object* v_nodes_1400_; lean_object* v_constraintIdx_1401_; lean_object* v___f_1402_; uint8_t v___x_1403_; 
v_nodes_1398_ = lean_ctor_get(v_x_1396_, 0);
lean_inc(v_nodes_1398_);
v_constraintIdx_1399_ = lean_ctor_get(v_x_1396_, 1);
lean_inc(v_constraintIdx_1399_);
lean_dec_ref(v_x_1396_);
v_nodes_1400_ = lean_ctor_get(v_x_1397_, 0);
lean_inc(v_nodes_1400_);
v_constraintIdx_1401_ = lean_ctor_get(v_x_1397_, 1);
lean_inc(v_constraintIdx_1401_);
lean_dec_ref(v_x_1397_);
v___f_1402_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1402_, 0, v_inst_1395_);
v___x_1403_ = l_instDecidableEqList___redArg(v___f_1402_, v_nodes_1398_, v_nodes_1400_);
if (v___x_1403_ == 0)
{
lean_dec(v_constraintIdx_1401_);
lean_dec(v_constraintIdx_1399_);
return v___x_1403_;
}
else
{
lean_object* v___x_1404_; uint8_t v___x_1405_; 
v___x_1404_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_1405_ = l_instDecidableEqList___redArg(v___x_1404_, v_constraintIdx_1399_, v_constraintIdx_1401_);
return v___x_1405_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg___boxed(lean_object* v_inst_1406_, lean_object* v_x_1407_, lean_object* v_x_1408_){
_start:
{
uint8_t v_res_1409_; lean_object* v_r_1410_; 
v_res_1409_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(v_inst_1406_, v_x_1407_, v_x_1408_);
v_r_1410_ = lean_box(v_res_1409_);
return v_r_1410_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq(lean_object* v_F_1411_, lean_object* v_inst_1412_, lean_object* v_x_1413_, lean_object* v_x_1414_){
_start:
{
uint8_t v___x_1415_; 
v___x_1415_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(v_inst_1412_, v_x_1413_, v_x_1414_);
return v___x_1415_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___boxed(lean_object* v_F_1416_, lean_object* v_inst_1417_, lean_object* v_x_1418_, lean_object* v_x_1419_){
_start:
{
uint8_t v_res_1420_; lean_object* v_r_1421_; 
v_res_1420_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq(v_F_1416_, v_inst_1417_, v_x_1418_, v_x_1419_);
v_r_1421_ = lean_box(v_res_1420_);
return v_r_1421_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___redArg(lean_object* v_inst_1422_, lean_object* v_x_1423_, lean_object* v_x_1424_){
_start:
{
uint8_t v___x_1425_; 
v___x_1425_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(v_inst_1422_, v_x_1423_, v_x_1424_);
return v___x_1425_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___redArg___boxed(lean_object* v_inst_1426_, lean_object* v_x_1427_, lean_object* v_x_1428_){
_start:
{
uint8_t v_res_1429_; lean_object* v_r_1430_; 
v_res_1429_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___redArg(v_inst_1426_, v_x_1427_, v_x_1428_);
v_r_1430_ = lean_box(v_res_1429_);
return v_r_1430_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag(lean_object* v_F_1431_, lean_object* v_inst_1432_, lean_object* v_x_1433_, lean_object* v_x_1434_){
_start:
{
uint8_t v___x_1435_; 
v___x_1435_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(v_inst_1432_, v_x_1433_, v_x_1434_);
return v___x_1435_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag___boxed(lean_object* v_F_1436_, lean_object* v_inst_1437_, lean_object* v_x_1438_, lean_object* v_x_1439_){
_start:
{
uint8_t v_res_1440_; lean_object* v_r_1441_; 
v_res_1440_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag(v_F_1436_, v_inst_1437_, v_x_1438_, v_x_1439_);
v_r_1441_ = lean_box(v_res_1440_);
return v_r_1441_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg(lean_object* v_inst_1455_, lean_object* v_x_1456_){
_start:
{
lean_object* v_nodes_1457_; lean_object* v_constraintIdx_1458_; lean_object* v___x_1460_; uint8_t v_isShared_1461_; uint8_t v_isSharedCheck_1493_; 
v_nodes_1457_ = lean_ctor_get(v_x_1456_, 0);
v_constraintIdx_1458_ = lean_ctor_get(v_x_1456_, 1);
v_isSharedCheck_1493_ = !lean_is_exclusive(v_x_1456_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1460_ = v_x_1456_;
v_isShared_1461_ = v_isSharedCheck_1493_;
goto v_resetjp_1459_;
}
else
{
lean_inc(v_constraintIdx_1458_);
lean_inc(v_nodes_1457_);
lean_dec(v_x_1456_);
v___x_1460_ = lean_box(0);
v_isShared_1461_ = v_isSharedCheck_1493_;
goto v_resetjp_1459_;
}
v_resetjp_1459_:
{
lean_object* v___f_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1469_; 
v___f_1462_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__0));
v___x_1463_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_1464_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__4));
v___x_1465_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4);
v___x_1466_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionNode_repr___boxed), 4, 2);
lean_closure_set(v___x_1466_, 0, lean_box(0));
lean_closure_set(v___x_1466_, 1, v_inst_1455_);
v___x_1467_ = l_List_repr___redArg(v___x_1466_, v_nodes_1457_);
if (v_isShared_1461_ == 0)
{
lean_ctor_set_tag(v___x_1460_, 4);
lean_ctor_set(v___x_1460_, 1, v___x_1467_);
lean_ctor_set(v___x_1460_, 0, v___x_1465_);
v___x_1469_ = v___x_1460_;
goto v_reusejp_1468_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v___x_1465_);
lean_ctor_set(v_reuseFailAlloc_1492_, 1, v___x_1467_);
v___x_1469_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1468_;
}
v_reusejp_1468_:
{
uint8_t v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; 
v___x_1470_ = 0;
v___x_1471_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1471_, 0, v___x_1469_);
lean_ctor_set_uint8(v___x_1471_, sizeof(void*)*1, v___x_1470_);
v___x_1472_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1472_, 0, v___x_1464_);
lean_ctor_set(v___x_1472_, 1, v___x_1471_);
v___x_1473_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_1474_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1474_, 0, v___x_1472_);
lean_ctor_set(v___x_1474_, 1, v___x_1473_);
v___x_1475_ = lean_box(1);
v___x_1476_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1476_, 0, v___x_1474_);
lean_ctor_set(v___x_1476_, 1, v___x_1475_);
v___x_1477_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg___closed__6));
v___x_1478_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1478_, 0, v___x_1476_);
lean_ctor_set(v___x_1478_, 1, v___x_1477_);
v___x_1479_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1479_, 0, v___x_1478_);
lean_ctor_set(v___x_1479_, 1, v___x_1463_);
v___x_1480_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__9);
v___x_1481_ = l_List_repr_x27___redArg(v___f_1462_, v_constraintIdx_1458_);
v___x_1482_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1482_, 0, v___x_1480_);
lean_ctor_set(v___x_1482_, 1, v___x_1481_);
v___x_1483_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1483_, 0, v___x_1482_);
lean_ctor_set_uint8(v___x_1483_, sizeof(void*)*1, v___x_1470_);
v___x_1484_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1484_, 0, v___x_1479_);
lean_ctor_set(v___x_1484_, 1, v___x_1483_);
v___x_1485_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10);
v___x_1486_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_1487_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1487_, 0, v___x_1486_);
lean_ctor_set(v___x_1487_, 1, v___x_1484_);
v___x_1488_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_1489_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1489_, 0, v___x_1487_);
lean_ctor_set(v___x_1489_, 1, v___x_1488_);
v___x_1490_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1490_, 0, v___x_1485_);
lean_ctor_set(v___x_1490_, 1, v___x_1489_);
v___x_1491_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1491_, 0, v___x_1490_);
lean_ctor_set_uint8(v___x_1491_, sizeof(void*)*1, v___x_1470_);
return v___x_1491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr(lean_object* v_F_1494_, lean_object* v_inst_1495_, lean_object* v_x_1496_, lean_object* v_prec_1497_){
_start:
{
lean_object* v___x_1498_; 
v___x_1498_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg(v_inst_1495_, v_x_1496_);
return v___x_1498_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___boxed(lean_object* v_F_1499_, lean_object* v_inst_1500_, lean_object* v_x_1501_, lean_object* v_prec_1502_){
_start:
{
lean_object* v_res_1503_; 
v_res_1503_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr(v_F_1499_, v_inst_1500_, v_x_1501_, v_prec_1502_);
lean_dec(v_prec_1502_);
return v_res_1503_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag___redArg(lean_object* v_inst_1504_){
_start:
{
lean_object* v___x_1505_; 
v___x_1505_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___boxed), 4, 2);
lean_closure_set(v___x_1505_, 0, lean_box(0));
lean_closure_set(v___x_1505_, 1, v_inst_1504_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag(lean_object* v_F_1506_, lean_object* v_inst_1507_){
_start:
{
lean_object* v___x_1508_; 
v___x_1508_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___boxed), 4, 2);
lean_closure_set(v___x_1508_, 0, lean_box(0));
lean_closure_set(v___x_1508_, 1, v_inst_1507_);
return v___x_1508_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(lean_object* v_fo_1509_, lean_object* v_e_1510_, lean_object* v_x_1511_){
_start:
{
switch(lean_obj_tag(v_x_1511_))
{
case 0:
{
lean_object* v_v_1512_; lean_object* v_evalVar_1513_; lean_object* v___x_1514_; 
lean_dec_ref(v_fo_1509_);
v_v_1512_ = lean_ctor_get(v_x_1511_, 0);
lean_inc_ref(v_v_1512_);
lean_dec_ref_known(v_x_1511_, 1);
v_evalVar_1513_ = lean_ctor_get(v_e_1510_, 1);
lean_inc(v_evalVar_1513_);
lean_dec_ref(v_e_1510_);
v___x_1514_ = lean_apply_1(v_evalVar_1513_, v_v_1512_);
return v___x_1514_;
}
case 1:
{
lean_object* v_evalIsFirstRow_1515_; 
lean_dec_ref(v_fo_1509_);
v_evalIsFirstRow_1515_ = lean_ctor_get(v_e_1510_, 2);
lean_inc(v_evalIsFirstRow_1515_);
lean_dec_ref(v_e_1510_);
return v_evalIsFirstRow_1515_;
}
case 2:
{
lean_object* v_evalIsLastRow_1516_; 
lean_dec_ref(v_fo_1509_);
v_evalIsLastRow_1516_ = lean_ctor_get(v_e_1510_, 3);
lean_inc(v_evalIsLastRow_1516_);
lean_dec_ref(v_e_1510_);
return v_evalIsLastRow_1516_;
}
case 3:
{
lean_object* v_evalIsTransition_1517_; 
lean_dec_ref(v_fo_1509_);
v_evalIsTransition_1517_ = lean_ctor_get(v_e_1510_, 4);
lean_inc(v_evalIsTransition_1517_);
lean_dec_ref(v_e_1510_);
return v_evalIsTransition_1517_;
}
case 4:
{
lean_object* v_c_1518_; lean_object* v_evalConst_1519_; lean_object* v___x_1520_; 
lean_dec_ref(v_fo_1509_);
v_c_1518_ = lean_ctor_get(v_x_1511_, 0);
lean_inc(v_c_1518_);
lean_dec_ref_known(v_x_1511_, 1);
v_evalConst_1519_ = lean_ctor_get(v_e_1510_, 0);
lean_inc(v_evalConst_1519_);
lean_dec_ref(v_e_1510_);
v___x_1520_ = lean_apply_1(v_evalConst_1519_, v_c_1518_);
return v___x_1520_;
}
case 5:
{
lean_object* v_toRingOps_1521_; lean_object* v_toSemiringOps_1522_; lean_object* v_x_1523_; lean_object* v_y_1524_; lean_object* v_add_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; 
v_toRingOps_1521_ = lean_ctor_get(v_fo_1509_, 0);
v_toSemiringOps_1522_ = lean_ctor_get(v_toRingOps_1521_, 0);
v_x_1523_ = lean_ctor_get(v_x_1511_, 0);
lean_inc(v_x_1523_);
v_y_1524_ = lean_ctor_get(v_x_1511_, 1);
lean_inc(v_y_1524_);
lean_dec_ref_known(v_x_1511_, 2);
v_add_1525_ = lean_ctor_get(v_toSemiringOps_1522_, 3);
lean_inc(v_add_1525_);
lean_inc_ref(v_e_1510_);
lean_inc_ref(v_fo_1509_);
v___x_1526_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_x_1523_);
v___x_1527_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_y_1524_);
v___x_1528_ = lean_apply_2(v_add_1525_, v___x_1526_, v___x_1527_);
return v___x_1528_;
}
case 6:
{
lean_object* v_toRingOps_1529_; lean_object* v_x_1530_; lean_object* v_y_1531_; lean_object* v_sub_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; 
v_toRingOps_1529_ = lean_ctor_get(v_fo_1509_, 0);
v_x_1530_ = lean_ctor_get(v_x_1511_, 0);
lean_inc(v_x_1530_);
v_y_1531_ = lean_ctor_get(v_x_1511_, 1);
lean_inc(v_y_1531_);
lean_dec_ref_known(v_x_1511_, 2);
v_sub_1532_ = lean_ctor_get(v_toRingOps_1529_, 1);
lean_inc(v_sub_1532_);
lean_inc_ref(v_e_1510_);
lean_inc_ref(v_fo_1509_);
v___x_1533_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_x_1530_);
v___x_1534_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_y_1531_);
v___x_1535_ = lean_apply_2(v_sub_1532_, v___x_1533_, v___x_1534_);
return v___x_1535_;
}
case 7:
{
lean_object* v_toRingOps_1536_; lean_object* v_toSemiringOps_1537_; lean_object* v_x_1538_; lean_object* v_sub_1539_; lean_object* v_zero_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; 
v_toRingOps_1536_ = lean_ctor_get(v_fo_1509_, 0);
v_toSemiringOps_1537_ = lean_ctor_get(v_toRingOps_1536_, 0);
v_x_1538_ = lean_ctor_get(v_x_1511_, 0);
lean_inc(v_x_1538_);
lean_dec_ref_known(v_x_1511_, 1);
v_sub_1539_ = lean_ctor_get(v_toRingOps_1536_, 1);
lean_inc(v_sub_1539_);
v_zero_1540_ = lean_ctor_get(v_toSemiringOps_1537_, 0);
lean_inc(v_zero_1540_);
v___x_1541_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_x_1538_);
v___x_1542_ = lean_apply_2(v_sub_1539_, v_zero_1540_, v___x_1541_);
return v___x_1542_;
}
default: 
{
lean_object* v_toRingOps_1543_; lean_object* v_toSemiringOps_1544_; lean_object* v_x_1545_; lean_object* v_y_1546_; lean_object* v_mul_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; 
v_toRingOps_1543_ = lean_ctor_get(v_fo_1509_, 0);
v_toSemiringOps_1544_ = lean_ctor_get(v_toRingOps_1543_, 0);
v_x_1545_ = lean_ctor_get(v_x_1511_, 0);
lean_inc(v_x_1545_);
v_y_1546_ = lean_ctor_get(v_x_1511_, 1);
lean_inc(v_y_1546_);
lean_dec_ref_known(v_x_1511_, 2);
v_mul_1547_ = lean_ctor_get(v_toSemiringOps_1544_, 4);
lean_inc(v_mul_1547_);
lean_inc_ref(v_e_1510_);
lean_inc_ref(v_fo_1509_);
v___x_1548_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_x_1545_);
v___x_1549_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1509_, v_e_1510_, v_y_1546_);
v___x_1550_ = lean_apply_2(v_mul_1547_, v___x_1548_, v___x_1549_);
return v___x_1550_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr(lean_object* v_F_1551_, lean_object* v_K_1552_, lean_object* v_fo_1553_, lean_object* v_e_1554_, lean_object* v_x_1555_){
_start:
{
lean_object* v___x_1556_; 
v___x_1556_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalExpr___redArg(v_fo_1553_, v_e_1554_, v_x_1555_);
return v___x_1556_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg(lean_object* v_fo_1557_, lean_object* v_e_1558_, lean_object* v_n_1559_, lean_object* v_values_1560_, lean_object* v_x_1561_){
_start:
{
switch(lean_obj_tag(v_x_1561_))
{
case 0:
{
lean_object* v_v_1562_; lean_object* v_evalVar_1563_; lean_object* v___x_1564_; 
lean_dec_ref(v_fo_1557_);
v_v_1562_ = lean_ctor_get(v_x_1561_, 0);
lean_inc_ref(v_v_1562_);
lean_dec_ref_known(v_x_1561_, 1);
v_evalVar_1563_ = lean_ctor_get(v_e_1558_, 1);
lean_inc(v_evalVar_1563_);
lean_dec_ref(v_e_1558_);
v___x_1564_ = lean_apply_1(v_evalVar_1563_, v_v_1562_);
return v___x_1564_;
}
case 1:
{
lean_object* v_evalIsFirstRow_1565_; 
lean_dec_ref(v_fo_1557_);
v_evalIsFirstRow_1565_ = lean_ctor_get(v_e_1558_, 2);
lean_inc(v_evalIsFirstRow_1565_);
lean_dec_ref(v_e_1558_);
return v_evalIsFirstRow_1565_;
}
case 2:
{
lean_object* v_evalIsLastRow_1566_; 
lean_dec_ref(v_fo_1557_);
v_evalIsLastRow_1566_ = lean_ctor_get(v_e_1558_, 3);
lean_inc(v_evalIsLastRow_1566_);
lean_dec_ref(v_e_1558_);
return v_evalIsLastRow_1566_;
}
case 3:
{
lean_object* v_evalIsTransition_1567_; 
lean_dec_ref(v_fo_1557_);
v_evalIsTransition_1567_ = lean_ctor_get(v_e_1558_, 4);
lean_inc(v_evalIsTransition_1567_);
lean_dec_ref(v_e_1558_);
return v_evalIsTransition_1567_;
}
case 4:
{
lean_object* v_c_1568_; lean_object* v_evalConst_1569_; lean_object* v___x_1570_; 
lean_dec_ref(v_fo_1557_);
v_c_1568_ = lean_ctor_get(v_x_1561_, 0);
lean_inc(v_c_1568_);
lean_dec_ref_known(v_x_1561_, 1);
v_evalConst_1569_ = lean_ctor_get(v_e_1558_, 0);
lean_inc(v_evalConst_1569_);
lean_dec_ref(v_e_1558_);
v___x_1570_ = lean_apply_1(v_evalConst_1569_, v_c_1568_);
return v___x_1570_;
}
case 5:
{
lean_object* v_toRingOps_1571_; lean_object* v_toSemiringOps_1572_; lean_object* v_leftIdx_1573_; lean_object* v_rightIdx_1574_; lean_object* v_zero_1575_; lean_object* v_add_1576_; lean_object* v___y_1578_; uint8_t v___x_1583_; 
lean_dec_ref(v_e_1558_);
v_toRingOps_1571_ = lean_ctor_get(v_fo_1557_, 0);
lean_inc_ref(v_toRingOps_1571_);
lean_dec_ref(v_fo_1557_);
v_toSemiringOps_1572_ = lean_ctor_get(v_toRingOps_1571_, 0);
lean_inc_ref(v_toSemiringOps_1572_);
lean_dec_ref(v_toRingOps_1571_);
v_leftIdx_1573_ = lean_ctor_get(v_x_1561_, 0);
lean_inc(v_leftIdx_1573_);
v_rightIdx_1574_ = lean_ctor_get(v_x_1561_, 1);
lean_inc(v_rightIdx_1574_);
lean_dec_ref_known(v_x_1561_, 3);
v_zero_1575_ = lean_ctor_get(v_toSemiringOps_1572_, 0);
lean_inc(v_zero_1575_);
v_add_1576_ = lean_ctor_get(v_toSemiringOps_1572_, 3);
lean_inc(v_add_1576_);
lean_dec_ref(v_toSemiringOps_1572_);
v___x_1583_ = lean_nat_dec_lt(v_leftIdx_1573_, v_n_1559_);
if (v___x_1583_ == 0)
{
lean_dec(v_leftIdx_1573_);
lean_inc(v_zero_1575_);
v___y_1578_ = v_zero_1575_;
goto v___jp_1577_;
}
else
{
lean_object* v___x_1584_; 
v___x_1584_ = lean_array_fget_borrowed(v_values_1560_, v_leftIdx_1573_);
lean_dec(v_leftIdx_1573_);
lean_inc(v___x_1584_);
v___y_1578_ = v___x_1584_;
goto v___jp_1577_;
}
v___jp_1577_:
{
uint8_t v___x_1579_; 
v___x_1579_ = lean_nat_dec_lt(v_rightIdx_1574_, v_n_1559_);
if (v___x_1579_ == 0)
{
lean_object* v___x_1580_; 
lean_dec(v_rightIdx_1574_);
v___x_1580_ = lean_apply_2(v_add_1576_, v___y_1578_, v_zero_1575_);
return v___x_1580_;
}
else
{
lean_object* v___x_1581_; lean_object* v___x_1582_; 
lean_dec(v_zero_1575_);
v___x_1581_ = lean_array_fget_borrowed(v_values_1560_, v_rightIdx_1574_);
lean_dec(v_rightIdx_1574_);
lean_inc(v___x_1581_);
v___x_1582_ = lean_apply_2(v_add_1576_, v___y_1578_, v___x_1581_);
return v___x_1582_;
}
}
}
case 6:
{
lean_object* v_toRingOps_1585_; lean_object* v_leftIdx_1586_; lean_object* v_rightIdx_1587_; lean_object* v_toSemiringOps_1588_; lean_object* v_sub_1589_; lean_object* v___y_1591_; uint8_t v___x_1597_; 
lean_dec_ref(v_e_1558_);
v_toRingOps_1585_ = lean_ctor_get(v_fo_1557_, 0);
lean_inc_ref(v_toRingOps_1585_);
lean_dec_ref(v_fo_1557_);
v_leftIdx_1586_ = lean_ctor_get(v_x_1561_, 0);
lean_inc(v_leftIdx_1586_);
v_rightIdx_1587_ = lean_ctor_get(v_x_1561_, 1);
lean_inc(v_rightIdx_1587_);
lean_dec_ref_known(v_x_1561_, 3);
v_toSemiringOps_1588_ = lean_ctor_get(v_toRingOps_1585_, 0);
lean_inc_ref(v_toSemiringOps_1588_);
v_sub_1589_ = lean_ctor_get(v_toRingOps_1585_, 1);
lean_inc(v_sub_1589_);
lean_dec_ref(v_toRingOps_1585_);
v___x_1597_ = lean_nat_dec_lt(v_leftIdx_1586_, v_n_1559_);
if (v___x_1597_ == 0)
{
lean_object* v_zero_1598_; 
lean_dec(v_leftIdx_1586_);
v_zero_1598_ = lean_ctor_get(v_toSemiringOps_1588_, 0);
lean_inc(v_zero_1598_);
v___y_1591_ = v_zero_1598_;
goto v___jp_1590_;
}
else
{
lean_object* v___x_1599_; 
v___x_1599_ = lean_array_fget_borrowed(v_values_1560_, v_leftIdx_1586_);
lean_dec(v_leftIdx_1586_);
lean_inc(v___x_1599_);
v___y_1591_ = v___x_1599_;
goto v___jp_1590_;
}
v___jp_1590_:
{
uint8_t v___x_1592_; 
v___x_1592_ = lean_nat_dec_lt(v_rightIdx_1587_, v_n_1559_);
if (v___x_1592_ == 0)
{
lean_object* v_zero_1593_; lean_object* v___x_1594_; 
lean_dec(v_rightIdx_1587_);
v_zero_1593_ = lean_ctor_get(v_toSemiringOps_1588_, 0);
lean_inc(v_zero_1593_);
lean_dec_ref(v_toSemiringOps_1588_);
v___x_1594_ = lean_apply_2(v_sub_1589_, v___y_1591_, v_zero_1593_);
return v___x_1594_;
}
else
{
lean_object* v___x_1595_; lean_object* v___x_1596_; 
lean_dec_ref(v_toSemiringOps_1588_);
v___x_1595_ = lean_array_fget_borrowed(v_values_1560_, v_rightIdx_1587_);
lean_dec(v_rightIdx_1587_);
lean_inc(v___x_1595_);
v___x_1596_ = lean_apply_2(v_sub_1589_, v___y_1591_, v___x_1595_);
return v___x_1596_;
}
}
}
case 7:
{
lean_object* v_toRingOps_1600_; lean_object* v_toSemiringOps_1601_; lean_object* v_idx_1602_; lean_object* v_sub_1603_; lean_object* v_zero_1604_; uint8_t v___x_1605_; 
lean_dec_ref(v_e_1558_);
v_toRingOps_1600_ = lean_ctor_get(v_fo_1557_, 0);
lean_inc_ref(v_toRingOps_1600_);
lean_dec_ref(v_fo_1557_);
v_toSemiringOps_1601_ = lean_ctor_get(v_toRingOps_1600_, 0);
lean_inc_ref(v_toSemiringOps_1601_);
v_idx_1602_ = lean_ctor_get(v_x_1561_, 0);
lean_inc(v_idx_1602_);
lean_dec_ref_known(v_x_1561_, 2);
v_sub_1603_ = lean_ctor_get(v_toRingOps_1600_, 1);
lean_inc(v_sub_1603_);
lean_dec_ref(v_toRingOps_1600_);
v_zero_1604_ = lean_ctor_get(v_toSemiringOps_1601_, 0);
lean_inc(v_zero_1604_);
lean_dec_ref(v_toSemiringOps_1601_);
v___x_1605_ = lean_nat_dec_lt(v_idx_1602_, v_n_1559_);
if (v___x_1605_ == 0)
{
lean_object* v___x_1606_; 
lean_dec(v_idx_1602_);
lean_inc(v_zero_1604_);
v___x_1606_ = lean_apply_2(v_sub_1603_, v_zero_1604_, v_zero_1604_);
return v___x_1606_;
}
else
{
lean_object* v___x_1607_; lean_object* v___x_1608_; 
v___x_1607_ = lean_array_fget_borrowed(v_values_1560_, v_idx_1602_);
lean_dec(v_idx_1602_);
lean_inc(v___x_1607_);
v___x_1608_ = lean_apply_2(v_sub_1603_, v_zero_1604_, v___x_1607_);
return v___x_1608_;
}
}
default: 
{
lean_object* v_toRingOps_1609_; lean_object* v_toSemiringOps_1610_; lean_object* v_leftIdx_1611_; lean_object* v_rightIdx_1612_; lean_object* v_zero_1613_; lean_object* v_mul_1614_; lean_object* v___y_1616_; uint8_t v___x_1621_; 
lean_dec_ref(v_e_1558_);
v_toRingOps_1609_ = lean_ctor_get(v_fo_1557_, 0);
lean_inc_ref(v_toRingOps_1609_);
lean_dec_ref(v_fo_1557_);
v_toSemiringOps_1610_ = lean_ctor_get(v_toRingOps_1609_, 0);
lean_inc_ref(v_toSemiringOps_1610_);
lean_dec_ref(v_toRingOps_1609_);
v_leftIdx_1611_ = lean_ctor_get(v_x_1561_, 0);
lean_inc(v_leftIdx_1611_);
v_rightIdx_1612_ = lean_ctor_get(v_x_1561_, 1);
lean_inc(v_rightIdx_1612_);
lean_dec_ref_known(v_x_1561_, 3);
v_zero_1613_ = lean_ctor_get(v_toSemiringOps_1610_, 0);
lean_inc(v_zero_1613_);
v_mul_1614_ = lean_ctor_get(v_toSemiringOps_1610_, 4);
lean_inc(v_mul_1614_);
lean_dec_ref(v_toSemiringOps_1610_);
v___x_1621_ = lean_nat_dec_lt(v_leftIdx_1611_, v_n_1559_);
if (v___x_1621_ == 0)
{
lean_dec(v_leftIdx_1611_);
lean_inc(v_zero_1613_);
v___y_1616_ = v_zero_1613_;
goto v___jp_1615_;
}
else
{
lean_object* v___x_1622_; 
v___x_1622_ = lean_array_fget_borrowed(v_values_1560_, v_leftIdx_1611_);
lean_dec(v_leftIdx_1611_);
lean_inc(v___x_1622_);
v___y_1616_ = v___x_1622_;
goto v___jp_1615_;
}
v___jp_1615_:
{
uint8_t v___x_1617_; 
v___x_1617_ = lean_nat_dec_lt(v_rightIdx_1612_, v_n_1559_);
if (v___x_1617_ == 0)
{
lean_object* v___x_1618_; 
lean_dec(v_rightIdx_1612_);
v___x_1618_ = lean_apply_2(v_mul_1614_, v___y_1616_, v_zero_1613_);
return v___x_1618_;
}
else
{
lean_object* v___x_1619_; lean_object* v___x_1620_; 
lean_dec(v_zero_1613_);
v___x_1619_ = lean_array_fget_borrowed(v_values_1560_, v_rightIdx_1612_);
lean_dec(v_rightIdx_1612_);
lean_inc(v___x_1619_);
v___x_1620_ = lean_apply_2(v_mul_1614_, v___y_1616_, v___x_1619_);
return v___x_1620_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg___boxed(lean_object* v_fo_1623_, lean_object* v_e_1624_, lean_object* v_n_1625_, lean_object* v_values_1626_, lean_object* v_x_1627_){
_start:
{
lean_object* v_res_1628_; 
v_res_1628_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg(v_fo_1623_, v_e_1624_, v_n_1625_, v_values_1626_, v_x_1627_);
lean_dec_ref(v_values_1626_);
lean_dec(v_n_1625_);
return v_res_1628_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode(lean_object* v_F_1629_, lean_object* v_K_1630_, lean_object* v_fo_1631_, lean_object* v_e_1632_, lean_object* v_n_1633_, lean_object* v_values_1634_, lean_object* v_x_1635_){
_start:
{
lean_object* v___x_1636_; 
v___x_1636_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg(v_fo_1631_, v_e_1632_, v_n_1633_, v_values_1634_, v_x_1635_);
return v___x_1636_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___boxed(lean_object* v_F_1637_, lean_object* v_K_1638_, lean_object* v_fo_1639_, lean_object* v_e_1640_, lean_object* v_n_1641_, lean_object* v_values_1642_, lean_object* v_x_1643_){
_start:
{
lean_object* v_res_1644_; 
v_res_1644_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode(v_F_1637_, v_K_1638_, v_fo_1639_, v_e_1640_, v_n_1641_, v_values_1642_, v_x_1643_);
lean_dec_ref(v_values_1642_);
lean_dec(v_n_1641_);
return v_res_1644_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes_spec__0___redArg(lean_object* v_fo_1645_, lean_object* v_e_1646_, lean_object* v_x_1647_, lean_object* v_x_1648_){
_start:
{
if (lean_obj_tag(v_x_1648_) == 0)
{
lean_dec_ref(v_e_1646_);
lean_dec_ref(v_fo_1645_);
return v_x_1647_;
}
else
{
lean_object* v_head_1649_; lean_object* v_tail_1650_; lean_object* v_fst_1651_; lean_object* v_snd_1652_; lean_object* v___x_1654_; uint8_t v_isShared_1655_; uint8_t v_isSharedCheck_1664_; 
v_head_1649_ = lean_ctor_get(v_x_1648_, 0);
lean_inc(v_head_1649_);
v_tail_1650_ = lean_ctor_get(v_x_1648_, 1);
lean_inc(v_tail_1650_);
lean_dec_ref_known(v_x_1648_, 2);
v_fst_1651_ = lean_ctor_get(v_x_1647_, 0);
v_snd_1652_ = lean_ctor_get(v_x_1647_, 1);
v_isSharedCheck_1664_ = !lean_is_exclusive(v_x_1647_);
if (v_isSharedCheck_1664_ == 0)
{
v___x_1654_ = v_x_1647_;
v_isShared_1655_ = v_isSharedCheck_1664_;
goto v_resetjp_1653_;
}
else
{
lean_inc(v_snd_1652_);
lean_inc(v_fst_1651_);
lean_dec(v_x_1647_);
v___x_1654_ = lean_box(0);
v_isShared_1655_ = v_isSharedCheck_1664_;
goto v_resetjp_1653_;
}
v_resetjp_1653_:
{
lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1661_; 
v___x_1656_ = lean_unsigned_to_nat(1u);
v___x_1657_ = lean_nat_add(v_fst_1651_, v___x_1656_);
lean_inc_ref(v_e_1646_);
lean_inc_ref(v_fo_1645_);
v___x_1658_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNode___redArg(v_fo_1645_, v_e_1646_, v_fst_1651_, v_snd_1652_, v_head_1649_);
lean_dec(v_fst_1651_);
v___x_1659_ = lean_array_push(v_snd_1652_, v___x_1658_);
if (v_isShared_1655_ == 0)
{
lean_ctor_set(v___x_1654_, 1, v___x_1659_);
lean_ctor_set(v___x_1654_, 0, v___x_1657_);
v___x_1661_ = v___x_1654_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1663_; 
v_reuseFailAlloc_1663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1663_, 0, v___x_1657_);
lean_ctor_set(v_reuseFailAlloc_1663_, 1, v___x_1659_);
v___x_1661_ = v_reuseFailAlloc_1663_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
v_x_1647_ = v___x_1661_;
v_x_1648_ = v_tail_1650_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes___redArg(lean_object* v_fo_1665_, lean_object* v_e_1666_, lean_object* v_nodes_1667_){
_start:
{
lean_object* v_toRingOps_1668_; lean_object* v_toSemiringOps_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1682_; 
v_toRingOps_1668_ = lean_ctor_get(v_fo_1665_, 0);
lean_inc_ref(v_toRingOps_1668_);
v_toSemiringOps_1669_ = lean_ctor_get(v_toRingOps_1668_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v_toRingOps_1668_);
if (v_isSharedCheck_1682_ == 0)
{
lean_object* v_unused_1683_; 
v_unused_1683_ = lean_ctor_get(v_toRingOps_1668_, 1);
lean_dec(v_unused_1683_);
v___x_1671_ = v_toRingOps_1668_;
v_isShared_1672_ = v_isSharedCheck_1682_;
goto v_resetjp_1670_;
}
else
{
lean_inc(v_toSemiringOps_1669_);
lean_dec(v_toRingOps_1668_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1682_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v_zero_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1677_; 
v_zero_1673_ = lean_ctor_get(v_toSemiringOps_1669_, 0);
lean_inc(v_zero_1673_);
lean_dec_ref(v_toSemiringOps_1669_);
v___x_1674_ = lean_unsigned_to_nat(0u);
v___x_1675_ = lean_mk_array(v___x_1674_, v_zero_1673_);
if (v_isShared_1672_ == 0)
{
lean_ctor_set(v___x_1671_, 1, v___x_1675_);
lean_ctor_set(v___x_1671_, 0, v___x_1674_);
v___x_1677_ = v___x_1671_;
goto v_reusejp_1676_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v___x_1674_);
lean_ctor_set(v_reuseFailAlloc_1681_, 1, v___x_1675_);
v___x_1677_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1676_;
}
v_reusejp_1676_:
{
lean_object* v___x_1678_; lean_object* v_snd_1679_; lean_object* v___x_1680_; 
v___x_1678_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes_spec__0___redArg(v_fo_1665_, v_e_1666_, v___x_1677_, v_nodes_1667_);
v_snd_1679_ = lean_ctor_get(v___x_1678_, 1);
lean_inc(v_snd_1679_);
lean_dec_ref(v___x_1678_);
v___x_1680_ = lean_array_to_list(v_snd_1679_);
return v___x_1680_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes(lean_object* v_F_1684_, lean_object* v_K_1685_, lean_object* v_fo_1686_, lean_object* v_e_1687_, lean_object* v_nodes_1688_){
_start:
{
lean_object* v___x_1689_; 
v___x_1689_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes___redArg(v_fo_1686_, v_e_1687_, v_nodes_1688_);
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes_spec__0(lean_object* v_K_1690_, lean_object* v_F_1691_, lean_object* v_fo_1692_, lean_object* v_e_1693_, lean_object* v_x_1694_, lean_object* v_x_1695_){
_start:
{
lean_object* v___x_1696_; 
v___x_1696_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes_spec__0___redArg(v_fo_1692_, v_e_1693_, v_x_1694_, v_x_1695_);
return v___x_1696_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg(lean_object* v_dag_1697_){
_start:
{
lean_object* v_constraintIdx_1698_; lean_object* v___x_1699_; 
v_constraintIdx_1698_ = lean_ctor_get(v_dag_1697_, 1);
v___x_1699_ = l_List_lengthTR___redArg(v_constraintIdx_1698_);
return v___x_1699_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg___boxed(lean_object* v_dag_1700_){
_start:
{
lean_object* v_res_1701_; 
v_res_1701_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg(v_dag_1700_);
lean_dec_ref(v_dag_1700_);
return v_res_1701_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount(lean_object* v_F_1702_, lean_object* v_dag_1703_){
_start:
{
lean_object* v___x_1704_; 
v___x_1704_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg(v_dag_1703_);
return v___x_1704_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___boxed(lean_object* v_F_1705_, lean_object* v_dag_1706_){
_start:
{
lean_object* v_res_1707_; 
v_res_1707_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount(v_F_1705_, v_dag_1706_);
lean_dec_ref(v_dag_1706_);
return v_res_1707_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes___redArg(lean_object* v_fo_1708_, lean_object* v_dag_1709_, lean_object* v_e_1710_){
_start:
{
lean_object* v_nodes_1711_; lean_object* v___x_1712_; 
v_nodes_1711_ = lean_ctor_get(v_dag_1709_, 0);
lean_inc(v_nodes_1711_);
lean_dec_ref(v_dag_1709_);
v___x_1712_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicEvaluator_evalNodes___redArg(v_fo_1708_, v_e_1710_, v_nodes_1711_);
return v___x_1712_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes(lean_object* v_F_1713_, lean_object* v_K_1714_, lean_object* v_fo_1715_, lean_object* v_dag_1716_, lean_object* v_e_1717_){
_start:
{
lean_object* v___x_1718_; 
v___x_1718_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes___redArg(v_fo_1715_, v_dag_1716_, v_e_1717_);
return v___x_1718_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg(lean_object* v_values_1719_, lean_object* v_fo_1720_, lean_object* v_a_1721_, lean_object* v_a_1722_){
_start:
{
if (lean_obj_tag(v_a_1721_) == 0)
{
lean_object* v___x_1723_; 
lean_dec_ref(v_fo_1720_);
v___x_1723_ = l_List_reverse___redArg(v_a_1722_);
return v___x_1723_;
}
else
{
lean_object* v_head_1724_; lean_object* v_tail_1725_; lean_object* v___x_1727_; uint8_t v_isShared_1728_; uint8_t v_isSharedCheck_1740_; 
v_head_1724_ = lean_ctor_get(v_a_1721_, 0);
v_tail_1725_ = lean_ctor_get(v_a_1721_, 1);
v_isSharedCheck_1740_ = !lean_is_exclusive(v_a_1721_);
if (v_isSharedCheck_1740_ == 0)
{
v___x_1727_ = v_a_1721_;
v_isShared_1728_ = v_isSharedCheck_1740_;
goto v_resetjp_1726_;
}
else
{
lean_inc(v_tail_1725_);
lean_inc(v_head_1724_);
lean_dec(v_a_1721_);
v___x_1727_ = lean_box(0);
v_isShared_1728_ = v_isSharedCheck_1740_;
goto v_resetjp_1726_;
}
v_resetjp_1726_:
{
lean_object* v___y_1730_; lean_object* v___x_1735_; 
v___x_1735_ = l_List_get_x3fInternal___redArg(v_values_1719_, v_head_1724_);
if (lean_obj_tag(v___x_1735_) == 0)
{
lean_object* v_toRingOps_1736_; lean_object* v_toSemiringOps_1737_; lean_object* v_zero_1738_; 
v_toRingOps_1736_ = lean_ctor_get(v_fo_1720_, 0);
v_toSemiringOps_1737_ = lean_ctor_get(v_toRingOps_1736_, 0);
v_zero_1738_ = lean_ctor_get(v_toSemiringOps_1737_, 0);
lean_inc(v_zero_1738_);
v___y_1730_ = v_zero_1738_;
goto v___jp_1729_;
}
else
{
lean_object* v_val_1739_; 
v_val_1739_ = lean_ctor_get(v___x_1735_, 0);
lean_inc(v_val_1739_);
lean_dec_ref_known(v___x_1735_, 1);
v___y_1730_ = v_val_1739_;
goto v___jp_1729_;
}
v___jp_1729_:
{
lean_object* v___x_1732_; 
if (v_isShared_1728_ == 0)
{
lean_ctor_set(v___x_1727_, 1, v_a_1722_);
lean_ctor_set(v___x_1727_, 0, v___y_1730_);
v___x_1732_ = v___x_1727_;
goto v_reusejp_1731_;
}
else
{
lean_object* v_reuseFailAlloc_1734_; 
v_reuseFailAlloc_1734_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1734_, 0, v___y_1730_);
lean_ctor_set(v_reuseFailAlloc_1734_, 1, v_a_1722_);
v___x_1732_ = v_reuseFailAlloc_1734_;
goto v_reusejp_1731_;
}
v_reusejp_1731_:
{
v_a_1721_ = v_tail_1725_;
v_a_1722_ = v___x_1732_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg___boxed(lean_object* v_values_1741_, lean_object* v_fo_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_){
_start:
{
lean_object* v_res_1745_; 
v_res_1745_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg(v_values_1741_, v_fo_1742_, v_a_1743_, v_a_1744_);
lean_dec(v_values_1741_);
return v_res_1745_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints___redArg(lean_object* v_fo_1746_, lean_object* v_dag_1747_, lean_object* v_e_1748_){
_start:
{
lean_object* v_constraintIdx_1749_; lean_object* v_values_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; 
v_constraintIdx_1749_ = lean_ctor_get(v_dag_1747_, 1);
lean_inc(v_constraintIdx_1749_);
lean_inc_ref(v_fo_1746_);
v_values_1750_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes___redArg(v_fo_1746_, v_dag_1747_, v_e_1748_);
v___x_1751_ = lean_box(0);
v___x_1752_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg(v_values_1750_, v_fo_1746_, v_constraintIdx_1749_, v___x_1751_);
lean_dec(v_values_1750_);
return v___x_1752_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints(lean_object* v_F_1753_, lean_object* v_K_1754_, lean_object* v_fo_1755_, lean_object* v_dag_1756_, lean_object* v_e_1757_){
_start:
{
lean_object* v___x_1758_; 
v___x_1758_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints___redArg(v_fo_1755_, v_dag_1756_, v_e_1757_);
return v___x_1758_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0(lean_object* v_K_1759_, lean_object* v_values_1760_, lean_object* v_fo_1761_, lean_object* v_a_1762_, lean_object* v_a_1763_){
_start:
{
lean_object* v___x_1764_; 
v___x_1764_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg(v_values_1760_, v_fo_1761_, v_a_1762_, v_a_1763_);
return v___x_1764_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___boxed(lean_object* v_K_1765_, lean_object* v_values_1766_, lean_object* v_fo_1767_, lean_object* v_a_1768_, lean_object* v_a_1769_){
_start:
{
lean_object* v_res_1770_; 
v_res_1770_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0(v_K_1765_, v_values_1766_, v_fo_1767_, v_a_1768_, v_a_1769_);
lean_dec(v_values_1766_);
return v_res_1770_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction_decEq(lean_object* v_x_1771_, lean_object* v_x_1772_){
_start:
{
lean_object* v_message_1773_; lean_object* v_count_1774_; lean_object* v_busIndex_1775_; lean_object* v_countWeight_1776_; lean_object* v_message_1777_; lean_object* v_count_1778_; lean_object* v_busIndex_1779_; lean_object* v_countWeight_1780_; lean_object* v___x_1781_; uint8_t v___x_1782_; 
v_message_1773_ = lean_ctor_get(v_x_1771_, 0);
lean_inc(v_message_1773_);
v_count_1774_ = lean_ctor_get(v_x_1771_, 1);
lean_inc(v_count_1774_);
v_busIndex_1775_ = lean_ctor_get(v_x_1771_, 2);
lean_inc(v_busIndex_1775_);
v_countWeight_1776_ = lean_ctor_get(v_x_1771_, 3);
lean_inc(v_countWeight_1776_);
lean_dec_ref(v_x_1771_);
v_message_1777_ = lean_ctor_get(v_x_1772_, 0);
lean_inc(v_message_1777_);
v_count_1778_ = lean_ctor_get(v_x_1772_, 1);
lean_inc(v_count_1778_);
v_busIndex_1779_ = lean_ctor_get(v_x_1772_, 2);
lean_inc(v_busIndex_1779_);
v_countWeight_1780_ = lean_ctor_get(v_x_1772_, 3);
lean_inc(v_countWeight_1780_);
lean_dec_ref(v_x_1772_);
v___x_1781_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_1782_ = l_instDecidableEqList___redArg(v___x_1781_, v_message_1773_, v_message_1777_);
if (v___x_1782_ == 0)
{
lean_dec(v_countWeight_1780_);
lean_dec(v_busIndex_1779_);
lean_dec(v_count_1778_);
lean_dec(v_countWeight_1776_);
lean_dec(v_busIndex_1775_);
lean_dec(v_count_1774_);
return v___x_1782_;
}
else
{
uint8_t v___x_1783_; 
v___x_1783_ = lean_nat_dec_eq(v_count_1774_, v_count_1778_);
lean_dec(v_count_1778_);
lean_dec(v_count_1774_);
if (v___x_1783_ == 0)
{
lean_dec(v_countWeight_1780_);
lean_dec(v_busIndex_1779_);
lean_dec(v_countWeight_1776_);
lean_dec(v_busIndex_1775_);
return v___x_1783_;
}
else
{
uint8_t v___x_1784_; 
v___x_1784_ = lean_nat_dec_eq(v_busIndex_1775_, v_busIndex_1779_);
lean_dec(v_busIndex_1779_);
lean_dec(v_busIndex_1775_);
if (v___x_1784_ == 0)
{
lean_dec(v_countWeight_1780_);
lean_dec(v_countWeight_1776_);
return v___x_1784_;
}
else
{
uint8_t v___x_1785_; 
v___x_1785_ = lean_nat_dec_eq(v_countWeight_1776_, v_countWeight_1780_);
lean_dec(v_countWeight_1780_);
lean_dec(v_countWeight_1776_);
return v___x_1785_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction_decEq___boxed(lean_object* v_x_1786_, lean_object* v_x_1787_){
_start:
{
uint8_t v_res_1788_; lean_object* v_r_1789_; 
v_res_1788_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction_decEq(v_x_1786_, v_x_1787_);
v_r_1789_ = lean_box(v_res_1788_);
return v_r_1789_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction(lean_object* v_x_1790_, lean_object* v_x_1791_){
_start:
{
uint8_t v___x_1792_; 
v___x_1792_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction_decEq(v_x_1790_, v_x_1791_);
return v___x_1792_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction___boxed(lean_object* v_x_1793_, lean_object* v_x_1794_){
_start:
{
uint8_t v_res_1795_; lean_object* v_r_1796_; 
v_res_1795_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction(v_x_1793_, v_x_1794_);
v_r_1796_ = lean_box(v_res_1795_);
return v_r_1796_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__8(void){
_start:
{
lean_object* v___x_1812_; lean_object* v___x_1813_; 
v___x_1812_ = lean_unsigned_to_nat(12u);
v___x_1813_ = lean_nat_to_int(v___x_1812_);
return v___x_1813_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg(lean_object* v_x_1817_){
_start:
{
lean_object* v_message_1818_; lean_object* v_count_1819_; lean_object* v_busIndex_1820_; lean_object* v_countWeight_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; uint8_t v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; 
v_message_1818_ = lean_ctor_get(v_x_1817_, 0);
lean_inc(v_message_1818_);
v_count_1819_ = lean_ctor_get(v_x_1817_, 1);
lean_inc(v_count_1819_);
v_busIndex_1820_ = lean_ctor_get(v_x_1817_, 2);
lean_inc(v_busIndex_1820_);
v_countWeight_1821_ = lean_ctor_get(v_x_1817_, 3);
lean_inc(v_countWeight_1821_);
lean_dec_ref(v_x_1817_);
v___x_1822_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_1823_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__3));
v___x_1824_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10);
v___x_1825_ = lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg(v_message_1818_);
v___x_1826_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1826_, 0, v___x_1824_);
lean_ctor_set(v___x_1826_, 1, v___x_1825_);
v___x_1827_ = 0;
v___x_1828_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1828_, 0, v___x_1826_);
lean_ctor_set_uint8(v___x_1828_, sizeof(void*)*1, v___x_1827_);
v___x_1829_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1829_, 0, v___x_1823_);
lean_ctor_set(v___x_1829_, 1, v___x_1828_);
v___x_1830_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_1831_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1831_, 0, v___x_1829_);
lean_ctor_set(v___x_1831_, 1, v___x_1830_);
v___x_1832_ = lean_box(1);
v___x_1833_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1833_, 0, v___x_1831_);
lean_ctor_set(v___x_1833_, 1, v___x_1832_);
v___x_1834_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__5));
v___x_1835_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1835_, 0, v___x_1833_);
lean_ctor_set(v___x_1835_, 1, v___x_1834_);
v___x_1836_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1835_);
lean_ctor_set(v___x_1836_, 1, v___x_1822_);
v___x_1837_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4);
v___x_1838_ = l_Nat_reprFast(v_count_1819_);
v___x_1839_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1839_, 0, v___x_1838_);
v___x_1840_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1840_, 0, v___x_1837_);
lean_ctor_set(v___x_1840_, 1, v___x_1839_);
v___x_1841_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1841_, 0, v___x_1840_);
lean_ctor_set_uint8(v___x_1841_, sizeof(void*)*1, v___x_1827_);
v___x_1842_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1842_, 0, v___x_1836_);
lean_ctor_set(v___x_1842_, 1, v___x_1841_);
v___x_1843_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1843_, 0, v___x_1842_);
lean_ctor_set(v___x_1843_, 1, v___x_1830_);
v___x_1844_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1844_, 0, v___x_1843_);
lean_ctor_set(v___x_1844_, 1, v___x_1832_);
v___x_1845_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__7));
v___x_1846_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1846_, 0, v___x_1844_);
lean_ctor_set(v___x_1846_, 1, v___x_1845_);
v___x_1847_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1847_, 0, v___x_1846_);
lean_ctor_set(v___x_1847_, 1, v___x_1822_);
v___x_1848_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__8, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__8_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__8);
v___x_1849_ = l_Nat_reprFast(v_busIndex_1820_);
v___x_1850_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1850_, 0, v___x_1849_);
v___x_1851_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1851_, 0, v___x_1848_);
lean_ctor_set(v___x_1851_, 1, v___x_1850_);
v___x_1852_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1852_, 0, v___x_1851_);
lean_ctor_set_uint8(v___x_1852_, sizeof(void*)*1, v___x_1827_);
v___x_1853_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1853_, 0, v___x_1847_);
lean_ctor_set(v___x_1853_, 1, v___x_1852_);
v___x_1854_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1854_, 0, v___x_1853_);
lean_ctor_set(v___x_1854_, 1, v___x_1830_);
v___x_1855_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1855_, 0, v___x_1854_);
lean_ctor_set(v___x_1855_, 1, v___x_1832_);
v___x_1856_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg___closed__10));
v___x_1857_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1857_, 0, v___x_1855_);
lean_ctor_set(v___x_1857_, 1, v___x_1856_);
v___x_1858_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1858_, 0, v___x_1857_);
lean_ctor_set(v___x_1858_, 1, v___x_1822_);
v___x_1859_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10);
v___x_1860_ = l_Nat_reprFast(v_countWeight_1821_);
v___x_1861_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1861_, 0, v___x_1860_);
v___x_1862_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1862_, 0, v___x_1859_);
lean_ctor_set(v___x_1862_, 1, v___x_1861_);
v___x_1863_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1863_, 0, v___x_1862_);
lean_ctor_set_uint8(v___x_1863_, sizeof(void*)*1, v___x_1827_);
v___x_1864_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1864_, 0, v___x_1858_);
lean_ctor_set(v___x_1864_, 1, v___x_1863_);
v___x_1865_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__16);
v___x_1866_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_1867_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1867_, 0, v___x_1866_);
lean_ctor_set(v___x_1867_, 1, v___x_1864_);
v___x_1868_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_1869_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1869_, 0, v___x_1867_);
lean_ctor_set(v___x_1869_, 1, v___x_1868_);
v___x_1870_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1870_, 0, v___x_1865_);
lean_ctor_set(v___x_1870_, 1, v___x_1869_);
v___x_1871_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1871_, 0, v___x_1870_);
lean_ctor_set_uint8(v___x_1871_, sizeof(void*)*1, v___x_1827_);
return v___x_1871_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr(lean_object* v_x_1872_, lean_object* v_prec_1873_){
_start:
{
lean_object* v___x_1874_; 
v___x_1874_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___redArg(v_x_1872_);
return v___x_1874_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr___boxed(lean_object* v_x_1875_, lean_object* v_prec_1876_){
_start:
{
lean_object* v_res_1877_; 
v_res_1877_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction_repr(v_x_1875_, v_prec_1876_);
lean_dec(v_prec_1876_);
return v_res_1877_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg(lean_object* v_fo_1880_, lean_object* v_interaction_1881_, lean_object* v_values_1882_){
_start:
{
lean_object* v_message_1883_; lean_object* v_count_1884_; lean_object* v_busIndex_1885_; lean_object* v_countWeight_1886_; lean_object* v___x_1888_; uint8_t v_isShared_1889_; uint8_t v_isSharedCheck_1903_; 
v_message_1883_ = lean_ctor_get(v_interaction_1881_, 0);
v_count_1884_ = lean_ctor_get(v_interaction_1881_, 1);
v_busIndex_1885_ = lean_ctor_get(v_interaction_1881_, 2);
v_countWeight_1886_ = lean_ctor_get(v_interaction_1881_, 3);
v_isSharedCheck_1903_ = !lean_is_exclusive(v_interaction_1881_);
if (v_isSharedCheck_1903_ == 0)
{
v___x_1888_ = v_interaction_1881_;
v_isShared_1889_ = v_isSharedCheck_1903_;
goto v_resetjp_1887_;
}
else
{
lean_inc(v_countWeight_1886_);
lean_inc(v_busIndex_1885_);
lean_inc(v_count_1884_);
lean_inc(v_message_1883_);
lean_dec(v_interaction_1881_);
v___x_1888_ = lean_box(0);
v_isShared_1889_ = v_isSharedCheck_1903_;
goto v_resetjp_1887_;
}
v_resetjp_1887_:
{
lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; 
v___x_1890_ = lean_box(0);
lean_inc_ref(v_fo_1880_);
v___x_1891_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints_spec__0___redArg(v_values_1882_, v_fo_1880_, v_message_1883_, v___x_1890_);
v___x_1892_ = l_List_get_x3fInternal___redArg(v_values_1882_, v_count_1884_);
if (lean_obj_tag(v___x_1892_) == 0)
{
lean_object* v_toRingOps_1893_; lean_object* v_toSemiringOps_1894_; lean_object* v_zero_1895_; lean_object* v___x_1897_; 
v_toRingOps_1893_ = lean_ctor_get(v_fo_1880_, 0);
lean_inc_ref(v_toRingOps_1893_);
lean_dec_ref(v_fo_1880_);
v_toSemiringOps_1894_ = lean_ctor_get(v_toRingOps_1893_, 0);
lean_inc_ref(v_toSemiringOps_1894_);
lean_dec_ref(v_toRingOps_1893_);
v_zero_1895_ = lean_ctor_get(v_toSemiringOps_1894_, 0);
lean_inc(v_zero_1895_);
lean_dec_ref(v_toSemiringOps_1894_);
if (v_isShared_1889_ == 0)
{
lean_ctor_set(v___x_1888_, 1, v_zero_1895_);
lean_ctor_set(v___x_1888_, 0, v___x_1891_);
v___x_1897_ = v___x_1888_;
goto v_reusejp_1896_;
}
else
{
lean_object* v_reuseFailAlloc_1898_; 
v_reuseFailAlloc_1898_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1898_, 0, v___x_1891_);
lean_ctor_set(v_reuseFailAlloc_1898_, 1, v_zero_1895_);
lean_ctor_set(v_reuseFailAlloc_1898_, 2, v_busIndex_1885_);
lean_ctor_set(v_reuseFailAlloc_1898_, 3, v_countWeight_1886_);
v___x_1897_ = v_reuseFailAlloc_1898_;
goto v_reusejp_1896_;
}
v_reusejp_1896_:
{
return v___x_1897_;
}
}
else
{
lean_object* v_val_1899_; lean_object* v___x_1901_; 
lean_dec_ref(v_fo_1880_);
v_val_1899_ = lean_ctor_get(v___x_1892_, 0);
lean_inc(v_val_1899_);
lean_dec_ref_known(v___x_1892_, 1);
if (v_isShared_1889_ == 0)
{
lean_ctor_set(v___x_1888_, 1, v_val_1899_);
lean_ctor_set(v___x_1888_, 0, v___x_1891_);
v___x_1901_ = v___x_1888_;
goto v_reusejp_1900_;
}
else
{
lean_object* v_reuseFailAlloc_1902_; 
v_reuseFailAlloc_1902_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1902_, 0, v___x_1891_);
lean_ctor_set(v_reuseFailAlloc_1902_, 1, v_val_1899_);
lean_ctor_set(v_reuseFailAlloc_1902_, 2, v_busIndex_1885_);
lean_ctor_set(v_reuseFailAlloc_1902_, 3, v_countWeight_1886_);
v___x_1901_ = v_reuseFailAlloc_1902_;
goto v_reusejp_1900_;
}
v_reusejp_1900_:
{
return v___x_1901_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg___boxed(lean_object* v_fo_1904_, lean_object* v_interaction_1905_, lean_object* v_values_1906_){
_start:
{
lean_object* v_res_1907_; 
v_res_1907_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg(v_fo_1904_, v_interaction_1905_, v_values_1906_);
lean_dec(v_values_1906_);
return v_res_1907_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval(lean_object* v_K_1908_, lean_object* v_fo_1909_, lean_object* v_interaction_1910_, lean_object* v_values_1911_){
_start:
{
lean_object* v___x_1912_; 
v___x_1912_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg(v_fo_1909_, v_interaction_1910_, v_values_1911_);
return v___x_1912_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___boxed(lean_object* v_K_1913_, lean_object* v_fo_1914_, lean_object* v_interaction_1915_, lean_object* v_values_1916_){
_start:
{
lean_object* v_res_1917_; 
v_res_1917_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval(v_K_1913_, v_fo_1914_, v_interaction_1915_, v_values_1916_);
lean_dec(v_values_1916_);
return v_res_1917_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(lean_object* v_inst_1918_, lean_object* v_x_1919_, lean_object* v_x_1920_){
_start:
{
lean_object* v_constraints_1921_; lean_object* v_interactions_1922_; lean_object* v_constraints_1923_; lean_object* v_interactions_1924_; uint8_t v___x_1925_; 
v_constraints_1921_ = lean_ctor_get(v_x_1919_, 0);
lean_inc_ref(v_constraints_1921_);
v_interactions_1922_ = lean_ctor_get(v_x_1919_, 1);
lean_inc(v_interactions_1922_);
lean_dec_ref(v_x_1919_);
v_constraints_1923_ = lean_ctor_get(v_x_1920_, 0);
lean_inc_ref(v_constraints_1923_);
v_interactions_1924_ = lean_ctor_get(v_x_1920_, 1);
lean_inc(v_interactions_1924_);
lean_dec_ref(v_x_1920_);
v___x_1925_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicExpressionDag_decEq___redArg(v_inst_1918_, v_constraints_1921_, v_constraints_1923_);
if (v___x_1925_ == 0)
{
lean_dec(v_interactions_1924_);
lean_dec(v_interactions_1922_);
return v___x_1925_;
}
else
{
lean_object* v___x_1926_; uint8_t v___x_1927_; 
v___x_1926_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicInteraction___boxed), 2, 0);
v___x_1927_ = l_instDecidableEqList___redArg(v___x_1926_, v_interactions_1922_, v_interactions_1924_);
return v___x_1927_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg___boxed(lean_object* v_inst_1928_, lean_object* v_x_1929_, lean_object* v_x_1930_){
_start:
{
uint8_t v_res_1931_; lean_object* v_r_1932_; 
v_res_1931_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(v_inst_1928_, v_x_1929_, v_x_1930_);
v_r_1932_ = lean_box(v_res_1931_);
return v_r_1932_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq(lean_object* v_F_1933_, lean_object* v_inst_1934_, lean_object* v_x_1935_, lean_object* v_x_1936_){
_start:
{
uint8_t v___x_1937_; 
v___x_1937_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(v_inst_1934_, v_x_1935_, v_x_1936_);
return v___x_1937_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___boxed(lean_object* v_F_1938_, lean_object* v_inst_1939_, lean_object* v_x_1940_, lean_object* v_x_1941_){
_start:
{
uint8_t v_res_1942_; lean_object* v_r_1943_; 
v_res_1942_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq(v_F_1938_, v_inst_1939_, v_x_1940_, v_x_1941_);
v_r_1943_ = lean_box(v_res_1942_);
return v_r_1943_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___redArg(lean_object* v_inst_1944_, lean_object* v_x_1945_, lean_object* v_x_1946_){
_start:
{
uint8_t v___x_1947_; 
v___x_1947_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(v_inst_1944_, v_x_1945_, v_x_1946_);
return v___x_1947_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___redArg___boxed(lean_object* v_inst_1948_, lean_object* v_x_1949_, lean_object* v_x_1950_){
_start:
{
uint8_t v_res_1951_; lean_object* v_r_1952_; 
v_res_1951_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___redArg(v_inst_1948_, v_x_1949_, v_x_1950_);
v_r_1952_ = lean_box(v_res_1951_);
return v_r_1952_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag(lean_object* v_F_1953_, lean_object* v_inst_1954_, lean_object* v_x_1955_, lean_object* v_x_1956_){
_start:
{
uint8_t v___x_1957_; 
v___x_1957_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(v_inst_1954_, v_x_1955_, v_x_1956_);
return v___x_1957_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag___boxed(lean_object* v_F_1958_, lean_object* v_inst_1959_, lean_object* v_x_1960_, lean_object* v_x_1961_){
_start:
{
uint8_t v_res_1962_; lean_object* v_r_1963_; 
v_res_1962_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag(v_F_1958_, v_inst_1959_, v_x_1960_, v_x_1961_);
v_r_1963_ = lean_box(v_res_1962_);
return v_r_1963_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg(lean_object* v_inst_1976_, lean_object* v_x_1977_){
_start:
{
lean_object* v_constraints_1978_; lean_object* v_interactions_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_2013_; 
v_constraints_1978_ = lean_ctor_get(v_x_1977_, 0);
v_interactions_1979_ = lean_ctor_get(v_x_1977_, 1);
v_isSharedCheck_2013_ = !lean_is_exclusive(v_x_1977_);
if (v_isSharedCheck_2013_ == 0)
{
v___x_1981_ = v_x_1977_;
v_isShared_1982_ = v_isSharedCheck_2013_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_interactions_1979_);
lean_inc(v_constraints_1978_);
lean_dec(v_x_1977_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_2013_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1989_; 
v___x_1983_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicInteraction___closed__0));
v___x_1984_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_1985_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__3));
v___x_1986_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__10);
v___x_1987_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicExpressionDag_repr___redArg(v_inst_1976_, v_constraints_1978_);
if (v_isShared_1982_ == 0)
{
lean_ctor_set_tag(v___x_1981_, 4);
lean_ctor_set(v___x_1981_, 1, v___x_1987_);
lean_ctor_set(v___x_1981_, 0, v___x_1986_);
v___x_1989_ = v___x_1981_;
goto v_reusejp_1988_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v___x_1986_);
lean_ctor_set(v_reuseFailAlloc_2012_, 1, v___x_1987_);
v___x_1989_ = v_reuseFailAlloc_2012_;
goto v_reusejp_1988_;
}
v_reusejp_1988_:
{
uint8_t v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; 
v___x_1990_ = 0;
v___x_1991_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1991_, 0, v___x_1989_);
lean_ctor_set_uint8(v___x_1991_, sizeof(void*)*1, v___x_1990_);
v___x_1992_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1992_, 0, v___x_1985_);
lean_ctor_set(v___x_1992_, 1, v___x_1991_);
v___x_1993_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_1994_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1994_, 0, v___x_1992_);
lean_ctor_set(v___x_1994_, 1, v___x_1993_);
v___x_1995_ = lean_box(1);
v___x_1996_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1996_, 0, v___x_1994_);
lean_ctor_set(v___x_1996_, 1, v___x_1995_);
v___x_1997_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg___closed__5));
v___x_1998_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1998_, 0, v___x_1996_);
lean_ctor_set(v___x_1998_, 1, v___x_1997_);
v___x_1999_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1999_, 0, v___x_1998_);
lean_ctor_set(v___x_1999_, 1, v___x_1984_);
v___x_2000_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__7);
v___x_2001_ = l_List_repr___redArg(v___x_1983_, v_interactions_1979_);
v___x_2002_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2002_, 0, v___x_2000_);
lean_ctor_set(v___x_2002_, 1, v___x_2001_);
v___x_2003_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2003_, 0, v___x_2002_);
lean_ctor_set_uint8(v___x_2003_, sizeof(void*)*1, v___x_1990_);
v___x_2004_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2004_, 0, v___x_1999_);
lean_ctor_set(v___x_2004_, 1, v___x_2003_);
v___x_2005_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10);
v___x_2006_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_2007_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2007_, 0, v___x_2006_);
lean_ctor_set(v___x_2007_, 1, v___x_2004_);
v___x_2008_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_2009_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2009_, 0, v___x_2007_);
lean_ctor_set(v___x_2009_, 1, v___x_2008_);
v___x_2010_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2010_, 0, v___x_2005_);
lean_ctor_set(v___x_2010_, 1, v___x_2009_);
v___x_2011_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2011_, 0, v___x_2010_);
lean_ctor_set_uint8(v___x_2011_, sizeof(void*)*1, v___x_1990_);
return v___x_2011_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr(lean_object* v_F_2014_, lean_object* v_inst_2015_, lean_object* v_x_2016_, lean_object* v_prec_2017_){
_start:
{
lean_object* v___x_2018_; 
v___x_2018_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg(v_inst_2015_, v_x_2016_);
return v___x_2018_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___boxed(lean_object* v_F_2019_, lean_object* v_inst_2020_, lean_object* v_x_2021_, lean_object* v_prec_2022_){
_start:
{
lean_object* v_res_2023_; 
v_res_2023_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr(v_F_2019_, v_inst_2020_, v_x_2021_, v_prec_2022_);
lean_dec(v_prec_2022_);
return v_res_2023_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag___redArg(lean_object* v_inst_2024_){
_start:
{
lean_object* v___x_2025_; 
v___x_2025_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___boxed), 4, 2);
lean_closure_set(v___x_2025_, 0, lean_box(0));
lean_closure_set(v___x_2025_, 1, v_inst_2024_);
return v___x_2025_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag(lean_object* v_F_2026_, lean_object* v_inst_2027_){
_start:
{
lean_object* v___x_2028_; 
v___x_2028_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___boxed), 4, 2);
lean_closure_set(v___x_2028_, 0, lean_box(0));
lean_closure_set(v___x_2028_, 1, v_inst_2027_);
return v___x_2028_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___redArg(lean_object* v_dag_2029_){
_start:
{
lean_object* v_constraints_2030_; lean_object* v___x_2031_; 
v_constraints_2030_ = lean_ctor_get(v_dag_2029_, 0);
v___x_2031_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_constraintCount___redArg(v_constraints_2030_);
return v___x_2031_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___redArg___boxed(lean_object* v_dag_2032_){
_start:
{
lean_object* v_res_2033_; 
v_res_2033_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___redArg(v_dag_2032_);
lean_dec_ref(v_dag_2032_);
return v_res_2033_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount(lean_object* v_F_2034_, lean_object* v_dag_2035_){
_start:
{
lean_object* v___x_2036_; 
v___x_2036_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___redArg(v_dag_2035_);
return v___x_2036_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount___boxed(lean_object* v_F_2037_, lean_object* v_dag_2038_){
_start:
{
lean_object* v_res_2039_; 
v_res_2039_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_constraintCount(v_F_2037_, v_dag_2038_);
lean_dec_ref(v_dag_2038_);
return v_res_2039_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___redArg(lean_object* v_dag_2040_){
_start:
{
lean_object* v_interactions_2041_; lean_object* v___x_2042_; 
v_interactions_2041_ = lean_ctor_get(v_dag_2040_, 1);
v___x_2042_ = l_List_lengthTR___redArg(v_interactions_2041_);
return v___x_2042_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___redArg___boxed(lean_object* v_dag_2043_){
_start:
{
lean_object* v_res_2044_; 
v_res_2044_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___redArg(v_dag_2043_);
lean_dec_ref(v_dag_2043_);
return v_res_2044_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount(lean_object* v_F_2045_, lean_object* v_dag_2046_){
_start:
{
lean_object* v___x_2047_; 
v___x_2047_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___redArg(v_dag_2046_);
return v___x_2047_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount___boxed(lean_object* v_F_2048_, lean_object* v_dag_2049_){
_start:
{
lean_object* v_res_2050_; 
v_res_2050_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_interactionCount(v_F_2048_, v_dag_2049_);
lean_dec_ref(v_dag_2049_);
return v_res_2050_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalNodes___redArg(lean_object* v_fo_2051_, lean_object* v_dag_2052_, lean_object* v_e_2053_){
_start:
{
lean_object* v_constraints_2054_; lean_object* v___x_2055_; 
v_constraints_2054_ = lean_ctor_get(v_dag_2052_, 0);
lean_inc_ref(v_constraints_2054_);
lean_dec_ref(v_dag_2052_);
v___x_2055_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalNodes___redArg(v_fo_2051_, v_constraints_2054_, v_e_2053_);
return v___x_2055_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalNodes(lean_object* v_F_2056_, lean_object* v_K_2057_, lean_object* v_fo_2058_, lean_object* v_dag_2059_, lean_object* v_e_2060_){
_start:
{
lean_object* v___x_2061_; 
v___x_2061_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalNodes___redArg(v_fo_2058_, v_dag_2059_, v_e_2060_);
return v___x_2061_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalConstraints___redArg(lean_object* v_fo_2062_, lean_object* v_dag_2063_, lean_object* v_e_2064_){
_start:
{
lean_object* v_constraints_2065_; lean_object* v___x_2066_; 
v_constraints_2065_ = lean_ctor_get(v_dag_2063_, 0);
lean_inc_ref(v_constraints_2065_);
lean_dec_ref(v_dag_2063_);
v___x_2066_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicExpressionDag_evalConstraints___redArg(v_fo_2062_, v_constraints_2065_, v_e_2064_);
return v___x_2066_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalConstraints(lean_object* v_F_2067_, lean_object* v_K_2068_, lean_object* v_fo_2069_, lean_object* v_dag_2070_, lean_object* v_e_2071_){
_start:
{
lean_object* v___x_2072_; 
v___x_2072_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalConstraints___redArg(v_fo_2069_, v_dag_2070_, v_e_2071_);
return v___x_2072_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg(lean_object* v_fo_2073_, lean_object* v_values_2074_, lean_object* v_a_2075_, lean_object* v_a_2076_){
_start:
{
if (lean_obj_tag(v_a_2075_) == 0)
{
lean_object* v___x_2077_; 
lean_dec_ref(v_fo_2073_);
v___x_2077_ = l_List_reverse___redArg(v_a_2076_);
return v___x_2077_;
}
else
{
lean_object* v_head_2078_; lean_object* v_tail_2079_; lean_object* v___x_2081_; uint8_t v_isShared_2082_; uint8_t v_isSharedCheck_2088_; 
v_head_2078_ = lean_ctor_get(v_a_2075_, 0);
v_tail_2079_ = lean_ctor_get(v_a_2075_, 1);
v_isSharedCheck_2088_ = !lean_is_exclusive(v_a_2075_);
if (v_isSharedCheck_2088_ == 0)
{
v___x_2081_ = v_a_2075_;
v_isShared_2082_ = v_isSharedCheck_2088_;
goto v_resetjp_2080_;
}
else
{
lean_inc(v_tail_2079_);
lean_inc(v_head_2078_);
lean_dec(v_a_2075_);
v___x_2081_ = lean_box(0);
v_isShared_2082_ = v_isSharedCheck_2088_;
goto v_resetjp_2080_;
}
v_resetjp_2080_:
{
lean_object* v___x_2083_; lean_object* v___x_2085_; 
lean_inc_ref(v_fo_2073_);
v___x_2083_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicInteraction_eval___redArg(v_fo_2073_, v_head_2078_, v_values_2074_);
if (v_isShared_2082_ == 0)
{
lean_ctor_set(v___x_2081_, 1, v_a_2076_);
lean_ctor_set(v___x_2081_, 0, v___x_2083_);
v___x_2085_ = v___x_2081_;
goto v_reusejp_2084_;
}
else
{
lean_object* v_reuseFailAlloc_2087_; 
v_reuseFailAlloc_2087_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2087_, 0, v___x_2083_);
lean_ctor_set(v_reuseFailAlloc_2087_, 1, v_a_2076_);
v___x_2085_ = v_reuseFailAlloc_2087_;
goto v_reusejp_2084_;
}
v_reusejp_2084_:
{
v_a_2075_ = v_tail_2079_;
v_a_2076_ = v___x_2085_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg___boxed(lean_object* v_fo_2089_, lean_object* v_values_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_){
_start:
{
lean_object* v_res_2093_; 
v_res_2093_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg(v_fo_2089_, v_values_2090_, v_a_2091_, v_a_2092_);
lean_dec(v_values_2090_);
return v_res_2093_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions___redArg(lean_object* v_fo_2094_, lean_object* v_dag_2095_, lean_object* v_e_2096_){
_start:
{
lean_object* v_interactions_2097_; lean_object* v_values_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; 
v_interactions_2097_ = lean_ctor_get(v_dag_2095_, 1);
lean_inc(v_interactions_2097_);
lean_inc_ref(v_fo_2094_);
v_values_2098_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalNodes___redArg(v_fo_2094_, v_dag_2095_, v_e_2096_);
v___x_2099_ = lean_box(0);
v___x_2100_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg(v_fo_2094_, v_values_2098_, v_interactions_2097_, v___x_2099_);
lean_dec(v_values_2098_);
return v___x_2100_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions(lean_object* v_F_2101_, lean_object* v_K_2102_, lean_object* v_fo_2103_, lean_object* v_dag_2104_, lean_object* v_e_2105_){
_start:
{
lean_object* v___x_2106_; 
v___x_2106_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions___redArg(v_fo_2103_, v_dag_2104_, v_e_2105_);
return v___x_2106_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0(lean_object* v_K_2107_, lean_object* v_fo_2108_, lean_object* v_values_2109_, lean_object* v_a_2110_, lean_object* v_a_2111_){
_start:
{
lean_object* v___x_2112_; 
v___x_2112_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___redArg(v_fo_2108_, v_values_2109_, v_a_2110_, v_a_2111_);
return v___x_2112_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0___boxed(lean_object* v_K_2113_, lean_object* v_fo_2114_, lean_object* v_values_2115_, lean_object* v_a_2116_, lean_object* v_a_2117_){
_start:
{
lean_object* v_res_2118_; 
v_res_2118_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions_spec__0(v_K_2113_, v_fo_2114_, v_values_2115_, v_a_2116_, v_a_2117_);
lean_dec(v_values_2115_);
return v_res_2118_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___lam__0(lean_object* v_inst_2119_, lean_object* v_a_2120_, lean_object* v_b_2121_){
_start:
{
uint8_t v___x_2122_; 
v___x_2122_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqVerifierSinglePreprocessedData_decEq___redArg(v_inst_2119_, v_a_2120_, v_b_2121_);
return v___x_2122_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___lam__0___boxed(lean_object* v_inst_2123_, lean_object* v_a_2124_, lean_object* v_b_2125_){
_start:
{
uint8_t v_res_2126_; lean_object* v_r_2127_; 
v_res_2126_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___lam__0(v_inst_2123_, v_a_2124_, v_b_2125_);
v_r_2127_ = lean_box(v_res_2126_);
return v_r_2127_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(lean_object* v_inst_2128_, lean_object* v_inst_2129_, lean_object* v_x_2130_, lean_object* v_x_2131_){
_start:
{
lean_object* v_preprocessedData_2132_; lean_object* v_params_2133_; lean_object* v_symbolicConstraints_2134_; lean_object* v_maxConstraintDegree_2135_; uint8_t v_isRequired_2136_; lean_object* v_unusedVariables_2137_; lean_object* v_preprocessedData_2138_; lean_object* v_params_2139_; lean_object* v_symbolicConstraints_2140_; lean_object* v_maxConstraintDegree_2141_; uint8_t v_isRequired_2142_; lean_object* v_unusedVariables_2143_; lean_object* v___f_2147_; uint8_t v___x_2148_; 
v_preprocessedData_2132_ = lean_ctor_get(v_x_2130_, 0);
lean_inc(v_preprocessedData_2132_);
v_params_2133_ = lean_ctor_get(v_x_2130_, 1);
lean_inc_ref(v_params_2133_);
v_symbolicConstraints_2134_ = lean_ctor_get(v_x_2130_, 2);
lean_inc_ref(v_symbolicConstraints_2134_);
v_maxConstraintDegree_2135_ = lean_ctor_get(v_x_2130_, 3);
lean_inc(v_maxConstraintDegree_2135_);
v_isRequired_2136_ = lean_ctor_get_uint8(v_x_2130_, sizeof(void*)*5);
v_unusedVariables_2137_ = lean_ctor_get(v_x_2130_, 4);
lean_inc(v_unusedVariables_2137_);
lean_dec_ref(v_x_2130_);
v_preprocessedData_2138_ = lean_ctor_get(v_x_2131_, 0);
lean_inc(v_preprocessedData_2138_);
v_params_2139_ = lean_ctor_get(v_x_2131_, 1);
lean_inc_ref(v_params_2139_);
v_symbolicConstraints_2140_ = lean_ctor_get(v_x_2131_, 2);
lean_inc_ref(v_symbolicConstraints_2140_);
v_maxConstraintDegree_2141_ = lean_ctor_get(v_x_2131_, 3);
lean_inc(v_maxConstraintDegree_2141_);
v_isRequired_2142_ = lean_ctor_get_uint8(v_x_2131_, sizeof(void*)*5);
v_unusedVariables_2143_ = lean_ctor_get(v_x_2131_, 4);
lean_inc(v_unusedVariables_2143_);
lean_dec_ref(v_x_2131_);
v___f_2147_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_2147_, 0, v_inst_2129_);
v___x_2148_ = l_Option_instDecidableEq___redArg(v___f_2147_, v_preprocessedData_2132_, v_preprocessedData_2138_);
if (v___x_2148_ == 0)
{
lean_dec(v_unusedVariables_2143_);
lean_dec(v_maxConstraintDegree_2141_);
lean_dec_ref(v_symbolicConstraints_2140_);
lean_dec_ref(v_params_2139_);
lean_dec(v_unusedVariables_2137_);
lean_dec(v_maxConstraintDegree_2135_);
lean_dec_ref(v_symbolicConstraints_2134_);
lean_dec_ref(v_params_2133_);
lean_dec_ref(v_inst_2128_);
return v___x_2148_;
}
else
{
uint8_t v___x_2149_; 
v___x_2149_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingParams_decEq(v_params_2133_, v_params_2139_);
if (v___x_2149_ == 0)
{
lean_dec(v_unusedVariables_2143_);
lean_dec(v_maxConstraintDegree_2141_);
lean_dec_ref(v_symbolicConstraints_2140_);
lean_dec(v_unusedVariables_2137_);
lean_dec(v_maxConstraintDegree_2135_);
lean_dec_ref(v_symbolicConstraints_2134_);
lean_dec_ref(v_inst_2128_);
return v___x_2149_;
}
else
{
uint8_t v___x_2150_; 
v___x_2150_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicConstraintsDag_decEq___redArg(v_inst_2128_, v_symbolicConstraints_2134_, v_symbolicConstraints_2140_);
if (v___x_2150_ == 0)
{
lean_dec(v_unusedVariables_2143_);
lean_dec(v_maxConstraintDegree_2141_);
lean_dec(v_unusedVariables_2137_);
lean_dec(v_maxConstraintDegree_2135_);
return v___x_2150_;
}
else
{
uint8_t v___x_2151_; 
v___x_2151_ = lean_nat_dec_eq(v_maxConstraintDegree_2135_, v_maxConstraintDegree_2141_);
lean_dec(v_maxConstraintDegree_2141_);
lean_dec(v_maxConstraintDegree_2135_);
if (v___x_2151_ == 0)
{
lean_dec(v_unusedVariables_2143_);
lean_dec(v_unusedVariables_2137_);
return v___x_2151_;
}
else
{
if (v_isRequired_2136_ == 0)
{
if (v_isRequired_2142_ == 0)
{
goto v___jp_2144_;
}
else
{
lean_dec(v_unusedVariables_2143_);
lean_dec(v_unusedVariables_2137_);
return v_isRequired_2136_;
}
}
else
{
if (v_isRequired_2142_ == 0)
{
lean_dec(v_unusedVariables_2143_);
lean_dec(v_unusedVariables_2137_);
return v_isRequired_2142_;
}
else
{
goto v___jp_2144_;
}
}
}
}
}
}
v___jp_2144_:
{
lean_object* v___x_2145_; uint8_t v___x_2146_; 
v___x_2145_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqSymbolicVariable___boxed), 2, 0);
v___x_2146_ = l_instDecidableEqList___redArg(v___x_2145_, v_unusedVariables_2137_, v_unusedVariables_2143_);
return v___x_2146_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg___boxed(lean_object* v_inst_2152_, lean_object* v_inst_2153_, lean_object* v_x_2154_, lean_object* v_x_2155_){
_start:
{
uint8_t v_res_2156_; lean_object* v_r_2157_; 
v_res_2156_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(v_inst_2152_, v_inst_2153_, v_x_2154_, v_x_2155_);
v_r_2157_ = lean_box(v_res_2156_);
return v_r_2157_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq(lean_object* v_F_2158_, lean_object* v_Digest_2159_, lean_object* v_inst_2160_, lean_object* v_inst_2161_, lean_object* v_x_2162_, lean_object* v_x_2163_){
_start:
{
uint8_t v___x_2164_; 
v___x_2164_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(v_inst_2160_, v_inst_2161_, v_x_2162_, v_x_2163_);
return v___x_2164_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___boxed(lean_object* v_F_2165_, lean_object* v_Digest_2166_, lean_object* v_inst_2167_, lean_object* v_inst_2168_, lean_object* v_x_2169_, lean_object* v_x_2170_){
_start:
{
uint8_t v_res_2171_; lean_object* v_r_2172_; 
v_res_2171_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq(v_F_2165_, v_Digest_2166_, v_inst_2167_, v_inst_2168_, v_x_2169_, v_x_2170_);
v_r_2172_ = lean_box(v_res_2171_);
return v_r_2172_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___redArg(lean_object* v_inst_2173_, lean_object* v_inst_2174_, lean_object* v_x_2175_, lean_object* v_x_2176_){
_start:
{
uint8_t v___x_2177_; 
v___x_2177_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(v_inst_2173_, v_inst_2174_, v_x_2175_, v_x_2176_);
return v___x_2177_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___redArg___boxed(lean_object* v_inst_2178_, lean_object* v_inst_2179_, lean_object* v_x_2180_, lean_object* v_x_2181_){
_start:
{
uint8_t v_res_2182_; lean_object* v_r_2183_; 
v_res_2182_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___redArg(v_inst_2178_, v_inst_2179_, v_x_2180_, v_x_2181_);
v_r_2183_ = lean_box(v_res_2182_);
return v_r_2183_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey(lean_object* v_F_2184_, lean_object* v_Digest_2185_, lean_object* v_inst_2186_, lean_object* v_inst_2187_, lean_object* v_x_2188_, lean_object* v_x_2189_){
_start:
{
uint8_t v___x_2190_; 
v___x_2190_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(v_inst_2186_, v_inst_2187_, v_x_2188_, v_x_2189_);
return v___x_2190_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey___boxed(lean_object* v_F_2191_, lean_object* v_Digest_2192_, lean_object* v_inst_2193_, lean_object* v_inst_2194_, lean_object* v_x_2195_, lean_object* v_x_2196_){
_start:
{
uint8_t v_res_2197_; lean_object* v_r_2198_; 
v_res_2197_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey(v_F_2191_, v_Digest_2192_, v_inst_2193_, v_inst_2194_, v_x_2195_, v_x_2196_);
v_r_2198_ = lean_box(v_res_2197_);
return v_r_2198_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_2208_; lean_object* v___x_2209_; 
v___x_2208_ = lean_unsigned_to_nat(20u);
v___x_2209_ = lean_nat_to_int(v___x_2208_);
return v___x_2209_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_2216_; lean_object* v___x_2217_; 
v___x_2216_ = lean_unsigned_to_nat(23u);
v___x_2217_ = lean_nat_to_int(v___x_2216_);
return v___x_2217_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg(lean_object* v_inst_2227_, lean_object* v_inst_2228_, lean_object* v_x_2229_){
_start:
{
lean_object* v_preprocessedData_2230_; lean_object* v_params_2231_; lean_object* v_symbolicConstraints_2232_; lean_object* v_maxConstraintDegree_2233_; uint8_t v_isRequired_2234_; lean_object* v_unusedVariables_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; uint8_t v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; 
v_preprocessedData_2230_ = lean_ctor_get(v_x_2229_, 0);
lean_inc(v_preprocessedData_2230_);
v_params_2231_ = lean_ctor_get(v_x_2229_, 1);
lean_inc_ref(v_params_2231_);
v_symbolicConstraints_2232_ = lean_ctor_get(v_x_2229_, 2);
lean_inc_ref(v_symbolicConstraints_2232_);
v_maxConstraintDegree_2233_ = lean_ctor_get(v_x_2229_, 3);
lean_inc(v_maxConstraintDegree_2233_);
v_isRequired_2234_ = lean_ctor_get_uint8(v_x_2229_, sizeof(void*)*5);
v_unusedVariables_2235_ = lean_ctor_get(v_x_2229_, 4);
lean_inc(v_unusedVariables_2235_);
lean_dec_ref(v_x_2229_);
v___x_2236_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicVariable___closed__0));
v___x_2237_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_2238_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__3));
v___x_2239_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__4);
v___x_2240_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___boxed), 4, 2);
lean_closure_set(v___x_2240_, 0, lean_box(0));
lean_closure_set(v___x_2240_, 1, v_inst_2228_);
v___x_2241_ = lean_unsigned_to_nat(0u);
v___x_2242_ = l_Option_repr___redArg(v___x_2240_, v_preprocessedData_2230_, v___x_2241_);
v___x_2243_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2239_);
lean_ctor_set(v___x_2243_, 1, v___x_2242_);
v___x_2244_ = 0;
v___x_2245_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2245_, 0, v___x_2243_);
lean_ctor_set_uint8(v___x_2245_, sizeof(void*)*1, v___x_2244_);
v___x_2246_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2246_, 0, v___x_2238_);
lean_ctor_set(v___x_2246_, 1, v___x_2245_);
v___x_2247_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_2248_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2248_, 0, v___x_2246_);
lean_ctor_set(v___x_2248_, 1, v___x_2247_);
v___x_2249_ = lean_box(1);
v___x_2250_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2250_, 0, v___x_2248_);
lean_ctor_set(v___x_2250_, 1, v___x_2249_);
v___x_2251_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__6));
v___x_2252_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2252_, 0, v___x_2250_);
lean_ctor_set(v___x_2252_, 1, v___x_2251_);
v___x_2253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2253_, 0, v___x_2252_);
lean_ctor_set(v___x_2253_, 1, v___x_2237_);
v___x_2254_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4);
v___x_2255_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg(v_params_2231_);
v___x_2256_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2256_, 0, v___x_2254_);
lean_ctor_set(v___x_2256_, 1, v___x_2255_);
v___x_2257_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2257_, 0, v___x_2256_);
lean_ctor_set_uint8(v___x_2257_, sizeof(void*)*1, v___x_2244_);
v___x_2258_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2258_, 0, v___x_2253_);
lean_ctor_set(v___x_2258_, 1, v___x_2257_);
v___x_2259_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2259_, 0, v___x_2258_);
lean_ctor_set(v___x_2259_, 1, v___x_2247_);
v___x_2260_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2260_, 0, v___x_2259_);
lean_ctor_set(v___x_2260_, 1, v___x_2249_);
v___x_2261_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__8));
v___x_2262_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2262_, 0, v___x_2260_);
lean_ctor_set(v___x_2262_, 1, v___x_2261_);
v___x_2263_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2263_, 0, v___x_2262_);
lean_ctor_set(v___x_2263_, 1, v___x_2237_);
v___x_2264_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__9, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__9_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__9);
v___x_2265_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprSymbolicConstraintsDag_repr___redArg(v_inst_2227_, v_symbolicConstraints_2232_);
v___x_2266_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2266_, 0, v___x_2264_);
lean_ctor_set(v___x_2266_, 1, v___x_2265_);
v___x_2267_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2267_, 0, v___x_2266_);
lean_ctor_set_uint8(v___x_2267_, sizeof(void*)*1, v___x_2244_);
v___x_2268_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2268_, 0, v___x_2263_);
lean_ctor_set(v___x_2268_, 1, v___x_2267_);
v___x_2269_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2269_, 0, v___x_2268_);
lean_ctor_set(v___x_2269_, 1, v___x_2247_);
v___x_2270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2270_, 0, v___x_2269_);
lean_ctor_set(v___x_2270_, 1, v___x_2249_);
v___x_2271_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__11));
v___x_2272_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2272_, 0, v___x_2270_);
lean_ctor_set(v___x_2272_, 1, v___x_2271_);
v___x_2273_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2273_, 0, v___x_2272_);
lean_ctor_set(v___x_2273_, 1, v___x_2237_);
v___x_2274_ = l_Nat_reprFast(v_maxConstraintDegree_2233_);
v___x_2275_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2274_);
v___x_2276_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2276_, 0, v___x_2264_);
lean_ctor_set(v___x_2276_, 1, v___x_2275_);
v___x_2277_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2277_, 0, v___x_2276_);
lean_ctor_set_uint8(v___x_2277_, sizeof(void*)*1, v___x_2244_);
v___x_2278_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2278_, 0, v___x_2273_);
lean_ctor_set(v___x_2278_, 1, v___x_2277_);
v___x_2279_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2279_, 0, v___x_2278_);
lean_ctor_set(v___x_2279_, 1, v___x_2247_);
v___x_2280_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2280_, 0, v___x_2279_);
lean_ctor_set(v___x_2280_, 1, v___x_2249_);
v___x_2281_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__13));
v___x_2282_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2282_, 0, v___x_2280_);
lean_ctor_set(v___x_2282_, 1, v___x_2281_);
v___x_2283_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2283_, 0, v___x_2282_);
lean_ctor_set(v___x_2283_, 1, v___x_2237_);
v___x_2284_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__13);
v___x_2285_ = l_Bool_repr___redArg(v_isRequired_2234_);
v___x_2286_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2286_, 0, v___x_2284_);
lean_ctor_set(v___x_2286_, 1, v___x_2285_);
v___x_2287_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2287_, 0, v___x_2286_);
lean_ctor_set_uint8(v___x_2287_, sizeof(void*)*1, v___x_2244_);
v___x_2288_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2288_, 0, v___x_2283_);
lean_ctor_set(v___x_2288_, 1, v___x_2287_);
v___x_2289_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2289_, 0, v___x_2288_);
lean_ctor_set(v___x_2289_, 1, v___x_2247_);
v___x_2290_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2290_, 0, v___x_2289_);
lean_ctor_set(v___x_2290_, 1, v___x_2249_);
v___x_2291_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg___closed__15));
v___x_2292_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2292_, 0, v___x_2290_);
lean_ctor_set(v___x_2292_, 1, v___x_2291_);
v___x_2293_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2293_, 0, v___x_2292_);
lean_ctor_set(v___x_2293_, 1, v___x_2237_);
v___x_2294_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__7);
v___x_2295_ = l_List_repr___redArg(v___x_2236_, v_unusedVariables_2235_);
v___x_2296_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2296_, 0, v___x_2294_);
lean_ctor_set(v___x_2296_, 1, v___x_2295_);
v___x_2297_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2297_, 0, v___x_2296_);
lean_ctor_set_uint8(v___x_2297_, sizeof(void*)*1, v___x_2244_);
v___x_2298_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2298_, 0, v___x_2293_);
lean_ctor_set(v___x_2298_, 1, v___x_2297_);
v___x_2299_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10);
v___x_2300_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_2301_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2301_, 0, v___x_2300_);
lean_ctor_set(v___x_2301_, 1, v___x_2298_);
v___x_2302_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_2303_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2303_, 0, v___x_2301_);
lean_ctor_set(v___x_2303_, 1, v___x_2302_);
v___x_2304_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2304_, 0, v___x_2299_);
lean_ctor_set(v___x_2304_, 1, v___x_2303_);
v___x_2305_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2305_, 0, v___x_2304_);
lean_ctor_set_uint8(v___x_2305_, sizeof(void*)*1, v___x_2244_);
return v___x_2305_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr(lean_object* v_F_2306_, lean_object* v_Digest_2307_, lean_object* v_inst_2308_, lean_object* v_inst_2309_, lean_object* v_x_2310_, lean_object* v_prec_2311_){
_start:
{
lean_object* v___x_2312_; 
v___x_2312_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___redArg(v_inst_2308_, v_inst_2309_, v_x_2310_);
return v___x_2312_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___boxed(lean_object* v_F_2313_, lean_object* v_Digest_2314_, lean_object* v_inst_2315_, lean_object* v_inst_2316_, lean_object* v_x_2317_, lean_object* v_prec_2318_){
_start:
{
lean_object* v_res_2319_; 
v_res_2319_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr(v_F_2313_, v_Digest_2314_, v_inst_2315_, v_inst_2316_, v_x_2317_, v_prec_2318_);
lean_dec(v_prec_2318_);
return v_res_2319_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey___redArg(lean_object* v_inst_2320_, lean_object* v_inst_2321_){
_start:
{
lean_object* v___x_2322_; 
v___x_2322_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___boxed), 6, 4);
lean_closure_set(v___x_2322_, 0, lean_box(0));
lean_closure_set(v___x_2322_, 1, lean_box(0));
lean_closure_set(v___x_2322_, 2, v_inst_2320_);
lean_closure_set(v___x_2322_, 3, v_inst_2321_);
return v___x_2322_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey(lean_object* v_F_2323_, lean_object* v_Digest_2324_, lean_object* v_inst_2325_, lean_object* v_inst_2326_){
_start:
{
lean_object* v___x_2327_; 
v___x_2327_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___boxed), 6, 4);
lean_closure_set(v___x_2327_, 0, lean_box(0));
lean_closure_set(v___x_2327_, 1, lean_box(0));
lean_closure_set(v___x_2327_, 2, v_inst_2325_);
lean_closure_set(v___x_2327_, 3, v_inst_2326_);
return v___x_2327_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___lam__0(lean_object* v_inst_2328_, lean_object* v_inst_2329_, lean_object* v_a_2330_, lean_object* v_b_2331_){
_start:
{
uint8_t v___x_2332_; 
v___x_2332_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqStarkVerifyingKey_decEq___redArg(v_inst_2328_, v_inst_2329_, v_a_2330_, v_b_2331_);
return v___x_2332_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___lam__0___boxed(lean_object* v_inst_2333_, lean_object* v_inst_2334_, lean_object* v_a_2335_, lean_object* v_b_2336_){
_start:
{
uint8_t v_res_2337_; lean_object* v_r_2338_; 
v_res_2337_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___lam__0(v_inst_2333_, v_inst_2334_, v_a_2335_, v_b_2336_);
v_r_2338_ = lean_box(v_res_2337_);
return v_r_2338_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(lean_object* v_inst_2339_, lean_object* v_inst_2340_, lean_object* v_x_2341_, lean_object* v_x_2342_){
_start:
{
lean_object* v_params_2343_; lean_object* v_perAir_2344_; lean_object* v_traceHeightConstraints_2345_; lean_object* v_params_2346_; lean_object* v_perAir_2347_; lean_object* v_traceHeightConstraints_2348_; uint8_t v___x_2349_; 
v_params_2343_ = lean_ctor_get(v_x_2341_, 0);
lean_inc_ref(v_params_2343_);
v_perAir_2344_ = lean_ctor_get(v_x_2341_, 1);
lean_inc(v_perAir_2344_);
v_traceHeightConstraints_2345_ = lean_ctor_get(v_x_2341_, 2);
lean_inc(v_traceHeightConstraints_2345_);
lean_dec_ref(v_x_2341_);
v_params_2346_ = lean_ctor_get(v_x_2342_, 0);
lean_inc_ref(v_params_2346_);
v_perAir_2347_ = lean_ctor_get(v_x_2342_, 1);
lean_inc(v_perAir_2347_);
v_traceHeightConstraints_2348_ = lean_ctor_get(v_x_2342_, 2);
lean_inc(v_traceHeightConstraints_2348_);
lean_dec_ref(v_x_2342_);
v___x_2349_ = lp_swirl_x2dfv_Fundamentals_Runtime_instDecidableEqSystemParams_decEq(v_params_2343_, v_params_2346_);
if (v___x_2349_ == 0)
{
lean_dec(v_traceHeightConstraints_2348_);
lean_dec(v_perAir_2347_);
lean_dec(v_traceHeightConstraints_2345_);
lean_dec(v_perAir_2344_);
lean_dec_ref(v_inst_2340_);
lean_dec_ref(v_inst_2339_);
return v___x_2349_;
}
else
{
lean_object* v___f_2350_; uint8_t v___x_2351_; 
v___f_2350_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_2350_, 0, v_inst_2339_);
lean_closure_set(v___f_2350_, 1, v_inst_2340_);
v___x_2351_ = l_instDecidableEqList___redArg(v___f_2350_, v_perAir_2344_, v_perAir_2347_);
if (v___x_2351_ == 0)
{
lean_dec(v_traceHeightConstraints_2348_);
lean_dec(v_traceHeightConstraints_2345_);
return v___x_2351_;
}
else
{
lean_object* v___x_2352_; uint8_t v___x_2353_; 
v___x_2352_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqLinearConstraint___boxed), 2, 0);
v___x_2353_ = l_instDecidableEqList___redArg(v___x_2352_, v_traceHeightConstraints_2345_, v_traceHeightConstraints_2348_);
return v___x_2353_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg___boxed(lean_object* v_inst_2354_, lean_object* v_inst_2355_, lean_object* v_x_2356_, lean_object* v_x_2357_){
_start:
{
uint8_t v_res_2358_; lean_object* v_r_2359_; 
v_res_2358_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(v_inst_2354_, v_inst_2355_, v_x_2356_, v_x_2357_);
v_r_2359_ = lean_box(v_res_2358_);
return v_r_2359_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq(lean_object* v_F_2360_, lean_object* v_Digest_2361_, lean_object* v_inst_2362_, lean_object* v_inst_2363_, lean_object* v_x_2364_, lean_object* v_x_2365_){
_start:
{
uint8_t v___x_2366_; 
v___x_2366_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(v_inst_2362_, v_inst_2363_, v_x_2364_, v_x_2365_);
return v___x_2366_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___boxed(lean_object* v_F_2367_, lean_object* v_Digest_2368_, lean_object* v_inst_2369_, lean_object* v_inst_2370_, lean_object* v_x_2371_, lean_object* v_x_2372_){
_start:
{
uint8_t v_res_2373_; lean_object* v_r_2374_; 
v_res_2373_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq(v_F_2367_, v_Digest_2368_, v_inst_2369_, v_inst_2370_, v_x_2371_, v_x_2372_);
v_r_2374_ = lean_box(v_res_2373_);
return v_r_2374_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___redArg(lean_object* v_inst_2375_, lean_object* v_inst_2376_, lean_object* v_x_2377_, lean_object* v_x_2378_){
_start:
{
uint8_t v___x_2379_; 
v___x_2379_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(v_inst_2375_, v_inst_2376_, v_x_2377_, v_x_2378_);
return v___x_2379_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___redArg___boxed(lean_object* v_inst_2380_, lean_object* v_inst_2381_, lean_object* v_x_2382_, lean_object* v_x_2383_){
_start:
{
uint8_t v_res_2384_; lean_object* v_r_2385_; 
v_res_2384_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___redArg(v_inst_2380_, v_inst_2381_, v_x_2382_, v_x_2383_);
v_r_2385_ = lean_box(v_res_2384_);
return v_r_2385_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0(lean_object* v_F_2386_, lean_object* v_Digest_2387_, lean_object* v_inst_2388_, lean_object* v_inst_2389_, lean_object* v_x_2390_, lean_object* v_x_2391_){
_start:
{
uint8_t v___x_2392_; 
v___x_2392_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(v_inst_2388_, v_inst_2389_, v_x_2390_, v_x_2391_);
return v___x_2392_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0___boxed(lean_object* v_F_2393_, lean_object* v_Digest_2394_, lean_object* v_inst_2395_, lean_object* v_inst_2396_, lean_object* v_x_2397_, lean_object* v_x_2398_){
_start:
{
uint8_t v_res_2399_; lean_object* v_r_2400_; 
v_res_2399_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0(v_F_2393_, v_Digest_2394_, v_inst_2395_, v_inst_2396_, v_x_2397_, v_x_2398_);
v_r_2400_ = lean_box(v_res_2399_);
return v_r_2400_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__6(void){
_start:
{
lean_object* v___x_2413_; lean_object* v___x_2414_; 
v___x_2413_ = lean_unsigned_to_nat(26u);
v___x_2414_ = lean_nat_to_int(v___x_2413_);
return v___x_2414_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg(lean_object* v_inst_2415_, lean_object* v_inst_2416_, lean_object* v_x_2417_){
_start:
{
lean_object* v_params_2418_; lean_object* v_perAir_2419_; lean_object* v_traceHeightConstraints_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v___x_2426_; uint8_t v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; 
v_params_2418_ = lean_ctor_get(v_x_2417_, 0);
lean_inc_ref(v_params_2418_);
v_perAir_2419_ = lean_ctor_get(v_x_2417_, 1);
lean_inc(v_perAir_2419_);
v_traceHeightConstraints_2420_ = lean_ctor_get(v_x_2417_, 2);
lean_inc(v_traceHeightConstraints_2420_);
lean_dec_ref(v_x_2417_);
v___x_2421_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprLinearConstraint___closed__0));
v___x_2422_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_2423_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__1));
v___x_2424_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__4);
v___x_2425_ = lp_swirl_x2dfv_Fundamentals_Runtime_instReprSystemParams_repr___redArg(v_params_2418_);
v___x_2426_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2426_, 0, v___x_2424_);
lean_ctor_set(v___x_2426_, 1, v___x_2425_);
v___x_2427_ = 0;
v___x_2428_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2428_, 0, v___x_2426_);
lean_ctor_set_uint8(v___x_2428_, sizeof(void*)*1, v___x_2427_);
v___x_2429_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2429_, 0, v___x_2423_);
lean_ctor_set(v___x_2429_, 1, v___x_2428_);
v___x_2430_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_2431_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2429_);
lean_ctor_set(v___x_2431_, 1, v___x_2430_);
v___x_2432_ = lean_box(1);
v___x_2433_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2433_, 0, v___x_2431_);
lean_ctor_set(v___x_2433_, 1, v___x_2432_);
v___x_2434_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__3));
v___x_2435_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2435_, 0, v___x_2433_);
lean_ctor_set(v___x_2435_, 1, v___x_2434_);
v___x_2436_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2436_, 0, v___x_2435_);
lean_ctor_set(v___x_2436_, 1, v___x_2422_);
v___x_2437_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingKey_repr___boxed), 6, 4);
lean_closure_set(v___x_2437_, 0, lean_box(0));
lean_closure_set(v___x_2437_, 1, lean_box(0));
lean_closure_set(v___x_2437_, 2, v_inst_2415_);
lean_closure_set(v___x_2437_, 3, v_inst_2416_);
v___x_2438_ = l_List_repr___redArg(v___x_2437_, v_perAir_2419_);
v___x_2439_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2439_, 0, v___x_2424_);
lean_ctor_set(v___x_2439_, 1, v___x_2438_);
v___x_2440_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2440_, 0, v___x_2439_);
lean_ctor_set_uint8(v___x_2440_, sizeof(void*)*1, v___x_2427_);
v___x_2441_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2441_, 0, v___x_2436_);
lean_ctor_set(v___x_2441_, 1, v___x_2440_);
v___x_2442_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2442_, 0, v___x_2441_);
lean_ctor_set(v___x_2442_, 1, v___x_2430_);
v___x_2443_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2443_, 0, v___x_2442_);
lean_ctor_set(v___x_2443_, 1, v___x_2432_);
v___x_2444_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__5));
v___x_2445_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2445_, 0, v___x_2443_);
lean_ctor_set(v___x_2445_, 1, v___x_2444_);
v___x_2446_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2446_, 0, v___x_2445_);
lean_ctor_set(v___x_2446_, 1, v___x_2422_);
v___x_2447_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__6, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__6_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg___closed__6);
v___x_2448_ = l_List_repr___redArg(v___x_2421_, v_traceHeightConstraints_2420_);
v___x_2449_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2449_, 0, v___x_2447_);
lean_ctor_set(v___x_2449_, 1, v___x_2448_);
v___x_2450_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2450_, 0, v___x_2449_);
lean_ctor_set_uint8(v___x_2450_, sizeof(void*)*1, v___x_2427_);
v___x_2451_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2451_, 0, v___x_2446_);
lean_ctor_set(v___x_2451_, 1, v___x_2450_);
v___x_2452_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10);
v___x_2453_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_2454_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2454_, 0, v___x_2453_);
lean_ctor_set(v___x_2454_, 1, v___x_2451_);
v___x_2455_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_2456_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2456_, 0, v___x_2454_);
lean_ctor_set(v___x_2456_, 1, v___x_2455_);
v___x_2457_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2457_, 0, v___x_2452_);
lean_ctor_set(v___x_2457_, 1, v___x_2456_);
v___x_2458_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2458_, 0, v___x_2457_);
lean_ctor_set_uint8(v___x_2458_, sizeof(void*)*1, v___x_2427_);
return v___x_2458_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr(lean_object* v_F_2459_, lean_object* v_Digest_2460_, lean_object* v_inst_2461_, lean_object* v_inst_2462_, lean_object* v_x_2463_, lean_object* v_prec_2464_){
_start:
{
lean_object* v___x_2465_; 
v___x_2465_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg(v_inst_2461_, v_inst_2462_, v_x_2463_);
return v___x_2465_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___boxed(lean_object* v_F_2466_, lean_object* v_Digest_2467_, lean_object* v_inst_2468_, lean_object* v_inst_2469_, lean_object* v_x_2470_, lean_object* v_prec_2471_){
_start:
{
lean_object* v_res_2472_; 
v_res_2472_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr(v_F_2466_, v_Digest_2467_, v_inst_2468_, v_inst_2469_, v_x_2470_, v_prec_2471_);
lean_dec(v_prec_2471_);
return v_res_2472_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0___redArg(lean_object* v_inst_2473_, lean_object* v_inst_2474_){
_start:
{
lean_object* v___x_2475_; 
v___x_2475_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___boxed), 6, 4);
lean_closure_set(v___x_2475_, 0, lean_box(0));
lean_closure_set(v___x_2475_, 1, lean_box(0));
lean_closure_set(v___x_2475_, 2, v_inst_2473_);
lean_closure_set(v___x_2475_, 3, v_inst_2474_);
return v___x_2475_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0(lean_object* v_F_2476_, lean_object* v_Digest_2477_, lean_object* v_inst_2478_, lean_object* v_inst_2479_){
_start:
{
lean_object* v___x_2480_; 
v___x_2480_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___boxed), 6, 4);
lean_closure_set(v___x_2480_, 0, lean_box(0));
lean_closure_set(v___x_2480_, 1, lean_box(0));
lean_closure_set(v___x_2480_, 2, v_inst_2478_);
lean_closure_set(v___x_2480_, 3, v_inst_2479_);
return v___x_2480_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_VerifierSinglePreprocessedData_mapDigest___redArg(lean_object* v_map_2481_, lean_object* v_data_2482_){
_start:
{
lean_object* v_commit_2483_; lean_object* v_hypercubeDim_2484_; lean_object* v_stackingWidth_2485_; lean_object* v___x_2487_; uint8_t v_isShared_2488_; uint8_t v_isSharedCheck_2493_; 
v_commit_2483_ = lean_ctor_get(v_data_2482_, 0);
v_hypercubeDim_2484_ = lean_ctor_get(v_data_2482_, 1);
v_stackingWidth_2485_ = lean_ctor_get(v_data_2482_, 2);
v_isSharedCheck_2493_ = !lean_is_exclusive(v_data_2482_);
if (v_isSharedCheck_2493_ == 0)
{
v___x_2487_ = v_data_2482_;
v_isShared_2488_ = v_isSharedCheck_2493_;
goto v_resetjp_2486_;
}
else
{
lean_inc(v_stackingWidth_2485_);
lean_inc(v_hypercubeDim_2484_);
lean_inc(v_commit_2483_);
lean_dec(v_data_2482_);
v___x_2487_ = lean_box(0);
v_isShared_2488_ = v_isSharedCheck_2493_;
goto v_resetjp_2486_;
}
v_resetjp_2486_:
{
lean_object* v___x_2489_; lean_object* v___x_2491_; 
v___x_2489_ = lean_apply_1(v_map_2481_, v_commit_2483_);
if (v_isShared_2488_ == 0)
{
lean_ctor_set(v___x_2487_, 0, v___x_2489_);
v___x_2491_ = v___x_2487_;
goto v_reusejp_2490_;
}
else
{
lean_object* v_reuseFailAlloc_2492_; 
v_reuseFailAlloc_2492_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2492_, 0, v___x_2489_);
lean_ctor_set(v_reuseFailAlloc_2492_, 1, v_hypercubeDim_2484_);
lean_ctor_set(v_reuseFailAlloc_2492_, 2, v_stackingWidth_2485_);
v___x_2491_ = v_reuseFailAlloc_2492_;
goto v_reusejp_2490_;
}
v_reusejp_2490_:
{
return v___x_2491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_VerifierSinglePreprocessedData_mapDigest(lean_object* v_Digest_2494_, lean_object* v_Digest_x27_2495_, lean_object* v_map_2496_, lean_object* v_data_2497_){
_start:
{
lean_object* v___x_2498_; 
v___x_2498_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_VerifierSinglePreprocessedData_mapDigest___redArg(v_map_2496_, v_data_2497_);
return v___x_2498_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_mapDigest___redArg(lean_object* v_map_2499_, lean_object* v_key_2500_){
_start:
{
lean_object* v_preprocessedData_2501_; 
v_preprocessedData_2501_ = lean_ctor_get(v_key_2500_, 0);
lean_inc(v_preprocessedData_2501_);
if (lean_obj_tag(v_preprocessedData_2501_) == 0)
{
lean_object* v_params_2502_; lean_object* v_symbolicConstraints_2503_; lean_object* v_maxConstraintDegree_2504_; uint8_t v_isRequired_2505_; lean_object* v_unusedVariables_2506_; lean_object* v___x_2508_; uint8_t v_isShared_2509_; uint8_t v_isSharedCheck_2514_; 
lean_dec(v_map_2499_);
v_params_2502_ = lean_ctor_get(v_key_2500_, 1);
v_symbolicConstraints_2503_ = lean_ctor_get(v_key_2500_, 2);
v_maxConstraintDegree_2504_ = lean_ctor_get(v_key_2500_, 3);
v_isRequired_2505_ = lean_ctor_get_uint8(v_key_2500_, sizeof(void*)*5);
v_unusedVariables_2506_ = lean_ctor_get(v_key_2500_, 4);
v_isSharedCheck_2514_ = !lean_is_exclusive(v_key_2500_);
if (v_isSharedCheck_2514_ == 0)
{
lean_object* v_unused_2515_; 
v_unused_2515_ = lean_ctor_get(v_key_2500_, 0);
lean_dec(v_unused_2515_);
v___x_2508_ = v_key_2500_;
v_isShared_2509_ = v_isSharedCheck_2514_;
goto v_resetjp_2507_;
}
else
{
lean_inc(v_unusedVariables_2506_);
lean_inc(v_maxConstraintDegree_2504_);
lean_inc(v_symbolicConstraints_2503_);
lean_inc(v_params_2502_);
lean_dec(v_key_2500_);
v___x_2508_ = lean_box(0);
v_isShared_2509_ = v_isSharedCheck_2514_;
goto v_resetjp_2507_;
}
v_resetjp_2507_:
{
lean_object* v___x_2510_; lean_object* v___x_2512_; 
v___x_2510_ = lean_box(0);
if (v_isShared_2509_ == 0)
{
lean_ctor_set(v___x_2508_, 0, v___x_2510_);
v___x_2512_ = v___x_2508_;
goto v_reusejp_2511_;
}
else
{
lean_object* v_reuseFailAlloc_2513_; 
v_reuseFailAlloc_2513_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v_reuseFailAlloc_2513_, 0, v___x_2510_);
lean_ctor_set(v_reuseFailAlloc_2513_, 1, v_params_2502_);
lean_ctor_set(v_reuseFailAlloc_2513_, 2, v_symbolicConstraints_2503_);
lean_ctor_set(v_reuseFailAlloc_2513_, 3, v_maxConstraintDegree_2504_);
lean_ctor_set(v_reuseFailAlloc_2513_, 4, v_unusedVariables_2506_);
lean_ctor_set_uint8(v_reuseFailAlloc_2513_, sizeof(void*)*5, v_isRequired_2505_);
v___x_2512_ = v_reuseFailAlloc_2513_;
goto v_reusejp_2511_;
}
v_reusejp_2511_:
{
return v___x_2512_;
}
}
}
else
{
lean_object* v_params_2516_; lean_object* v_symbolicConstraints_2517_; lean_object* v_maxConstraintDegree_2518_; uint8_t v_isRequired_2519_; lean_object* v_unusedVariables_2520_; lean_object* v___x_2522_; uint8_t v_isShared_2523_; uint8_t v_isSharedCheck_2536_; 
v_params_2516_ = lean_ctor_get(v_key_2500_, 1);
v_symbolicConstraints_2517_ = lean_ctor_get(v_key_2500_, 2);
v_maxConstraintDegree_2518_ = lean_ctor_get(v_key_2500_, 3);
v_isRequired_2519_ = lean_ctor_get_uint8(v_key_2500_, sizeof(void*)*5);
v_unusedVariables_2520_ = lean_ctor_get(v_key_2500_, 4);
v_isSharedCheck_2536_ = !lean_is_exclusive(v_key_2500_);
if (v_isSharedCheck_2536_ == 0)
{
lean_object* v_unused_2537_; 
v_unused_2537_ = lean_ctor_get(v_key_2500_, 0);
lean_dec(v_unused_2537_);
v___x_2522_ = v_key_2500_;
v_isShared_2523_ = v_isSharedCheck_2536_;
goto v_resetjp_2521_;
}
else
{
lean_inc(v_unusedVariables_2520_);
lean_inc(v_maxConstraintDegree_2518_);
lean_inc(v_symbolicConstraints_2517_);
lean_inc(v_params_2516_);
lean_dec(v_key_2500_);
v___x_2522_ = lean_box(0);
v_isShared_2523_ = v_isSharedCheck_2536_;
goto v_resetjp_2521_;
}
v_resetjp_2521_:
{
lean_object* v_val_2524_; lean_object* v___x_2526_; uint8_t v_isShared_2527_; uint8_t v_isSharedCheck_2535_; 
v_val_2524_ = lean_ctor_get(v_preprocessedData_2501_, 0);
v_isSharedCheck_2535_ = !lean_is_exclusive(v_preprocessedData_2501_);
if (v_isSharedCheck_2535_ == 0)
{
v___x_2526_ = v_preprocessedData_2501_;
v_isShared_2527_ = v_isSharedCheck_2535_;
goto v_resetjp_2525_;
}
else
{
lean_inc(v_val_2524_);
lean_dec(v_preprocessedData_2501_);
v___x_2526_ = lean_box(0);
v_isShared_2527_ = v_isSharedCheck_2535_;
goto v_resetjp_2525_;
}
v_resetjp_2525_:
{
lean_object* v___x_2528_; lean_object* v___x_2530_; 
v___x_2528_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_VerifierSinglePreprocessedData_mapDigest___redArg(v_map_2499_, v_val_2524_);
if (v_isShared_2527_ == 0)
{
lean_ctor_set(v___x_2526_, 0, v___x_2528_);
v___x_2530_ = v___x_2526_;
goto v_reusejp_2529_;
}
else
{
lean_object* v_reuseFailAlloc_2534_; 
v_reuseFailAlloc_2534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2534_, 0, v___x_2528_);
v___x_2530_ = v_reuseFailAlloc_2534_;
goto v_reusejp_2529_;
}
v_reusejp_2529_:
{
lean_object* v___x_2532_; 
if (v_isShared_2523_ == 0)
{
lean_ctor_set(v___x_2522_, 0, v___x_2530_);
v___x_2532_ = v___x_2522_;
goto v_reusejp_2531_;
}
else
{
lean_object* v_reuseFailAlloc_2533_; 
v_reuseFailAlloc_2533_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v_reuseFailAlloc_2533_, 0, v___x_2530_);
lean_ctor_set(v_reuseFailAlloc_2533_, 1, v_params_2516_);
lean_ctor_set(v_reuseFailAlloc_2533_, 2, v_symbolicConstraints_2517_);
lean_ctor_set(v_reuseFailAlloc_2533_, 3, v_maxConstraintDegree_2518_);
lean_ctor_set(v_reuseFailAlloc_2533_, 4, v_unusedVariables_2520_);
lean_ctor_set_uint8(v_reuseFailAlloc_2533_, sizeof(void*)*5, v_isRequired_2519_);
v___x_2532_ = v_reuseFailAlloc_2533_;
goto v_reusejp_2531_;
}
v_reusejp_2531_:
{
return v___x_2532_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_mapDigest(lean_object* v_F_2538_, lean_object* v_Digest_2539_, lean_object* v_Digest_x27_2540_, lean_object* v_map_2541_, lean_object* v_key_2542_){
_start:
{
lean_object* v___x_2543_; 
v___x_2543_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_mapDigest___redArg(v_map_2541_, v_key_2542_);
return v___x_2543_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest_spec__0___redArg(lean_object* v_f_2544_, lean_object* v_a_2545_, lean_object* v_a_2546_){
_start:
{
if (lean_obj_tag(v_a_2545_) == 0)
{
lean_object* v___x_2547_; 
lean_dec(v_f_2544_);
v___x_2547_ = l_List_reverse___redArg(v_a_2546_);
return v___x_2547_;
}
else
{
lean_object* v_head_2548_; lean_object* v_tail_2549_; lean_object* v___x_2551_; uint8_t v_isShared_2552_; uint8_t v_isSharedCheck_2558_; 
v_head_2548_ = lean_ctor_get(v_a_2545_, 0);
v_tail_2549_ = lean_ctor_get(v_a_2545_, 1);
v_isSharedCheck_2558_ = !lean_is_exclusive(v_a_2545_);
if (v_isSharedCheck_2558_ == 0)
{
v___x_2551_ = v_a_2545_;
v_isShared_2552_ = v_isSharedCheck_2558_;
goto v_resetjp_2550_;
}
else
{
lean_inc(v_tail_2549_);
lean_inc(v_head_2548_);
lean_dec(v_a_2545_);
v___x_2551_ = lean_box(0);
v_isShared_2552_ = v_isSharedCheck_2558_;
goto v_resetjp_2550_;
}
v_resetjp_2550_:
{
lean_object* v___x_2553_; lean_object* v___x_2555_; 
lean_inc(v_f_2544_);
v___x_2553_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_mapDigest___redArg(v_f_2544_, v_head_2548_);
if (v_isShared_2552_ == 0)
{
lean_ctor_set(v___x_2551_, 1, v_a_2546_);
lean_ctor_set(v___x_2551_, 0, v___x_2553_);
v___x_2555_ = v___x_2551_;
goto v_reusejp_2554_;
}
else
{
lean_object* v_reuseFailAlloc_2557_; 
v_reuseFailAlloc_2557_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2557_, 0, v___x_2553_);
lean_ctor_set(v_reuseFailAlloc_2557_, 1, v_a_2546_);
v___x_2555_ = v_reuseFailAlloc_2557_;
goto v_reusejp_2554_;
}
v_reusejp_2554_:
{
v_a_2545_ = v_tail_2549_;
v_a_2546_ = v___x_2555_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest___redArg(lean_object* v_f_2559_, lean_object* v_key_2560_){
_start:
{
lean_object* v_params_2561_; lean_object* v_perAir_2562_; lean_object* v_traceHeightConstraints_2563_; lean_object* v___x_2565_; uint8_t v_isShared_2566_; uint8_t v_isSharedCheck_2572_; 
v_params_2561_ = lean_ctor_get(v_key_2560_, 0);
v_perAir_2562_ = lean_ctor_get(v_key_2560_, 1);
v_traceHeightConstraints_2563_ = lean_ctor_get(v_key_2560_, 2);
v_isSharedCheck_2572_ = !lean_is_exclusive(v_key_2560_);
if (v_isSharedCheck_2572_ == 0)
{
v___x_2565_ = v_key_2560_;
v_isShared_2566_ = v_isSharedCheck_2572_;
goto v_resetjp_2564_;
}
else
{
lean_inc(v_traceHeightConstraints_2563_);
lean_inc(v_perAir_2562_);
lean_inc(v_params_2561_);
lean_dec(v_key_2560_);
v___x_2565_ = lean_box(0);
v_isShared_2566_ = v_isSharedCheck_2572_;
goto v_resetjp_2564_;
}
v_resetjp_2564_:
{
lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2570_; 
v___x_2567_ = lean_box(0);
v___x_2568_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest_spec__0___redArg(v_f_2559_, v_perAir_2562_, v___x_2567_);
if (v_isShared_2566_ == 0)
{
lean_ctor_set(v___x_2565_, 1, v___x_2568_);
v___x_2570_ = v___x_2565_;
goto v_reusejp_2569_;
}
else
{
lean_object* v_reuseFailAlloc_2571_; 
v_reuseFailAlloc_2571_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2571_, 0, v_params_2561_);
lean_ctor_set(v_reuseFailAlloc_2571_, 1, v___x_2568_);
lean_ctor_set(v_reuseFailAlloc_2571_, 2, v_traceHeightConstraints_2563_);
v___x_2570_ = v_reuseFailAlloc_2571_;
goto v_reusejp_2569_;
}
v_reusejp_2569_:
{
return v___x_2570_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest(lean_object* v_F_2573_, lean_object* v_Digest_2574_, lean_object* v_Digest_x27_2575_, lean_object* v_f_2576_, lean_object* v_key_2577_){
_start:
{
lean_object* v___x_2578_; 
v___x_2578_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest___redArg(v_f_2576_, v_key_2577_);
return v___x_2578_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest_spec__0(lean_object* v_F_2579_, lean_object* v_Digest_2580_, lean_object* v_Digest_x27_2581_, lean_object* v_f_2582_, lean_object* v_a_2583_, lean_object* v_a_2584_){
_start:
{
lean_object* v___x_2585_; 
v___x_2585_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest_spec__0___redArg(v_f_2582_, v_a_2583_, v_a_2584_);
return v___x_2585_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg(lean_object* v_inst_2586_, lean_object* v_inst_2587_, lean_object* v_x_2588_, lean_object* v_x_2589_){
_start:
{
lean_object* v_inner_2590_; lean_object* v_preHash_2591_; lean_object* v_inner_2592_; lean_object* v_preHash_2593_; uint8_t v___x_2594_; 
v_inner_2590_ = lean_ctor_get(v_x_2588_, 0);
lean_inc_ref(v_inner_2590_);
v_preHash_2591_ = lean_ctor_get(v_x_2588_, 1);
lean_inc(v_preHash_2591_);
lean_dec_ref(v_x_2588_);
v_inner_2592_ = lean_ctor_get(v_x_2589_, 0);
lean_inc_ref(v_inner_2592_);
v_preHash_2593_ = lean_ctor_get(v_x_2589_, 1);
lean_inc(v_preHash_2593_);
lean_dec_ref(v_x_2589_);
lean_inc_ref(v_inst_2587_);
v___x_2594_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey0_decEq___redArg(v_inst_2586_, v_inst_2587_, v_inner_2590_, v_inner_2592_);
if (v___x_2594_ == 0)
{
lean_dec(v_preHash_2593_);
lean_dec(v_preHash_2591_);
lean_dec_ref(v_inst_2587_);
return v___x_2594_;
}
else
{
lean_object* v___x_2595_; uint8_t v___x_2596_; 
v___x_2595_ = lean_apply_2(v_inst_2587_, v_preHash_2591_, v_preHash_2593_);
v___x_2596_ = lean_unbox(v___x_2595_);
return v___x_2596_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg___boxed(lean_object* v_inst_2597_, lean_object* v_inst_2598_, lean_object* v_x_2599_, lean_object* v_x_2600_){
_start:
{
uint8_t v_res_2601_; lean_object* v_r_2602_; 
v_res_2601_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg(v_inst_2597_, v_inst_2598_, v_x_2599_, v_x_2600_);
v_r_2602_ = lean_box(v_res_2601_);
return v_r_2602_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq(lean_object* v_F_2603_, lean_object* v_Digest_2604_, lean_object* v_inst_2605_, lean_object* v_inst_2606_, lean_object* v_x_2607_, lean_object* v_x_2608_){
_start:
{
uint8_t v___x_2609_; 
v___x_2609_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg(v_inst_2605_, v_inst_2606_, v_x_2607_, v_x_2608_);
return v___x_2609_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___boxed(lean_object* v_F_2610_, lean_object* v_Digest_2611_, lean_object* v_inst_2612_, lean_object* v_inst_2613_, lean_object* v_x_2614_, lean_object* v_x_2615_){
_start:
{
uint8_t v_res_2616_; lean_object* v_r_2617_; 
v_res_2616_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq(v_F_2610_, v_Digest_2611_, v_inst_2612_, v_inst_2613_, v_x_2614_, v_x_2615_);
v_r_2617_ = lean_box(v_res_2616_);
return v_r_2617_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___redArg(lean_object* v_inst_2618_, lean_object* v_inst_2619_, lean_object* v_x_2620_, lean_object* v_x_2621_){
_start:
{
uint8_t v___x_2622_; 
v___x_2622_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg(v_inst_2618_, v_inst_2619_, v_x_2620_, v_x_2621_);
return v___x_2622_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___redArg___boxed(lean_object* v_inst_2623_, lean_object* v_inst_2624_, lean_object* v_x_2625_, lean_object* v_x_2626_){
_start:
{
uint8_t v_res_2627_; lean_object* v_r_2628_; 
v_res_2627_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___redArg(v_inst_2623_, v_inst_2624_, v_x_2625_, v_x_2626_);
v_r_2628_ = lean_box(v_res_2627_);
return v_r_2628_;
}
}
LEAN_EXPORT uint8_t lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey(lean_object* v_F_2629_, lean_object* v_Digest_2630_, lean_object* v_inst_2631_, lean_object* v_inst_2632_, lean_object* v_x_2633_, lean_object* v_x_2634_){
_start:
{
uint8_t v___x_2635_; 
v___x_2635_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey_decEq___redArg(v_inst_2631_, v_inst_2632_, v_x_2633_, v_x_2634_);
return v___x_2635_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey___boxed(lean_object* v_F_2636_, lean_object* v_Digest_2637_, lean_object* v_inst_2638_, lean_object* v_inst_2639_, lean_object* v_x_2640_, lean_object* v_x_2641_){
_start:
{
uint8_t v_res_2642_; lean_object* v_r_2643_; 
v_res_2642_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instDecidableEqMultiStarkVerifyingKey(v_F_2636_, v_Digest_2637_, v_inst_2638_, v_inst_2639_, v_x_2640_, v_x_2641_);
v_r_2643_ = lean_box(v_res_2642_);
return v_r_2643_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg(lean_object* v_inst_2656_, lean_object* v_inst_2657_, lean_object* v_x_2658_){
_start:
{
lean_object* v_inner_2659_; lean_object* v_preHash_2660_; lean_object* v___x_2662_; uint8_t v_isShared_2663_; uint8_t v_isSharedCheck_2694_; 
v_inner_2659_ = lean_ctor_get(v_x_2658_, 0);
v_preHash_2660_ = lean_ctor_get(v_x_2658_, 1);
v_isSharedCheck_2694_ = !lean_is_exclusive(v_x_2658_);
if (v_isSharedCheck_2694_ == 0)
{
v___x_2662_ = v_x_2658_;
v_isShared_2663_ = v_isSharedCheck_2694_;
goto v_resetjp_2661_;
}
else
{
lean_inc(v_preHash_2660_);
lean_inc(v_inner_2659_);
lean_dec(v_x_2658_);
v___x_2662_ = lean_box(0);
v_isShared_2663_ = v_isSharedCheck_2694_;
goto v_resetjp_2661_;
}
v_resetjp_2661_:
{
lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2670_; 
v___x_2664_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__5));
v___x_2665_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__3));
v___x_2666_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__4);
v___x_2667_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_inst_2657_);
v___x_2668_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey0_repr___redArg(v_inst_2656_, v_inst_2657_, v_inner_2659_);
if (v_isShared_2663_ == 0)
{
lean_ctor_set_tag(v___x_2662_, 4);
lean_ctor_set(v___x_2662_, 1, v___x_2668_);
lean_ctor_set(v___x_2662_, 0, v___x_2666_);
v___x_2670_ = v___x_2662_;
goto v_reusejp_2669_;
}
else
{
lean_object* v_reuseFailAlloc_2693_; 
v_reuseFailAlloc_2693_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2693_, 0, v___x_2666_);
lean_ctor_set(v_reuseFailAlloc_2693_, 1, v___x_2668_);
v___x_2670_ = v_reuseFailAlloc_2693_;
goto v_reusejp_2669_;
}
v_reusejp_2669_:
{
uint8_t v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; lean_object* v___x_2679_; lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; 
v___x_2671_ = 0;
v___x_2672_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2672_, 0, v___x_2670_);
lean_ctor_set_uint8(v___x_2672_, sizeof(void*)*1, v___x_2671_);
v___x_2673_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2673_, 0, v___x_2665_);
lean_ctor_set(v___x_2673_, 1, v___x_2672_);
v___x_2674_ = ((lean_object*)(lp_swirl_x2dfv_List_repr_x27___at___00Fundamentals_VerifyingKey_instReprTraceWidth_repr_spec__0___redArg___closed__4));
v___x_2675_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2675_, 0, v___x_2673_);
lean_ctor_set(v___x_2675_, 1, v___x_2674_);
v___x_2676_ = lean_box(1);
v___x_2677_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2677_, 0, v___x_2675_);
lean_ctor_set(v___x_2677_, 1, v___x_2676_);
v___x_2678_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg___closed__5));
v___x_2679_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2679_, 0, v___x_2677_);
lean_ctor_set(v___x_2679_, 1, v___x_2678_);
v___x_2680_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2680_, 0, v___x_2679_);
lean_ctor_set(v___x_2680_, 1, v___x_2664_);
v___x_2681_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprStarkVerifyingParams_repr___redArg___closed__10);
v___x_2682_ = lean_apply_2(v_inst_2657_, v_preHash_2660_, v___x_2667_);
v___x_2683_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2683_, 0, v___x_2681_);
lean_ctor_set(v___x_2683_, 1, v___x_2682_);
v___x_2684_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2684_, 0, v___x_2683_);
lean_ctor_set_uint8(v___x_2684_, sizeof(void*)*1, v___x_2671_);
v___x_2685_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2685_, 0, v___x_2680_);
lean_ctor_set(v___x_2685_, 1, v___x_2684_);
v___x_2686_ = lean_obj_once(&lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10, &lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10_once, _init_lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprVerifierSinglePreprocessedData_repr___redArg___closed__10);
v___x_2687_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__17));
v___x_2688_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2688_, 0, v___x_2687_);
lean_ctor_set(v___x_2688_, 1, v___x_2685_);
v___x_2689_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprTraceWidth_repr___redArg___closed__18));
v___x_2690_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2690_, 0, v___x_2688_);
lean_ctor_set(v___x_2690_, 1, v___x_2689_);
v___x_2691_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2691_, 0, v___x_2686_);
lean_ctor_set(v___x_2691_, 1, v___x_2690_);
v___x_2692_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2692_, 0, v___x_2691_);
lean_ctor_set_uint8(v___x_2692_, sizeof(void*)*1, v___x_2671_);
return v___x_2692_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr(lean_object* v_F_2695_, lean_object* v_Digest_2696_, lean_object* v_inst_2697_, lean_object* v_inst_2698_, lean_object* v_x_2699_, lean_object* v_prec_2700_){
_start:
{
lean_object* v___x_2701_; 
v___x_2701_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___redArg(v_inst_2697_, v_inst_2698_, v_x_2699_);
return v___x_2701_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___boxed(lean_object* v_F_2702_, lean_object* v_Digest_2703_, lean_object* v_inst_2704_, lean_object* v_inst_2705_, lean_object* v_x_2706_, lean_object* v_prec_2707_){
_start:
{
lean_object* v_res_2708_; 
v_res_2708_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr(v_F_2702_, v_Digest_2703_, v_inst_2704_, v_inst_2705_, v_x_2706_, v_prec_2707_);
lean_dec(v_prec_2707_);
return v_res_2708_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey___redArg(lean_object* v_inst_2709_, lean_object* v_inst_2710_){
_start:
{
lean_object* v___x_2711_; 
v___x_2711_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___boxed), 6, 4);
lean_closure_set(v___x_2711_, 0, lean_box(0));
lean_closure_set(v___x_2711_, 1, lean_box(0));
lean_closure_set(v___x_2711_, 2, v_inst_2709_);
lean_closure_set(v___x_2711_, 3, v_inst_2710_);
return v___x_2711_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey(lean_object* v_F_2712_, lean_object* v_Digest_2713_, lean_object* v_inst_2714_, lean_object* v_inst_2715_){
_start:
{
lean_object* v___x_2716_; 
v___x_2716_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_VerifyingKey_instReprMultiStarkVerifyingKey_repr___boxed), 6, 4);
lean_closure_set(v___x_2716_, 0, lean_box(0));
lean_closure_set(v___x_2716_, 1, lean_box(0));
lean_closure_set(v___x_2716_, 2, v_inst_2714_);
lean_closure_set(v___x_2716_, 3, v_inst_2715_);
return v___x_2716_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_mapDigest___redArg(lean_object* v_map_2717_, lean_object* v_key_2718_){
_start:
{
lean_object* v_inner_2719_; lean_object* v_preHash_2720_; lean_object* v___x_2722_; uint8_t v_isShared_2723_; uint8_t v_isSharedCheck_2729_; 
v_inner_2719_ = lean_ctor_get(v_key_2718_, 0);
v_preHash_2720_ = lean_ctor_get(v_key_2718_, 1);
v_isSharedCheck_2729_ = !lean_is_exclusive(v_key_2718_);
if (v_isSharedCheck_2729_ == 0)
{
v___x_2722_ = v_key_2718_;
v_isShared_2723_ = v_isSharedCheck_2729_;
goto v_resetjp_2721_;
}
else
{
lean_inc(v_preHash_2720_);
lean_inc(v_inner_2719_);
lean_dec(v_key_2718_);
v___x_2722_ = lean_box(0);
v_isShared_2723_ = v_isSharedCheck_2729_;
goto v_resetjp_2721_;
}
v_resetjp_2721_:
{
lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2727_; 
lean_inc(v_map_2717_);
v___x_2724_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey0_mapDigest___redArg(v_map_2717_, v_inner_2719_);
v___x_2725_ = lean_apply_1(v_map_2717_, v_preHash_2720_);
if (v_isShared_2723_ == 0)
{
lean_ctor_set(v___x_2722_, 1, v___x_2725_);
lean_ctor_set(v___x_2722_, 0, v___x_2724_);
v___x_2727_ = v___x_2722_;
goto v_reusejp_2726_;
}
else
{
lean_object* v_reuseFailAlloc_2728_; 
v_reuseFailAlloc_2728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2728_, 0, v___x_2724_);
lean_ctor_set(v_reuseFailAlloc_2728_, 1, v___x_2725_);
v___x_2727_ = v_reuseFailAlloc_2728_;
goto v_reusejp_2726_;
}
v_reusejp_2726_:
{
return v___x_2727_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_mapDigest(lean_object* v_F_2730_, lean_object* v_Digest_2731_, lean_object* v_Digest_x27_2732_, lean_object* v_map_2733_, lean_object* v_key_2734_){
_start:
{
lean_object* v___x_2735_; 
v___x_2735_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_mapDigest___redArg(v_map_2733_, v_key_2734_);
return v___x_2735_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___redArg(lean_object* v_vk_2736_){
_start:
{
lean_object* v_inner_2737_; lean_object* v_perAir_2738_; lean_object* v___x_2739_; 
v_inner_2737_ = lean_ctor_get(v_vk_2736_, 0);
v_perAir_2738_ = lean_ctor_get(v_inner_2737_, 1);
v___x_2739_ = l_List_lengthTR___redArg(v_perAir_2738_);
return v___x_2739_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___redArg___boxed(lean_object* v_vk_2740_){
_start:
{
lean_object* v_res_2741_; 
v_res_2741_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___redArg(v_vk_2740_);
lean_dec_ref(v_vk_2740_);
return v_res_2741_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount(lean_object* v_F_2742_, lean_object* v_Digest_2743_, lean_object* v_vk_2744_){
_start:
{
lean_object* v___x_2745_; 
v___x_2745_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___redArg(v_vk_2744_);
return v___x_2745_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount___boxed(lean_object* v_F_2746_, lean_object* v_Digest_2747_, lean_object* v_vk_2748_){
_start:
{
lean_object* v_res_2749_; 
v_res_2749_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_airCount(v_F_2746_, v_Digest_2747_, v_vk_2748_);
lean_dec_ref(v_vk_2748_);
return v_res_2749_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___redArg(lean_object* v_vk_2750_, lean_object* v_air_2751_){
_start:
{
lean_object* v_inner_2752_; lean_object* v_perAir_2753_; lean_object* v___x_2754_; lean_object* v_params_2755_; lean_object* v_numPublicValues_2756_; 
v_inner_2752_ = lean_ctor_get(v_vk_2750_, 0);
v_perAir_2753_ = lean_ctor_get(v_inner_2752_, 1);
v___x_2754_ = l_List_get___redArg(v_perAir_2753_, v_air_2751_);
v_params_2755_ = lean_ctor_get(v___x_2754_, 1);
lean_inc_ref(v_params_2755_);
lean_dec(v___x_2754_);
v_numPublicValues_2756_ = lean_ctor_get(v_params_2755_, 1);
lean_inc(v_numPublicValues_2756_);
lean_dec_ref(v_params_2755_);
return v_numPublicValues_2756_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___redArg___boxed(lean_object* v_vk_2757_, lean_object* v_air_2758_){
_start:
{
lean_object* v_res_2759_; 
v_res_2759_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___redArg(v_vk_2757_, v_air_2758_);
lean_dec_ref(v_vk_2757_);
return v_res_2759_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount(lean_object* v_F_2760_, lean_object* v_Digest_2761_, lean_object* v_vk_2762_, lean_object* v_air_2763_){
_start:
{
lean_object* v___x_2764_; 
v___x_2764_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___redArg(v_vk_2762_, v_air_2763_);
return v___x_2764_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount___boxed(lean_object* v_F_2765_, lean_object* v_Digest_2766_, lean_object* v_vk_2767_, lean_object* v_air_2768_){
_start:
{
lean_object* v_res_2769_; 
v_res_2769_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_MultiStarkVerifyingKey_publicValueCount(v_F_2765_, v_Digest_2766_, v_vk_2767_, v_air_2768_);
lean_dec_ref(v_vk_2767_);
return v_res_2769_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalConstraints___redArg(lean_object* v_fo_2770_, lean_object* v_vk_2771_, lean_object* v_e_2772_){
_start:
{
lean_object* v_symbolicConstraints_2773_; lean_object* v___x_2774_; 
v_symbolicConstraints_2773_ = lean_ctor_get(v_vk_2771_, 2);
lean_inc_ref(v_symbolicConstraints_2773_);
lean_dec_ref(v_vk_2771_);
v___x_2774_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalConstraints___redArg(v_fo_2770_, v_symbolicConstraints_2773_, v_e_2772_);
return v___x_2774_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalConstraints(lean_object* v_F_2775_, lean_object* v_Digest_2776_, lean_object* v_K_2777_, lean_object* v_fo_2778_, lean_object* v_vk_2779_, lean_object* v_e_2780_){
_start:
{
lean_object* v___x_2781_; 
v___x_2781_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalConstraints___redArg(v_fo_2778_, v_vk_2779_, v_e_2780_);
return v___x_2781_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalInteractions___redArg(lean_object* v_fo_2782_, lean_object* v_vk_2783_, lean_object* v_e_2784_){
_start:
{
lean_object* v_symbolicConstraints_2785_; lean_object* v___x_2786_; 
v_symbolicConstraints_2785_ = lean_ctor_get(v_vk_2783_, 2);
lean_inc_ref(v_symbolicConstraints_2785_);
lean_dec_ref(v_vk_2783_);
v___x_2786_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_SymbolicConstraintsDag_evalInteractions___redArg(v_fo_2782_, v_symbolicConstraints_2785_, v_e_2784_);
return v___x_2786_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalInteractions(lean_object* v_F_2787_, lean_object* v_Digest_2788_, lean_object* v_K_2789_, lean_object* v_fo_2790_, lean_object* v_vk_2791_, lean_object* v_e_2792_){
_start:
{
lean_object* v___x_2793_; 
v___x_2793_ = lp_swirl_x2dfv_Fundamentals_VerifyingKey_StarkVerifyingKey_evalInteractions___redArg(v_fo_2790_, v_vk_2791_, v_e_2792_);
return v___x_2793_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Config(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_FieldOps(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_VerifyingKey(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_Runtime_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_FieldOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
