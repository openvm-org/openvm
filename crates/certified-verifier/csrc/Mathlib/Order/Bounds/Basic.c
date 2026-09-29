// Lean compiler output
// Module: Mathlib.Order.Bounds.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.Antisymmetrization public import Mathlib.Order.Bounds.Defs public import Mathlib.Order.Directed public import Mathlib.Order.BoundedOrder.Monotone public import Mathlib.Order.Interval.Set.Basic
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Nat_decidableBallLTTR___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticBddDefault___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticBddDefault"};
static const lean_object* lp_mathlib_tacticBddDefault___closed__0 = (const lean_object*)&lp_mathlib_tacticBddDefault___closed__0_value;
static const lean_ctor_object lp_mathlib_tacticBddDefault___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticBddDefault___closed__0_value),LEAN_SCALAR_PTR_LITERAL(253, 17, 67, 45, 28, 85, 122, 106)}};
static const lean_object* lp_mathlib_tacticBddDefault___closed__1 = (const lean_object*)&lp_mathlib_tacticBddDefault___closed__1_value;
static const lean_string_object lp_mathlib_tacticBddDefault___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "bddDefault"};
static const lean_object* lp_mathlib_tacticBddDefault___closed__2 = (const lean_object*)&lp_mathlib_tacticBddDefault___closed__2_value;
static const lean_ctor_object lp_mathlib_tacticBddDefault___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_tacticBddDefault___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_tacticBddDefault___closed__3 = (const lean_object*)&lp_mathlib_tacticBddDefault___closed__3_value;
static const lean_ctor_object lp_mathlib_tacticBddDefault___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticBddDefault___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_tacticBddDefault___closed__3_value)}};
static const lean_object* lp_mathlib_tacticBddDefault___closed__4 = (const lean_object*)&lp_mathlib_tacticBddDefault___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_tacticBddDefault = (const lean_object*)&lp_mathlib_tacticBddDefault___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__8_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__14_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "OrderTop.bddAbove"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__16_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__17;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "OrderTop"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__18_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "bddAbove"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__19_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(56, 115, 10, 82, 184, 170, 227, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(238, 100, 132, 193, 178, 8, 43, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__21_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__22_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "OrderBot.bddBelow"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__23_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__24;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "OrderBot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__25 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__25_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "bddBelow"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__26 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__26_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(138, 76, 152, 81, 44, 99, 224, 67)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(41, 50, 249, 246, 117, 232, 222, 201)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__28 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__28_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__29 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__29_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableIsLeast___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableIsLeast___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableIsLeast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableIsLeast___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableIsLeast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableIsLeast___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_ofIsLUB___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_ofIsLUB(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_ofIsGLB___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_ofIsGLB(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_ofIsLUBofIsGLB___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_ofIsLUBofIsGLB(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot___redArg(lean_object* v_a_1_){
_start:
{
lean_inc(v_a_1_);
return v_a_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot___redArg___boxed(lean_object* v_a_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_IsLeast_orderBot___redArg(v_a_2_);
lean_dec(v_a_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_s_6_, lean_object* v_a_7_, lean_object* v_h_8_){
_start:
{
lean_inc(v_a_7_);
return v_a_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLeast_orderBot___boxed(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_s_11_, lean_object* v_a_12_, lean_object* v_h_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_IsLeast_orderBot(v_00_u03b1_9_, v_inst_10_, v_s_11_, v_a_12_, v_h_13_);
lean_dec(v_a_12_);
lean_dec_ref(v_inst_10_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop___redArg(lean_object* v_a_15_){
_start:
{
lean_inc(v_a_15_);
return v_a_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop___redArg___boxed(lean_object* v_a_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_IsGreatest_orderTop___redArg(v_a_16_);
lean_dec(v_a_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_, lean_object* v_s_20_, lean_object* v_a_21_, lean_object* v_h_22_){
_start:
{
lean_inc(v_a_21_);
return v_a_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsGreatest_orderTop___boxed(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_, lean_object* v_s_25_, lean_object* v_a_26_, lean_object* v_h_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_IsGreatest_orderTop(v_00_u03b1_23_, v_inst_24_, v_s_25_, v_a_26_, v_h_27_);
lean_dec(v_a_26_);
lean_dec_ref(v_inst_24_);
return v_res_28_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__17(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__16));
v___x_77_ = l_String_toRawSubstring_x27(v___x_76_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__24(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__23));
v___x_91_ = l_String_toRawSubstring_x27(v___x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1(lean_object* v_x_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
lean_object* v___x_106_; uint8_t v___x_107_; 
v___x_106_ = ((lean_object*)(lp_mathlib_tacticBddDefault___closed__1));
v___x_107_ = l_Lean_Syntax_isOfKind(v_x_103_, v___x_106_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = lean_box(1);
v___x_109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v_a_105_);
return v___x_109_;
}
else
{
lean_object* v_quotContext_110_; lean_object* v_currMacroScope_111_; lean_object* v_ref_112_; uint8_t v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v_quotContext_110_ = lean_ctor_get(v_a_104_, 1);
v_currMacroScope_111_ = lean_ctor_get(v_a_104_, 2);
v_ref_112_ = lean_ctor_get(v_a_104_, 5);
v___x_113_ = 0;
v___x_114_ = l_Lean_SourceInfo_fromRef(v_ref_112_, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__3));
v___x_116_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__4));
lean_inc_n(v___x_114_, 16);
v___x_117_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_114_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
v___x_118_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__6));
v___x_119_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__8));
v___x_120_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__9));
v___x_121_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_114_);
lean_ctor_set(v___x_121_, 1, v___x_120_);
v___x_122_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__11));
v___x_123_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__13));
v___x_124_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__14));
v___x_125_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__15));
v___x_126_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_126_, 0, v___x_114_);
lean_ctor_set(v___x_126_, 1, v___x_124_);
v___x_127_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__17, &lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__17_once, _init_lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__17);
v___x_128_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__20));
lean_inc_n(v_currMacroScope_111_, 2);
lean_inc_n(v_quotContext_110_, 2);
v___x_129_ = l_Lean_addMacroScope(v_quotContext_110_, v___x_128_, v_currMacroScope_111_);
v___x_130_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__22));
v___x_131_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_131_, 0, v___x_114_);
lean_ctor_set(v___x_131_, 1, v___x_127_);
lean_ctor_set(v___x_131_, 2, v___x_129_);
lean_ctor_set(v___x_131_, 3, v___x_130_);
lean_inc_ref(v___x_126_);
v___x_132_ = l_Lean_Syntax_node2(v___x_114_, v___x_125_, v___x_126_, v___x_131_);
v___x_133_ = l_Lean_Syntax_node1(v___x_114_, v___x_118_, v___x_132_);
v___x_134_ = l_Lean_Syntax_node1(v___x_114_, v___x_123_, v___x_133_);
v___x_135_ = l_Lean_Syntax_node1(v___x_114_, v___x_122_, v___x_134_);
lean_inc_ref(v___x_121_);
v___x_136_ = l_Lean_Syntax_node2(v___x_114_, v___x_119_, v___x_121_, v___x_135_);
v___x_137_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__24, &lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__24_once, _init_lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__24);
v___x_138_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__27));
v___x_139_ = l_Lean_addMacroScope(v_quotContext_110_, v___x_138_, v_currMacroScope_111_);
v___x_140_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___closed__29));
v___x_141_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_141_, 0, v___x_114_);
lean_ctor_set(v___x_141_, 1, v___x_137_);
lean_ctor_set(v___x_141_, 2, v___x_139_);
lean_ctor_set(v___x_141_, 3, v___x_140_);
v___x_142_ = l_Lean_Syntax_node2(v___x_114_, v___x_125_, v___x_126_, v___x_141_);
v___x_143_ = l_Lean_Syntax_node1(v___x_114_, v___x_118_, v___x_142_);
v___x_144_ = l_Lean_Syntax_node1(v___x_114_, v___x_123_, v___x_143_);
v___x_145_ = l_Lean_Syntax_node1(v___x_114_, v___x_122_, v___x_144_);
v___x_146_ = l_Lean_Syntax_node2(v___x_114_, v___x_119_, v___x_121_, v___x_145_);
v___x_147_ = l_Lean_Syntax_node2(v___x_114_, v___x_118_, v___x_136_, v___x_146_);
v___x_148_ = l_Lean_Syntax_node2(v___x_114_, v___x_116_, v___x_117_, v___x_147_);
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v_a_105_);
return v___x_149_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1___boxed(lean_object* v_x_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib___aux__Mathlib__Order__Bounds__Basic______macroRules__tacticBddDefault__1(v_x_150_, v_a_151_, v_a_152_);
lean_dec_ref(v_a_151_);
return v_res_153_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableIsLeast___redArg___lam__0(lean_object* v_inst_154_, uint8_t v___x_155_, lean_object* v_n_156_, lean_object* v_h_157_){
_start:
{
lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_158_ = lean_apply_1(v_inst_154_, v_n_156_);
v___x_159_ = lean_unbox(v___x_158_);
if (v___x_159_ == 0)
{
return v___x_155_;
}
else
{
uint8_t v___x_160_; 
v___x_160_ = 0;
return v___x_160_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableIsLeast___redArg___lam__0___boxed(lean_object* v_inst_161_, lean_object* v___x_162_, lean_object* v_n_163_, lean_object* v_h_164_){
_start:
{
uint8_t v___x_80__boxed_165_; uint8_t v_res_166_; lean_object* v_r_167_; 
v___x_80__boxed_165_ = lean_unbox(v___x_162_);
v_res_166_ = lp_mathlib_Nat_instDecidableIsLeast___redArg___lam__0(v_inst_161_, v___x_80__boxed_165_, v_n_163_, v_h_164_);
v_r_167_ = lean_box(v_res_166_);
return v_r_167_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableIsLeast___redArg(lean_object* v_n_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; uint8_t v___x_171_; 
lean_inc_ref(v_inst_169_);
lean_inc(v_n_168_);
v___x_170_ = lean_apply_1(v_inst_169_, v_n_168_);
v___x_171_ = lean_unbox(v___x_170_);
if (v___x_171_ == 0)
{
uint8_t v___x_172_; 
lean_dec_ref(v_inst_169_);
lean_dec(v_n_168_);
v___x_172_ = lean_unbox(v___x_170_);
return v___x_172_;
}
else
{
lean_object* v___f_173_; uint8_t v___x_174_; 
v___f_173_ = lean_alloc_closure((void*)(lp_mathlib_Nat_instDecidableIsLeast___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_173_, 0, v_inst_169_);
lean_closure_set(v___f_173_, 1, v___x_170_);
v___x_174_ = l_Nat_decidableBallLTTR___redArg(v_n_168_, v___f_173_);
return v___x_174_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableIsLeast___redArg___boxed(lean_object* v_n_175_, lean_object* v_inst_176_){
_start:
{
uint8_t v_res_177_; lean_object* v_r_178_; 
v_res_177_ = lp_mathlib_Nat_instDecidableIsLeast___redArg(v_n_175_, v_inst_176_);
v_r_178_ = lean_box(v_res_177_);
return v_r_178_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableIsLeast(lean_object* v_p_179_, lean_object* v_n_180_, lean_object* v_inst_181_){
_start:
{
uint8_t v___x_182_; 
v___x_182_ = lp_mathlib_Nat_instDecidableIsLeast___redArg(v_n_180_, v_inst_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableIsLeast___boxed(lean_object* v_p_183_, lean_object* v_n_184_, lean_object* v_inst_185_){
_start:
{
uint8_t v_res_186_; lean_object* v_r_187_; 
v_res_186_ = lp_mathlib_Nat_instDecidableIsLeast(v_p_183_, v_n_184_, v_inst_185_);
v_r_187_ = lean_box(v_res_186_);
return v_r_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_ofIsLUB___redArg(lean_object* v_inst_188_, lean_object* v_sup_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_190_, 0, v_inst_188_);
lean_ctor_set(v___x_190_, 1, v_sup_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_ofIsLUB(lean_object* v_00_u03b1_191_, lean_object* v_inst_192_, lean_object* v_sup_193_, lean_object* v_isLUB__pair_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v_inst_192_);
lean_ctor_set(v___x_195_, 1, v_sup_193_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_ofIsGLB___redArg(lean_object* v_inst_196_, lean_object* v_sup_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_198_, 0, v_inst_196_);
lean_ctor_set(v___x_198_, 1, v_sup_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_ofIsGLB(lean_object* v_00_u03b1_199_, lean_object* v_inst_200_, lean_object* v_sup_201_, lean_object* v_isLUB__pair_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_203_, 0, v_inst_200_);
lean_ctor_set(v___x_203_, 1, v_sup_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_ofIsLUBofIsGLB___redArg(lean_object* v_inst_204_, lean_object* v_sup_205_, lean_object* v_inf_206_){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_207_, 0, v_inst_204_);
lean_ctor_set(v___x_207_, 1, v_sup_205_);
v___x_208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_inf_206_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_ofIsLUBofIsGLB(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_, lean_object* v_sup_211_, lean_object* v_inf_212_, lean_object* v_isLUB__pair_213_, lean_object* v_isGLB__pair_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Lattice_ofIsLUBofIsGLB___redArg(v_inst_210_, v_sup_211_, v_inf_212_);
return v___x_215_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Monotone(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Monotone(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Monotone(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Monotone(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
