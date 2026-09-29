// Lean compiler output
// Module: Mathlib.Algebra.Order.Nonneg.Module
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.RingHom public import Mathlib.Algebra.Order.Module.Defs public import Mathlib.Algebra.Order.Nonneg.Basic
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
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_isType_x27___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchFVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 212, 98, 212, 98, 99, 115, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 166, 229, 100, 42, 142, 126, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Nonneg"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8_value),LEAN_SCALAR_PTR_LITERAL(21, 207, 92, 229, 255, 79, 166, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Module"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__10_value),LEAN_SCALAR_PTR_LITERAL(149, 21, 255, 91, 22, 197, 184, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 172, 180, 154, 131, 242, 52, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "termR≥0"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__13_value),LEAN_SCALAR_PTR_LITERAL(12, 210, 145, 232, 130, 125, 254, 16)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "R≥0"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__14_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8_value),LEAN_SCALAR_PTR_LITERAL(221, 99, 24, 205, 44, 73, 232, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__9_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "R"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__14;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(10, 150, 1, 122, 163, 250, 19, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(10, 150, 1, 122, 163, 250, 19, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(99, 199, 61, 239, 123, 123, 35, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__2_value),LEAN_SCALAR_PTR_LITERAL(62, 116, 43, 91, 63, 55, 139, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__19_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__4_value),LEAN_SCALAR_PTR_LITERAL(86, 139, 173, 148, 245, 19, 10, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__20_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 1, 14, 26, 90, 211, 131, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__21_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8_value),LEAN_SCALAR_PTR_LITERAL(209, 41, 96, 65, 2, 157, 72, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__22_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__10_value),LEAN_SCALAR_PTR_LITERAL(169, 97, 25, 219, 124, 213, 193, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__23_value),((lean_object*)(((size_t)(203535399) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(180, 219, 143, 177, 191, 42, 2, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__24_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(187, 8, 18, 107, 150, 13, 133, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__26_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__26_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(27, 207, 40, 162, 157, 171, 242, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__28_value),((lean_object*)(((size_t)(6) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(83, 24, 224, 182, 170, 160, 117, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__29_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__30_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__31_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_isType_x27___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchExpr___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchFVar___boxed, .m_arity = 10, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__8));
v___x_50_ = l_String_toRawSubstring_x27(v___x_49_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__14(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__13));
v___x_69_ = l_String_toRawSubstring_x27(v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1(lean_object* v_x_114_, lean_object* v_a_115_, lean_object* v_a_116_){
_start:
{
lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_117_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__14));
v___x_118_ = l_Lean_Syntax_isOfKind(v_x_114_, v___x_117_);
if (v___x_118_ == 0)
{
lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_119_ = lean_box(1);
v___x_120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v_a_116_);
return v___x_120_;
}
else
{
lean_object* v_quotContext_121_; lean_object* v_currMacroScope_122_; lean_object* v_ref_123_; uint8_t v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v_quotContext_121_ = lean_ctor_get(v_a_115_, 1);
v_currMacroScope_122_ = lean_ctor_get(v_a_115_, 2);
v_ref_123_ = lean_ctor_get(v_a_115_, 5);
v___x_124_ = 0;
v___x_125_ = l_Lean_SourceInfo_fromRef(v_ref_123_, v___x_124_);
v___x_126_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4));
v___x_127_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5, &lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5_once, _init_lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__5);
v___x_128_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6));
lean_inc_n(v_currMacroScope_122_, 2);
lean_inc_n(v_quotContext_121_, 2);
v___x_129_ = l_Lean_addMacroScope(v_quotContext_121_, v___x_128_, v_currMacroScope_122_);
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__10));
lean_inc_n(v___x_125_, 3);
v___x_131_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_131_, 0, v___x_125_);
lean_ctor_set(v___x_131_, 1, v___x_127_);
lean_ctor_set(v___x_131_, 2, v___x_129_);
lean_ctor_set(v___x_131_, 3, v___x_130_);
v___x_132_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__12));
v___x_133_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__14, &lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__14_once, _init_lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__14);
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__15));
v___x_135_ = l_Lean_addMacroScope(v_quotContext_121_, v___x_134_, v_currMacroScope_122_);
v___x_136_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__31));
v___x_137_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_137_, 0, v___x_125_);
lean_ctor_set(v___x_137_, 1, v___x_133_);
lean_ctor_set(v___x_137_, 2, v___x_135_);
lean_ctor_set(v___x_137_, 3, v___x_136_);
v___x_138_ = l_Lean_Syntax_node1(v___x_125_, v___x_132_, v___x_137_);
v___x_139_ = l_Lean_Syntax_node2(v___x_125_, v___x_126_, v___x_131_, v___x_138_);
v___x_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v_a_116_);
return v___x_140_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___boxed(lean_object* v_x_141_, lean_object* v_a_142_, lean_object* v_a_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1(v_x_141_, v_a_142_, v_a_143_);
lean_dec_ref(v_a_142_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg(lean_object* v___y_145_){
_start:
{
lean_object* v_subExpr_147_; lean_object* v_expr_148_; lean_object* v___x_149_; 
v_subExpr_147_ = lean_ctor_get(v___y_145_, 3);
v_expr_148_ = lean_ctor_get(v_subExpr_147_, 0);
lean_inc_ref(v_expr_148_);
v___x_149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_149_, 0, v_expr_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg___boxed(lean_object* v___y_150_, lean_object* v___y_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg(v___y_150_);
lean_dec_ref(v___y_150_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0(lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg(v___y_153_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___boxed(lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0(v___y_161_, v___y_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
lean_dec(v___y_164_);
lean_dec_ref(v___y_163_);
lean_dec(v___y_162_);
lean_dec_ref(v___y_161_);
return v_res_168_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__0(lean_object* v_x_169_){
_start:
{
lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______macroRules____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__6));
v___x_171_ = l_Lean_Expr_isConstOf(v_x_169_, v___x_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__0___boxed(lean_object* v_x_172_){
_start:
{
uint8_t v_res_173_; lean_object* v_r_174_; 
v_res_173_ = lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__0(v_x_172_);
lean_dec_ref(v_x_172_);
v_r_174_ = lean_box(v_res_173_);
return v_r_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__1(lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_183_, 0, v___y_175_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__1___boxed(lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__1(v___y_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__2(lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v_ref_200_; uint8_t v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v_ref_200_ = lean_ctor_get(v___y_197_, 5);
v___x_201_ = 0;
v___x_202_ = l_Lean_SourceInfo_fromRef(v_ref_200_, v___x_201_);
v___x_203_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__14));
v___x_204_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0__termR_u22650___closed__15));
lean_inc(v___x_202_);
v___x_205_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_202_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
v___x_206_ = l_Lean_Syntax_node1(v___x_202_, v___x_203_, v___x_205_);
v___x_207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__2___boxed(lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__2(v___y_208_, v___y_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
lean_dec(v___y_209_);
lean_dec_ref(v___y_208_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3(lean_object* v___f_222_, lean_object* v___f_223_, lean_object* v___f_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_232_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1_spec__0___redArg(v___y_225_);
lean_dec_ref(v___x_232_);
v___x_233_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_233_, 0, v___f_222_);
v___x_234_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___closed__2));
v___x_235_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_235_, 0, v___x_233_);
lean_closure_set(v___x_235_, 1, v___x_234_);
lean_inc_ref(v___f_223_);
v___x_236_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_236_, 0, v___x_235_);
lean_closure_set(v___x_236_, 1, v___f_223_);
v___x_237_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_238_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_236_, v___f_223_, v___x_237_, v___y_225_, v___y_226_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
if (lean_obj_tag(v___x_238_) == 0)
{
lean_object* v___x_239_; 
lean_dec_ref_known(v___x_238_, 1);
v___x_239_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
return v___x_239_;
}
else
{
lean_object* v_a_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
lean_dec_ref(v___f_224_);
v_a_240_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_247_ == 0)
{
v___x_242_ = v___x_238_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_a_240_);
lean_dec(v___x_238_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_a_240_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3___boxed(lean_object* v___f_248_, lean_object* v___f_249_, lean_object* v___f_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___lam__3(v___f_248_, v___f_249_, v___f_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_);
lean_dec(v___y_256_);
lean_dec_ref(v___y_255_);
lean_dec(v___y_254_);
lean_dec_ref(v___y_253_);
lean_dec(v___y_252_);
lean_dec_ref(v___y_251_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1(lean_object* v_a_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_281_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__4));
v___x_282_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___closed__7));
v___x_283_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_281_, v___x_282_, v_a_274_, v_a_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1___boxed(lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib___private_Mathlib_Algebra_Order_Nonneg_Module_0____aux__Mathlib__Algebra__Order__Nonneg__Module______delab__app____private__Mathlib__Algebra__Order__Nonneg__Module__0__termR_u22650__1(v_a_284_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_);
lean_dec(v_a_289_);
lean_dec_ref(v_a_288_);
lean_dec(v_a_287_);
lean_dec_ref(v_a_286_);
lean_dec(v_a_285_);
lean_dec_ref(v_a_284_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul___redArg___lam__0(lean_object* v_inst_292_, lean_object* v_c_293_, lean_object* v_x_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lean_apply_2(v_inst_292_, v_c_293_, v_x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul___redArg(lean_object* v_inst_296_){
_start:
{
lean_object* v___f_297_; 
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_297_, 0, v_inst_296_);
return v___f_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul(lean_object* v_R_298_, lean_object* v_S_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_){
_start:
{
lean_object* v___f_303_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_303_, 0, v_inst_302_);
return v___f_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMul___boxed(lean_object* v_R_304_, lean_object* v_S_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_inst_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Nonneg_instSMul(v_R_304_, v_S_305_, v_inst_306_, v_inst_307_, v_inst_308_);
lean_dec_ref(v_inst_307_);
lean_dec_ref(v_inst_306_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMulWithZero___redArg(lean_object* v_inst_310_){
_start:
{
lean_object* v___f_311_; 
v___f_311_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_311_, 0, v_inst_310_);
return v___f_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMulWithZero(lean_object* v_R_312_, lean_object* v_S_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___f_318_; 
v___f_318_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_318_, 0, v_inst_317_);
return v___f_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instSMulWithZero___boxed(lean_object* v_R_319_, lean_object* v_S_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib_Nonneg_instSMulWithZero(v_R_319_, v_S_320_, v_inst_321_, v_inst_322_, v_inst_323_, v_inst_324_);
lean_dec(v_inst_323_);
lean_dec_ref(v_inst_322_);
lean_dec_ref(v_inst_321_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instModule___redArg(lean_object* v_inst_326_){
_start:
{
lean_object* v___f_327_; 
v___f_327_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_327_, 0, v_inst_326_);
return v___f_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instModule(lean_object* v_R_328_, lean_object* v_M_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_){
_start:
{
lean_object* v___f_335_; 
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_335_, 0, v_inst_334_);
return v___f_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instModule___boxed(lean_object* v_R_336_, lean_object* v_M_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_Nonneg_instModule(v_R_336_, v_M_337_, v_inst_338_, v_inst_339_, v_inst_340_, v_inst_341_, v_inst_342_);
lean_dec_ref(v_inst_341_);
lean_dec_ref(v_inst_339_);
lean_dec_ref(v_inst_338_);
return v_res_343_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_RingHom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_RingHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_RingHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_RingHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(builtin);
}
#ifdef __cplusplus
}
#endif
