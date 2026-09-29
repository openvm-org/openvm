// Lean compiler output
// Module: Mathlib.Algebra.Order.GroupWithZero.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.Algebra.Notation.Pi.Defs public import Mathlib.Algebra.Order.GroupWithZero.Defs public import Mathlib.Algebra.Order.ZeroLEOne public import Mathlib.Tactic.Bound.Attribute public import Mathlib.Tactic.Monotonicity.Attr import Mathlib.Data.Set.Function public import Mathlib.Data.Int.Order.Basic
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
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_natLitMatcher___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_isType_x27___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchFVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchLambda___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 212, 98, 212, 98, 99, 115, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 166, 229, 100, 42, 142, 126, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(175, 127, 223, 15, 86, 131, 123, 38)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(32, 199, 8, 202, 34, 205, 5, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(89, 236, 175, 7, 241, 225, 0, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 7, .m_data = "termα>0"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(74, 178, 24, 122, 200, 193, 207, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 3, .m_data = "α>0"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__14_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "term{_:_//_}"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(12, 133, 82, 74, 101, 189, 164, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__10;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 206, 72, 126, 215, 111, 61)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(26, 242, 46, 175, 203, 191, 82, 215)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(98, 8, 55, 112, 18, 213, 110, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(227, 77, 188, 177, 196, 194, 213, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(127, 92, 126, 224, 193, 188, 223, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(48, 198, 27, 202, 90, 79, 146, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__19_value),((lean_object*)(((size_t)(2098617137) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(61, 114, 33, 44, 166, 103, 65, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__20_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(254, 143, 189, 251, 231, 196, 21, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__22_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(242, 66, 119, 16, 160, 27, 174, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__24_value),((lean_object*)(((size_t)(6) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(70, 36, 41, 238, 144, 213, 75, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__25_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__27_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "//"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__28_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_<_"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(192, 242, 106, 74, 199, 131, 133, 95)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__30_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__32_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__33_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__34_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__35_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subtype"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__5___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_natLitMatcher___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_isType_x27___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchExpr___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchFVar___boxed, .m_arity = 10, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__3___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__4___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___boxed, .m_arity = 12, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_45_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__3));
v___x_46_ = l_String_toRawSubstring_x27(v___x_45_);
return v___x_46_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__10(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9));
v___x_55_ = l_String_toRawSubstring_x27(v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__14));
v___x_114_ = l_Lean_Syntax_isOfKind(v_x_110_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = lean_box(1);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_112_);
return v___x_116_;
}
else
{
lean_object* v_quotContext_117_; lean_object* v_currMacroScope_118_; lean_object* v_ref_119_; uint8_t v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v_quotContext_117_ = lean_ctor_get(v_a_111_, 1);
v_currMacroScope_118_ = lean_ctor_get(v_a_111_, 2);
v_ref_119_ = lean_ctor_get(v_a_111_, 5);
v___x_120_ = 0;
v___x_121_ = l_Lean_SourceInfo_fromRef(v_ref_119_, v___x_120_);
v___x_122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__1));
v___x_123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__2));
lean_inc_n(v___x_121_, 11);
v___x_124_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_121_);
lean_ctor_set(v___x_124_, 1, v___x_123_);
v___x_125_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4, &lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4_once, _init_lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__4);
v___x_126_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__5));
lean_inc_n(v_currMacroScope_118_, 2);
lean_inc_n(v_quotContext_117_, 2);
v___x_127_ = l_Lean_addMacroScope(v_quotContext_117_, v___x_126_, v_currMacroScope_118_);
v___x_128_ = lean_box(0);
v___x_129_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_129_, 0, v___x_121_);
lean_ctor_set(v___x_129_, 1, v___x_125_);
lean_ctor_set(v___x_129_, 2, v___x_127_);
lean_ctor_set(v___x_129_, 3, v___x_128_);
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__7));
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__8));
v___x_132_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_121_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
v___x_133_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__10, &lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__10_once, _init_lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__10);
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__11));
v___x_135_ = l_Lean_addMacroScope(v_quotContext_117_, v___x_134_, v_currMacroScope_118_);
v___x_136_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__27));
v___x_137_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_137_, 0, v___x_121_);
lean_ctor_set(v___x_137_, 1, v___x_133_);
lean_ctor_set(v___x_137_, 2, v___x_135_);
lean_ctor_set(v___x_137_, 3, v___x_136_);
v___x_138_ = l_Lean_Syntax_node2(v___x_121_, v___x_130_, v___x_132_, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__28));
v___x_140_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_121_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v___x_141_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__30));
v___x_142_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__32));
v___x_143_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__33));
v___x_144_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_121_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v___x_145_ = l_Lean_Syntax_node1(v___x_121_, v___x_142_, v___x_144_);
v___x_146_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__34));
v___x_147_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_121_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
lean_inc_ref(v___x_129_);
v___x_148_ = l_Lean_Syntax_node3(v___x_121_, v___x_141_, v___x_145_, v___x_147_, v___x_129_);
v___x_149_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__35));
v___x_150_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_150_, 0, v___x_121_);
lean_ctor_set(v___x_150_, 1, v___x_149_);
v___x_151_ = l_Lean_Syntax_node6(v___x_121_, v___x_122_, v___x_124_, v___x_129_, v___x_138_, v___x_140_, v___x_148_, v___x_150_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_a_112_);
return v___x_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___boxed(lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______macroRules____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1(v_x_153_, v_a_154_, v_a_155_);
lean_dec_ref(v_a_154_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg(lean_object* v___y_157_){
_start:
{
lean_object* v_subExpr_159_; lean_object* v_expr_160_; lean_object* v___x_161_; 
v_subExpr_159_ = lean_ctor_get(v___y_157_, 3);
v_expr_160_ = lean_ctor_get(v_subExpr_159_, 0);
lean_inc_ref(v_expr_160_);
v___x_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_161_, 0, v_expr_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg___boxed(lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg(v___y_162_);
lean_dec_ref(v___y_162_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0(lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg(v___y_165_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___boxed(lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0(v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
return v_res_180_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0(lean_object* v_x_184_){
_start:
{
lean_object* v___x_185_; uint8_t v___x_186_; 
v___x_185_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___closed__1));
v___x_186_ = l_Lean_Expr_isConstOf(v_x_184_, v___x_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0___boxed(lean_object* v_x_187_){
_start:
{
uint8_t v_res_188_; lean_object* v_r_189_; 
v_res_188_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__0(v_x_187_);
lean_dec_ref(v_x_187_);
v_r_189_ = lean_box(v_res_188_);
return v_r_189_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1(lean_object* v_x_195_){
_start:
{
lean_object* v___x_196_; uint8_t v___x_197_; 
v___x_196_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___closed__2));
v___x_197_ = l_Lean_Expr_isConstOf(v_x_195_, v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1___boxed(lean_object* v_x_198_){
_start:
{
uint8_t v_res_199_; lean_object* v_r_200_; 
v_res_199_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__1(v_x_198_);
lean_dec_ref(v_x_198_);
v_r_200_ = lean_box(v_res_199_);
return v_r_200_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2(lean_object* v_x_206_){
_start:
{
lean_object* v___x_207_; uint8_t v___x_208_; 
v___x_207_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___closed__2));
v___x_208_ = l_Lean_Expr_isConstOf(v_x_206_, v___x_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2___boxed(lean_object* v_x_209_){
_start:
{
uint8_t v_res_210_; lean_object* v_r_211_; 
v_res_210_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__2(v_x_209_);
lean_dec_ref(v_x_209_);
v_r_211_ = lean_box(v_res_210_);
return v_r_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__3(lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_ref_219_; uint8_t v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v_ref_219_ = lean_ctor_get(v___y_216_, 5);
v___x_220_ = 0;
v___x_221_ = l_Lean_SourceInfo_fromRef(v_ref_219_, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__14));
v___x_223_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0__term_u03b1_x3e0___closed__15));
lean_inc(v___x_221_);
v___x_224_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_221_);
lean_ctor_set(v___x_224_, 1, v___x_223_);
v___x_225_ = l_Lean_Syntax_node1(v___x_221_, v___x_222_, v___x_224_);
v___x_226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_226_, 0, v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__3___boxed(lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__3(v___y_227_, v___y_228_, v___y_229_, v___y_230_, v___y_231_, v___y_232_);
lean_dec(v___y_232_);
lean_dec_ref(v___y_231_);
lean_dec(v___y_230_);
lean_dec_ref(v___y_229_);
lean_dec(v___y_228_);
lean_dec_ref(v___y_227_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__4(lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_243_, 0, v___y_235_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__4___boxed(lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__4(v___y_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_, v___y_250_);
lean_dec(v___y_250_);
lean_dec_ref(v___y_249_);
lean_dec(v___y_248_);
lean_dec_ref(v___y_247_);
lean_dec(v___y_246_);
lean_dec_ref(v___y_245_);
return v_res_252_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__5(lean_object* v_n_253_, lean_object* v_x_254_){
_start:
{
uint8_t v___x_255_; 
v___x_255_ = lean_expr_eqv(v_x_254_, v_n_253_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__5___boxed(lean_object* v_n_256_, lean_object* v_x_257_){
_start:
{
uint8_t v_res_258_; lean_object* v_r_259_; 
v_res_258_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__5(v_n_256_, v_x_257_);
lean_dec_ref(v_x_257_);
lean_dec_ref(v_n_256_);
v_r_259_ = lean_box(v_res_258_);
return v_r_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6(lean_object* v___f_262_, lean_object* v___x_263_, lean_object* v___f_264_, lean_object* v___f_265_, lean_object* v_n_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_){
_start:
{
lean_object* v___f_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__5___boxed), 2, 1);
lean_closure_set(v___f_275_, 0, v_n_266_);
v___x_276_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_276_, 0, v___f_262_);
lean_inc_ref(v___x_263_);
v___x_277_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_277_, 0, v___x_276_);
lean_closure_set(v___x_277_, 1, v___x_263_);
lean_inc_ref(v___f_264_);
v___x_278_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_278_, 0, v___x_277_);
lean_closure_set(v___x_278_, 1, v___f_264_);
v___x_279_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_279_, 0, v___f_265_);
v___x_280_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_280_, 0, v___x_279_);
lean_closure_set(v___x_280_, 1, v___x_263_);
v___x_281_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___closed__0));
v___x_282_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_282_, 0, v___x_280_);
lean_closure_set(v___x_282_, 1, v___x_281_);
v___x_283_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_283_, 0, v___x_282_);
lean_closure_set(v___x_283_, 1, v___f_264_);
v___x_284_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_284_, 0, v___x_278_);
lean_closure_set(v___x_284_, 1, v___x_283_);
v___x_285_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_285_, 0, v___f_275_);
v___x_286_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_284_, v___x_285_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___boxed(lean_object* v___f_287_, lean_object* v___x_288_, lean_object* v___f_289_, lean_object* v___f_290_, lean_object* v_n_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6(v___f_287_, v___x_288_, v___f_289_, v___f_290_, v_n_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v___y_296_);
lean_dec_ref(v___y_295_);
lean_dec(v___y_294_);
lean_dec_ref(v___y_293_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7(lean_object* v___f_307_, lean_object* v___f_308_, lean_object* v___f_309_, lean_object* v___f_310_, lean_object* v___f_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___f_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_319_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1_spec__0___redArg(v___y_312_);
lean_dec_ref(v___x_319_);
v___x_320_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_320_, 0, v___f_307_);
v___x_321_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___closed__2));
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__6___boxed), 13, 4);
lean_closure_set(v___f_322_, 0, v___f_308_);
lean_closure_set(v___f_322_, 1, v___x_321_);
lean_closure_set(v___f_322_, 2, v___f_309_);
lean_closure_set(v___f_322_, 3, v___f_310_);
v___x_323_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_323_, 0, v___x_320_);
lean_closure_set(v___x_323_, 1, v___x_321_);
v___x_324_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_324_, 0, v___x_321_);
lean_closure_set(v___x_324_, 1, v___f_322_);
v___x_325_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_326_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_323_, v___x_324_, v___x_325_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
if (lean_obj_tag(v___x_326_) == 0)
{
lean_object* v___x_327_; 
lean_dec_ref_known(v___x_326_, 1);
v___x_327_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
return v___x_327_;
}
else
{
lean_object* v_a_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_335_; 
lean_dec_ref(v___f_311_);
v_a_328_ = lean_ctor_get(v___x_326_, 0);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_335_ == 0)
{
v___x_330_ = v___x_326_;
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_a_328_);
lean_dec(v___x_326_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_333_; 
if (v_isShared_331_ == 0)
{
v___x_333_ = v___x_330_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_a_328_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7___boxed(lean_object* v___f_336_, lean_object* v___f_337_, lean_object* v___f_338_, lean_object* v___f_339_, lean_object* v___f_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___lam__7(v___f_336_, v___f_337_, v___f_338_, v___f_339_, v___f_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_);
lean_dec(v___y_346_);
lean_dec_ref(v___y_345_);
lean_dec(v___y_344_);
lean_dec_ref(v___y_343_);
lean_dec(v___y_342_);
lean_dec_ref(v___y_341_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1(lean_object* v_a_368_, lean_object* v_a_369_, lean_object* v_a_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_375_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__6));
v___x_376_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___closed__9));
v___x_377_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_375_, v___x_376_, v_a_368_, v_a_369_, v_a_370_, v_a_371_, v_a_372_, v_a_373_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1___boxed(lean_object* v_a_378_, lean_object* v_a_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib___private_Mathlib_Algebra_Order_GroupWithZero_Basic_0____aux__Mathlib__Algebra__Order__GroupWithZero__Basic______delab__app____private__Mathlib__Algebra__Order__GroupWithZero__Basic__0__term_u03b1_x3e0__1(v_a_378_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
lean_dec(v_a_383_);
lean_dec_ref(v_a_382_);
lean_dec(v_a_381_);
lean_dec_ref(v_a_380_);
lean_dec(v_a_379_);
lean_dec_ref(v_a_378_);
return v_res_385_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_ZeroLEOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
