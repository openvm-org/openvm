// Lean compiler output
// Module: Mathlib.Data.Set.Notation
// Imports: public import Init public meta import Init public import Mathlib.Util.Notation3 public meta import Mathlib.Lean.Expr.ExtraRecognizers public import Mathlib.Data.Set.Operations
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
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchLambda___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPCoercions___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value;
static const lean_string_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Notation"};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__1 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__1_value;
static const lean_string_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_↓∩_"};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__2 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(182, 98, 3, 59, 80, 203, 164, 154)}};
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(98, 83, 98, 112, 55, 53, 94, 118)}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value;
static const lean_string_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__4 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__5 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__5_value;
static const lean_string_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ↓∩ "};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__6 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__6_value)}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__7 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__7_value;
static const lean_string_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__8 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__9 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__9_value),((lean_object*)(((size_t)(67) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__10 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__5_value),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__7_value),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__10_value)}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__11 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(67) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__11_value)}};
static const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__12 = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_Notation_term___u2193_u2229__ = (const lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__12_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__3 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__5 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__5_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__7 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__7_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__8 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__9 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__9_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__10 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__10_value;
static lean_once_cell_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__11;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(182, 98, 3, 59, 80, 203, 164, 154)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__12 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__12_value)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__13 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__14 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__14_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 9, .m_data = "term_⁻¹'_"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__15 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__15_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(78, 198, 251, 91, 103, 37, 163, 106)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__16 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__16_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Subtype.val"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__17 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__17_value;
static lean_once_cell_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__18;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subtype"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__19 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__19_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "val"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__20 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__20_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(69, 191, 88, 200, 184, 193, 168, 219)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__22 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__22_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__23 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__23_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 3, .m_data = "⁻¹'"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__24 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__24_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__25 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__25_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__26 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__26_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__27 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__27_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "typeOf"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__28 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__28_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value_aux_1),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value_aux_2),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(26, 238, 102, 207, 116, 185, 165, 59)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "type_of%"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__30 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__30_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__31 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__31_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__32 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__32_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value_aux_1),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value_aux_2),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33_value;
static lean_once_cell_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__34;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__36 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__36_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35_value)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__37 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__37_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__37_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__38 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__38_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__36_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__38_value)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__39 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__39_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__40 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__40_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value_aux_1),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value_aux_2),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__40_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__42 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__42_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Membership"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mem"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 217, 109, 94, 255, 55, 82, 109)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(224, 90, 126, 237, 128, 148, 153, 69)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__2___boxed(lean_object*);
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elem"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(20, 190, 239, 85, 165, 199, 80, 79)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___boxed(lean_object*);
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "preimage"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__0_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 75, 132, 28, 148, 34, 22, 147)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "A"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__0_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(125, 144, 3, 41, 245, 105, 96, 211)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__1_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__1_value)} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__2 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__2_value;
static const lean_string_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "B"};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__3 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__3_value;
static const lean_ctor_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__3_value),LEAN_SCALAR_PTR_LITERAL(112, 162, 160, 92, 17, 139, 201, 28)}};
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__4 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__4_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__4_value)} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__5 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__0_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__1_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__2 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__2_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__3 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__3_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__4 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__4_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__5___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__5 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__5_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___boxed, .m_arity = 13, .m_num_fixed = 6, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__4_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__3_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__5_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__0_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__1_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__2_value)} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__6 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__6_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__7 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__7_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__8 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__8_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__6_value)} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__9 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__9_value;
static const lean_closure_object lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__8_value),((lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__9_value)} };
static const lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__10 = (const lean_object*)&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_instCoeHeadElem(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__0;
static const lean_closure_object lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "coeNotation"};
static const lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 100, 71, 170, 251, 12, 50, 58)}};
static const lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↑"};
static const lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set_Notation_delabSetImageSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___closed__0 = (const lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___closed__0_value;
static const lean_closure_object lp_mathlib_Set_Notation_delabSetImageSubtype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPCoercions___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___closed__1 = (const lean_object*)&lp_mathlib_Set_Notation_delabSetImageSubtype___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__11(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__10));
v___x_51_ = l_String_toRawSubstring_x27(v___x_50_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__18(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_65_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__17));
v___x_66_ = l_String_toRawSubstring_x27(v___x_65_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__34(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = ((lean_object*)(lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__0));
v___x_98_ = l_String_toRawSubstring_x27(v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1(lean_object* v_x_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_122_ = ((lean_object*)(lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3));
lean_inc(v_x_119_);
v___x_123_ = l_Lean_Syntax_isOfKind(v_x_119_, v___x_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_dec(v_x_119_);
v___x_124_ = lean_box(1);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_121_);
return v___x_125_;
}
else
{
lean_object* v_quotContext_126_; lean_object* v_currMacroScope_127_; lean_object* v_ref_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; uint8_t v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v_quotContext_126_ = lean_ctor_get(v_a_120_, 1);
v_currMacroScope_127_ = lean_ctor_get(v_a_120_, 2);
v_ref_128_ = lean_ctor_get(v_a_120_, 5);
v___x_129_ = lean_unsigned_to_nat(0u);
v___x_130_ = l_Lean_Syntax_getArg(v_x_119_, v___x_129_);
v___x_131_ = lean_unsigned_to_nat(2u);
v___x_132_ = l_Lean_Syntax_getArg(v_x_119_, v___x_131_);
lean_dec(v_x_119_);
v___x_133_ = 0;
v___x_134_ = l_Lean_SourceInfo_fromRef(v_ref_128_, v___x_133_);
v___x_135_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__4));
v___x_136_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__6));
v___x_137_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__7));
lean_inc_n(v___x_134_, 23);
v___x_138_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_134_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__9));
v___x_140_ = lean_obj_once(&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__11, &lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__11_once, _init_lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__11);
v___x_141_ = lean_box(0);
lean_inc_n(v_currMacroScope_127_, 3);
lean_inc_n(v_quotContext_126_, 3);
v___x_142_ = l_Lean_addMacroScope(v_quotContext_126_, v___x_141_, v_currMacroScope_127_);
v___x_143_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__14));
v___x_144_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_144_, 0, v___x_134_);
lean_ctor_set(v___x_144_, 1, v___x_140_);
lean_ctor_set(v___x_144_, 2, v___x_142_);
lean_ctor_set(v___x_144_, 3, v___x_143_);
v___x_145_ = l_Lean_Syntax_node1(v___x_134_, v___x_139_, v___x_144_);
v___x_146_ = l_Lean_Syntax_node2(v___x_134_, v___x_136_, v___x_138_, v___x_145_);
v___x_147_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__16));
v___x_148_ = lean_obj_once(&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__18, &lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__18_once, _init_lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__18);
v___x_149_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21));
v___x_150_ = l_Lean_addMacroScope(v_quotContext_126_, v___x_149_, v_currMacroScope_127_);
v___x_151_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__23));
v___x_152_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_152_, 0, v___x_134_);
lean_ctor_set(v___x_152_, 1, v___x_148_);
lean_ctor_set(v___x_152_, 2, v___x_150_);
lean_ctor_set(v___x_152_, 3, v___x_151_);
v___x_153_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__24));
v___x_154_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_134_);
lean_ctor_set(v___x_154_, 1, v___x_153_);
v___x_155_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__25));
v___x_156_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_156_, 0, v___x_134_);
lean_ctor_set(v___x_156_, 1, v___x_155_);
v___x_157_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__27));
v___x_158_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__29));
v___x_159_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__30));
v___x_160_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_134_);
lean_ctor_set(v___x_160_, 1, v___x_159_);
lean_inc(v___x_130_);
v___x_161_ = l_Lean_Syntax_node2(v___x_134_, v___x_158_, v___x_160_, v___x_130_);
v___x_162_ = l_Lean_Syntax_node1(v___x_134_, v___x_157_, v___x_161_);
v___x_163_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__31));
v___x_164_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_134_);
lean_ctor_set(v___x_164_, 1, v___x_163_);
lean_inc_ref_n(v___x_164_, 2);
lean_inc_ref_n(v___x_156_, 2);
lean_inc_n(v___x_146_, 2);
v___x_165_ = l_Lean_Syntax_node5(v___x_134_, v___x_135_, v___x_146_, v___x_132_, v___x_156_, v___x_162_, v___x_164_);
v___x_166_ = l_Lean_Syntax_node3(v___x_134_, v___x_147_, v___x_152_, v___x_154_, v___x_165_);
v___x_167_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__33));
v___x_168_ = lean_obj_once(&lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__34, &lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__34_once, _init_lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__34);
v___x_169_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35));
v___x_170_ = l_Lean_addMacroScope(v_quotContext_126_, v___x_169_, v_currMacroScope_127_);
v___x_171_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__39));
v___x_172_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_172_, 0, v___x_134_);
lean_ctor_set(v___x_172_, 1, v___x_168_);
lean_ctor_set(v___x_172_, 2, v___x_170_);
lean_ctor_set(v___x_172_, 3, v___x_171_);
v___x_173_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__41));
v___x_174_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__42));
v___x_175_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_134_);
lean_ctor_set(v___x_175_, 1, v___x_174_);
v___x_176_ = l_Lean_Syntax_node1(v___x_134_, v___x_173_, v___x_175_);
v___x_177_ = l_Lean_Syntax_node1(v___x_134_, v___x_157_, v___x_176_);
lean_inc_ref(v___x_172_);
v___x_178_ = l_Lean_Syntax_node2(v___x_134_, v___x_167_, v___x_172_, v___x_177_);
v___x_179_ = l_Lean_Syntax_node1(v___x_134_, v___x_157_, v___x_178_);
v___x_180_ = l_Lean_Syntax_node5(v___x_134_, v___x_135_, v___x_146_, v___x_130_, v___x_156_, v___x_179_, v___x_164_);
v___x_181_ = l_Lean_Syntax_node1(v___x_134_, v___x_157_, v___x_180_);
v___x_182_ = l_Lean_Syntax_node2(v___x_134_, v___x_167_, v___x_172_, v___x_181_);
v___x_183_ = l_Lean_Syntax_node1(v___x_134_, v___x_157_, v___x_182_);
v___x_184_ = l_Lean_Syntax_node5(v___x_134_, v___x_135_, v___x_146_, v___x_166_, v___x_156_, v___x_183_, v___x_164_);
v___x_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
lean_ctor_set(v___x_185_, 1, v_a_121_);
return v___x_185_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___boxed(lean_object* v_x_186_, lean_object* v_a_187_, lean_object* v_a_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1(v_x_186_, v_a_187_, v_a_188_);
lean_dec_ref(v_a_187_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(lean_object* v___y_190_){
_start:
{
lean_object* v_subExpr_192_; lean_object* v_expr_193_; lean_object* v___x_194_; 
v_subExpr_192_ = lean_ctor_get(v___y_190_, 3);
v_expr_193_ = lean_ctor_get(v_subExpr_192_, 0);
lean_inc_ref(v_expr_193_);
v___x_194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_194_, 0, v_expr_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg___boxed(lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(v___y_195_);
lean_dec_ref(v___y_195_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0(lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(v___y_198_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___boxed(lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0(v___y_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
lean_dec(v___y_209_);
lean_dec_ref(v___y_208_);
lean_dec(v___y_207_);
lean_dec_ref(v___y_206_);
return v_res_213_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0(lean_object* v_x_219_){
_start:
{
lean_object* v___x_220_; uint8_t v___x_221_; 
v___x_220_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___closed__2));
v___x_221_ = l_Lean_Expr_isConstOf(v_x_219_, v___x_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0___boxed(lean_object* v_x_222_){
_start:
{
uint8_t v_res_223_; lean_object* v_r_224_; 
v_res_223_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__0(v_x_222_);
lean_dec_ref(v_x_222_);
v_r_224_ = lean_box(v_res_223_);
return v_r_224_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__1(lean_object* v_x_225_){
_start:
{
lean_object* v___x_226_; uint8_t v___x_227_; 
v___x_226_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__35));
v___x_227_ = l_Lean_Expr_isConstOf(v_x_225_, v___x_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__1___boxed(lean_object* v_x_228_){
_start:
{
uint8_t v_res_229_; lean_object* v_r_230_; 
v_res_229_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__1(v_x_228_);
lean_dec_ref(v_x_228_);
v_r_230_ = lean_box(v_res_229_);
return v_r_230_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__2(lean_object* v_x_231_){
_start:
{
lean_object* v___x_232_; uint8_t v___x_233_; 
v___x_232_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21));
v___x_233_ = l_Lean_Expr_isConstOf(v_x_231_, v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__2___boxed(lean_object* v_x_234_){
_start:
{
uint8_t v_res_235_; lean_object* v_r_236_; 
v_res_235_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__2(v_x_234_);
lean_dec_ref(v_x_234_);
v_r_236_ = lean_box(v_res_235_);
return v_r_236_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3(lean_object* v_x_241_){
_start:
{
lean_object* v___x_242_; uint8_t v___x_243_; 
v___x_242_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___closed__1));
v___x_243_ = l_Lean_Expr_isConstOf(v_x_241_, v___x_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3___boxed(lean_object* v_x_244_){
_start:
{
uint8_t v_res_245_; lean_object* v_r_246_; 
v_res_245_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__3(v_x_244_);
lean_dec_ref(v_x_244_);
v_r_246_ = lean_box(v_res_245_);
return v_r_246_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4(lean_object* v_x_251_){
_start:
{
lean_object* v___x_252_; uint8_t v___x_253_; 
v___x_252_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___closed__1));
v___x_253_ = l_Lean_Expr_isConstOf(v_x_251_, v___x_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4___boxed(lean_object* v_x_254_){
_start:
{
uint8_t v_res_255_; lean_object* v_r_256_; 
v_res_255_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__4(v_x_254_);
lean_dec_ref(v_x_254_);
v_r_256_ = lean_box(v_res_255_);
return v_r_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__5(lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v___y_257_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__5___boxed(lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__5(v___y_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_);
lean_dec(v___y_272_);
lean_dec_ref(v___y_271_);
lean_dec(v___y_270_);
lean_dec_ref(v___y_269_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
return v_res_274_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__6(lean_object* v_n_275_, lean_object* v_x_276_){
_start:
{
uint8_t v___x_277_; 
v___x_277_ = lean_expr_eqv(v_x_276_, v_n_275_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__6___boxed(lean_object* v_n_278_, lean_object* v_x_279_){
_start:
{
uint8_t v_res_280_; lean_object* v_r_281_; 
v_res_280_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__6(v_n_278_, v_x_279_);
lean_dec_ref(v_x_279_);
lean_dec_ref(v_n_278_);
v_r_281_ = lean_box(v_res_280_);
return v_r_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__7(lean_object* v___f_282_, lean_object* v___f_283_, lean_object* v___f_284_, lean_object* v___x_285_, lean_object* v_n_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
lean_object* v___f_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__6___boxed), 2, 1);
lean_closure_set(v___f_295_, 0, v_n_286_);
v___x_296_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_296_, 0, v___f_282_);
lean_inc_ref_n(v___f_283_, 2);
v___x_297_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_297_, 0, v___x_296_);
lean_closure_set(v___x_297_, 1, v___f_283_);
v___x_298_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_298_, 0, v___f_284_);
v___x_299_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_299_, 0, v___x_298_);
lean_closure_set(v___x_299_, 1, v___f_283_);
v___x_300_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_300_, 0, v___x_297_);
lean_closure_set(v___x_300_, 1, v___x_299_);
v___x_301_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_301_, 0, v___x_300_);
lean_closure_set(v___x_301_, 1, v___f_283_);
v___x_302_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_302_, 0, v___x_301_);
lean_closure_set(v___x_302_, 1, v___x_285_);
v___x_303_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_303_, 0, v___f_295_);
v___x_304_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_302_, v___x_303_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__7___boxed(lean_object* v___f_305_, lean_object* v___f_306_, lean_object* v___f_307_, lean_object* v___x_308_, lean_object* v_n_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__7(v___f_305_, v___f_306_, v___f_307_, v___x_308_, v_n_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_);
lean_dec(v___y_316_);
lean_dec_ref(v___y_315_);
lean_dec(v___y_314_);
lean_dec_ref(v___y_313_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__8(lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
lean_object* v_ref_328_; uint8_t v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v_ref_328_ = lean_ctor_get(v___y_325_, 5);
v___x_329_ = 0;
v___x_330_ = l_Lean_SourceInfo_fromRef(v_ref_328_, v___x_329_);
v___x_331_ = ((lean_object*)(lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__3));
v___x_332_ = ((lean_object*)(lp_mathlib_Set_Notation_term___u2193_u2229___00__closed__6));
lean_inc(v___x_330_);
v___x_333_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_330_);
lean_ctor_set(v___x_333_, 1, v___x_332_);
v___x_334_ = l_Lean_Syntax_node3(v___x_330_, v___x_331_, v_a_319_, v___x_333_, v_a_320_);
v___x_335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_335_, 0, v___x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__8___boxed(lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__8(v_a_336_, v_a_337_, v___y_338_, v___y_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
lean_dec(v___y_343_);
lean_dec_ref(v___y_342_);
lean_dec(v___y_341_);
lean_dec_ref(v___y_340_);
lean_dec(v___y_339_);
lean_dec_ref(v___y_338_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9(lean_object* v___f_356_, lean_object* v___f_357_, lean_object* v___f_358_, lean_object* v___f_359_, lean_object* v___f_360_, lean_object* v___f_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v___x_369_; lean_object* v_a_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_410_; 
v___x_369_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(v___y_362_);
v_a_370_ = lean_ctor_get(v___x_369_, 0);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_369_);
if (v_isSharedCheck_410_ == 0)
{
v___x_372_ = v___x_369_;
v_isShared_373_ = v_isSharedCheck_410_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_a_370_);
lean_dec(v___x_369_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_410_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___f_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_374_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_374_, 0, v___f_356_);
v___x_375_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_375_, 0, v___f_357_);
lean_inc_ref_n(v___f_358_, 4);
v___x_376_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_376_, 0, v___x_375_);
lean_closure_set(v___x_376_, 1, v___f_358_);
v___x_377_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__1));
v___x_378_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__2));
v___f_379_ = lean_alloc_closure((void*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__7___boxed), 13, 4);
lean_closure_set(v___f_379_, 0, v___f_359_);
lean_closure_set(v___f_379_, 1, v___f_358_);
lean_closure_set(v___f_379_, 2, v___f_360_);
lean_closure_set(v___f_379_, 3, v___x_378_);
v___x_380_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_380_, 0, v___x_376_);
lean_closure_set(v___x_380_, 1, v___x_378_);
v___x_381_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_381_, 0, v___x_374_);
lean_closure_set(v___x_381_, 1, v___x_380_);
v___x_382_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_382_, 0, v___x_381_);
lean_closure_set(v___x_382_, 1, v___f_358_);
v___x_383_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_383_, 0, v___f_361_);
v___x_384_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_384_, 0, v___x_383_);
lean_closure_set(v___x_384_, 1, v___f_358_);
v___x_385_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_385_, 0, v___f_358_);
lean_closure_set(v___x_385_, 1, v___f_379_);
v___x_386_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_386_, 0, v___x_384_);
lean_closure_set(v___x_386_, 1, v___x_385_);
v___x_387_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_387_, 0, v___x_382_);
lean_closure_set(v___x_387_, 1, v___x_386_);
v___x_388_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__4));
v___x_389_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___closed__5));
v___x_390_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_391_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_387_, v___x_389_, v___x_390_, v___y_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
if (lean_obj_tag(v___x_391_) == 0)
{
lean_object* v_a_392_; lean_object* v___x_394_; 
v_a_392_ = lean_ctor_get(v___x_391_, 0);
lean_inc(v_a_392_);
lean_dec_ref_known(v___x_391_, 1);
if (v_isShared_373_ == 0)
{
lean_ctor_set_tag(v___x_372_, 1);
v___x_394_ = v___x_372_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_a_370_);
v___x_394_ = v_reuseFailAlloc_401_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
lean_object* v___x_395_; 
lean_inc_ref(v___x_394_);
v___x_395_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_392_, v___x_377_, v___x_394_, v___y_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
if (lean_obj_tag(v___x_395_) == 0)
{
lean_object* v_a_396_; lean_object* v___x_397_; 
v_a_396_ = lean_ctor_get(v___x_395_, 0);
lean_inc(v_a_396_);
lean_dec_ref_known(v___x_395_, 1);
v___x_397_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_392_, v___x_388_, v___x_394_, v___y_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
lean_dec(v_a_392_);
if (lean_obj_tag(v___x_397_) == 0)
{
lean_object* v_a_398_; lean_object* v___f_399_; lean_object* v___x_400_; 
v_a_398_ = lean_ctor_get(v___x_397_, 0);
lean_inc(v_a_398_);
lean_dec_ref_known(v___x_397_, 1);
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__8___boxed), 9, 2);
lean_closure_set(v___f_399_, 0, v_a_396_);
lean_closure_set(v___f_399_, 1, v_a_398_);
v___x_400_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_399_, v___y_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
return v___x_400_;
}
else
{
lean_dec(v_a_396_);
return v___x_397_;
}
}
else
{
lean_dec_ref(v___x_394_);
lean_dec(v_a_392_);
return v___x_395_;
}
}
}
else
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_409_; 
lean_del_object(v___x_372_);
lean_dec(v_a_370_);
v_a_402_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_409_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_409_ == 0)
{
v___x_404_ = v___x_391_;
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_391_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_409_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_407_; 
if (v_isShared_405_ == 0)
{
v___x_407_ = v___x_404_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_408_; 
v_reuseFailAlloc_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_408_, 0, v_a_402_);
v___x_407_ = v_reuseFailAlloc_408_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
return v___x_407_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9___boxed(lean_object* v___f_411_, lean_object* v___f_412_, lean_object* v___f_413_, lean_object* v___f_414_, lean_object* v___f_415_, lean_object* v___f_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___lam__9(v___f_411_, v___f_412_, v___f_413_, v___f_414_, v___f_415_, v___f_416_, v___y_417_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
lean_dec(v___y_418_);
lean_dec_ref(v___y_417_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1(lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_453_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__7));
v___x_454_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___closed__10));
v___x_455_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_453_, v___x_454_, v_a_446_, v_a_447_, v_a_448_, v_a_449_, v_a_450_, v_a_451_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1___boxed(lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1(v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_);
lean_dec(v_a_461_);
lean_dec_ref(v_a_460_);
lean_dec(v_a_459_);
lean_dec_ref(v_a_458_);
lean_dec(v_a_457_);
lean_dec_ref(v_a_456_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_instCoeHeadElem(lean_object* v_00_u03b1_464_, lean_object* v_s_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lean_box(0);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg(lean_object* v_child_467_, lean_object* v_childIdx_468_, lean_object* v_x_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
lean_object* v_subExpr_477_; lean_object* v_optionsPerPos_478_; lean_object* v_currNamespace_479_; lean_object* v_openDecls_480_; uint8_t v_inPattern_481_; lean_object* v_depth_482_; lean_object* v_lctxInitIndices_483_; lean_object* v_pos_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v_subExpr_477_ = lean_ctor_get(v___y_470_, 3);
v_optionsPerPos_478_ = lean_ctor_get(v___y_470_, 0);
v_currNamespace_479_ = lean_ctor_get(v___y_470_, 1);
v_openDecls_480_ = lean_ctor_get(v___y_470_, 2);
v_inPattern_481_ = lean_ctor_get_uint8(v___y_470_, sizeof(void*)*6);
v_depth_482_ = lean_ctor_get(v___y_470_, 4);
v_lctxInitIndices_483_ = lean_ctor_get(v___y_470_, 5);
v_pos_484_ = lean_ctor_get(v_subExpr_477_, 1);
v___x_485_ = l_Lean_SubExpr_Pos_push(v_pos_484_, v_childIdx_468_);
v___x_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_486_, 0, v_child_467_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
lean_inc(v_lctxInitIndices_483_);
lean_inc(v_depth_482_);
lean_inc(v_openDecls_480_);
lean_inc(v_currNamespace_479_);
lean_inc(v_optionsPerPos_478_);
v___x_487_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_487_, 0, v_optionsPerPos_478_);
lean_ctor_set(v___x_487_, 1, v_currNamespace_479_);
lean_ctor_set(v___x_487_, 2, v_openDecls_480_);
lean_ctor_set(v___x_487_, 3, v___x_486_);
lean_ctor_set(v___x_487_, 4, v_depth_482_);
lean_ctor_set(v___x_487_, 5, v_lctxInitIndices_483_);
lean_ctor_set_uint8(v___x_487_, sizeof(void*)*6, v_inPattern_481_);
lean_inc(v___y_475_);
lean_inc_ref(v___y_474_);
lean_inc(v___y_473_);
lean_inc_ref(v___y_472_);
lean_inc(v___y_471_);
v___x_488_ = lean_apply_7(v_x_469_, v___x_487_, v___y_471_, v___y_472_, v___y_473_, v___y_474_, v___y_475_, lean_box(0));
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg___boxed(lean_object* v_child_489_, lean_object* v_childIdx_490_, lean_object* v_x_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg(v_child_489_, v_childIdx_490_, v_x_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_);
lean_dec(v___y_497_);
lean_dec_ref(v___y_496_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg(lean_object* v_x_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v___x_508_; lean_object* v_a_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_508_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(v___y_501_);
v_a_509_ = lean_ctor_get(v___x_508_, 0);
lean_inc(v_a_509_);
lean_dec_ref(v___x_508_);
v___x_510_ = l_Lean_Expr_appArg_x21(v_a_509_);
lean_dec(v_a_509_);
v___x_511_ = lean_unsigned_to_nat(1u);
v___x_512_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg(v___x_510_, v___x_511_, v_x_500_, v___y_501_, v___y_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg___boxed(lean_object* v_x_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg(v_x_513_, v___y_514_, v___y_515_, v___y_516_, v___y_517_, v___y_518_, v___y_519_);
lean_dec(v___y_519_);
lean_dec_ref(v___y_518_);
lean_dec(v___y_517_);
lean_dec_ref(v___y_516_);
lean_dec(v___y_515_);
lean_dec_ref(v___y_514_);
return v_res_521_;
}
}
static lean_object* _init_lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__0(void){
_start:
{
lean_object* v___x_522_; lean_object* v_dummy_523_; 
v___x_522_ = lean_box(0);
v_dummy_523_ = l_Lean_Expr_sort___override(v___x_522_);
return v_dummy_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0(lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_){
_start:
{
lean_object* v___x_536_; lean_object* v_a_537_; lean_object* v_dummy_538_; lean_object* v_nargs_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; uint8_t v___x_546_; 
v___x_536_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Set_Notation___aux__Mathlib__Data__Set__Notation______delab__app__Set__Notation__term___u2193_u2229____1_spec__0___redArg(v___y_529_);
v_a_537_ = lean_ctor_get(v___x_536_, 0);
lean_inc(v_a_537_);
lean_dec_ref(v___x_536_);
v_dummy_538_ = lean_obj_once(&lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__0, &lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__0_once, _init_lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__0);
v_nargs_539_ = l_Lean_Expr_getAppNumArgs(v_a_537_);
lean_inc(v_nargs_539_);
v___x_540_ = lean_mk_array(v_nargs_539_, v_dummy_538_);
v___x_541_ = lean_unsigned_to_nat(1u);
v___x_542_ = lean_nat_sub(v_nargs_539_, v___x_541_);
lean_dec(v_nargs_539_);
v___x_543_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_537_, v___x_540_, v___x_542_);
v___x_544_ = lean_array_get_size(v___x_543_);
v___x_545_ = lean_unsigned_to_nat(4u);
v___x_546_ = lean_nat_dec_eq(v___x_544_, v___x_545_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; 
lean_dec_ref(v___x_543_);
v___x_547_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_547_;
}
else
{
lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; uint8_t v___x_573_; 
v___x_548_ = lean_unsigned_to_nat(0u);
v___x_549_ = lean_array_fget(v___x_543_, v___x_548_);
v___x_570_ = lean_unsigned_to_nat(2u);
v___x_571_ = lean_array_fget(v___x_543_, v___x_570_);
lean_dec_ref(v___x_543_);
v___x_572_ = ((lean_object*)(lp_mathlib_Set_Notation___aux__Mathlib__Data__Set__Notation______macroRules__Set__Notation__term___u2193_u2229____1___closed__21));
v___x_573_ = l_Lean_Expr_isAppOfArity(v___x_571_, v___x_572_, v___x_570_);
lean_dec(v___x_571_);
if (v___x_573_ == 0)
{
lean_object* v___x_574_; 
v___x_574_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_574_) == 0)
{
lean_dec_ref_known(v___x_574_, 1);
goto v___jp_550_;
}
else
{
lean_object* v_a_575_; lean_object* v___x_577_; uint8_t v_isShared_578_; uint8_t v_isSharedCheck_582_; 
lean_dec(v___x_549_);
v_a_575_ = lean_ctor_get(v___x_574_, 0);
v_isSharedCheck_582_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_582_ == 0)
{
v___x_577_ = v___x_574_;
v_isShared_578_ = v_isSharedCheck_582_;
goto v_resetjp_576_;
}
else
{
lean_inc(v_a_575_);
lean_dec(v___x_574_);
v___x_577_ = lean_box(0);
v_isShared_578_ = v_isSharedCheck_582_;
goto v_resetjp_576_;
}
v_resetjp_576_:
{
lean_object* v___x_580_; 
if (v_isShared_578_ == 0)
{
v___x_580_ = v___x_577_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_581_; 
v_reuseFailAlloc_581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_581_, 0, v_a_575_);
v___x_580_ = v_reuseFailAlloc_581_;
goto v_reusejp_579_;
}
v_reusejp_579_:
{
return v___x_580_;
}
}
}
}
else
{
goto v___jp_550_;
}
v___jp_550_:
{
lean_object* v___x_551_; 
v___x_551_ = lp_mathlib_Lean_Expr_coeTypeSet_x3f(v___x_549_);
lean_dec(v___x_549_);
if (lean_obj_tag(v___x_551_) == 1)
{
lean_object* v___x_552_; lean_object* v___x_553_; 
lean_dec_ref_known(v___x_551_, 1);
v___x_552_ = ((lean_object*)(lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__1));
v___x_553_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg(v___x_552_, v___y_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_);
if (lean_obj_tag(v___x_553_) == 0)
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_568_; 
v_a_554_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_568_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_568_ == 0)
{
v___x_556_ = v___x_553_;
v_isShared_557_ = v_isSharedCheck_568_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_553_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_568_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v_ref_558_; uint8_t v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_566_; 
v_ref_558_ = lean_ctor_get(v___y_533_, 5);
v___x_559_ = 0;
v___x_560_ = l_Lean_SourceInfo_fromRef(v_ref_558_, v___x_559_);
v___x_561_ = ((lean_object*)(lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__3));
v___x_562_ = ((lean_object*)(lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___closed__4));
lean_inc(v___x_560_);
v___x_563_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_563_, 0, v___x_560_);
lean_ctor_set(v___x_563_, 1, v___x_562_);
v___x_564_ = l_Lean_Syntax_node2(v___x_560_, v___x_561_, v___x_563_, v_a_554_);
if (v_isShared_557_ == 0)
{
lean_ctor_set(v___x_556_, 0, v___x_564_);
v___x_566_ = v___x_556_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v___x_564_);
v___x_566_ = v_reuseFailAlloc_567_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
return v___x_566_;
}
}
}
else
{
return v___x_553_;
}
}
else
{
lean_object* v___x_569_; 
lean_dec(v___x_551_);
v___x_569_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_569_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0___boxed(lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_Set_Notation_delabSetImageSubtype___lam__0(v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype(lean_object* v_a_593_, lean_object* v_a_594_, lean_object* v_a_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_){
_start:
{
lean_object* v___f_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v___f_600_ = ((lean_object*)(lp_mathlib_Set_Notation_delabSetImageSubtype___closed__0));
v___x_601_ = ((lean_object*)(lp_mathlib_Set_Notation_delabSetImageSubtype___closed__1));
v___x_602_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_601_, v___f_600_, v_a_593_, v_a_594_, v_a_595_, v_a_596_, v_a_597_, v_a_598_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_Notation_delabSetImageSubtype___boxed(lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v_a_607_, lean_object* v_a_608_, lean_object* v_a_609_){
_start:
{
lean_object* v_res_610_; 
v_res_610_ = lp_mathlib_Set_Notation_delabSetImageSubtype(v_a_603_, v_a_604_, v_a_605_, v_a_606_, v_a_607_, v_a_608_);
lean_dec(v_a_608_);
lean_dec_ref(v_a_607_);
lean_dec(v_a_606_);
lean_dec_ref(v_a_605_);
lean_dec(v_a_604_);
lean_dec_ref(v_a_603_);
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0(lean_object* v_00_u03b1_611_, lean_object* v_child_612_, lean_object* v_childIdx_613_, lean_object* v_x_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_){
_start:
{
lean_object* v___x_622_; 
v___x_622_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___redArg(v_child_612_, v_childIdx_613_, v_x_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_, v___y_620_);
return v___x_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0___boxed(lean_object* v_00_u03b1_623_, lean_object* v_child_624_, lean_object* v_childIdx_625_, lean_object* v_x_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0_spec__0(v_00_u03b1_623_, v_child_624_, v_childIdx_625_, v_x_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_);
lean_dec(v___y_632_);
lean_dec_ref(v___y_631_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec_ref(v___y_627_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0(lean_object* v_00_u03b1_635_, lean_object* v_x_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___redArg(v_x_636_, v___y_637_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0___boxed(lean_object* v_00_u03b1_645_, lean_object* v_x_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Set_Notation_delabSetImageSubtype_spec__0(v_00_u03b1_645_, v_x_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_, v___y_652_);
lean_dec(v___y_652_);
lean_dec_ref(v___y_651_);
lean_dec(v___y_650_);
lean_dec_ref(v___y_649_);
lean_dec(v___y_648_);
lean_dec_ref(v___y_647_);
return v_res_654_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Notation3(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Notation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Notation3(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Notation(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Notation3(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Notation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Notation3(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Notation(builtin);
}
#ifdef __cplusplus
}
#endif
