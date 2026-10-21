// Lean compiler output
// Module: Mathlib.Order.InitialSeg
// Imports: public import Init public meta import Init public import Mathlib.Data.Sum.Order public import Mathlib.Order.Hom.Lex public import Mathlib.Order.RelIso.Set public import Mathlib.Order.UpperLower.Basic public import Mathlib.Order.WellFounded
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchLambda(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchLambda___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_InitialSeg_term___u227ci___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "InitialSeg"};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__0 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__0_value;
static const lean_string_object lp_mathlib_InitialSeg_term___u227ci___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≼i_"};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__1 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__1_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 22, 191, 14, 132, 248, 30, 171)}};
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(124, 71, 79, 74, 5, 153, 8, 179)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__2 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__2_value;
static const lean_string_object lp_mathlib_InitialSeg_term___u227ci___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__3 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__3_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__4 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__4_value;
static const lean_string_object lp_mathlib_InitialSeg_term___u227ci___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≼i "};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__5 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__5_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__5_value)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__6 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__6_value;
static const lean_string_object lp_mathlib_InitialSeg_term___u227ci___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__7 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__7_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__8 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__8_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__8_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__9 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__9_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__4_value),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__6_value),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__9_value)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__10 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__10_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ci___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__2_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__10_value)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ci___00__closed__11 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_InitialSeg_term___u227ci__ = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__11_value;
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value;
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value;
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value;
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__3 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__3_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4_value;
static lean_once_cell_t lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 22, 191, 14, 132, 248, 30, 171)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__7 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__7_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6_value)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__8 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__8_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__9 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__9_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__7_value),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__9_value)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__10 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__10_value;
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__11 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__11_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__0 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__0_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__1 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2264i___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≤i_"};
static const lean_object* lp_mathlib_term___u2264i___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2264i___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2264i___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 232, 75, 43, 117, 254, 72)}};
static const lean_object* lp_mathlib_term___u2264i___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2264i___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≤i "};
static const lean_object* lp_mathlib_term___u2264i___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2264i___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2264i___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2264i___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2264i___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__8_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2264i___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2264i___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__4_value),((lean_object*)&lp_mathlib_term___u2264i___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2264i___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2264i___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u2264i___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2264i___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(24) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2264i___00__closed__5_value)}};
static const lean_object* lp_mathlib_term___u2264i___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2264i__ = (const lean_object*)&lp_mathlib_term___u2264i___00__closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__10_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__12_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__13_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__14_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__14_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__15_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__16_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__12_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__16_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__17_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_<_"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__18_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(192, 242, 106, 74, 199, 131, 133, 95)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__19_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value_aux_1),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(215, 94, 65, 66, 49, 100, 151, 85)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__22_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__23_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__24 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "β"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__3_value),LEAN_SCALAR_PTR_LITERAL(163, 67, 89, 131, 111, 186, 232, 248)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__4_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__4_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__3___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___boxed, .m_arity = 12, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__3_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__4_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__5_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__3_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__6_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__5_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__6_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_instCoeRelEmbedding___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_InitialSeg_instCoeRelEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_InitialSeg_instCoeRelEmbedding___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_InitialSeg_instCoeRelEmbedding___closed__0 = (const lean_object*)&lp_mathlib_InitialSeg_instCoeRelEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_instCoeRelEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toInitialSeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toInitialSeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_InitialSeg_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_InitialSeg_refl___closed__0;
static lean_once_cell_t lp_mathlib_InitialSeg_refl___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_InitialSeg_refl___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_refl(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_InitialSeg_instInhabited___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_InitialSeg_instInhabited___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_InitialSeg_ofIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_InitialSeg_ofIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_InitialSeg_ofIsEmpty___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_ofIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_leAdd___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_InitialSeg_leAdd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_InitialSeg_leAdd___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_InitialSeg_leAdd___closed__0 = (const lean_object*)&lp_mathlib_InitialSeg_leAdd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_leAdd(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_InitialSeg_term___u227ai___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≺i_"};
static const lean_object* lp_mathlib_InitialSeg_term___u227ai___00__closed__0 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__0_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ai___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 22, 191, 14, 132, 248, 30, 171)}};
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ai___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(189, 135, 138, 17, 131, 186, 117, 159)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ai___00__closed__1 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__1_value;
static const lean_string_object lp_mathlib_InitialSeg_term___u227ai___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≺i "};
static const lean_object* lp_mathlib_InitialSeg_term___u227ai___00__closed__2 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__2_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ai___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__2_value)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ai___00__closed__3 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__3_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ai___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__4_value),((lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__3_value),((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__9_value)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ai___00__closed__4 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__4_value;
static const lean_ctor_object lp_mathlib_InitialSeg_term___u227ai___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__4_value)}};
static const lean_object* lp_mathlib_InitialSeg_term___u227ai___00__closed__5 = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_InitialSeg_term___u227ai__ = (const lean_object*)&lp_mathlib_InitialSeg_term___u227ai___00__closed__5_value;
static const lean_string_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "PrincipalSeg"};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__0 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__0_value;
static lean_once_cell_t lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 43, 12, 152, 215, 32, 180, 50)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__3 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__3_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2_value)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__4 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__4_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__5 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__5_value;
static const lean_ctor_object lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__3_value),((lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__5_value)}};
static const lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__6 = (const lean_object*)&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__PrincipalSeg__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__PrincipalSeg__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___x3ci___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_<i_"};
static const lean_object* lp_mathlib_term___x3ci___00__closed__0 = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___x3ci___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x3ci___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 1, 28, 111, 182, 213, 236, 255)}};
static const lean_object* lp_mathlib_term___x3ci___00__closed__1 = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__1_value;
static const lean_string_object lp_mathlib_term___x3ci___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " <i "};
static const lean_object* lp_mathlib_term___x3ci___00__closed__2 = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___x3ci___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___x3ci___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___x3ci___00__closed__3 = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___x3ci___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_InitialSeg_term___u227ci___00__closed__4_value),((lean_object*)&lp_mathlib_term___x3ci___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2264i___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___x3ci___00__closed__4 = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___x3ci___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___x3ci___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(24) << 1) | 1)),((lean_object*)&lp_mathlib_term___x3ci___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___x3ci___00__closed__5 = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___x3ci__ = (const lean_object*)&lp_mathlib_term___x3ci___00__closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___x3ci____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___x3ci____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__1___boxed, .m_arity = 12, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__1_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__5_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___closed__0 = (const lean_object*)&lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeFunForall___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PrincipalSeg_instCoeFunForall___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PrincipalSeg_instCoeFunForall___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PrincipalSeg_instCoeFunForall___closed__0 = (const lean_object*)&lp_mathlib_PrincipalSeg_instCoeFunForall___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeFunForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_hasCoeInitialSeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transInitial___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transInitial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_relIsoTrans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_relIsoTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transRelIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transRelIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PrincipalSeg_ofElement___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PrincipalSeg_ofElement___redArg___closed__0 = (const lean_object*)&lp_mathlib_PrincipalSeg_ofElement___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofElement___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofElement(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofIsEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PrincipalSeg_pemptyToPUnit___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PrincipalSeg_pemptyToPUnit___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_pemptyToPUnit;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toInitialSeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toInitialSeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toInitialSeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ci___00__closed__0));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ci___00__closed__2));
lean_inc(v_x_54_);
v___x_58_ = l_Lean_Syntax_isOfKind(v_x_54_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec(v_x_54_);
v___x_59_ = lean_box(1);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v_a_56_);
return v___x_60_;
}
else
{
lean_object* v_quotContext_61_; lean_object* v_currMacroScope_62_; lean_object* v_ref_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_quotContext_61_ = lean_ctor_get(v_a_55_, 1);
v_currMacroScope_62_ = lean_ctor_get(v_a_55_, 2);
v_ref_63_ = lean_ctor_get(v_a_55_, 5);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = l_Lean_Syntax_getArg(v_x_54_, v___x_64_);
v___x_66_ = lean_unsigned_to_nat(2u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_54_, v___x_66_);
lean_dec(v_x_54_);
v___x_68_ = 0;
v___x_69_ = l_Lean_SourceInfo_fromRef(v_ref_63_, v___x_68_);
v___x_70_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5, &lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5_once, _init_lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5);
v___x_72_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__10));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4));
lean_inc(v_x_87_);
v___x_91_ = l_Lean_Syntax_isOfKind(v_x_87_, v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_93_; 
lean_dec(v_x_87_);
v___x_92_ = lean_box(0);
v___x_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_89_);
return v___x_93_;
}
else
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = l_Lean_Syntax_getArg(v_x_87_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__1));
lean_inc(v___x_95_);
v___x_97_ = l_Lean_Syntax_isOfKind(v___x_95_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v___x_95_);
lean_dec(v_x_87_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_89_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(1u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_87_, v___x_100_);
lean_dec(v_x_87_);
v___x_102_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_101_);
v___x_103_ = l_Lean_Syntax_matchesNull(v___x_101_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v___x_101_);
lean_dec(v___x_95_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_89_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v_ref_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_106_ = l_Lean_Syntax_getArg(v___x_101_, v___x_94_);
v___x_107_ = l_Lean_Syntax_getArg(v___x_101_, v___x_100_);
lean_dec(v___x_101_);
v_ref_108_ = l_Lean_replaceRef(v___x_95_, v_a_88_);
lean_dec(v___x_95_);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_109_);
lean_dec(v_ref_108_);
v___x_111_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ci___00__closed__2));
v___x_112_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ci___00__closed__5));
lean_inc(v___x_110_);
v___x_113_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v___x_114_ = l_Lean_Syntax_node3(v___x_110_, v___x_111_, v___x_106_, v___x_113_, v___x_107_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_89_);
return v___x_115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__10));
v___x_164_ = l_String_toRawSubstring_x27(v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1(lean_object* v_x_190_, lean_object* v_a_191_, lean_object* v_a_192_){
_start:
{
lean_object* v___x_193_; uint8_t v___x_194_; 
v___x_193_ = ((lean_object*)(lp_mathlib_term___u2264i___00__closed__1));
lean_inc(v_x_190_);
v___x_194_ = l_Lean_Syntax_isOfKind(v_x_190_, v___x_193_);
if (v___x_194_ == 0)
{
lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec(v_x_190_);
v___x_195_ = lean_box(1);
v___x_196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
lean_ctor_set(v___x_196_, 1, v_a_192_);
return v___x_196_;
}
else
{
lean_object* v_quotContext_197_; lean_object* v_currMacroScope_198_; lean_object* v_ref_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; uint8_t v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v_quotContext_197_ = lean_ctor_get(v_a_191_, 1);
v_currMacroScope_198_ = lean_ctor_get(v_a_191_, 2);
v_ref_199_ = lean_ctor_get(v_a_191_, 5);
v___x_200_ = lean_unsigned_to_nat(0u);
v___x_201_ = l_Lean_Syntax_getArg(v_x_190_, v___x_200_);
v___x_202_ = lean_unsigned_to_nat(2u);
v___x_203_ = l_Lean_Syntax_getArg(v_x_190_, v___x_202_);
lean_dec(v_x_190_);
v___x_204_ = 0;
v___x_205_ = l_Lean_SourceInfo_fromRef(v_ref_199_, v___x_204_);
v___x_206_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4));
v___x_207_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1));
v___x_208_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__2));
lean_inc_n(v___x_205_, 14);
v___x_209_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_205_);
lean_ctor_set(v___x_209_, 1, v___x_208_);
v___x_210_ = lean_obj_once(&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5, &lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5_once, _init_lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__5);
v___x_211_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6));
lean_inc_n(v_currMacroScope_198_, 2);
lean_inc_n(v_quotContext_197_, 2);
v___x_212_ = l_Lean_addMacroScope(v_quotContext_197_, v___x_211_, v_currMacroScope_198_);
v___x_213_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__10));
v___x_214_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_214_, 0, v___x_205_);
lean_ctor_set(v___x_214_, 1, v___x_210_);
lean_ctor_set(v___x_214_, 2, v___x_212_);
lean_ctor_set(v___x_214_, 3, v___x_213_);
v___x_215_ = l_Lean_Syntax_node2(v___x_205_, v___x_207_, v___x_209_, v___x_214_);
v___x_216_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12));
v___x_217_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4));
v___x_218_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6));
v___x_219_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__7));
v___x_220_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_205_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
v___x_221_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__9));
v___x_222_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11, &lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11_once, _init_lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11);
v___x_223_ = lean_box(0);
v___x_224_ = l_Lean_addMacroScope(v_quotContext_197_, v___x_223_, v_currMacroScope_198_);
v___x_225_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__17));
v___x_226_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_226_, 0, v___x_205_);
lean_ctor_set(v___x_226_, 1, v___x_222_);
lean_ctor_set(v___x_226_, 2, v___x_224_);
lean_ctor_set(v___x_226_, 3, v___x_225_);
v___x_227_ = l_Lean_Syntax_node1(v___x_205_, v___x_221_, v___x_226_);
lean_inc(v___x_227_);
v___x_228_ = l_Lean_Syntax_node2(v___x_205_, v___x_218_, v___x_220_, v___x_227_);
v___x_229_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__19));
v___x_230_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21));
v___x_231_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__22));
v___x_232_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_205_);
lean_ctor_set(v___x_232_, 1, v___x_231_);
v___x_233_ = l_Lean_Syntax_node2(v___x_205_, v___x_230_, v___x_232_, v___x_227_);
v___x_234_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__23));
v___x_235_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_205_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
lean_inc(v___x_233_);
v___x_236_ = l_Lean_Syntax_node3(v___x_205_, v___x_229_, v___x_233_, v___x_235_, v___x_233_);
v___x_237_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__24));
v___x_238_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_205_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = l_Lean_Syntax_node3(v___x_205_, v___x_217_, v___x_228_, v___x_236_, v___x_238_);
lean_inc(v___x_239_);
v___x_240_ = l_Lean_Syntax_node4(v___x_205_, v___x_216_, v___x_201_, v___x_203_, v___x_239_, v___x_239_);
v___x_241_ = l_Lean_Syntax_node2(v___x_205_, v___x_206_, v___x_215_, v___x_240_);
v___x_242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
lean_ctor_set(v___x_242_, 1, v_a_192_);
return v___x_242_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___boxed(lean_object* v_x_243_, lean_object* v_a_244_, lean_object* v_a_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1(v_x_243_, v_a_244_, v_a_245_);
lean_dec_ref(v_a_244_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg(lean_object* v___y_247_){
_start:
{
lean_object* v_subExpr_249_; lean_object* v_expr_250_; lean_object* v___x_251_; 
v_subExpr_249_ = lean_ctor_get(v___y_247_, 3);
v_expr_250_ = lean_ctor_get(v_subExpr_249_, 0);
lean_inc_ref(v_expr_250_);
v___x_251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_251_, 0, v_expr_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg___boxed(lean_object* v___y_252_, lean_object* v___y_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg(v___y_252_);
lean_dec_ref(v___y_252_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0(lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg(v___y_255_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___boxed(lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0(v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
return v_res_270_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__0(lean_object* v_x_271_){
_start:
{
lean_object* v___x_272_; uint8_t v___x_273_; 
v___x_272_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__6));
v___x_273_ = l_Lean_Expr_isConstOf(v_x_271_, v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__0___boxed(lean_object* v_x_274_){
_start:
{
uint8_t v_res_275_; lean_object* v_r_276_; 
v_res_275_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__0(v_x_274_);
lean_dec_ref(v_x_274_);
v_r_276_ = lean_box(v_res_275_);
return v_r_276_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1(lean_object* v_x_282_){
_start:
{
lean_object* v___x_283_; uint8_t v___x_284_; 
v___x_283_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___closed__2));
v___x_284_ = l_Lean_Expr_isConstOf(v_x_282_, v___x_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1___boxed(lean_object* v_x_285_){
_start:
{
uint8_t v_res_286_; lean_object* v_r_287_; 
v_res_286_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__1(v_x_285_);
lean_dec_ref(v_x_285_);
v_r_287_ = lean_box(v_res_286_);
return v_r_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__3(lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_296_, 0, v___y_288_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__3___boxed(lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__3(v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
return v_res_305_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4(lean_object* v_n_306_, lean_object* v_x_307_){
_start:
{
uint8_t v___x_308_; 
v___x_308_ = lean_expr_eqv(v_x_307_, v_n_306_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4___boxed(lean_object* v_n_309_, lean_object* v_x_310_){
_start:
{
uint8_t v_res_311_; lean_object* v_r_312_; 
v_res_311_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4(v_n_309_, v_x_310_);
lean_dec_ref(v_x_310_);
lean_dec_ref(v_n_309_);
v_r_312_ = lean_box(v_res_311_);
return v_r_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__5(lean_object* v___f_313_, lean_object* v___x_314_, lean_object* v___f_315_, lean_object* v___f_316_, lean_object* v_n_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_){
_start:
{
lean_object* v___f_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___f_326_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4___boxed), 2, 1);
lean_closure_set(v___f_326_, 0, v_n_317_);
v___x_327_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_327_, 0, v___f_313_);
v___x_328_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_328_, 0, v___x_327_);
lean_closure_set(v___x_328_, 1, v___x_314_);
v___x_329_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_329_, 0, v___x_328_);
lean_closure_set(v___x_329_, 1, v___f_315_);
v___x_330_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_330_, 0, v___f_316_);
v___x_331_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_331_, 0, v___x_329_);
lean_closure_set(v___x_331_, 1, v___x_330_);
v___x_332_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_332_, 0, v___f_326_);
v___x_333_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_331_, v___x_332_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__5___boxed(lean_object* v___f_334_, lean_object* v___x_335_, lean_object* v___f_336_, lean_object* v___f_337_, lean_object* v_n_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__5(v___f_334_, v___x_335_, v___f_336_, v___f_337_, v_n_338_, v___y_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_);
lean_dec(v___y_345_);
lean_dec_ref(v___y_344_);
lean_dec(v___y_343_);
lean_dec_ref(v___y_342_);
lean_dec(v___y_341_);
lean_dec_ref(v___y_340_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2(lean_object* v___f_348_, lean_object* v___x_349_, lean_object* v___f_350_, lean_object* v_n_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_){
_start:
{
lean_object* v___f_360_; lean_object* v___f_361_; lean_object* v___x_362_; 
v___f_360_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__4___boxed), 2, 1);
lean_closure_set(v___f_360_, 0, v_n_351_);
lean_inc_ref(v___x_349_);
v___f_361_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__5___boxed), 13, 4);
lean_closure_set(v___f_361_, 0, v___f_348_);
lean_closure_set(v___f_361_, 1, v___x_349_);
lean_closure_set(v___f_361_, 2, v___f_350_);
lean_closure_set(v___f_361_, 3, v___f_360_);
v___x_362_ = lp_mathlib_Mathlib_Notation3_matchLambda(v___x_349_, v___f_361_, v___y_352_, v___y_353_, v___y_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2___boxed(lean_object* v___f_363_, lean_object* v___x_364_, lean_object* v___f_365_, lean_object* v_n_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2(v___f_363_, v___x_364_, v___f_365_, v_n_366_, v___y_367_, v___y_368_, v___y_369_, v___y_370_, v___y_371_, v___y_372_, v___y_373_);
lean_dec(v___y_373_);
lean_dec_ref(v___y_372_);
lean_dec(v___y_371_);
lean_dec_ref(v___y_370_);
lean_dec(v___y_369_);
lean_dec_ref(v___y_368_);
return v_res_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__10(lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_){
_start:
{
lean_object* v_ref_385_; uint8_t v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v_ref_385_ = lean_ctor_get(v___y_382_, 5);
v___x_386_ = 0;
v___x_387_ = l_Lean_SourceInfo_fromRef(v_ref_385_, v___x_386_);
v___x_388_ = ((lean_object*)(lp_mathlib_term___u2264i___00__closed__1));
v___x_389_ = ((lean_object*)(lp_mathlib_term___u2264i___00__closed__2));
lean_inc(v___x_387_);
v___x_390_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_387_);
lean_ctor_set(v___x_390_, 1, v___x_389_);
v___x_391_ = l_Lean_Syntax_node3(v___x_387_, v___x_388_, v_a_376_, v___x_390_, v_a_377_);
v___x_392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__10___boxed(lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__10(v_a_393_, v_a_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_);
lean_dec(v___y_400_);
lean_dec_ref(v___y_399_);
lean_dec(v___y_398_);
lean_dec_ref(v___y_397_);
lean_dec(v___y_396_);
lean_dec_ref(v___y_395_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6(lean_object* v___f_413_, lean_object* v___f_414_, lean_object* v___f_415_, lean_object* v___f_416_, lean_object* v___f_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v___x_425_; lean_object* v_a_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_462_; 
v___x_425_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg(v___y_418_);
v_a_426_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_462_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_462_ == 0)
{
v___x_428_ = v___x_425_;
v_isShared_429_ = v_isSharedCheck_462_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_a_426_);
lean_dec(v___x_425_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_462_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___f_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___f_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_430_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_430_, 0, v___f_413_);
v___x_431_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__1));
v___x_432_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__2));
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2___boxed), 12, 3);
lean_closure_set(v___f_433_, 0, v___f_414_);
lean_closure_set(v___f_433_, 1, v___x_432_);
lean_closure_set(v___f_433_, 2, v___f_415_);
v___x_434_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_434_, 0, v___x_430_);
lean_closure_set(v___x_434_, 1, v___x_432_);
v___x_435_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__4));
v___x_436_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__5));
v___f_437_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2___boxed), 12, 3);
lean_closure_set(v___f_437_, 0, v___f_416_);
lean_closure_set(v___f_437_, 1, v___x_436_);
lean_closure_set(v___f_437_, 2, v___f_417_);
v___x_438_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_438_, 0, v___x_434_);
lean_closure_set(v___x_438_, 1, v___x_436_);
v___x_439_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_439_, 0, v___x_432_);
lean_closure_set(v___x_439_, 1, v___f_433_);
v___x_440_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_440_, 0, v___x_438_);
lean_closure_set(v___x_440_, 1, v___x_439_);
v___x_441_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_441_, 0, v___x_436_);
lean_closure_set(v___x_441_, 1, v___f_437_);
v___x_442_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_443_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_440_, v___x_441_, v___x_442_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_);
if (lean_obj_tag(v___x_443_) == 0)
{
lean_object* v_a_444_; lean_object* v___x_446_; 
v_a_444_ = lean_ctor_get(v___x_443_, 0);
lean_inc(v_a_444_);
lean_dec_ref_known(v___x_443_, 1);
if (v_isShared_429_ == 0)
{
lean_ctor_set_tag(v___x_428_, 1);
v___x_446_ = v___x_428_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_a_426_);
v___x_446_ = v_reuseFailAlloc_453_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
lean_object* v___x_447_; 
lean_inc_ref(v___x_446_);
v___x_447_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_444_, v___x_435_, v___x_446_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v_a_448_; lean_object* v___x_449_; 
v_a_448_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_a_448_);
lean_dec_ref_known(v___x_447_, 1);
v___x_449_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_444_, v___x_431_, v___x_446_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_);
lean_dec(v_a_444_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; lean_object* v___f_451_; lean_object* v___x_452_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref_known(v___x_449_, 1);
v___f_451_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__10___boxed), 9, 2);
lean_closure_set(v___f_451_, 0, v_a_450_);
lean_closure_set(v___f_451_, 1, v_a_448_);
v___x_452_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_451_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_);
return v___x_452_;
}
else
{
lean_dec(v_a_448_);
return v___x_449_;
}
}
else
{
lean_dec_ref(v___x_446_);
lean_dec(v_a_444_);
return v___x_447_;
}
}
}
else
{
lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_461_; 
lean_del_object(v___x_428_);
lean_dec(v_a_426_);
v_a_454_ = lean_ctor_get(v___x_443_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_443_);
if (v_isSharedCheck_461_ == 0)
{
v___x_456_ = v___x_443_;
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_443_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___x_459_; 
if (v_isShared_457_ == 0)
{
v___x_459_ = v___x_456_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v_a_454_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___boxed(lean_object* v___f_463_, lean_object* v___f_464_, lean_object* v___f_465_, lean_object* v___f_466_, lean_object* v___f_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v_res_475_; 
v_res_475_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6(v___f_463_, v___f_464_, v___f_465_, v___f_466_, v___f_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1(lean_object* v_a_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_498_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__4));
v___x_499_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__7));
v___x_500_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_498_, v___x_499_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___boxed(lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1(v_a_501_, v_a_502_, v_a_503_, v_a_504_, v_a_505_, v_a_506_);
lean_dec(v_a_506_);
lean_dec_ref(v_a_505_);
lean_dec(v_a_504_);
lean_dec_ref(v_a_503_);
lean_dec(v_a_502_);
lean_dec_ref(v_a_501_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_instCoeRelEmbedding___lam__0(lean_object* v_self_509_, lean_object* v___y_510_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = lean_apply_1(v_self_509_, v___y_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_instCoeRelEmbedding(lean_object* v_00_u03b1_513_, lean_object* v_00_u03b2_514_, lean_object* v_r_515_, lean_object* v_s_516_){
_start:
{
lean_object* v___f_517_; 
v___f_517_ = ((lean_object*)(lp_mathlib_InitialSeg_instCoeRelEmbedding___closed__0));
return v___f_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding___redArg(lean_object* v_f_518_){
_start:
{
lean_inc(v_f_518_);
return v_f_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding___redArg___boxed(lean_object* v_f_519_){
_start:
{
lean_object* v_res_520_; 
v_res_520_ = lp_mathlib_InitialSeg_toOrderEmbedding___redArg(v_f_519_);
lean_dec(v_f_519_);
return v_res_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding(lean_object* v_00_u03b1_521_, lean_object* v_00_u03b2_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_f_525_){
_start:
{
lean_inc(v_f_525_);
return v_f_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_toOrderEmbedding___boxed(lean_object* v_00_u03b1_526_, lean_object* v_00_u03b2_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_f_530_){
_start:
{
lean_object* v_res_531_; 
v_res_531_ = lp_mathlib_InitialSeg_toOrderEmbedding(v_00_u03b1_526_, v_00_u03b2_527_, v_inst_528_, v_inst_529_, v_f_530_);
lean_dec(v_f_530_);
lean_dec_ref(v_inst_529_);
lean_dec_ref(v_inst_528_);
return v_res_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toInitialSeg___redArg(lean_object* v_f_532_){
_start:
{
lean_object* v___f_533_; 
v___f_533_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_533_, 0, v_f_532_);
return v___f_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toInitialSeg(lean_object* v_00_u03b1_534_, lean_object* v_00_u03b2_535_, lean_object* v_r_536_, lean_object* v_s_537_, lean_object* v_f_538_){
_start:
{
lean_object* v___f_539_; 
v___f_539_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_539_, 0, v_f_538_);
return v___f_539_;
}
}
static lean_object* _init_lp_mathlib_InitialSeg_refl___closed__0(void){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_540_;
}
}
static lean_object* _init_lp_mathlib_InitialSeg_refl___closed__1(void){
_start:
{
lean_object* v___x_541_; lean_object* v___f_542_; 
v___x_541_ = lean_obj_once(&lp_mathlib_InitialSeg_refl___closed__0, &lp_mathlib_InitialSeg_refl___closed__0_once, _init_lp_mathlib_InitialSeg_refl___closed__0);
v___f_542_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_542_, 0, v___x_541_);
return v___f_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_refl(lean_object* v_00_u03b1_543_, lean_object* v_r_544_){
_start:
{
lean_object* v___f_545_; 
v___f_545_ = lean_obj_once(&lp_mathlib_InitialSeg_refl___closed__1, &lp_mathlib_InitialSeg_refl___closed__1_once, _init_lp_mathlib_InitialSeg_refl___closed__1);
return v___f_545_;
}
}
static lean_object* _init_lp_mathlib_InitialSeg_instInhabited___closed__0(void){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_mathlib_InitialSeg_refl(lean_box(0), lean_box(0));
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_instInhabited(lean_object* v_00_u03b1_547_, lean_object* v_r_548_){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = lean_obj_once(&lp_mathlib_InitialSeg_instInhabited___closed__0, &lp_mathlib_InitialSeg_instInhabited___closed__0_once, _init_lp_mathlib_InitialSeg_instInhabited___closed__0);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_trans___redArg(lean_object* v_f_550_, lean_object* v_g_551_){
_start:
{
lean_object* v___f_552_; 
v___f_552_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_552_, 0, v_f_550_);
lean_closure_set(v___f_552_, 1, v_g_551_);
return v___f_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_trans(lean_object* v_00_u03b1_553_, lean_object* v_00_u03b2_554_, lean_object* v_00_u03b3_555_, lean_object* v_r_556_, lean_object* v_s_557_, lean_object* v_t_558_, lean_object* v_f_559_, lean_object* v_g_560_){
_start:
{
lean_object* v___f_561_; 
v___f_561_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_561_, 0, v_f_559_);
lean_closure_set(v___f_561_, 1, v_g_560_);
return v___f_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm___redArg___lam__0(lean_object* v_f_562_, lean_object* v___y_563_){
_start:
{
lean_object* v___x_564_; 
v___x_564_ = lean_apply_1(v_f_562_, v___y_563_);
return v___x_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm___redArg___lam__1(lean_object* v_g_565_, lean_object* v___y_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lean_apply_1(v_g_565_, v___y_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm___redArg(lean_object* v_f_568_, lean_object* v_g_569_){
_start:
{
lean_object* v___f_570_; lean_object* v___f_571_; lean_object* v___x_572_; 
v___f_570_ = lean_alloc_closure((void*)(lp_mathlib_InitialSeg_antisymm___redArg___lam__0), 2, 1);
lean_closure_set(v___f_570_, 0, v_f_568_);
v___f_571_ = lean_alloc_closure((void*)(lp_mathlib_InitialSeg_antisymm___redArg___lam__1), 2, 1);
lean_closure_set(v___f_571_, 0, v_g_569_);
v___x_572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_572_, 0, v___f_570_);
lean_ctor_set(v___x_572_, 1, v___f_571_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_antisymm(lean_object* v_00_u03b1_573_, lean_object* v_00_u03b2_574_, lean_object* v_r_575_, lean_object* v_s_576_, lean_object* v_inst_577_, lean_object* v_f_578_, lean_object* v_g_579_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = lp_mathlib_InitialSeg_antisymm___redArg(v_f_578_, v_g_579_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_codRestrict___redArg(lean_object* v_f_581_){
_start:
{
lean_object* v___f_582_; 
v___f_582_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_582_, 0, v_f_581_);
return v___f_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_codRestrict(lean_object* v_00_u03b1_583_, lean_object* v_00_u03b2_584_, lean_object* v_r_585_, lean_object* v_s_586_, lean_object* v_p_587_, lean_object* v_f_588_, lean_object* v_H_589_){
_start:
{
lean_object* v___f_590_; 
v___f_590_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_590_, 0, v_f_588_);
return v___f_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_ofIsEmpty(lean_object* v_00_u03b1_592_, lean_object* v_00_u03b2_593_, lean_object* v_r_594_, lean_object* v_s_595_, lean_object* v_inst_596_){
_start:
{
lean_object* v___f_597_; 
v___f_597_ = ((lean_object*)(lp_mathlib_InitialSeg_ofIsEmpty___closed__0));
return v___f_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_leAdd___lam__0(lean_object* v_val_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_599_, 0, v_val_598_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg_leAdd(lean_object* v_00_u03b1_601_, lean_object* v_00_u03b2_602_, lean_object* v_r_603_, lean_object* v_s_604_){
_start:
{
lean_object* v___f_605_; 
v___f_605_ = ((lean_object*)(lp_mathlib_InitialSeg_leAdd___closed__0));
return v___f_605_;
}
}
static lean_object* _init_lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1(void){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; 
v___x_623_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__0));
v___x_624_ = l_String_toRawSubstring_x27(v___x_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1(lean_object* v_x_638_, lean_object* v_a_639_, lean_object* v_a_640_){
_start:
{
lean_object* v___x_641_; uint8_t v___x_642_; 
v___x_641_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ai___00__closed__1));
lean_inc(v_x_638_);
v___x_642_ = l_Lean_Syntax_isOfKind(v_x_638_, v___x_641_);
if (v___x_642_ == 0)
{
lean_object* v___x_643_; lean_object* v___x_644_; 
lean_dec(v_x_638_);
v___x_643_ = lean_box(1);
v___x_644_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_644_, 0, v___x_643_);
lean_ctor_set(v___x_644_, 1, v_a_640_);
return v___x_644_;
}
else
{
lean_object* v_quotContext_645_; lean_object* v_currMacroScope_646_; lean_object* v_ref_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; uint8_t v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; 
v_quotContext_645_ = lean_ctor_get(v_a_639_, 1);
v_currMacroScope_646_ = lean_ctor_get(v_a_639_, 2);
v_ref_647_ = lean_ctor_get(v_a_639_, 5);
v___x_648_ = lean_unsigned_to_nat(0u);
v___x_649_ = l_Lean_Syntax_getArg(v_x_638_, v___x_648_);
v___x_650_ = lean_unsigned_to_nat(2u);
v___x_651_ = l_Lean_Syntax_getArg(v_x_638_, v___x_650_);
lean_dec(v_x_638_);
v___x_652_ = 0;
v___x_653_ = l_Lean_SourceInfo_fromRef(v_ref_647_, v___x_652_);
v___x_654_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4));
v___x_655_ = lean_obj_once(&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1, &lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1_once, _init_lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1);
v___x_656_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2));
lean_inc(v_currMacroScope_646_);
lean_inc(v_quotContext_645_);
v___x_657_ = l_Lean_addMacroScope(v_quotContext_645_, v___x_656_, v_currMacroScope_646_);
v___x_658_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__6));
lean_inc_n(v___x_653_, 2);
v___x_659_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_659_, 0, v___x_653_);
lean_ctor_set(v___x_659_, 1, v___x_655_);
lean_ctor_set(v___x_659_, 2, v___x_657_);
lean_ctor_set(v___x_659_, 3, v___x_658_);
v___x_660_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12));
v___x_661_ = l_Lean_Syntax_node2(v___x_653_, v___x_660_, v___x_649_, v___x_651_);
v___x_662_ = l_Lean_Syntax_node2(v___x_653_, v___x_654_, v___x_659_, v___x_661_);
v___x_663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_663_, 0, v___x_662_);
lean_ctor_set(v___x_663_, 1, v_a_640_);
return v___x_663_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___boxed(lean_object* v_x_664_, lean_object* v_a_665_, lean_object* v_a_666_){
_start:
{
lean_object* v_res_667_; 
v_res_667_ = lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1(v_x_664_, v_a_665_, v_a_666_);
lean_dec_ref(v_a_665_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__PrincipalSeg__1(lean_object* v_x_668_, lean_object* v_a_669_, lean_object* v_a_670_){
_start:
{
lean_object* v___x_671_; uint8_t v___x_672_; 
v___x_671_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4));
lean_inc(v_x_668_);
v___x_672_ = l_Lean_Syntax_isOfKind(v_x_668_, v___x_671_);
if (v___x_672_ == 0)
{
lean_object* v___x_673_; lean_object* v___x_674_; 
lean_dec(v_x_668_);
v___x_673_ = lean_box(0);
v___x_674_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_674_, 0, v___x_673_);
lean_ctor_set(v___x_674_, 1, v_a_670_);
return v___x_674_;
}
else
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; uint8_t v___x_678_; 
v___x_675_ = lean_unsigned_to_nat(0u);
v___x_676_ = l_Lean_Syntax_getArg(v_x_668_, v___x_675_);
v___x_677_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__InitialSeg__1___closed__1));
lean_inc(v___x_676_);
v___x_678_ = l_Lean_Syntax_isOfKind(v___x_676_, v___x_677_);
if (v___x_678_ == 0)
{
lean_object* v___x_679_; lean_object* v___x_680_; 
lean_dec(v___x_676_);
lean_dec(v_x_668_);
v___x_679_ = lean_box(0);
v___x_680_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_680_, 0, v___x_679_);
lean_ctor_set(v___x_680_, 1, v_a_670_);
return v___x_680_;
}
else
{
lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; uint8_t v___x_684_; 
v___x_681_ = lean_unsigned_to_nat(1u);
v___x_682_ = l_Lean_Syntax_getArg(v_x_668_, v___x_681_);
lean_dec(v_x_668_);
v___x_683_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_682_);
v___x_684_ = l_Lean_Syntax_matchesNull(v___x_682_, v___x_683_);
if (v___x_684_ == 0)
{
lean_object* v___x_685_; lean_object* v___x_686_; 
lean_dec(v___x_682_);
lean_dec(v___x_676_);
v___x_685_ = lean_box(0);
v___x_686_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_686_, 0, v___x_685_);
lean_ctor_set(v___x_686_, 1, v_a_670_);
return v___x_686_;
}
else
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v_ref_689_; uint8_t v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_687_ = l_Lean_Syntax_getArg(v___x_682_, v___x_675_);
v___x_688_ = l_Lean_Syntax_getArg(v___x_682_, v___x_681_);
lean_dec(v___x_682_);
v_ref_689_ = l_Lean_replaceRef(v___x_676_, v_a_669_);
lean_dec(v___x_676_);
v___x_690_ = 0;
v___x_691_ = l_Lean_SourceInfo_fromRef(v_ref_689_, v___x_690_);
lean_dec(v_ref_689_);
v___x_692_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ai___00__closed__1));
v___x_693_ = ((lean_object*)(lp_mathlib_InitialSeg_term___u227ai___00__closed__2));
lean_inc(v___x_691_);
v___x_694_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_694_, 0, v___x_691_);
lean_ctor_set(v___x_694_, 1, v___x_693_);
v___x_695_ = l_Lean_Syntax_node3(v___x_691_, v___x_692_, v___x_687_, v___x_694_, v___x_688_);
v___x_696_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_696_, 0, v___x_695_);
lean_ctor_set(v___x_696_, 1, v_a_670_);
return v___x_696_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__PrincipalSeg__1___boxed(lean_object* v_x_697_, lean_object* v_a_698_, lean_object* v_a_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______unexpand__PrincipalSeg__1(v_x_697_, v_a_698_, v_a_699_);
lean_dec(v_a_698_);
return v_res_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___x3ci____1(lean_object* v_x_717_, lean_object* v_a_718_, lean_object* v_a_719_){
_start:
{
lean_object* v___x_720_; uint8_t v___x_721_; 
v___x_720_ = ((lean_object*)(lp_mathlib_term___x3ci___00__closed__1));
lean_inc(v_x_717_);
v___x_721_ = l_Lean_Syntax_isOfKind(v_x_717_, v___x_720_);
if (v___x_721_ == 0)
{
lean_object* v___x_722_; lean_object* v___x_723_; 
lean_dec(v_x_717_);
v___x_722_ = lean_box(1);
v___x_723_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_722_);
lean_ctor_set(v___x_723_, 1, v_a_719_);
return v___x_723_;
}
else
{
lean_object* v_quotContext_724_; lean_object* v_currMacroScope_725_; lean_object* v_ref_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; uint8_t v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; 
v_quotContext_724_ = lean_ctor_get(v_a_718_, 1);
v_currMacroScope_725_ = lean_ctor_get(v_a_718_, 2);
v_ref_726_ = lean_ctor_get(v_a_718_, 5);
v___x_727_ = lean_unsigned_to_nat(0u);
v___x_728_ = l_Lean_Syntax_getArg(v_x_717_, v___x_727_);
v___x_729_ = lean_unsigned_to_nat(2u);
v___x_730_ = l_Lean_Syntax_getArg(v_x_717_, v___x_729_);
lean_dec(v_x_717_);
v___x_731_ = 0;
v___x_732_ = l_Lean_SourceInfo_fromRef(v_ref_726_, v___x_731_);
v___x_733_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__4));
v___x_734_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__1));
v___x_735_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__2));
lean_inc_n(v___x_732_, 14);
v___x_736_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_736_, 0, v___x_732_);
lean_ctor_set(v___x_736_, 1, v___x_735_);
v___x_737_ = lean_obj_once(&lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1, &lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1_once, _init_lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__1);
v___x_738_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2));
lean_inc_n(v_currMacroScope_725_, 2);
lean_inc_n(v_quotContext_724_, 2);
v___x_739_ = l_Lean_addMacroScope(v_quotContext_724_, v___x_738_, v_currMacroScope_725_);
v___x_740_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__6));
v___x_741_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_741_, 0, v___x_732_);
lean_ctor_set(v___x_741_, 1, v___x_737_);
lean_ctor_set(v___x_741_, 2, v___x_739_);
lean_ctor_set(v___x_741_, 3, v___x_740_);
v___x_742_ = l_Lean_Syntax_node2(v___x_732_, v___x_734_, v___x_736_, v___x_741_);
v___x_743_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ci____1___closed__12));
v___x_744_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__4));
v___x_745_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__6));
v___x_746_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__7));
v___x_747_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_747_, 0, v___x_732_);
lean_ctor_set(v___x_747_, 1, v___x_746_);
v___x_748_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__9));
v___x_749_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11, &lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11_once, _init_lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__11);
v___x_750_ = lean_box(0);
v___x_751_ = l_Lean_addMacroScope(v_quotContext_724_, v___x_750_, v_currMacroScope_725_);
v___x_752_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__17));
v___x_753_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_753_, 0, v___x_732_);
lean_ctor_set(v___x_753_, 1, v___x_749_);
lean_ctor_set(v___x_753_, 2, v___x_751_);
lean_ctor_set(v___x_753_, 3, v___x_752_);
v___x_754_ = l_Lean_Syntax_node1(v___x_732_, v___x_748_, v___x_753_);
lean_inc(v___x_754_);
v___x_755_ = l_Lean_Syntax_node2(v___x_732_, v___x_745_, v___x_747_, v___x_754_);
v___x_756_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__19));
v___x_757_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__21));
v___x_758_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__22));
v___x_759_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_759_, 0, v___x_732_);
lean_ctor_set(v___x_759_, 1, v___x_758_);
v___x_760_ = l_Lean_Syntax_node2(v___x_732_, v___x_757_, v___x_759_, v___x_754_);
v___x_761_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__23));
v___x_762_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_762_, 0, v___x_732_);
lean_ctor_set(v___x_762_, 1, v___x_761_);
lean_inc(v___x_760_);
v___x_763_ = l_Lean_Syntax_node3(v___x_732_, v___x_756_, v___x_760_, v___x_762_, v___x_760_);
v___x_764_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___u2264i____1___closed__24));
v___x_765_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_765_, 0, v___x_732_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
v___x_766_ = l_Lean_Syntax_node3(v___x_732_, v___x_744_, v___x_755_, v___x_763_, v___x_765_);
lean_inc(v___x_766_);
v___x_767_ = l_Lean_Syntax_node4(v___x_732_, v___x_743_, v___x_728_, v___x_730_, v___x_766_, v___x_766_);
v___x_768_ = l_Lean_Syntax_node2(v___x_732_, v___x_733_, v___x_742_, v___x_767_);
v___x_769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_769_, 0, v___x_768_);
lean_ctor_set(v___x_769_, 1, v_a_719_);
return v___x_769_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___x3ci____1___boxed(lean_object* v_x_770_, lean_object* v_a_771_, lean_object* v_a_772_){
_start:
{
lean_object* v_res_773_; 
v_res_773_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______macroRules__term___x3ci____1(v_x_770_, v_a_771_, v_a_772_);
lean_dec_ref(v_a_771_);
return v_res_773_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__0(lean_object* v_x_774_){
_start:
{
lean_object* v___x_775_; uint8_t v___x_776_; 
v___x_775_ = ((lean_object*)(lp_mathlib_InitialSeg___aux__Mathlib__Order__InitialSeg______macroRules__InitialSeg__term___u227ai____1___closed__2));
v___x_776_ = l_Lean_Expr_isConstOf(v_x_774_, v___x_775_);
return v___x_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__0___boxed(lean_object* v_x_777_){
_start:
{
uint8_t v_res_778_; lean_object* v_r_779_; 
v_res_778_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__0(v_x_777_);
lean_dec_ref(v_x_777_);
v_r_779_ = lean_box(v_res_778_);
return v_r_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__13(lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_){
_start:
{
lean_object* v_ref_789_; uint8_t v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v_ref_789_ = lean_ctor_get(v___y_786_, 5);
v___x_790_ = 0;
v___x_791_ = l_Lean_SourceInfo_fromRef(v_ref_789_, v___x_790_);
v___x_792_ = ((lean_object*)(lp_mathlib_term___x3ci___00__closed__1));
v___x_793_ = ((lean_object*)(lp_mathlib_term___x3ci___00__closed__2));
lean_inc(v___x_791_);
v___x_794_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_794_, 0, v___x_791_);
lean_ctor_set(v___x_794_, 1, v___x_793_);
v___x_795_ = l_Lean_Syntax_node3(v___x_791_, v___x_792_, v_a_780_, v___x_794_, v_a_781_);
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v___x_795_);
return v___x_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__13___boxed(lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__13(v_a_797_, v_a_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_, v___y_804_);
lean_dec(v___y_804_);
lean_dec_ref(v___y_803_);
lean_dec(v___y_802_);
lean_dec_ref(v___y_801_);
lean_dec(v___y_800_);
lean_dec_ref(v___y_799_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__1(lean_object* v___f_807_, lean_object* v___f_808_, lean_object* v___f_809_, lean_object* v___f_810_, lean_object* v___f_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_){
_start:
{
lean_object* v___x_819_; lean_object* v_a_820_; lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_856_; 
v___x_819_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1_spec__0___redArg(v___y_812_);
v_a_820_ = lean_ctor_get(v___x_819_, 0);
v_isSharedCheck_856_ = !lean_is_exclusive(v___x_819_);
if (v_isSharedCheck_856_ == 0)
{
v___x_822_ = v___x_819_;
v_isShared_823_ = v_isSharedCheck_856_;
goto v_resetjp_821_;
}
else
{
lean_inc(v_a_820_);
lean_dec(v___x_819_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_856_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___f_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___f_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; 
v___x_824_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_824_, 0, v___f_807_);
v___x_825_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__1));
v___x_826_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__2));
v___f_827_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2___boxed), 12, 3);
lean_closure_set(v___f_827_, 0, v___f_808_);
lean_closure_set(v___f_827_, 1, v___x_826_);
lean_closure_set(v___f_827_, 2, v___f_809_);
v___x_828_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_828_, 0, v___x_824_);
lean_closure_set(v___x_828_, 1, v___x_826_);
v___x_829_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__4));
v___x_830_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__6___closed__5));
v___f_831_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___lam__2___boxed), 12, 3);
lean_closure_set(v___f_831_, 0, v___f_810_);
lean_closure_set(v___f_831_, 1, v___x_830_);
lean_closure_set(v___f_831_, 2, v___f_811_);
v___x_832_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_832_, 0, v___x_828_);
lean_closure_set(v___x_832_, 1, v___x_830_);
v___x_833_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_833_, 0, v___x_826_);
lean_closure_set(v___x_833_, 1, v___f_827_);
v___x_834_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_834_, 0, v___x_832_);
lean_closure_set(v___x_834_, 1, v___x_833_);
v___x_835_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_835_, 0, v___x_830_);
lean_closure_set(v___x_835_, 1, v___f_831_);
v___x_836_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_837_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_834_, v___x_835_, v___x_836_, v___y_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
if (lean_obj_tag(v___x_837_) == 0)
{
lean_object* v_a_838_; lean_object* v___x_840_; 
v_a_838_ = lean_ctor_get(v___x_837_, 0);
lean_inc(v_a_838_);
lean_dec_ref_known(v___x_837_, 1);
if (v_isShared_823_ == 0)
{
lean_ctor_set_tag(v___x_822_, 1);
v___x_840_ = v___x_822_;
goto v_reusejp_839_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v_a_820_);
v___x_840_ = v_reuseFailAlloc_847_;
goto v_reusejp_839_;
}
v_reusejp_839_:
{
lean_object* v___x_841_; 
lean_inc_ref(v___x_840_);
v___x_841_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_838_, v___x_829_, v___x_840_, v___y_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
if (lean_obj_tag(v___x_841_) == 0)
{
lean_object* v_a_842_; lean_object* v___x_843_; 
v_a_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc(v_a_842_);
lean_dec_ref_known(v___x_841_, 1);
v___x_843_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_838_, v___x_825_, v___x_840_, v___y_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
lean_dec(v_a_838_);
if (lean_obj_tag(v___x_843_) == 0)
{
lean_object* v_a_844_; lean_object* v___f_845_; lean_object* v___x_846_; 
v_a_844_ = lean_ctor_get(v___x_843_, 0);
lean_inc(v_a_844_);
lean_dec_ref_known(v___x_843_, 1);
v___f_845_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__13___boxed), 9, 2);
lean_closure_set(v___f_845_, 0, v_a_844_);
lean_closure_set(v___f_845_, 1, v_a_842_);
v___x_846_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_845_, v___y_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
return v___x_846_;
}
else
{
lean_dec(v_a_842_);
return v___x_843_;
}
}
else
{
lean_dec_ref(v___x_840_);
lean_dec(v_a_838_);
return v___x_841_;
}
}
}
else
{
lean_object* v_a_848_; lean_object* v___x_850_; uint8_t v_isShared_851_; uint8_t v_isSharedCheck_855_; 
lean_del_object(v___x_822_);
lean_dec(v_a_820_);
v_a_848_ = lean_ctor_get(v___x_837_, 0);
v_isSharedCheck_855_ = !lean_is_exclusive(v___x_837_);
if (v_isSharedCheck_855_ == 0)
{
v___x_850_ = v___x_837_;
v_isShared_851_ = v_isSharedCheck_855_;
goto v_resetjp_849_;
}
else
{
lean_inc(v_a_848_);
lean_dec(v___x_837_);
v___x_850_ = lean_box(0);
v_isShared_851_ = v_isSharedCheck_855_;
goto v_resetjp_849_;
}
v_resetjp_849_:
{
lean_object* v___x_853_; 
if (v_isShared_851_ == 0)
{
v___x_853_ = v___x_850_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_854_; 
v_reuseFailAlloc_854_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_854_, 0, v_a_848_);
v___x_853_ = v_reuseFailAlloc_854_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
return v___x_853_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__1___boxed(lean_object* v___f_857_, lean_object* v___f_858_, lean_object* v___f_859_, lean_object* v___f_860_, lean_object* v___f_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_){
_start:
{
lean_object* v_res_869_; 
v_res_869_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___lam__1(v___f_857_, v___f_858_, v___f_859_, v___f_860_, v___f_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_, v___y_866_, v___y_867_);
lean_dec(v___y_867_);
lean_dec_ref(v___y_866_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
return v_res_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1(lean_object* v_a_881_, lean_object* v_a_882_, lean_object* v_a_883_, lean_object* v_a_884_, lean_object* v_a_885_, lean_object* v_a_886_){
_start:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; 
v___x_888_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___u2264i____1___closed__4));
v___x_889_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___closed__3));
v___x_890_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_888_, v___x_889_, v_a_881_, v_a_882_, v_a_883_, v_a_884_, v_a_885_, v_a_886_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1___boxed(lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_, lean_object* v_a_894_, lean_object* v_a_895_, lean_object* v_a_896_, lean_object* v_a_897_){
_start:
{
lean_object* v_res_898_; 
v_res_898_ = lp_mathlib___aux__Mathlib__Order__InitialSeg______delab__app__term___x3ci____1(v_a_891_, v_a_892_, v_a_893_, v_a_894_, v_a_895_, v_a_896_);
lean_dec(v_a_896_);
lean_dec_ref(v_a_895_);
lean_dec(v_a_894_);
lean_dec_ref(v_a_893_);
lean_dec(v_a_892_);
lean_dec_ref(v_a_891_);
return v_res_898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___lam__0(lean_object* v_self_899_, lean_object* v___y_900_){
_start:
{
lean_object* v_toRelEmbedding_901_; lean_object* v___x_902_; 
v_toRelEmbedding_901_ = lean_ctor_get(v_self_899_, 0);
lean_inc(v_toRelEmbedding_901_);
lean_dec_ref(v_self_899_);
v___x_902_ = lean_apply_1(v_toRelEmbedding_901_, v___y_900_);
return v___x_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding(lean_object* v_00_u03b1_904_, lean_object* v_00_u03b2_905_, lean_object* v_r_906_, lean_object* v_s_907_){
_start:
{
lean_object* v___f_908_; 
v___f_908_ = ((lean_object*)(lp_mathlib_PrincipalSeg_instCoeOutRelEmbedding___closed__0));
return v___f_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeFunForall___lam__0(lean_object* v_f_909_, lean_object* v___y_910_){
_start:
{
lean_object* v_toRelEmbedding_911_; lean_object* v___x_912_; 
v_toRelEmbedding_911_ = lean_ctor_get(v_f_909_, 0);
lean_inc(v_toRelEmbedding_911_);
lean_dec_ref(v_f_909_);
v___x_912_ = lean_apply_1(v_toRelEmbedding_911_, v___y_910_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_instCoeFunForall(lean_object* v_00_u03b1_914_, lean_object* v_00_u03b2_915_, lean_object* v_r_916_, lean_object* v_s_917_){
_start:
{
lean_object* v___f_918_; 
v___f_918_ = ((lean_object*)(lp_mathlib_PrincipalSeg_instCoeFunForall___closed__0));
return v___f_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_hasCoeInitialSeg(lean_object* v_00_u03b1_919_, lean_object* v_00_u03b2_920_, lean_object* v_r_921_, lean_object* v_s_922_, lean_object* v_inst_923_){
_start:
{
lean_object* v___f_924_; 
v___f_924_ = ((lean_object*)(lp_mathlib_PrincipalSeg_instCoeFunForall___closed__0));
return v___f_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transInitial___redArg(lean_object* v_f_925_, lean_object* v_g_926_){
_start:
{
lean_object* v_toRelEmbedding_927_; lean_object* v_top_928_; lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_937_; 
v_toRelEmbedding_927_ = lean_ctor_get(v_f_925_, 0);
v_top_928_ = lean_ctor_get(v_f_925_, 1);
v_isSharedCheck_937_ = !lean_is_exclusive(v_f_925_);
if (v_isSharedCheck_937_ == 0)
{
v___x_930_ = v_f_925_;
v_isShared_931_ = v_isSharedCheck_937_;
goto v_resetjp_929_;
}
else
{
lean_inc(v_top_928_);
lean_inc(v_toRelEmbedding_927_);
lean_dec(v_f_925_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_937_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v___f_932_; lean_object* v___x_933_; lean_object* v___x_935_; 
lean_inc(v_g_926_);
v___f_932_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_932_, 0, v_toRelEmbedding_927_);
lean_closure_set(v___f_932_, 1, v_g_926_);
v___x_933_ = lean_apply_1(v_g_926_, v_top_928_);
if (v_isShared_931_ == 0)
{
lean_ctor_set(v___x_930_, 1, v___x_933_);
lean_ctor_set(v___x_930_, 0, v___f_932_);
v___x_935_ = v___x_930_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v___f_932_);
lean_ctor_set(v_reuseFailAlloc_936_, 1, v___x_933_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transInitial(lean_object* v_00_u03b1_938_, lean_object* v_00_u03b2_939_, lean_object* v_00_u03b3_940_, lean_object* v_r_941_, lean_object* v_s_942_, lean_object* v_t_943_, lean_object* v_f_944_, lean_object* v_g_945_){
_start:
{
lean_object* v___x_946_; 
v___x_946_ = lp_mathlib_PrincipalSeg_transInitial___redArg(v_f_944_, v_g_945_);
return v___x_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_trans___redArg(lean_object* v_f_947_, lean_object* v_g_948_){
_start:
{
lean_object* v_toRelEmbedding_949_; lean_object* v___x_950_; 
v_toRelEmbedding_949_ = lean_ctor_get(v_g_948_, 0);
lean_inc(v_toRelEmbedding_949_);
lean_dec_ref(v_g_948_);
v___x_950_ = lp_mathlib_PrincipalSeg_transInitial___redArg(v_f_947_, v_toRelEmbedding_949_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_trans(lean_object* v_00_u03b1_951_, lean_object* v_00_u03b2_952_, lean_object* v_00_u03b3_953_, lean_object* v_r_954_, lean_object* v_s_955_, lean_object* v_t_956_, lean_object* v_inst_957_, lean_object* v_f_958_, lean_object* v_g_959_){
_start:
{
lean_object* v___x_960_; 
v___x_960_ = lp_mathlib_PrincipalSeg_trans___redArg(v_f_958_, v_g_959_);
return v___x_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_relIsoTrans___redArg(lean_object* v_f_961_, lean_object* v_g_962_){
_start:
{
lean_object* v_toRelEmbedding_963_; lean_object* v_top_964_; lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_973_; 
v_toRelEmbedding_963_ = lean_ctor_get(v_g_962_, 0);
v_top_964_ = lean_ctor_get(v_g_962_, 1);
v_isSharedCheck_973_ = !lean_is_exclusive(v_g_962_);
if (v_isSharedCheck_973_ == 0)
{
v___x_966_ = v_g_962_;
v_isShared_967_ = v_isSharedCheck_973_;
goto v_resetjp_965_;
}
else
{
lean_inc(v_top_964_);
lean_inc(v_toRelEmbedding_963_);
lean_dec(v_g_962_);
v___x_966_ = lean_box(0);
v_isShared_967_ = v_isSharedCheck_973_;
goto v_resetjp_965_;
}
v_resetjp_965_:
{
lean_object* v___f_968_; lean_object* v___f_969_; lean_object* v___x_971_; 
v___f_968_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_968_, 0, v_f_961_);
v___f_969_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_969_, 0, v___f_968_);
lean_closure_set(v___f_969_, 1, v_toRelEmbedding_963_);
if (v_isShared_967_ == 0)
{
lean_ctor_set(v___x_966_, 0, v___f_969_);
v___x_971_ = v___x_966_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_972_; 
v_reuseFailAlloc_972_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_972_, 0, v___f_969_);
lean_ctor_set(v_reuseFailAlloc_972_, 1, v_top_964_);
v___x_971_ = v_reuseFailAlloc_972_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
return v___x_971_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_relIsoTrans(lean_object* v_00_u03b1_974_, lean_object* v_00_u03b2_975_, lean_object* v_00_u03b3_976_, lean_object* v_r_977_, lean_object* v_s_978_, lean_object* v_t_979_, lean_object* v_f_980_, lean_object* v_g_981_){
_start:
{
lean_object* v___x_982_; 
v___x_982_ = lp_mathlib_PrincipalSeg_relIsoTrans___redArg(v_f_980_, v_g_981_);
return v___x_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transRelIso___redArg(lean_object* v_f_983_, lean_object* v_g_984_){
_start:
{
lean_object* v___f_985_; lean_object* v___x_986_; 
v___f_985_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_985_, 0, v_g_984_);
v___x_986_ = lp_mathlib_PrincipalSeg_transInitial___redArg(v_f_983_, v___f_985_);
return v___x_986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_transRelIso(lean_object* v_00_u03b1_987_, lean_object* v_00_u03b2_988_, lean_object* v_00_u03b3_989_, lean_object* v_r_990_, lean_object* v_s_991_, lean_object* v_t_992_, lean_object* v_f_993_, lean_object* v_g_994_){
_start:
{
lean_object* v___x_995_; 
v___x_995_ = lp_mathlib_PrincipalSeg_transRelIso___redArg(v_f_993_, v_g_994_);
return v___x_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofElement___redArg(lean_object* v_a_997_){
_start:
{
lean_object* v___f_998_; lean_object* v___x_999_; 
v___f_998_ = ((lean_object*)(lp_mathlib_PrincipalSeg_ofElement___redArg___closed__0));
v___x_999_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_999_, 0, v___f_998_);
lean_ctor_set(v___x_999_, 1, v_a_997_);
return v___x_999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofElement(lean_object* v_00_u03b1_1000_, lean_object* v_r_1001_, lean_object* v_a_1002_){
_start:
{
lean_object* v___x_1003_; 
v___x_1003_ = lp_mathlib_PrincipalSeg_ofElement___redArg(v_a_1002_);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_codRestrict___redArg(lean_object* v_f_1004_){
_start:
{
lean_object* v_toRelEmbedding_1005_; lean_object* v_top_1006_; lean_object* v___x_1008_; uint8_t v_isShared_1009_; uint8_t v_isSharedCheck_1014_; 
v_toRelEmbedding_1005_ = lean_ctor_get(v_f_1004_, 0);
v_top_1006_ = lean_ctor_get(v_f_1004_, 1);
v_isSharedCheck_1014_ = !lean_is_exclusive(v_f_1004_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_1008_ = v_f_1004_;
v_isShared_1009_ = v_isSharedCheck_1014_;
goto v_resetjp_1007_;
}
else
{
lean_inc(v_top_1006_);
lean_inc(v_toRelEmbedding_1005_);
lean_dec(v_f_1004_);
v___x_1008_ = lean_box(0);
v_isShared_1009_ = v_isSharedCheck_1014_;
goto v_resetjp_1007_;
}
v_resetjp_1007_:
{
lean_object* v___f_1010_; lean_object* v___x_1012_; 
v___f_1010_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1010_, 0, v_toRelEmbedding_1005_);
if (v_isShared_1009_ == 0)
{
lean_ctor_set(v___x_1008_, 0, v___f_1010_);
v___x_1012_ = v___x_1008_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1013_; 
v_reuseFailAlloc_1013_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1013_, 0, v___f_1010_);
lean_ctor_set(v_reuseFailAlloc_1013_, 1, v_top_1006_);
v___x_1012_ = v_reuseFailAlloc_1013_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
return v___x_1012_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_codRestrict(lean_object* v_00_u03b1_1015_, lean_object* v_00_u03b2_1016_, lean_object* v_r_1017_, lean_object* v_s_1018_, lean_object* v_p_1019_, lean_object* v_f_1020_, lean_object* v_H_1021_, lean_object* v_H_u2082_1022_){
_start:
{
lean_object* v___x_1023_; 
v___x_1023_ = lp_mathlib_PrincipalSeg_codRestrict___redArg(v_f_1020_);
return v___x_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofIsEmpty___redArg(lean_object* v_b_1024_){
_start:
{
lean_object* v___f_1025_; lean_object* v___x_1026_; 
v___f_1025_ = ((lean_object*)(lp_mathlib_InitialSeg_ofIsEmpty___closed__0));
v___x_1026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1026_, 0, v___f_1025_);
lean_ctor_set(v___x_1026_, 1, v_b_1024_);
return v___x_1026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PrincipalSeg_ofIsEmpty(lean_object* v_00_u03b1_1027_, lean_object* v_00_u03b2_1028_, lean_object* v_s_1029_, lean_object* v_r_1030_, lean_object* v_inst_1031_, lean_object* v_b_1032_, lean_object* v_H_1033_){
_start:
{
lean_object* v___x_1034_; 
v___x_1034_ = lp_mathlib_PrincipalSeg_ofIsEmpty___redArg(v_b_1032_);
return v___x_1034_;
}
}
static lean_object* _init_lp_mathlib_PrincipalSeg_pemptyToPUnit___closed__0(void){
_start:
{
lean_object* v___x_1035_; lean_object* v___x_1036_; 
v___x_1035_ = lean_box(0);
v___x_1036_ = lp_mathlib_PrincipalSeg_ofIsEmpty___redArg(v___x_1035_);
return v___x_1036_;
}
}
static lean_object* _init_lp_mathlib_PrincipalSeg_pemptyToPUnit(void){
_start:
{
lean_object* v___x_1037_; 
v___x_1037_ = lean_obj_once(&lp_mathlib_PrincipalSeg_pemptyToPUnit___closed__0, &lp_mathlib_PrincipalSeg_pemptyToPUnit___closed__0_once, _init_lp_mathlib_PrincipalSeg_pemptyToPUnit___closed__0);
return v___x_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toInitialSeg___redArg(lean_object* v_f_1038_){
_start:
{
lean_object* v___f_1039_; 
v___f_1039_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1039_, 0, v_f_1038_);
return v___f_1039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toInitialSeg(lean_object* v_00_u03b1_1040_, lean_object* v_00_u03b2_1041_, lean_object* v_inst_1042_, lean_object* v_inst_1043_, lean_object* v_f_1044_){
_start:
{
lean_object* v___f_1045_; 
v___f_1045_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1045_, 0, v_f_1044_);
return v___f_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toInitialSeg___boxed(lean_object* v_00_u03b1_1046_, lean_object* v_00_u03b2_1047_, lean_object* v_inst_1048_, lean_object* v_inst_1049_, lean_object* v_f_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_mathlib_OrderIso_toInitialSeg(v_00_u03b1_1046_, v_00_u03b2_1047_, v_inst_1048_, v_inst_1049_, v_f_1050_);
lean_dec_ref(v_inst_1049_);
lean_dec_ref(v_inst_1048_);
return v_res_1051_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_UpperLower_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_InitialSeg(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_UpperLower_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PrincipalSeg_pemptyToPUnit = _init_lp_mathlib_PrincipalSeg_pemptyToPUnit();
lean_mark_persistent(lp_mathlib_PrincipalSeg_pemptyToPUnit);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_InitialSeg(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_UpperLower_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_InitialSeg(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_UpperLower_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_InitialSeg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_InitialSeg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_InitialSeg(builtin);
}
#ifdef __cplusplus
}
#endif
