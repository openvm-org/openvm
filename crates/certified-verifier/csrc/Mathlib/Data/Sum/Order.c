// Lean compiler output
// Module: Mathlib.Data.Sum.Order
// Imports: public import Init public meta import Init public import Mathlib.Order.Heyting.Basic public import Mathlib.Order.Hom.Basic public import Mathlib.Order.Lex public import Mathlib.Order.WithBot
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
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
uint8_t l_Sum_instDecidableRelSumLex___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqSum_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_optionEquivSumPUnit(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumComm(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* l_Sum_instDecidableRelSumLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumEmpty(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_emptySum(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instLESum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instLTSum(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Sum_instPreorderSum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sum_instPreorderSum___closed__0 = (const lean_object*)&lp_mathlib_Sum_instPreorderSum___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPreorderSum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPreorderSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Sum"};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__0 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__0_value;
static const lean_string_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Lex"};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1_value;
static const lean_string_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_⊕ₗ_"};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__2 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 106, 118, 161, 227, 189, 67, 81)}};
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 59, 74, 0, 15, 168, 215, 244)}};
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(183, 249, 121, 185, 54, 229, 39, 22)}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value;
static const lean_string_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__4 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__5 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__5_value;
static const lean_string_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ⊕ₗ "};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__6 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__6_value)}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__7 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__7_value;
static const lean_string_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__8 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__9 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__9_value),((lean_object*)(((size_t)(29) << 1) | 1))}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__10 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__5_value),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__7_value),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__10_value)}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__11 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3_value),((lean_object*)(((size_t)(30) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__11_value)}};
static const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__12 = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Sum_Lex_term___u2295_u2097__ = (const lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__12_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__0 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__0_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__1 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__1_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__2 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__2_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__3 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "_root_.Lex"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__5 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__6;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__7 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(184, 175, 53, 50, 212, 152, 178, 8)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(112, 109, 209, 0, 60, 125, 100, 111)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__8 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(47, 205, 122, 164, 96, 181, 7, 42)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__9 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__10 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__9_value)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__11 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__12 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__10_value),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__12_value)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__13 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__13_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__14 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__14_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__15 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__15_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__16 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__16_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__18 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__18_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value_aux_1),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__20 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__20_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__21 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__21_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__22 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__22_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__23 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__23_value;
static lean_once_cell_t lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__24;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 106, 118, 161, 227, 189, 67, 81)}};
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 59, 74, 0, 15, 168, 215, 244)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__25 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__25_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__25_value)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__26 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__26_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__27 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__27_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊕_"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__28 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__28_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(255, 117, 43, 15, 38, 4, 232, 178)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__29 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__29_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊕"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__30 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__30_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__31 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__31_value;
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__0___boxed(lean_object*);
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 106, 118, 161, 227, 189, 67, 81)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__1_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__1_value)} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "β"};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(163, 67, 89, 131, 111, 186, 232, 248)}};
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__4_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__4_value)} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__0 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__0_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__1 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__1_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__0_value),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__1_value)} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__2 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__2_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__3 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__3_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__4 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__4_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__2_value)} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__5 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__5_value;
static const lean_closure_object lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__4_value),((lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__5_value)} };
static const lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__6 = (const lean_object*)&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Sum_inl_u2097___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sum_inl_u2097___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Sum_inl_u2097___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_inl_u2097(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_inr_u2097___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_inr_u2097(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_LE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_LT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLT___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__0;
static lean_once_cell_t lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_preorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_preorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex_linearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_boundedOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_boundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumCongr___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderIso_sumCongr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_toEmbedding___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderIso_sumCongr___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderIso_sumCongr___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_OrderIso_sumCongr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderIso_sumCongr___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderIso_sumCongr___redArg___closed__1 = (const lean_object*)&lp_mathlib_OrderIso_sumCongr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_OrderIso_sumCongr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderIso_sumCongr___redArg___closed__0_value),((lean_object*)&lp_mathlib_OrderIso_sumCongr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_OrderIso_sumCongr___redArg___closed__2 = (const lean_object*)&lp_mathlib_OrderIso_sumCongr___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_sumComm___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_sumComm___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumComm(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_sumAssoc___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_sumAssoc___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumDualDistrib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexDualAntidistrib(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_sumLexEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_sumLexEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_emptySumLex___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_emptySumLex___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_emptySumLex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sum_Order_0__Option_elim_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sum_Order_0__Option_elim_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0;
static lean_once_cell_t lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__1;
static lean_once_cell_t lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__2;
static lean_once_cell_t lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__3;
static lean_once_cell_t lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__0;
static lean_once_cell_t lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_orderIsoSumLexPUnit(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_instLESum(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b2_2_, lean_object* v_inst_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instLTSum(lean_object* v_00_u03b1_6_, lean_object* v_00_u03b2_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_box(0);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPreorderSum(lean_object* v_00_u03b1_14_, lean_object* v_00_u03b2_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = ((lean_object*)(lp_mathlib_Sum_instPreorderSum___closed__0));
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPreorderSum___boxed(lean_object* v_00_u03b1_19_, lean_object* v_00_u03b2_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Sum_instPreorderSum(v_00_u03b1_19_, v_00_u03b2_20_, v_inst_21_, v_inst_22_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder___redArg(lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Sum_instPreorderSum(lean_box(0), lean_box(0), v_inst_24_, v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder___redArg___boxed(lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Sum_instPartialOrder___redArg(v_inst_27_, v_inst_28_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_Sum_instPreorderSum(lean_box(0), lean_box(0), v_inst_32_, v_inst_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_instPartialOrder___boxed(lean_object* v_00_u03b1_35_, lean_object* v_00_u03b2_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Sum_instPartialOrder(v_00_u03b1_35_, v_00_u03b2_36_, v_inst_37_, v_inst_38_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
return v_res_39_;
}
}
static lean_object* _init_lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__6(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__5));
v___x_80_ = l_String_toRawSubstring_x27(v___x_79_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__24(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__23));
v___x_119_ = l_String_toRawSubstring_x27(v___x_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1(lean_object* v_x_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v___x_136_; uint8_t v___x_137_; 
v___x_136_ = ((lean_object*)(lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3));
lean_inc(v_x_133_);
v___x_137_ = l_Lean_Syntax_isOfKind(v_x_133_, v___x_136_);
if (v___x_137_ == 0)
{
lean_object* v___x_138_; lean_object* v___x_139_; 
lean_dec(v_x_133_);
v___x_138_ = lean_box(1);
v___x_139_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v_a_135_);
return v___x_139_;
}
else
{
lean_object* v_quotContext_140_; lean_object* v_currMacroScope_141_; lean_object* v_ref_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; uint8_t v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v_quotContext_140_ = lean_ctor_get(v_a_134_, 1);
v_currMacroScope_141_ = lean_ctor_get(v_a_134_, 2);
v_ref_142_ = lean_ctor_get(v_a_134_, 5);
v___x_143_ = lean_unsigned_to_nat(0u);
v___x_144_ = l_Lean_Syntax_getArg(v_x_133_, v___x_143_);
v___x_145_ = lean_unsigned_to_nat(2u);
v___x_146_ = l_Lean_Syntax_getArg(v_x_133_, v___x_145_);
lean_dec(v_x_133_);
v___x_147_ = 0;
v___x_148_ = l_Lean_SourceInfo_fromRef(v_ref_142_, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__4));
v___x_150_ = lean_obj_once(&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__6, &lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__6_once, _init_lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__6);
v___x_151_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__8));
lean_inc_n(v_currMacroScope_141_, 2);
lean_inc_n(v_quotContext_140_, 2);
v___x_152_ = l_Lean_addMacroScope(v_quotContext_140_, v___x_151_, v_currMacroScope_141_);
v___x_153_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__13));
lean_inc_n(v___x_148_, 10);
v___x_154_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_154_, 0, v___x_148_);
lean_ctor_set(v___x_154_, 1, v___x_150_);
lean_ctor_set(v___x_154_, 2, v___x_152_);
lean_ctor_set(v___x_154_, 3, v___x_153_);
v___x_155_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__15));
v___x_156_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__17));
v___x_157_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__19));
v___x_158_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__20));
v___x_159_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_148_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
v___x_160_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__22));
v___x_161_ = lean_obj_once(&lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__24, &lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__24_once, _init_lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__24);
v___x_162_ = lean_box(0);
v___x_163_ = l_Lean_addMacroScope(v_quotContext_140_, v___x_162_, v_currMacroScope_141_);
v___x_164_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__27));
v___x_165_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_165_, 0, v___x_148_);
lean_ctor_set(v___x_165_, 1, v___x_161_);
lean_ctor_set(v___x_165_, 2, v___x_163_);
lean_ctor_set(v___x_165_, 3, v___x_164_);
v___x_166_ = l_Lean_Syntax_node1(v___x_148_, v___x_160_, v___x_165_);
v___x_167_ = l_Lean_Syntax_node2(v___x_148_, v___x_157_, v___x_159_, v___x_166_);
v___x_168_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__29));
v___x_169_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__30));
v___x_170_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_148_);
lean_ctor_set(v___x_170_, 1, v___x_169_);
v___x_171_ = l_Lean_Syntax_node3(v___x_148_, v___x_168_, v___x_144_, v___x_170_, v___x_146_);
v___x_172_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__31));
v___x_173_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_148_);
lean_ctor_set(v___x_173_, 1, v___x_172_);
v___x_174_ = l_Lean_Syntax_node3(v___x_148_, v___x_156_, v___x_167_, v___x_171_, v___x_173_);
v___x_175_ = l_Lean_Syntax_node1(v___x_148_, v___x_155_, v___x_174_);
v___x_176_ = l_Lean_Syntax_node2(v___x_148_, v___x_149_, v___x_154_, v___x_175_);
v___x_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v_a_135_);
return v___x_177_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___boxed(lean_object* v_x_178_, lean_object* v_a_179_, lean_object* v_a_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1(v_x_178_, v_a_179_, v_a_180_);
lean_dec_ref(v_a_179_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg(lean_object* v___y_182_){
_start:
{
lean_object* v_subExpr_184_; lean_object* v_expr_185_; lean_object* v___x_186_; 
v_subExpr_184_ = lean_ctor_get(v___y_182_, 3);
v_expr_185_ = lean_ctor_get(v_subExpr_184_, 0);
lean_inc_ref(v_expr_185_);
v___x_186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_186_, 0, v_expr_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg___boxed(lean_object* v___y_187_, lean_object* v___y_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg(v___y_187_);
lean_dec_ref(v___y_187_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0(lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg(v___y_190_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___boxed(lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0(v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
lean_dec(v___y_203_);
lean_dec_ref(v___y_202_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
return v_res_205_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__0(lean_object* v_x_206_){
_start:
{
lean_object* v___x_207_; uint8_t v___x_208_; 
v___x_207_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______macroRules__Sum__Lex__term___u2295_u2097____1___closed__9));
v___x_208_ = l_Lean_Expr_isConstOf(v_x_206_, v___x_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__0___boxed(lean_object* v_x_209_){
_start:
{
uint8_t v_res_210_; lean_object* v_r_211_; 
v_res_210_ = lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__0(v_x_209_);
lean_dec_ref(v_x_209_);
v_r_211_ = lean_box(v_res_210_);
return v_r_211_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1(lean_object* v_x_214_){
_start:
{
lean_object* v___x_215_; uint8_t v___x_216_; 
v___x_215_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___closed__0));
v___x_216_ = l_Lean_Expr_isConstOf(v_x_214_, v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1___boxed(lean_object* v_x_217_){
_start:
{
uint8_t v_res_218_; lean_object* v_r_219_; 
v_res_218_ = lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__1(v_x_217_);
lean_dec_ref(v_x_217_);
v_r_219_ = lean_box(v_res_218_);
return v_r_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__2(lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v_ref_229_; uint8_t v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v_ref_229_ = lean_ctor_get(v___y_226_, 5);
v___x_230_ = 0;
v___x_231_ = l_Lean_SourceInfo_fromRef(v_ref_229_, v___x_230_);
v___x_232_ = ((lean_object*)(lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__3));
v___x_233_ = ((lean_object*)(lp_mathlib_Sum_Lex_term___u2295_u2097___00__closed__6));
lean_inc(v___x_231_);
v___x_234_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_231_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
v___x_235_ = l_Lean_Syntax_node3(v___x_231_, v___x_232_, v_a_220_, v___x_234_, v_a_221_);
v___x_236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__2___boxed(lean_object* v_a_237_, lean_object* v_a_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__2(v_a_237_, v_a_238_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
lean_dec(v___y_244_);
lean_dec_ref(v___y_243_);
lean_dec(v___y_242_);
lean_dec_ref(v___y_241_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3(lean_object* v___f_257_, lean_object* v___f_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
lean_object* v___x_266_; lean_object* v_a_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_299_; 
v___x_266_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1_spec__0___redArg(v___y_259_);
v_a_267_ = lean_ctor_get(v___x_266_, 0);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_266_);
if (v_isSharedCheck_299_ == 0)
{
v___x_269_ = v___x_266_;
v_isShared_270_ = v_isSharedCheck_299_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_a_267_);
lean_dec(v___x_266_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_299_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_271_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_271_, 0, v___f_257_);
v___x_272_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_272_, 0, v___f_258_);
v___x_273_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__1));
v___x_274_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__2));
v___x_275_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_275_, 0, v___x_272_);
lean_closure_set(v___x_275_, 1, v___x_274_);
v___x_276_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__4));
v___x_277_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___closed__5));
v___x_278_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_278_, 0, v___x_275_);
lean_closure_set(v___x_278_, 1, v___x_277_);
v___x_279_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_280_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_271_, v___x_278_, v___x_279_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
if (lean_obj_tag(v___x_280_) == 0)
{
lean_object* v_a_281_; lean_object* v___x_283_; 
v_a_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_a_281_);
lean_dec_ref_known(v___x_280_, 1);
if (v_isShared_270_ == 0)
{
lean_ctor_set_tag(v___x_269_, 1);
v___x_283_ = v___x_269_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_267_);
v___x_283_ = v_reuseFailAlloc_290_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
lean_object* v___x_284_; 
lean_inc_ref(v___x_283_);
v___x_284_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_281_, v___x_276_, v___x_283_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; lean_object* v___x_286_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_284_, 1);
v___x_286_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_281_, v___x_273_, v___x_283_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v_a_281_);
if (lean_obj_tag(v___x_286_) == 0)
{
lean_object* v_a_287_; lean_object* v___f_288_; lean_object* v___x_289_; 
v_a_287_ = lean_ctor_get(v___x_286_, 0);
lean_inc(v_a_287_);
lean_dec_ref_known(v___x_286_, 1);
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__2___boxed), 9, 2);
lean_closure_set(v___f_288_, 0, v_a_287_);
lean_closure_set(v___f_288_, 1, v_a_285_);
v___x_289_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_288_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
return v___x_289_;
}
else
{
lean_dec(v_a_285_);
return v___x_286_;
}
}
else
{
lean_dec_ref(v___x_283_);
lean_dec(v_a_281_);
return v___x_284_;
}
}
}
else
{
lean_object* v_a_291_; lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_298_; 
lean_del_object(v___x_269_);
lean_dec(v_a_267_);
v_a_291_ = lean_ctor_get(v___x_280_, 0);
v_isSharedCheck_298_ = !lean_is_exclusive(v___x_280_);
if (v_isSharedCheck_298_ == 0)
{
v___x_293_ = v___x_280_;
v_isShared_294_ = v_isSharedCheck_298_;
goto v_resetjp_292_;
}
else
{
lean_inc(v_a_291_);
lean_dec(v___x_280_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_298_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___x_296_; 
if (v_isShared_294_ == 0)
{
v___x_296_ = v___x_293_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v_a_291_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3___boxed(lean_object* v___f_300_, lean_object* v___f_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___lam__3(v___f_300_, v___f_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_);
lean_dec(v___y_307_);
lean_dec_ref(v___y_306_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1(lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_, lean_object* v_a_327_, lean_object* v_a_328_){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_330_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__3));
v___x_331_ = ((lean_object*)(lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___closed__6));
v___x_332_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_330_, v___x_331_, v_a_323_, v_a_324_, v_a_325_, v_a_326_, v_a_327_, v_a_328_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1___boxed(lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_Sum_Lex___aux__Mathlib__Data__Sum__Order______delab__app__Sum__Lex__term___u2295_u2097____1(v_a_333_, v_a_334_, v_a_335_, v_a_336_, v_a_337_, v_a_338_);
lean_dec(v_a_338_);
lean_dec_ref(v_a_337_);
lean_dec(v_a_336_);
lean_dec_ref(v_a_335_);
lean_dec(v_a_334_);
lean_dec_ref(v_a_333_);
return v_res_340_;
}
}
static lean_object* _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0(void){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_inl_u2097___redArg(lean_object* v_x_342_){
_start:
{
lean_object* v___x_343_; lean_object* v_toFun_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_343_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
v_toFun_344_ = lean_ctor_get(v___x_343_, 0);
v___x_345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_345_, 0, v_x_342_);
lean_inc(v_toFun_344_);
v___x_346_ = lean_apply_1(v_toFun_344_, v___x_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_inl_u2097(lean_object* v_00_u03b1_347_, lean_object* v_00_u03b2_348_, lean_object* v_x_349_){
_start:
{
lean_object* v___x_350_; lean_object* v_toFun_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_350_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
v_toFun_351_ = lean_ctor_get(v___x_350_, 0);
v___x_352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_352_, 0, v_x_349_);
lean_inc(v_toFun_351_);
v___x_353_ = lean_apply_1(v_toFun_351_, v___x_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_inr_u2097___redArg(lean_object* v_x_354_){
_start:
{
lean_object* v___x_355_; lean_object* v_toFun_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_355_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
v_toFun_356_ = lean_ctor_get(v___x_355_, 0);
v___x_357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_357_, 0, v_x_354_);
lean_inc(v_toFun_356_);
v___x_358_ = lean_apply_1(v_toFun_356_, v___x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_inr_u2097(lean_object* v_00_u03b1_359_, lean_object* v_00_u03b2_360_, lean_object* v_x_361_){
_start:
{
lean_object* v___x_362_; lean_object* v_toFun_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_362_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
v_toFun_363_ = lean_ctor_get(v___x_362_, 0);
v___x_364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_364_, 0, v_x_361_);
lean_inc(v_toFun_363_);
v___x_365_ = lean_apply_1(v_toFun_363_, v___x_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_LE(lean_object* v_00_u03b1_366_, lean_object* v_00_u03b2_367_, lean_object* v_inst_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lean_box(0);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_LT(lean_object* v_00_u03b1_371_, lean_object* v_00_u03b2_372_, lean_object* v_inst_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lean_box(0);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLT___lam__0(lean_object* v___x_376_, lean_object* v___y_377_){
_start:
{
lean_object* v_toFun_378_; lean_object* v___x_379_; 
v_toFun_378_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_toFun_378_);
lean_dec_ref(v___x_376_);
v___x_379_ = lean_apply_1(v_toFun_378_, v___y_377_);
return v___x_379_;
}
}
static lean_object* _init_lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__0(void){
_start:
{
lean_object* v___x_380_; lean_object* v___f_381_; 
v___x_380_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
v___f_381_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex_toLexRelIsoLT___lam__0), 2, 1);
lean_closure_set(v___f_381_, 0, v___x_380_);
return v___f_381_;
}
}
static lean_object* _init_lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1(void){
_start:
{
lean_object* v___f_382_; lean_object* v___x_383_; 
v___f_382_ = lean_obj_once(&lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__0, &lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__0_once, _init_lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__0);
v___x_383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_383_, 0, v___f_382_);
lean_ctor_set(v___x_383_, 1, v___f_382_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLT(lean_object* v_00_u03b1_384_, lean_object* v_00_u03b2_385_, lean_object* v_inst_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lean_obj_once(&lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1, &lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1_once, _init_lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_toLexRelIsoLE(lean_object* v_00_u03b1_389_, lean_object* v_00_u03b2_390_, lean_object* v_inst_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lean_obj_once(&lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1, &lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1_once, _init_lp_mathlib_Sum_Lex_toLexRelIsoLT___closed__1);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_preorder(lean_object* v_00_u03b1_394_, lean_object* v_00_u03b2_395_, lean_object* v_inst_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = ((lean_object*)(lp_mathlib_Sum_instPreorderSum___closed__0));
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_preorder___boxed(lean_object* v_00_u03b1_399_, lean_object* v_00_u03b2_400_, lean_object* v_inst_401_, lean_object* v_inst_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_Sum_Lex_preorder(v_00_u03b1_399_, v_00_u03b2_400_, v_inst_401_, v_inst_402_);
lean_dec_ref(v_inst_402_);
lean_dec_ref(v_inst_401_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder___redArg(lean_object* v_inst_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_Sum_Lex_preorder(lean_box(0), lean_box(0), v_inst_404_, v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder___redArg___boxed(lean_object* v_inst_407_, lean_object* v_inst_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_Sum_Lex_partialOrder___redArg(v_inst_407_, v_inst_408_);
lean_dec_ref(v_inst_408_);
lean_dec_ref(v_inst_407_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder(lean_object* v_00_u03b1_410_, lean_object* v_00_u03b2_411_, lean_object* v_inst_412_, lean_object* v_inst_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lp_mathlib_Sum_Lex_preorder(lean_box(0), lean_box(0), v_inst_412_, v_inst_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_partialOrder___boxed(lean_object* v_00_u03b1_415_, lean_object* v_00_u03b2_416_, lean_object* v_inst_417_, lean_object* v_inst_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib_Sum_Lex_partialOrder(v_00_u03b1_415_, v_00_u03b2_416_, v_inst_417_, v_inst_418_);
lean_dec_ref(v_inst_418_);
lean_dec_ref(v_inst_417_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__0(lean_object* v_toDecidableLE_420_, lean_object* v_toDecidableLE_421_, lean_object* v_a_422_, lean_object* v_b_423_){
_start:
{
uint8_t v___x_424_; 
lean_inc_ref(v_b_423_);
lean_inc_ref(v_a_422_);
v___x_424_ = l_Sum_instDecidableRelSumLex___redArg(v_toDecidableLE_420_, v_toDecidableLE_421_, v_a_422_, v_b_423_);
if (v___x_424_ == 0)
{
lean_dec_ref(v_a_422_);
return v_b_423_;
}
else
{
lean_dec_ref(v_b_423_);
return v_a_422_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__1(lean_object* v_toDecidableLE_425_, lean_object* v_toDecidableLE_426_, lean_object* v_a_427_, lean_object* v_b_428_){
_start:
{
uint8_t v___x_429_; 
lean_inc_ref(v_b_428_);
lean_inc_ref(v_a_427_);
v___x_429_ = l_Sum_instDecidableRelSumLex___redArg(v_toDecidableLE_425_, v_toDecidableLE_426_, v_a_427_, v_b_428_);
if (v___x_429_ == 0)
{
lean_dec_ref(v_b_428_);
return v_a_427_;
}
else
{
lean_dec_ref(v_a_427_);
return v_b_428_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2(lean_object* v_toDecidableEq_430_, lean_object* v_a_431_, lean_object* v_b_432_){
_start:
{
lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_433_ = lean_apply_2(v_toDecidableEq_430_, v_a_431_, v_b_432_);
v___x_434_ = lean_unbox(v___x_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2___boxed(lean_object* v_toDecidableEq_435_, lean_object* v_a_436_, lean_object* v_b_437_){
_start:
{
uint8_t v_res_438_; lean_object* v_r_439_; 
v_res_438_ = lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2(v_toDecidableEq_435_, v_a_436_, v_b_437_);
v_r_439_ = lean_box(v_res_438_);
return v_r_439_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sum_Lex_linearOrder___redArg___lam__4(lean_object* v_toDecidableLT_440_, lean_object* v_toDecidableLT_441_, lean_object* v___f_442_, lean_object* v___f_443_, lean_object* v_a_444_, lean_object* v_b_445_){
_start:
{
uint8_t v___x_446_; 
lean_inc_ref(v_b_445_);
lean_inc_ref(v_a_444_);
v___x_446_ = l_Sum_instDecidableRelSumLex___redArg(v_toDecidableLT_440_, v_toDecidableLT_441_, v_a_444_, v_b_445_);
if (v___x_446_ == 0)
{
uint8_t v___x_447_; 
v___x_447_ = l_instDecidableEqSum_decEq___redArg(v___f_442_, v___f_443_, v_a_444_, v_b_445_);
if (v___x_447_ == 0)
{
uint8_t v___x_448_; 
v___x_448_ = 2;
return v___x_448_;
}
else
{
uint8_t v___x_449_; 
v___x_449_ = 1;
return v___x_449_;
}
}
else
{
uint8_t v___x_450_; 
lean_dec_ref(v_b_445_);
lean_dec_ref(v_a_444_);
lean_dec_ref(v___f_443_);
lean_dec_ref(v___f_442_);
v___x_450_ = 0;
return v___x_450_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg___lam__4___boxed(lean_object* v_toDecidableLT_451_, lean_object* v_toDecidableLT_452_, lean_object* v___f_453_, lean_object* v___f_454_, lean_object* v_a_455_, lean_object* v_b_456_){
_start:
{
uint8_t v_res_457_; lean_object* v_r_458_; 
v_res_457_ = lp_mathlib_Sum_Lex_linearOrder___redArg___lam__4(v_toDecidableLT_451_, v_toDecidableLT_452_, v___f_453_, v___f_454_, v_a_455_, v_b_456_);
v_r_458_ = lean_box(v_res_457_);
return v_r_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder___redArg(lean_object* v_inst_459_, lean_object* v_inst_460_){
_start:
{
lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v_toPartialOrder_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v_toPartialOrder_466_; lean_object* v___x_467_; lean_object* v_toDecidableLE_468_; lean_object* v_toDecidableEq_469_; lean_object* v_toDecidableLT_470_; lean_object* v_toDecidableLE_471_; lean_object* v_toDecidableEq_472_; lean_object* v_toDecidableLT_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_488_; 
v___x_461_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_459_);
v___x_462_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_461_);
v_toPartialOrder_463_ = lean_ctor_get(v___x_462_, 0);
lean_inc_ref(v_toPartialOrder_463_);
lean_dec_ref(v___x_462_);
v___x_464_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_460_);
v___x_465_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_464_);
v_toPartialOrder_466_ = lean_ctor_get(v___x_465_, 0);
lean_inc_ref(v_toPartialOrder_466_);
lean_dec_ref(v___x_465_);
v___x_467_ = lp_mathlib_Sum_Lex_preorder(lean_box(0), lean_box(0), v_toPartialOrder_463_, v_toPartialOrder_466_);
lean_dec_ref(v_toPartialOrder_466_);
lean_dec_ref(v_toPartialOrder_463_);
v_toDecidableLE_468_ = lean_ctor_get(v_inst_459_, 4);
lean_inc_ref(v_toDecidableLE_468_);
v_toDecidableEq_469_ = lean_ctor_get(v_inst_459_, 5);
lean_inc_ref(v_toDecidableEq_469_);
v_toDecidableLT_470_ = lean_ctor_get(v_inst_459_, 6);
lean_inc_ref(v_toDecidableLT_470_);
lean_dec_ref(v_inst_459_);
v_toDecidableLE_471_ = lean_ctor_get(v_inst_460_, 4);
v_toDecidableEq_472_ = lean_ctor_get(v_inst_460_, 5);
v_toDecidableLT_473_ = lean_ctor_get(v_inst_460_, 6);
v_isSharedCheck_488_ = !lean_is_exclusive(v_inst_460_);
if (v_isSharedCheck_488_ == 0)
{
lean_object* v_unused_489_; lean_object* v_unused_490_; lean_object* v_unused_491_; lean_object* v_unused_492_; 
v_unused_489_ = lean_ctor_get(v_inst_460_, 3);
lean_dec(v_unused_489_);
v_unused_490_ = lean_ctor_get(v_inst_460_, 2);
lean_dec(v_unused_490_);
v_unused_491_ = lean_ctor_get(v_inst_460_, 1);
lean_dec(v_unused_491_);
v_unused_492_ = lean_ctor_get(v_inst_460_, 0);
lean_dec(v_unused_492_);
v___x_475_ = v_inst_460_;
v_isShared_476_ = v_isSharedCheck_488_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_toDecidableLT_473_);
lean_inc(v_toDecidableEq_472_);
lean_inc(v_toDecidableLE_471_);
lean_dec(v_inst_460_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_488_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___f_477_; lean_object* v___f_478_; lean_object* v___f_479_; lean_object* v___f_480_; lean_object* v___f_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_486_; 
lean_inc_ref_n(v_toDecidableLE_471_, 2);
lean_inc_ref_n(v_toDecidableLE_468_, 2);
v___f_477_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex_linearOrder___redArg___lam__0), 4, 2);
lean_closure_set(v___f_477_, 0, v_toDecidableLE_468_);
lean_closure_set(v___f_477_, 1, v_toDecidableLE_471_);
v___f_478_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex_linearOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_478_, 0, v_toDecidableLE_468_);
lean_closure_set(v___f_478_, 1, v_toDecidableLE_471_);
v___f_479_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_479_, 0, v_toDecidableEq_469_);
v___f_480_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex_linearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_480_, 0, v_toDecidableEq_472_);
lean_inc_ref(v___f_480_);
lean_inc_ref(v___f_479_);
lean_inc_ref(v_toDecidableLT_473_);
lean_inc_ref(v_toDecidableLT_470_);
v___f_481_ = lean_alloc_closure((void*)(lp_mathlib_Sum_Lex_linearOrder___redArg___lam__4___boxed), 6, 4);
lean_closure_set(v___f_481_, 0, v_toDecidableLT_470_);
lean_closure_set(v___f_481_, 1, v_toDecidableLT_473_);
lean_closure_set(v___f_481_, 2, v___f_479_);
lean_closure_set(v___f_481_, 3, v___f_480_);
v___x_482_ = lean_alloc_closure((void*)(l_Sum_instDecidableRelSumLex___boxed), 8, 6);
lean_closure_set(v___x_482_, 0, lean_box(0));
lean_closure_set(v___x_482_, 1, lean_box(0));
lean_closure_set(v___x_482_, 2, lean_box(0));
lean_closure_set(v___x_482_, 3, lean_box(0));
lean_closure_set(v___x_482_, 4, v_toDecidableLE_468_);
lean_closure_set(v___x_482_, 5, v_toDecidableLE_471_);
v___x_483_ = lean_alloc_closure((void*)(l_instDecidableEqSum___boxed), 6, 4);
lean_closure_set(v___x_483_, 0, lean_box(0));
lean_closure_set(v___x_483_, 1, lean_box(0));
lean_closure_set(v___x_483_, 2, v___f_479_);
lean_closure_set(v___x_483_, 3, v___f_480_);
v___x_484_ = lean_alloc_closure((void*)(l_Sum_instDecidableRelSumLex___boxed), 8, 6);
lean_closure_set(v___x_484_, 0, lean_box(0));
lean_closure_set(v___x_484_, 1, lean_box(0));
lean_closure_set(v___x_484_, 2, lean_box(0));
lean_closure_set(v___x_484_, 3, lean_box(0));
lean_closure_set(v___x_484_, 4, v_toDecidableLT_470_);
lean_closure_set(v___x_484_, 5, v_toDecidableLT_473_);
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 6, v___x_484_);
lean_ctor_set(v___x_475_, 5, v___x_483_);
lean_ctor_set(v___x_475_, 4, v___x_482_);
lean_ctor_set(v___x_475_, 3, v___f_481_);
lean_ctor_set(v___x_475_, 2, v___f_478_);
lean_ctor_set(v___x_475_, 1, v___f_477_);
lean_ctor_set(v___x_475_, 0, v___x_467_);
v___x_486_ = v___x_475_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v___x_467_);
lean_ctor_set(v_reuseFailAlloc_487_, 1, v___f_477_);
lean_ctor_set(v_reuseFailAlloc_487_, 2, v___f_478_);
lean_ctor_set(v_reuseFailAlloc_487_, 3, v___f_481_);
lean_ctor_set(v_reuseFailAlloc_487_, 4, v___x_482_);
lean_ctor_set(v_reuseFailAlloc_487_, 5, v___x_483_);
lean_ctor_set(v_reuseFailAlloc_487_, 6, v___x_484_);
v___x_486_ = v_reuseFailAlloc_487_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
return v___x_486_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_linearOrder(lean_object* v_00_u03b1_493_, lean_object* v_00_u03b2_494_, lean_object* v_inst_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib_Sum_Lex_linearOrder___redArg(v_inst_495_, v_inst_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderBot___redArg(lean_object* v_inst_498_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_499_, 0, v_inst_498_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderBot(lean_object* v_00_u03b1_500_, lean_object* v_00_u03b2_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_inst_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_505_, 0, v_inst_503_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderTop___redArg(lean_object* v_inst_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_507_, 0, v_inst_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_orderTop(lean_object* v_00_u03b1_508_, lean_object* v_00_u03b2_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_513_, 0, v_inst_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_boundedOrder___redArg(lean_object* v_inst_514_, lean_object* v_inst_515_){
_start:
{
lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_516_, 0, v_inst_514_);
v___x_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_517_, 0, v_inst_515_);
v___x_518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_518_, 0, v___x_517_);
lean_ctor_set(v___x_518_, 1, v___x_516_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_Lex_boundedOrder(lean_object* v_00_u03b1_519_, lean_object* v_00_u03b2_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_Sum_Lex_boundedOrder___redArg(v_inst_523_, v_inst_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumCongr___redArg___lam__0(lean_object* v_f_526_, lean_object* v___y_527_){
_start:
{
lean_object* v___x_528_; lean_object* v_toFun_529_; lean_object* v___x_530_; 
v___x_528_ = lp_mathlib_Equiv_symm___redArg(v_f_526_);
v_toFun_529_ = lean_ctor_get(v___x_528_, 0);
lean_inc(v_toFun_529_);
lean_dec_ref(v___x_528_);
v___x_530_ = lean_apply_1(v_toFun_529_, v___y_527_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumCongr___redArg(lean_object* v_ea_536_, lean_object* v_eb_537_){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
v___x_538_ = ((lean_object*)(lp_mathlib_OrderIso_sumCongr___redArg___closed__2));
v___x_539_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_538_, v_ea_536_);
v___x_540_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_538_, v_eb_537_);
v___x_541_ = lp_mathlib_Equiv_sumCongr___redArg(v___x_539_, v___x_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumCongr(lean_object* v_00_u03b1_u2081_542_, lean_object* v_00_u03b1_u2082_543_, lean_object* v_00_u03b2_u2081_544_, lean_object* v_00_u03b2_u2082_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_ea_550_, lean_object* v_eb_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lp_mathlib_OrderIso_sumCongr___redArg(v_ea_550_, v_eb_551_);
return v___x_552_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_sumComm___closed__0(void){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumComm(lean_object* v_00_u03b1_554_, lean_object* v_00_u03b2_555_, lean_object* v_inst_556_, lean_object* v_inst_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lean_obj_once(&lp_mathlib_OrderIso_sumComm___closed__0, &lp_mathlib_OrderIso_sumComm___closed__0_once, _init_lp_mathlib_OrderIso_sumComm___closed__0);
return v___x_558_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_sumAssoc___closed__0(void){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lp_mathlib_Equiv_sumAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumAssoc(lean_object* v_00_u03b1_560_, lean_object* v_00_u03b2_561_, lean_object* v_00_u03b3_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_){
_start:
{
lean_object* v___x_566_; 
v___x_566_ = lean_obj_once(&lp_mathlib_OrderIso_sumAssoc___closed__0, &lp_mathlib_OrderIso_sumAssoc___closed__0_once, _init_lp_mathlib_OrderIso_sumAssoc___closed__0);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumDualDistrib(lean_object* v_00_u03b1_567_, lean_object* v_00_u03b2_568_, lean_object* v_inst_569_, lean_object* v_inst_570_){
_start:
{
lean_object* v___x_571_; 
v___x_571_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexCongr___redArg(lean_object* v_ea_572_, lean_object* v_eb_573_){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
v___x_574_ = lean_obj_once(&lp_mathlib_Sum_inl_u2097___redArg___closed__0, &lp_mathlib_Sum_inl_u2097___redArg___closed__0_once, _init_lp_mathlib_Sum_inl_u2097___redArg___closed__0);
v___x_575_ = ((lean_object*)(lp_mathlib_OrderIso_sumCongr___redArg___closed__2));
v___x_576_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_575_, v_ea_572_);
v___x_577_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_575_, v_eb_573_);
v___x_578_ = lp_mathlib_Equiv_sumCongr___redArg(v___x_576_, v___x_577_);
v___x_579_ = lp_mathlib_Equiv_trans___redArg(v___x_578_, v___x_574_);
v___x_580_ = lp_mathlib_Equiv_trans___redArg(v___x_574_, v___x_579_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexCongr(lean_object* v_00_u03b1_u2081_581_, lean_object* v_00_u03b1_u2082_582_, lean_object* v_00_u03b2_u2081_583_, lean_object* v_00_u03b2_u2082_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_inst_587_, lean_object* v_inst_588_, lean_object* v_ea_589_, lean_object* v_eb_590_){
_start:
{
lean_object* v___x_591_; 
v___x_591_ = lp_mathlib_OrderIso_sumLexCongr___redArg(v_ea_589_, v_eb_590_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexAssoc(lean_object* v_00_u03b1_592_, lean_object* v_00_u03b2_593_, lean_object* v_00_u03b3_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_){
_start:
{
lean_object* v___x_598_; 
v___x_598_ = lean_obj_once(&lp_mathlib_OrderIso_sumAssoc___closed__0, &lp_mathlib_OrderIso_sumAssoc___closed__0_once, _init_lp_mathlib_OrderIso_sumAssoc___closed__0);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexDualAntidistrib(lean_object* v_00_u03b1_599_, lean_object* v_00_u03b2_600_, lean_object* v_inst_601_, lean_object* v_inst_602_){
_start:
{
lean_object* v___x_603_; 
v___x_603_ = lean_obj_once(&lp_mathlib_OrderIso_sumComm___closed__0, &lp_mathlib_OrderIso_sumComm___closed__0_once, _init_lp_mathlib_OrderIso_sumComm___closed__0);
return v___x_603_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_sumLexEmpty___closed__0(void){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = lp_mathlib_Equiv_sumEmpty(lean_box(0), lean_box(0), lean_box(0));
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_sumLexEmpty(lean_object* v_00_u03b1_605_, lean_object* v_00_u03b2_606_, lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_inst_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lean_obj_once(&lp_mathlib_OrderIso_sumLexEmpty___closed__0, &lp_mathlib_OrderIso_sumLexEmpty___closed__0_once, _init_lp_mathlib_OrderIso_sumLexEmpty___closed__0);
return v___x_610_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_emptySumLex___closed__0(void){
_start:
{
lean_object* v___x_611_; 
v___x_611_ = lp_mathlib_Equiv_emptySum(lean_box(0), lean_box(0), lean_box(0));
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_emptySumLex(lean_object* v_00_u03b1_612_, lean_object* v_00_u03b2_613_, lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_inst_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lean_obj_once(&lp_mathlib_OrderIso_emptySumLex___closed__0, &lp_mathlib_OrderIso_emptySumLex___closed__0_once, _init_lp_mathlib_OrderIso_emptySumLex___closed__0);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sum_Order_0__Option_elim_match__1_splitter___redArg(lean_object* v_x_618_, lean_object* v_x_619_, lean_object* v_x_620_, lean_object* v_h__1_621_, lean_object* v_h__2_622_){
_start:
{
if (lean_obj_tag(v_x_618_) == 0)
{
lean_object* v___x_623_; 
lean_dec(v_h__1_621_);
v___x_623_ = lean_apply_2(v_h__2_622_, v_x_619_, v_x_620_);
return v___x_623_;
}
else
{
lean_object* v_val_624_; lean_object* v___x_625_; 
lean_dec(v_h__2_622_);
v_val_624_ = lean_ctor_get(v_x_618_, 0);
lean_inc(v_val_624_);
lean_dec_ref_known(v_x_618_, 1);
v___x_625_ = lean_apply_3(v_h__1_621_, v_val_624_, v_x_619_, v_x_620_);
return v___x_625_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sum_Order_0__Option_elim_match__1_splitter(lean_object* v_00_u03b1_626_, lean_object* v_00_u03b2_627_, lean_object* v_motive_628_, lean_object* v_x_629_, lean_object* v_x_630_, lean_object* v_x_631_, lean_object* v_h__1_632_, lean_object* v_h__2_633_){
_start:
{
if (lean_obj_tag(v_x_629_) == 0)
{
lean_object* v___x_634_; 
lean_dec(v_h__1_632_);
v___x_634_ = lean_apply_2(v_h__2_633_, v_x_630_, v_x_631_);
return v___x_634_;
}
else
{
lean_object* v_val_635_; lean_object* v___x_636_; 
lean_dec(v_h__2_633_);
v_val_635_ = lean_ctor_get(v_x_629_, 0);
lean_inc(v_val_635_);
lean_dec_ref_known(v_x_629_, 1);
v___x_636_ = lean_apply_3(v_h__1_632_, v_val_635_, v_x_630_, v_x_631_);
return v___x_636_;
}
}
}
static lean_object* _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0(void){
_start:
{
lean_object* v___x_637_; 
v___x_637_ = lp_mathlib_Equiv_optionEquivSumPUnit(lean_box(0));
return v___x_637_;
}
}
static lean_object* _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__1(void){
_start:
{
lean_object* v___x_638_; 
v___x_638_ = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return v___x_638_;
}
}
static lean_object* _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__2(void){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_639_;
}
}
static lean_object* _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__3(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; 
v___x_640_ = lean_obj_once(&lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__2, &lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__2_once, _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__2);
v___x_641_ = lean_obj_once(&lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__1, &lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__1_once, _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__1);
v___x_642_ = lp_mathlib_Equiv_trans___redArg(v___x_641_, v___x_640_);
return v___x_642_;
}
}
static lean_object* _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__4(void){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; 
v___x_643_ = lean_obj_once(&lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__3, &lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__3_once, _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__3);
v___x_644_ = lean_obj_once(&lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0, &lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0_once, _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0);
v___x_645_ = lp_mathlib_Equiv_trans___redArg(v___x_644_, v___x_643_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_orderIsoPUnitSumLex(lean_object* v_00_u03b1_646_, lean_object* v_inst_647_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = lean_obj_once(&lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__4, &lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__4_once, _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__4);
return v___x_648_;
}
}
static lean_object* _init_lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__0(void){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_649_;
}
}
static lean_object* _init_lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__1(void){
_start:
{
lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_650_ = lean_obj_once(&lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__0, &lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__0_once, _init_lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__0);
v___x_651_ = lean_obj_once(&lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0, &lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0_once, _init_lp_mathlib_WithBot_orderIsoPUnitSumLex___closed__0);
v___x_652_ = lp_mathlib_Equiv_trans___redArg(v___x_651_, v___x_650_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_orderIsoSumLexPUnit(lean_object* v_00_u03b1_653_, lean_object* v_inst_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lean_obj_once(&lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__1, &lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__1_once, _init_lp_mathlib_WithTop_orderIsoSumLexPUnit___closed__1);
return v___x_655_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
}
#ifdef __cplusplus
}
#endif
